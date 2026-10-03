#!/usr/bin/env python3
"""Phase 120: CryptoBench + B-factor flexibility feature. Cryptic pockets
are fundamentally about LOCAL FLEXIBILITY -- regions that are rigid in
the apo crystal but flexible enough to open a pocket upon ligand binding.
Crystallographic B-factors (already present in every PDB/mmCIF file, free
to extract, no extra computation) are a direct experimental proxy for
local flexibility/disorder. Never tried in this project. Per-chain
z-scored (raw B-factor scale varies hugely between structures/resolution/
refinement, so within-chain RELATIVE flexibility is the meaningful
signal) and added as ONE extra feature column, positioned so the
with-vs-without ablation only needs to zero that ONE column (same
architecture both arms -- avoids the RNG-shift confound discovered in
Phase119, where changing use_aux_head shifted weight initialization)."""
import json, pickle
from pathlib import Path

C117 = pickle.load(open("/tmp/p117_cells.pkl", "rb"))
s117 = lambda i: (C117[i]["source"] if isinstance(C117[i]["source"], str) else "".join(C117[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 120 - CryptoBench + B-factor Flexibility Feature

Cryptic pockets are, by definition, regions that are apo-invisible but
become pockets upon binding -- they require local flexibility. B-factors
(crystallographic thermal/disorder factors, free in every PDB file) are a
direct experimental measure of that. Per-chain z-scored and added as one
extra feature column. Ingests fresh (Phase117's blob doesn't have this),
then trains with-vs-without (same architecture, just zeroing the one
column for the ablation) in the same run for direct comparison.
''')

md(r'''## Cell 1 - Install dependencies (identical to Phase117)''')
code(s117(2))

md(r'''## Cell 2 - Setup, fetch CryptoBench's dataset.json + folds.json + noncryptic-pockets.json (identical to Phase117)''')
code(s117(4))

md(r'''## Cell 3 - Fetch apo structures (identical to Phase117)''')
code(s117(6))

md(r'''## Cell 4 - Parse apo chains, build labels -- NEW: also extract per-residue CA B-factor, per-chain z-scored''')
code(s117(8)
     .replace('seq, ca, per_res, resnums = [], [], [], []',
               'seq, ca, per_res, resnums, bfac = [], [], [], [], []')
     .replace('ca.append(r["CA"].coord)',
               'ca.append(r["CA"].coord)\n        bfac.append(r["CA"].get_bfactor())')
     .replace('return dict(pdb_id=pid, chain_id=cid, seq="".join(seq),',
               'bfac = np.asarray(bfac, dtype=np.float32)\n'
               '    bfac_z = (bfac - bfac.mean()) / (bfac.std() + 1e-6)   # per-chain relative flexibility\n'
               '    return dict(pdb_id=pid, chain_id=cid, seq="".join(seq), bfac=bfac_z,'))

md(r'''## Cell 5 - Featuriser definitions (identical to Phase117)''')
code(s117(10))

md(r'''## Cell 6 - Compute topology (identical to Phase117)''')
code(s117(12))

md(r'''## Cell 7 - Recompute backbone dihedrals (identical to Phase117)''')
code(s117(14))

md(r'''## Cell 8 - ESM2 embeddings (35M, identical to Phase117)''')
code(s117(16))

md(r'''## Cell 9 - Assemble PyG graphs -- NEW: insert the B-factor column right after ESM, before topology; track BFAC_COL_IDX for the ablation''')
code(s117(18)
     .replace('N_PRE = 21 + 6 + ESM_DIM',
               'BFAC_COL_IDX = 21 + 6 + ESM_DIM   # index of the new B-factor column\n'
               'N_PRE = 21 + 6 + ESM_DIM + 1        # +1 for B-factor, now part of the "pre-topology" block')
     .replace('x = torch.cat([onehot, dihed[i], esm_feats[i], topo_z], dim=1)',
               'bfac_col = torch.from_numpy(r["bfac"]).unsqueeze(1).float()\n'
               '    x = torch.cat([onehot, dihed[i], esm_feats[i], bfac_col, topo_z], dim=1)'))

md(r'''## Cell 10 - Carve val split, integrity checks, save (identical to Phase117 except output filename)''')
code(s117(20).replace('phase117_cryptobench_graphs_aux.pt', 'phase120_cryptobench_graphs_bfactor.pt')
             .replace('"source": "CryptoBench (Skrhak et al. 2025, btae745) + noncryptic-pockets aux label"',
                       '"bfac_col_idx": BFAC_COL_IDX, '
                       '"source": "CryptoBench (Skrhak et al. 2025, btae745) + per-chain z-scored B-factor feature"'))

md(r'''## Cell 11 - Model (identical to Phase118/119/121; imports needed here since fresh ingestion never needed them)''')
code(r'''
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import average_precision_score, roc_auc_score

class EGCL(nn.Module):
    def __init__(s,h,ea=2,act=nn.SiLU,uc=True):
        super().__init__(); s.uc=uc
        s.e=nn.Sequential(nn.Linear(2*h+1+ea,h),act(),nn.Linear(h,h),act())
        s.n=nn.Sequential(nn.Linear(2*h,h),act(),nn.Linear(h,h))
        s.a=nn.Sequential(nn.Linear(h,1),nn.Sigmoid())
        if uc:
            s.c=nn.Sequential(nn.Linear(h,h),act(),nn.Linear(h,1,bias=False))
            nn.init.xavier_uniform_(s.c[-1].weight,gain=0.001)
        s.norm=nn.LayerNorm(h)
    def forward(s,h,x,ei,ea):
        src,dst=ei; rel=x[src]-x[dst]; d2=(rel**2).sum(-1,keepdim=True)
        m=s.e(torch.cat([h[src],h[dst],d2,ea],-1)); m=m*s.a(m)
        agg=torch.zeros_like(h).index_add_(0,dst,m)
        deg=torch.zeros(h.size(0),1,device=h.device).index_add_(0,dst,torch.ones_like(d2)).clamp(min=1)
        h=s.norm(h+s.n(torch.cat([h,agg/deg],-1)))
        if s.uc:
            w=s.c(m)*rel/(d2.sqrt()+1.0); x=x+torch.zeros_like(x).index_add_(0,dst,w)/deg
        return h,x

class Fus(nn.Module):
    def __init__(s,h,t):
        super().__init__(); s.p=nn.Linear(t,h); s.g=nn.Sequential(nn.Linear(h+t,h),nn.Sigmoid())
    def forward(s,h,t): return h+s.g(torch.cat([h,t],-1))*s.p(t)

class LocalizingEGNN(nn.Module):
    def __init__(s, n_pre, td, h=128, L=4, dr=0.1, use_loc_head=True):
        super().__init__(); s.n_pre=n_pre; s.use_loc_head=use_loc_head
        s.embed=nn.Sequential(nn.Linear(n_pre+td,h),nn.SiLU(),nn.Linear(h,h))
        s.ls=nn.ModuleList([EGCL(h,uc=(i<L-1)) for i in range(L)])
        s.fs=nn.ModuleList([Fus(h,td) for _ in range(L)])
        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))
        if use_loc_head:
            s.loc_head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Linear(h//2,1))

    def forward(s, d):
        h=s.embed(d.x); pos=d.pos; t=d.x[:,s.n_pre:]
        for l,f in zip(s.ls,s.fs): h,pos=l(h,pos,d.edge_index,d.edge_attr); h=f(h,t)
        logits = s.head(h).squeeze(-1)
        pred_center = None
        if s.use_loc_head:
            batch = d.batch if hasattr(d, "batch") and d.batch is not None \
                    else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            loc_logit = s.loc_head(h).squeeze(-1)
            attn = pyg_softmax(loc_logit, batch)
            weighted = attn.unsqueeze(-1) * pos
            n_graphs = int(batch.max().item()) + 1
            pred_center = torch.zeros(n_graphs, 3, device=h.device).index_add_(0, batch, weighted)
        return logits, pred_center

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
splits = {s: [d for d in dataset if d.split == s] for s in ("train","val","test")}
# TOPO_DIM as a bare global stays at its PRE-filter value (35) throughout
# ingestion -- only Cell 10's SAVE line computes the correct POST-filter
# value (int(keep.sum())=34, after dropping near-constant columns) inline,
# without ever reassigning the variable itself. Phase118/119/121 never hit
# this because they reload TOPO_DIM from a saved blob (already correct);
# this combined ingest+train notebook needs the fix explicitly:
TOPO_DIM = int(keep.sum())
print(f"device {DEVICE} | node dim {dataset[0].x.size(1)} | N_PRE {N_PRE} | TOPO_DIM {TOPO_DIM} | BFAC_COL_IDX {BFAC_COL_IDX} | "
      f"pos_weight {POS_WEIGHT:.2f} | " + " | ".join(f"{k} {len(v)}" for k,v in splits.items()))
WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)
''')

md(r'''## Cell 12 - Training machinery (identical pattern to Phase121)''')
code(r'''
def run_epoch(model, loader, opt=None, lam=0.0):
    tr = opt is not None; model.train(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    huber = nn.SmoothL1Loss(beta=4.0)
    ps, ys, cd = [], [], []
    for b in loader:
        b = b.to(DEVICE)
        with torch.set_grad_enabled(tr):
            logits, pred_c = model(b)
            loss = crit(logits, b.y)
            if lam > 0 and pred_c is not None:
                true_c = b.true_center.view(-1, 3)
                loss = loss + lam * huber(pred_c, true_c)
                cd.append((pred_c - true_c).norm(dim=1).detach().cpu().numpy())
        if tr:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        ps.append(torch.sigmoid(logits).detach().cpu().numpy()); ys.append(b.y.cpu().numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)
    out = {"pr": average_precision_score(y, p), "roc": roc_auc_score(y, p)}
    if cd:
        d_ = np.concatenate(cd)
        out["dcc"] = float((d_ < 4.0).mean()); out["mean_center_err"] = float(d_.mean())
    return out


def fit(use_loc_head, lam, seed, epochs=40, patience=8, batch=16):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl = DataLoader(splits["train"], batch_size=batch, shuffle=True)
    vl = DataLoader(splits["val"], batch_size=batch)
    best, state, bad = -1, None, 0
    for ep in range(epochs):
        run_epoch(model, tl, opt, lam)
        v = run_epoch(model, vl, lam=lam)
        sch.step()
        if v["pr"] > best:
            best, state, bad = v["pr"], {k:t.clone() for k,t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval()
    return model, best
''')

md(r'''## Cell 13 - Train WITH B-factor first (data pristine), then zero the one B-factor column for WITHOUT, same seed=0, same run (same architecture both arms -- avoids the RNG-shift confound from Phase119)''')
code(r'''
BEST_LAMBDA = 0.1
results = []
t0 = time.time()

m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=0)
v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
torch.save(m.state_dict(), WORK/"cache/cryptobench_withbfac_seed0.pt")
results.append(dict(arm="with B-factor", val_pr=val_pr, val_dcc=v.get("dcc", float("nan"))))
print(f"  with B-factor    seed0: val PR {val_pr:.4f}  val DCC {v.get('dcc', float('nan')):.4f}  [{(time.time()-t0)/60:.1f}m]")

for d in dataset:
    d.x[:, BFAC_COL_IDX] = 0.0   # zero JUST the B-factor column -- same architecture, isolates its contribution
m2, val_pr2 = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=0)
v2 = run_epoch(m2, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
torch.save(m2.state_dict(), WORK/"cache/cryptobench_nobfac_seed0.pt")
results.append(dict(arm="without B-factor (zeroed)", val_pr=val_pr2, val_dcc=v2.get("dcc", float("nan"))))
print(f"  without B-factor seed0: val PR {val_pr2:.4f}  val DCC {v2.get('dcc', float('nan')):.4f}  [{(time.time()-t0)/60:.1f}m]")

print(f"\n--- B-factor ablation (single seed=0, same run) ---")
with_pr = results[0]["val_pr"]
for r in results:
    print(f"  {r['arm']:<30} val PR {r['val_pr']:.4f}  ({r['val_pr']-with_pr:+.4f} vs with-B-factor)  val DCC {r['val_dcc']:.4f}")
json.dump(results, open(WORK/"phase120_bfactor_results.json", "w"), indent=2)
print("\nsaved -> phase120_bfactor_results.json")
''')

md(r'''
---

## Reading this

- **with-B-factor clearly beats without** (gap outside typical
  single-seed noise) -- crystallographic flexibility carries real signal
  for cryptic-pocket prediction, worth an 8-seed confirmation.
- **Close/tied** -- B-factor doesn't add exploitable signal beyond what
  topology+sequence already capture.
- **without beats with** -- B-factor might be adding noise (crystal
  packing artifacts, resolution-dependent noise unrelated to true
  flexibility) rather than signal.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase120_cryptobench_bfactor.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
