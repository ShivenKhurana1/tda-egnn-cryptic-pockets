#!/usr/bin/env python3
"""Phase 161: top-K local-attention localization head. The existing
localization head predicts the pocket center as a softmax-weighted
average over EVERY residue in the chain -- for a 300+ residue protein
where only ~20-30 residues form the real pocket, that global softmax
can be diluted by the long tail of low-scoring residues, pulling the
predicted center away from the true cluster. This masks the per-residue
localization logit to the top-K scoring residues before the softmax,
forcing the model to commit to a local region instead of averaging
globally. Zero new parameters (same loc_head shape), so the existing
main-pipeline 3B pretrained checkpoints (Phase97/101/107) load
unchanged -- transfer learning stays available, unlike the last three
architecture/feature experiments. Fine-tunes the SAME 8 pretrained
checkpoints Phase129 used, 1:1, for a direct paired-by-seed comparison
against Phase129's existing results (val PR mean 0.4023, test DCC
0.3077) with zero need to retrain the baseline arm."""
import json
from pathlib import Path

nb129 = json.load(open("/Users/shivenkhurana/bioproj/phase129_cryptobench_esm3b_transfer.ipynb"))
c129 = nb129["cells"]
s129 = lambda i: (c129[i]["source"] if isinstance(c129[i]["source"], str) else "".join(c129[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 161 - Top-K Local-Attention Localization Head (8 seeds)

The existing localization head predicts a pocket center as a softmax-
weighted average over *every* residue in the chain. This tests whether
masking that softmax to only the top-K (K=20) highest-scoring residues
before averaging - forcing a local, committed prediction instead of a
global one that can be diluted by a long tail of low-scoring residues -
improves localization accuracy. Zero new parameters, so the same 8
main-pipeline 3B pretrained checkpoints Phase129 used are reused
directly, 1:1, for a paired-by-seed comparison against Phase129's
existing results (val PR mean 0.4023, test AUC 0.9111/MCC 0.4664/DCC
0.3077) - no need to retrain the baseline arm.
''')

md(r'''## Cell 1 - Install dependencies (identical to Phase129)''')
code(s129(2))

md(r'''## Cell 2 - Setup, load Phase 48's CryptoBench-3B blob (identical to Phase129)''')
code(s129(4))

md(r'''## Cell 3 - Model, with an optional top-K mask on the localization softmax (only the readout logic changes -- same parameter shapes as every prior phase)''')
code(r'''
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
    """Same TopologicalEGNN as every prior phase. The localization head's
    PARAMETERS are unchanged (same loc_head shape) -- only the readout
    computation is new: `topk` (None = original global-softmax behavior,
    an int = mask to that many highest-scoring residues per graph before
    the softmax). Existing pretrained checkpoints load into this class
    with zero shape mismatch regardless of the `topk` setting used at
    inference/fine-tune time."""
    def __init__(s, n_pre, td, h=128, L=4, dr=0.1, use_loc_head=True, topk=None):
        super().__init__(); s.n_pre=n_pre; s.use_loc_head=use_loc_head; s.topk=topk
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
            if s.topk is not None:
                n_graphs_ = int(batch.max().item()) + 1
                masked = loc_logit.clone()
                for g in range(n_graphs_):
                    idx = (batch == g).nonzero(as_tuple=True)[0]
                    if len(idx) > s.topk:
                        vals = loc_logit[idx]
                        _, top_local = torch.topk(vals, s.topk)
                        keep_mask = torch.zeros(len(idx), dtype=torch.bool, device=h.device)
                        keep_mask[top_local] = True
                        masked[idx[~keep_mask]] = float("-inf")
                loc_logit = masked
            attn = pyg_softmax(loc_logit, batch)                    # sums to 1 per graph
            weighted = attn.unsqueeze(-1) * pos                     # (N,3)
            n_graphs = int(batch.max().item()) + 1
            pred_center = torch.zeros(n_graphs, 3, device=h.device).index_add_(0, batch, weighted)
        return logits, pred_center
''')

md(r'''## Cell 4 - Training machinery (identical to Phase129)''')
code(s129(8))

md(r'''## Cell 5 - Fine-tune all 8 seeds with topk=20, each from the SAME pretrained checkpoint Phase129 used''')
code(r'''
PRETRAINED_FILES = [f"clean_deploy_esm3b_seed{s}.pt" for s in range(8)]   # identical to Phase129
TOPK = 20

BEST_LAMBDA = 0.1
results = []
t0 = time.time()
for seed in range(8):
    pretrained = locate(PRETRAINED_FILES[seed])
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=True, topk=TOPK).to(DEVICE)
    state = torch.load(pretrained, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, f"architecture mismatch: missing={missing} unexpected={unexpected}"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    tl = DataLoader(splits["train"], batch_size=16, shuffle=True)
    vl = DataLoader(splits["val"], batch_size=16)
    best, state_best, bad, patience = -1, None, 0, 8
    for ep in range(40):
        run_epoch(model, tl, opt, BEST_LAMBDA)
        v = run_epoch(model, vl, lam=BEST_LAMBDA)
        sch.step()
        if v["pr"] > best:
            best, state_best, bad = v["pr"], {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state_best); model.eval()
    v_final = run_epoch(model, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    val_dcc = v_final.get("dcc", float("nan"))
    torch.save(model.state_dict(), WORK / f"cache/cryptobench_esm3b_topk_seed{seed}.pt")
    results.append(dict(seed=seed, val_pr=best, val_dcc=val_dcc))
    print(f"  topk seed {seed} (pretrained: {pretrained.name}): val PR {best:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")

all_pr = [r["val_pr"] for r in results]
all_dcc = [r["val_dcc"] for r in results]
print(f"\n--- topk=20, 8-seed summary ---")
print(f"val PR:  mean {np.mean(all_pr):.4f}  std {np.std(all_pr):.4f}")
print(f"val DCC: mean {np.mean(all_dcc):.4f}  std {np.std(all_dcc):.4f}")
print(f"\nfor reference, Phase129 (no topk masking): val PR mean 0.4023, val DCC values in that phase's log")
json.dump({"seeds": results, "all_pr": all_pr, "all_dcc": all_dcc},
          open(WORK/"phase161_topk_8seed_results.json", "w"), indent=2)
print("\nsaved -> phase161_topk_8seed_results.json")
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase161_cryptobench_topk_localization.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
