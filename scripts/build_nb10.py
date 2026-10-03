#!/usr/bin/env python3
"""Phase 10: close the P2Rank gap - restore dihedrals + add ESM2 embeddings."""
import json
from pathlib import Path

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 10 - Closing the P2Rank Gap: Dihedrals + ESM2 Embeddings

Phase 7 showed the model **losing** to P2Rank (test PR-AUC 0.4209 vs 0.4589, 
EGNN winning only 125/293 chains). The likely reason is visible in the numbers
already collected: the `seq-only` arm - one-hot amino acid identity, nothing
else - scores PR-AUC **0.289**, while P2Rank brings genuine physicochemical and
evolutionary features to bear. The model has been competing on structure and
topology alone, with almost no per-residue chemistry. This phase gives it some.

### Two additions, one deliberately cheap and one higher-leverage

1. **Backbone dihedrals, actually computed.** At the 6,000-structure scale
   (Phase 6a) these were zeroed out to avoid a second full parse - a documented,
   deliberate shortcut, not an oversight. This phase pays that cost.

2. **ESM2 per-residue embeddings** (`esm2_t12_35M_UR50D`, 480-dim, ~136 MB).
   This is the real lever. A protein language model trained on ~65M sequences
   encodes evolutionary conservation, secondary-structure propensity, and local
   chemical context - exactly the signal a classical PSSM would supply, without
   needing a multi-sequence-alignment search against a multi-GB database (which
   is not practical to run in a Kaggle session for 2,056 chains). ESM2 needs only
   the sequence, is fast in batch on a T4, and is the standard substitute for
   PSSM features in 2024+ structural biology models.

### What this tests, and what it doesn't

If augmented features close or reverse the P2Rank gap, that demonstrates the
architecture and topology pipeline were never the bottleneck - the *input*
information was. If the gap persists, that is evidence the model's real
disadvantage lies elsewhere (training scale, P2Rank's own feature engineering,
or the equal-selectivity regime P2Rank was tuned for).

This phase evaluates on the **same fixed train/val/test split** used in Phase 7,
for direct comparability. It does not re-run the full 10-fold ablation - that is
the natural follow-up if this shows a clear win, not a step to spend ~4 GPU-hours
on before knowing whether the cheaper change helps at all.
''')

md(r'''## Cell 1 - Setup''')

code(r'''
!pip install -q torch-geometric==2.6.1 biopython==1.84 transformers==4.46.3
''')

code(r'''
import gzip, json, os, time, urllib.request, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

warnings.filterwarnings("ignore")

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

blob = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)
dataset = blob["dataset"]
TOPO_DIM = len(blob["topo_names"])
N_SEQ = int(dataset[0].x.size(1)) - TOPO_DIM
print(f"chains {len(dataset):,} | node dim {dataset[0].x.size(1)} "
      f"(seq {N_SEQ} + topo {TOPO_DIM}) | splits present: "
      f"{ {s: sum(1 for d in dataset if d.split==s) for s in ('train','val','test')} }")

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
''')

md(r'''
## Cell 2 - Recompute real backbone dihedrals

Same predicate and same PPBuilder chain-walk used in Phase 1, re-run over the
6,000-structure set. Encoded exactly as before: (sin, cos) pairs plus a validity
mask for the undefined termini angles - angles are circular, so raw radians would
make -pi and +pi look maximally distant to the network.
''')

code(r'''
from Bio.PDB import PDBParser, MMCIFParser, PPBuilder

AA3={"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
     "MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE"}
RAW=Path("/tmp/pdb_raw"); RAW.mkdir(parents=True, exist_ok=True)

need = sorted({d.pdb_id for d in dataset})
def fetch(pid):
    for ext in ("pdb","cif"):
        p=RAW/f"{pid}.{ext}.gz"
        if p.exists() and p.stat().st_size>500: return pid,p
    for ext in ("pdb","cif"):
        try:
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.{ext}.gz",timeout=60) as r:
                b=r.read()
            if len(b)>500:
                p=RAW/f"{pid}.{ext}.gz"; p.write_bytes(b); return pid,p
        except Exception: continue
    return pid,None

t0=time.time()
with ThreadPoolExecutor(max_workers=16) as ex: got=list(ex.map(fetch, need))
paths={p:f for p,f in got if f}
print(f"fetched {len(paths):,}/{len(need):,} in {(time.time()-t0)/60:.1f} min")

def dihedrals_for(pid, path, cid, n_expect):
    """(N,4) sin/cos phi,psi + (N,2) validity mask, or None if misaligned."""
    try:
        parser = MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True, PERMISSIVE=True)
        with gzip.open(path,"rt") as fh:
            model = next(parser.get_structure(pid, fh).get_models())
        ch = model[cid]
    except Exception:
        return None
    residues=[r for r in ch if r.get_resname().strip().upper() in AA3 and "CA" in r]
    if len(residues) != n_expect:
        return None
    ppmap={}
    for pp in PPBuilder().build_peptides(ch, aa_only=False):
        for res,(phi,psi) in zip(pp, pp.get_phi_psi_list()):
            ppmap[res.id]=(phi,psi)
    phis,psis=[],[]
    for r in residues:
        phi,psi = ppmap.get(r.id, (None,None))
        phis.append(phi); psis.append(psi)
    def enc(angles):
        v=torch.tensor([a if a is not None else 0.0 for a in angles], dtype=torch.float32)
        m=torch.tensor([a is not None for a in angles], dtype=torch.float32)
        return torch.stack([torch.sin(v)*m, torch.cos(v)*m],dim=1), m.unsqueeze(1)
    pf,pm = enc(phis); sf,sm = enc(psis)
    return torch.cat([pf,sf,pm,sm],dim=1)   # (N,6): matches Phase 1 layout exactly

from joblib import Parallel, delayed
t0=time.time()
dihed = Parallel(n_jobs=max(1, os.cpu_count() or 2), verbose=1)(
    delayed(dihedrals_for)(d.pdb_id, paths[d.pdb_id], d.chain_id, int(d.num_nodes))
    if d.pdb_id in paths else None
    for d in dataset)
n_ok = sum(1 for x in dihed if x is not None)
print(f"\ndihedrals recovered for {n_ok:,}/{len(dataset):,} chains in {(time.time()-t0)/60:.1f} min")
''')

md(r'''
## Cell 3 - ESM2 embeddings

Batched by length to keep padding waste low; long chains (>1000 residues, none
here by construction since Phase 6a capped at 1000) would be chunked, but the
existing `max_residues=1000` filter means every chain fits directly.
''')

code(r'''
from transformers import AutoTokenizer, AutoModel

ESM_NAME = "facebook/esm2_t12_35M_UR50D"
tok = AutoTokenizer.from_pretrained(ESM_NAME)
esm = AutoModel.from_pretrained(ESM_NAME).to(DEVICE).eval()
ESM_DIM = esm.config.hidden_size
print(f"{ESM_NAME}: {sum(p.numel() for p in esm.parameters())/1e6:.0f}M params, "
      f"hidden {ESM_DIM}, device {DEVICE}")

@torch.no_grad()
def embed_batch(seqs, max_len=1002):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = esm(**enc).last_hidden_state          # (B, L+2, H) incl BOS/EOS
    embs = []
    for i, s in enumerate(seqs):
        n = len(s)
        embs.append(out[i, 1:1+n, :].float().cpu())   # strip BOS, keep exactly n residues
    return embs

order = sorted(range(len(dataset)), key=lambda i: len(dataset[i].seq))
BATCH = 16
t0=time.time()
esm_feats = [None]*len(dataset)
for bi in range(0, len(order), BATCH):
    idx = order[bi:bi+BATCH]
    seqs = [dataset[i].seq for i in idx]
    embs = embed_batch(seqs)
    for i, e in zip(idx, embs):
        esm_feats[i] = e
    if bi % (BATCH*20) == 0:
        print(f"  {bi}/{len(order)}  [{(time.time()-t0)/60:.1f}m]")

n_res = sum(int(d.num_nodes) for d in dataset)
print(f"\nESM2 embeddings for {len(dataset):,} chains / {n_res:,} residues "
      f"in {(time.time()-t0)/60:.1f} min ({1000*(time.time()-t0)/n_res:.2f} ms/residue)")
del esm; torch.cuda.empty_cache()
''')

md(r'''## Cell 4 - Assemble augmented features and verify alignment''')

code(r'''
kept, dropped = [], 0
for i, d in enumerate(dataset):
    if dihed[i] is None or esm_feats[i] is None:
        dropped += 1; continue
    n = int(d.num_nodes)
    if dihed[i].shape[0] != n or esm_feats[i].shape[0] != n:
        dropped += 1; continue
    onehot = d.x[:, :21]                     # first 21 cols of the original block
    topo_z = d.x[:, N_SEQ:]                  # standardized topo block, unchanged
    d.x = torch.cat([onehot, dihed[i], esm_feats[i], topo_z], dim=1)
    kept.append(d)

print(f"kept {len(kept):,}/{len(dataset):,} chains (dropped {dropped}: parse/alignment failures)")
N_PRE = 21 + 6 + ESM_DIM       # everything before the topo block
print(f"new node feature dim: {kept[0].x.size(1)}  (21 onehot + 6 dihedral + "
      f"{ESM_DIM} ESM + {TOPO_DIM} topo)")
for d in kept[:50]:
    assert torch.isfinite(d.x).all(), f"{d.pdb_id}_{d.chain_id}: non-finite features"
print("finite-value spot check: OK")

for s in ("train","val","test"):
    ds = [d for d in kept if d.split == s]
    print(f"  {s:<6} {len(ds):>5} chains")

torch.save({"dataset": kept, "n_pre_topo": N_PRE, "topo_dim": TOPO_DIM,
            "esm_dim": ESM_DIM, "pos_weight": blob["pos_weight"]},
           "/kaggle/working/phase10_graphs.pt")
print("saved -> /kaggle/working/phase10_graphs.pt")
''')

md(r'''
## Cell 5 - Model (topology fusion unchanged; embed layer widened)

Architecture is otherwise identical to every prior phase - same `EGCL`, same
gated `TopoFusion` on the topology block. Only the pre-topology input width
changes, from 27 to `21 + 6 + ESM_DIM`.
''')

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

class Net(nn.Module):
    def __init__(s, n_pre, td, h=128, L=4, dr=0.1):
        super().__init__(); s.n_pre=n_pre
        s.embed=nn.Sequential(nn.Linear(n_pre+td,h),nn.SiLU(),nn.Linear(h,h))
        s.layers=nn.ModuleList([EGCL(h,uc=(i<L-1)) for i in range(L)])
        s.fusions=nn.ModuleList([Fus(h,td) for _ in range(L)])
        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))
    def forward(s,d):
        h=s.embed(d.x); pos=d.pos; t=d.x[:,s.n_pre:]
        for l,f in zip(s.layers,s.fusions): h,pos=l(h,pos,d.edge_index,d.edge_attr); h=f(h,t)
        return s.head(h).squeeze(-1)

def run_epoch(model, loader, opt=None):
    tr = opt is not None; model.train(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([blob["pos_weight"]], device=DEVICE))
    ps,ys=[],[]
    for b in loader:
        b=b.to(DEVICE)
        with torch.set_grad_enabled(tr):
            lg=model(b); loss=crit(lg,b.y)
        if tr:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        ps.append(torch.sigmoid(lg).detach().cpu().numpy()); ys.append(b.y.cpu().numpy())
    p,y=np.concatenate(ps),np.concatenate(ys)
    return {"pr":average_precision_score(y,p),"roc":roc_auc_score(y,p),"p":p,"y":y}

def best_mcc(p,y):
    return max(matthews_corrcoef(y,(p>=t).astype(int))
               for t in np.linspace(0.05,0.95,37) if (p>=t).sum())
''')

md(r'''
## Cell 6 - Train on the fixed split, compare directly to Phase 7's numbers

Reference values, same split, same protocol (Phase 7, kernel
`tda-egnn-phase7-baseline-fpocket` v4):

| model | test PR-AUC | test ROC-AUC | test MCC |
|---|---|---|---|
| EGNN, seq+topo (Phase 7) | 0.4209 | 0.8550 | 0.407 |
| P2Rank (external, no training) | 0.4589 | 0.8467 | 0.457 |

Trained twice here: **augmented full** (dihedral+ESM+topo) and **augmented
seq-only** (dihedral+ESM, topo zeroed) - the second isolates how much of any
gain is from richer per-residue features alone versus the combination.
''')

code(r'''
splits = {s: [d for d in kept if d.split == s] for s in ("train","val","test")}
EPOCHS, PATIENCE, BATCH = 40, 8, 16
ORIG_X = [d.x.clone() for d in kept]

def apply_arm(a):
    for d, x0 in zip(kept, ORIG_X):
        t = x0[:, N_PRE:].clone()
        if a == "no-topo": t.zero_()
        d.x = torch.cat([x0[:, :N_PRE], t], dim=1)

def fit_eval(arm, tag):
    apply_arm(arm)
    torch.manual_seed(0); np.random.seed(0)
    m = Net(N_PRE, TOPO_DIM).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    tl = DataLoader(splits["train"], batch_size=BATCH, shuffle=True)
    vl = DataLoader(splits["val"], batch_size=BATCH)
    best, state, bad = -1, None, 0
    t0=time.time()
    for ep in range(EPOCHS):
        run_epoch(m, tl, opt); v = run_epoch(m, vl); sch.step()
        if v["pr"] > best: best, state, bad = v["pr"], {k:t.clone() for k,t in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
        if ep % 5 == 0: print(f"  [{tag}] ep {ep:2d} val PR {v['pr']:.4f}")
    m.load_state_dict(state); m.eval()
    te = run_epoch(m, DataLoader(splits["test"], batch_size=BATCH))
    mcc = best_mcc(te["p"], te["y"])
    print(f"  [{tag}] DONE in {(time.time()-t0)/60:.1f}m | val PR {best:.4f} "
          f"| test PR {te['pr']:.4f} ROC {te['roc']:.4f} MCC {mcc:.4f}")
    return {"arm": tag, "val_pr": best, "test_pr": te["pr"], "test_roc": te["roc"], "test_mcc": mcc}

results = {}
for arm, tag in (("full", "augmented-full"), ("no-topo", "augmented-no-topo")):
    results[tag] = fit_eval(arm, tag)

for d, x0 in zip(kept, ORIG_X): d.x = x0
''')

md(r'''## Cell 7 - Verdict''')

code(r'''
REFERENCE = {
    "P2Rank (external)":            {"test_pr": 0.4589, "test_roc": 0.8467, "test_mcc": 0.457},
    "EGNN seq+topo (Phase 7)":       {"test_pr": 0.4209, "test_roc": 0.8550, "test_mcc": 0.407},
}

print(f"{'model':<32}{'test PR-AUC':>13}{'test ROC-AUC':>14}{'test MCC':>10}")
print("-"*70)
for name, r in REFERENCE.items():
    print(f"{name:<32}{r['test_pr']:>13.4f}{r['test_roc']:>14.4f}{r['test_mcc']:>10.3f}")
for tag, r in results.items():
    print(f"{tag:<32}{r['test_pr']:>13.4f}{r['test_roc']:>14.4f}{r['test_mcc']:>10.3f}")

full = results["augmented-full"]
old = REFERENCE["EGNN seq+topo (Phase 7)"]
p2 = REFERENCE["P2Rank (external)"]
print(f"\naugmented-full vs Phase 7 EGNN : PR {full['test_pr']-old['test_pr']:+.4f}  "
      f"ROC {full['test_roc']-old['test_roc']:+.4f}  MCC {full['test_mcc']-old['test_mcc']:+.4f}")
print(f"augmented-full vs P2Rank       : PR {full['test_pr']-p2['test_pr']:+.4f}  "
      f"ROC {full['test_roc']-p2['test_roc']:+.4f}  MCC {full['test_mcc']-p2['test_mcc']:+.4f}")
print(f"\ncontribution of topology given rich seq features: "
      f"{results['augmented-full']['test_pr']-results['augmented-no-topo']['test_pr']:+.4f} PR-AUC "
      f"(single split -- not a substitute for the 10-fold result)")

verdict = "CLOSES/REVERSES the P2Rank gap" if full["test_pr"] >= p2["test_pr"] else \
          "IMPROVES over Phase 7 but does not beat P2Rank" if full["test_pr"] > old["test_pr"] else \
          "does NOT improve over Phase 7 -- richer seq features were not the bottleneck"
print(f"\nVERDICT: {verdict}")

json.dump({"results": results, "reference": REFERENCE},
          open("/kaggle/working/phase10_results.json","w"), indent=2, default=float)
print("saved -> phase10_results.json")
''')

md(r'''
---

## Reading this

Three outcomes, three different next steps:

- **Beats P2Rank** - the architecture and topology were fine all along; the model
  was starved of per-residue chemistry. Worth then re-running the full Phase 6c
  10-fold ablation with these features to get a properly powered final number,
  and re-checking the fusion gates now that there is more for them to route.
- **Improves but doesn't fully close the gap** - evolutionary features help, but
  P2Rank's edge (curated SASA-derived features, larger training corpus) is not
  fully replicated by ESM2 alone. Report the improvement honestly as partial.
- **No improvement** - the bottleneck was elsewhere (training scale, architecture
  capacity, or the equal-selectivity regime P2Rank's random forest was tuned
  for), and the seq-only weakness observed earlier was a symptom, not the cause.

Whichever result lands, it is a real answer to "can you make the model better",
not just an attempt.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase10_esm_dihedral_boost.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
