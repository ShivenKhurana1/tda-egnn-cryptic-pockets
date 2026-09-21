#!/usr/bin/env python3
"""Phase 16: fix Phase 15's coordinate-frame bug and rescore, using cached
raw features and trained checkpoints -- no re-featurization needed."""
import json
from pathlib import Path

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 16 — Fixing Phase 15's Frame Bug (No Re-Featurization Needed)

Phase 15 reported near-zero DCC (pooled 0.0018) on COACH420/HOLO4K, against a
Phase 14 result that showed a clear, validated win on our own held-out split.
That combination — works on one distribution, catastrophically fails on
another — is a red flag for a bug, not a real finding, and the bug was found:

**The model's `pred_center` output is in the per-chain CENTERED coordinate
frame** (`d.pos` is built as `ca - ca.mean(0)`, mean ≈ 1e-6 after centering).
**The true ligand positions (`ligs`) are in the RAW crystal frame** (never
centered — a spot-check on one chain showed a 45+ Angstrom offset between the
two frames). Comparing a centered-frame prediction against a raw-frame target
with a 4 Angstrom success threshold fails almost by construction, regardless
of whether the prediction is otherwise any good.

This is exactly the class of bug the Phase 1 centering work was originally
about, just resurfacing in a new spot: Phase 12/13 never hit it because they
stayed in the raw frame throughout; Phase 14 avoided it by correctly
transforming its OWN training target into the centered frame. Phase 15 is the
only place a raw-frame ground truth got compared against a centered-frame
prediction without conversion.

**The fix:** add back the per-chain centering offset (`ca.mean(0)`, using the
same raw CA coordinates already cached) to `pred_center` before scoring.

**Why this is fast:** Phase 15 cached the raw per-chain features (`x`, `pos`,
`edge_index`, `edge_attr`, `ca`, `ligs` for all 2,082 clean benchmark chains)
and the trained ensemble checkpoints specifically so a mistake at this stage
wouldn't cost another ~170 minutes of fetch+parse+ESM+topology. It didn't.
''')

md(r'''## Cell 1 — Setup''')
code(r'''
!pip install -q torch-geometric==2.6.1
''')

md(r'''## Cell 2 — Load cached features and trained checkpoints (no fetching)''')
code(r'''
import json, pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

with open(locate("phase15_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)
print(f"loaded cached features for {len(raw_cache):,} chains (no re-fetch/re-featurize)")

blob10 = torch.load(locate("phase10_graphs.pt"), map_location="cpu", weights_only=False)
N_PRE = int(blob10["n_pre_topo"]); TOPO_DIM = int(blob10["topo_dim"])
blob6b = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)
mu, sd = blob6b["topo_mean"], blob6b["topo_std"]

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
    except RuntimeError: pass
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
print(f"device {DEVICE}")
''')

md(r'''## Cell 2 — Model definition (identical to Phase 14/15), load the 3 trained checkpoints''')
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


ensemble = []
for seed in (0,1,2):
    cands = []
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): cands.extend(r.rglob(f"loc_deploy_seed{seed}.pt"))
    assert cands, f"loc_deploy_seed{seed}.pt not found -- attach Phase 15's kernel output"
    state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
    m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE); m.load_state_dict(state); m.eval()
    ensemble.append(m)
print(f"loaded {len(ensemble)} trained checkpoints from Phase 15 (no retraining)")
''')

md(r'''
## Cell 3 — Inference with the FIX: convert `pred_center` back to the raw crystal
frame before it ever gets compared against `ligs`

`entry["ca"].mean(0)` is the same offset that was subtracted to build `pos` in
Phase 12/15's featurizer — adding it back undoes exactly that, and nothing
else. `ligs` were never touched, so no change is needed there.
''')
code(r'''
@torch.no_grad()
def predict_chain(entry):
    x = torch.from_numpy(entry["x"]).clone()
    x[:, N_PRE:] = (x[:, N_PRE:] - torch.from_numpy(mu)) / torch.from_numpy(sd)
    d = type("D", (), {})()
    d.x=x.to(DEVICE); d.pos=torch.from_numpy(entry["pos"]).to(DEVICE)
    d.edge_index=torch.from_numpy(entry["edge_index"]).to(DEVICE)
    d.edge_attr=torch.from_numpy(entry["edge_attr"]).to(DEVICE)
    d.batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
    seed_logits, seed_centers = [], []
    for m in ensemble:
        lg, pc = m(d)
        seed_logits.append(torch.sigmoid(lg).cpu().numpy())
        seed_centers.append(pc.cpu().numpy()[0])
    probs = np.mean(seed_logits, axis=0)
    center_model_frame = np.mean(seed_centers, axis=0)
    offset = entry["ca"].mean(axis=0)                  # THE FIX: same offset that built `pos`
    center_raw_frame = center_model_frame + offset      # back to the frame `ligs` is already in
    return probs, center_raw_frame


import time
t0 = time.time()
by_structure = defaultdict(list)
for (src_,pid,cid), entry in raw_cache.items():
    by_structure[(src_,pid)].append((cid, entry))

predictions = {}
for (src_, pid), chain_list in by_structure.items():
    ligs = chain_list[0][1]["ligs"]
    best_conf, best_center = -np.inf, None
    for cid, entry in chain_list:
        probs, center = predict_chain(entry)
        conf = float(probs.sum())
        if conf > best_conf:
            best_conf, best_center = conf, center
    predictions[(src_,pid)] = (best_center, ligs)

print(f"predicted centers (RAW frame, corrected) for {len(predictions)} structures "
      f"in {(time.time()-t0)/60:.1f} min")
''')

md(r'''## Cell 4 — Rescore with the corrected frame''')
code(r'''
def dcc_dca(pred_center, ligs):
    if pred_center is None or not ligs: return None, None
    best_dcc, best_dca = np.inf, np.inf
    for lig_atoms in ligs:
        if len(lig_atoms) == 0: continue
        com = lig_atoms.mean(axis=0)
        best_dcc = min(best_dcc, float(np.linalg.norm(pred_center - com)))
        best_dca = min(best_dca, float(np.linalg.norm(lig_atoms - pred_center, axis=1).min()))
    return (best_dcc if np.isfinite(best_dcc) else None,
            best_dca if np.isfinite(best_dca) else None)


PHASE12_REFERENCE = {"coach420": {"dcc": 0.3678, "dca": 0.6782}, "holo4k": {"dcc": 0.4033, "dca": 0.7053}}
P2RANK_REFERENCE  = {"coach420": {"dcc": 0.4706, "dca": 0.6941}, "holo4k": {"dcc": 0.5509, "dca": 0.7796}}
PHASE13_POOLED_TUNED_DCC = 0.4355

results = {"coach420": [], "holo4k": []}
for (src_, pid), (center, ligs) in predictions.items():
    dcc, dca = dcc_dca(center, ligs)
    if dcc is not None: results[src_].append((dcc, dca))

print(f"{'benchmark':<10}{'n':>6}{'DCC succ':>10}{'DCA succ':>10}")
print("-"*36)
summary = {}
all_dcc = []
for src_ in ("coach420","holo4k"):
    vals = results[src_]
    if not vals: print(f"{src_:<10}{0:>6}{'--':>10}{'--':>10}"); continue
    dcc_s = np.mean([1.0 if d<4.0 else 0.0 for d,_ in vals])
    dca_s = np.mean([1.0 if a<4.0 else 0.0 for _,a in vals])
    summary[src_] = {"n": len(vals), "dcc": float(dcc_s), "dca": float(dca_s)}
    all_dcc.extend([1.0 if d<4.0 else 0.0 for d,_ in vals])
    print(f"{src_:<10}{len(vals):>6}{dcc_s:>10.3f}{dca_s:>10.3f}")

pooled_dcc = float(np.mean(all_dcc)) if all_dcc else float("nan")

print(f"\n{'':30}{'DCC':>8}")
print("-"*40)
for src_ in ("coach420","holo4k"):
    if src_ not in summary: continue
    p12v = PHASE12_REFERENCE[src_]["dcc"]; p2v = P2RANK_REFERENCE[src_]["dcc"]
    ours = summary[src_]["dcc"]
    print(f"  {src_:<10} ours {ours:.4f}  |  Phase12(heuristic) {p12v:.4f}  ({ours-p12v:+.4f})  |  "
          f"P2Rank {p2v:.4f}  ({ours-p2v:+.4f})")

print(f"\n  pooled DCC: {pooled_dcc:.4f}  vs Phase 13 tuned-heuristic pooled: "
      f"{PHASE13_POOLED_TUNED_DCC:.4f}  ({pooled_dcc-PHASE13_POOLED_TUNED_DCC:+.4f})")

# sanity: distance distribution should now look like a real localization
# problem (mix of small and large errors), not a uniform "always ~40-80A off"
all_d = [d for v in results.values() for d,_ in v]
print(f"\n  distance sanity check: median {np.median(all_d):.1f}A, "
      f"p10 {np.percentile(all_d,10):.1f}A, p90 {np.percentile(all_d,90):.1f}A "
      f"(pre-fix this was uniformly 40-100+A; a real result should have a mix)")

json.dump({"per_benchmark": summary, "pooled_dcc": pooled_dcc,
           "phase12_reference": PHASE12_REFERENCE, "p2rank_reference": P2RANK_REFERENCE,
           "phase13_pooled_tuned_dcc": PHASE13_POOLED_TUNED_DCC,
           "distance_median": float(np.median(all_d))},
          open("/kaggle/working/phase16_summary.json", "w"), indent=2)
print("\nsaved -> phase16_summary.json")
''')

md(r'''
---

## Reading this

The distance-distribution sanity check at the bottom of Cell 4 is worth reading
before the headline numbers: if the fix is correct, distances should now span a
real range (some structures close, some far) rather than sitting uniformly in
the 40-100+ Angstrom band Phase 15 produced. That shape, not just the summary
statistic, is what confirms the frame bug is actually gone rather than
partially patched.

With the bug fixed, this is now the number to compare against Phase 12
(classification-only model + best heuristic) and P2Rank — the same comparisons
Phase 15 intended to make.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase16_frame_fix_rescore.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
