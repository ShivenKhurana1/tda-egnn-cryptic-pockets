#!/usr/bin/env python3
"""Phase 19: fix cross-chain arbitration for multi-chain HOLO4K structures.
No retraining, no re-featurization -- reuses Phase 15's cached data and
trained checkpoints, tests better per-chain confidence criteria."""
import json
from pathlib import Path

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 19 - Fixing Cross-Chain Arbitration on HOLO4K

Phase 17's visualization surfaced a specific, diagnosable failure mode: on the
worst miss (`holo4k/1OFU`, 56.8 A off), the true ligand sat entirely
disconnected from the protein chain the model examined - the model almost
certainly picked the **wrong chain** to represent that multi-chain structure,
not a bad prediction on the chain it did pick.

Every prior phase (15/16/18) selected which chain represents a HOLO4K
structure by **summed classification probability** across that chain's
residues. That criterion is biased toward chains with many moderately-
confident residues over a chain with a few sharply-confident ones at the true
pocket - exactly backwards for finding a *tight* site. This notebook tests
better alternatives.

**Why this is cheap:** COACH420 has one target chain per structure, so this
only affects HOLO4K (968 of 1,149 clean structures - the large majority of the
benchmark). Everything needed is already cached from Phase 15/18 - raw
features and trained checkpoints - so this is pure inference and a different
selection rule, no retraining, no re-featurization.

**Same discipline as Phase 13:** the winning criterion is chosen on a 35% tune
split of HOLO4K structures and scored exactly once on the other 65% - picking
a criterion and reporting its score on the same data it was picked from would
repeat the mistake that phase was built to avoid.
''')

md(r'''## Cell 1 - Setup, load cache and checkpoints (no fetching)''')
code(r'''
!pip install -q torch-geometric==2.6.1
''')

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
print(f"loaded cached features for {len(raw_cache):,} chains")

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

md(r'''## Cell 2 - Model + trained checkpoints (identical to Phase 15/16, no retraining)''')
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
    state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
    m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE); m.load_state_dict(state); m.eval()
    ensemble.append(m)
print(f"loaded {len(ensemble)} trained checkpoints (no retraining)")
''')

md(r'''## Cell 3 - Inference: compute per-chain probs + predicted center ONCE, cache in memory''')
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
    offset = entry["ca"].mean(axis=0)
    center_raw = np.mean(seed_centers, axis=0) + offset
    return probs, center_raw


import time
t0 = time.time()
chain_outputs = {}   # (src,pid,cid) -> {probs, center, n_res}
for key, entry in raw_cache.items():
    probs, center = predict_chain(entry)
    chain_outputs[key] = {"probs": probs, "center": center, "n_res": len(probs)}
print(f"inference on {len(chain_outputs):,} chains in {(time.time()-t0)/60:.1f} min "
      f"(cached in memory -- selection-criterion experiments below are instant)")
''')

md(r'''## Cell 4 - Candidate per-chain confidence criteria''')
code(r'''
CRITERIA = {
    "sum_probs":      lambda o: float(o["probs"].sum()),                      # current (Phase 15/16/18)
    "max_prob":       lambda o: float(o["probs"].max()),
    "mean_prob":      lambda o: float(o["probs"].mean()),
    "top10_mean":     lambda o: float(np.sort(o["probs"])[-10:].mean()) if o["n_res"]>=1 else 0.0,
    "count_above_05": lambda o: float((o["probs"] >= 0.5).sum()),
    "max_minus_mean":lambda o: float(o["probs"].max() - o["probs"].mean()),  # peakedness
}
print(f"{len(CRITERIA)} candidate chain-selection criteria")
''')

md(r'''## Cell 5 - Ground truth + tune/test split of HOLO4K structures''')
code(r'''
def dcc_dca(pred_center, ligs):
    best_dcc, best_dca = np.inf, np.inf
    for lig_atoms in ligs:
        if len(lig_atoms) == 0: continue
        com = lig_atoms.mean(axis=0)
        d_ = float(np.linalg.norm(pred_center - com))
        if d_ < best_dcc:
            best_dcc = d_
            best_dca = float(np.linalg.norm(lig_atoms - pred_center, axis=1).min())
    return (best_dcc if np.isfinite(best_dcc) else None, best_dca if np.isfinite(best_dca) else None)


by_structure = defaultdict(list)
ligs_by_structure = {}
for (src_,pid,cid), entry in raw_cache.items():
    by_structure[(src_,pid)].append(cid)
    ligs_by_structure[(src_,pid)] = entry["ligs"]

holo_structs = sorted(pid for src_,pid in by_structure if src_ == "holo4k")
coach_structs = sorted(pid for src_,pid in by_structure if src_ == "coach420")
print(f"HOLO4K structures (multi-chain arbitration applies): {len(holo_structs)}")
print(f"COACH420 structures (single chain, unaffected by this fix): {len(coach_structs)}")

rng = np.random.RandomState(777)
perm = list(holo_structs); rng.shuffle(perm)
n_tune = max(1, int(0.35 * len(perm)))
tune_structs, test_structs = set(perm[:n_tune]), set(perm[n_tune:])
print(f"tune {len(tune_structs)} / test {len(test_structs)}")
''')

md(r'''## Cell 6 - Score every criterion on TUNE, pick the winner''')
code(r'''
def score_structs(struct_ids, criterion_fn):
    hits, n = 0, 0
    for pid in struct_ids:
        key = ("holo4k", pid)
        chains = by_structure[key]
        best_score, best_center = -np.inf, None
        for cid in chains:
            o = chain_outputs[("holo4k", pid, cid)]
            sc = criterion_fn(o)
            if sc > best_score: best_score, best_center = sc, o["center"]
        dcc, dca = dcc_dca(best_center, ligs_by_structure[key])
        if dcc is None: continue
        n += 1; hits += int(dcc < 4.0)
    return hits / max(n, 1), n


print(f"{'criterion':<18}{'tune DCC':>10}{'n':>6}")
print("-"*36)
tune_scores = {}
for name, fn in CRITERIA.items():
    dcc, n = score_structs(tune_structs, fn)
    tune_scores[name] = dcc
    print(f"{name:<18}{dcc:>10.4f}{n:>6}")

best_name = max(tune_scores, key=tune_scores.get)
print(f"\nselected on TUNE: {best_name} (DCC {tune_scores[best_name]:.4f})")
''')

md(r'''## Cell 7 - One-shot evaluation on TEST: winner vs current (sum_probs)''')
code(r'''
base_dcc, base_n = score_structs(test_structs, CRITERIA["sum_probs"])
new_dcc, new_n = score_structs(test_structs, CRITERIA[best_name])

print(f"{'':30}{'DCC':>8}{'n':>6}")
print("-"*44)
print(f"{'current (sum_probs)':<30}{base_dcc:>8.4f}{base_n:>6}")
print(f"{'winner: '+best_name:<30}{new_dcc:>8.4f}{new_n:>6}")
print(f"\nimprovement on HOLO4K test structures: {new_dcc-base_dcc:+.4f} DCC")

print(f"\n(COACH420 is single-chain per structure -- this fix does not apply there, "
      f"its DCC is unchanged from Phase 16's 0.5115)")

json.dump({"tune_scores": tune_scores, "selected": best_name,
           "test_baseline_sum_probs": {"dcc": base_dcc, "n": base_n},
           "test_winner": {"dcc": new_dcc, "n": new_n, "criterion": best_name},
           "delta": new_dcc - base_dcc},
          open("/kaggle/working/phase19_summary.json", "w"), indent=2)
print("\nsaved -> phase19_summary.json")
''')

md(r'''
---

## Reading this

The `sum_probs` vs winning-criterion comparison on the held-out test structures
is the number that matters - chosen on a disjoint tune split, so this is a fair
test of whether a smarter chain-selection rule actually recovers accuracy on
genuinely multi-chain structures, not an artifact of picking whatever worked
best on the exact structures being reported.

If a criterion wins clearly, the fix is a one-line change (swap the selection
function) with no retraining required - the model and its predictions are
unchanged, only which chain's answer gets used for a multi-chain structure.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase19_crosschain_arbitration.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
