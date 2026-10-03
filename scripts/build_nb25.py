#!/usr/bin/env python3
"""Phase 25: matched 3-seed comparison across all 6 arms (H0/H1/H2/burial/
full/no_topo) using ONLY seeds 0-2, which every arm has trained. Resolves
Phase 23 v2's ensemble-size confound with ZERO new training -- pure
rescoring on cached checkpoints and cached benchmark features. CPU-friendly
since there's no backprop involved."""
import json
from pathlib import Path

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 25 - Matched 3-Seed Comparison (Fixing the Thread 1 Confound, No Training)

Phase 23 v2 found `full`/`no_topo` reference checkpoints only had 3 seeds
loaded while H0/H1/H2/burial had 8 -- an ensemble-size mismatch that made
"burial beats full" untrustworthy. This notebook fixes it the cheap way:
score **all six arms using only seeds 0-2**, since every arm has those three
trained. No new training, just loading checkpoints and running inference --
CPU-friendly, should take minutes not hours.

This does not replace the eventual fully-matched 8-seed comparison (queued as
a separate, longer CPU training job) -- it is the fast, confound-free
intermediate check.
''')

md(r'''## Cell 1 - Setup''')
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

blob10 = torch.load(locate("phase10_graphs.pt"), map_location="cpu", weights_only=False)
N_PRE = int(blob10["n_pre_topo"]); TOPO_DIM = int(blob10["topo_dim"])
blob6b = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)
mu_b, sd_b = blob6b["topo_mean"], blob6b["topo_std"]
TOPO_NAMES = blob6b["topo_names"]; BURIAL_NAMES = set(blob6b["burial_names"])

DEVICE = torch.device("cpu")
print(f"device {DEVICE} (rescoring only -- no training in this notebook)")

MASKS = {
    "H0_only":     np.array([n.startswith("H0_") for n in TOPO_NAMES]),
    "H1_only":     np.array([n.startswith("H1_") for n in TOPO_NAMES]),
    "H2_only":     np.array([n.startswith("H2_") for n in TOPO_NAMES]),
    "burial_only": np.array([n in BURIAL_NAMES for n in TOPO_NAMES]),
}
assert sum(m.sum() for m in MASKS.values()) == len(TOPO_NAMES)
print("masks reconstructed and verified")
''')

md(r'''## Cell 2 - Model class (identical to every prior localization phase)''')
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
''')

md(r'''## Cell 3 - Load ONLY seeds 0-2 for every arm (matched, no new training)''')
code(r'''
def load_seeds(prefix, seeds):
    out = {}
    for seed in seeds:
        cands = []
        for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
            if r.exists(): cands.extend(r.rglob(f"{prefix}{seed}.pt"))
        if not cands: continue
        state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
        m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE); m.load_state_dict(state); m.eval()
        out[seed] = m
    return out

arm_models = {}
for arm_name in ("H0_only", "H1_only", "H2_only", "burial_only"):
    arm_models[arm_name] = load_seeds(f"{arm_name}_seed", (0,1,2))
arm_models["full"] = load_seeds("loc_deploy_seed", (0,1,2))
arm_models["no_topo"] = load_seeds("notopo_deploy_seed", (0,1,2))

for name, models in arm_models.items():
    print(f"  {name:<14} {len(models)} checkpoints (seeds {sorted(models)})")
assert all(len(m) == 3 for m in arm_models.values()), "expected exactly 3 matched seeds per arm"
print("\nall six arms matched at 3 seeds -- no ensemble-size confound this time")
''')

md(r'''## Cell 4 - Load cached benchmark features, score all 6 arms (matched)''')
code(r'''
with open(locate("phase15_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)
print(f"loaded cached features for {len(raw_cache):,} chains")

@torch.no_grad()
def predict(entry, model_dict, mask=None):
    x = torch.from_numpy(entry["x"]).clone()
    if mask is None:
        x[:, N_PRE:] = 0.0
    else:
        t = x[:, N_PRE:].clone()
        keep = torch.from_numpy(mask) if mask is not True else torch.ones(t.size(1), dtype=torch.bool)
        t = (t - torch.from_numpy(mu_b)) / torch.from_numpy(sd_b)
        t[:, ~keep] = 0.0
        x[:, N_PRE:] = t
    d = type("D", (), {})()
    d.x=x.to(DEVICE); d.pos=torch.from_numpy(entry["pos"]).to(DEVICE)
    d.edge_index=torch.from_numpy(entry["edge_index"]).to(DEVICE)
    d.edge_attr=torch.from_numpy(entry["edge_attr"]).to(DEVICE)
    d.batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
    offset = entry["ca"].mean(axis=0)
    seed_logits, seed_centers = [], []
    for m in model_dict.values():
        lg, pc = m(d)
        seed_logits.append(torch.sigmoid(lg).cpu().numpy())
        seed_centers.append(pc.cpu().numpy()[0])
    return float(np.mean(seed_logits, axis=0).sum()), np.mean(seed_centers, axis=0) + offset


def dcc(pred_center, ligs):
    if pred_center is None or not ligs: return None
    best = np.inf
    for lig_atoms in ligs:
        if len(lig_atoms) == 0: continue
        best = min(best, float(np.linalg.norm(pred_center - lig_atoms.mean(axis=0))))
    return best if np.isfinite(best) else None


by_structure = defaultdict(list)
for key, entry in raw_cache.items():
    by_structure[key[:2]].append((key[2], entry))

ARM_MASK_FOR_SCORING = {**MASKS, "full": True, "no_topo": None}

import time
t0 = time.time()
print(f"{'arm':<14}{'DCC (3-seed matched)':>22}{'n':>6}")
print("-"*42)
summary = {}
for arm_name, mask in ARM_MASK_FOR_SCORING.items():
    models = arm_models[arm_name]
    hits, n = 0, 0
    for key, chain_list in by_structure.items():
        ligs = raw_cache[(key[0], key[1], chain_list[0][0])]["ligs"]
        best_conf, best_center = -np.inf, None
        for cid, entry in chain_list:
            conf, center = predict(entry, models, mask)
            if conf > best_conf: best_conf, best_center = conf, center
        d_ = dcc(best_center, ligs)
        if d_ is None: continue
        n += 1; hits += int(d_ < 4.0)
    rate = hits / max(n, 1)
    summary[arm_name] = {"dcc": rate, "n": n}
    print(f"{arm_name:<14}{rate:>22.4f}{n:>6}")
print(f"\nscored all 6 arms in {(time.time()-t0)/60:.1f} min (CPU)")
''')

md(r'''## Cell 5 - Matched ranking''')
code(r'''
base = summary["no_topo"]["dcc"]
full = summary["full"]["dcc"]
total_gain = full - base

print(f"no_topo baseline: {base:.4f}")
print(f"full (all topo):  {full:.4f}  (total gain: {total_gain:+.4f})\n")

ranked = sorted(("H0_only","H1_only","H2_only","burial_only"), key=lambda n: -summary[n]["dcc"])
print(f"{'arm':<14}{'DCC':>10}{'gain over no_topo':>20}{'% of full gain':>18}")
print("-"*62)
for name in ranked:
    g = summary[name]["dcc"] - base
    pct = 100 * g / total_gain if total_gain > 1e-9 else float("nan")
    print(f"{name:<14}{summary[name]['dcc']:>10.4f}{g:>+20.4f}{pct:>17.1f}%")

print(f"\nmatched-3-seed ranking: {' > '.join(ranked)}")
print("(compare against Phase23 v1's 3-seed unmatched ranking: H0 > H2 > H1 > burial,")
print(" and Phase23 v2's 8-seed MISMATCHED ranking: burial > H2 > H1 > H0)")

json.dump({"summary": summary, "base": base, "full": full, "total_gain": total_gain,
           "ranked": ranked}, open("/kaggle/working/phase25_summary.json", "w"), indent=2)
print("\nsaved -> phase25_summary.json")
''')

md(r'''
---

This is a genuinely matched comparison (3 seeds, every arm, identical
methodology) -- more trustworthy than either prior pass, though still not the
full 8-seed rigor Phase 22 gave the headline topology-vs-no-topology finding.
If this ranking agrees with either prior pass, that convergent evidence is
worth noting. If it disagrees with both, the honest conclusion is that this
specific decomposition question needs the full 8-seed matched treatment
(queued separately) before any ranking can be trusted.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase25_matched_3seed.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
