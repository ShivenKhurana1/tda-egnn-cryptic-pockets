#!/usr/bin/env python3
"""Phase 27b: rescore-only fix for Phase 27's key-schema bug (by_structure
assumed a 3-tuple key like Phase15's combined-benchmark cache; Phase24's
HOLO4K-only cache actually uses a 2-tuple (pid,cid) key -- IndexError at
scoring, AFTER all 5 new seeds finished training successfully). Rather than
retrain (5 hours), the already-trained seed 3-7 checkpoints were downloaded
from the errored kernel and re-uploaded as a Kaggle DATASET (kernel_sources
can't attach an errored kernel's outputs, but dataset_sources have no such
restriction) -- pure inference here, no training, should take minutes."""
import json
from pathlib import Path

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 27b - Rescore-Only Fix (Thread 2 Seed Validation, No Retraining)

Phase 27 trained all 5 new seeds (3-7) successfully but crashed during
scoring: `by_structure[key[:2]]` assumed the 3-tuple key schema Phase15's
COMBINED COACH420+HOLO4K cache uses, but Phase24's cache is HOLO4K-only and
uses a plain `(pid, cid)` 2-tuple -- `IndexError: tuple index out of range`.

Rather than pay another ~5hr CPU training cost, the 5 already-trained
checkpoints were downloaded from the errored kernel's output (Kaggle still
lets you download files from an errored run even though you can't attach it
as a kernel_source) and re-uploaded as a dataset. This notebook just loads
all 8 seeds (0-2 from Phase24, 3-7 from the new dataset) and rescores --
zero training, should run in minutes even on CPU.
''')

md(r'''## Cell 1 - Setup''')
code(r'''
!pip install -q torch-geometric==2.6.1
''')

code(r'''
import json, pickle, time
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

DEVICE = torch.device("cpu")
print(f"device {DEVICE} (rescoring only -- no training in this notebook)")
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

md(r'''## Cell 3 - Load all 8 seeds (0-2 from Phase24, 3-7 from the new dataset)''')
code(r'''
def load_seeds(prefix, seeds):
    out = {}
    for seed in seeds:
        cands = []
        for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
            if r.exists(): cands.extend(r.rglob(f"{prefix}{seed}.pt"))
        if not cands: continue
        state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
        mm = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE); mm.load_state_dict(state); mm.eval()
        out[seed] = mm
    return out

ensemble8 = load_seeds("holo_clean_deploy_seed", range(8))
print(f"loaded {len(ensemble8)} 'holo_clean_deploy' checkpoints (seeds {sorted(ensemble8)})")
assert len(ensemble8) == 8, "expected all 8 seeds -- check kernel_sources/dataset_sources"

with open(locate("phase24_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)
print(f"loaded cached HOLO4K benchmark features for {len(raw_cache):,} chains")
''')

md(r'''## Cell 4 - Score pooled 8-seed ensemble + per-seed breakdown (key-schema FIXED)''')
code(r'''
@torch.no_grad()
def predict(entry, model_dict):
    x = torch.from_numpy(entry["x"]).clone()
    t = (x[:, N_PRE:] - torch.from_numpy(blob6b["topo_mean"])) / torch.from_numpy(blob6b["topo_std"])
    x[:, N_PRE:] = t
    d = type("D", (), {})()
    d.x=x.to(DEVICE); d.pos=torch.from_numpy(entry["pos"]).to(DEVICE)
    d.edge_index=torch.from_numpy(entry["edge_index"]).to(DEVICE)
    d.edge_attr=torch.from_numpy(entry["edge_attr"]).to(DEVICE)
    d.batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
    offset = entry["ca"].mean(axis=0)
    seed_logits, seed_centers = [], []
    for mm in model_dict.values():
        lg, pc = mm(d)
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
for (pid, cid), entry in raw_cache.items():
    by_structure[pid].append((cid, entry))
print(f"grouped into {len(by_structure):,} structures")

def score_ensemble(model_dict):
    hits, n = 0, 0
    for pid, chain_list in by_structure.items():
        ligs = chain_list[0][1]["ligs"]
        best_conf, best_center = -np.inf, None
        for cid, entry in chain_list:
            conf, center = predict(entry, model_dict)
            if conf > best_conf: best_conf, best_center = conf, center
        d_ = dcc(best_center, ligs)
        if d_ is None: continue
        n += 1; hits += int(d_ < 4.0)
    return hits / max(n, 1), n

for k in (3, 8):
    seeds = tuple(range(k))
    subset = {s: ensemble8[s] for s in seeds if s in ensemble8}
    rate, n = score_ensemble(subset)
    print(f"{k}-seed ensemble DCC: {rate:.4f}  (n={n})")

per_seed = []
for seed in range(8):
    rate, n = score_ensemble({seed: ensemble8[seed]})
    per_seed.append(rate)
    print(f"  seed {seed}: DCC {rate:.4f}")
per_seed = np.array(per_seed)
print(f"\nper-seed mean {per_seed.mean():.4f}  std {per_seed.std():.4f}")
''')

md(r'''## Cell 5 - Compare to the live P2Rank number from Phase 24 (same-run, same 1000 structures)''')
code(r'''
LIVE_P2RANK_DCC = 0.6580
ours_3seed = score_ensemble({s: ensemble8[s] for s in (0,1,2)})[0]
ours_8seed = score_ensemble({s: ensemble8[s] for s in range(8)})[0]

gap_3 = ours_3seed - LIVE_P2RANK_DCC
gap_8 = ours_8seed - LIVE_P2RANK_DCC
noise = per_seed.std()

print(f"ours (3-seed, Phase24 original): {ours_3seed:.4f}   gap to P2Rank: {gap_3:+.4f}")
print(f"ours (8-seed, this notebook):    {ours_8seed:.4f}   gap to P2Rank: {gap_8:+.4f}")
print(f"per-seed noise (std): {noise:.4f}")
print(f"8-seed gap / noise ratio: {abs(gap_8)/noise:.2f}x" if noise > 1e-9 else "noise ~0")

json.dump({"ours_3seed": ours_3seed, "ours_8seed": ours_8seed, "live_p2rank": LIVE_P2RANK_DCC,
           "gap_3seed": gap_3, "gap_8seed": gap_8, "per_seed": per_seed.tolist(),
           "noise_std": float(noise)},
          open("/kaggle/working/phase27b_summary.json", "w"), indent=2)
print("\nsaved -> phase27b_summary.json")
''')

md(r'''
---

Rescore-only recovery of Phase 27's already-completed training -- same
methodology and conclusion target as Phase 21's COACH420 seed-variance check,
just reached via a cheaper path after the key-schema bug.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase27b_holo4k_rescore.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
