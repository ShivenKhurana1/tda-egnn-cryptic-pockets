#!/usr/bin/env python3
"""Phase 151: COACH420 round-3, 8-seed DCA re-eval. Phase56 already
computed 8-seed DCC (full8_dcc=0.5932) but its dcc_rate() helper
discarded the DCA half of dcc_dca()'s return value. All 8 checkpoints
(clean_deploy_scaled3_seed0..7.pt, from Phase53+Phase56) and the cached
raw features (phase53_raw_features.pkl, includes full ligand atoms)
already exist -- this is pure re-inference, no retraining."""
import json
from pathlib import Path

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 151 — COACH420 Round-3, 8-Seed: DCA Re-Eval

Phase56 computed 8-seed DCC (0.5932) on the round-3 pool but its
dcc_rate() helper discarded the DCA (distance-to-nearest-ligand-atom,
the looser companion metric) half of dcc_dca()'s return value. All 8
checkpoints and the cached raw features (with full ligand atom
coordinates) already exist on Kaggle -- this is pure re-inference to
recover the DCA number, no retraining.
''')

md(r'''## Cell 1 — Install dependencies''')
code(r'''!pip install -q torch-geometric==2.6.1 biopython==1.84''')

md(r'''## Cell 2 — Setup: load Phase 52's graphs blob (N_PRE/TOPO_DIM) and Phase 51's topology stats (mu/sd)''')
code(r'''
import json, pickle, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax

warnings.filterwarnings("ignore")

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

blob = torch.load(locate("phase52_graphs.pt"), map_location="cpu", weights_only=False)
N_PRE = int(blob["n_pre_topo"]); TOPO_DIM = int(blob["topo_dim"])
blob6b = torch.load(locate("phase51_graphs.pt"), map_location="cpu", weights_only=False)
mu, sd = blob6b["topo_mean"], blob6b["topo_std"]

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
print(f"device {DEVICE} | N_PRE {N_PRE} | TOPO_DIM {TOPO_DIM}")
''')

md(r'''## Cell 3 — Model (identical to every prior localization phase)''')
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

md(r'''## Cell 4 — Load all 8 checkpoints directly (no retraining -- Phase53 gave seeds 0-2, Phase56 gave seeds 3-7)''')
code(r'''
models = {}
for seed in range(8):
    cands = []
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): cands.extend(r.rglob(f"clean_deploy_scaled3_seed{seed}.pt"))
    assert cands, f"clean_deploy_scaled3_seed{seed}.pt not found"
    state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
    m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE); m.load_state_dict(state); m.eval()
    models[seed] = m
print(f"loaded {len(models)} checkpoints (seeds 0-7), no retraining")

with open(locate("phase53_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)
print(f"loaded cached (round-3 pool) COACH420 features for {len(raw_cache):,} chains")
''')

md(r'''## Cell 5 — Per-seed and 8-seed-ensemble DCC AND DCA (fixes Phase56's dropped DCA)''')
code(r'''
import time

def dcc_dca(pred_center, ligs):
    if pred_center is None or not ligs: return None, None
    best_dcc, best_dca = np.inf, np.inf
    for lig_atoms in ligs:
        if len(lig_atoms) == 0: continue
        com = lig_atoms.mean(axis=0)
        d_ = float(np.linalg.norm(pred_center - com))
        if d_ < best_dcc:
            best_dcc = d_
            best_dca = float(np.linalg.norm(lig_atoms - pred_center, axis=1).min())
    return (best_dcc if np.isfinite(best_dcc) else None, best_dca if np.isfinite(best_dca) else None)


@torch.no_grad()
def predict_centers_all_seeds(entry):
    x = torch.from_numpy(entry["x"]).clone()
    x[:, N_PRE:] = (x[:, N_PRE:] - torch.from_numpy(mu)) / torch.from_numpy(sd)
    d = type("D", (), {})()
    d.x=x.to(DEVICE); d.pos=torch.from_numpy(entry["pos"]).to(DEVICE)
    d.edge_index=torch.from_numpy(entry["edge_index"]).to(DEVICE)
    d.edge_attr=torch.from_numpy(entry["edge_attr"]).to(DEVICE)
    d.batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
    offset = entry["ca"].mean(axis=0)
    out = {}
    for seed, m in models.items():
        _, pc = m(d)
        out[seed] = pc.cpu().numpy()[0] + offset
    return out


t0 = time.time()
per_chain_centers, per_chain_ligs = {}, {}
for key, entry in raw_cache.items():
    per_chain_centers[key] = predict_centers_all_seeds(entry)
    per_chain_ligs[key] = entry["ligs"]
print(f"inference (8 seeds x {len(raw_cache)} chains) in {(time.time()-t0)/60:.1f} min")


def dcc_dca_rate(center_fn):
    hits_dcc, hits_dca, n = 0, 0, 0
    for key in raw_cache:
        c = center_fn(key)
        if c is None: continue
        d_, a_ = dcc_dca(c, per_chain_ligs[key])
        if d_ is None: continue
        n += 1; hits_dcc += int(d_ < 4.0); hits_dca += int(a_ < 4.0)
    return hits_dcc / max(n, 1), hits_dca / max(n, 1), n


print(f"\n{'per-seed DCC / DCA (individual models, no ensembling)':<60}")
print("-"*60)
seed_dccs, seed_dcas = [], []
for seed in range(8):
    dcc, dca, n = dcc_dca_rate(lambda k, s=seed: per_chain_centers[k][s])
    seed_dccs.append(dcc); seed_dcas.append(dca)
    print(f"  seed {seed}: DCC {dcc:.4f}  DCA {dca:.4f}  (n={n})")

full8_dcc, full8_dca, full8_n = dcc_dca_rate(lambda k: np.mean(list(per_chain_centers[k].values()), axis=0))
print(f"\nfull 8-seed ensemble: DCC {full8_dcc:.4f}  DCA {full8_dca:.4f}  (n={full8_n})")
print(f"(Phase56 reported DCC {0.5932:.4f} for this same ensemble -- {'MATCHES' if abs(full8_dcc-0.5932)<0.001 else 'MISMATCH, investigate'})")

P2RANK_THIS_RUN = 0.520885
LIT_DCA = {"fpocket": 0.444, "P2Rank": 0.628, "EquiPocket": 0.656}
print(f"\n{'method':<32}{'DCC':>8}{'DCA':>8}")
print("-"*48)
for name, dca in LIT_DCA.items():
    print(f"{name+' (literature, full 420)':<32}{'--':>8}{dca:>8.3f}")
print(f"{'P2Rank (this run, DCC only)':<32}{P2RANK_THIS_RUN:>8.3f}{'--':>8}")
print(f"{'OURS (8-seed ensemble, round-3)':<32}{full8_dcc:>8.4f}{full8_dca:>8.4f}")

json.dump({"seed_dccs": seed_dccs, "seed_dcas": seed_dcas,
           "full8_dcc": full8_dcc, "full8_dca": full8_dca, "n": full8_n,
           "p2rank_this_run_dcc": P2RANK_THIS_RUN, "literature_dca": LIT_DCA},
          open("/kaggle/working/phase151_dca_summary.json", "w"), indent=2)
print("\nsaved -> phase151_dca_summary.json")
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase151_coach420_round3_dca_eval.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
