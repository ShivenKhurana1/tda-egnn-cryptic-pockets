#!/usr/bin/env python3
"""Phase 17: prediction visualizations - reuses Phase 16's cached data, no re-fetch."""
import json
from pathlib import Path

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 17 - Visualizing Predictions

Renders 3D views of the trained model's per-residue binding-site probabilities
and predicted pocket center against the true ligand, on real COACH420/HOLO4K
structures the model has never been trained on.

Reuses Phase 15's cached raw features and Phase 16's trained checkpoints -
no re-fetching or re-featurization, so this runs in a couple of minutes.

**Selection is deliberately not cherry-picked for best cases only.** Examples
are chosen to span the actual outcome distribution - clear hits, a borderline
case, and a genuine miss - because showing only successes would misrepresent
what Phase 16's 53% pooled DCC success rate actually looks like.
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
import matplotlib.pyplot as plt

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

md(r'''## Cell 2 - Model + trained checkpoints (identical to Phase 16)''')
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
print(f"loaded {len(ensemble)} trained checkpoints")
''')

md(r'''## Cell 3 - Inference (same frame-corrected logic as Phase 16)''')
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


def dcc_dca(pred_center, ligs):
    best_dcc, best_dca, best_lig = np.inf, np.inf, None
    for lig_atoms in ligs:
        if len(lig_atoms) == 0: continue
        com = lig_atoms.mean(axis=0)
        dcc = float(np.linalg.norm(pred_center - com))
        if dcc < best_dcc:
            best_dcc = dcc; best_lig = lig_atoms
            best_dca = float(np.linalg.norm(lig_atoms - pred_center, axis=1).min())
    return best_dcc, best_dca, best_lig


by_structure = defaultdict(list)
for (src_,pid,cid), entry in raw_cache.items():
    by_structure[(src_,pid)].append((cid, entry))

records = []
for (src_, pid), chain_list in by_structure.items():
    ligs = chain_list[0][1]["ligs"]
    best_conf, best_center, best_probs, best_entry = -np.inf, None, None, None
    for cid, entry in chain_list:
        probs, center = predict_chain(entry)
        conf = float(probs.sum())
        if conf > best_conf:
            best_conf, best_center, best_probs, best_entry = conf, center, probs, entry
    dcc, dca, true_lig = dcc_dca(best_center, ligs)
    records.append({"src": src_, "pid": pid, "dcc": dcc, "dca": dca,
                    "pred_center": best_center, "true_lig": true_lig,
                    "ca_raw": best_entry["ca"], "probs": best_probs})

n_before = len(records)
records = [r for r in records if np.isfinite(r["dcc"]) and r["true_lig"] is not None]
records.sort(key=lambda r: r["dcc"])
print(f"scored {n_before} structures, {len(records)} with a valid matched ligand "
      f"({n_before - len(records)} excluded: no usable ligand atoms found)")
print(f"median DCC: {np.median([r['dcc'] for r in records]):.2f} A")
''')

md(r'''
## Cell 4 - Pick honest examples: best, median, and a genuine miss

Not cherry-picked for best-case-only. One clean hit, one borderline case near
the 4 Angstrom threshold, and one real miss - the actual range of outcomes.
''')
code(r'''
n = len(records)
selection = {
    "clear_hit_1":   records[0],
    "clear_hit_2":   records[max(1, n//20)],
    "borderline":    min(records, key=lambda r: abs(r["dcc"] - 4.0)),
    "typical":       records[n // 2],
    "miss":          records[-1],
}
for name, r in selection.items():
    print(f"  {name:<14} {r['src']}/{r['pid']}  DCC={r['dcc']:.2f}A  DCA={r['dca']:.2f}A  "
          f"n_res={len(r['probs'])}")
''')

md(r'''## Cell 5 - Render''')
code(r'''
def plot_chain(ax, r, title):
    ca = r["ca_raw"]; probs = r["probs"]
    pred_c = r["pred_center"]; true_lig = r["true_lig"]
    true_c = true_lig.mean(axis=0)

    ax.plot(ca[:,0], ca[:,1], ca[:,2], lw=0.7, alpha=0.35, color="gray", zorder=1)
    order = np.argsort(probs)                      # draw high-probability points on top
    sc = ax.scatter(ca[order,0], ca[order,1], ca[order,2],
                    c=probs[order], cmap="viridis", s=14 + 40*probs[order],
                    vmin=0, vmax=max(0.5, probs.max()), alpha=0.85, zorder=2,
                    edgecolors="none")
    ax.scatter(*true_lig.T, s=10, alpha=0.5, color="gold", marker="o", zorder=3, label="true ligand atoms")
    ax.scatter(*true_c, s=260, color="gold", marker="*", edgecolors="black",
              linewidths=1.2, zorder=5, label="true ligand center")
    ax.scatter(*pred_c, s=220, color="red", marker="X", edgecolors="black",
              linewidths=1.2, zorder=5, label="predicted center")

    hit = "HIT" if r["dcc"] < 4.0 else "MISS"
    ax.set_title(f"{title}\n{r['src']}/{r['pid']}  DCC={r['dcc']:.1f}A ({hit})", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    return sc


fig = plt.figure(figsize=(18, 11))
titles = {"clear_hit_1": "Clear hit", "clear_hit_2": "Clear hit (2)",
         "borderline": "Borderline (~4A threshold)", "typical": "Typical (median)", "miss": "Miss"}
for i, (key, r) in enumerate(selection.items()):
    ax = fig.add_subplot(2, 3, i+1, projection="3d")
    sc = plot_chain(ax, r, titles[key])
    if i == 0:
        ax.legend(loc="upper left", fontsize=7, bbox_to_anchor=(-0.15, 1.05))

cbar_ax = fig.add_axes([0.70, 0.08, 0.2, 0.02])
fig.colorbar(sc, cax=cbar_ax, orientation="horizontal", label="predicted binding-site probability")
fig.suptitle("TopologicalEGNN pocket predictions on COACH420/HOLO4K (external, unseen structures)",
            fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0.06, 1, 0.96])
plt.savefig("/kaggle/working/predictions_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("saved -> predictions_overview.png")
''')

md(r'''## Cell 6 - One large, detailed view of the best hit''')
code(r'''
best = selection["clear_hit_1"]
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
sc = plot_chain(ax, best, "Best prediction, full detail")
ax.legend(loc="upper left", fontsize=9)
plt.colorbar(sc, ax=ax, shrink=0.6, label="predicted probability")
plt.tight_layout()
plt.savefig("/kaggle/working/best_prediction_detail.png", dpi=180, bbox_inches="tight")
plt.show()
print("saved -> best_prediction_detail.png")
''')

md(r'''
---

These are real predictions on structures the model never trained on - COACH420
and HOLO4K chains that survived the leakage filter, run through the exact
checkpoints scored in Phase 16. Gold markers are ground truth (from the actual
crystal structure's bound ligand), red is the model's prediction, and point
color/size is the per-residue binding-site probability the classification head
assigns.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase17_visualize_predictions.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
