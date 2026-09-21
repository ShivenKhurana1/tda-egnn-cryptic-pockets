#!/usr/bin/env python3
"""Phase 11b: quick-win accuracy levers on the fixed split - focal loss, seed ensembling."""
import json
from pathlib import Path

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 11b — Quick Accuracy Levers (fixed split, fast iteration)

Two cheap things worth checking while the properly-powered Phase 11 nested CV
runs in parallel. Both reuse `phase10_graphs.pt` directly — no re-embedding,
each variant trains in a few minutes on the existing fixed split, so this is
exploration, not the number to cite.

1. **Focal loss vs weighted BCE.** At ~6.8% positives, standard weighted BCE
   treats every hard example the same once it is correctly up-weighted. Focal
   loss additionally down-weights *easy* examples during training, focusing
   gradient on the boundary cases — often a direct win on PR-AUC for exactly
   this kind of imbalance, and it is a one-line change to the loss function.

2. **Seed ensembling.** Averaging predicted probabilities across independently
   seeded runs of the same architecture almost always reduces variance and
   nudges PR-AUC up a little. Cheap to check, cheap to adopt if it helps.

Whichever combination wins here is a candidate to fold into a future properly-
powered run — not a replacement for one.
''')

md(r'''## Cell 1 — Setup''')

code(r'''
!pip install -q torch-geometric==2.6.1
''')

code(r'''
import json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

blob = torch.load(locate("phase10_graphs.pt"), map_location="cpu", weights_only=False)
dataset = blob["dataset"]; N_PRE = int(blob["n_pre_topo"]); TOPO_DIM = int(blob["topo_dim"])
POS_WEIGHT = float(blob["pos_weight"])
GPU=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU=True
    except RuntimeError: pass
DEVICE = torch.device("cuda" if GPU else "cpu")
splits = {s: [d for d in dataset if d.split==s] for s in ("train","val","test")}
print(f"{DEVICE} | node dim {dataset[0].x.size(1)} | pos_weight {POS_WEIGHT:.2f}")
print({s: len(v) for s,v in splits.items()})
''')

md(r'''## Cell 2 — Model + both loss functions''')

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
    def __init__(s,n_pre,td,h=128,L=4,dr=0.1):
        super().__init__(); s.n_pre=n_pre
        s.embed=nn.Sequential(nn.Linear(n_pre+td,h),nn.SiLU(),nn.Linear(h,h))
        s.ls=nn.ModuleList([EGCL(h,uc=(i<L-1)) for i in range(L)])
        s.fs=nn.ModuleList([Fus(h,td) for _ in range(L)])
        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))
    def forward(s,d):
        h=s.embed(d.x); pos=d.pos; t=d.x[:,s.n_pre:]
        for l,f in zip(s.ls,s.fs): h,pos=l(h,pos,d.edge_index,d.edge_attr); h=f(h,t)
        return s.head(h).squeeze(-1)

class FocalLoss(nn.Module):
    """Binary focal loss with a positive-class weight, on logits."""
    def __init__(s, pos_weight, gamma=2.0):
        super().__init__(); s.pw=pos_weight; s.gamma=gamma
    def forward(s, logits, y):
        p = torch.sigmoid(logits)
        pt = torch.where(y==1, p, 1-p).clamp(min=1e-6, max=1-1e-6)
        w = torch.where(y==1, s.pw, 1.0)
        return (-w * (1-pt).pow(s.gamma) * pt.log()).mean()

def run_epoch(model, loader, crit, opt=None):
    tr = opt is not None; model.train(tr)
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

BCE = lambda: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
FOCAL = lambda: FocalLoss(POS_WEIGHT, gamma=2.0)
''')

md(r'''## Cell 3 — Loss comparison, 3 seeds each''')

code(r'''
EPOCHS, PATIENCE, BATCH = 40, 8, 16
tl = DataLoader(splits["train"], batch_size=BATCH, shuffle=True)
vl = DataLoader(splits["val"], batch_size=BATCH)
te_loader = DataLoader(splits["test"], batch_size=BATCH)

def train_one(crit_fn, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net(N_PRE, TOPO_DIM).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = crit_fn()
    best, state, bad = -1, None, 0
    for ep in range(EPOCHS):
        run_epoch(m, tl, crit, opt); v = run_epoch(m, vl, crit); sch.step()
        if v["pr"] > best: best, state, bad = v["pr"], {k:t.clone() for k,t in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    m.load_state_dict(state); m.eval()
    return m, best

t0=time.time()
loss_results = {}
test_preds = {}   # for ensembling
for loss_name, crit_fn in (("bce", BCE), ("focal", FOCAL)):
    seeded = []
    for seed in (0,1,2):
        m, val_pr = train_one(crit_fn, seed)
        te = run_epoch(m, te_loader, crit_fn())
        mcc = best_mcc(te["p"], te["y"])
        seeded.append({"seed": seed, "val_pr": val_pr, "test_pr": te["pr"],
                       "test_roc": te["roc"], "test_mcc": mcc})
        test_preds[(loss_name, seed)] = te["p"]
        y_ref = te["y"]
        print(f"  [{loss_name}] seed {seed} val PR {val_pr:.4f} test PR {te['pr']:.4f} "
              f"ROC {te['roc']:.4f} MCC {mcc:.4f}  [{(time.time()-t0)/60:.1f}m]")
    loss_results[loss_name] = seeded

print(f"\ndone in {(time.time()-t0)/60:.1f} min")
''')

md(r'''## Cell 4 — Ensemble and verdict''')

code(r'''
print(f"{'config':<20}{'mean test PR':>15}{'std':>8}")
for name, runs in loss_results.items():
    prs = np.array([r["test_pr"] for r in runs])
    print(f"{name:<20}{prs.mean():>15.4f}{prs.std():>8.4f}")

for name in ("bce", "focal"):
    ens = np.mean([test_preds[(name, s)] for s in (0,1,2)], axis=0)
    pr = average_precision_score(y_ref, ens); roc = roc_auc_score(y_ref, ens)
    mcc = best_mcc(ens, y_ref)
    print(f"{name+'-ensemble(3seed)':<20}{pr:>15.4f} (ROC {roc:.4f}, MCC {mcc:.3f})")

# cross-loss ensemble: bce + focal, all 6 runs
all_ens = np.mean([test_preds[k] for k in test_preds], axis=0)
pr_all = average_precision_score(y_ref, all_ens)
print(f"{'bce+focal ensemble(6)':<20}{pr_all:>15.4f}")

REFERENCE = {"P2Rank": 0.4589, "Phase7 EGNN": 0.4209, "Phase10 augmented-full (1 seed)": 0.5002}
print("\nreference points:")
for k,v in REFERENCE.items(): print(f"  {k:<32} {v:.4f}")

best_name = max(loss_results, key=lambda n: np.mean([r["test_pr"] for r in loss_results[n]]))
best_mean = np.mean([r["test_pr"] for r in loss_results[best_name]])
print(f"\nbest single-seed-mean config: {best_name} ({best_mean:.4f})")
recommendation = "focal loss" if best_name=="focal" and best_mean > np.mean([r['test_pr'] for r in loss_results['bce']]) else "standard weighted BCE (focal did not help)"
print(f"recommendation for the next properly-powered run: {recommendation}")

json.dump({"loss_results": loss_results,
           "ensemble_bce": float(average_precision_score(y_ref, np.mean([test_preds[('bce',s)] for s in (0,1,2)],axis=0))),
           "ensemble_focal": float(average_precision_score(y_ref, np.mean([test_preds[('focal',s)] for s in (0,1,2)],axis=0))),
           "ensemble_all6": float(pr_all)},
          open("/kaggle/working/phase11b_results.json","w"), indent=2, default=float)
print("saved -> phase11b_results.json")
''')

md(r'''
---

## Using this

This notebook is exploratory and runs on a single fixed split — treat every
number here as a *hypothesis* about what might help, not a validated result.
If focal loss or ensembling shows a clear, consistent gain across all 3 seeds,
it is a reasonable ingredient to carry into the next properly-powered nested-CV
run (Phase 11 / a future Phase 12) rather than something to report on its own.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase11b_quick_levers.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
