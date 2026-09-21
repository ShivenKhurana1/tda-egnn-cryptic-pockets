#!/usr/bin/env python3
"""Phase 118 -- CryptoBench train+eval WITH the auxiliary "any-pocket"
head, testing whether the noncryptic-pocket data (Phase117) actually
helps. Model gets a THIRD head (aux_head, structurally identical to the
main classification head) predicting y_aux (cryptic OR noncryptic), added
to the loss with weight lam_aux. Sweeps lam_aux in {0.0, 0.1, 0.3} on a
SINGLE seed first (lam_aux=0.0 is the exact Phase42/43 baseline control,
included in the same run for a same-seed, same-run comparison) -- cheap
sanity pass before committing to multi-seed validation of any apparent
winner, per this project's standing (and this session's expensively
re-learned) discipline."""
import json, pickle
from pathlib import Path

C20 = pickle.load(open("/tmp/p20_cells.pkl", "rb"))
s20 = lambda i: (C20[i]["source"] if isinstance(C20[i]["source"], str) else "".join(C20[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 118 — CryptoBench: Does the Auxiliary "Any-Pocket" Loss Help?

Tests Phase117's new `y_aux` label (cryptic OR noncryptic pocket) as an
auxiliary multi-task target. Adds a THIRD head (`aux_head`) to
`LocalizingEGNN`, structurally identical to the existing classification
head, predicting `y_aux` with its own BCE loss term weighted by
`lam_aux`. `lam_aux=0.0` is mathematically identical to Phase42/43's
proven baseline (the aux head still exists and trains, but contributes
nothing to the loss) -- included in the SAME sweep, same seed, same run
for a clean apples-to-apples comparison, avoiding the asymmetric-seed
trap this session already got burned by twice (COACH420 ESM3B 3-seed
reversal, the h128/L6 capacity mirage).
''')

md(r'''## Cell 1 — Setup, load Phase 117's aux-label CryptoBench blob''')
code(r'''
!pip install -q torch-geometric==2.6.1
''')
code(r'''
import json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              matthews_corrcoef, f1_score, accuracy_score)

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

blob = torch.load(locate("phase117_cryptobench_graphs_aux.pt"), map_location="cpu", weights_only=False)
dataset = blob["dataset"]
N_PRE = int(blob["n_pre_topo"]); TOPO_DIM = int(blob["topo_dim"])
POS_WEIGHT = float(blob["pos_weight"]); POS_WEIGHT_AUX = float(blob["pos_weight_aux"])
splits = {s: [d for d in dataset if d.split == s] for s in ("train","val","test")}

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
print(f"device {DEVICE} | node dim {dataset[0].x.size(1)} | pos_weight {POS_WEIGHT:.2f} | "
      f"pos_weight_aux {POS_WEIGHT_AUX:.2f} | " +
      " | ".join(f"{k} {len(v)}" for k,v in splits.items()))

WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)
''')

md(r'''## Cell 2 — Model, with a NEW aux_head (structurally identical to the main head)''')
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
    """Same TopologicalEGNN as every prior phase, plus the localization head,
    plus (new) an optional auxiliary "any-pocket" classification head --
    structurally identical to the main head, predicting a broader/easier
    superset target (y_aux: cryptic OR noncryptic pocket) to test whether
    it helps the shared trunk learn better intermediate features."""
    def __init__(s, n_pre, td, h=128, L=4, dr=0.1, use_loc_head=True, use_aux_head=False):
        super().__init__(); s.n_pre=n_pre; s.use_loc_head=use_loc_head; s.use_aux_head=use_aux_head
        s.embed=nn.Sequential(nn.Linear(n_pre+td,h),nn.SiLU(),nn.Linear(h,h))
        s.ls=nn.ModuleList([EGCL(h,uc=(i<L-1)) for i in range(L)])
        s.fs=nn.ModuleList([Fus(h,td) for _ in range(L)])
        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))
        if use_loc_head:
            s.loc_head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Linear(h//2,1))
        if use_aux_head:
            s.aux_head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))

    def forward(s, d):
        h=s.embed(d.x); pos=d.pos; t=d.x[:,s.n_pre:]
        for l,f in zip(s.ls,s.fs): h,pos=l(h,pos,d.edge_index,d.edge_attr); h=f(h,t)
        logits = s.head(h).squeeze(-1)
        aux_logits = s.aux_head(h).squeeze(-1) if s.use_aux_head else None
        pred_center = None
        if s.use_loc_head:
            batch = d.batch if hasattr(d, "batch") and d.batch is not None \
                    else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            loc_logit = s.loc_head(h).squeeze(-1)
            attn = pyg_softmax(loc_logit, batch)                    # sums to 1 per graph
            weighted = attn.unsqueeze(-1) * pos                     # (N,3)
            n_graphs = int(batch.max().item()) + 1
            pred_center = torch.zeros(n_graphs, 3, device=h.device).index_add_(0, batch, weighted)
        return logits, pred_center, aux_logits
''')

md(r'''## Cell 3 — Training machinery: run_epoch/fit accept lam_aux, model always builds aux_head (lam_aux=0.0 makes it a no-op on the loss, exact Phase42/43 baseline control)''')
code(r'''
def run_epoch(model, loader, opt=None, lam=0.0, lam_aux=0.0):
    tr = opt is not None; model.train(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    aux_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT_AUX], device=DEVICE))
    huber = nn.SmoothL1Loss(beta=4.0)
    ps, ys, cd = [], [], []
    for b in loader:
        b = b.to(DEVICE)
        with torch.set_grad_enabled(tr):
            logits, pred_c, aux_logits = model(b)
            loss = crit(logits, b.y)
            if lam > 0 and pred_c is not None:
                true_c = b.true_center.view(-1, 3)
                loss = loss + lam * huber(pred_c, true_c)
                cd.append((pred_c - true_c).norm(dim=1).detach().cpu().numpy())
            if lam_aux > 0 and aux_logits is not None:
                loss = loss + lam_aux * aux_crit(aux_logits, b.y_aux)
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


def fit(use_loc_head, lam, seed, lam_aux=0.0, epochs=40, patience=8, batch=16):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head, use_aux_head=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl = DataLoader(splits["train"], batch_size=batch, shuffle=True)
    vl = DataLoader(splits["val"], batch_size=batch)
    best, state, bad = -1, None, 0
    for ep in range(epochs):
        run_epoch(model, tl, opt, lam, lam_aux)
        v = run_epoch(model, vl, lam=lam, lam_aux=lam_aux)
        sch.step()
        if v["pr"] > best:
            best, state, bad = v["pr"], {k:t.clone() for k,t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval()
    return model, best
''')

md(r'''## Cell 4 — Sweep lam_aux (single seed=0 each, same run) -- lam_aux=0.0 is the exact baseline control''')
code(r'''
BEST_LAMBDA = 0.1   # localization-loss weight, unchanged from Phase42/43
LAM_AUX_SWEEP = [0.0, 0.1, 0.3]
sweep_results = []
t0 = time.time()
for lam_aux in LAM_AUX_SWEEP:
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, lam_aux=lam_aux, seed=0)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA, lam_aux=lam_aux)
    val_dcc = v.get("dcc", float("nan"))
    torch.save(m.state_dict(), WORK/f"cache/cryptobench_aux_lam{lam_aux}_seed0.pt")
    sweep_results.append(dict(lam_aux=lam_aux, val_pr=val_pr, val_dcc=val_dcc))
    print(f"  lam_aux={lam_aux:<4} seed0: val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")

print(f"\n--- lam_aux sweep (single seed=0, same run) ---")
baseline_pr = next(r["val_pr"] for r in sweep_results if r["lam_aux"] == 0.0)
for r in sweep_results:
    print(f"  lam_aux={r['lam_aux']:<4}  val PR {r['val_pr']:.4f}  ({r['val_pr']-baseline_pr:+.4f} vs lam_aux=0 baseline)  val DCC {r['val_dcc']:.4f}")
json.dump(sweep_results, open(WORK/"phase118_sweep_results.json", "w"), indent=2)
print("\nsaved -> phase118_sweep_results.json")
''')

md(r'''
---

## Reading this

- **Best lam_aux clearly beats lam_aux=0.0 (val PR gain outside typical
  single-seed noise, roughly >0.01-0.015 based on this project's observed
  noise floors elsewhere)** -- worth a proper 3-seed (then 8-seed)
  confirmation on the winning lam_aux, then a full CryptoBench test-set
  evaluation (AUC/AUPRC/MCC, mirroring Phase43), before trusting it.
- **All lam_aux values cluster within noise of the baseline** -- the
  noncryptic-pocket auxiliary signal doesn't help at 35M-ESM scale;
  would still be worth knowing before deciding whether to try it on the
  3B pipeline too, but wouldn't be a promising lead on its own.
- **lam_aux clearly HURTS** -- the auxiliary task may be too easy/
  uncorrelated and is diluting gradient signal from the main task; not
  worth pursuing further with this exact formulation.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase118_cryptobench_train_aux_sweep.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
