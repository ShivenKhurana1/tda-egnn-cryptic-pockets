#!/usr/bin/env python3
"""Phase 8: hyperparameter sweep + fusion-gate analysis."""
import json
from pathlib import Path

C = []
md = lambda s: C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code = lambda s: C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 8 - Capacity Sweep and Fusion-Gate Analysis

Two loose ends from the main result.

**1. The architecture was never tuned.** `h_dim=128, 4 layers, dropout 0.1` was
chosen once and frozen deliberately - changing the model while changing the
evaluation protocol would have confounded the ablation. That reason is now spent,
and Phase 5 showed severe overfitting (train PR 0.97 vs val 0.43 at small scale).

**2. The fusion gates sat at ~0.50** - sigmoid's initialisation value - in the
41-chain run, meaning the gating mechanism designed to make topology use
*interpretable* was not visibly learning anything. With 50× more data it may
behave differently, and that is worth measuring rather than assuming.

### Protocol

All selection is on the **validation split only**. The test split is not touched
here - reporting a tuned configuration's test score selected *on* test would be
the exact leakage the last three phases worked to eliminate. The winning config
should be re-run through the Phase 6c ablation to get a clean test number.
''')

md(r'''## Cell 1 - Setup''')

code(r'''
!pip install -q torch-geometric==2.6.1
''')

code(r'''
import copy, itertools, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"),Path("/kaggle/working"),Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c,key=lambda p:p.stat().st_size)

blob=torch.load(locate("phase6b_graphs.pt"),map_location="cpu",weights_only=False)
dataset=blob["dataset"]; TOPO_NAMES=blob["topo_names"]; TOPO_DIM=len(TOPO_NAMES)
N_SEQ=int(dataset[0].x.size(1))-TOPO_DIM; POS_WEIGHT=float(blob["pos_weight"])
GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU {torch.cuda.get_device_name(0)} probe PASS")
    except RuntimeError as e: print("GPU unusable:",e)
DEVICE=torch.device("cuda" if GPU_OK else "cpu")
splits={s:[d for d in dataset if d.split==s] for s in ("train","val","test")}
print(f"train {len(splits['train'])} | val {len(splits['val'])} | test {len(splits['test'])} (untouched)")
''')

md(r'''## Cell 2 - Model (parameterised)''')

code(r'''
class EGCL(nn.Module):
    def __init__(s,h,ea=2,act=nn.SiLU,uc=True):
        super().__init__(); s.uc=uc
        s.edge_mlp=nn.Sequential(nn.Linear(2*h+1+ea,h),act(),nn.Linear(h,h),act())
        s.node_mlp=nn.Sequential(nn.Linear(2*h,h),act(),nn.Linear(h,h))
        s.att=nn.Sequential(nn.Linear(h,1),nn.Sigmoid())
        if uc:
            s.coord_mlp=nn.Sequential(nn.Linear(h,h),act(),nn.Linear(h,1,bias=False))
            nn.init.xavier_uniform_(s.coord_mlp[-1].weight,gain=0.001)
        s.norm=nn.LayerNorm(h)
    def forward(s,h,x,ei,ea):
        src,dst=ei; rel=x[src]-x[dst]; d2=(rel**2).sum(-1,keepdim=True)
        m=s.edge_mlp(torch.cat([h[src],h[dst],d2,ea],-1)); m=m*s.att(m)
        agg=torch.zeros_like(h).index_add_(0,dst,m)
        deg=torch.zeros(h.size(0),1,device=h.device).index_add_(0,dst,torch.ones_like(d2)).clamp(min=1)
        h=s.norm(h+s.node_mlp(torch.cat([h,agg/deg],-1)))
        if s.uc:
            w=s.coord_mlp(m)*rel/(d2.sqrt()+1.0)
            x=x+torch.zeros_like(x).index_add_(0,dst,w)/deg
        return h,x

class TopoFusion(nn.Module):
    def __init__(s,h,t):
        super().__init__(); s.proj=nn.Linear(t,h); s.gate=nn.Sequential(nn.Linear(h+t,h),nn.Sigmoid())
    def forward(s,h,t):
        g=s.gate(torch.cat([h,t],-1)); s.last_gate=g.detach(); return h+g*s.proj(t)

class TopologicalEGNN(nn.Module):
    def __init__(s,n_seq,td,h=128,L=4,ea=2,dr=0.1):
        super().__init__(); s.n_seq=n_seq
        s.embed=nn.Sequential(nn.Linear(n_seq+td,h),nn.SiLU(),nn.Linear(h,h))
        s.layers=nn.ModuleList([EGCL(h,ea,uc=(i<L-1)) for i in range(L)])
        s.fusions=nn.ModuleList([TopoFusion(h,td) for _ in range(L)])
        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))
    def forward(s,d):
        h=s.embed(d.x); pos=d.pos; t=d.x[:,s.n_seq:]
        for l,f in zip(s.layers,s.fusions):
            h,pos=l(h,pos,d.edge_index,d.edge_attr); h=f(h,t)
        return s.head(h).squeeze(-1)

def run_epoch(model,loader,opt=None):
    tr=opt is not None; model.train(tr)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT],device=DEVICE))
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
    return {"pr":average_precision_score(y,p),"roc":roc_auc_score(y,p)}
''')

md(r'''
## Cell 3 - Sweep

A deliberately small grid. With ~15 min per configuration, an exhaustive search
is not affordable, and the honest framing is "a coarse sweep around the frozen
default", not "we found the optimum".
''')

code(r'''
GRID=[
  dict(h=128,L=4,dr=0.1,wd=1e-4,lr=1e-3),   # the frozen default, as reference
  dict(h=64, L=4,dr=0.1,wd=1e-4,lr=1e-3),   # less capacity
  dict(h=128,L=4,dr=0.3,wd=1e-4,lr=1e-3),   # more dropout
  dict(h=128,L=4,dr=0.1,wd=1e-2,lr=1e-3),   # more weight decay
  dict(h=128,L=6,dr=0.2,wd=1e-3,lr=1e-3),   # deeper + regularised
  dict(h=192,L=4,dr=0.2,wd=1e-3,lr=1e-3),   # wider + regularised
]
EPOCHS,PATIENCE,BATCH=40,8,16
tl=DataLoader(splits["train"],batch_size=BATCH,shuffle=True)
vl=DataLoader(splits["val"],batch_size=BATCH)

results=[]; t0=time.time()
for gi,g in enumerate(GRID):
    torch.manual_seed(0); np.random.seed(0)
    m=TopologicalEGNN(N_SEQ,TOPO_DIM,h=g["h"],L=g["L"],dr=g["dr"]).to(DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=g["lr"],weight_decay=g["wd"])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    best,bad,best_tr,best_state=-1,0,None,None
    for ep in range(EPOCHS):
        tr=run_epoch(m,tl,opt); v=run_epoch(m,vl); sch.step()
        if v["pr"]>best:
            best,best_tr,bad=v["pr"],tr["pr"],0
            best_state={k:t.clone() for k,t in m.state_dict().items()}
        else:
            bad+=1
            if bad>=PATIENCE: break
    n_par=sum(p.numel() for p in m.parameters())
    results.append({**g,"val_pr":best,"train_pr":best_tr,"gap":best_tr-best,
                    "params":n_par,"epochs":ep+1})
    print(f"  [{gi+1}/{len(GRID)}] h={g['h']} L={g['L']} dr={g['dr']} wd={g['wd']:.0e} "
          f"-> val PR {best:.4f} (train {best_tr:.4f}, gap {best_tr-best:+.3f}, "
          f"{n_par/1e6:.2f}M par, {ep+1} ep) [{(time.time()-t0)/60:.0f}m]")
    if gi==0: base_state=best_state

print(f"\nsweep done in {(time.time()-t0)/60:.1f} min")
results.sort(key=lambda r:-r["val_pr"])
print(f"\n{'h':>5}{'L':>3}{'drop':>6}{'wd':>8}{'params':>9}{'val PR':>9}{'train PR':>10}{'gap':>8}")
for r in results:
    print(f"{r['h']:>5}{r['L']:>3}{r['dr']:>6.1f}{r['wd']:>8.0e}"
          f"{r['params']/1e6:>8.2f}M{r['val_pr']:>9.4f}{r['train_pr']:>10.4f}{r['gap']:>+8.3f}")
best_cfg=results[0]
print(f"\nbest on VAL: h={best_cfg['h']} L={best_cfg['L']} dropout={best_cfg['dr']} "
      f"wd={best_cfg['wd']:.0e} -> val PR {best_cfg['val_pr']:.4f}")
print(f"default was val PR {[r for r in results if r['h']==128 and r['L']==4 and r['dr']==0.1 and r['wd']==1e-4][0]['val_pr']:.4f}")
Path("/kaggle/working/phase8_sweep.json").write_text(json.dumps(results,indent=2,default=float))
''')

md(r'''
## Cell 4 - Fusion-gate analysis

Three questions: are the gates moving off their 0.50 initialisation, do they vary
*per residue* (selective routing) or sit flat (a constant scale factor), and do
binding-site residues gate topology differently from the rest?

A flat gate would mean the mechanism is not doing what it was designed to do -
worth knowing, since the interpretability claim rests on it.
''')

code(r'''
torch.manual_seed(0)
m=TopologicalEGNN(N_SEQ,TOPO_DIM,h=best_cfg["h"],L=best_cfg["L"],dr=best_cfg["dr"]).to(DEVICE)
opt=torch.optim.AdamW(m.parameters(),lr=best_cfg["lr"],weight_decay=best_cfg["wd"])
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
best,bad,state=-1,0,None
for ep in range(EPOCHS):
    run_epoch(m,tl,opt); v=run_epoch(m,vl); sch.step()
    if v["pr"]>best: best,bad,state=v["pr"],0,{k:t.clone() for k,t in m.state_dict().items()}
    else:
        bad+=1
        if bad>=PATIENCE: break
m.load_state_dict(state); m.eval()
print(f"retrained best config: val PR {best:.4f}\n")

gates_pos,gates_neg,per_layer=[],[],[[] for _ in range(best_cfg["L"])]
with torch.no_grad():
    for d in splits["val"][:120]:
        dd=d.clone().to(DEVICE); _=m(dd); y=d.y.numpy().astype(bool)
        for li,f in enumerate(m.fusions):
            g=f.last_gate.cpu().numpy()
            per_layer[li].append(g.mean(1))
            if li==best_cfg["L"]-1:
                gates_pos.append(g.mean(1)[y]); gates_neg.append(g.mean(1)[~y])

print(f"{'layer':<8}{'mean gate':>12}{'std across residues':>22}")
for li,vals in enumerate(per_layer):
    v=np.concatenate(vals)
    print(f"{li:<8}{v.mean():>12.4f}{v.std():>22.4f}")

gp=np.concatenate(gates_pos); gn=np.concatenate(gates_neg)
print(f"\nfinal-layer gate, binding-site residues : {gp.mean():.4f} +-{gp.std():.4f}")
print(f"final-layer gate, non-site residues     : {gn.mean():.4f} +-{gn.std():.4f}")
print(f"difference: {gp.mean()-gn.mean():+.4f}")
allg=np.concatenate([np.concatenate(v) for v in per_layer])
print(f"\ndeviation from 0.5 init: {abs(allg.mean()-0.5):.4f} | "
      f"across-residue spread: {allg.std():.4f}")
print("VERDICT:", "gates are learning selective, per-residue routing"
      if allg.std()>0.05 and abs(allg.mean()-0.5)>0.02
      else "gates remain near init and roughly flat -- the mechanism is acting as a "
           "constant scale, so the interpretability claim is NOT supported")
''')

md(r'''
---

## Using this

The winning configuration is selected on **validation only**. To report a test
number for it, re-run the Phase 6c 10-fold ablation with that configuration -
quoting a test score for a config chosen on test would reintroduce exactly the
leakage the earlier phases removed.

If the gate verdict comes back flat, say so. "We designed a gated fusion for
interpretability and measured that it degenerates to a constant scale factor" is
a real finding about the architecture, and considerably more credible than
claiming interpretability that the numbers do not support.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase8_sweep_gates.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
