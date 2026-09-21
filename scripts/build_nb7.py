#!/usr/bin/env python3
"""Phase 7 (v3): P2Rank baseline comparison."""
import json
from pathlib import Path
C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 7 — Baseline Comparison: `TopologicalEGNN` vs P2Rank

Every result so far compares the model to *its own ablations*. That shows which
inputs matter, not whether the model is any good. "Beats our seq-only arm" does
not answer "better than what?"

**P2Rank** (Krivák & Hoksza, *J. Cheminformatics* 2018) is the current standard
ligand-binding-site predictor: a random forest over solvent-accessible-surface
points, distributed pre-trained. It is a strong, widely-cited baseline and, unlike
a geometric-only method, it is genuinely competitive.

### Why not fpocket

fpocket was attempted first and **segfaults on this image** (exit −11): its Voronoi
stage calls Ubuntu's system qhull 2015.2, which rejects the `Qz Qt` option
combination fpocket passes (`qvoronoi p i Pp Qz Qt`). That is a build/ABI
incompatibility, not a property of the structures. Rather than patch a
third-party build, P2Rank is used — a prebuilt JVM binary with no native
dependency, and a harder baseline to beat.

### Why this comparison is fair

P2Rank emits **per-residue probabilities** in `*_residues.csv`, so no
pocket-to-residue conversion is needed — its output is directly comparable to the
model's per-residue sigmoid, on identical residues with identical labels.

An important caveat to state in any writeup: P2Rank predicts *ligandable* sites
generally, while our labels mark residues near the ligand **present in that
crystal**. A real, unoccupied pocket counts against P2Rank here. The comparison
is therefore mildly favourable to the model, and the equal-selectivity block
below is the fairest single number.
''')

md(r'''## Cell 1 — Java + P2Rank''')

code(r'''
!pip install -q torch-geometric==2.6.1 biopython==1.84
!apt-get -qq update > /dev/null 2>&1 && apt-get -qq install -y default-jre-headless > /dev/null 2>&1
''')

code(r'''
import gzip, json, os, shutil, subprocess, time, urllib.request, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch, torch.nn as nn
warnings.filterwarnings("ignore")

print(subprocess.run(["java","-version"],capture_output=True,text=True).stderr.splitlines()[0])

t0=time.time(); PRANK=None
try:
    subprocess.run("curl -sL https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz "
                   "-o /tmp/p2.tar.gz && tar xzf /tmp/p2.tar.gz -C /tmp", shell=True, check=True, timeout=1200)
    cand=next(Path("/tmp").glob("p2rank_*/prank"))
    os.chmod(cand,0o755); PRANK=str(cand)
except Exception as e:
    raise RuntimeError(f"P2Rank unavailable: {type(e).__name__}: {e}")
print(f"P2Rank ready in {time.time()-t0:.0f}s -> {PRANK}")
''')

md(r'''## Cell 2 — Train the model on the fixed split''')

code(r'''
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"),Path("/kaggle/working"),Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c,key=lambda p:p.stat().st_size)

blob=torch.load(locate("phase6b_graphs.pt"),map_location="cpu",weights_only=False)
dataset=blob["dataset"]; TOPO_NAMES=blob["topo_names"]; TOPO_DIM=len(TOPO_NAMES)
N_SEQ=int(dataset[0].x.size(1))-TOPO_DIM; POS_WEIGHT=float(blob["pos_weight"])
GPU=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU=True
    except RuntimeError: pass
DEVICE=torch.device("cuda" if GPU else "cpu")
splits={s:[d for d in dataset if d.split==s] for s in ("train","val","test")}
print(f"{DEVICE} | train {len(splits['train'])} val {len(splits['val'])} test {len(splits['test'])}")

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
    def __init__(s,ns,td,h=128,L=4):
        super().__init__(); s.ns=ns
        s.emb=nn.Sequential(nn.Linear(ns+td,h),nn.SiLU(),nn.Linear(h,h))
        s.ls=nn.ModuleList([EGCL(h,uc=(i<L-1)) for i in range(L)])
        s.fs=nn.ModuleList([Fus(h,td) for _ in range(L)])
        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(0.1),nn.Linear(h//2,1))
    def forward(s,d):
        h=s.emb(d.x); pos=d.pos; t=d.x[:,s.ns:]
        for l,f in zip(s.ls,s.fs): h,pos=l(h,pos,d.edge_index,d.edge_attr); h=f(h,t)
        return s.head(h).squeeze(-1)

def ep(m,loader,opt=None):
    tr=opt is not None; m.train(tr)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT],device=DEVICE))
    ps,ys=[],[]
    for b in loader:
        b=b.to(DEVICE)
        with torch.set_grad_enabled(tr):
            lg=m(b); loss=crit(lg,b.y)
        if tr:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        ps.append(torch.sigmoid(lg).detach().cpu().numpy()); ys.append(b.y.cpu().numpy())
    p,y=np.concatenate(ps),np.concatenate(ys)
    return {"pr":average_precision_score(y,p),"roc":roc_auc_score(y,p)}

torch.manual_seed(0)
model=Net(N_SEQ,TOPO_DIM).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40)
tl=DataLoader(splits["train"],batch_size=16,shuffle=True); vl=DataLoader(splits["val"],batch_size=16)
best,st,bad=-1,None,0
for e in range(40):
    ep(model,tl,opt); v=ep(model,vl); sch.step()
    if v["pr"]>best: best,st,bad=v["pr"],{k:t.clone() for k,t in model.state_dict().items()},0
    else:
        bad+=1
        if bad>=8: break
model.load_state_dict(st); model.eval()
print(f"val PR {best:.4f}")
model_pred={}
with torch.no_grad():
    for d in splits["test"]:
        model_pred[(d.pdb_id,d.chain_id)]=torch.sigmoid(model(d.clone().to(DEVICE))).cpu().numpy()
print(f"predictions for {len(model_pred)} test chains")
''')

md(r'''## Cell 3 — Run P2Rank in batch mode''')

code(r'''
RAW=Path("/tmp/pdb_raw"); RAW.mkdir(parents=True,exist_ok=True)
W=Path("/tmp/p2in"); shutil.rmtree(W,ignore_errors=True); W.mkdir()
OUT=Path("/tmp/p2out"); shutil.rmtree(OUT,ignore_errors=True)

need=sorted({d.pdb_id for d in splits["test"]})
def fetch(pid):
    p=RAW/f"{pid}.pdb.gz"
    if p.exists() and p.stat().st_size>500: return pid,p
    try:
        with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.pdb.gz",timeout=60) as r:
            b=r.read()
        if len(b)>500: p.write_bytes(b); return pid,p
    except Exception: pass
    return pid,None
with ThreadPoolExecutor(max_workers=16) as ex: got=list(ex.map(fetch,need))
paths={p:f for p,f in got if f}
for pid,gz in paths.items():
    plain=W/f"{pid}.pdb"
    if not plain.exists():
        with gzip.open(gz,"rb") as fi, open(plain,"wb") as fo: shutil.copyfileobj(fi,fo)
print(f"prepared {len(paths)} structures")

ds_file=W/"test.ds"
ds_file.write_text("\n".join(str(W/f"{p}.pdb") for p in sorted(paths))+"\n")

# probe one structure first so a systematic failure surfaces immediately
probe=W/"probe.ds"; probe.write_text(str(W/f"{sorted(paths)[0]}.pdb")+"\n")
r=subprocess.run([PRANK,"predict",str(probe),"-o",str(OUT/"probe"),"-threads","2"],
                 capture_output=True,text=True,timeout=900)
print(f"probe rc={r.returncode}")
print((r.stdout or "")[-600:])
if r.returncode!=0: print("STDERR:",(r.stderr or "")[-800:])
probe_files=sorted(x.name for x in (OUT/"probe").rglob("*.csv"))
print("probe csv outputs:",probe_files[:6])
assert probe_files, "P2Rank produced no CSV -- inspect the output above"
''')

code(r'''
t0=time.time()
r=subprocess.run([PRANK,"predict",str(ds_file),"-o",str(OUT/"all"),
                  "-threads",str(max(1,os.cpu_count() or 2))],
                 capture_output=True,text=True,timeout=7200)
print(f"P2Rank rc={r.returncode} in {(time.time()-t0)/60:.1f} min")
print((r.stdout or "")[-800:])
csvs={p.name:p for p in (OUT/"all").rglob("*_residues.csv")}
print(f"residue-level outputs: {len(csvs)}")
if not csvs:
    print("no *_residues.csv; falling back to predictions.csv residue_ids")
    csvs={p.name:p for p in (OUT/"all").rglob("*_predictions.csv")}
    print(f"pocket-level outputs: {len(csvs)}")
''')

md(r'''## Cell 4 — Align and compare''')

code(r'''
import csv as _csv
from Bio.PDB import PDBParser
AA3={"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
     "MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE"}

def p2rank_scores(pid):
    """-> {(chain, resnum): probability}"""
    f=(OUT/"all"/f"{pid}.pdb_residues.csv")
    if not f.exists():
        cand=[p for n,p in csvs.items() if n.startswith(pid)]
        if not cand: return None
        f=cand[0]
    out={}
    with open(f) as fh:
        for row in _csv.DictReader(fh):
            row={k.strip():(v.strip() if isinstance(v,str) else v) for k,v in row.items() if k}
            ch=row.get("chain","")
            lab=row.get("residue_label","")
            sc=row.get("probability") or row.get("zscore") or row.get("score") or "0"
            num="".join(c for c in lab if c.isdigit() or c=="-")
            if not ch or not num: continue
            try: out[(ch,int(num))]=float(sc)
            except ValueError: continue
    return out or None

def residue_keys(pid,cid):
    p=paths.get(pid)
    if p is None: return None
    try:
        with gzip.open(p,"rt") as fh:
            mdl=next(PDBParser(QUIET=True,PERMISSIVE=True).get_structure(pid,fh).get_models())
        ch=mdl[cid]
    except Exception: return None
    return [(cid,r.id[1]) for r in ch
            if r.get_resname().strip().upper() in AA3 and "CA" in r]

rows=[]; skipped=0
for d in splits["test"]:
    sc=p2rank_scores(d.pdb_id)
    keys=residue_keys(d.pdb_id,d.chain_id)
    if sc is None or keys is None or len(keys)!=int(d.num_nodes): skipped+=1; continue
    p2=np.array([sc.get(k,0.0) for k in keys],dtype=np.float32)
    if p2.max()<=0: skipped+=1; continue
    rows.append({"y":d.y.numpy(),"m":model_pred[(d.pdb_id,d.chain_id)],"p2":p2})

y=np.concatenate([r["y"] for r in rows]); mp=np.concatenate([r["m"] for r in rows])
p2=np.concatenate([r["p2"] for r in rows])
print(f"compared on {len(rows)} chains / {len(y):,} residues "
      f"({int(y.sum()):,} pos, {100*y.mean():.2f}%) | skipped {skipped}\n")

def prf(pred,y):
    tp=int(((pred==1)&(y==1)).sum())
    pr=tp/max(pred.sum(),1); rc=tp/max(y.sum(),1)
    return pr,rc,2*pr*rc/max(pr+rc,1e-9),matthews_corrcoef(y,pred)

print(f"{'method':<22}{'PR-AUC':>10}{'ROC-AUC':>10}{'prec':>8}{'rec':>8}{'F1':>8}{'MCC':>8}")
print("-"*74)
print(f"{'random':<22}{y.mean():>10.4f}{0.5:>10.4f}")
for nm,s in (("P2Rank",p2),("TopologicalEGNN",mp)):
    bf,bt=0,0.5
    for t in np.linspace(0.02,0.98,97):
        pd_=(s>=t).astype(int)
        if pd_.sum():
            f=prf(pd_,y)[2]
            if f>bf: bf,bt=f,t
    pr,rc,f1,mcc=prf((s>=bt).astype(int),y)
    print(f"{nm:<22}{average_precision_score(y,s):>10.4f}{roc_auc_score(y,s):>10.4f}"
          f"{pr:>8.3f}{rc:>8.3f}{f1:>8.3f}{mcc:>8.3f}")

print("\n--- equal selectivity (same fraction of residues called positive) ---")
for frac in (0.05,0.10,0.20):
    line=[]
    for nm,s in (("P2Rank",p2),("EGNN",mp)):
        thr=np.quantile(s,1-frac); pr,rc,f1,mcc=prf((s>=thr).astype(int),y)
        line.append(f"{nm} prec {pr:.3f} rec {rc:.3f} MCC {mcc:.3f}")
    print(f"  top {int(frac*100):>2}%: " + " | ".join(line))

# per-chain paired comparison
per=[(average_precision_score(r["y"],r["p2"]) if r["y"].sum() else np.nan,
      average_precision_score(r["y"],r["m"]) if r["y"].sum() else np.nan) for r in rows]
per=np.array([x for x in per if not np.isnan(x).any()])
d_=per[:,1]-per[:,0]
from scipy.stats import wilcoxon
print(f"\nper-chain PR-AUC: P2Rank {per[:,0].mean():.4f} | EGNN {per[:,1].mean():.4f}")
print(f"paired diff {d_.mean():+.4f} +-{d_.std():.4f} | EGNN wins {int((d_>0).sum())}/{len(d_)} "
      f"chains | Wilcoxon p={wilcoxon(d_).pvalue:.2e}")

json.dump({"n_chains":len(rows),"n_res":int(len(y)),
           "p2rank_pr":float(average_precision_score(y,p2)),
           "p2rank_roc":float(roc_auc_score(y,p2)),
           "egnn_pr":float(average_precision_score(y,mp)),
           "egnn_roc":float(roc_auc_score(y,mp)),
           "paired_mean_diff":float(d_.mean()),
           "egnn_win_rate":float((d_>0).mean())},
          open("/kaggle/working/phase7_baseline.json","w"),indent=2)
print("saved -> phase7_baseline.json")
''')

md(r'''
---

## Reading this

P2Rank is a genuinely strong, pre-trained baseline, not a straw man. Beating it
on a sequence-clustered split would be a real claim; losing to it is equally
informative and should be reported as-is.

**Two caveats that must accompany any number here.**

1. **Label mismatch favours us.** Our labels mark residues near the ligand in
   *that crystal*; P2Rank predicts ligandable sites in general. A correctly
   identified but unoccupied pocket is scored as a false positive against it.
2. **P2Rank never trained on our split.** Its training set may overlap our test
   chains, which cuts the other way. Neither effect is quantified here, so treat
   the comparison as indicative rather than a controlled benchmark.

The per-chain paired test is the most robust line: it asks how often the model
wins on an individual protein, which is insensitive to a few large chains
dominating the pooled metric.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase7_baseline_p2rank.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
