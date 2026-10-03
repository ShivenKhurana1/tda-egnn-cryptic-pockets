#!/usr/bin/env python3
"""Phase 9: M-CSA catalytic-residue task."""
import json
from pathlib import Path

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 9 - Catalytic Residues (M-CSA): The Harder Task

Every result so far predicts **binding pockets** - residues within 4.5 Å of a
ligand. A pocket has ~20 contact residues; a catalytic machinery has ~4. Those
are different biological questions, and only the second is "active site
prediction" in the strict sense.

The **Catalytic Site Atlas (M-CSA)** provides literature-curated catalytic
residues: 951 PDB entries, 4,408 annotated residues, **4.6 per entry**.

### Why this is a separate dataset, not a second head

The obvious design - add a catalytic output head to the existing model - does not
work here. M-CSA covers 951 entries; the Phase 6 set is 2,056 cluster
representatives sampled from 126,921. Expected overlap is **~15 entries**, far too
few to train a head on. So this builds a dedicated M-CSA dataset end to end.

### What makes it hard

At ~4.6 positives in a ~300-residue chain, the positive rate is **around 1.5 %**
versus 6.82 % for pockets. PR-AUC baselines drop accordingly, and the interesting
question is whether topology's contribution *survives* - or grows - when the
target shrinks to the catalytic core.

> **Header caution:** in `curated_data.csv` the column headers are offset from the
> data. The `PDB code` column holds the *residue name* (`Asp`), and the actual
> entry is in `PDB`. Reading it naively yields 102 bogus "entries" instead of 951.
''')

md(r'''## Cell 1 - Fetch and parse M-CSA annotations''')

code(r'''
!pip install -q torch-geometric==2.6.1 gudhi==3.10.1 biopython==1.84
''')

code(r'''
import csv, gzip, io, os, shutil, subprocess, time, urllib.request, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")

URL="https://www.ebi.ac.uk/thornton-srv/m-csa/media/flat_files/curated_data.csv"
raw=urllib.request.urlopen(URL,timeout=120).read().decode("utf-8",errors="replace")
rows=list(csv.DictReader(io.StringIO(raw)))
print(f"M-CSA rows: {len(rows):,}")

# NB: headers are offset -- 'PDB' is the entry; 'PDB code' holds the residue NAME.
cat=defaultdict(set)          # (PDB, chain) -> {resnum}
for r in rows:
    if r.get("residue/reactant/product/cofactor") != "residue": continue
    pid=(r.get("PDB") or "").strip().upper()
    ch=(r.get("chain/kegg compound") or "").strip()
    num=(r.get("resid/chebi id") or "").strip()
    if not pid or not ch or not num.lstrip("-").isdigit(): continue
    cat[(pid,ch)].add(int(num))

entries=sorted({p for p,_ in cat})
print(f"annotated (pdb,chain) pairs : {len(cat):,}")
print(f"unique PDB entries          : {len(entries):,}")
print(f"catalytic residues total    : {sum(len(v) for v in cat.values()):,}")
print(f"mean per chain              : {np.mean([len(v) for v in cat.values()]):.1f}")
''')

md(r'''## Cell 2 - Download structures and build labelled chains''')

code(r'''
from Bio.PDB import PDBParser, MMCIFParser
AA3_TO_1={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
RAW=Path("/tmp/mcsa"); RAW.mkdir(parents=True,exist_ok=True)
MIN_RES,MAX_RES=50,1000

def fetch(pid):
    for ext in ("pdb","cif"):
        p=RAW/f"{pid}.{ext}.gz"
        if p.exists() and p.stat().st_size>500: return pid,p
    for ext in ("pdb","cif"):
        try:
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.{ext}.gz",timeout=60) as r:
                b=r.read()
            if len(b)>500:
                p=RAW/f"{pid}.{ext}.gz"; p.write_bytes(b); return pid,p
        except Exception: continue
    return pid,None

t0=time.time()
with ThreadPoolExecutor(max_workers=16) as ex: got=list(ex.map(fetch,entries))
paths={p:f for p,f in got if f}
print(f"fetched {len(paths):,}/{len(entries):,} in {(time.time()-t0)/60:.1f} min")

def build_chain(pid,path,wanted):
    try:
        parser=MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True,PERMISSIVE=True)
        with gzip.open(path,"rt") as fh:
            model=next(parser.get_structure(pid,fh).get_models())
    except Exception: return []
    out=[]
    for cid,catnums in wanted:
        try: ch=model[cid]
        except KeyError: continue
        residues=[r for r in ch if r.get_resname().strip().upper() in AA3_TO_1 and "CA" in r]
        if not (MIN_RES<=len(residues)<=MAX_RES): continue
        ca,seq,per_res,y=[],[],[],[]
        for r in residues:
            ca.append(r["CA"].coord)
            seq.append(AA3_TO_1.get(r.get_resname().strip().upper(),"X"))
            xyz=[a.coord for a in r.get_unpacked_list() if a.element not in ("H","D")]
            per_res.append(np.asarray(xyz,dtype=np.float32).reshape(-1,3))
            y.append(1.0 if r.id[1] in catnums else 0.0)
        y=np.asarray(y,dtype=np.float32)
        if y.sum()<2: continue          # need >=2 annotated residues actually matched
        out.append({"pdb_id":pid,"chain_id":cid,"seq":"".join(seq),
                    "ca":np.asarray(ca,dtype=np.float32),"y":y,"atoms":per_res})
    return out

from joblib import Parallel, delayed
want=defaultdict(list)
for (pid,ch),nums in cat.items():
    if pid in paths: want[pid].append((ch,nums))

t0=time.time()
batches=Parallel(n_jobs=max(1,os.cpu_count() or 2),verbose=1)(
    delayed(build_chain)(pid,paths[pid],w) for pid,w in sorted(want.items()))
recs=[r for b in batches for r in b]
nres=sum(len(r["y"]) for r in recs); npos=sum(int(r["y"].sum()) for r in recs)
print(f"\nchains {len(recs):,} | residues {nres:,} | catalytic {npos:,} "
      f"({100*npos/max(nres,1):.2f}%) | {(time.time()-t0)/60:.1f} min")
print(f"mean catalytic residues per chain: {npos/max(len(recs),1):.1f}")
''')

md(r'''## Cell 3 - Cluster, pick representatives''')

code(r'''
seqs={f"{r['pdb_id']}_{r['chain_id']}":r["seq"] for r in recs}
by_id={f"{r['pdb_id']}_{r['chain_id']}":r for r in recs}
mm=None
try:
    subprocess.run("curl -sL https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -o /tmp/mm.tar.gz "
                   "&& tar xzf /tmp/mm.tar.gz -C /tmp",shell=True,check=True,timeout=600)
    c=Path("/tmp/mmseqs/bin/mmseqs")
    if c.exists(): os.chmod(c,0o755); mm=str(c)
except Exception as e: print("mmseqs unavailable:",type(e).__name__)

W=Path("/tmp/mw"); shutil.rmtree(W,ignore_errors=True); W.mkdir()
(W/"in.fasta").write_text("".join(f">{k}\n{v}\n" for k,v in seqs.items()))
subprocess.run([mm,"easy-cluster",str(W/"in.fasta"),str(W/"res"),str(W/"tmp"),
                "--min-seq-id","0.30","-c","0.8","--cov-mode","1","-v","1"],
               check=True,capture_output=True,timeout=1800)
rep=defaultdict(list)
for line in (W/"res_cluster.tsv").read_text().splitlines():
    a,b=line.split("\t"); rep[a].append(b)
clusters={f"C{i:05d}":v for i,v in enumerate(sorted(rep.values(),key=lambda v:(-len(v),v[0])))}
reps={k:max(v,key=lambda c:(len(seqs[c]),c)) for k,v in clusters.items()}
print(f"{len(seqs):,} chains -> {len(clusters):,} clusters -> {len(reps):,} representatives")
''')

md(r'''## Cell 4 - Topology (identical featuriser to Phase 2b/6b)''')

code(r'''
import gudhi
MAX_HOM_DIM=2; BETTI_RADII=(2.0,3.0,5.0); NBHD_RADIUS=12.0; MAX_NBHD_ATOMS=300
FPD=6+2+len(BETTI_RADII); TOPO_DIM=FPD*(MAX_HOM_DIM+1)+2
TOPO_NAMES=[f"H{d}_{s}" for d in range(MAX_HOM_DIM+1)
            for s in ["n_bars","total_pers","max_pers","mean_pers","std_pers","entropy",
                      "mean_birth","mean_death",*[f"betti_r{r:g}" for r in BETTI_RADII]]]+\
           ["nbhd_n_atoms","nbhd_gyration"]
BURIAL_NAMES=["nbhd_n_atoms","nbhd_gyration"]

def _dg(pts):
    pts=np.ascontiguousarray(pts,dtype=np.float64); st=None; sq=True
    if len(pts)>=5:
        try: st=gudhi.AlphaComplex(points=pts).create_simplex_tree()
        except Exception: st=None
    if st is None:
        st=gudhi.RipsComplex(points=pts,max_edge_length=float(NBHD_RADIUS)).create_simplex_tree(max_dimension=MAX_HOM_DIM+1); sq=False
    st.compute_persistence(homology_coeff_field=2,persistence_dim_max=True)
    o={}
    for d in range(MAX_HOM_DIM+1):
        b=np.asarray(st.persistence_intervals_in_dimension(d),dtype=np.float64).reshape(-1,2)
        o[d]=np.sqrt(np.maximum(b,0.0)) if (sq and len(b)) else b
    return o

def _vec(dg,pts):
    f=[]
    for d in range(MAX_HOM_DIM+1):
        bars=dg.get(d,np.zeros((0,2)))
        fin=bars[np.isfinite(bars).all(axis=1)] if len(bars) else bars.reshape(-1,2)
        if len(fin)==0: f.extend([0.0]*FPD); continue
        b,dt=fin[:,0],fin[:,1]; pers=np.maximum(dt-b,0.0); tot=float(pers.sum())
        ent=0.0
        if tot>0:
            p=pers/tot; nz=p[p>0]; ent=float(-(nz*np.log(nz)).sum())
        f.extend([float(len(fin)),tot,float(pers.max()),float(pers.mean()),float(pers.std()),
                  ent,float(b.mean()),float(dt.mean()),
                  *[float(((b<=r)&(dt>r)).sum()) for r in BETTI_RADII]])
    c=pts-pts.mean(0,keepdims=True)
    f.extend([float(len(pts)),float(np.sqrt((c**2).sum(1).mean())) if len(pts) else 0.0])
    return np.asarray(f,dtype=np.float32)

def topo_chain(ca,per_res):
    atoms=np.concatenate([a for a in per_res if len(a)],axis=0)
    tree=cKDTree(atoms); out=np.zeros((len(ca),TOPO_DIM),np.float32)
    for i,c in enumerate(ca):
        idx=tree.query_ball_point(c,NBHD_RADIUS)
        if len(idx)>MAX_NBHD_ATOMS: _,idx=tree.query(c,k=MAX_NBHD_ATOMS)
        pts=atoms[np.asarray(idx,dtype=int)]
        if len(pts)>=4: out[i]=_vec(_dg(pts),pts)
    return out

sel=[by_id[c] for c in reps.values() if c in by_id]
n_res=sum(len(r["y"]) for r in sel)
print(f"topology on {len(sel):,} chains / {n_res:,} residues "
      f"(ETA {n_res*0.0064/60:.0f} min)")
t0=time.time()
topo_all=Parallel(n_jobs=max(1,os.cpu_count() or 2),verbose=5)(
    delayed(topo_chain)(r["ca"],r["atoms"]) for r in sel)
print(f"done in {(time.time()-t0)/60:.1f} min")
''')

md(r'''## Cell 5 - Graphs and 5-fold ablation''')

code(r'''
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

AA="ACDEFGHIKLMNPQRSTVWYX"; A2I={a:i for i,a in enumerate(AA)}
def rgc(pos,r=8.0,k=32):
    n=pos.size(0); pos=pos-pos.mean(0,keepdim=True)
    d=torch.cdist(pos,pos).clone(); d.fill_diagonal_(float("inf")); m=d<=r
    if k<n:
        _,i=torch.topk(d,k,dim=1,largest=False); keep=torch.zeros_like(m); keep.scatter_(1,i,True); m&=keep
    m=m|m.t(); t,s=m.nonzero(as_tuple=True); return torch.stack([s,t]).long()

topo_cat=np.concatenate(topo_all,0)
keep=topo_cat.std(0)>1e-9
names=[TOPO_NAMES[i] for i in range(TOPO_DIM) if keep[i]]
mu,sd=topo_cat[:,keep].mean(0),topo_cat[:,keep].std(0); sd[sd<1e-6]=1.0
BURIAL_IDX=[names.index(n) for n in BURIAL_NAMES]
N_SEQ=27; TD=len(names)

ds=[]
for r,t in zip(sel,topo_all):
    pos=torch.from_numpy(r["ca"]); pos=pos-pos.mean(0,keepdim=True); n=pos.size(0)
    oh=torch.zeros(n,21); oh[torch.arange(n),torch.tensor([A2I.get(c,20) for c in r["seq"]])]=1.0
    x=torch.cat([oh,torch.zeros(n,6),torch.from_numpy(((t[:,keep]-mu)/sd).astype(np.float32))],1)
    ei=rgc(pos); d_=(pos[ei[0]]-pos[ei[1]]).norm(dim=1,keepdim=True)
    sep=(ei[0]-ei[1]).abs().float().unsqueeze(1)
    g=Data(x=x,pos=pos,edge_index=ei,edge_attr=torch.cat([d_,torch.log1p(sep)],1),
           y=torch.from_numpy(r["y"])); g.num_nodes=n
    ds.append(g)
POS_WEIGHT=float((sum(int(d.num_nodes) for d in ds)-sum(int(d.y.sum()) for d in ds))/max(sum(int(d.y.sum()) for d in ds),1))
print(f"{len(ds):,} graphs | node dim {ds[0].x.size(1)} | pos_weight {POS_WEIGHT:.1f} "
      f"| positive rate {100*sum(int(d.y.sum()) for d in ds)/sum(int(d.num_nodes) for d in ds):.2f}%")
''')

code(r'''
GPU=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU=True
    except RuntimeError: pass
DEVICE=torch.device("cuda" if GPU else "cpu")

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

def ep(model,loader,opt=None):
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

ORIG=[d.x.clone() for d in ds]
def arm(a):
    for d,x0 in zip(ds,ORIG):
        t=x0[:,N_SEQ:].clone()
        if a=="seq-only": t.zero_()
        elif a=="burial-only":
            k=torch.zeros_like(t); k[:,BURIAL_IDX]=t[:,BURIAL_IDX]; t=k
        d.x=torch.cat([x0[:,:N_SEQ],t],1)

ARMS=["seq-only","burial-only","full"]; NF=5
folds=list(KFold(n_splits=NF,shuffle=True,random_state=1337).split(np.arange(len(ds))))
recs2=[]; t0=time.time()
for a in ARMS:
    arm(a)
    for fi,(tr_i,te_i) in enumerate(folds):
        rng=np.random.RandomState(7); perm=rng.permutation(tr_i); nv=max(1,int(0.15*len(perm)))
        va=[ds[i] for i in perm[:nv]]; trn=[ds[i] for i in perm[nv:]]; te=[ds[i] for i in te_i]
        torch.manual_seed(0)
        m=Net(N_SEQ,TD).to(DEVICE); o=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=40)
        best,bad,st=-1,0,None
        for e in range(40):
            ep(m,DataLoader(trn,batch_size=16,shuffle=True),o); v=ep(m,DataLoader(va,batch_size=16)); sch.step()
            if v["pr"]>best: best,bad,st=v["pr"],0,{k:t.clone() for k,t in m.state_dict().items()}
            else:
                bad+=1
                if bad>=8: break
        m.load_state_dict(st)
        r=ep(m,DataLoader(te,batch_size=16))
        recs2.append({"arm":a,"fold":fi,"val_pr":best,"test_pr":r["pr"],"test_roc":r["roc"]})
        print(f"  {a:<12} fold {fi} val PR {best:.4f} test PR {r['pr']:.4f} ROC {r['roc']:.4f} "
              f"[{(time.time()-t0)/60:.0f}m]")
for d,x0 in zip(ds,ORIG): d.x=x0
''')

md(r'''## Cell 6 - Results''')

code(r'''
import json
from scipy.stats import ttest_rel, wilcoxon
gm=lambda a,k: np.array([r[k] for r in recs2 if r["arm"]==a])
base=sum(int(d.y.sum()) for d in ds)/sum(int(d.num_nodes) for d in ds)
print(f"positive (catalytic) base rate: {base:.4f}\n")
print(f"{'arm':<14}{'test PR-AUC':>18}{'test ROC-AUC':>18}{'lift over base':>17}")
for a in ARMS:
    pr,rc=gm(a,"test_pr"),gm(a,"test_roc")
    print(f"{a:<14}{pr.mean():>10.4f} +-{pr.std():.4f}{rc.mean():>11.4f} +-{rc.std():.4f}"
          f"{pr.mean()/base:>16.1f}x")
print()
for x,y_ in (("full","burial-only"),("full","seq-only")):
    d=gm(x,"test_pr")-gm(y_,"test_pr")
    t=ttest_rel(gm(x,"test_pr"),gm(y_,"test_pr")).pvalue
    try: w=wilcoxon(d).pvalue
    except ValueError: w=float("nan")
    print(f"  {x} - {y_:<14} PR {d.mean():+.4f} +-{d.std():.4f} | {(d>0).sum()}/{NF} | "
          f"t p={t:.4f} | W p={w:.4f}")
json.dump(recs2,open("/kaggle/working/phase9_mcsa.json","w"),indent=2,default=float)
print("\nsaved -> phase9_mcsa.json")
''')

md(r'''
---

## Interpreting

This is a **much harder target** - roughly 1.5 % positives against 6.82 % for
pockets, on a far smaller dataset. Absolute PR-AUC will be low; the meaningful
number is **lift over the base rate**, not the raw value.

Two readings:

- **Topology's advantage persists or grows** → persistent homology is capturing
  something about catalytic geometry specifically, not merely pocket burial. That
  is the stronger scientific claim available to this project.
- **The advantage disappears** → topology is a pocket detector, and catalytic
  residues need chemistry (conservation, electrostatics, protonation) that
  geometry alone does not supply. Also a clean, reportable finding.

With only 5 folds this is again power-limited - treat it as a directional result
and say so, rather than repeating the Phase 5b mistake of over-reading an
underpowered comparison.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase9_mcsa_catalytic.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
