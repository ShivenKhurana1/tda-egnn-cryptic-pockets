
# ============================================================
# ORIGINAL CELL 2
# ============================================================
!pip install -q torch-geometric==2.6.1 scipy scikit-learn


# ============================================================
# ORIGINAL CELL 3
# ============================================================

import copy, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, f1_score, precision_score, recall_score
from sklearn.cluster import KMeans

def locate(f):
    hits=[]
    for root in (Path('/kaggle/input'), Path('/kaggle/working'), Path('.')):
        if root.exists(): hits.extend(root.rglob(f))
    if not hits: raise FileNotFoundError(f'Could not find {f}')
    return max(hits, key=lambda p:p.stat().st_size)

blob_path=locate('phase159_cryptobench_anmmsf_graphs.pt')
print('USING:',blob_path)
print('SIZE GB:',blob_path.stat().st_size/1024**3)
blob=torch.load(blob_path,map_location='cpu',weights_only=False)
dataset=blob['dataset']
N_PRE=int(blob['n_pre_topo']); TOPO_DIM=int(blob['topo_dim']); MSF_COL_IDX=int(blob['msf_col_idx']); POS_WEIGHT=float(blob['pos_weight'])
assert N_PRE==508 and TOPO_DIM==34 and MSF_COL_IDX==507
print('node dim:',dataset[0].x.shape[1])
print('official splits:',{s:sum(d.split==s for d in dataset) for s in ('train','val','test')})

# Label-free graph-level representation: mean pooled 480-d ESM embedding.
emb=np.stack([d.x[:,27:507].float().mean(0).numpy() for d in dataset])
emb=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-12)

# Robust embedding-cluster holdout. We cluster proteins in embedding space and NEVER
# split a cluster across train/validation/test. KMeans is used instead of a distance
# threshold so one giant cluster cannot absorb most of the dataset. This is a
# label-free similarity-based holdout, not a verified sequence-family split.
N_CLUSTERS=30
km=KMeans(n_clusters=N_CLUSTERS, random_state=166, n_init=20, max_iter=300)
labels=km.fit_predict(emb)
sizes=np.bincount(labels, minlength=N_CLUSTERS)
print('embedding clusters:',N_CLUSTERS,'largest cluster:',int(sizes.max()),'smallest cluster:',int(sizes.min()))

# Deterministic whole-cluster assignment targeting 70/10/20% by graph count.
# Try many deterministic orderings and retain the closest feasible assignment.
targets=np.array([0.70,0.10,0.20])*len(dataset)
rng=np.random.default_rng(166)
best=None
for trial in range(2000):
    order=np.arange(N_CLUSTERS)
    rng.shuffle(order)
    order=sorted(order,key=lambda k:sizes[k],reverse=True)
    counts=np.zeros(3,dtype=int)
    assign={}
    for k in order:
        ratios=(counts+sizes[k])/targets
        # Avoid starving validation/test while keeping the final sizes close to target.
        j=int(np.argmin(ratios))
        assign[k]=j
        counts[j]+=sizes[k]
    score=float(np.sum(((counts-targets)/targets)**2))
    if best is None or score<best[0]:
        best=(score,counts.copy(),assign.copy())
_,counts,assign=best

hard_split={0:'train_hard',1:'val_hard',2:'test_hard'}
for i,d in enumerate(dataset):
    d.hard_split=hard_split[assign[int(labels[i])]]
print('hard split sizes:',{s:sum(d.hard_split==s for d in dataset) for s in hard_split.values()})
print('hard split positive residues:',{s:int(sum(int(d.y.sum()) for d in dataset if d.hard_split==s)) for s in hard_split.values()})
print('hard split cluster counts:',{s:sum(assign[k]==j for k in range(N_CLUSTERS) for j in [hard_split.keys().__iter__()]) if False else None for s in []})

# Save the exact label-free split recipe for reproducibility.
split_manifest={
    'method':'KMeans on L2-normalized graph-level mean ESM embeddings',
    'embedding_columns':'27:507',
    'n_clusters':N_CLUSTERS,
    'random_state':166,
    'n_init':20,
    'target_fractions':{'train_hard':0.70,'val_hard':0.10,'test_hard':0.20},
    'cluster_sizes':sizes.tolist(),
    'cluster_assignment':{str(k):hard_split[assign[k]] for k in range(N_CLUSTERS)},
    'split_sizes':{s:sum(d.hard_split==s for d in dataset) for s in hard_split.values()}
}
json.dump(split_manifest,open('/kaggle/working/phase166_hard_split_manifest.json','w'),indent=2)


# ============================================================
# ORIGINAL CELL 4
# ============================================================
class EGCL(nn.Module):
    def __init__(self,h,ea=2,act=nn.SiLU,uc=True):
        super().__init__(); self.uc=uc
        self.e=nn.Sequential(nn.Linear(2*h+1+ea,h),act(),nn.Linear(h,h),act())
        self.n=nn.Sequential(nn.Linear(2*h,h),act(),nn.Linear(h,h))
        self.a=nn.Sequential(nn.Linear(h,1),nn.Sigmoid())
        if uc:
            self.c=nn.Sequential(nn.Linear(h,h),act(),nn.Linear(h,1,bias=False))
            nn.init.xavier_uniform_(self.c[-1].weight,gain=0.001)
        self.norm=nn.LayerNorm(h)
    def forward(self,h,x,ei,ea):
        src,dst=ei; rel=x[src]-x[dst]; d2=(rel**2).sum(-1,keepdim=True)
        m=self.e(torch.cat([h[src],h[dst],d2,ea],-1)); m=m*self.a(m)
        agg=torch.zeros_like(h).index_add_(0,dst,m)
        deg=torch.zeros(h.size(0),1,device=h.device).index_add_(0,dst,torch.ones_like(d2)).clamp(min=1)
        h=self.norm(h+self.n(torch.cat([h,agg/deg],-1)))
        if self.uc:
            w=self.c(m)*rel/(d2.sqrt()+1.0)
            x=x+torch.zeros_like(x).index_add_(0,dst,w)/deg
        return h,x

class Fus(nn.Module):
    def __init__(self,h,t):
        super().__init__(); self.p=nn.Linear(t,h); self.g=nn.Sequential(nn.Linear(h+t,h),nn.Sigmoid())
    def forward(self,h,t): return h+self.g(torch.cat([h,t],-1))*self.p(t)

class LocalizingEGNN(nn.Module):
    def __init__(self,n_pre,td,h=128,L=4,dr=0.1):
        super().__init__(); self.n_pre=n_pre
        self.embed=nn.Sequential(nn.Linear(n_pre+td,h),nn.SiLU(),nn.Linear(h,h))
        self.ls=nn.ModuleList([EGCL(h,uc=(i<L-1)) for i in range(L)])
        self.fs=nn.ModuleList([Fus(h,td) for _ in range(L)])
        self.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))
        self.loc_head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Linear(h//2,1))
    def forward(self,d):
        h=self.embed(d.x); pos=d.pos; t=d.x[:,self.n_pre:]
        for l,f in zip(self.ls,self.fs): h,pos=l(h,pos,d.edge_index,d.edge_attr); h=f(h,t)
        logits=self.head(h).squeeze(-1)
        batch=d.batch if getattr(d,'batch',None) is not None else torch.zeros(h.size(0),dtype=torch.long,device=h.device)
        loc_logit=self.loc_head(h).squeeze(-1); attn=pyg_softmax(loc_logit,batch)
        pred_center=torch.zeros(int(batch.max().item())+1,3,device=h.device).index_add_(0,batch,attn.unsqueeze(-1)*pos)
        return logits,pred_center


# ============================================================
# ORIGINAL CELL 5
# ============================================================

# One configuration per Kaggle kernel keeps CPU runtime bounded.
TOPO_NAMES = list(blob['topo_names'])
assert len(TOPO_NAMES) == 34
ARMS = {'full_model': [507] + list(range(508,542))}

def masked_dataset(ds,arm):
    out=[]
    for d in ds:
        d2=copy.copy(d); d2.x=d.x.clone(); keep=torch.zeros(d2.x.size(1),dtype=torch.bool)
        keep[0:27]=True       # structure
        keep[27:507]=True     # ESM
        keep[ARMS[arm]]=True  # TDA and/or ANM
        d2.x[:,~keep]=0.0
        out.append(d2)
    return out

hard_splits={s:[d for d in dataset if d.hard_split==s] for s in ('train_hard','val_hard','test_hard')}
masked={arm:{s:masked_dataset(hard_splits[s],arm) for s in hard_splits} for arm in ARMS}
for arm in ARMS:
    print(arm, 'extra feature dims=',len(ARMS[arm]), {s:len(masked[arm][s]) for s in hard_splits})


# ============================================================
# ORIGINAL CELL 7
# ============================================================
for arm, idx in ARMS.items():
    print(f"{arm:20s} {len(idx):2d} extra features")
    print(idx)


# ============================================================
# ORIGINAL CELL 8
# ============================================================
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    if major >= 7:
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cpu')

print("DEVICE:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability())
WORK=Path('/kaggle/working')
(WORK/'cache').mkdir(parents=True,exist_ok=True)
print('DEVICE:',DEVICE)

def focal_loss_with_logits(logits,targets,alpha=0.9,gamma=2.0):
    bce=nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    p=torch.sigmoid(logits); pt=p*targets+(1-p)*(1-targets); at=alpha*targets+(1-alpha)*(1-targets)
    return (at*(1-pt).pow(gamma)*bce).mean()

def run_epoch(model,loader,opt=None,lam=0.1):
    train=opt is not None; model.train(train); huber=nn.SmoothL1Loss(beta=4.0); ps=[]; ys=[]; errs=[]
    for b in loader:
        b=b.to(DEVICE)
        with torch.set_grad_enabled(train):
            logits,pc=model(b); loss=focal_loss_with_logits(logits,b.y)+lam*huber(pc,b.true_center.view(-1,3))
        if train:
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        ps.append(torch.sigmoid(logits).detach().cpu().numpy()); ys.append(b.y.cpu().numpy()); errs.append((pc-b.true_center.view(-1,3)).norm(dim=1).detach().cpu().numpy())
    p=np.concatenate(ps); y=np.concatenate(ys); e=np.concatenate(errs); pred=p>=0.5
    return {'pr_auc':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),'mcc':float(matthews_corrcoef(y,pred)),'f1':float(f1_score(y,pred,zero_division=0)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'dcc4':float((e<4).mean()),'mean_center_error':float(e.mean())}

def train_arm(arm,seed,epochs=40,patience=8,batch=16,lam=0.1):
    torch.manual_seed(seed); np.random.seed(seed)
    model=LocalizingEGNN(N_PRE,TOPO_DIM).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    tl=DataLoader(masked[arm]['train_hard'],batch_size=batch,shuffle=True); vl=DataLoader(masked[arm]['val_hard'],batch_size=batch)
    best=-1; state=None; bad=0
    for ep in range(epochs):
        run_epoch(model,tl,opt,lam); v=run_epoch(model,vl,lam=lam); sch.step()
        if v['pr_auc']>best: best=v['pr_auc']; state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(state); model.eval(); val=run_epoch(model,vl,lam=lam); te=run_epoch(model,DataLoader(masked[arm]['test_hard'],batch_size=batch),lam=lam)
    path=WORK/f'cache/phase166_{arm}_seed{seed}.pt'; torch.save(model.state_dict(),path)
    return val,te,str(path)


# ============================================================
# ORIGINAL CELL 10
# ============================================================

SEEDS=list(range(8)); results=[]; t0=time.time()
for arm in ARMS:
    for seed in SEEDS:
        val,te,path=train_arm(arm,seed)
        row={'arm':arm,'seed':seed,'val':val,'test':te,'checkpoint':path}
        results.append(row)
        print(f"{arm:18s} seed={seed} val_PR={val['pr_auc']:.4f} val_ROC={val['roc_auc']:.4f} test_PR={te['pr_auc']:.4f} test_ROC={te['roc_auc']:.4f} MCC={te['mcc']:.4f} DCC4={te['dcc4']:.3f} elapsed={(time.time()-t0)/60:.1f}m")
json.dump(results,open(WORK/f'phase166_full_model_results.json','w'),indent=2)


# ============================================================
# ORIGINAL CELL 11
# ============================================================

import pandas as pd
rows=[]
for r in results:
    rows.append({'arm':r['arm'],'seed':r['seed'],**{f'val_{k}':v for k,v in r['val'].items()},**{f'test_{k}':v for k,v in r['test'].items()}})
df=pd.DataFrame(rows)
df.to_csv(WORK/f'phase166_full_model_seed_results.csv',index=False)
summary=df.groupby('arm').agg({'seed':['count'],'val_pr_auc':['mean','std'],'test_pr_auc':['mean','std'],'test_roc_auc':['mean','std'],'test_mcc':['mean','std'],'test_f1':['mean','std'],'test_precision':['mean','std'],'test_recall':['mean','std'],'test_dcc4':['mean','std'],'test_mean_center_error':['mean','std']}).reset_index()
summary.columns=['arm','n_seeds','val_pr_auc_mean','val_pr_auc_std','test_pr_auc_mean','test_pr_auc_std','test_roc_auc_mean','test_roc_auc_std','test_mcc_mean','test_mcc_std','test_f1_mean','test_f1_std','test_precision_mean','test_precision_std','test_recall_mean','test_recall_std','test_dcc4_mean','test_dcc4_std','test_center_error_mean','test_center_error_std']
summary.to_csv(WORK/f'phase166_full_model_summary.csv',index=False)
print(summary.to_string(index=False))


# ============================================================
# ORIGINAL CELL 13
# ============================================================
# Cross-arm statistics deferred until all configurations share completed outputs.

