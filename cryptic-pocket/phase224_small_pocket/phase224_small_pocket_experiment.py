# ============================================================
# PHASE 224 — SMALL-POCKET LOSS EXPERIMENT
#
# Isolated from Phase 166.
# Does not modify the frozen manuscript or submission package.
# ============================================================

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as pyg_softmax

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.cluster import KMeans


# ============================================================
# PHASE 166 DATA LOADING + HARD SPLIT
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
# PHASE 166 ARCHITECTURE
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
# PHASE 224 FEATURE MASKING
# ============================================================

TOPO_NAMES = list(blob['topo_names'])
assert len(TOPO_NAMES) == 34

ARMS = {
    'baseline': [507] + list(range(508, 542)),
    'small_w1': [507] + list(range(508, 542)),
    'small_w2': [507] + list(range(508, 542)),
}


def masked_dataset(ds, arm):
    out = []

    for d in ds:
        d2 = copy.copy(d)
        d2.x = d.x.clone()

        keep = torch.zeros(
            d2.x.size(1),
            dtype=torch.bool
        )

        keep[0:27] = True
        keep[27:507] = True
        keep[ARMS[arm]] = True

        d2.x[:, ~keep] = 0.0

        out.append(d2)

    return out


hard_splits = {
    s: [d for d in dataset if d.hard_split == s]
    for s in ('train_hard', 'val_hard', 'test_hard')
}

masked = {
    arm: {
        s: masked_dataset(hard_splits[s], arm)
        for s in hard_splits
    }
    for arm in ARMS
}

print(
    "224 arms:",
    {
        arm: {
            s: len(masked[arm][s])
            for s in hard_splits
        }
        for arm in ARMS
    }
)


# ============================================================
# DEVICE
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




# ============================================================
# PHASE 224 LOSS
# ============================================================

def focal_loss_with_logits_weighted(
    logits,
    targets,
    residue_weights=None,
    alpha=0.9,
    gamma=2.0,
):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction='none',
    )

    p = torch.sigmoid(logits)

    pt = (
        p * targets
        + (1 - p) * (1 - targets)
    )

    at = (
        alpha * targets
        + (1 - alpha) * (1 - targets)
    )

    focal = (
        at
        * (1 - pt).pow(gamma)
        * bce
    )

    if residue_weights is not None:
        focal = focal * residue_weights

    return focal.mean()


# ============================================================
# GRAPH INDEXING
# ============================================================

def attach_graph_indices(ds):
    out = []

    for graph_id, d in enumerate(ds):
        d2 = copy.copy(d)

        d2.graph_index = torch.full(
            (d2.x.size(0),),
            graph_id,
            dtype=torch.long,
        )

        out.append(d2)

    return out


# ============================================================
# TRAINING-ONLY POCKET SIZE WEIGHTS
# ============================================================

def compute_train_size_weights(train_ds):
    pocket_sizes = np.array(
        [
            max(1, int(d.y.sum().item()))
            for d in train_ds
        ],
        dtype=float,
    )

    median_size = float(
        np.median(pocket_sizes)
    )

    weights = {}

    settings = {
        'small_w1': (0.5, 3.0),
        'small_w2': (1.0, 6.0),
    }

    for arm, (beta, max_weight) in settings.items():

        w = (
            median_size / pocket_sizes
        ) ** beta

        w = np.clip(
            w,
            1.0,
            max_weight,
        )

        weights[arm] = torch.tensor(
            w,
            dtype=torch.float32,
            device=DEVICE,
        )

    return median_size, weights


# ============================================================
# EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    opt=None,
    lam=0.1,
    arm='baseline',
    size_weights=None,
):
    train = opt is not None

    model.train(train)

    huber = nn.SmoothL1Loss(beta=4.0)

    ps = []
    ys = []
    errs = []

    for b in loader:

        b = b.to(DEVICE)

        with torch.set_grad_enabled(train):

            logits, pc = model(b)

            residue_weights = None

            if arm != 'baseline':

                if size_weights is None:
                    raise RuntimeError(
                        "Missing training-derived size weights."
                    )

                residue_weights = (
                    size_weights[b.graph_index]
                )

                # Only positive residues get the
                # additional small-pocket weight.
                residue_weights = torch.where(
                    b.y > 0.5,
                    residue_weights,
                    torch.ones_like(
                        residue_weights
                    ),
                )

            loss = (
                focal_loss_with_logits_weighted(
                    logits,
                    b.y,
                    residue_weights,
                )
                + lam * huber(
                    pc,
                    b.true_center.view(-1, 3),
                )
            )

        if train:

            opt.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

        ps.append(
            torch.sigmoid(logits)
            .detach()
            .cpu()
            .numpy()
        )

        ys.append(
            b.y.cpu().numpy()
        )

        errs.append(
            (
                pc
                - b.true_center.view(-1, 3)
            )
            .norm(dim=1)
            .detach()
            .cpu()
            .numpy()
        )

    p = np.concatenate(ps)
    y = np.concatenate(ys)
    e = np.concatenate(errs)

    pred = p >= 0.5

    return {
        'pr_auc': float(
            average_precision_score(y, p)
        ),
        'roc_auc': float(
            roc_auc_score(y, p)
        ),
        'mcc': float(
            matthews_corrcoef(y, pred)
        ),
        'f1': float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        'precision': float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        'recall': float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        'dcc4': float(
            (e < 4).mean()
        ),
        'mean_center_error': float(
            e.mean()
        ),
    }


# ============================================================
# TRAIN ONE ARM / ONE SEED
# ============================================================

def train_arm(
    arm,
    seed,
    epochs=40,
    patience=8,
    batch=16,
    lam=0.1,
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LocalizingEGNN(
        N_PRE,
        TOPO_DIM,
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
    )

    train_ds = attach_graph_indices(
        masked[arm]['train_hard']
    )

    val_ds = attach_graph_indices(
        masked[arm]['val_hard']
    )

    test_ds = attach_graph_indices(
        masked[arm]['test_hard']
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch,
    )

    median_size, size_weights = (
        compute_train_size_weights(
            train_ds
        )
    )

    best = -1.0
    state = None
    bad = 0

    for epoch in range(epochs):

        run_epoch(
            model,
            train_loader,
            opt,
            lam=lam,
            arm=arm,
            size_weights=size_weights.get(arm),
        )

        val = run_epoch(
            model,
            val_loader,
            lam=lam,
            arm=arm,
            size_weights=None,
        )

        sch.step()

        if val['pr_auc'] > best:

            best = val['pr_auc']

            state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            bad = 0

        else:

            bad += 1

            if bad >= patience:
                break

    if state is None:
        raise RuntimeError(
            "No validation checkpoint was selected."
        )

    model.load_state_dict(state)
    model.eval()

    val = run_epoch(
        model,
        val_loader,
        lam=lam,
        arm=arm,
        size_weights=None,
    )

    test = run_epoch(
        model,
        test_loader,
        lam=lam,
        arm=arm,
        size_weights=None,
    )

    outdir = Path(
        "phase224_small_pocket/cache"
    )
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = (
        outdir
        / f"phase224_{arm}_seed{seed}.pt"
    )

    torch.save(
        model.state_dict(),
        checkpoint,
    )

    return (
        val,
        test,
        str(checkpoint),
        median_size,
    )


# ============================================================
# 24 RUNS: 3 ARMS × 8 SEEDS
# ============================================================

SEEDS = list(range(8))

results = []

t0 = time.time()

for arm in ARMS:

    for seed in SEEDS:

        val, test, checkpoint, median_size = (
            train_arm(
                arm,
                seed,
            )
        )

        row = {
            'arm': arm,
            'seed': seed,
            'train_median_pocket_size': median_size,
            'val': val,
            'test': test,
            'checkpoint': checkpoint,
        }

        results.append(row)

        print(
            f"{arm:10s} "
            f"seed={seed} "
            f"val_PR={val['pr_auc']:.4f} "
            f"val_ROC={val['roc_auc']:.4f} "
            f"test_PR={test['pr_auc']:.4f} "
            f"test_ROC={test['roc_auc']:.4f} "
            f"MCC={test['mcc']:.4f} "
            f"DCC4={test['dcc4']:.3f} "
            f"elapsed={(time.time()-t0)/60:.1f}m"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

outdir = Path(
    "phase224_small_pocket"
)

with open(
    outdir / "phase224_small_pocket_results.json",
    "w",
) as f:
    json.dump(
        results,
        f,
        indent=2,
    )


import pandas as pd

rows = []

for r in results:

    rows.append({
        'arm': r['arm'],
        'seed': r['seed'],
        'train_median_pocket_size':
            r['train_median_pocket_size'],
        **{
            f"val_{k}": v
            for k, v in r['val'].items()
        },
        **{
            f"test_{k}": v
            for k, v in r['test'].items()
        },
    })

df = pd.DataFrame(rows)

df.to_csv(
    outdir / "phase224_small_pocket_seed_results.csv",
    index=False,
)

summary = (
    df.groupby('arm')
    .agg({
        'seed': ['count'],
        'val_pr_auc': ['mean', 'std'],
        'test_pr_auc': ['mean', 'std'],
        'test_roc_auc': ['mean', 'std'],
        'test_mcc': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'test_precision': ['mean', 'std'],
        'test_recall': ['mean', 'std'],
        'test_dcc4': ['mean', 'std'],
        'test_mean_center_error': ['mean', 'std'],
    })
    .reset_index()
)

summary.columns = [
    'arm',
    'n_seeds',
    'val_pr_auc_mean',
    'val_pr_auc_std',
    'test_pr_auc_mean',
    'test_pr_auc_std',
    'test_roc_auc_mean',
    'test_roc_auc_std',
    'test_mcc_mean',
    'test_mcc_std',
    'test_f1_mean',
    'test_f1_std',
    'test_precision_mean',
    'test_precision_std',
    'test_recall_mean',
    'test_recall_std',
    'test_dcc4_mean',
    'test_dcc4_std',
    'test_center_error_mean',
    'test_center_error_std',
]

summary.to_csv(
    outdir / "phase224_small_pocket_summary.csv",
    index=False,
)

print()
print(
    "===== PHASE 224 SUMMARY ====="
)
print(
    summary.to_string(index=False)
)
