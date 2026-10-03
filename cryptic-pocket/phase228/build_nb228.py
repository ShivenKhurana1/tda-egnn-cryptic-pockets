#!/usr/bin/env python3
"""Phase 228: does topology add anything beyond rich plain geometry? 6 arms x 8 seeds.

Reuses Phase164's model, loss, training loop and split verbatim. The ONLY changes: the topology/geometry
side channel is 51 wide for every arm (34 TDA + 17 geometry, unused columns zeroed) so all arms share
the identical architecture, and the arm definitions."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
nb164 = json.load(open(HERE.parent / "phase164_cryptobench_tda_component_ablation.ipynb"))
src = lambda i: "".join(nb164["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r'''
# Phase 228 - Does topology add anything beyond plain geometry?

**Motivation.** Phase 164 found a 2-feature burial baseline (atoms within 12 Å + gyration) already recovers
most of the TDA gain (test PR-AUC 0.2664 vs all-TDA 0.2676; persistence-only vs burial: PR p=0.74, ROC +0.0105,
8/8 seeds). The obvious objection is that "TDA" is a fancy burial measure. This run tests it directly against a
deliberately strong non-topological feature set (Phase 227: 17 features -- neighbour counts at 6 radii, relative
SASA, half-sphere exposure, local shape eigen-ratios, ray buriedness, convex-hull depth, residue depth).

**Design (fixed before any result).** Same Phase159 graphs, official split, Phase164 model/loss/optimizer/
40-epoch max/patience 8/8 seeds. Every arm keeps structure + ESM; ANM-MSF stays excluded. The gated-fusion side
channel is 51 wide for every arm (34 TDA + 17 geometry, unselected columns zeroed), so architecture is
identical across arms.

| arm | side-channel content |
|---|---|
| esm_structure | none |
| geom | 17 geometry |
| tda | 34 TDA (persistence + Betti + burial) |
| persistence | 23 persistence summaries only (no Betti, no burial) |
| geom_tda | geometry + all TDA |
| geom_persistence | geometry + persistence only |

**Primary question (pre-specified):** `geom_persistence` vs `geom` on validation PR-AUC -- does *topology proper* add
to geometry? Secondary: `geom_tda` vs `geom`; `tda` vs `geom` (which is better alone); `geom` vs `esm_structure`
(is geometry itself the gain); test PR-AUC/ROC-AUC for all of these. Paired by seed; paired t and Wilcoxon;
both must agree to call an effect confirmed. Test data is never used for selection.
''')

md("## Cell 1 - Install")
code("!pip install -q torch-geometric==2.6.1 scipy")

md("## Cell 2 - Load Phase159 graphs and Phase227 geometry; append 17 standardized geometry columns")
code(r'''
import copy, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, f1_score, precision_score, recall_score
from scipy.stats import ttest_rel, wilcoxon

def locate(f):
    hits=[]
    for root in (Path('/kaggle/input'), Path('/kaggle/working'), Path('.')):
        if root.exists(): hits.extend(root.rglob(f))
    if not hits: raise FileNotFoundError(f'Could not find {f}')
    return max(hits, key=lambda p:p.stat().st_size)

blob=torch.load(locate('phase159_cryptobench_anmmsf_graphs.pt'),map_location='cpu',weights_only=False)
geo=torch.load(locate('phase227_geometry_features.pt'),map_location='cpu',weights_only=False)
dataset=blob['dataset']
N_PRE=int(blob['n_pre_topo']); TDA_DIM=int(blob['topo_dim']); MSF_COL_IDX=int(blob['msf_col_idx']); POS_WEIGHT=float(blob['pos_weight'])
assert N_PRE==508 and TDA_DIM==34 and MSF_COL_IDX==507
GEOM_NAMES=list(geo['names']); GEOM_DIM=len(GEOM_NAMES); assert GEOM_DIM==17
SIDE_DIM=TDA_DIM+GEOM_DIM   # 51: identical side-channel width for every arm
mu=torch.from_numpy(np.asarray(geo['train_mean'],np.float32)); sd=torch.from_numpy(np.asarray(geo['train_std'],np.float32))

n_bad=0
for d in dataset:
    f=geo['feats'][(str(d.pdb_id).lower(),str(d.chain_id))]
    assert f.shape==(d.x.size(0),GEOM_DIM), (d.pdb_id,f.shape,d.x.shape)
    z=((torch.from_numpy(f)-mu)/sd).clamp(-5,5)          # train-split statistics only
    d.x=torch.cat([d.x,z],dim=1)
assert dataset[0].x.size(1)==N_PRE+SIDE_DIM
splits={s:[d for d in dataset if d.split==s] for s in ('train','val','test')}
print('node dim:',dataset[0].x.shape[1],'splits:',{k:len(v) for k,v in splits.items()})

GPU_OK=False
if torch.cuda.is_available():
    try:
        q=torch.randn(256,256,device='cuda'); (q@q).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print('GPU:',torch.cuda.get_device_name(0),'PASS')
    except Exception as e: print('GPU probe failed:',e)
DEVICE=torch.device('cuda' if GPU_OK else 'cpu')
WORK=Path('/kaggle/working'); (WORK/'cache').mkdir(exist_ok=True)
''')

md("## Cell 3 - Model (Phase164, unchanged; side channel width = 51)")
code(src(4).replace("model=LocalizingEGNN(N_PRE,TOPO_DIM)", "model=LocalizingEGNN(N_PRE,SIDE_DIM)"))

md("## Cell 4 - Arms")
code(r'''
TOPO_NAMES=list(blob['topo_names']); assert len(TOPO_NAMES)==34
PERSISTENCE_LOCAL=[0,1,2,3,4,5,6, 10,11,12,13,14,15,16,17, 21,22,23,24,25,26,27,28]
TDA_COLS=[N_PRE+i for i in range(34)]
PERS_COLS=[N_PRE+i for i in PERSISTENCE_LOCAL]
GEOM_COLS=[N_PRE+34+i for i in range(GEOM_DIM)]
ARMS={
 'esm_structure':[],
 'geom':GEOM_COLS,
 'tda':TDA_COLS,
 'persistence':PERS_COLS,
 'geom_tda':sorted(GEOM_COLS+TDA_COLS),
 'geom_persistence':sorted(GEOM_COLS+PERS_COLS),
}
def masked_dataset(ds,arm):
    out=[]
    for d in ds:
        d2=copy.copy(d); d2.x=d.x.clone(); keep=torch.zeros(d2.x.size(1),dtype=torch.bool)
        keep[0:507]=True            # structure + ESM (ANM-MSF column 507 stays excluded)
        keep[ARMS[arm]]=True
        d2.x[:,~keep]=0.0
        out.append(d2)
    return out
masked={arm:{s:masked_dataset(splits[s],arm) for s in splits} for arm in ARMS}
for arm in ARMS: print(f'{arm:18s} side-channel features: {len(ARMS[arm])}')
''')

md("## Cell 5 - Training machinery (Phase164, unchanged apart from side-channel width)")
code(src(6).replace("model=LocalizingEGNN(N_PRE,TOPO_DIM).to(DEVICE)","model=LocalizingEGNN(N_PRE,SIDE_DIM).to(DEVICE)").replace("phase163_{arm}_seed{seed}","phase228_{arm}_seed{seed}"))

md("## Cell 6 - Run 6 arms × 8 seeds (48 trainings)")
code(src(8).replace("phase164_tda_component_results.json","phase228_geometry_control_results.json"))

md("## Cell 7 - Summary table")
code(src(9).replace("phase164_tda_component_seed_results.csv","phase228_seed_results.csv").replace("phase164_tda_component_summary.csv","phase228_summary.csv"))

md("## Cell 8 - Pre-specified paired statistics")
code(r'''
import pandas as pd
def vec(arm,split,metric):
    rr=sorted([r for r in results if r['arm']==arm],key=lambda x:x['seed'])
    return np.array([r[split][metric] for r in rr],float)
COMPARISONS=[('geom_persistence','geom'),('geom_tda','geom'),('tda','geom'),('persistence','geom'),
             ('geom','esm_structure'),('tda','esm_structure'),('geom_tda','tda'),('geom_persistence','persistence')]
rows=[]
for a,b in COMPARISONS:
    for split,metric in [('val','pr_auc'),('test','pr_auc'),('test','roc_auc'),('test','mcc'),('test','dcc4'),('test','mean_center_error')]:
        x,y=vec(a,split,metric),vec(b,split,metric); dlt=x-y
        try: wp=wilcoxon(x,y).pvalue
        except Exception: wp=float('nan')
        rows.append(dict(a=a,b=b,split=split,metric=metric,mean_a=x.mean(),mean_b=y.mean(),mean_delta=dlt.mean(),
                         a_wins=int((dlt>0).sum()),t_p=float(ttest_rel(x,y).pvalue),wilcoxon_p=float(wp),
                         cohens_d=float(dlt.mean()/(dlt.std(ddof=1)+1e-12))))
st=pd.DataFrame(rows); st.to_csv(WORK/'phase228_paired_statistics.csv',index=False)
pd.set_option('display.width',250)
print(st[st.metric.isin(['pr_auc','roc_auc'])].round(4).to_string(index=False))
''')

md('''
## Interpretation (filled after the run, not before)

Read `phase228_paired_statistics.csv`. The primary line is `geom_persistence − geom` on `val pr_auc`; a result counts
as confirmed only if paired t-test and Wilcoxon both fall below 0.05. Do not use test metrics to choose anything.
''')

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
o = HERE / "phase228_geometry_control.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
