#!/usr/bin/env python3
"""Phase 232: the geometry control on the manuscript's own evaluation (Phase166 hard split, 185-protein test set),
with residue-probability export so experimental-site metrics can be computed exactly as in Phases 167-189."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
nb228 = json.load(open(HERE.parent / "phase228" / "phase228_geometry_control.ipynb"))
c228 = nb228["cells"]
s228 = lambda i: "".join(c228[i]["source"])
# cells of nb228: 0 md,1 install,2 load,3 md,4 model,5 md,6 arms,7 md,8 train machinery,9 md,10 loop,11 md,12 summary,...
def find(prefix):
    for i, c in enumerate(c228):
        if c["cell_type"] == "code" and "".join(c["source"]).lstrip().startswith(prefix): return i
    raise KeyError(prefix)

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r'''
# Phase 232 — Geometry control on the manuscript's own evaluation (Phase 166 hard split, 185-protein test set)

Phase 228 ran the geometry-vs-topology control on the *official* CryptoBench split with CryptoBench's labels.
The manuscript's experimental-site results (Phases 167-189) instead use the **Phase 166 embedding-cluster hard split**
(668 / 89 / 185 graphs) and ligand-contact labels derived from holo structures. Models trained on the official
train split have seen many of those 185 proteins, so Phase 228's checkpoints cannot be scored there. This notebook
retrains the arms on the Phase 166 hard split (identical deterministic recipe) and **exports every test residue
probability** for downstream experimental-site analysis.

Arms (same 51-wide side channel for all): `esm_structure`, `tda` (34), `geom` (17), `geom_tda`, `geom_persistence`.
8 seeds each = 40 trainings. Primary question (fixed in advance): `geom_persistence` vs `geom`.
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1 scipy scikit-learn")
md("## Cell 2 — Load Phase 159 graphs + Phase 227 geometry, build the Phase 166 hard split")
code(s228(find("import copy")).replace("from scipy.stats import ttest_rel, wilcoxon", "from scipy.stats import ttest_rel, wilcoxon\nfrom sklearn.cluster import KMeans")
     .replace("splits={s:[d for d in dataset if d.split==s] for s in ('train','val','test')}\nprint('node dim:',dataset[0].x.shape[1],'splits:',{k:len(v) for k,v in splits.items()})", "OFFICIAL={s:sum(d.split==s for d in dataset) for s in ('train','val','test')}\nprint('official split:',OFFICIAL)"))
md("## Cell 3 — Phase 166 hard split (verbatim recipe) and consistency assertion")
code(r'''
emb=np.stack([d.x[:,27:507].float().mean(0).numpy() for d in dataset])
emb=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-12)
N_CLUSTERS=30
km=KMeans(n_clusters=N_CLUSTERS, random_state=166, n_init=20, max_iter=300)
labels=km.fit_predict(emb)
sizes=np.bincount(labels, minlength=N_CLUSTERS)
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
sizes_hard={s:sum(d.hard_split==s for d in dataset) for s in hard_split.values()}
print('hard split sizes:',sizes_hard)
# must equal the Phase 166 manifest (668 / 89 / 185) or the split is not the manuscript's
assert sizes_hard=={'train_hard':668,'val_hard':89,'test_hard':185}, sizes_hard
EXPECTED_CLUSTER_ASSIGNMENT={str(k):v for k,v in {0:'train_hard',1:'test_hard',2:'train_hard',3:'val_hard',4:'test_hard',5:'train_hard',6:'train_hard',7:'val_hard',8:'train_hard',9:'train_hard',10:'train_hard',11:'train_hard',12:'train_hard',13:'train_hard',14:'train_hard',15:'test_hard',16:'test_hard',17:'train_hard',18:'train_hard',19:'train_hard',20:'train_hard',21:'train_hard',22:'train_hard',23:'test_hard',24:'val_hard',25:'test_hard',26:'train_hard',27:'train_hard',28:'train_hard',29:'val_hard'}.items()}
got={str(k):hard_split[assign[k]] for k in range(N_CLUSTERS)}
assert got==EXPECTED_CLUSTER_ASSIGNMENT, 'cluster assignment differs from the Phase 166 manifest'
print('cluster assignment identical to phase166_hard_split_manifest.json')
splits={'train':[d for d in dataset if d.hard_split=='train_hard'],'val':[d for d in dataset if d.hard_split=='val_hard'],'test':[d for d in dataset if d.hard_split=='test_hard']}
''')
md("## Cell 4 — Model (Phase 164/228, unchanged; side channel 51)")
code(s228(find("class EGCL")))
md("## Cell 5 — Arms")
code(s228(find("TOPO_NAMES=list")).replace("ARMS={\n 'esm_structure':[],\n 'geom':GEOM_COLS,\n 'tda':TDA_COLS,\n 'persistence':PERS_COLS,\n 'geom_tda':sorted(GEOM_COLS+TDA_COLS),\n 'geom_persistence':sorted(GEOM_COLS+PERS_COLS),\n}",
     "ARMS={\n 'esm_structure':[],\n 'tda':TDA_COLS,\n 'geom':GEOM_COLS,\n 'geom_tda':sorted(GEOM_COLS+TDA_COLS),\n 'geom_persistence':sorted(GEOM_COLS+PERS_COLS),\n}"))
md("## Cell 6 — Training machinery (unchanged) with test-probability export")
train_src = s228(find("def focal_loss_with_logits")).replace("phase163_","phase232_").replace("phase228_{arm}_seed{seed}", "phase232_{arm}_seed{seed}")
train_src = train_src.replace("model=LocalizingEGNN(N_PRE,SIDE_DIM).to(DEVICE)", "model=LocalizingEGNN(N_PRE,SIDE_DIM).to(DEVICE)")
train_src += r'''

@torch.no_grad()
def export_test_probs(model,arm):
    model.eval(); out={}
    for d in masked[arm]['test']:
        b=d.clone().to(DEVICE) if hasattr(d,'clone') else d.to(DEVICE)
        logits,_=model(b)
        out[(str(d.pdb_id).lower(),str(d.chain_id))]=torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    return out
'''
code(train_src)
md("## Cell 7 — 5 arms x 8 seeds = 40 trainings")
code(r'''
SEEDS=list(range(8)); results=[]; PROBS={}; t0=time.time()
for arm in ARMS:
    for seed in SEEDS:
        val,te,path=train_arm(arm,seed)
        m=LocalizingEGNN(N_PRE,SIDE_DIM).to(DEVICE); m.load_state_dict(torch.load(path,map_location=DEVICE)); PROBS[(arm,seed)]=export_test_probs(m,arm)
        results.append({'arm':arm,'seed':seed,'val':val,'test':te,'checkpoint':path})
        print(f"{arm:18s} seed={seed} val_PR={val['pr_auc']:.4f} test_PR={te['pr_auc']:.4f} test_ROC={te['roc_auc']:.4f} elapsed={(time.time()-t0)/60:.1f}m")
json.dump(results,open(WORK/'phase232_results.json','w'),indent=2)
torch.save(PROBS,WORK/'phase232_test_probabilities.pt')
print('saved phase232_results.json and phase232_test_probabilities.pt')
''')
md("## Cell 8 — Summary (protein-level analysis is done locally on the exported probabilities)")
code(r'''
import pandas as pd
rows=[{'arm':r['arm'],'seed':r['seed'],**{f'val_{k}':v for k,v in r['val'].items()},**{f'test_{k}':v for k,v in r['test'].items()}} for r in results]
df=pd.DataFrame(rows); df.to_csv(WORK/'phase232_seed_results.csv',index=False)
print(df.groupby('arm')[['val_pr_auc','test_pr_auc','test_roc_auc']].agg(['mean','std']).round(4).to_string())
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
o = HERE / "phase232_geometry_control_hardsplit.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
