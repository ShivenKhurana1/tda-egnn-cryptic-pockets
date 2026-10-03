#!/usr/bin/env python3
"""Phase 11: nested 10-fold CV of augmented model, paired against P2Rank per fold."""
import json
from pathlib import Path

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 11 - Validating the Win: Nested CV vs P2Rank, Matched Per Fold

Phase 10 showed the ESM2+dihedral-augmented model beating P2Rank on **one fixed
split**: PR-AUC 0.5002 vs 0.4589. That is encouraging, but it is exactly the
kind of single-split claim that Phase 5b already taught this project not to
trust - the same protocol gave +0.054 on one split and +0.0423 (properly, over
10 folds) on the real measurement, and a single split's sign even flipped once
before. A one-shot win against P2Rank deserves the same skepticism.

This notebook applies the **Phase 6c protocol** (10-fold KFold over 2,056
non-redundant clusters - homology-clean by construction, since each chain is its
cluster's sole representative) to the augmented features, and adds one thing
Phase 7 could not do on a single split: **P2Rank is run once on all 2,056 chains
and cached, then matched against the model's test set in every fold**, giving a
paired, fold-level comparison against P2Rank instead of a single-split point
estimate.

Two questions, both properly powered this time:

1. Does topology still help once the model has rich per-residue features?
   (`full` vs `burial-only`, mirroring Phase 6c exactly)
2. Does the augmented model reliably beat P2Rank, or was Phase 10 a lucky split?
   (`full` vs P2Rank, matched per fold - a new comparison, not available before)
''')

md(r'''## Cell 1 - Setup''')

code(r'''
!pip install -q torch-geometric==2.6.1 biopython==1.84
!apt-get -qq update > /dev/null 2>&1 && apt-get -qq install -y default-jre-headless > /dev/null 2>&1
''')

code(r'''
import gzip, json, os, shutil, subprocess, time, urllib.request, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

blob10 = torch.load(locate("phase10_graphs.pt"), map_location="cpu", weights_only=False)
blob6b = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)
dataset = blob10["dataset"]
N_PRE = int(blob10["n_pre_topo"])
TOPO_DIM = int(blob10["topo_dim"])
POS_WEIGHT = float(blob10["pos_weight"])
TOPO_NAMES = blob6b["topo_names"]          # same 34-column block, reused for naming/indexing
BURIAL_IDX = [TOPO_NAMES.index(n) for n in blob6b["burial_names"]]
assert len(TOPO_NAMES) == TOPO_DIM

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")

n_res = sum(int(d.num_nodes) for d in dataset)
n_pos = sum(int(d.y.sum()) for d in dataset)
print(f"chains {len(dataset):,} | residues {n_res:,} ({100*n_pos/n_res:.2f}% positive) "
      f"| node dim {dataset[0].x.size(1)} (pre-topo {N_PRE} + topo {TOPO_DIM})")
assert len({d.cluster for d in dataset}) == len(dataset), "not 1 chain per cluster"
print("verified: KFold over chains is homology-clean")
''')

md(r'''
## Cell 2 - Run P2Rank once on all 2,056 chains, cache per-residue scores

The expensive part (running P2Rank) happens exactly once here, up front, so it
never needs to be repeated inside the 60-run training loop below. Scores are
aligned to each graph's exact residue order and cached as a plain array,
`None` where a chain could not be matched (dropped from the P2Rank comparison
only - the model's own training/evaluation on that chain is unaffected).
''')

code(r'''
from Bio.PDB import PDBParser

AA3={"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
     "MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE"}
RAW=Path("/tmp/pdb_raw"); RAW.mkdir(parents=True, exist_ok=True)

t0=time.time(); PRANK=None
subprocess.run("curl -sL https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz "
               "-o /tmp/p2.tar.gz && tar xzf /tmp/p2.tar.gz -C /tmp", shell=True, check=True, timeout=1200)
PRANK = str(next(Path("/tmp").glob("p2rank_*/prank")))
os.chmod(PRANK, 0o755)
print(f"P2Rank ready in {time.time()-t0:.0f}s -> {PRANK}")

need = sorted({d.pdb_id for d in dataset})
def fetch(pid):
    p=RAW/f"{pid}.pdb.gz"
    if p.exists() and p.stat().st_size>500: return pid,p
    try:
        with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.pdb.gz",timeout=60) as r:
            b=r.read()
        if len(b)>500: p.write_bytes(b); return pid,p
    except Exception: pass
    return pid,None

t0=time.time()
with ThreadPoolExecutor(max_workers=16) as ex: got=list(ex.map(fetch, need))
paths={p:f for p,f in got if f}
print(f"fetched {len(paths):,}/{len(need):,} in {(time.time()-t0)/60:.1f} min")

W=Path("/tmp/p2in"); shutil.rmtree(W, ignore_errors=True); W.mkdir()
OUT=Path("/tmp/p2out"); shutil.rmtree(OUT, ignore_errors=True)
for pid, gz in paths.items():
    plain = W / f"{pid}.pdb"
    if not plain.exists():
        with gzip.open(gz,"rb") as fi, open(plain,"wb") as fo: shutil.copyfileobj(fi,fo)
ds_file = W/"all.ds"
ds_file.write_text("\n".join(str(W/f"{p}.pdb") for p in sorted(paths))+"\n")

t0=time.time()
r = subprocess.run([PRANK,"predict",str(ds_file),"-o",str(OUT/"all"),
                    "-threads", str(max(1, os.cpu_count() or 2))],
                   capture_output=True, text=True, timeout=10800)
print(f"P2Rank rc={r.returncode} in {(time.time()-t0)/60:.1f} min")
if r.returncode != 0:
    print("STDERR tail:", (r.stderr or "")[-1000:])
csv_files = {p.stem.replace(".pdb_residues",""): p for p in (OUT/"all").rglob("*_residues.csv")}
print(f"residue-level outputs: {len(csv_files):,}/{len(paths):,}")
''')

code(r'''
import csv as _csv

def p2rank_chain_scores(pid):
    f = csv_files.get(pid)
    if f is None: return None
    out = {}
    with open(f) as fh:
        for row in _csv.DictReader(fh):
            row = {k.strip(): (v.strip() if isinstance(v,str) else v) for k,v in row.items() if k}
            ch = row.get("chain","")
            lab = row.get("residue_label","")
            sc = row.get("probability") or row.get("zscore") or row.get("score") or "0"
            num = "".join(c for c in lab if c.isdigit() or c=="-")
            if not ch or not num: continue
            try: out[(ch,int(num))] = float(sc)
            except ValueError: continue
    return out or None

def residue_keys(pid, cid, n_expect):
    p = paths.get(pid)
    if p is None: return None
    try:
        with gzip.open(p,"rt") as fh:
            model_ = next(PDBParser(QUIET=True, PERMISSIVE=True).get_structure(pid, fh).get_models())
        ch = model_[cid]
    except Exception:
        return None
    keys = [(cid, r.id[1]) for r in ch if r.get_resname().strip().upper() in AA3 and "CA" in r]
    return keys if len(keys) == n_expect else None

t0=time.time()
p2_cache = {}
p2_scores = [None]*len(dataset)
matched = 0
for i, d in enumerate(dataset):
    sc = p2_cache.get(d.pdb_id)
    if sc is None and d.pdb_id not in p2_cache:
        sc = p2rank_chain_scores(d.pdb_id); p2_cache[d.pdb_id] = sc
    if sc is None: continue
    keys = residue_keys(d.pdb_id, d.chain_id, int(d.num_nodes))
    if keys is None: continue
    arr = np.array([sc.get(k, 0.0) for k in keys], dtype=np.float32)
    if arr.max() <= 0: continue
    p2_scores[i] = arr; matched += 1

print(f"P2Rank aligned for {matched:,}/{len(dataset):,} chains ({(time.time()-t0)/60:.1f} min)")
''')

md(r'''## Cell 3 - Model (identical to every prior phase, wider input)''')

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

def run_epoch(model, loader, opt=None):
    tr = opt is not None; model.train(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
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
''')

md(r'''
## Cell 4 - Nested 10-fold CV, 2 seeds, 3 arms, plus matched P2Rank per fold

Same discipline as Phase 6c: no test fold ever influences early stopping, and
`KFold` over chains is homology-clean because every chain is its cluster's sole
representative (asserted above). P2Rank's fold-level score is computed from the
Cell-2 cache - no re-running P2Rank inside the loop.
''')

code(r'''
N_FOLDS, SEEDS, EPOCHS, PATIENCE, BATCH = 10, [0,1], 40, 8, 16
ARMS = ["no-topo", "burial-only", "full"]
ORIG_X = [d.x.clone() for d in dataset]

def apply_arm(a):
    for d, x0 in zip(dataset, ORIG_X):
        t = x0[:, N_PRE:].clone()
        if a == "no-topo": t.zero_()
        elif a == "burial-only":
            k = torch.zeros_like(t); k[:,BURIAL_IDX]=t[:,BURIAL_IDX]; t=k
        d.x = torch.cat([x0[:,:N_PRE], t], dim=1)

idx_all = np.arange(len(dataset))
folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=1337).split(idx_all))

# P2Rank per fold: computed once, independent of arm/seed
p2rank_fold = []
for fi, (_, te_idx) in enumerate(folds):
    ys, ps = [], []
    for i in te_idx:
        if p2_scores[i] is None: continue
        ys.append(dataset[i].y.numpy()); ps.append(p2_scores[i])
    if ys:
        y_, p_ = np.concatenate(ys), np.concatenate(ps)
        p2rank_fold.append({"fold": fi, "pr": average_precision_score(y_,p_),
                            "roc": roc_auc_score(y_,p_), "n": len(y_), "n_chains": len(ys)})
    else:
        p2rank_fold.append({"fold": fi, "pr": np.nan, "roc": np.nan, "n": 0, "n_chains": 0})
print("P2Rank per-fold coverage:", [r["n_chains"] for r in p2rank_fold])

records = []
t0 = time.time()
for arm in ARMS:
    apply_arm(arm)
    for fi, (tr_idx, te_idx) in enumerate(folds):
        for seed in SEEDS:
            rng = np.random.RandomState(500+seed)
            perm = rng.permutation(tr_idx)
            nv = max(1, int(0.15*len(perm)))
            va_ds=[dataset[i] for i in perm[:nv]]; tr_ds=[dataset[i] for i in perm[nv:]]
            te_ds=[dataset[i] for i in te_idx]

            torch.manual_seed(seed); np.random.seed(seed)
            m = Net(N_PRE, TOPO_DIM).to(DEVICE)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
            tl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
            vl = DataLoader(va_ds, batch_size=BATCH)
            best, state, bad = -1, None, 0
            for ep in range(EPOCHS):
                run_epoch(m, tl, opt); v = run_epoch(m, vl); sch.step()
                if v["pr"] > best: best, state, bad = v["pr"], {k:t.clone() for k,t in m.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= PATIENCE: break
            m.load_state_dict(state)
            te = run_epoch(m, DataLoader(te_ds, batch_size=BATCH))
            records.append({"arm": arm, "fold": fi, "seed": seed, "val_pr": best,
                            "test_pr": te["pr"], "test_roc": te["roc"],
                            "test_mcc": best_mcc(te["p"], te["y"]),
                            "n_test": int(len(te["y"])), "n_pos": int(te["y"].sum())})
        r = records[-1]
        print(f"  {arm:<12} fold {fi:<2} | val PR {r['val_pr']:.3f} | test PR {r['test_pr']:.3f} "
              f"ROC {r['test_roc']:.3f} [{(time.time()-t0)/60:.0f}m]")

for d, x0 in zip(dataset, ORIG_X): d.x = x0
print(f"\n{len(records)} runs in {(time.time()-t0)/60:.1f} min")
json.dump({"records": records, "p2rank_fold": p2rank_fold},
          open("/kaggle/working/phase11_records.json","w"), indent=2, default=float)
''')

md(r'''## Cell 5 - Statistics: topology contribution, and model vs P2Rank''')

code(r'''
from scipy.stats import ttest_rel, wilcoxon

def fold_means(arm, metric):
    return np.array([np.mean([r[metric] for r in records if r["arm"]==arm and r["fold"]==f])
                     for f in range(N_FOLDS)])

print(f"{'arm':<14}{'test PR-AUC':>18}{'test ROC-AUC':>18}{'test MCC':>16}")
print("-"*66)
for arm in ARMS:
    pr,roc,mcc = (fold_means(arm,m) for m in ("test_pr","test_roc","test_mcc"))
    print(f"{arm:<14}{pr.mean():>10.4f} +-{pr.std():.4f}{roc.mean():>11.4f} +-{roc.std():.4f}"
          f"{mcc.mean():>9.4f} +-{mcc.std():.4f}")
p2_pr = np.array([r["pr"] for r in p2rank_fold]); p2_roc = np.array([r["roc"] for r in p2rank_fold])
print(f"{'P2Rank (matched)':<14}{np.nanmean(p2_pr):>10.4f} +-{np.nanstd(p2_pr):.4f}"
      f"{np.nanmean(p2_roc):>11.4f} +-{np.nanstd(p2_roc):.4f}{'':>16}")

print("\n" + "="*78)
print(f"TOPOLOGY CONTRIBUTION (n={N_FOLDS} independent folds)")
d = fold_means("full","test_pr") - fold_means("burial-only","test_pr")
t_p = ttest_rel(fold_means("full","test_pr"), fold_means("burial-only","test_pr")).pvalue
w_p = wilcoxon(d).pvalue
print(f"  full - burial-only  PR-AUC  {d.mean():+.4f} +-{d.std():.4f} | {(d>0).sum()}/{N_FOLDS} wins "
      f"| t p={t_p:.5f} | W p={w_p:.4f}")
print(f"  95% CI: [{d.mean()-1.96*d.std()/np.sqrt(N_FOLDS):+.4f}, {d.mean()+1.96*d.std()/np.sqrt(N_FOLDS):+.4f}]")

print("\n" + "="*78)
print("MODEL vs P2RANK (matched per fold, n=%d)" % N_FOLDS)
model_pr = fold_means("full","test_pr")
valid = ~np.isnan(p2_pr)
dm = model_pr[valid] - p2_pr[valid]
print(f"  full - P2Rank       PR-AUC  {dm.mean():+.4f} +-{dm.std():.4f} | {(dm>0).sum()}/{valid.sum()} wins")
if valid.sum() >= 3:
    print(f"  paired t p={ttest_rel(model_pr[valid], p2_pr[valid]).pvalue:.5f} | "
          f"Wilcoxon p={wilcoxon(dm).pvalue:.4f}")

json.dump({
    "topology_contribution": {"mean": float(d.mean()), "std": float(d.std()),
                              "wins": int((d>0).sum()), "t_p": float(t_p), "w_p": float(w_p)},
    "vs_p2rank": {"mean": float(dm.mean()), "std": float(dm.std()),
                 "wins": int((dm>0).sum()), "n": int(valid.sum())},
    "arm_summary": {a: {m: [float(fold_means(a,m).mean()), float(fold_means(a,m).std())]
                        for m in ("test_pr","test_roc","test_mcc")} for a in ARMS},
    "p2rank_summary": {"pr": [float(np.nanmean(p2_pr)), float(np.nanstd(p2_pr))],
                       "roc": [float(np.nanmean(p2_roc)), float(np.nanstd(p2_roc))]},
}, open("/kaggle/working/phase11_summary.json","w"), indent=2)
print("\nsaved -> phase11_summary.json")
''')

md(r'''## Cell 6 - Plot''')

code(r'''
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2, figsize=(12.5,4.3))
ax[0].boxplot([fold_means(a,"test_pr") for a in ARMS] + [p2_pr[~np.isnan(p2_pr)]],
             tick_labels=ARMS+["P2Rank"])
ax[0].set_ylabel("test PR-AUC"); ax[0].set_title(f"{N_FOLDS}-fold CV, augmented features")

dd = fold_means("full","test_pr")[valid] - p2_pr[valid]
ax[1].bar(range(len(dd)), dd, color=["tab:green" if x>0 else "tab:red" for x in dd])
ax[1].axhline(0,color="k",lw=1); ax[1].axhline(dd.mean(),color="tab:blue",ls="--",label=f"mean {dd.mean():+.4f}")
ax[1].set_xlabel("fold"); ax[1].set_ylabel("PR-AUC diff"); ax[1].set_title("full - P2Rank per fold")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
''')

md(r'''
---

## Reading this

Two independent claims, both now at the fold-count that made Phase 6c
trustworthy:

- **Topology contribution** confirms (or revises) whether persistent homology
  still adds value once the model has rich per-residue chemistry from ESM2 -
  the Phase 10 single-split number (+0.0646) was suggestive, this is the
  measurement to actually cite.
- **Model vs P2Rank** is a genuinely new comparison this project did not have
  before: matched per fold rather than a single split, with the same paired
  statistics used everywhere else. If the win holds across most folds with a
  small p-value, that is real evidence of parity or superiority against a
  published, pre-trained standard - a substantially stronger claim than Phase
  10's point estimate.

If either number looks weaker than the single-split result, trust this one -
that has been the consistent lesson of every prior over-generalization in this
project.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase11_augmented_nested_cv.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
