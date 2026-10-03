#!/usr/bin/env python3
"""Phase 255: Phase 11's nested-CV geometry control (Phase 230), on the 7,513-chain scale-3 pool.
usage: build_nb255.py <a|b|c|d|e> <output_dir>    folds: a=0,1  b=2,3  c=4,5  d=6,7  e=8,9   (2 seeds x 4 arms per fold)"""
import json, sys
from pathlib import Path

TAG = sys.argv[1]; OUT_DIR = Path(sys.argv[2]).resolve()
FOLDS = {"a": [0, 1], "b": [2, 3], "c": [4, 5], "d": [6, 7], "e": [8, 9]}[TAG]
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
nb11 = json.load(open(ROOT / "phase11_augmented_nested_cv.ipynb"))
src11 = lambda i: "".join(nb11["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase 255{TAG} - Classification geometry control at the 7,513-chain pool (folds {FOLDS[0]}, {FOLDS[1]})

Phase 230 (2,055 chains, ESM2-35M): geometry beat topology alone (+0.0129 PR-AUC, 10/10 folds) and topology added a small residual
over geometry (+0.0097, 9/10 folds, PR-AUC only). Phases 240/241 found that residual grows at 650M/3B embedding scale. The untested
explanation offered for why the main pipeline keeps a residual while CryptoBench (700 chains) does not is training-set size. This
runs the identical control on the round-3 pool (3.65x the chains) at ESM2-35M.

Design identical to Phase 230: 10-fold KFold (seed 1337) over the pool (every chain is its own <30%-identity cluster, so the folds are
homology-clean), 2 seeds, 15% of each training fold for early stopping, 40 epochs, patience 8; identical 51-wide side channel for every
arm. Arms: `no-topo`, `full` (34 topology), `geom` (17), `geom+topo` (51). **Primary comparison (fixed in advance): `geom+topo` vs `geom`
on test PR-AUC, paired over the 10 folds** (this kernel runs folds {FOLDS}; the five kernels are merged locally). No thresholded metric.
''')
md("## Cell 1 - Install")
code("!pip install -q torch-geometric==2.6.1")
md("## Cell 2 - Load the Phase 52 pool and the Phase 250 geometry (3 chunks); append 17 standardized geometry columns")
code(r'''
import json, os, time, warnings
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

blob = torch.load(locate("phase52_graphs.pt"), map_location="cpu", weights_only=False)
N_PRE = int(blob["n_pre_topo"]); TOPO_DIM = int(blob["topo_dim"]); POS_WEIGHT = float(blob["pos_weight"])
assert N_PRE == 507 and TOPO_DIM == 34, (N_PRE, TOPO_DIM)
feats = {}
for t in "abc":
    g = torch.load(locate(f"phase250{t}_geometry_scale3.pt"), map_location="cpu", weights_only=False)
    feats.update(g["feats"]); print(f"geometry chunk {t}: {g['n_chains']}/{g['n_total']} accepted")
GEOM_DIM = 17; SIDE_DIM = TOPO_DIM + GEOM_DIM

dataset, dropped = [], 0
for d in blob["dataset"]:                                       # blob order, restricted to chains with geometry (identical in every kernel)
    if (str(d.pdb_id), str(d.chain_id)) in feats: dataset.append(d)
    else: dropped += 1
del blob
assert dropped <= 0.02 * (len(dataset) + dropped), f"too many chains without geometry: {dropped}"
allg = np.concatenate([feats[(str(d.pdb_id), str(d.chain_id))] for d in dataset], 0)
g_mu = torch.from_numpy(allg.mean(0).astype(np.float32)); g_sd = torch.from_numpy(np.maximum(allg.std(0), 1e-6).astype(np.float32)); del allg
for d in dataset:
    f = feats[(str(d.pdb_id), str(d.chain_id))]; assert f.shape == (d.x.size(0), GEOM_DIM)
    d.x = torch.cat([d.x, ((torch.from_numpy(f) - g_mu) / g_sd).clamp(-5, 5)], dim=1)
del feats
print(f"chains kept {len(dataset):,}, dropped for missing geometry {dropped}; node dim {dataset[0].x.size(1)} = {N_PRE} + {TOPO_DIM} + {GEOM_DIM}")

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
if hasattr(dataset[0], "cluster"):
    assert len({d.cluster for d in dataset}) == len(dataset), "not 1 chain per cluster -- KFold would leak homologs"
    print("verified: one chain per cluster, KFold over chains is homology-clean")
else:
    print("WARNING: no `cluster` attribute on the graphs; homology-cleanliness of the folds is not re-verified here")
''')
md("## Cell 3 - Model and training machinery (Phase 11, unchanged)")
code(src11(8))
md(f"## Cell 4 - Arms and the nested CV loop (folds {FOLDS})")
code(f'''
FOLDS_TO_RUN = {FOLDS}
''' + r'''
N_FOLDS, SEEDS, EPOCHS, PATIENCE, BATCH = 10, [0,1], 40, 8, 16
ARMS = ["no-topo", "full", "geom", "geom+topo"]
KEEP_SIDE = {"no-topo": [], "full": list(range(TOPO_DIM)), "geom": list(range(TOPO_DIM, SIDE_DIM)), "geom+topo": list(range(SIDE_DIM))}
SIDE0 = [d.x[:, N_PRE:].clone() for d in dataset]        # original side channels only; the wide ESM block is never duplicated
def apply_arm(a):
    m = torch.zeros(SIDE_DIM); m[KEEP_SIDE[a]] = 1.0
    for d, s0 in zip(dataset, SIDE0): d.x[:, N_PRE:] = s0 * m

folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=1337).split(np.arange(len(dataset))))
records = []; t0 = time.time()
for arm in ARMS:
    apply_arm(arm)
    for fi in FOLDS_TO_RUN:
        tr_idx, te_idx = folds[fi]
        for seed in SEEDS:
            rng = np.random.RandomState(500+seed)
            perm = rng.permutation(tr_idx)
            nv = max(1, int(0.15*len(perm)))
            va_ds=[dataset[i] for i in perm[:nv]]; tr_ds=[dataset[i] for i in perm[nv:]]
            te_ds=[dataset[i] for i in te_idx]
            torch.manual_seed(seed); np.random.seed(seed)
            m = Net(N_PRE, SIDE_DIM).to(DEVICE)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
            tl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True); vl = DataLoader(va_ds, batch_size=BATCH)
            best, state, bad = -1, None, 0
            for ep in range(EPOCHS):
                run_epoch(m, tl, opt); v = run_epoch(m, vl); sch.step()
                if v["pr"] > best: best, state, bad = v["pr"], {k:t.clone() for k,t in m.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= PATIENCE: break
            m.load_state_dict(state)
            te = run_epoch(m, DataLoader(te_ds, batch_size=BATCH))
            records.append({"arm": arm, "fold": fi, "seed": seed, "val_pr": float(best), "test_pr": float(te["pr"]),
                            "test_roc": float(te["roc"]), "n_test": int(len(te["y"])), "n_pos": int(te["y"].sum()), "epochs": ep + 1})
            json.dump({"records": records, "folds": FOLDS_TO_RUN}, open("/kaggle/working/phase255__TAG___records.json", "w"), indent=2, default=float)
            print(f"  {arm:<10} fold {fi:<2} seed {seed} | val PR {best:.3f} | test PR {te['pr']:.3f} ROC {te['roc']:.3f} [{(time.time()-t0)/60:.0f}m]", flush=True)
print(f"\n{len(records)} runs in {(time.time()-t0)/60:.1f} min")
'''.replace("__TAG__", TAG))
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT_DIR.mkdir(parents=True, exist_ok=True)
o = OUT_DIR / f"phase255{TAG}_cls_geometry_scale3.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
meta = {"id": f"blackpearls1/phase255{TAG}-cls-geometry-scale3", "title": f"Phase255{TAG} Cls Geometry Scale3", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase52-esm-dihedral-scale3", "blackpearls1/phase250a-geometry-scale3",
                           "blackpearls1/phase250b-geometry-scale3", "blackpearls1/phase250c-geometry-scale3"]}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
