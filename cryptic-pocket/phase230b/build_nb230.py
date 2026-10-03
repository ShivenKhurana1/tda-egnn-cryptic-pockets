#!/usr/bin/env python3
"""Phase 230: the Phase 11 nested-CV topology ablation, re-run with a plain-geometry control.

usage: build_nb230.py <tag a|b>   (a = folds 0-4, b = folds 5-9; two GPU kernels in parallel)"""
import json, sys
from pathlib import Path

TAG = sys.argv[1] if len(sys.argv) > 1 else "a"
FOLDS = {"a": [0, 1, 2, 3, 4], "b": [5, 6, 7, 8, 9]}[TAG]
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
nb11 = json.load(open(ROOT / "phase11_augmented_nested_cv.ipynb"))
src = lambda i: "".join(nb11["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase 230{TAG} - Does the main-pipeline topology gain survive a plain-geometry control? (folds {FOLDS[0]}–{FOLDS[-1]})

Phase 11 (10-fold x 2 seeds, 2,056 homology-clean chains, ESM2-35M + dihedrals): topology added +0.032 PR-AUC
over a 2-feature burial baseline, 10/10 folds. Phase 228 (CryptoBench) then showed topology adds nothing
beyond 17 richer plain-geometry features. This is the same control on the main pipeline.

**Design (fixed before any result).** Identical to Phase 11 (KFold seed 1337, 2 seeds, 15% of the training folds
for early stopping, 40 epochs, patience 8, same model/loss). The gated side channel is 51 wide for every arm
(34 TDA + 17 geometry, unselected columns zeroed). Arms: `no-topo`, `full` (all 34 TDA = Phase 11's full),
`geom` (17), `geom+topo` (51). Chains that fail the geometry alignment check are dropped from EVERY arm.

**Primary comparison:** `geom+topo` vs `geom` on test PR-AUC, paired over the 10 folds (this notebook runs half of
them; the two halves are merged locally). No thresholded metric is reported: Phase 11's MCC picked its threshold on
the test predictions, so it is omitted here. Test data never influences early stopping.
''')
md("## Cell 1 - Setup")
code("!pip install -q torch-geometric==2.6.1 biopython==1.84")
md("## Cell 2 - Load Phase 10 graphs and the Phase 229 geometry")
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

blob10 = torch.load(locate("phase10_graphs.pt"), map_location="cpu", weights_only=False)
geo = torch.load(locate("phase229_geometry_features.pt"), map_location="cpu", weights_only=False)
N_PRE = int(blob10["n_pre_topo"]); TOPO_DIM = int(blob10["topo_dim"]); POS_WEIGHT = float(blob10["pos_weight"])
assert TOPO_DIM == 34
GEOM_DIM = len(geo["names"]); assert GEOM_DIM == 17
SIDE_DIM = TOPO_DIM + GEOM_DIM
mu = torch.from_numpy(np.asarray(geo["mean"], np.float32)); sd = torch.from_numpy(np.asarray(geo["std"], np.float32))

dataset = []
dropped = 0
for d in blob10["dataset"]:
    k = (str(d.pdb_id), str(d.chain_id))
    if k not in geo["feats"]:
        dropped += 1; continue
    f = geo["feats"][k]
    assert f.shape == (d.x.size(0), GEOM_DIM)
    z = ((torch.from_numpy(f) - mu) / sd).clamp(-5, 5)      # label-free, all-chain statistics (as the TDA block)
    d.x = torch.cat([d.x, z], dim=1)
    dataset.append(d)
del blob10
print(f"chains kept {len(dataset):,}, dropped for alignment {dropped}; node dim {dataset[0].x.size(1)} = {N_PRE} + {TOPO_DIM} + {GEOM_DIM}")

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
assert len({d.cluster for d in dataset}) == len(dataset), "not 1 chain per cluster"
''')
md("## Cell 3 - Model and training machinery (Phase 11, unchanged)")
code(src(8))
md("## Cell 4 - Arms and the nested CV loop")
code(f'''
FOLDS_TO_RUN = {FOLDS}
''' + r'''
N_FOLDS, SEEDS, EPOCHS, PATIENCE, BATCH = 10, [0,1], 40, 8, 16
ARMS = ["no-topo", "full", "geom", "geom+topo"]
ORIG_X = [d.x.clone() for d in dataset]
TOPO_COLS = list(range(N_PRE, N_PRE + TOPO_DIM))
GEOM_COLS = list(range(N_PRE + TOPO_DIM, N_PRE + SIDE_DIM))
KEEP = {"no-topo": [], "full": TOPO_COLS, "geom": GEOM_COLS, "geom+topo": TOPO_COLS + GEOM_COLS}

def apply_arm(a):
    for d, x0 in zip(dataset, ORIG_X):
        x = x0.clone(); mask = torch.ones(x.size(1), dtype=torch.bool)
        mask[N_PRE:] = False; mask[KEEP[a]] = True
        x[:, ~mask] = 0.0
        d.x = x

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
            records.append({"arm": arm, "fold": fi, "seed": seed, "val_pr": best, "test_pr": float(te["pr"]),
                            "test_roc": float(te["roc"]), "n_test": int(len(te["y"])), "n_pos": int(te["y"].sum())})
        r = records[-1]
        print(f"  {arm:<10} fold {fi:<2} | val PR {r['val_pr']:.3f} | test PR {r['test_pr']:.3f} ROC {r['test_roc']:.3f} [{(time.time()-t0)/60:.0f}m]")
print(f"\n{len(records)} runs in {(time.time()-t0)/60:.1f} min")
json.dump({"records": records, "folds": FOLDS_TO_RUN}, open(f"/kaggle/working/phase230__TAG__records.json","w"), indent=2, default=float)
'''.replace("__TAG__", TAG))
md("## Cell 5 - Partial summary (final paired statistics are computed after merging both halves)")
code(r'''
import pandas as pd
df = pd.DataFrame(records)
print(df.groupby("arm")[["test_pr","test_roc"]].mean().round(4).sort_values("test_pr", ascending=False))
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
o = HERE / f"phase230{TAG}_geometry_control_main.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
