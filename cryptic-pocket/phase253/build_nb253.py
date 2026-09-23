#!/usr/bin/env python3
"""Phase 253: localization geometry control at the 7,513-chain scale-3 pool -- TRAINING half.
usage: build_nb253.py <a|b> <output_dir>   (a: no-topo + full ; b: geom + geom+topo ; 8 seeds each = 16 models/kernel)
Splices Phase 53's model and training-machinery cells verbatim (width-parameterised). Scoring is a separate kernel (Phase 254)
so a scoring bug cannot cost hours of training."""
import json, sys
from pathlib import Path

TAG = sys.argv[1]; OUT_DIR = Path(sys.argv[2]).resolve()
ARMS = {"a": ["no-topo", "full"], "b": ["geom", "geom+topo"]}[TAG]
HERE = Path(__file__).resolve().parent
nb53 = json.load(open(HERE.parents[1] / "phase53_scaled3_coach420.ipynb"))
s53 = lambda i: "".join(nb53["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase 253{TAG} — Localization geometry control at the 7,513-chain pool: training ({", ".join(ARMS)})

Topology's localization gain is largest on this pool (Phase 86: pooled COACH420 DCC 0.5932 with topology vs 0.4286 without,
8 seeds), and it is the paper's headline localization result. Phases 236-247 (2,055-chain pool) found geometry matches or
beats topology alone at every embedding scale and the residual beyond geometry is not established for localization. This
asks the same question with 3.5x the training data.

Training set = Phase 53's COACH420-clean 7,181 chains (Phase 251, identical for every arm and kernel), true centers from Phase 251,
92/8 train/val split (RandomState 42), Phase 53's model and training loop (lambda 0.1, 40 epochs, patience 8, batch 16, ESM2-35M).
Every arm uses an identical 51-wide side channel (34 topology + 17 geometry columns, unselected zeroed). This kernel trains
arms {ARMS} x 8 seeds and saves the checkpoints; the scorer (Phase 254) loads all 32 from the four arms.
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1")
md("## Cell 2 — Load the Phase 52 blob, the Phase 251 fixed training list + true centers, and the Phase 250 geometry (3 chunks)")
code(r'''
import gzip, json, os, time, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings("ignore")

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)
blob = torch.load(locate("phase52_graphs.pt"), map_location="cpu", weights_only=False)
N_PRE = int(blob["n_pre_topo"]); TOPO_DIM = int(blob["topo_dim"]); POS_WEIGHT = float(blob["pos_weight"])
by_key = {(str(d.pdb_id), str(d.chain_id)): d for d in blob["dataset"]}
del blob
assert N_PRE == 507 and TOPO_DIM == 34, (N_PRE, TOPO_DIM)

clean = torch.load(locate("phase251_clean_centers.pt"), map_location="cpu", weights_only=False)
feats = {}
for t in "abc":
    g = torch.load(locate(f"phase250{t}_geometry_scale3.pt"), map_location="cpu", weights_only=False)
    feats.update(g["feats"]); print(f"geometry chunk {t}: {g['n_chains']}/{g['n_total']} accepted")
GEOM_DIM = 17; SIDE_DIM = TOPO_DIM + GEOM_DIM

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")

dataset, miss_geo = [], 0
for k in clean["order_keys"]:                       # Phase 53's exact chain order
    if k not in feats: miss_geo += 1; continue
    d = by_key[k]; d.true_center = torch.from_numpy(clean["centers"][k]); dataset.append(d)
print(f"clean training chains {len(clean['order_keys']):,}; dropped for missing geometry {miss_geo}; using {len(dataset):,}")
assert miss_geo <= 0.02 * len(clean["order_keys"])

# label-free z-scoring statistics over the training chains
allg = np.concatenate([feats[(str(d.pdb_id), str(d.chain_id))] for d in dataset], 0)
g_mu = torch.from_numpy(allg.mean(0).astype(np.float32)); g_sd = torch.from_numpy(np.maximum(allg.std(0), 1e-6).astype(np.float32))
del allg
for d in dataset:
    f = feats[(str(d.pdb_id), str(d.chain_id))]; assert f.shape == (d.x.size(0), GEOM_DIM)
    d.x = torch.cat([d.x, ((torch.from_numpy(f) - g_mu) / g_sd).clamp(-5, 5)], dim=1)
del feats, by_key
torch.save(dict(g_mu=g_mu, g_sd=g_sd, n_train=len(dataset)), WORK / f"phase253__TAG___geom_stats.pt")

rng = np.random.RandomState(42)
perm = rng.permutation(len(dataset))
n_val = max(1, int(0.08 * len(perm)))
splits = {"val": [dataset[i] for i in perm[:n_val]], "train": [dataset[i] for i in perm[n_val:]]}
print(f"node dim {dataset[0].x.size(1)} = {N_PRE} + {TOPO_DIM} + {GEOM_DIM}; train {len(splits['train'])} / val {len(splits['val'])}")
'''.replace("__TAG__", TAG))
md("## Cell 3 — Model (Phase 53, verbatim)")
code(s53(11))
md("## Cell 4 — Training machinery (Phase 53, verbatim except the side-channel width)")
code(s53(15).replace("LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head)", "LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=use_loc_head)"))
md(f"## Cell 5 — Train arms {ARMS} x 8 seeds")
code(f'''
ARMS = {ARMS}
''' + r'''
KEEP_SIDE = {"no-topo": [], "full": list(range(TOPO_DIM)), "geom": list(range(TOPO_DIM, SIDE_DIM)), "geom+topo": list(range(SIDE_DIM))}
SIDE0 = [d.x[:, N_PRE:].clone() for d in dataset]          # original side channels only (the wide ESM block is never copied)
def apply_arm(a):
    m = torch.zeros(SIDE_DIM); m[KEEP_SIDE[a]] = 1.0
    for d, s0 in zip(dataset, SIDE0): d.x[:, N_PRE:] = s0 * m

BEST_LAMBDA = 0.1
train_log = []; t0 = time.time()
for arm in ARMS:
    apply_arm(arm)
    for seed in range(8):
        m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
        v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
        torch.save(m.state_dict(), WORK/f"cache/loc253_{arm.replace('+','_')}_seed{seed}.pt")
        train_log.append({"arm": arm, "seed": seed, "val_pr": float(val_pr), "val_dcc": float(v.get("dcc", float("nan")))})
        json.dump(train_log, open(WORK/"phase253__TAG___train_log.json", "w"), indent=2)
        print(f"  [{arm:9s}] seed {seed}: val PR {val_pr:.4f}  val DCC {v.get('dcc',float('nan')):.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)
print("done; checkpoints in cache/")
'''.replace("__TAG__", TAG))
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT_DIR.mkdir(parents=True, exist_ok=True)
o = OUT_DIR / f"phase253{TAG}_loc_train_scale3.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
meta = {"id": f"blackpearls1/phase253{TAG}-loc-train-scale3", "title": f"Phase253{TAG} Loc Train Scale3", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase52-esm-dihedral-scale3", "blackpearls1/phase251-clean-centers",
                           "blackpearls1/phase250a-geometry-scale3", "blackpearls1/phase250b-geometry-scale3",
                           "blackpearls1/phase250c-geometry-scale3"]}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
