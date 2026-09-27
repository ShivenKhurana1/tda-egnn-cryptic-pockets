#!/usr/bin/env python3
"""Phase 271: main-pipeline ESM2-3B pretraining (homology-clean pool of Phase 266a) with the Phase 270 protein-movement block.
Arms: full (topology only, movement columns zeroed = matched control) and full+flex (topology + 10 movement features); identical 44-wide side channel.
usage: build_nb271.py <full|full+flex> <output_root>
Same recipe as Phase 97 (3B, COACH420-clean 7,178 chains from Phase 251, 92/8 split RandomState 42, lambda 0.1, 40 epochs, patience 8, batch 16)
but every arm has an identical 51-wide side channel (34 topology + 17 geometry, unselected columns zeroed), so checkpoints of every arm
share one architecture and can all seed the CryptoBench fine-tune (Phase 264). 8 seeds per arm; node features stay fp16 in RAM and are
cast to fp32 per batch (halves the ~19 GB the fp32 upcast needed)."""
import json, sys
from pathlib import Path

ARM = sys.argv[1]; ROOT_OUT = Path(sys.argv[2]).resolve(); SLUG = {"full": "271fu", "full+flex": "271ff"}[ARM]
HERE = Path(__file__).resolve().parent
nb97 = json.load(open(HERE.parents[1] / "phase97_esm3b_coach420_train.ipynb"))
s97 = lambda i: "".join(nb97["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase 271 — ESM2-3B pretraining, homology-clean pool, protein-movement block: arm `{ARM}` (8 seeds)

Seeds the movement-feature CryptoBench transfer (Phase 272). Same pool, split and recipe as Phase 266b, plus 10 elastic-network movement features (Phase 270)
replacing the geometry block. Both arms share the 44-wide side channel; `full` zeroes the movement columns.
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1")
md("## Cell 2 — Load the Phase 96 3B blob, Phase 251 clean list + centres, Phase 250 geometry")
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
blob = torch.load(locate("phase96_graphs_esm3b.pt"), map_location="cpu", weights_only=False)
N_PRE = int(blob["n_pre_topo"]); TOPO_DIM = int(blob["topo_dim"]); POS_WEIGHT = float(blob["pos_weight"])
by_key = {(str(d.pdb_id), str(d.chain_id)): d for d in blob["dataset"]}
del blob
print("N_PRE", N_PRE, "TOPO_DIM", TOPO_DIM, "chains", len(by_key), "x dtype", next(iter(by_key.values())).x.dtype)
assert N_PRE == 2587 and TOPO_DIM == 34, (N_PRE, TOPO_DIM)

clean = torch.load(locate("phase251_clean_centers.pt"), map_location="cpu", weights_only=False)
_ex = json.load(open(locate("phase266a_pool_exclusions.json"))); _drop = {(a, b) for a, b in _ex["drop"]}
_n0 = len(clean["order_keys"]); clean["order_keys"] = [k for k in clean["order_keys"] if (str(k[0]), str(k[1])) not in _drop]
print(f"Phase 266a exclusions: dropped {_n0 - len(clean['order_keys'])} of {_n0} pool chains; {len(clean['order_keys'])} remain")
assert 0 < _n0 - len(clean["order_keys"]) < 0.6 * _n0
_fx = torch.load(locate("phase270_flex_features.pt"), map_location="cpu", weights_only=False)
feats = _fx["pool"]; GEOM_DIM = len(_fx["names"]); assert GEOM_DIM == 10, GEOM_DIM; print("movement features:", _fx["names"]); del _fx
SIDE_DIM = TOPO_DIM + GEOM_DIM

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")

dataset, miss_geo, miss_blob = [], 0, 0
for k in clean["order_keys"]:
    if k not in by_key: miss_blob += 1; continue
    if k not in feats: miss_geo += 1; continue
    d = by_key[k]; d.true_center = torch.from_numpy(clean["centers"][k]); dataset.append(d)
print(f"clean training chains {len(clean['order_keys']):,}; missing from 3B blob {miss_blob}; missing geometry {miss_geo}; using {len(dataset):,}")
assert miss_blob == 0 and miss_geo <= 0.02 * len(clean["order_keys"])

allg = np.concatenate([feats[(str(d.pdb_id), str(d.chain_id))] for d in dataset], 0)
g_mu = torch.from_numpy(allg.mean(0).astype(np.float32)); g_sd = torch.from_numpy(np.maximum(allg.std(0), 1e-6).astype(np.float32))
del allg
for d in dataset:
    f = feats[(str(d.pdb_id), str(d.chain_id))]; assert f.shape == (d.x.size(0), GEOM_DIM)
    d.x = torch.cat([d.x.half(), ((torch.from_numpy(f) - g_mu) / g_sd).clamp(-5, 5).half()], dim=1)       # stays fp16 in RAM
del feats, by_key
torch.save(dict(g_mu=g_mu, g_sd=g_sd, n_train=len(dataset)), WORK / "phase271_flex_stats.pt")

rng = np.random.RandomState(42)
perm = rng.permutation(len(dataset))
n_val = max(1, int(0.08 * len(perm)))
splits = {"val": [dataset[i] for i in perm[:n_val]], "train": [dataset[i] for i in perm[n_val:]]}
print(f"node dim {dataset[0].x.size(1)} = {N_PRE} + {TOPO_DIM} + {GEOM_DIM}; train {len(splits['train'])} / val {len(splits['val'])}")
assert dataset[0].x.size(1) == N_PRE + SIDE_DIM
''')
md("## Cell 3 — Model (Phase 97, verbatim)")
code(s97(11))
md("## Cell 4 — Training machinery (Phase 97, side-channel width parameterised, fp16 -> fp32 cast per batch)")
t = s97(15).replace("LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head)", "LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=use_loc_head)")
assert t.count("b = b.to(DEVICE)") == 1
code(t.replace("b = b.to(DEVICE)", "b = b.to(DEVICE); b.x = b.x.float()"))
md(f"## Cell 5 — Train arm `{ARM}` x 8 seeds")
code(f'''
ARM = "{ARM}"
''' + r'''
KEEP_SIDE = {"full": list(range(TOPO_DIM)), "full+flex": list(range(SIDE_DIM))}
m_ = torch.zeros(SIDE_DIM, dtype=torch.float16); m_[KEEP_SIDE[ARM]] = 1.0
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_

BEST_LAMBDA = 0.1
train_log = []; t0 = time.time()
for seed in range(8):
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    torch.save(m.state_dict(), WORK / f"cache/pretrain271_{ARM.replace('+','_')}_seed{seed}.pt")
    train_log.append({"arm": ARM, "seed": seed, "val_pr": float(val_pr), "val_dcc": float(v.get("dcc", float("nan")))})
    json.dump(train_log, open(WORK / "phase271_train_log.json", "w"), indent=2)
    print(f"  [{ARM}] seed {seed}: val PR {val_pr:.4f}  val DCC {v.get('dcc', float('nan')):.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)
print("done; checkpoints in cache/")
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / f"phase{SLUG}_pretrain_3b_flex.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = {"id": f"blackpearls1/phase{SLUG}-pretrain-3b-flex", "title": f"Phase{SLUG} Pretrain 3b Flex", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase96-esm-dihedral-esm3b-v3", "blackpearls1/phase251-clean-centers",
                           "blackpearls1/phase266a-pool-exclusions", "blackpearls1/phase270-flex-features"]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
