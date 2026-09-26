#!/usr/bin/env python3
"""Phase 262: CryptoBench, ESM2-3B, official split -- do the 17 plain-geometry features improve the strongest hidden-pocket setup?
One kernel per arm (none / full=topology / geom / geom+topo), 8 seeds each, trained FROM SCRATCH (transfer is a separate step).
Saves every model's per-residue probabilities and predicted pocket centers on val and test, so ensembles and thresholds are analysed offline.
usage: build_nb262.py <none|full|geom|geom+topo> <output_root>"""
import json, sys
from pathlib import Path

ARM = sys.argv[1]; ROOT_OUT = Path(sys.argv[2]).resolve()
SLUG = {"none": "no", "full": "fu", "geom": "ge", "geom+topo": "gt"}[ARM]
HERE = Path(__file__).resolve().parent
nb129 = json.load(open(HERE.parents[1] / "phase129_cryptobench_esm3b_transfer.ipynb"))
s129 = lambda i: "".join(nb129["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase 262 {SLUG} — CryptoBench + ESM2-3B + plain geometry: arm `{ARM}` (8 seeds, from scratch)

The strongest published-style hidden-pocket setup in this project (ESM2-3B on CryptoBench's official split; Phase 55/129) predates the
finding that 17 plain geometry features beat the topology block. This tests whether geometry improves it. Same blob, split, model, loss
(BCE + 0.1 x Huber pocket-center), optimiser and early stopping as Phase 129; every arm has an identical 51-wide side channel (34 topology +
17 geometry, unselected columns zeroed). Selection uses validation only; test predictions are saved, not used for any choice.
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1")
md("## Cell 2 — Setup: Phase 48's CryptoBench-3B blob + Phase 227's geometry")
code(s129(4) + r'''

geo = torch.load(locate("phase227_geometry_features.pt"), map_location="cpu", weights_only=False)
GEOM_DIM = len(geo["names"]); assert GEOM_DIM == 17
SIDE_DIM = TOPO_DIM + GEOM_DIM
_tr = np.concatenate([geo["feats"][(str(d.pdb_id).lower(), str(d.chain_id))] for d in dataset if d.split == "train"])   # official-train statistics only
g_mu = torch.from_numpy(_tr.mean(0).astype(np.float32)); g_sd = torch.from_numpy((_tr.std(0) + 1e-6).astype(np.float32))
missing = 0
for d in dataset:
    k = (str(d.pdb_id).lower(), str(d.chain_id))
    f = geo["feats"].get(k)
    if f is None or f.shape[0] != d.x.size(0): missing += 1; continue
    d.x = torch.cat([d.x, ((torch.from_numpy(f) - g_mu) / g_sd).clamp(-5, 5)], dim=1)
assert missing == 0, f"{missing} chains could not be matched to geometry"
splits = {s: [d for d in dataset if d.split == s] for s in ("train", "val", "test")}
print(f"node dim {dataset[0].x.size(1)} = {N_PRE} + {TOPO_DIM} + {GEOM_DIM}; splits", {k: len(v) for k, v in splits.items()})
assert dataset[0].x.size(1) == N_PRE + SIDE_DIM
''')
md("## Cell 3 — Model (Phase 129, verbatim)")
code(s129(6))
md("## Cell 4 — Training machinery (Phase 129, width-parameterised, no pretrained weights)")
code(s129(8).replace("LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head)", "LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=use_loc_head)"))
md(f"## Cell 5 — Train arm `{ARM}` x 8 seeds; save per-residue probabilities and predicted centers for val and test")
code(f'''
ARM = "{ARM}"
''' + r'''
KEEP_SIDE = {"none": [], "full": list(range(TOPO_DIM)), "geom": list(range(TOPO_DIM, SIDE_DIM)), "geom+topo": list(range(SIDE_DIM))}
m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE[ARM]] = 1.0
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_

@torch.no_grad()
def predict(model, split):
    out = {}
    for d in splits[split]:
        b = d.clone().to(DEVICE); logits, pc = model(b)
        out[(str(d.pdb_id).lower(), str(d.chain_id))] = dict(p=torch.sigmoid(logits).cpu().numpy().astype(np.float32),
                                                             c=pc.cpu().numpy()[0], y=d.y.numpy().astype(np.int8),
                                                             tc=d.true_center.numpy())
    return out

BEST_LAMBDA = 0.1
PREDS = {"arm": ARM, "seeds": {}}; t0 = time.time()
for seed in range(8):
    model, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
    PREDS["seeds"][seed] = {"val": predict(model, "val"), "test": predict(model, "test"), "val_pr": float(val_pr)}
    torch.save(PREDS, WORK / f"phase262_{ARM.replace('+','_')}_preds.pt")          # incremental: a late crash loses one seed, not all
    print(f"  [{ARM}] seed {seed}: val PR {val_pr:.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)
print("done")
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / f"phase262{SLUG}_cb3b_geometry.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = {"id": f"blackpearls1/phase262-{SLUG}-cb3b-geometry", "title": f"Phase262 {SLUG} Cb3b Geometry", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase48-cryptobench-esm3b", "blackpearls1/phase227-geometry-ingestion"]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
