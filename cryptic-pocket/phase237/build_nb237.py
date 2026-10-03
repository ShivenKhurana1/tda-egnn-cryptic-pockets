#!/usr/bin/env python3
"""Phase 237: resolve the Phase 236 per-seed vs 8-seed-ensemble disagreement for geom+topo vs geom.
Reloads the 32 Phase 236 checkpoints (no retraining), rescoring on the identical benchmark, then:
seed-subset analysis by ensemble size and a two-way (seeds x structures) bootstrap."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
nb236 = json.load(open(HERE.parent / "phase236" / "phase236_localization_geometry_control.ipynb"))
c = nb236["cells"]
s236 = lambda i: "".join(c[i]["source"])
def find(prefix):
    for i, cell in enumerate(c):
        if cell["cell_type"] == "code" and "".join(cell["source"]).lstrip().startswith(prefix): return i
    raise KeyError(prefix)

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r'''
# Phase 237 - Per-seed vs ensemble: does geometry + topology beat geometry for localization?

Phase 236's pre-specified primary comparison (seed-paired pooled DCC@4A, geometry+topology vs geometry) was null
(+0.0051, 3/8 seeds, p = 0.62 / 0.92), but the 8-seed *ensemble* comparison was +0.0254 (structure-bootstrap CI
+0.004 to +0.047). One ensemble per arm cannot separate a real ensemble-level benefit from seed luck, and the structure
bootstrap ignores seed variance entirely. This notebook reloads the 32 saved checkpoints (no retraining), rescoring the
identical 1,149 benchmark structures, and asks:

1. How does the geometry+topology - geometry difference change with ensemble size k (all C(8,k) seed subsets, paired)?
2. What is the 8-seed-ensemble difference when BOTH seeds and structures are resampled (two-way bootstrap)?
''')
md("## Cell 1 - Install")
code("!pip install -q torch-geometric==2.6.1 biopython==1.84")
md("## Cell 2 - Setup (Phase 236 cell, unchanged)")
code(s236(4))
md("## Cell 3 - Model class (Phase 22/236, unchanged) and arm definitions")
code(s236(8) + r'''

import time
ARMS = ["no-topo", "full", "geom", "geom+topo"]
TOPO_COLS = list(range(N_PRE, N_PRE + TOPO_DIM))
GEOM_COLS = list(range(N_PRE + TOPO_DIM, N_PRE + SIDE_DIM))
KEEP = {"no-topo": [], "full": TOPO_COLS, "geom": GEOM_COLS, "geom+topo": TOPO_COLS + GEOM_COLS}
''')
md("## Cell 4 - Load the 32 Phase 236 checkpoints")
code(r'''
models = {a: {} for a in ARMS}
for arm in ARMS:
    for seed in range(8):
        fn = f"loc236_{arm.replace('+','_')}_seed{seed}.pt"
        state = torch.load(locate(fn), map_location=DEVICE)
        m = LocalizingEGNN(N_PRE, SIDE_DIM).to(DEVICE); m.load_state_dict(state); m.eval()
        models[arm][seed] = m
print({a: len(v) for a, v in models.items()}, "checkpoints loaded")
''')
md("## Cell 5 - Benchmark features + geometry, and scoring (Phase 236 cells, unchanged)")
code(next("".join(x["source"]) for x in c if x["cell_type"] == "code" and "phase15_raw_features" in "".join(x["source"])))
code(s236(find("from collections import defaultdict")))
md("## Cell 6 - Sanity: reproduce Phase 236's per-seed and ensemble numbers")
code(r'''
per_seed = {a: np.array([rate(hits_for(a, (s,))) for s in range(8)]) for a in ARMS}
ens = {a: rate(hits_for(a, tuple(range(8)))) for a in ARMS}
print({a: round(float(per_seed[a].mean()), 4) for a in ARMS})
print({a: round(ens[a], 4) for a in ARMS})
# Phase 236 reported per-seed means no-topo .4441 full .4990 geom .5296 geom+topo .5347; ensembles .507 .5333 .5552 .5806
assert abs(per_seed["geom"].mean() - 0.5296) < 0.002 and abs(ens["geom+topo"] - 0.5806) < 0.002, "does not reproduce Phase 236"
print("reproduces Phase 236")
''')
md("## Cell 7 - Ensemble-size analysis: all seed subsets of each size, paired across arms")
code(r'''
from itertools import combinations
import pandas as pd
COMPS = [("geom+topo", "geom"), ("full", "no-topo"), ("geom", "full"), ("geom", "no-topo")]
rows = []
for a, b in COMPS:
    for k in (1, 2, 3, 4, 6, 8):
        ds = np.array([rate(hits_for(a, S)) - rate(hits_for(b, S)) for S in combinations(range(8), k)])
        rows.append(dict(a=a, b=b, k=k, n_subsets=len(ds), mean_delta=float(ds.mean()), sd=float(ds.std()), frac_positive=float((ds > 0).mean())))
df_k = pd.DataFrame(rows); df_k.to_csv("/kaggle/working/phase237_ensemble_size.csv", index=False)
print(df_k.round(4).to_string(index=False))
# two disjoint 4-seed ensembles per arm (35 ways to split 8 seeds into two halves)
print("\nDisjoint 4+4 splits, geom+topo - geom (each half is an independent 4-model ensemble):")
halves = []
for S in combinations(range(8), 4):
    if 0 not in S: continue
    T = tuple(x for x in range(8) if x not in S)
    for H in (S, T): halves.append(rate(hits_for("geom+topo", H)) - rate(hits_for("geom", H)))
halves = np.array(halves)
print(f"  {len(halves)} half-ensemble deltas: mean {halves.mean():+.4f}  sd {halves.std():.4f}  positive {int((halves>0).sum())}/{len(halves)}")
''')
md("## Cell 8 - Two-way bootstrap (seeds x structures) of the 8-seed-ensemble difference")
code(r'''
rng = np.random.default_rng(237)
B = 1000
res = {}
for a, b in COMPS:
    deltas = []
    for _ in range(B):
        seeds = tuple(int(x) for x in rng.integers(0, 8, size=8))          # resample the 8 seeds with replacement
        ha, hb = hits_for(a, seeds), hits_for(b, seeds)
        keys = sorted(set(ha) & set(hb)); va = np.array([ha[k] for k in keys]); vb = np.array([hb[k] for k in keys])
        idx = rng.integers(0, len(keys), size=len(keys))                     # resample structures
        deltas.append(va[idx].mean() - vb[idx].mean())
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    res[f"{a} - {b}"] = dict(mean=float(deltas.mean()), lo=float(lo), hi=float(hi), frac_pos=float((deltas > 0).mean()))
    print(f"{a+' - '+b:<22} two-way bootstrap ensemble delta {deltas.mean():+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  P(delta>0) = {(deltas>0).mean():.3f}")
json.dump(res, open("/kaggle/working/phase237_twoway_bootstrap.json", "w"), indent=2)
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
o = HERE / "phase237_ensemble_vs_seed_analysis.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
