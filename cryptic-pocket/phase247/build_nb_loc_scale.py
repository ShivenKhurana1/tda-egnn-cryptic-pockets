#!/usr/bin/env python3
"""Phase 236's localization geometry control, generalized to an arbitrary main-pipeline embedding scale.
Trains on the scale's own graph blob (Phase 238/239), scores on the scale's own re-embedded benchmark cache
(Phase 244/245). Same 4 arms x 8 seeds, same true-center recovery, same scoring code as Phase 236.

usage: build_nb_loc_scale.py <650m|3b> <phase_num> <output_dir>
"""
import json, sys
from pathlib import Path

SCALE, PHASE = sys.argv[1], sys.argv[2]
OUT_DIR = Path(sys.argv[3]).resolve()
ROOT = Path(__file__).resolve().parents[2]
CFG = {
    "650m": dict(train_blob="phase238_graphs_esm650m.pt", bench_cache="phase244_raw_features_esm650m.pt",
                train_src="blackpearls1/phase238-esm650m-ingestion", bench_src="blackpearls1/phase244-bench-esm650m",
                label="ESM2-650M"),
    "3b": dict(train_blob="phase239_graphs_esm3b.pt", bench_cache="phase245_raw_features_esm3b.pt",
              train_src="blackpearls1/phase239-esm3b-ingestion", bench_src="blackpearls1/phase245-bench-esm3b",
              label="ESM2-3B"),
}[SCALE]

nb236 = json.load(open(ROOT / "cryptic-pocket" / "phase236" / "phase236_localization_geometry_control.ipynb"))
c236 = nb236["cells"]
s236 = lambda i: "".join(c236[i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase {PHASE} — Localization geometry control at {CFG["label"]}

Phase 236 (35M): topology's classification-side residual over geometry (+0.0097 PR-AUC) was confirmed only on
PR-AUC; the analogous localization residual (geometry+topology vs geometry, pooled DCC@4A) was NULL per seed
(+0.0051, 3/8 seeds) but trended positive with ensemble size (Phase 237). Phases 238-241 then found the
CLASSIFICATION residual becomes clearly confirmed (both PR-AUC and ROC-AUC) at 650M and 3B. This asks whether
the LOCALIZATION residual does the same, using {CFG["label"]} training data (Phase {"238" if SCALE=="650m" else "239"})
and a losslessly re-embedded {CFG["label"]} benchmark cache (Phase {"244" if SCALE=="650m" else "245"}, decoded
from Phase 15's one-hot block, no re-fetch).

Protocol identical to Phase 236: training pool = the scale's own 2,055-chain blob, 92/8 split (RandomState 42),
`LocalizingEGNN` (lambda 0.1, 40 epochs, patience 8, batch 16), 4 arms x 8 seeds = 32 trainings, scored on the
1,149-unit COACH420/HOLO4K benchmark, pooled DCC@4A. **Primary comparison (fixed before any result): `geom+topo`
vs `geom`, seed-paired pooled DCC.**
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1 biopython==1.84")
md(f"## Cell 2 — Load the {CFG['label']} training graphs and geometry (Phase 229, scale-independent)")
code(s236(4).replace('locate("phase10_graphs.pt")', f'locate("{CFG["train_blob"]}")'))
md("## Cell 3 — Recover true ligand centers for the training set and split 92/8 (Phase 22, verbatim)")
code(s236(6))
md("## Cell 4 — Model (Phase 22, verbatim)")
code(s236(8))
md("## Cell 5 — Training machinery (Phase 22, verbatim except side-channel width)")
code(s236(10))
md("## Cell 6 — Arms; train 4 arms x 8 seeds")
code(s236(12).replace("loc236_{arm.replace('+','_')}_seed{seed}", f"loc{PHASE}_{{arm.replace('+','_')}}_seed{{seed}}"))
md(f"## Cell 7 — Load the re-embedded {CFG['label']} benchmark cache (Phase {'244' if SCALE=='650m' else '245'}) and its geometry (Phase 235, scale-independent)")
code(f'''
bench = torch.load(locate("{CFG["bench_cache"]}"), map_location="cpu", weights_only=False)
raw_cache = bench["cache"]
geo_bm = torch.load(locate("phase235_benchmark_geometry.pt"), map_location="cpu", weights_only=False)
blob6b_bench = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)
mu_b, sd_b = blob6b_bench["topo_mean"], blob6b_bench["topo_std"]
del blob6b_bench
valid_keys = {{k for k in raw_cache if k in geo_bm["feats"]}}
print(f"benchmark chains: {{len(raw_cache):,}} cached ({CFG['label']}), {{len(valid_keys):,}} with geometry (dropped {{len(raw_cache)-len(valid_keys)}} for every arm)")
assert bench["n_pre"] == N_PRE, f"benchmark N_PRE {{bench['n_pre']}} != training N_PRE {{N_PRE}}"
''')
md("## Cell 8 — Score all arms and seeds (Phase 22/236 scoring, arm masks applied to the benchmark features)")
code(s236(16))
md("## Cell 9 — Per-seed pooled DCC, seed-paired statistics (primary comparison first), structure-level bootstrap on the 8-seed ensembles")
code(s236(18).replace('phase236_paired_stats.csv', f'phase{PHASE}_paired_stats.csv')
             .replace('phase236_summary.json', f'phase{PHASE}_summary.json')
             .replace('phase236_train_log.json', f'phase{PHASE}_train_log.json'))

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT_DIR.mkdir(parents=True, exist_ok=True)
o = OUT_DIR / f"phase{PHASE}_localization_geometry_control_{SCALE}.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")

meta = {
  "id": f"blackpearls1/phase{PHASE}-localization-geometry-{SCALE}",
  "title": f"Phase{PHASE} Localization Geometry {SCALE.upper()}",
  "code_file": o.name,
  "language": "python", "kernel_type": "notebook", "is_private": True,
  "enable_gpu": True, "machine_shape": "NvidiaTeslaT4", "enable_tpu": False, "enable_internet": True,
  "dataset_sources": [], "competition_sources": [],
  "kernel_sources": [CFG["train_src"], CFG["bench_src"], "blackpearls1/phase229-geometry-main-pool",
                     "blackpearls1/phase235-benchmark-geometry", "blackpearls1/tda-egnn-phase6b-topology-scale"],
}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {OUT_DIR/'kernel-metadata.json'}")
