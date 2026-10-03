#!/usr/bin/env python3
"""Phase 230's geometry control, generalized to an arbitrary ESM embedding scale.
Same 2,056-chain pool, same folds, same arms/model/training -- only the blob (and hence ESM_DIM) changes.

usage: build_nb_scale_control.py <650m|3b> <a|b> <phase_num> <output_dir>
  a = folds 0-4, b = folds 5-9 (two GPU kernels in parallel, as Phase 230 did)
"""
import json, sys
from pathlib import Path

SCALE, TAG, PHASE = sys.argv[1], sys.argv[2], sys.argv[3]
OUT_DIR = Path(sys.argv[4]).resolve()
FOLDS = {"a": [0, 1, 2, 3, 4], "b": [5, 6, 7, 8, 9]}[TAG]
ROOT = Path(__file__).resolve().parents[2]
CFG = {"650m": dict(blob="phase238_graphs_esm650m.pt", src="blackpearls1/phase238-esm650m-ingestion", label="ESM2-650M"),
       "3b": dict(blob="phase239_graphs_esm3b.pt", src="blackpearls1/phase239-esm3b-ingestion", label="ESM2-3B")}[SCALE]

nb11 = json.load(open(ROOT / "phase11_augmented_nested_cv.ipynb"))
src11 = lambda i: "".join(nb11["cells"][i]["source"])
nb230 = json.load(open(ROOT / "cryptic-pocket" / "phase230a" / "phase230a_geometry_control_main.ipynb"))
src230 = lambda i: "".join(nb230["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase {PHASE}{TAG} - Geometry control at {CFG["label"]} (folds {FOLDS[0]}-{FOLDS[-1]})

Phase 230 ran the geometry-vs-topology control at ESM2-35M on the 2,055-chain pool (main result: geometry beats
topology alone +0.0129 PR-AUC 10/10 folds; topology adds a small residual over geometry, +0.0097 PR-AUC 9/10
folds, p=0.017/0.037). This repeats it at {CFG["label"]} -- same 2,056-chain pool, same 10-fold split (seed
1337), same model/loss/arms; only the ESM embedding (and hence node width) differs. Open question from Future
Work: does the small residual topology effect grow, shrink, or vanish with embedding scale?

Arms: `no-topo`, `full` (34 TDA), `geom` (17), `geom+topo` (51). **Primary comparison (fixed before any result):
`geom+topo` vs `geom` on test PR-AUC, paired over the 10 folds** (this notebook runs half; halves are merged
locally, same script as Phase 230).
''')
md("## Cell 1 - Setup")
code("!pip install -q torch-geometric==2.6.1 biopython==1.84")
md(f"## Cell 2 - Load the {CFG['label']} graphs ({CFG['blob']}) and the Phase 229 geometry")
code(src230(4).replace('locate("phase10_graphs.pt")', f'locate("{CFG["blob"]}")'))
md("## Cell 3 - Model and training machinery (Phase 11, unchanged)")
code(src11(8))
md("## Cell 4 - Arms and the nested CV loop")
code(src230(8).replace('FOLDS_TO_RUN = [0, 1, 2, 3, 4]', f'FOLDS_TO_RUN = {FOLDS}')
              .replace('phase230arecords.json', f'phase{PHASE}{TAG}_records.json'))
md("## Cell 5 - Partial summary")
code(src230(10))

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT_DIR.mkdir(parents=True, exist_ok=True)
o = OUT_DIR / f"phase{PHASE}{TAG}_geometry_control_{SCALE}.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")

meta = {
  "id": f"blackpearls1/phase{PHASE}{TAG}-geometry-control-{SCALE}",
  "title": f"Phase{PHASE}{TAG} Geometry Control {SCALE.upper()}",
  "code_file": o.name,
  "language": "python", "kernel_type": "notebook", "is_private": True,
  "enable_gpu": True, "machine_shape": "NvidiaTeslaT4", "enable_tpu": False, "enable_internet": True,
  "dataset_sources": [], "competition_sources": [],
  "kernel_sources": [CFG["src"], "blackpearls1/phase229-geometry-main-pool"],
}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {OUT_DIR/'kernel-metadata.json'}")
