#!/usr/bin/env python3
"""Phase 256 timing probe: how long does one epoch of the Phase 255 classification training take on Kaggle CPU?"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
nb = json.load(open(HERE.parent / "phase255" / "a" / "phase255a_cls_geometry_scale3.ipynb"))
cells = nb["cells"]
src = lambda i: "".join(cells[i]["source"])
loop = src(8)
head = loop[:loop.index("records = []")]           # arms / apply_arm / folds definitions, unchanged
C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})
md("# Phase 256 - CPU timing probe for the Phase 255 classification training (2 epochs, geom+topo arm, fold 0, seed 0)")
code(src(2)); code(src(4)); code(src(6))
code(r'''
import torch, time
print("threads:", torch.get_num_threads(), "| device:", DEVICE, "| cpus:", os.cpu_count())
''')
code(head + r'''
apply_arm("geom+topo")
tr_idx, te_idx = folds[0]
rng = np.random.RandomState(500); perm = rng.permutation(tr_idx); nv = max(1, int(0.15*len(perm)))
va_ds=[dataset[i] for i in perm[:nv]]; tr_ds=[dataset[i] for i in perm[nv:]]
print("train chains", len(tr_ds), "val chains", len(va_ds))
torch.manual_seed(0); np.random.seed(0)
m = Net(N_PRE, SIDE_DIM).to(DEVICE)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
tl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True); vl = DataLoader(va_ds, batch_size=BATCH)
for ep in range(2):
    t0 = time.time(); run_epoch(m, tl, opt); t1 = time.time(); v = run_epoch(m, vl); t2 = time.time()
    print(f"epoch {ep}: train {t1-t0:.0f}s  val {t2-t1:.0f}s  total {t2-t0:.0f}s  val PR {v['pr']:.3f}", flush=True)
''')
out = {"cells": C, "metadata": nb["metadata"], "nbformat": 4, "nbformat_minor": 5}
o = HERE / "phase256_cpu_timing.ipynb"; o.write_text(json.dumps(out, indent=1) + "\n"); print("wrote", o)
meta = {"id": "blackpearls1/phase256-cpu-timing", "title": "Phase256 CPU Timing", "code_file": o.name, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase52-esm-dihedral-scale3", "blackpearls1/phase250a-geometry-scale3",
                           "blackpearls1/phase250b-geometry-scale3", "blackpearls1/phase250c-geometry-scale3"]}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
