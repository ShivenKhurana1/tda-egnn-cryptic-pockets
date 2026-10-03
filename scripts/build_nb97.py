#!/usr/bin/env python3
"""Phase 97: COACH420 TRAIN-ONLY, splitting Phase93's combined train+score
kernel in two. Root cause found via Phase93 v6's memory diagnostics: the
kernel has only 32.9GB total RAM, and it's ALREADY at 31.9GB used / 1.0GB
available right after training finishes -- BEFORE any cleanup, and
cleanup barely moves the needle (31.8GB after freeing the 23.6GB
dataset+splits+blob10). This isn't a "forgot to free something" bug, it's
that training a 3-seed ensemble on the 7,513-chain/2621-dim pool already
uses ~all available RAM, leaving no room to ALSO load a fresh 3B-param
ESM model afterward in the SAME process, no matter how efficiently that
second load is done (v6 already used low_cpu_mem_usage). Fix: run
training and evaluation as separate Kaggle kernels, each getting its own
fresh 32.9GB budget. This notebook is the train-only half -- identical to
Phase93's Cells 1-8 (setup through training), nothing else. Never loads
ESM2-3B live (training uses the pre-baked embeddings already in Phase96's
blob)."""
import json, pickle
from pathlib import Path

C20 = pickle.load(open("/tmp/p20_cells.pkl", "rb"))
s20 = lambda i: (C20[i]["source"] if isinstance(C20[i]["source"], str) else "".join(C20[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 97 - COACH420 ESM2-3B: TRAIN-ONLY

Split from Phase93 after 6 straight failures all crashing ~30s into
loading a SECOND ESM2-3B model post-training. Diagnostics (Phase93 v6)
proved the kernel's 32.9GB RAM is already ~97% used right when training
finishes, regardless of what's freed afterward -- training itself is the
whole budget. This notebook does ONLY setup + clean-training-set
construction + training (identical to Phase93's Cells 1-8, which have
succeeded 6/6 times so far -- only what came AFTER ever failed). Saves
checkpoints for Phase98 (score-only, fresh kernel) to pick up.
''')

md(r'''## Cell 1 - Setup (loads Phase 96's ESM2-3B blob)''')
code(s20(2))
code(s20(3).replace('locate("phase10_graphs.pt")', 'locate("phase96_graphs_esm3b.pt")')
           .replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")')
           .replace('dataset = blob10["dataset"]',
                     'dataset = blob10["dataset"]\n'
                     'for _d in dataset: _d.x = _d.x.float()   # Phase96 stores fp16 on disk (quota fix); upcast once here'))

md(r'''## Cell 2 - Fetch ALL 420 COACH420 structures (needed to compute training-side homology exclusion; identical to Phase 20/31/38/53/93)''')
code(s20(5))

md(r'''## Cell 3 - Clean the TRAINING side: exclude any chain homologous to COACH420 (identical to Phase 20 v2/31/38/53/93)''')
code(s20(7))

md(r'''## Cell 4 - Recover true ligand centers for the cleaned training set (identical to Phase 20/31/38/53/93/14)''')
code(s20(9))

md(r'''## Cell 5 - Model (identical to every prior localization phase; N_PRE/TOPO_DIM read dynamically so this works unchanged at the new 2621-dim node width)''')
code(s20(11))

md(r'''## Cell 6 - Equivariance check before training anything (identical to Phase 20/31/38/53/93)''')
code(s20(13))

md(r'''## Cell 7 - Training machinery (identical to Phase 14/20/31/38/53/93)''')
code(s20(15))

md(r'''## Cell 8 - Train the clean-training ensemble (3 seeds, lambda=0.1) on the ESM2-3B blob''')
code(s20(17).replace("clean_deploy_seed{seed}.pt", "clean_deploy_esm3b_seed{seed}.pt"))

code(r'''
print(f"\nN_TRAIN_CHAINS = {len(dataset):,}")
print("checkpoints saved: clean_deploy_esm3b_seed0.pt, seed1.pt, seed2.pt")
print("Phase97 (train-only) complete -- Phase98 (score-only, separate kernel) picks up from here.")
''')

md(r'''
---

Checkpoints `clean_deploy_esm3b_seed{0,1,2}.pt` are the deliverable.
Phase98 loads these fresh (new kernel, new 32.9GB budget) along with
Phase51's small topology blob (for `topo_mean`/`topo_std`/`topo_names`)
and Phase96's blob is NEVER loaded there -- scoring doesn't need the
7,513-chain training pool at all, only the trained weights.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase97_esm3b_coach420_train.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
