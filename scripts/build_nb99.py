#!/usr/bin/env python3
"""Phase 99: HOLO4K TRAIN-ONLY, splitting Phase94's combined train+score
kernel in two -- mirrors Phase97's fix for COACH420 (see that script's
docstring for the full RAM-ceiling diagnosis: the kernel has only 32.9GB
total RAM, already ~97% used right when training finishes, so a SECOND
live ESM2-3B load in the same process for benchmark featurization
reliably crashes regardless of what's freed first). This notebook is
Phase94's Cells 1-7 (setup through training) only. Saves a small JSON
with the exact training-pool size alongside the checkpoints so Phase100
(score-only) doesn't have to guess or hardcode it."""
import json, pickle
from pathlib import Path

C24 = pickle.load(open("/tmp/p24_cells.pkl", "rb"))
s24 = lambda i: (C24[i]["source"] if isinstance(C24[i]["source"], str) else "".join(C24[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 99 — HOLO4K ESM2-3B: TRAIN-ONLY

Split from Phase94 for the same reason Phase93->97/98 was split: the
kernel's 32.9GB RAM is already ~97% used right when training finishes,
regardless of what's freed afterward. Cells 1-7 only (identical to
Phase94, which succeeded through this point every single time -- only
the second ESM2-3B load afterward ever failed).
''')

md(r'''## Cell 1 — Setup (loads Phase 96's ESM2-3B blob, kept in fp16 -- CPU RAM fix: v3's diagnostic proved training itself needs meaningfully more working memory than inference on CPU (no separate VRAM pool to offload gradients/activations to), and the dataset's fp32 upcast alone consumed nearly the whole 32.9GB budget. Keeping the resident dataset at fp16 (~11.8GB vs ~23.6GB) frees ~11.8GB for training; batches are upcast to fp32 per-batch in the equivariance-check probe and in run_epoch below, since nn.Linear needs matching dtype)''')
code(s24(2))
code(s24(3).replace('locate("phase10_graphs.pt")', 'locate("phase96_graphs_esm3b.pt")')
           .replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")'))

code(r'''
def _mem(label):
    try:
        info = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            info[k] = int(v.strip().split()[0])
        total = info.get("MemTotal", 0)/1e6; avail = info.get("MemAvailable", 0)/1e6
        print(f"  [{label}] host RAM: {total-avail:.1f}GB used / {total:.1f}GB total ({avail:.1f}GB available)")
    except Exception as e:
        print(f"  [{label}] host RAM: error {e}")
_mem("after loading phase96 blob (CPU kernel -- checking if RAM budget differs from GPU kernels)")
''')

md(r'''## Cell 2 — Sample the SAME ~1,000 HOLO4K structures (seed=2024, identical to Phase 24/33/39/54/94)''')
code(s24(5))

md(r'''## Cell 3 — Clean the TRAINING side (identical logic; pool size 7513, unchanged from Phase54/94)''')
code(s24(7).replace("2056", "7513"))

md(r'''## Cell 4 — Recover true ligand centers (memory-hardened: periodic gc.collect() during the ~7k-chain loop, plus freeing `paths` before the final assembly -- Phase99 v2's diagnostic showed only ~2.5GB headroom after loading Phase96's blob, and the crash landed right after this loop finished, consistent with fragmentation across ~7k parse/free cycles eating that margin)''')
code(s24(9).replace('t0=time.time(); n_ok=0; n_geom_fail=0; n_no_ligand=0\nfor d in dataset:',
                     'import gc\nt0=time.time(); n_ok=0; n_geom_fail=0; n_no_ligand=0\nfor _loop_i, d in enumerate(dataset):\n'
                     '    if _loop_i % 1000 == 0 and _loop_i > 0: gc.collect()')
           .replace('print(f"\\ntrue centers recovered:',
                     'del paths; gc.collect()\n'
                     'print(f"\\ntrue centers recovered:'))

md(r'''## Cell 5 — Model (identical to every prior localization phase; N_PRE/TOPO_DIM read dynamically so this works unchanged at the new 2621-dim node width)''')
code(s24(11))

md(r'''## Cell 6 — Training machinery (identical to Phase 24/33/39/54/94/14)''')
code(s24(13).replace('    for b in loader:\n        b = b.to(DEVICE)',
                      '    for b in loader:\n        b = b.to(DEVICE); b.x = b.x.float()   # fp16-on-disk fix: upcast per-batch, not the whole resident dataset'))

md(r'''## Cell 7 — Train the clean ensemble (3 seeds) on the ESM2-3B blob''')
code(s24(15).replace("holo_clean_deploy_seed{seed}.pt", "holo_clean_deploy_esm3b_seed{seed}.pt"))

code(r'''
N_TRAIN_CHAINS = len(dataset)
json.dump({"n_train_chains": N_TRAIN_CHAINS}, open(WORK/"phase99_train_meta.json", "w"))
print(f"\nN_TRAIN_CHAINS = {N_TRAIN_CHAINS:,}")
print("checkpoints saved: holo_clean_deploy_esm3b_seed0.pt, seed1.pt, seed2.pt")
print("Phase99 (train-only) complete -- Phase100 (score-only, separate kernel) picks up from here.")
''')

md(r'''
---

Checkpoints `holo_clean_deploy_esm3b_seed{0,1,2}.pt` + `phase99_train_meta.json`
are the deliverable. Phase100 loads these fresh (new kernel, new 32.9GB
budget) along with Phase51's small topology blob -- Phase96's big blob is
never loaded there.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase99_esm3b_holo4k_train.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
