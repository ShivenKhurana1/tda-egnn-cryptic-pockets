#!/usr/bin/env python3
"""Phase 101: COACH420 ESM2-3B 8-seed extension (train-only, mirrors
Phase97 exactly but trains seeds 3-7 instead of 0-2). Extends Phase98's
exciting 3-seed result (DCC 0.6053, already beating the 8-seed 35M
baseline of 0.5932) to the full 8-seed standard this project uses for
every major finding, following the established discipline (Phase21/32
did the same 5-more-seeds extension for earlier pools). CPU (GPU quota
still exhausted) -- Phase97 already proved this exact pipeline works
reliably; CPU just makes it slower, not incapable (unlike Phase99's
HOLO4K-specific RAM wall)."""
import json, pickle
from pathlib import Path

C20 = pickle.load(open("/tmp/p20_cells.pkl", "rb"))
s20 = lambda i: (C20[i]["source"] if isinstance(C20[i]["source"], str) else "".join(C20[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 101 — COACH420 ESM2-3B: 8-Seed Extension (seeds 3-7)

Phase98's 3-seed result (DCC 0.6053) already beats the 8-seed 35M
baseline (0.5932) and this-run P2Rank by +0.0844 -- the strongest single
lever gain in the project since the original data-scale-up (Phase31/32).
This trains 5 more seeds (3-7) on the identical clean training set
Phase97 used, so the full 8-seed ensemble can be rescored (Phase98-style)
against the same benchmark for a properly powered final number.
''')

md(r'''## Cell 1 — Setup (loads Phase 96's ESM2-3B blob, kept in fp16 -- CPU RAM fix: training itself needs meaningfully more working memory than inference on CPU (no separate VRAM pool to offload gradients/activations to), and eagerly upcasting the whole dataset to fp32 alone consumed nearly the entire 32.9GB budget, leaving no room for training. Keeping the resident dataset at fp16 (~11.8GB vs ~23.6GB) frees ~11.8GB; batches are upcast to fp32 per-batch below instead)''')
code(s20(2))
code(s20(3).replace('locate("phase10_graphs.pt")', 'locate("phase96_graphs_esm3b.pt")')
           .replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")'))

md(r'''## Cell 2 — Fetch ALL 420 COACH420 structures (identical to Phase 20/31/38/53/93/97 -- reproduces the same clean training set)''')
code(s20(5))

md(r'''## Cell 3 — Clean the TRAINING side: exclude any chain homologous to COACH420 (identical to Phase 20 v2/31/38/53/93/97)''')
code(s20(7))

md(r'''## Cell 4 — Recover true ligand centers for the cleaned training set (identical to Phase 20/31/38/53/93/97/14)''')
code(s20(9))

md(r'''## Cell 5 — Model (identical to every prior localization phase)''')
code(s20(11))

md(r'''## Cell 6 — Equivariance check before training anything (probe cast to fp32 -- nn.Linear needs matching dtype, dataset is fp16 now)''')
code(s20(13).replace('probe = dataset[0].clone()', 'probe = dataset[0].clone(); probe.x = probe.x.float()'))

md(r'''## Cell 7 — Training machinery (run_epoch upcasts each batch to fp32 -- fp16-on-disk fix)''')
code(s20(15).replace('    for b in loader:\n        b = b.to(DEVICE)',
                      '    for b in loader:\n        b = b.to(DEVICE); b.x = b.x.float()   # fp16-on-disk fix: upcast per-batch, not the whole resident dataset'))

md(r'''## Cell 8 — Train seeds 3-7 (extending Phase97's 0-2 to the full 8-seed ensemble)''')
code(s20(17).replace("clean_deploy_seed{seed}.pt", "clean_deploy_esm3b_seed{seed}.pt")
            .replace("for seed in (0,1,2):", "for seed in (3,4,5,6,7):"))

code(r'''
print(f"\nN_TRAIN_CHAINS = {len(dataset):,}")
print("checkpoints saved: clean_deploy_esm3b_seed3.pt through seed7.pt")
print("Phase101 (seeds 3-7) complete -- combine with Phase97 (seeds 0-2) for the full 8-seed ensemble.")
''')

md(r'''
---

Checkpoints `clean_deploy_esm3b_seed{3..7}.pt` are the deliverable.
Combined with Phase97's `seed{0,1,2}.pt`, gives the full 8-seed ensemble.
A rescore-only notebook (mirroring Phase98, loading BOTH Phase97 and
Phase101 as kernel_sources, iterating `range(8)` instead of `(0,1,2)`)
is the next step.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase101_esm3b_coach420_train_seeds37.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
