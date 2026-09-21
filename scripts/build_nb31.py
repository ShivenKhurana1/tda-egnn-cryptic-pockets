#!/usr/bin/env python3
"""Phase 31: Thread 3's payoff notebook -- rerun Phase 20's exact "clean
training + full COACH420 benchmark" methodology on phase30_graphs.pt (the
1.74x-scaled pool) instead of phase10_graphs.pt. Every cell is spliced
verbatim from Phase 20 (proven, bug-fixed v2 -- the coach_paths collision
fix is already baked in) except: (1) the two locate() calls in setup now
point at phase30/phase29 instead of phase10/phase6b, (2) output filenames
are namespaced phase31_* / *_scaled_* so nothing collides with Phase20's
own cached outputs if both are ever attached as sources together."""
import json, pickle
from pathlib import Path

C20 = pickle.load(open("/tmp/p20_cells.pkl", "rb"))
s20 = lambda i: (C20[i]["source"] if isinstance(C20[i]["source"], str) else "".join(C20[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 31 — Thread 3 Payoff: Clean Training + Full COACH420 on the Scaled Pool

Phase 20/21 established the reference result on the original 2,056-cluster
pool: clean training (leakage-filtered against COACH420) + full 420-structure
benchmark scoring gives ours DCC 0.477 (3-seed) / 0.516 (8-seed) vs literature
EquiPocket 0.423 (beats) vs this-run P2Rank 0.521 (loses by -0.044 at 3-seed,
-0.0052 at 8-seed -- Phase21 showed that gap is mostly noise).

This notebook reruns the IDENTICAL methodology (every cell spliced verbatim
from Phase 20's proven, bug-fixed v2) on Phase 30's 3,577-cluster pool
(1.74x bigger) to see whether more training data changes any of these
numbers. Same leakage-filter threshold (30% identity), same architecture,
same 3-seed first pass (consistent with every other phase's discipline of
screening at 3 seeds before committing to 8).
''')

md(r'''## Cell 1 — Setup (loads Phase 30/29's scaled blobs instead of Phase 10/6b's)''')
code(s20(2))
code(s20(3).replace('locate("phase10_graphs.pt")', 'locate("phase30_graphs.pt")')
           .replace('locate("phase6b_graphs.pt")', 'locate("phase29_graphs.pt")'))

md(r'''## Cell 2 — Fetch ALL 420 COACH420 structures (identical to Phase 20)''')
code(s20(5))

md(r'''## Cell 3 — Clean the TRAINING side: exclude any chain homologous to COACH420 (identical to Phase 20 v2, coach_paths fix included)''')
code(s20(7))

md(r'''## Cell 4 — Recover true ligand centers for the cleaned training set (identical to Phase 20/14)''')
code(s20(9))

md(r'''## Cell 5 — Model (identical to every prior localization phase)''')
code(s20(11))

md(r'''## Cell 6 — Equivariance check before training anything (identical to Phase 20)''')
code(s20(13))

md(r'''## Cell 7 — Training machinery (identical to Phase 14/20)''')
code(s20(15))

md(r'''## Cell 8 — Train the clean-training ensemble (3 seeds, lambda=0.1) on the SCALED pool''')
code(s20(17).replace('WORK/f"cache/clean_deploy_seed{seed}.pt"',
                      'WORK/f"cache/clean_deploy_scaled_seed{seed}.pt"'))

md(r'''## Cell 9 — Ground-truth ligands for ALL 420 COACH420 structures (identical to Phase 20)''')
code(s20(19))

md(r'''## Cell 10 — Featurize (identical pipeline: dihedral + ESM2 + topology, matched to the scaled topology block via blob6b/blob29)''')
code(s20(21).replace('cache/phase20_raw_features.pkl', 'cache/phase31_raw_features.pkl'))

md(r'''## Cell 11 — P2Rank on the FULL 420 (same-run controlled reference, identical to Phase 20)''')
code(s20(23).replace('Path("/tmp/p2_20")', 'Path("/tmp/p2_31")')
            .replace('Path("/tmp/p2_20_out")', 'Path("/tmp/p2_31_out")'))

md(r'''## Cell 12 — Score both, compare to literature AND to Phase 20/21's original-pool numbers''')
code(s20(25).replace('phase20_summary.json', 'phase31_summary.json'))

code(r'''
# --- extra: direct comparison to the original 2,056-cluster pool's numbers ---
ORIGINAL_POOL = {"3seed": {"dcc": 0.4770, "dca": 0.751}, "8seed": {"dcc": 0.5161}}
print(f"\n--- scaled pool (this notebook, 3-seed) vs original pool ---")
print(f"original 3-seed DCC: {ORIGINAL_POOL['3seed']['dcc']:.4f}  scaled 3-seed DCC: {ours_dcc_rate:.4f}  "
      f"delta: {ours_dcc_rate - ORIGINAL_POOL['3seed']['dcc']:+.4f}")
print(f"original 8-seed DCC: {ORIGINAL_POOL['8seed']['dcc']:.4f}  (Phase21, not yet matched here -- "
      f"this is only a 3-seed first pass on the scaled pool)")
print(f"training pool size: {len(dataset):,} clean chains (scaled) vs 1,858 (original, Phase20)")
''')

md(r'''
---

## Reading this

Three outcomes:
- **Scaled DCC clearly beats the original pool's 3-seed number (0.477) by
  more than the ~0.03 seed-to-seed noise scale** -- more data helps, worth
  extending to 8 seeds and doing the same for HOLO4K.
- **Scaled DCC is within noise of 0.477** -- more data (at this 1.74x scale)
  doesn't move the needle; the bottleneck is elsewhere (architecture,
  labels, or the localization head itself), not training-set size.
- **Scaled DCC is worse** -- would be surprising and worth investigating
  before trusting (check clean training set size, check for a bug) rather
  than accepting at face value.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase31_scaled_coach420.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
