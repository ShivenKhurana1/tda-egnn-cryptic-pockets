#!/usr/bin/env python3
"""Phase 33: does the scaled training pool also close Thread 2's still-open
HOLO4K gap? Mirrors Phase 24's exact "clean training + full HOLO4K sample
benchmark" methodology (every cell spliced verbatim, proven code) but loads
phase30_graphs.pt/phase29_graphs.pt (3,577-cluster pool) instead of
phase10_graphs.pt/phase6b_graphs.pt (2,056-cluster pool). Motivated by
Phase31/32's finding that training-data scale, not seed count, was the real
bottleneck on COACH420 -- worth testing whether the same holds for HOLO4K,
where Phase27b found a real, seed-validated -0.067 DCC loss to live P2Rank
on the ORIGINAL pool."""
import json, pickle
from pathlib import Path

C24 = pickle.load(open("/tmp/p24_cells.pkl", "rb"))
s24 = lambda i: (C24[i]["source"] if isinstance(C24[i]["source"], str) else "".join(C24[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 33 - Does the Scaled Pool Close Thread 2's HOLO4K Gap?

Phase 31/32 found the scaled 3,577-cluster training pool gives a robust,
8-seed-validated WIN over live P2Rank on COACH420 (+0.043 DCC) -- a much
better outcome than Phase20/21's near-parity result on the original
2,056-cluster pool. That strongly suggests training-data scale, not seed
count, was the real bottleneck.

Thread 2 (Phase24/27b) found the ORIGINAL pool has a genuine, only
partially-noise-explained LOSS to live P2Rank on HOLO4K specifically
(-0.067 DCC at 8 seeds, 2.3x the noise floor) -- qualitatively different
from COACH420's near-parity. This notebook tests whether the scaled pool
closes that gap too, using Phase 24's exact methodology (spliced verbatim,
proven code) pointed at the bigger pool. 3-seed first pass, same discipline
as every other new arm.
''')

md(r'''## Cell 1 - Setup (loads Phase 30/29's scaled blobs instead of Phase 10/6b's)''')
code(s24(2))
code(s24(3).replace('locate("phase10_graphs.pt")', 'locate("phase30_graphs.pt")')
           .replace('locate("phase6b_graphs.pt")', 'locate("phase29_graphs.pt")'))

md(r'''## Cell 2 - Sample the SAME ~1,000 HOLO4K structures (seed=2024, identical to Phase 24)''')
code(s24(5))

md(r'''## Cell 3 - Clean the TRAINING side (identical logic; pool size updated 2056 -> 3577)''')
code(s24(7).replace("2056", "3577"))

md(r'''## Cell 4 - Recover true ligand centers (identical to Phase 24/14)''')
code(s24(9))

md(r'''## Cell 5 - Model (identical to every prior localization phase)''')
code(s24(11))

md(r'''## Cell 6 - Training machinery (identical to Phase 24/14)''')
code(s24(13))

md(r'''## Cell 7 - Train the clean ensemble (3 seeds) on the SCALED pool''')
code(s24(15).replace("holo_clean_deploy_seed{seed}.pt", "holo_clean_deploy_scaled_seed{seed}.pt"))

md(r'''## Cell 8 - Ground truth + featurize the sampled 1,000 structures (identical to Phase 24)''')
code(s24(17))

md(r'''## Cell 9 - Featurize (dihedral + ESM2 + topology, identical pipeline; cache filename updated)''')
code(s24(19).replace("phase24_raw_features.pkl", "phase33_raw_features.pkl"))

md(r'''## Cell 10 - P2Rank on the same 1,000 structures (identical to Phase 24)''')
code(s24(21))

md(r'''## Cell 11 - Score, compare to literature AND to Phase 24/27b's original-pool numbers''')
code(s24(23).replace("2056", "3577").replace("phase24_summary.json", "phase33_summary.json"))

code(r'''
# --- extra: direct comparison to the original 2,056-cluster pool's numbers ---
ORIGINAL_POOL = {"3seed": {"dcc": 0.5601, "dca": 0.7806}, "8seed": {"dcc": 0.5912}}
print(f"\n--- scaled pool (this notebook, 3-seed) vs original pool ---")
print(f"original 3-seed DCC: {ORIGINAL_POOL['3seed']['dcc']:.4f}  scaled 3-seed DCC: {ours_dcc_rate:.4f}  "
      f"delta: {ours_dcc_rate - ORIGINAL_POOL['3seed']['dcc']:+.4f}")
print(f"original 8-seed DCC: {ORIGINAL_POOL['8seed']['dcc']:.4f}  (Phase27b, not yet matched here -- "
      f"this is only a 3-seed first pass on the scaled pool)")
print(f"training pool size: {len(dataset):,} clean chains (scaled) vs 1,760 (original, Phase24)")
print(f"this-run P2Rank DCC: {p2_dcc_rate:.4f}  (compare to Phase24/27b's 0.6580 -- should be close, "
      f"same 1000-structure sample, deterministic tool, only training-side data differs)")
''')

md(r'''
---

## Reading this

- **Scaled DCC beats the original pool's 8-seed number (0.5912) by more
  than the ~0.029 seed noise floor from Phase27b** -- more data helps here
  too, worth extending to 8 seeds (mirroring Phase 32's treatment of the
  COACH420 result).
- **Scaled DCC is within noise of 0.5912** -- COACH420 and HOLO4K may
  respond differently to training-data scale; worth understanding why
  before assuming they generalize identically.
- **Scaled DCC is still clearly behind live P2Rank (0.658)** -- would mean
  Thread 2's gap is NOT primarily a training-data-scale problem, unlike
  Thread 1's COACH420 finding -- a genuinely different, still-open question.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase33_scaled_holo4k.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
