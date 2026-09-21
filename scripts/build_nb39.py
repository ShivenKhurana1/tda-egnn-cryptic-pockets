#!/usr/bin/env python3
"""Phase 39: does pushing the training pool further (5,339 clusters, 2.6x
original) close HOLO4K's still-ambiguous gap (Phase33/34: 8-seed -0.027,
1.4x noise, "not confidently resolved")? Every cell spliced verbatim from
Phase 24 (proven methodology Phase33 already validated at 1.74x scale) but
loading phase37_graphs.pt/phase36_graphs.pt instead of
phase30_graphs.pt/phase29_graphs.pt."""
import json, pickle
from pathlib import Path

C24 = pickle.load(open("/tmp/p24_cells.pkl", "rb"))
s24 = lambda i: (C24[i]["source"] if isinstance(C24[i]["source"], str) else "".join(C24[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 39 — Round 2: Does Pushing Scale Further Close the HOLO4K Gap?

Phase 33/34 found scaling the pool 1.74x narrowed HOLO4K's gap to live
P2Rank from a confirmed real loss (-0.067, 2.3x noise, original pool) to
statistical ambiguity (-0.027, 1.4x noise) -- real progress, but not the
clean win COACH420 got. This notebook reruns the identical methodology
(spliced verbatim from Phase 24) on the round-2 pool (5,339 clusters, 2.6x
original) to see whether more scale finally resolves it. 3-seed first pass,
same discipline as every other new arm.
''')

md(r'''## Cell 1 — Setup (loads Phase 37/36's round-2 scaled blobs)''')
code(s24(2))
code(s24(3).replace('locate("phase10_graphs.pt")', 'locate("phase37_graphs.pt")')
           .replace('locate("phase6b_graphs.pt")', 'locate("phase36_graphs.pt")'))

md(r'''## Cell 2 — Sample the SAME ~1,000 HOLO4K structures (seed=2024, identical to Phase 24/33)''')
code(s24(5))

md(r'''## Cell 3 — Clean the TRAINING side (identical logic; pool size updated 2056 -> 5339)''')
code(s24(7).replace("2056", "5339"))

md(r'''## Cell 4 — Recover true ligand centers (identical to Phase 24/33/14)''')
code(s24(9))

md(r'''## Cell 5 — Model (identical to every prior localization phase)''')
code(s24(11))

md(r'''## Cell 6 — Training machinery (identical to Phase 24/33/14)''')
code(s24(13))

md(r'''## Cell 7 — Train the clean ensemble (3 seeds) on the ROUND-2 SCALED pool''')
code(s24(15).replace("holo_clean_deploy_seed{seed}.pt", "holo_clean_deploy_scaled2_seed{seed}.pt"))

md(r'''## Cell 8 — Ground truth + featurize the sampled 1,000 structures (identical to Phase 24/33)''')
code(s24(17))

md(r'''## Cell 9 — Featurize (dihedral + ESM2 + topology, identical pipeline; cache filename updated)''')
code(s24(19).replace("phase24_raw_features.pkl", "phase39_raw_features.pkl"))

md(r'''## Cell 10 — P2Rank on the same 1,000 structures (identical to Phase 24/33)''')
code(s24(21))

md(r'''## Cell 11 — Score, compare to literature AND to the pool-size progression so far''')
code(s24(23).replace("2056", "5339").replace("phase24_summary.json", "phase39_summary.json"))

code(r'''
# --- extra: direct comparison across all three pool sizes tested so far ---
print(f"\n--- HOLO4K DCC across training-pool scale ---")
print(f"{'pool':<28}{'clusters':>10}{'DCC (3-seed)':>14}{'DCC (8-seed)':>14}{'gap to P2Rank':>16}")
print("-"*82)
print(f"{'original (Phase24/27b)':<28}{2056:>10}{0.5601:>14.4f}{0.5912:>14.4f}{-0.0668:>16.4f}")
print(f"{'1.74x scaled (Phase33/34)':<28}{3577:>10}{0.6283:>14.4f}{0.6313:>14.4f}{-0.0267:>16.4f}")
print(f"{'2.6x scaled (this notebook)':<28}{5339:>10}{ours_dcc_rate:>14.4f}{'pending':>14}"
      f"{ours_dcc_rate-p2_dcc_rate:>16.4f}")
print(f"\ntraining pool size: {len(dataset):,} clean chains (round-2) vs 3,187 (Phase33) vs 1,760 (Phase24)")
print(f"this-run P2Rank DCC: {p2_dcc_rate:.4f}  (compare to Phase24/27b/33/34's 0.657974 -- "
      f"deterministic tool, same 1000-structure sample, should match closely)")
''')

md(r'''
---

## Reading this

- **Gap closes to near-zero or flips positive** -- matches the pattern
  COACH420 already showed; scale was the answer for both benchmarks, just
  needed more of it for HOLO4K's harder distribution.
- **Gap narrows further but stays negative and outside noise** -- HOLO4K
  may have a structural reason to lag P2Rank beyond training data volume
  (worth a dedicated investigation if this holds).
- **Gap stops improving / plateaus around -0.02 to -0.03** -- scale has
  diminishing returns specifically on this benchmark; further scale-up
  likely won't close it either.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase39_scaled2_holo4k.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
