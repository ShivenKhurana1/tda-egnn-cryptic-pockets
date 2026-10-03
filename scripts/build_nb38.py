#!/usr/bin/env python3
"""Phase 38: does pushing the training pool further (5,339 clusters, 2.6x
original) extend COACH420's win beyond Phase31/32's already-strong result
(8-seed DCC 0.564 vs P2Rank 0.521, +0.043)? Every cell spliced verbatim from
Phase 20 v2 (same proven methodology Phase31 already validated at 1.74x
scale) but loading phase37_graphs.pt/phase36_graphs.pt instead of
phase30_graphs.pt/phase29_graphs.pt."""
import json, pickle
from pathlib import Path

C20 = pickle.load(open("/tmp/p20_cells.pkl", "rb"))
s20 = lambda i: (C20[i]["source"] if isinstance(C20[i]["source"], str) else "".join(C20[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 38 - Round 2: Does Pushing Scale Further Extend the COACH420 Win?

Phase 31/32 established that scaling the training pool 1.74x (2,056 ->
3,577 clusters) turned COACH420's near-parity result into a robust,
8-seed-validated WIN over live P2Rank (+0.043 DCC). This notebook reruns
the identical methodology (every cell spliced verbatim from Phase 20 v2) on
the round-2 pool (5,339 clusters, 2.6x original, 1.49x Phase31's pool) to
see whether the win keeps growing, plateaus, or has already saturated.
3-seed first pass, same discipline as every other new arm.
''')

md(r'''## Cell 1 - Setup (loads Phase 37/36's round-2 scaled blobs)''')
code(s20(2))
code(s20(3).replace('locate("phase10_graphs.pt")', 'locate("phase37_graphs.pt")')
           .replace('locate("phase6b_graphs.pt")', 'locate("phase36_graphs.pt")'))

md(r'''## Cell 2 - Fetch ALL 420 COACH420 structures (identical to Phase 20/31)''')
code(s20(5))

md(r'''## Cell 3 - Clean the TRAINING side: exclude any chain homologous to COACH420 (identical to Phase 20 v2/31)''')
code(s20(7))

md(r'''## Cell 4 - Recover true ligand centers for the cleaned training set (identical to Phase 20/31/14)''')
code(s20(9))

md(r'''## Cell 5 - Model (identical to every prior localization phase)''')
code(s20(11))

md(r'''## Cell 6 - Equivariance check before training anything (identical to Phase 20/31)''')
code(s20(13))

md(r'''## Cell 7 - Training machinery (identical to Phase 14/20/31)''')
code(s20(15))

md(r'''## Cell 8 - Train the clean-training ensemble (3 seeds, lambda=0.1) on the ROUND-2 SCALED pool''')
code(s20(17).replace("clean_deploy_seed{seed}.pt", "clean_deploy_scaled2_seed{seed}.pt"))

md(r'''## Cell 9 - Ground-truth ligands for ALL 420 COACH420 structures (identical to Phase 20/31)''')
code(s20(19))

md(r'''## Cell 10 - Featurize (identical pipeline: dihedral + ESM2 + topology, matched to the round-2 topology block)''')
code(s20(21).replace("cache/phase20_raw_features.pkl", "cache/phase38_raw_features.pkl"))

md(r'''## Cell 11 - P2Rank on the FULL 420 (same-run controlled reference, identical to Phase 20/31)''')
code(s20(23).replace('Path("/tmp/p2_20")', 'Path("/tmp/p2_38")')
            .replace('Path("/tmp/p2_20_out")', 'Path("/tmp/p2_38_out")'))

md(r'''## Cell 12 - Score both, compare to literature AND to Phase 20/21/31/32's pool-size progression''')
code(s20(25).replace("phase20_summary.json", "phase38_summary.json"))

code(r'''
# --- extra: direct comparison across all three pool sizes tested so far ---
print(f"\n--- COACH420 DCC across training-pool scale ---")
print(f"{'pool':<28}{'clusters':>10}{'DCC (3-seed)':>14}{'DCC (8-seed)':>14}")
print("-"*66)
print(f"{'original (Phase20/21)':<28}{2056:>10}{0.4770:>14.4f}{0.5161:>14.4f}")
print(f"{'1.74x scaled (Phase31/32)':<28}{3577:>10}{0.5545:>14.4f}{0.5642:>14.4f}")
print(f"{'2.6x scaled (this notebook)':<28}{5339:>10}{ours_dcc_rate:>14.4f}{'pending':>14}")
print(f"\ntraining pool size: {len(dataset):,} clean chains (round-2) vs 3,330 (Phase31) vs 1,858 (Phase20)")
print(f"this-run P2Rank DCC: {p2_dcc_rate:.4f}  (compare to Phase20/21/31/32's 0.520885 -- "
      f"deterministic tool, should match closely, only training-side data differs)")
''')

md(r'''
---

## Reading this

- **DCC keeps climbing past 0.5642 (Phase32's 8-seed number)** -- scale
  hasn't saturated yet, worth pushing even further.
- **DCC plateaus near 0.55-0.57** -- diminishing returns setting in around
  this pool size; further scale-up may not be worth the compute.
- **DCC drops** -- would be surprising and worth investigating (possible
  data quality issue at this larger scale) rather than accepting at face
  value.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase38_scaled2_coach420.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
