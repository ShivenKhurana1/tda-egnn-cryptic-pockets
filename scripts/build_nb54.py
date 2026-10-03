#!/usr/bin/env python3
"""Phase 54: does HOLO4K's win hold at round-3 scale (7,513 clusters, 3.65x
original)? Every cell spliced verbatim from Phase 24 (the same proven
methodology Phase33/39 already validated at 1.74x/2.6x scale) but loading
phase52_graphs.pt/phase51_graphs.pt instead of phase37_graphs.pt/
phase36_graphs.pt."""
import json, pickle
from pathlib import Path

C24 = pickle.load(open("/tmp/p24_cells.pkl", "rb"))
s24 = lambda i: (C24[i]["source"] if isinstance(C24[i]["source"], str) else "".join(C24[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 54 - Round 3: Does HOLO4K's Win Hold at Even Bigger Scale?

HOLO4K's gap to live P2Rank flipped from a confirmed real loss (original
pool, -0.0668) to a clean win at round-2 scale (5,339 clusters, +0.0124,
8-seed). This reruns the identical methodology (spliced verbatim from
Phase 24) on the round-3 pool (7,513 clusters, 3.65x original) to see
whether the win holds or grows further. 3-seed first pass, same discipline
as every other new arm.
''')

md(r'''## Cell 1 - Setup (loads Phase 52/51's round-3 scaled blobs)''')
code(s24(2))
code(s24(3).replace('locate("phase10_graphs.pt")', 'locate("phase52_graphs.pt")')
           .replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")'))

md(r'''## Cell 2 - Sample the SAME ~1,000 HOLO4K structures (seed=2024, identical to Phase 24/33/39)''')
code(s24(5))

md(r'''## Cell 3 - Clean the TRAINING side (identical logic; pool size updated 2056 -> 7513)''')
code(s24(7).replace("2056", "7513"))

md(r'''## Cell 4 - Recover true ligand centers (identical to Phase 24/33/39/14)''')
code(s24(9))

md(r'''## Cell 5 - Model (identical to every prior localization phase)''')
code(s24(11))

md(r'''## Cell 6 - Training machinery (identical to Phase 24/33/39/14)''')
code(s24(13))

md(r'''## Cell 7 - Train the clean ensemble (3 seeds) on the ROUND-3 SCALED pool''')
code(s24(15).replace("holo_clean_deploy_seed{seed}.pt", "holo_clean_deploy_scaled3_seed{seed}.pt"))

md(r'''## Cell 8 - Ground truth + featurize the sampled 1,000 structures (identical to Phase 24/33/39)''')
code(s24(17))

md(r'''## Cell 9 - Featurize (dihedral + ESM2 + topology, identical pipeline; cache filename updated)''')
code(s24(19).replace("phase24_raw_features.pkl", "phase54_raw_features.pkl"))

md(r'''## Cell 10 - P2Rank on the same 1,000 structures (identical to Phase 24/33/39)''')
code(s24(21))

md(r'''## Cell 11 - Score, compare to literature AND to the pool-size progression so far''')
code(s24(23).replace("2056", "7513").replace("phase24_summary.json", "phase54_summary.json"))

code(r'''
print(f"\n--- HOLO4K DCC across training-pool scale ---")
print(f"{'pool':<28}{'clusters':>10}{'DCC (3-seed)':>14}{'DCC (8-seed)':>14}{'gap to P2Rank':>16}")
print("-"*82)
print(f"{'original (Phase24/27b)':<28}{2056:>10}{0.5601:>14.4f}{0.5912:>14.4f}{-0.0668:>16.4f}")
print(f"{'1.74x scaled (Phase33/34)':<28}{3577:>10}{0.6283:>14.4f}{0.6313:>14.4f}{-0.0267:>16.4f}")
print(f"{'2.6x scaled (Phase39/41)':<28}{5339:>10}{0.6483:>14.4f}{0.6703:>14.4f}{0.0124:>16.4f}")
print(f"{'3.65x scaled (this notebook)':<28}{7513:>10}{ours_dcc_rate:>14.4f}{'pending':>14}"
      f"{ours_dcc_rate-p2_dcc_rate:>16.4f}")
print(f"\ntraining pool size: {len(dataset):,} clean chains (round-3)")
print(f"this-run P2Rank DCC: {p2_dcc_rate:.4f}  (compare to prior rounds' 0.657974 -- "
      f"deterministic tool, same 1000-structure sample, should match closely)")
''')

md(r'''
---

## Reading this

- **Gap stays positive or grows** -- the win from round 2 is holding at
  even bigger scale, strengthening the "scale was the answer for both
  benchmarks" conclusion.
- **Gap shrinks back toward zero** -- worth investigating whether round 2's
  win was itself partly seed/scale luck (would need an 8-seed check here
  too before concluding either way).
- **Gap goes negative again** -- would be a genuinely surprising reversal,
  worth real scrutiny before accepting.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase54_scaled3_holo4k.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
