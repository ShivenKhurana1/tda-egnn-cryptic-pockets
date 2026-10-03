#!/usr/bin/env python3
"""Phase 56: is Phase 53's round-3 COACH420 result (+0.065 DCC, 3-seed,
3.65x pool) real, or has scale started to saturate (3-seed number sat
BELOW round-2's own 8-seed number for the first time)? Mirrors
Phase21/32/40's exact treatment but pointed at Phase 53's round-3 pool."""
import json, pickle
from pathlib import Path

C21 = pickle.load(open("/tmp/p21_cells.pkl", "rb"))
C53 = pickle.load(open("/tmp/p53_cells.pkl", "rb"))
s21 = lambda i: (C21[i]["source"] if isinstance(C21[i]["source"], str) else "".join(C21[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 56 - Is Phase 53's Round-3 COACH420 Result Real, or Has Scale Started to Saturate?

Phase 53 found the round-3 pool (7,513 clusters) gives DCC 0.5860 vs
P2Rank's 0.5209 (+0.0651, 3-seed) -- still a clear win, but this is the
FIRST round where the 3-seed number sat BELOW the prior round's own 8-seed
number (0.5956), and the round-2->round-3 increment (+0.0097) was smaller
than round-1->round-2's (+0.0218). This could mean scale is finally
saturating, or could just be normal 3-seed noise that resolves with more
seeds like every prior round did (Phase21: closed 88%; Phase32/40: grew
further). This notebook applies the identical 8-seed stress test to settle
which one it is.
''')

md(r'''## Cell 1 - Setup, rebuild the clean training set (Phase 53 logic, reused) on the ROUND-3 pool''')
code(s21(2))
code(s21(3).replace('locate("phase10_graphs.pt")', 'locate("phase52_graphs.pt")'))

md(r'''## Cell 1b - apt update (identical to Phase 21/32)''')
code(s21(4))

md(r'''## Cell 1c - Rebuild the leakage-clean set against COACH420 (identical logic; should reproduce Phase 53's 7,181 chains)''')
code(s21(5))

md(r'''## Cell 2 - Recover true ligand centers (Phase 14 logic, reused)''')
code(s21(7))

md(r'''## Cell 3 - Model (identical to every prior localization phase)''')
code(s21(9))

md(r'''## Cell 4 - Training machinery (identical to Phase 14)''')
code(s21(11))

md(r"""## Cell 5 - Load Phase 53's 3 checkpoints, train 5 MORE seeds (8 total)""")
code(s21(13).replace("clean_deploy_seed{seed}.pt", "clean_deploy_scaled3_seed{seed}.pt")
            .replace("attach Phase 20's kernel output", "attach Phase 53's kernel output")
            .replace("loaded {len(models)} checkpoints from Phase 20", "loaded {len(models)} checkpoints from Phase 53"))

md(r"""## Cell 6 - Load Phase 53's cached benchmark features (no re-fetch)""")
code(s21(15).replace('locate("phase20_raw_features.pkl")', 'locate("phase53_raw_features.pkl")')
            .replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")')
            .replace('loaded cached COACH420 features', 'loaded cached (round-3 pool) COACH420 features'))

md(r'''## Cell 7 - Per-seed and per-ensemble-size DCC on the cached benchmark''')
code(s21(17).replace("Phase 20's 3-seed ensemble reported 0.4770", "Phase 53's 3-seed ensemble reported 0.5860"))

md(r'''## Cell 8 - Ensemble size sweep: does averaging more seeds change the win?''')
code(s21(19).replace('P2RANK_THIS_RUN = 0.5209   # Phase 20\'s same-run, same-benchmark P2Rank number',
                      'P2RANK_THIS_RUN = 0.520885   # Phase53\'s same-run, same-benchmark P2Rank number '
                      '(deterministic tool -- matches every prior phase\'s number, confirms benchmark side identical)')
            .replace("Phase 20's 3-seed (0,1,2 only) for direct comparison:", "Phase 53's 3-seed (0,1,2 only) for direct comparison:")
            .replace("[Phase 20 reported 0.4770]", "[Phase 53 reported 0.5860]"))

md(r'''## Cell 9 - Verdict''')
code(s21(21).replace('WORK/"phase21_summary.json"', 'WORK/"phase56_summary.json"')
            .replace('saved -> phase21_summary.json', 'saved -> phase56_summary.json'))

code(r'''
print(f"\n--- COACH420 8-seed DCC across training-pool scale ---")
print(f"{'pool':<28}{'clusters':>10}{'DCC (8-seed)':>14}")
print("-"*52)
print(f"{'original (Phase21)':<28}{2056:>10}{0.5161:>14.4f}")
print(f"{'1.74x scaled (Phase32)':<28}{3577:>10}{0.5642:>14.4f}")
print(f"{'2.6x scaled (Phase40)':<28}{5339:>10}{0.5956:>14.4f}")
print(f"{'3.65x scaled (this notebook)':<28}{7513:>10}{full8_dcc:>14.4f}")
''')

md(r'''
---

## Reading this

- **Win grows or holds above 0.5956** -- scale still hasn't saturated even
  at this size, matching the pattern from every prior round.
- **Win shrinks toward Phase53's 3-seed number (0.5860) or below** --
  diminishing returns finally setting in around this pool size.
- **Win disappears or reverses** -- would break the clean monotonic trend
  seen at every prior scale and be worth investigating rather than
  accepting at face value.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase56_scaled3_seed_variance.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
