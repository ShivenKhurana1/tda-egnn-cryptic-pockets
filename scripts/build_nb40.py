#!/usr/bin/env python3
"""Phase 40: is Phase 38's biggest-yet COACH420 win (+0.055 DCC, 3-seed,
2.6x pool) real, or seed luck? Mirrors Phase 21/32's exact treatment
(train 5 more seeds, reuse cached benchmark features, per-seed noise floor
+ ensemble-size sweep) but pointed at Phase 38's round-2 scaled pool."""
import json, pickle
from pathlib import Path

C21 = pickle.load(open("/tmp/p21_cells.pkl", "rb"))
s21 = lambda i: (C21[i]["source"] if isinstance(C21[i]["source"], str) else "".join(C21[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 40 — Is Phase 38's Round-2 COACH420 Win Real, or Seed Luck?

Phase 38 found the round-2 pool (5,339 clusters) gives DCC 0.5763 vs P2Rank's
0.5209 (+0.0554, 3-seed) -- the biggest COACH420 win yet, continuing the
monotonic trend from Phase20/21 (original) through Phase31/32 (1.74x).
Every prior scale-up's win GREW with more seeds rather than shrinking
(Phase21: gap closed 88%; Phase32: gap widened +0.034 -> +0.043). This
notebook applies the identical stress test: train 5 more seeds (8 total) on
Phase 38's leakage-clean training set, rescore against Phase38's cached
benchmark features.
''')

md(r'''## Cell 1 — Setup, rebuild the clean training set (Phase 38 logic, reused) on the ROUND-2 pool''')
code(s21(2))
code(s21(3).replace('locate("phase10_graphs.pt")', 'locate("phase37_graphs.pt")'))

md(r'''## Cell 1b — apt update (identical to Phase 21/32)''')
code(s21(4))

md(r'''## Cell 1c — Rebuild the leakage-clean set against COACH420 (identical logic; should reproduce Phase 38's 5,041 chains)''')
code(s21(5))

md(r'''## Cell 2 — Recover true ligand centers (Phase 14 logic, reused)''')
code(s21(7))

md(r'''## Cell 3 — Model (identical to every prior localization phase)''')
code(s21(9))

md(r'''## Cell 4 — Training machinery (identical to Phase 14)''')
code(s21(11))

md(r"""## Cell 5 — Load Phase 38's 3 checkpoints, train 5 MORE seeds (8 total)""")
code(s21(13).replace("clean_deploy_seed{seed}.pt", "clean_deploy_scaled2_seed{seed}.pt")
            .replace('attach Phase 20\'s kernel output', 'attach Phase 38\'s kernel output')
            .replace('loaded {len(models)} checkpoints from Phase 20', 'loaded {len(models)} checkpoints from Phase 38'))

md(r"""## Cell 6 — Load Phase 38's cached benchmark features (no re-fetch)""")
code(s21(15).replace('locate("phase20_raw_features.pkl")', 'locate("phase38_raw_features.pkl")')
            .replace('locate("phase6b_graphs.pt")', 'locate("phase36_graphs.pt")')
            .replace('loaded cached COACH420 features', 'loaded cached (round-2 pool) COACH420 features'))

md(r'''## Cell 7 — Per-seed and per-ensemble-size DCC on the cached benchmark''')
code(s21(17).replace("Phase 20's 3-seed ensemble reported 0.4770", "Phase 38's 3-seed ensemble reported 0.5763"))

md(r'''## Cell 8 — Ensemble size sweep: does averaging more seeds change the win?''')
code(s21(19).replace('P2RANK_THIS_RUN = 0.5209   # Phase 20\'s same-run, same-benchmark P2Rank number',
                      'P2RANK_THIS_RUN = 0.520885   # Phase38\'s same-run, same-benchmark P2Rank number '
                      '(deterministic tool -- matches every prior phase\'s number, confirms benchmark side identical)')
            .replace("Phase 20's 3-seed (0,1,2 only) for direct comparison:", "Phase 38's 3-seed (0,1,2 only) for direct comparison:")
            .replace("[Phase 20 reported 0.4770]", "[Phase 38 reported 0.5763]"))

md(r'''## Cell 9 — Verdict''')
code(s21(21).replace('WORK/"phase21_summary.json"', 'WORK/"phase40_summary.json"')
            .replace('saved -> phase21_summary.json', 'saved -> phase40_summary.json'))

code(r'''
print(f"\n--- COACH420 8-seed DCC across training-pool scale ---")
print(f"{'pool':<28}{'clusters':>10}{'DCC (8-seed)':>14}")
print("-"*52)
print(f"{'original (Phase21)':<28}{2056:>10}{0.5161:>14.4f}")
print(f"{'1.74x scaled (Phase32)':<28}{3577:>10}{0.5642:>14.4f}")
print(f"{'2.6x scaled (this notebook)':<28}{5339:>10}{full8_dcc:>14.4f}")
''')

md(r'''
---

## Reading this

- **Win grows or holds** -- scale-up keeps paying off cleanly at every step
  tested so far, matching the pattern from every prior round.
- **Win shrinks toward zero but stays positive** -- diminishing returns
  setting in, still net positive.
- **Win disappears or reverses** -- would break the clean monotonic trend
  seen at 1.74x and be worth investigating rather than accepting at face
  value.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase40_scaled2_seed_variance.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
