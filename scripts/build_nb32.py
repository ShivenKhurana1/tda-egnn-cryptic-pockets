#!/usr/bin/env python3
"""Phase 32: is Phase 31's scaled-pool COACH420 win (+0.034 DCC over live
P2Rank, 3-seed) real, or seed luck? Mirrors Phase 21's exact treatment of
Phase 20's original-pool result (train 5 more seeds, reuse cached benchmark
features, per-seed noise floor + ensemble-size sweep) but pointed at
Phase 31's scaled pool instead. Every cell spliced verbatim from Phase 21
except filenames/checkpoint-prefixes renamed to the _scaled_ variants."""
import json, pickle
from pathlib import Path

C21 = pickle.load(open("/tmp/p21_cells.pkl", "rb"))
s21 = lambda i: (C21[i]["source"] if isinstance(C21[i]["source"], str) else "".join(C21[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 32 — Is Phase 31's Scaled-Pool COACH420 Win Real, or Seed Luck?

Phase 31 found the scaled 3,330-chain training pool (vs the original 1,858)
clearly BEATS live P2Rank on COACH420 for the first time in this project
(DCC 0.5545 vs 0.5209, +0.0336) -- but it's only a 3-seed pass, and that
margin is roughly the same scale as the seed-to-seed noise floor seen
elsewhere (Phase22: 0.0112-0.0193; Phase27b: 0.0288). Phase 21 already ran
this exact stress test on the ORIGINAL pool's -0.044 gap and found it mostly
closed with more seeds (down to -0.0052, 88% reduction) -- so seed count
alone can move a result by roughly this much. This notebook applies the
identical methodology to Phase 31's result: train 5 more seeds (8 total),
rescore against Phase31's cached benchmark features (no re-fetch), and see
whether the win holds, grows, or shrinks toward the noise floor.
''')

md(r'''## Cell 1 — Setup, rebuild the clean training set (Phase 31/20 logic, reused) on the SCALED pool''')
code(s21(2))
code(s21(3).replace('locate("phase10_graphs.pt")', 'locate("phase30_graphs.pt")'))

md(r'''## Cell 1b — apt update (identical to Phase 21)''')
code(s21(4))

md(r'''## Cell 1c — Rebuild the leakage-clean set against COACH420 (identical logic; should reproduce Phase 31's 3,330 chains)''')
code(s21(5))

md(r'''## Cell 2 — Recover true ligand centers (Phase 14 logic, reused)''')
code(s21(7))

md(r'''## Cell 3 — Model (identical to every prior localization phase)''')
code(s21(9))

md(r'''## Cell 4 — Training machinery (identical to Phase 14)''')
code(s21(11))

md(r"""## Cell 5 — Load Phase 31's 3 checkpoints, train 5 MORE seeds (8 total)""")
code(s21(13).replace("clean_deploy_seed{seed}.pt", "clean_deploy_scaled_seed{seed}.pt")
            .replace('attach Phase 20\'s kernel output', 'attach Phase 31\'s kernel output')
            .replace('loaded {len(models)} checkpoints from Phase 20', 'loaded {len(models)} checkpoints from Phase 31'))

md(r"""## Cell 6 — Load Phase 31's cached benchmark features (no re-fetch)""")
code(s21(15).replace('locate("phase20_raw_features.pkl")', 'locate("phase31_raw_features.pkl")')
            .replace('locate("phase6b_graphs.pt")', 'locate("phase29_graphs.pt")')
            .replace('loaded cached COACH420 features', 'loaded cached (scaled-pool) COACH420 features'))

md(r'''## Cell 7 — Per-seed and per-ensemble-size DCC on the cached benchmark''')
code(s21(17).replace("Phase 20's 3-seed ensemble reported 0.4770", "Phase 31's 3-seed ensemble reported 0.5545"))

md(r'''## Cell 8 — Ensemble size sweep: does averaging more seeds change the win?''')
code(s21(19).replace('P2RANK_THIS_RUN = 0.5209   # Phase 20\'s same-run, same-benchmark P2Rank number',
                      'P2RANK_THIS_RUN = 0.520885   # Phase 31\'s same-run, same-benchmark P2Rank number '
                      '(deterministic tool -- same value as Phase20/21\'s, confirms benchmark side is identical)')
            .replace("Phase 20's 3-seed (0,1,2 only) for direct comparison:", "Phase 31's 3-seed (0,1,2 only) for direct comparison:")
            .replace("[Phase 20 reported 0.4770]", "[Phase 31 reported 0.5545]"))

md(r'''## Cell 9 — Verdict''')
code(s21(21).replace('WORK/"phase21_summary.json"', 'WORK/"phase32_summary.json"')
            .replace('saved -> phase21_summary.json', 'saved -> phase32_summary.json'))

md(r'''
---

## Reading this

- **Win grows or holds (8-seed gap stays clearly positive, well beyond
  noise)** -- the scaled-pool improvement is real and robust, the strongest
  possible outcome for Thread 3.
- **Win shrinks toward zero but stays positive** -- some of the 3-seed
  margin was seed luck, but scale-up still helped on net (matches Phase21's
  precedent of gaps shrinking with more seeds, just from a better starting
  point).
- **Win disappears or reverses** -- would mean Phase 31's headline result
  was mostly a lucky 3-seed draw, and the "training-data scale was the
  bottleneck" conclusion needs to be walked back to "not yet shown" rather
  than accepted.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase32_scaled_seed_variance.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
