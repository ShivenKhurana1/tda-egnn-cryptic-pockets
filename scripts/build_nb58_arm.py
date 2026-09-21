#!/usr/bin/env python3
"""Generates one single-arm training notebook for Thread 1's round-3
decomposition. Each arm gets its OWN notebook (training-only, no benchmark
scoring) so it fits within Kaggle's 12hr CPU session limit -- the combined
5-arm notebook (Phase58) hit that limit after only finishing one arm
(no_topo, ~11.7hrs for 3 seeds on the 7,181-chain pool). Usage:
  python3 build_nb58_arm.py <ARM_NAME> <PHASE_NUM>
ARM_NAME in {H0_only, H1_only, H2_only, burial_only}."""
import json, pickle, sys
from pathlib import Path

ARM_NAME = sys.argv[1]
PHASE_NUM = sys.argv[2]

C58 = pickle.load(open("/tmp/p58_cells.pkl", "rb"))
s58 = lambda i: (C58[i]["source"] if isinstance(C58[i]["source"], str) else "".join(C58[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(rf'''
# Phase {PHASE_NUM} — Thread 1 Round-3, Single Arm: {ARM_NAME}

Trains ONLY the `{ARM_NAME}` arm (3 seeds) for Thread 1's round-3
decomposition -- split out from Phase58 after that notebook hit Kaggle's
12hr CPU session limit partway through (finished `no_topo`'s 3 seeds in
~11.7hrs, got cancelled before starting the next arm). Training-only, no
benchmark scoring here -- that happens in a separate combining notebook
once all arms are trained. Identical training methodology (same clean
COACH420 training set, same model, same `fit()`/`run_epoch()`) to Phase58.
''')

md(r'''## Cell 1 — Setup (loads Phase 52/51's round-3 scaled blobs, identical to Phase53/56/58)''')
code(s58(2))
code(s58(3))

md(r'''## Cell 2 — Fetch ALL 420 COACH420 structures (identical to Phase 53/56/58)''')
code(s58(5))

md(r'''## Cell 3 — Clean the TRAINING side: exclude any chain homologous to COACH420 (identical to Phase 53/56/58; reproduces the same 7,181 chains)''')
code(s58(7))

md(r'''## Cell 4 — Recover true ligand centers for the cleaned training set (identical to Phase 53/56/58)''')
code(s58(9))

md(r'''## Cell 5 — Model (identical to every prior localization phase)''')
code(s58(11))

md(r'''## Cell 6 — Training machinery (identical to Phase 14/20/53/56/58)''')
code(s58(13))

md(r'''## Cell 7 — Build H0 / H1 / H2 / burial column masks (identical to Phase 23/58, applied to the round-3 topology block)''')
code(s58(15))

md(rf'''## Cell 8 — Train ONLY the {ARM_NAME} arm (3 seeds)''')
code(rf'''
BEST_LAMBDA = 0.1
WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)

ORIG_X = [d.x.clone() for d in dataset]
def apply_mask(mask):
    for d, x0 in zip(dataset, ORIG_X):
        t = x0[:, N_PRE:].clone()
        keep = torch.from_numpy(mask)
        t[:, ~keep] = 0.0
        d.x = torch.cat([x0[:, :N_PRE], t], dim=1)

mask = MASKS["{ARM_NAME}"]
apply_mask(mask)

t0 = time.time()
arm_models = {{}}
for seed in (0,1,2):
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    arm_models[seed] = m
    torch.save(m.state_dict(), WORK/f"cache/{ARM_NAME}_scaled3_seed{{seed}}.pt")
    print(f"  [{ARM_NAME}] seed {{seed}}: val PR {{val_pr:.4f}}  "
          f"val DCC {{v.get('dcc',float('nan')):.4f}}  [{{(time.time()-t0)/60:.1f}}m]")

print(f"\ntrained {{len(arm_models)}} models for arm {ARM_NAME}")
''')

md(rf'''
---

`{ARM_NAME}_scaled3_seed{{0,1,2}}.pt` -- feed into the combining/scoring
notebook once all four decomposition arms (H0/H1/H2/burial) are trained,
alongside the already-available `full` (Phase53/56) and `no_topo`
(salvaged from Phase58's partial run, re-uploaded as a dataset) arms.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path(f"/Users/shivenkhurana/bioproj/phase{PHASE_NUM}_{ARM_NAME.lower()}_scale3.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
