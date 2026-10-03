#!/usr/bin/env python3
"""Generates the two 'finish the last seed' notebooks for H2_only (Phase64)
and burial_only (Phase65) -- both got cancelled by Kaggle's 12hr CPU limit
after finishing seeds 0-1 (out of 3), likely due to resource contention
from running 4 CPU sessions concurrently (Phase59/60 finished cleanly,
Phase61/62 didn't). These load the salvaged seed 0-1 checkpoints and train
ONLY seed 2, which should take ~1/3 the time and fit comfortably."""
import json, pickle, sys
from pathlib import Path

C58 = pickle.load(open("/tmp/p58_cells.pkl", "rb"))
s58 = lambda i: (C58[i]["source"] if isinstance(C58[i]["source"], str) else "".join(C58[i]["source"]))

def build(arm_name, phase_num, salvage_dataset_slug):
    C=[]
    md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
    code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

    md(f'''
# Phase {phase_num} - Thread 1 Round-3, Finish {arm_name} (Seed 2 Only)

`{arm_name}` got cancelled by Kaggle's 12hr CPU session limit after
finishing seeds 0-1 (of 3), likely from resource contention running 4 CPU
sessions concurrently (Phase59/60 finished cleanly at the same time,
Phase61/62 didn't). This trains ONLY the missing seed 2 -- should take
~1/3 the time of the original attempt and fit comfortably in one session.
''')

    md('''## Cell 1 - Setup (loads Phase 52/51's round-3 scaled blobs, identical to Phase53/56/58-62)''')
    code(s58(2))
    code(s58(3))

    md('''## Cell 2 - Fetch ALL 420 COACH420 structures (identical to Phase 53/56/58-62)''')
    code(s58(5))

    md('''## Cell 3 - Clean the TRAINING side: exclude any chain homologous to COACH420 (identical to Phase 53/56/58-62; reproduces the same 7,181 chains)''')
    code(s58(7))

    md('''## Cell 4 - Recover true ligand centers for the cleaned training set (identical to Phase 53/56/58-62)''')
    code(s58(9))

    md('''## Cell 5 - Model (identical to every prior localization phase)''')
    code(s58(11))

    md('''## Cell 6 - Training machinery (identical to Phase 14/20/53/56/58-62)''')
    code(s58(13))

    md('''## Cell 7 - Build H0 / H1 / H2 / burial column masks (identical to Phase 23/58-62)''')
    code(s58(15))

    md(f'''## Cell 8 - Train ONLY seed 2 for {arm_name} (seeds 0-1 salvaged from the cancelled run)''')
    code(f'''
BEST_LAMBDA = 0.1
WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)

ORIG_X = [d.x.clone() for d in dataset]
def apply_mask(mask):
    for d, x0 in zip(dataset, ORIG_X):
        t = x0[:, N_PRE:].clone()
        keep = torch.from_numpy(mask)
        t[:, ~keep] = 0.0
        d.x = torch.cat([x0[:, :N_PRE], t], dim=1)

mask = MASKS["{arm_name}"]
apply_mask(mask)

t0 = time.time()
seed = 2
m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
torch.save(m.state_dict(), WORK/f"cache/{arm_name}_scaled3_seed{{seed}}.pt")
print(f"  [{arm_name}] seed {{seed}}: val PR {{val_pr:.4f}}  "
      f"val DCC {{v.get('dcc',float('nan')):.4f}}  [{{(time.time()-t0)/60:.1f}}m]")
print(f"\\ntrained seed 2 for arm {arm_name}")
''')

    md(f'''
---

`{arm_name}_scaled3_seed2.pt` -- combined with the salvaged seeds 0-1
(dataset `{salvage_dataset_slug}`), this completes all 3 seeds for
`{arm_name}`, feeding into the combining/scoring notebook (Phase63)
alongside the other five arms.
''')

    nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
        "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
        "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
    o=Path(f"/Users/shivenkhurana/bioproj/phase{phase_num}_{arm_name.lower()}_seed2_scale3.ipynb")
    o.write_text(json.dumps(nb,indent=1)+"\n")
    print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")

build("H2_only", 64, "tda-egnn-phase61-h2only-scaled3-seeds01")
build("burial_only", 65, "tda-egnn-phase62-burial-scaled3-seeds01")
