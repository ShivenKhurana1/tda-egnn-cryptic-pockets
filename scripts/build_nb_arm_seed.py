#!/usr/bin/env python3
"""Generates one single-seed training notebook for Thread 1's round-3
8-seed extension. Usage: python3 build_nb_arm_seed.py <ARM_NAME> <SEED> <PHASE_NUM>
Single-seed granularity chosen after Phase61/62 got cut off training 3
seeds concurrently with 2 other notebooks (contention slowed them ~1.5x) --
1 seed per notebook keeps a large safety margin under the 12hr CPU cap
even under similar contention."""
import json, pickle, sys
from pathlib import Path

ARM_NAME = sys.argv[1]
SEED = int(sys.argv[2])
PHASE_NUM = sys.argv[3]

C58 = pickle.load(open("/tmp/p58_cells.pkl", "rb"))
s58 = lambda i: (C58[i]["source"] if isinstance(C58[i]["source"], str) else "".join(C58[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(f'''
# Phase {PHASE_NUM} - Thread 1 Round-3, 8-Seed Extension: {ARM_NAME} Seed {SEED}

Extends {ARM_NAME} from 3 to 8 seeds (training seed {SEED} of the 5 new
ones: 3-7) to confirm/refute Phase63's preliminary 3-seed finding (H0_only
significantly weaker than H1_only, ratio 2.20x -- but H1 didn't separate
from H2 or burial, and 3 seeds is much less statistical power than the
8-seed standard this project otherwise uses). Single-seed granularity for
safety under the 12hr CPU session cap, given contention issues observed
running multiple 3-seed arms concurrently.
''')

md(r'''## Cell 1 - Setup (loads Phase 52/51's round-3 scaled blobs, identical to Phase53/56/58-65)''')
code(s58(2))
code(s58(3))

md(r'''## Cell 2 - Fetch ALL 420 COACH420 structures (identical to Phase 53/56/58-65)''')
code(s58(5))

md(r'''## Cell 3 - Clean the TRAINING side: exclude any chain homologous to COACH420 (identical to Phase 53/56/58-65; reproduces the same 7,181 chains)''')
code(s58(7))

md(r'''## Cell 4 - Recover true ligand centers for the cleaned training set (identical to Phase 53/56/58-65)''')
code(s58(9))

md(r'''## Cell 5 - Model (identical to every prior localization phase)''')
code(s58(11))

md(r'''## Cell 6 - Training machinery (identical to Phase 14/20/53/56/58-65)''')
code(s58(13))

md(r'''## Cell 7 - Build H0 / H1 / H2 / burial column masks (identical to Phase 23/58-65)''')
code(s58(15))

md(f'''## Cell 8 - Train ONLY seed {SEED} for {ARM_NAME}''')
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

mask = np.zeros(len(TOPO_NAMES), dtype=bool) if "{ARM_NAME}" == "no_topo" else MASKS["{ARM_NAME}"]
apply_mask(mask)

t0 = time.time()
seed = {SEED}
m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
torch.save(m.state_dict(), WORK/f"cache/{ARM_NAME}_scaled3_seed{{seed}}.pt")
print(f"  [{ARM_NAME}] seed {{seed}}: val PR {{val_pr:.4f}}  "
      f"val DCC {{v.get('dcc',float('nan')):.4f}}  [{{(time.time()-t0)/60:.1f}}m]")
print(f"\\ntrained seed {SEED} for arm {ARM_NAME}")
''')

md(f'''
---

`{ARM_NAME}_scaled3_seed{SEED}.pt` -- one of five new seeds (3-7) needed to
extend {ARM_NAME} to 8 total, feeding into a follow-up combining/scoring
notebook once all four arms have all 8 seeds.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path(f"/Users/shivenkhurana/bioproj/phase{PHASE_NUM}_{ARM_NAME.lower()}_seed{SEED}_scale3.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
