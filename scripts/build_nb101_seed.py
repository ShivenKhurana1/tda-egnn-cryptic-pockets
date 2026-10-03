#!/usr/bin/env python3
"""Generates one single-seed COACH420 ESM2-3B training notebook (extends
Phase97's 0-2 to the full 8-seed ensemble). Single-seed granularity
because Phase99's timing (168/189/196 min/seed on CPU, ~17x GPU's 11
min/seed) means Phase101's original all-5-seeds-in-one-notebook design
risks exceeding the 12hr CPU session cap (5 seeds * ~200min ~= 16.7hr).
Mirrors the single-seed-per-notebook pattern Thread 1 already proved out
for exactly this kind of CPU-session-limit risk.
Usage: python3 build_nb101_seed.py <SEED> <PHASE_NUM>"""
import json, pickle, sys
from pathlib import Path

SEED = int(sys.argv[1])
PHASE_NUM = sys.argv[2]

C20 = pickle.load(open("/tmp/p20_cells.pkl", "rb"))
s20 = lambda i: (C20[i]["source"] if isinstance(C20[i]["source"], str) else "".join(C20[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(f'''
# Phase {PHASE_NUM} - COACH420 ESM2-3B 8-Seed Extension: Seed {SEED}

Single-seed slice of Phase101 (which risked exceeding the 12hr CPU
session cap training all 5 remaining seeds in one notebook). Identical
methodology to Phase97/101 (fp16-resident dataset, per-batch fp32
upcast) but trains only seed {SEED}.
''')

md(r'''## Cell 1 - Setup (loads Phase 96's ESM2-3B blob, kept in fp16)''')
code(s20(2))
code(s20(3).replace('locate("phase10_graphs.pt")', 'locate("phase96_graphs_esm3b.pt")')
           .replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")'))

md(r'''## Cell 2 - Fetch ALL 420 COACH420 structures''')
code(s20(5))

md(r'''## Cell 3 - Clean the TRAINING side''')
code(s20(7))

md(r'''## Cell 4 - Recover true ligand centers''')
code(s20(9))

md(r'''## Cell 5 - Model''')
code(s20(11))

md(r'''## Cell 6 - Equivariance check (probe cast to fp32)''')
code(s20(13).replace('probe = dataset[0].clone()', 'probe = dataset[0].clone(); probe.x = probe.x.float()'))

md(r'''## Cell 7 - Training machinery (per-batch fp32 upcast)''')
code(s20(15).replace('    for b in loader:\n        b = b.to(DEVICE)',
                      '    for b in loader:\n        b = b.to(DEVICE); b.x = b.x.float()   # fp16-on-disk fix: upcast per-batch'))

md(f'''## Cell 8 - Train ONLY seed {SEED}''')
code(f'''
BEST_LAMBDA = 0.1
WORK2 = Path("/kaggle/working"); (WORK2/"cache").mkdir(parents=True, exist_ok=True)
t0 = time.time()
seed = {SEED}
m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
torch.save(m.state_dict(), WORK2/f"cache/clean_deploy_esm3b_seed{{seed}}.pt")
val_dcc = v.get("dcc", float("nan"))
print(f"  seed {{seed}}: val PR {{val_pr:.4f}}  val DCC {{val_dcc:.4f}}  [{{(time.time()-t0)/60:.1f}}m]")
print(f"\\ntrained seed {SEED} on {{len(dataset):,}} chains (zero homology to COACH420, by construction)")
''')

md(f'''
---

`clean_deploy_esm3b_seed{SEED}.pt` -- one of 5 new seeds (3-7) needed to
extend the COACH420 ESM2-3B ensemble to 8 total, alongside Phase97's
seeds 0-2.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path(f"/Users/shivenkhurana/bioproj/phase{PHASE_NUM}_coach420_esm3b_seed{SEED}.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
