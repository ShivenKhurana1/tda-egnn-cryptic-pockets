#!/usr/bin/env python3
"""Single-seed confirmation of Phase112's promising partial result:
h=128/L=6/dropout=0.1 (deeper than Phase8's default h128/L4) scored val
PR 0.5511 vs the default's 0.5373 at seed 0 -- a real-looking gain
(+0.014 PR, +0.038 DCC). Phase112's full 6-config grid was cancelled by
the 12hr CPU cap after only 2/6 configs (CPU training is far slower than
expected for this pipeline, ~160-370 min/config) -- pivoting to directly
confirming/refuting this ONE promising signal across more seeds rather
than continuing the expensive exhaustive grid.
Usage: python3 build_nb112_seed.py <SEED> <PHASE_NUM>"""
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
# Phase {PHASE_NUM} — h128/L6 Capacity Confirmation: Seed {SEED}

Single-seed slice confirming Phase112's promising partial finding
(h=128/L=6/dr=0.1 beat the h128/L4 default at seed 0: val PR 0.5511 vs
0.5373). Trains seed {SEED} of this same deeper config on the identical
round-3 COACH420 clean training set.
''')

md(r'''## Cell 1 — Setup (loads Phase 52's round-3 35M-ESM blob)''')
code(s20(2))
code(s20(3).replace('locate("phase10_graphs.pt")', 'locate("phase52_graphs.pt")')
           .replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")'))

md(r'''## Cell 2 — Fetch ALL 420 COACH420 structures''')
code(s20(5))

md(r'''## Cell 3 — Clean the TRAINING side''')
code(s20(7))

md(r'''## Cell 4 — Recover true ligand centers''')
code(s20(9))

md(r'''## Cell 5 — Model class''')
code(s20(11))

md(r'''## Cell 6 — Training machinery (run_epoch only; capacity-agnostic)''')
code(s20(15).split("\n\ndef fit(")[0])

md(f'''## Cell 7 — Train h128/L6/dr.1 seed {SEED}''')
code(f'''
def fit_capacity(h, L, dr, lam, seed, epochs=40, patience=8, batch=16):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, TOPO_DIM, h=h, L=L, dr=dr, use_loc_head=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl = DataLoader(splits["train"], batch_size=batch, shuffle=True)
    vl = DataLoader(splits["val"], batch_size=batch)
    best, state, bad = -1, None, 0
    for ep in range(epochs):
        run_epoch(model, tl, opt, lam)
        v = run_epoch(model, vl, lam=lam)
        sch.step()
        if v["pr"] > best:
            best, state, bad = v["pr"], {{k:t.clone() for k,t in model.state_dict().items()}}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval()
    return model, best

WORK2 = Path("/kaggle/working"); (WORK2/"cache").mkdir(parents=True, exist_ok=True)
t0 = time.time()
seed = {SEED}
m, val_pr = fit_capacity(128, 6, 0.1, lam=0.1, seed=seed)
v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=0.1)
torch.save(m.state_dict(), WORK2/f"cache/capacity_h128L6_seed{{seed}}.pt")
val_dcc = v.get("dcc", float("nan"))
print(f"  h128_L6_dr.1 seed {{seed}}: val PR {{val_pr:.4f}}  val DCC {{val_dcc:.4f}}  [{{(time.time()-t0)/60:.1f}}m]")
print(f"\\ncompare: seed0 (Phase112) was val PR 0.5511, val DCC 0.4268; default h128/L4 seed0 was val PR 0.5373, val DCC 0.3885")
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path(f"/Users/shivenkhurana/bioproj/phase{PHASE_NUM}_h128L6_seed{SEED}.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
