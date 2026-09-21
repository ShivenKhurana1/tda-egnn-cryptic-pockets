#!/usr/bin/env python3
"""Phase 112: model-capacity sweep on ROUND-3 SCALE data. Phase8's
original capacity sweep (null result: h=128/L=4/dropout=0.1 tied with
5 other configs) was run on the ORIGINAL 2,056-chain pool -- never
repeated since the training pool grew 3.65x (round 3, 7,181 clean
chains for COACH420). A model "right-sized" to avoid overfitting a small
dataset may now be underfitting a much larger one. Uses the proven 35M
ESM round-3 blob (phase52_graphs.pt) -- NOT the 3B blob (Thread 5 showed
capacity-via-embedding doesn't help; this tests capacity via the GNN
itself, a different lever). Single-seed sweep first (cheap), 3-seed
confirmation of any apparent winner before trusting it (this project's
standing discipline)."""
import json, pickle
from pathlib import Path

C20 = pickle.load(open("/tmp/p20_cells.pkl", "rb"))
s20 = lambda i: (C20[i]["source"] if isinstance(C20[i]["source"], str) else "".join(C20[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 112 — Model-Capacity Sweep at Round-3 Scale

Phase8's capacity sweep (h=128/L=4/dropout=0.1 vs 5 alternatives) found
no benefit from tuning -- but that was on the ORIGINAL 2,056-chain pool.
The training pool has since grown 3.65x (round 3). A model sized to avoid
overfitting 2,056 chains may be underfitting 7,181. Tests GNN capacity
(hidden dim, depth, dropout) as a lever distinct from embedding capacity
(Thread 5, closed null) -- uses the cheap, proven 35M ESM round-3 blob.
''')

md(r'''## Cell 1 — Setup (loads Phase 52's round-3 35M-ESM blob)''')
code(s20(2))
code(s20(3).replace('locate("phase10_graphs.pt")', 'locate("phase52_graphs.pt")')
           .replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")'))

md(r'''## Cell 2 — Fetch ALL 420 COACH420 structures''')
code(s20(5))

md(r'''## Cell 3 — Clean the TRAINING side: exclude any chain homologous to COACH420''')
code(s20(7))

md(r'''## Cell 4 — Recover true ligand centers for the cleaned training set''')
code(s20(9))

md(r'''## Cell 5 — Model class (identical to every prior localization phase)''')
code(s20(11))

md(r'''## Cell 6 — Training machinery (run_epoch is capacity-agnostic, reused as-is; fit() is NOT used here -- replaced by the sweep loop below)''')
code(s20(15).split("\n\ndef fit(")[0])

md(r'''## Cell 7 — Capacity sweep: 6 configs, single seed each (cheap sanity pass)''')
code(r'''
CONFIGS = [
    dict(h=128, L=4, dr=0.1, name="h128_L4_dr.1 (Phase8 default)"),
    dict(h=128, L=6, dr=0.1, name="h128_L6_dr.1 (deeper)"),
    dict(h=256, L=4, dr=0.1, name="h256_L4_dr.1 (wider)"),
    dict(h=256, L=6, dr=0.1, name="h256_L6_dr.1 (wider+deeper)"),
    dict(h=256, L=4, dr=0.2, name="h256_L4_dr.2 (wider+more dropout)"),
    dict(h=384, L=4, dr=0.2, name="h384_L4_dr.2 (widest)"),
]

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
            best, state, bad = v["pr"], {k:t.clone() for k,t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval()
    return model, best

BEST_LAMBDA = 0.1
results = []
t0 = time.time()
for cfg in CONFIGS:
    n_params = None
    m, val_pr = fit_capacity(cfg["h"], cfg["L"], cfg["dr"], BEST_LAMBDA, seed=0)
    n_params = sum(p.numel() for p in m.parameters())
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    val_dcc = v.get("dcc", float("nan"))
    results.append(dict(**cfg, n_params=n_params, val_pr=val_pr, val_dcc=val_dcc))
    print(f"  {cfg['name']:<32} {n_params/1e6:.2f}M params  val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")

print(f"\n--- Capacity sweep results (round-3 scale, {len(dataset):,} training chains) ---")
print(f"{'config':<32}{'params':>10}{'val PR':>10}{'val DCC':>10}")
print("-"*62)
for r in sorted(results, key=lambda r: -r["val_pr"]):
    print(f"{r['name']:<32}{r['n_params']/1e6:>9.2f}M{r['val_pr']:>10.4f}{r['val_dcc']:>10.4f}")

best_cfg = max(results, key=lambda r: r["val_pr"])
print(f"\nbest by val PR: {best_cfg['name']} ({best_cfg['val_pr']:.4f})")
default_pr = next(r["val_pr"] for r in results if r["name"].startswith("h128_L4"))
print(f"default (Phase8 h128/L4/dr.1): {default_pr:.4f}")
print(f"gain over default: {best_cfg['val_pr']-default_pr:+.4f}")

json.dump(results, open(WORK/"phase112_sweep_results.json", "w"), indent=2)
print("\nsaved -> phase112_sweep_results.json")
''')

md(r'''
---

## Reading this

- **Best config clearly beats h128/L4/dr.1 default (val PR gain > ~0.01,
  outside typical single-seed noise)** -- worth a 3-seed confirmation
  pass on the winning config, then a full benchmark rescore, mirroring
  every other lever tested this project.
- **All configs cluster within noise of each other** -- reproduces
  Phase8's null finding even at round-3 scale; GNN capacity is not the
  remaining bottleneck either, narrowing the search further.
- **Larger configs clearly UNDERPERFORM** -- would suggest overfitting
  is creeping back in even at this larger scale, worth noting but not
  pursuing further.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase112_capacity_sweep_scale3.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
