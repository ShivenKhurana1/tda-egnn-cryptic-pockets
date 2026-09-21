#!/usr/bin/env python3
"""Phase 160: does the ANM mean-square-fluctuation (predicted flexibility)
feature help CryptoBench classification? Trains BOTH arms from scratch
(the wider N_PRE is shape-incompatible with existing pretrained
checkpoints, same reasoning as Phase158): full=MSF column intact,
no_msf=zero just that one column. 8 seeds each, paired by seed."""
import json
from pathlib import Path

nb43 = json.load(open("/Users/shivenkhurana/bioproj/phase43_cryptobench_train_eval.ipynb"))
c43 = nb43["cells"]
s43 = lambda i: (c43[i]["source"] if isinstance(c43[i]["source"], str) else "".join(c43[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 160 — ANM Flexibility Feature Ablation (8 seeds each arm)

Tests whether Phase159's new ANM mean-square-fluctuation (predicted
flexibility) feature helps CryptoBench classification. Trained from
scratch (the wider N_PRE is shape-incompatible with existing
pretrained checkpoints, same reasoning as Phase158). Two arms, 8 seeds
each, paired by seed:
- **full**: the MSF column intact.
- **no_msf**: that one column zeroed, everything else intact.
''')

md(r'''## Cell 1 — Install dependencies (identical to Phase43)''')
code(s43(2))

md(r'''## Cell 2 — Setup, load Phase 159's ANM-MSF CryptoBench blob''')
code(s43(3).replace('phase42_cryptobench_graphs.pt', 'phase159_cryptobench_anmmsf_graphs.pt')
           .replace('N_PRE = int(blob["n_pre_topo"]); TOPO_DIM = int(blob["topo_dim"]); POS_WEIGHT = float(blob["pos_weight"])',
                     'N_PRE = int(blob["n_pre_topo"]); TOPO_DIM = int(blob["topo_dim"]); POS_WEIGHT = float(blob["pos_weight"])\n'
                     'MSF_COL_IDX = int(blob["msf_col_idx"])'))

md(r'''## Cell 3 — Model (identical to every prior localization phase)''')
code(s43(5))

md(r'''## Cell 4 — Training machinery (identical to Phase43)''')
code(s43(7))

md(r'''## Cell 5 — Train both arms, 8 seeds each, paired by seed''')
code(r'''
BEST_LAMBDA = 0.1

def zero_msf(ds):
    import copy
    out = []
    for d in ds:
        d2 = copy.copy(d)
        d2.x = d.x.clone()
        d2.x[:, MSF_COL_IDX] = 0.0
        out.append(d2)
    return out

splits_nomsf = {s: zero_msf(v) for s, v in splits.items()}

results = {"full": [], "no_msf": []}
t0 = time.time()
for arm, use_splits in [("full", splits), ("no_msf", splits_nomsf)]:
    tl_base = use_splits["train"]; vl_base = use_splits["val"]
    for seed in range(8):
        torch.manual_seed(seed); np.random.seed(seed)
        model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=True).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
        tl = DataLoader(tl_base, batch_size=16, shuffle=True)
        vl = DataLoader(vl_base, batch_size=16)
        best, state, bad, patience = -1, None, 0, 8
        for ep in range(40):
            run_epoch(model, tl, opt, BEST_LAMBDA)
            v = run_epoch(model, vl, lam=BEST_LAMBDA)
            sch.step()
            if v["pr"] > best:
                best, state, bad = v["pr"], {k: t.clone() for k, t in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience: break
        model.load_state_dict(state); model.eval()
        v_final = run_epoch(model, DataLoader(vl_base, batch_size=16), lam=BEST_LAMBDA)
        val_dcc = v_final.get("dcc", float("nan"))
        torch.save(model.state_dict(), WORK / f"cache/cryptobench_anmmsf_{arm}_seed{seed}.pt")
        results[arm].append(dict(seed=seed, val_pr=best, val_dcc=val_dcc))
        print(f"  {arm:<8} seed {seed}: val PR {best:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")

for arm in ("full", "no_msf"):
    prs = [r["val_pr"] for r in results[arm]]
    print(f"\n--- {arm} arm, 8-seed summary ---")
    print(f"val PR: mean {np.mean(prs):.4f}  std {np.std(prs):.4f}  values {[round(x,4) for x in prs]}")

json.dump(results, open(WORK / "phase160_anmmsf_ablation_results.json", "w"), indent=2)
print("\nsaved -> phase160_anmmsf_ablation_results.json")
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase160_cryptobench_anmmsf_ablation.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
