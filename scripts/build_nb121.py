#!/usr/bin/env python3
"""Phase 121: CryptoBench topology ablation -- does persistent homology
help CRYPTIC-pocket localization/classification the same way Thread 1
proved it helps the main COACH420/HOLO4K pipeline's localization? Never
tested on CryptoBench specifically. Reuses Phase117's blob (topology
already baked in, N_PRE/TOPO_DIM stored dynamically) -- trains WITH
topology (full, matches Phase119's baseline exactly) first while data is
pristine, then zeroes the topology columns (d.x[:, N_PRE:] = 0, same
technique as Thread 1's arm-masking) for a no_topo arm, same seed=0/split,
same run for direct comparison."""
import json
from pathlib import Path

nb118 = json.load(open("/Users/shivenkhurana/bioproj/phase118_cryptobench_train_aux_sweep.ipynb"))
c118 = nb118["cells"]
s118 = lambda i: c118[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 121 — CryptoBench: Does Topology Help Cryptic-Pocket Prediction?

Thread 1 proved persistent homology significantly improves localization
on the main COACH420/HOLO4K pipeline (H1 > H0, topology overall beats
no-topology). Never tested on CryptoBench (a structurally different,
harder problem -- cryptic pockets are apo-INVISIBLE by definition, so
static-structure topology might carry less signal, or might matter MORE
since it captures cavity/void geometry that residue identity alone
can't). Trains full (with topology) and no_topo (topology zeroed) in the
SAME run, seed=0, for a direct comparison.
''')

md(r'''## Cell 1 — Install dependencies''')
code(s118(2))

md(r'''## Cell 2 — Setup, load Phase 117's blob (identical to Phase118/119)''')
code(s118(3))

md(r'''## Cell 3 — Model (identical to Phase118/119)''')
code(s118(5))

md(r'''## Cell 4 — Training machinery (identical to Phase119, no aux head, no masking -- this is Phase119's exact baseline arm, reused as the "full" topology arm here)''')
code(r'''
def run_epoch(model, loader, opt=None, lam=0.0):
    tr = opt is not None; model.train(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    huber = nn.SmoothL1Loss(beta=4.0)
    ps, ys, cd = [], [], []
    for b in loader:
        b = b.to(DEVICE)
        with torch.set_grad_enabled(tr):
            logits, pred_c, _ = model(b)
            loss = crit(logits, b.y)
            if lam > 0 and pred_c is not None:
                true_c = b.true_center.view(-1, 3)
                loss = loss + lam * huber(pred_c, true_c)
                cd.append((pred_c - true_c).norm(dim=1).detach().cpu().numpy())
        if tr:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        ps.append(torch.sigmoid(logits).detach().cpu().numpy()); ys.append(b.y.cpu().numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)
    out = {"pr": average_precision_score(y, p), "roc": roc_auc_score(y, p)}
    if cd:
        d_ = np.concatenate(cd)
        out["dcc"] = float((d_ < 4.0).mean()); out["mean_center_err"] = float(d_.mean())
    return out


def fit(use_loc_head, lam, seed, epochs=40, patience=8, batch=16):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head, use_aux_head=False).to(DEVICE)
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
''')

md(r'''## Cell 5 — Train "full" (with topology) first while data is pristine, then zero d.x[:, N_PRE:] for "no_topo", same seed=0, same run''')
code(r'''
BEST_LAMBDA = 0.1
results = []
t0 = time.time()

m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=0)
v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
torch.save(m.state_dict(), WORK/"cache/cryptobench_full_seed0.pt")
results.append(dict(arm="full (with topology)", val_pr=val_pr, val_dcc=v.get("dcc", float("nan"))))
print(f"  full (with topology)   seed0: val PR {val_pr:.4f}  val DCC {v.get('dcc', float('nan')):.4f}  [{(time.time()-t0)/60:.1f}m]")

for d in dataset:
    d.x[:, N_PRE:] = 0.0   # zero the topology block -- same technique as Thread 1's arm-masking
m2, val_pr2 = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=0)
v2 = run_epoch(m2, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
torch.save(m2.state_dict(), WORK/"cache/cryptobench_notopo_seed0.pt")
results.append(dict(arm="no_topo (topology zeroed)", val_pr=val_pr2, val_dcc=v2.get("dcc", float("nan"))))
print(f"  no_topo (zeroed)       seed0: val PR {val_pr2:.4f}  val DCC {v2.get('dcc', float('nan')):.4f}  [{(time.time()-t0)/60:.1f}m]")

print(f"\n--- topology ablation (single seed=0, same run) ---")
full_pr = results[0]["val_pr"]
for r in results:
    print(f"  {r['arm']:<30} val PR {r['val_pr']:.4f}  ({r['val_pr']-full_pr:+.4f} vs full)  val DCC {r['val_dcc']:.4f}")
json.dump(results, open(WORK/"phase121_topo_ablation_results.json", "w"), indent=2)
print("\nsaved -> phase121_topo_ablation_results.json")
''')

md(r'''
---

## Reading this

- **full clearly beats no_topo** (gap outside typical single-seed noise,
  ~>0.01-0.015 val PR) -- topology's benefit generalizes to cryptic
  pockets too, a strong finding worth an 8-seed confirmation and
  reporting in any writeup.
- **Close/tied** -- topology's contribution is specific to the main
  pipeline's task/data scale, doesn't clearly transfer to this harder,
  smaller-data problem.
- **no_topo beats full** -- would be a genuinely surprising reversal,
  worth real scrutiny (e.g. is topology just adding noise on 700 chains
  when 5 dims might already be plenty of capacity for the trunk).
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase121_cryptobench_topo_ablation.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
