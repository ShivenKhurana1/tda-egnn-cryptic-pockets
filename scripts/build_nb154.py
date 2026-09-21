#!/usr/bin/env python3
"""Phase 154: CryptoBench-specific GNN depth sweep, 650M scale, FROM
SCRATCH (not transfer, to keep it a clean architecture comparison,
matching how the main-pipeline's own depth sweep was done). The main
pipeline's h128L6 vs h128L4 sweep came back null at 8 seeds because the
main pipeline had already saturated its available training-data scale;
CryptoBench's smaller, harder task may have headroom the main pipeline
didn't. Trains L=6 (deeper) directly against Phase47's existing 8-seed
h128L4 baseline (MCC 0.3986) for a paired-by-seed comparison."""
import json
from pathlib import Path

nb46 = json.load(open("/Users/shivenkhurana/bioproj/phase46_cryptobench_train_eval_esm650m.ipynb"))
c46 = nb46["cells"]
s46 = lambda i: (c46[i]["source"] if isinstance(c46[i]["source"], str) else "".join(c46[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 154 — CryptoBench GNN Depth Sweep (650M, From Scratch, 8 Seeds)

Tests whether a deeper network (h128 L6, +2 EGCL/Fus layers vs the
default h128 L4) helps CryptoBench specifically. The main pipeline's own
depth sweep (Thread 6, this project) came back null at 8 seeds because
the main pipeline had already saturated its available training-data
scale and GNN capacity; CryptoBench's smaller (700 chains), harder
(cryptic pockets) task may have headroom the main pipeline didn't.
Trained from scratch (not transfer) to keep this a clean architecture
comparison against Phase47's existing 8-seed h128L4 baseline
(MCC 0.3986, AUC 0.8873), same blob, same protocol, only L changed.
''')

md(r'''## Cell 1 — Install dependencies''')
code(s46(2))

md(r'''## Cell 2 — Setup, load Phase 45's CryptoBench-650M blob (identical to Phase46/47)''')
code(s46(3))

md(r'''## Cell 3 — Model, with L exposed as a parameter (identical architecture to every prior phase, just configurable depth)''')
code(s46(5))

md(r'''## Cell 4 — Training machinery, fit() now takes L (identical to Phase46 otherwise)''')
code(r'''
def run_epoch(model, loader, opt=None, lam=0.0):
    tr = opt is not None; model.train(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    huber = nn.SmoothL1Loss(beta=4.0)
    ps, ys, cd = [], [], []
    for b in loader:
        b = b.to(DEVICE)
        with torch.set_grad_enabled(tr):
            logits, pred_c = model(b)
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


def fit(use_loc_head, lam, seed, L=4, epochs=40, patience=8, batch=16):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head, L=L).to(DEVICE)
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

md(r'''## Cell 5 — Train the h128L6 (deeper) ensemble, 8 seeds, from scratch''')
code(r'''
BEST_LAMBDA = 0.1
ensemble = []
results = []
t0 = time.time()
for seed in range(8):
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, L=6)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    ensemble.append(m)
    torch.save(m.state_dict(), WORK/f"cache/cryptobench_esm650m_deepL6_seed{seed}.pt")
    val_dcc = v.get("dcc", float("nan"))
    results.append(dict(seed=seed, val_pr=val_pr, val_dcc=val_dcc))
    print(f"  L6 seed {seed}: val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")

all_pr = [r["val_pr"] for r in results]
print(f"\n--- h128L6 (deeper), 8-seed summary ---")
print(f"val PR:  mean {np.mean(all_pr):.4f}  std {np.std(all_pr):.4f}  values {[round(x,4) for x in all_pr]}")
print(f"reference, Phase46/47's h128L4 (default) 8-seed val PR: see Phase47 log (not re-derived here)")
json.dump({"seeds": results, "all_pr": all_pr},
          open(WORK/"phase154_depthL6_8seed_results.json", "w"), indent=2)
print(f"ensemble of {len(ensemble)} trained on {len(splits['train']):,} cryptic-pocket chains")
''')

md(r'''## Cell 6 — Pick a decision threshold on VAL ONLY (never touch test for this), then evaluate on TEST''')
code(s46(11))

md(r'''## Cell 7 — Field-standard residue classification metrics on TEST (pooled)''')
code(s46(13))

md(r'''## Cell 8 — Our own localization-head metric''')
code(s46(15).replace('phase43_summary.json', 'phase154_depthL6_summary.json')
            .replace('saved -> phase43_summary.json', 'saved -> phase154_depthL6_summary.json'))

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase154_cryptobench_depth_sweep.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
