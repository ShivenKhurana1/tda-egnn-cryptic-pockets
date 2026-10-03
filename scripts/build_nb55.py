#!/usr/bin/env python3
"""Phase 55: is Phase 50's ESM2-3B CryptoBench result (AUC 0.887, MCC 0.414
-- essentially matching field SOTA's 0.42 on a 3-seed first pass)
seed-stable? Same stress test already applied to the 35M (Phase44) and
650M (Phase47) results, both confirmed stable/improving at 8 seeds. Trains
5 more seeds (8 total) on Phase 50's cached training set, reports per-seed
spread for every metric plus the 8-seed ensemble number."""
import json, pickle
from pathlib import Path

C50 = pickle.load(open("/tmp/p50_cells.pkl", "rb"))
s43 = lambda i: (C50[i]["source"] if isinstance(C50[i]["source"], str) else "".join(C50[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 55 - Is the ESM2-3B CryptoBench Result Seed-Stable?

Phase 50 (3 seeds, ESM2-3B) found AUC 0.887 and MCC 0.414 -- essentially
matching CryptoBench's own published SOTA (0.87 / 0.42), and the best
localization numbers seen on this benchmark (DCC@4A 0.236, median error
6.6A). Phase44/47 already confirmed the 35M and 650M results were
seed-stable (improved, not shrank, at 8 seeds); this notebook applies the
identical stress test to the 3B result -- the last capacity lever tested
in this project. Trains 5 more seeds (8 total) on Phase 50's cached
training set, reports the per-seed spread for every metric, and the
8-seed ensemble number.
''')

md(r'''## Cell 1 - Setup, load Phase 48's ESM2-3B CryptoBench blob''')
code(s43(2))
code(s43(3))

md(r'''## Cell 2 - Model (identical to every prior localization phase)''')
code(s43(5))

md(r'''## Cell 3 - Training machinery (identical to Phase 14/20/43)''')
code(s43(7))

md(r'''## Cell 4 - Load Phase 50's 3 checkpoints, train 5 MORE seeds (8 total)''')
code(r'''
ensemble = {}
for seed in (0,1,2):
    cands = []
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): cands.extend(r.rglob(f"cryptobench_esm3b_deploy_seed{seed}.pt"))
    assert cands, f"cryptobench_esm3b_deploy_seed{seed}.pt not found -- attach Phase 50's kernel output"
    state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
    m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE); m.load_state_dict(state); m.eval()
    ensemble[seed] = m
print(f"loaded {len(ensemble)} checkpoints from Phase 50 (no retraining)")

BEST_LAMBDA = 0.1
t0 = time.time()
for seed in (3,4,5,6,7):
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    ensemble[seed] = m
    torch.save(m.state_dict(), WORK/f"cache/cryptobench_esm3b_deploy_seed{seed}.pt")
    val_dcc = v.get("dcc", float("nan"))
    print(f"  seed {seed}: val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")
print(f"\n{len(ensemble)} total seeds available (0-7)")
''')

md(r'''## Cell 5 - Per-seed and ensemble predictions on val/test''')
code(r'''
@torch.no_grad()
def predict_seed(model, loader):
    """Returns pooled (probs, y, per-chain-DCC, per-chain-predicted-center,
    per-chain-true-center). Centers are kept raw (not yet turned into a
    distance) so an ensemble can average CENTERS across seeds before
    computing DCC -- averaging per-seed DISTANCES instead would be a
    different, wrong statistic (see Phase23/26's "ensemble-of-predictions
    vs mean-of-individual-scores are not the same" lesson)."""
    all_probs, all_y, all_dcc, all_centers, all_true_c = [], [], [], [], []
    for b in loader:
        b = b.to(DEVICE)
        lg, pc = model(b)
        p = torch.sigmoid(lg).cpu().numpy()
        all_probs.append(p); all_y.append(b.y.cpu().numpy())
        true_c = b.true_center.view(-1,3).cpu().numpy()
        pc_np = pc.cpu().numpy()
        d_ = np.linalg.norm(pc_np - true_c, axis=1)
        all_dcc.extend(d_.tolist())
        all_centers.append(pc_np); all_true_c.append(true_c)
    return (np.concatenate(all_probs), np.concatenate(all_y), np.array(all_dcc),
            np.concatenate(all_centers), np.concatenate(all_true_c))

val_loader = DataLoader(splits["val"], batch_size=16)
test_loader = DataLoader(splits["test"], batch_size=16)

val_preds, test_preds = {}, {}
for seed, m in ensemble.items():
    val_preds[seed] = predict_seed(m, val_loader)
    test_preds[seed] = predict_seed(m, test_loader)
print(f"predicted for {len(ensemble)} seeds x (val + test)")
''')

md(r'''## Cell 6 - Per-seed metrics (the noise floor)''')
code(r'''
def metrics_at(p, y, dcc, thresh):
    pred = (p >= thresh).astype(int)
    return {
        "auc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
        "mcc": matthews_corrcoef(y, pred), "f1": f1_score(y, pred),
        "dcc_rate_4A": float((dcc < 4.0).mean()), "mean_error_A": float(dcc.mean()),
    }

# per-seed threshold tuned on that seed's own val predictions (matches Phase43's
# val-only discipline, just repeated per seed instead of once for the ensemble)
per_seed_metrics = {}
for seed in range(8):
    vp, vy, *_ = val_preds[seed]
    best_t, best_mcc = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        mcc = matthews_corrcoef(vy, (vp >= t).astype(int))
        if mcc > best_mcc: best_mcc, best_t = mcc, t
    tp, ty, td, *_ = test_preds[seed]
    per_seed_metrics[seed] = {**metrics_at(tp, ty, td, best_t), "val_thresh": float(best_t)}

print(f"{'seed':<6}{'AUC':>8}{'AUPRC':>8}{'MCC':>8}{'F1':>8}{'DCC@4A':>8}{'thresh':>8}")
print("-"*54)
for seed in range(8):
    m_ = per_seed_metrics[seed]
    print(f"{seed:<6}{m_['auc']:>8.4f}{m_['auprc']:>8.4f}{m_['mcc']:>8.4f}{m_['f1']:>8.4f}"
          f"{m_['dcc_rate_4A']:>8.4f}{m_['val_thresh']:>8.2f}")

for key in ("auc","auprc","mcc","f1","dcc_rate_4A"):
    vals = np.array([per_seed_metrics[s][key] for s in range(8)])
    print(f"{key:<12} mean {vals.mean():.4f}  std {vals.std():.4f}  "
          f"range [{vals.min():.4f}, {vals.max():.4f}]")
''')

md(r'''## Cell 7 - 8-seed ensemble metrics (pooled predictions averaged across seeds)''')
code(r'''
val_p_ens = np.mean([val_preds[s][0] for s in range(8)], axis=0)
val_y_ens = val_preds[0][1]
test_p_ens = np.mean([test_preds[s][0] for s in range(8)], axis=0)
test_y_ens = test_preds[0][1]
# ensemble DCC = distance of the AVERAGED CENTER, not the average of
# per-seed distances (a different, wrong statistic -- see predict_seed's
# docstring)
test_centers_ens = np.mean([test_preds[s][3] for s in range(8)], axis=0)
test_true_c = test_preds[0][4]
test_dcc_ens = np.linalg.norm(test_centers_ens - test_true_c, axis=1)

best_t_ens, best_mcc_ens = 0.5, -1.0
for t in np.arange(0.05, 0.96, 0.01):
    mcc = matthews_corrcoef(val_y_ens, (val_p_ens >= t).astype(int))
    if mcc > best_mcc_ens: best_mcc_ens, best_t_ens = mcc, t

ens_metrics = metrics_at(test_p_ens, test_y_ens, test_dcc_ens, best_t_ens)
print(f"8-seed ensemble (val-tuned threshold {best_t_ens:.2f}):")
for k, v in ens_metrics.items():
    print(f"  {k}: {v:.4f}")

print(f"\nPhase 50's 3-seed number for direct comparison: AUC 0.8869, AUPRC 0.3695, "
      f"MCC 0.4144, F1 0.4389, DCC@4A 0.2363")
print(f"3-seed -> 8-seed AUC delta: {ens_metrics['auc']-0.8869:+.4f}")
print(f"3-seed -> 8-seed DCC@4A delta: {ens_metrics['dcc_rate_4A']-0.2363:+.4f}")

json.dump({"per_seed": per_seed_metrics, "ensemble_8seed": ens_metrics,
           "phase50_3seed": {"auc":0.8869,"auprc":0.3695,"mcc":0.4144,"f1":0.4389,"dcc_rate_4A":0.2363}},
          open(WORK/"phase55_summary.json", "w"), indent=2)
print("\nsaved -> phase55_summary.json")
''')

md(r'''
---

## Reading this

- **AUC/AUPRC/MCC hold or improve with more seeds, spread is tight** --
  Phase 50's result is real and stable; safe to cite as the headline
  CryptoBench finding.
- **Wide per-seed spread, ensemble number close to Phase50's 3-seed** --
  the model works but individual runs are noisy; report the ensemble
  number with the spread stated, not a single run.
- **8-seed number drops meaningfully below Phase50's 3-seed** -- the
  original 3-seed pass got lucky; the honest number is lower and should
  replace it in any writeup.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase55_cryptobench_seed_variance_esm3b.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
