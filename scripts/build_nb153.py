#!/usr/bin/env python3
"""Phase 153: CryptoBench final analysis -- statistical rigor pass.
Two things Phase148 lacked: (1) a chain-level bootstrap to formally test
whether the ensemble's MCC gain over 3B-transfer-alone is real or noise
(currently just eyeballed as "tied"), and (2) a regularized logistic-
regression meta-combiner as a more principled alternative to the 66-point
grid search, which is tuned on only 60 validation chains and plausibly
overfits that small, correlated-by-chain sample. Zero new training --
reuses all 24 existing 8-seed checkpoints (3 scales x 8 seeds)."""
import json
from pathlib import Path

nb130 = json.load(open("/Users/shivenkhurana/bioproj/phase130_cryptobench_combined.ipynb"))
c130 = nb130["cells"]
s130 = lambda i: c130[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 153 - CryptoBench Final Analysis: Bootstrap Significance + Regularized Combiner

Phase148's final ensemble result reported the MCC gain over 3B-transfer
alone as "not clearly distinguishable from noise" using an informal
noise-floor heuristic. This notebook makes that rigorous with a
chain-level bootstrap (residues within a chain are correlated, so
bootstrapping residues directly would understate the true uncertainty --
we resample whole chains instead), and separately tests whether a
regularized logistic-regression meta-combiner (4 parameters, fit by
maximum likelihood) is a more defensible alternative to the 66-point
grid search, which explicitly maximizes over many candidate weightings
using only 60 validation chains -- a search procedure with much higher
effective flexibility than it looks, and correspondingly more prone to
picking up validation-set noise. Zero new training: reuses the existing
24 checkpoints (35M/650M/3B, 8 transfer-learned seeds each).
''')

md(r'''## Cell 1 - Install dependencies''')
code(s130(2))

md(r'''## Cell 2 - Model + setup (identical to Phase130/131/148)''')
code(s130(4))

md(r'''## Cell 3 - Load each scale's blob + 8 checkpoints, predict on VAL and TEST (identical to Phase148)''')
code(r'''
SCALES = {
    "35M":  dict(blob="phase42_cryptobench_graphs.pt",         ckpt_prefix="cryptobench_transfer_seed"),
    "650M": dict(blob="phase45_cryptobench_graphs_esm650m.pt", ckpt_prefix="cryptobench_esm650m_transfer_seed"),
    "3B":   dict(blob="phase48_cryptobench_graphs_esm3b.pt",   ckpt_prefix="cryptobench_esm3b_transfer_seed"),
}

scale_val_probs, scale_val_y, scale_val_keys = {}, {}, {}
scale_test_probs, scale_test_y, scale_test_keys = {}, {}, {}
t0 = time.time()
for scale, cfg in SCALES.items():
    blob = torch.load(locate(cfg["blob"]), map_location="cpu", weights_only=False)
    dataset = blob["dataset"]
    N_PRE = int(blob["n_pre_topo"]); TOPO_DIM = int(blob["topo_dim"])
    val_set = [d for d in dataset if d.split == "val"]
    test_set = [d for d in dataset if d.split == "test"]
    val_loader = DataLoader(val_set, batch_size=16)
    test_loader = DataLoader(test_set, batch_size=16)
    scale_val_keys[scale] = [(d.pdb_id, d.chain_id, int(d.num_nodes)) for d in val_set]
    scale_test_keys[scale] = [(d.pdb_id, d.chain_id, int(d.num_nodes)) for d in test_set]

    val_seed_probs, test_seed_probs = [], []
    for seed in range(8):
        ckpt = locate(f"{cfg['ckpt_prefix']}{seed}.pt")
        m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m.eval()
        vp, vy = predict_seed(m, val_loader)
        tp, ty = predict_seed(m, test_loader)
        val_seed_probs.append(vp); test_seed_probs.append(tp)
        if seed == 0:
            scale_val_y[scale] = vy; scale_test_y[scale] = ty
    scale_val_probs[scale] = np.mean(val_seed_probs, axis=0)
    scale_test_probs[scale] = np.mean(test_seed_probs, axis=0)
    print(f"  {scale:<6} 8-seed val+test predictions ready ({cfg['ckpt_prefix']}*)  [{(time.time()-t0)/60:.1f}m]")

for split_name, keys_dict in [("val", scale_val_keys), ("test", scale_test_keys)]:
    ref = keys_dict["35M"]
    for scale in ("650M", "3B"):
        assert keys_dict[scale] == ref, f"{split_name}: {scale} key order mismatch vs 35M"
print(f"\nalignment verified on both val and test across all 3 scales")

SCALE_NAMES = ["35M", "650M", "3B"]
y_val_ref = scale_val_y["35M"]
y_test_ref = scale_test_y["35M"]
''')

md(r'''## Cell 4 - Candidate combination methods: solo-best, equal-weight, grid-search (2 variants), and a regularized logistic-regression meta-combiner''')
code(r'''
from sklearn.linear_model import LogisticRegression

def metrics_at(p, y, thresh):
    pred = (p >= thresh).astype(int)
    return {"auc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "mcc": matthews_corrcoef(y, pred), "f1": f1_score(y, pred)}

def best_thresh(p, y):
    best_t, best_mcc = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        mcc = matthews_corrcoef(y, (p >= t).astype(int))
        if mcc > best_mcc: best_mcc, best_t = mcc, t
    return best_t, best_mcc

# --- solo 3B-transfer (the strongest single scale, MCC 0.4664 in Phase150) ---
t_3b, _ = best_thresh(scale_val_probs["3B"], y_val_ref)

# --- equal-weight ensemble ---
equal_val = sum(scale_val_probs[s] for s in SCALE_NAMES) / 3.0
equal_test = sum(scale_test_probs[s] for s in SCALE_NAMES) / 3.0
t_equal, _ = best_thresh(equal_val, y_val_ref)

# --- grid search (66-point simplex), val-MCC-optimal and val-AUC-optimal (identical to Phase148) ---
best_w, best_val_mcc, best_val_auc_at_w = None, -1.0, None
grid_results = []
for w1 in np.arange(0.0, 1.01, 0.1):
    for w2 in np.arange(0.0, 1.01 - w1, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-9: continue
        w3 = max(w3, 0.0)
        weights = {"35M": w1, "650M": w2, "3B": w3}
        val_p = sum(weights[s] * scale_val_probs[s] for s in SCALE_NAMES)
        t, val_mcc = best_thresh(val_p, y_val_ref)
        val_auc = roc_auc_score(y_val_ref, val_p)
        grid_results.append(dict(w=weights, val_mcc=val_mcc, val_auc=val_auc, thresh=t))
        if val_mcc > best_val_mcc:
            best_val_mcc, best_w, best_val_auc_at_w = val_mcc, weights, val_auc
best_w_auc, best_val_auc = None, -1.0
for r in grid_results:
    if r["val_auc"] > best_val_auc: best_val_auc, best_w_auc = r["val_auc"], r["w"]

grid_mcc_val = sum(best_w[s] * scale_val_probs[s] for s in SCALE_NAMES)
grid_mcc_test = sum(best_w[s] * scale_test_probs[s] for s in SCALE_NAMES)
t_grid_mcc, _ = best_thresh(grid_mcc_val, y_val_ref)

grid_auc_val = sum(best_w_auc[s] * scale_val_probs[s] for s in SCALE_NAMES)
grid_auc_test = sum(best_w_auc[s] * scale_test_probs[s] for s in SCALE_NAMES)
t_grid_auc, _ = best_thresh(grid_auc_val, y_val_ref)

print(f"grid weights (val-MCC-optimal):  {best_w}")
print(f"grid weights (val-AUC-optimal):  {best_w_auc}")

# --- regularized logistic-regression meta-combiner ---
# Features = logit-transformed per-scale probabilities (numerically stable,
# treats each scale's own calibration as a signal the combiner can weigh
# rather than forcing a convex/simplex-constrained average). Fit by
# maximum likelihood on VAL residues only -- 4 parameters (3 weights + an
# intercept), a far lower-flexibility fit than a 66-point exhaustive grid
# search over the same 60 validation chains.
EPS = 1e-6
def logit(p): return np.log(np.clip(p, EPS, 1-EPS) / np.clip(1-p, EPS, 1-EPS))

X_val = np.stack([logit(scale_val_probs[s]) for s in SCALE_NAMES], axis=1)
X_test = np.stack([logit(scale_test_probs[s]) for s in SCALE_NAMES], axis=1)

lr = LogisticRegression(C=1.0, max_iter=1000)
lr.fit(X_val, y_val_ref)
print(f"LR combiner coefficients: {dict(zip(SCALE_NAMES, lr.coef_[0].round(3)))}  intercept: {lr.intercept_[0]:.3f}")

lr_val_p = lr.predict_proba(X_val)[:, 1]
lr_test_p = lr.predict_proba(X_test)[:, 1]
t_lr, _ = best_thresh(lr_val_p, y_val_ref)

METHODS = {
    "3B-transfer alone":            dict(test_p=scale_test_probs["3B"], thresh=t_3b),
    "equal-weight ensemble":        dict(test_p=equal_test,            thresh=t_equal),
    "grid-search (val-MCC-optimal)":dict(test_p=grid_mcc_test,         thresh=t_grid_mcc),
    "grid-search (val-AUC-optimal)":dict(test_p=grid_auc_test,         thresh=t_grid_auc),
    "LR meta-combiner":             dict(test_p=lr_test_p,             thresh=t_lr),
}

print(f"\n{'method':<32}{'AUC':>8}{'AUPRC':>8}{'MCC':>8}{'F1':>8}")
print("-"*64)
point_estimates = {}
for name, cfg in METHODS.items():
    r = metrics_at(cfg["test_p"], y_test_ref, cfg["thresh"])
    point_estimates[name] = r
    print(f"{name:<32}{r['auc']:>8.4f}{r['auprc']:>8.4f}{r['mcc']:>8.4f}{r['f1']:>8.4f}")
''')

md(r'''## Cell 5 - Chain-level bootstrap: is the ensemble's MCC gain over 3B-transfer-alone real, or test-set noise?''')
code(r'''
# Residues within one chain are correlated (same protein, same structural
# context), so resampling residues directly would understate the true
# sampling uncertainty. Resample whole CHAINS with replacement instead --
# same resampled chain order applied to every method in a given bootstrap
# iteration, so the paired differences between methods stay valid.
chain_lens = [k[2] for k in scale_test_keys["35M"]]
offsets = np.concatenate([[0], np.cumsum(chain_lens)])
n_chains = len(chain_lens)
assert offsets[-1] == len(y_test_ref), "chain lengths don't sum to residue count"

rng = np.random.RandomState(0)
B = 2000
boot_metrics = {name: {"auc": [], "mcc": []} for name in METHODS}
boot_diff_vs_3b = {name: {"auc": [], "mcc": []} for name in METHODS if name != "3B-transfer alone"}

t0 = time.time()
for b in range(B):
    chain_idx = rng.randint(0, n_chains, size=n_chains)
    res_idx = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in chain_idx])
    y_b = y_test_ref[res_idx]
    if y_b.sum() == 0 or y_b.sum() == len(y_b):
        continue  # degenerate resample (no positives or no negatives), skip
    iter_vals = {}
    for name, cfg in METHODS.items():
        p_b = cfg["test_p"][res_idx]
        auc_b = roc_auc_score(y_b, p_b)
        mcc_b = matthews_corrcoef(y_b, (p_b >= cfg["thresh"]).astype(int))
        boot_metrics[name]["auc"].append(auc_b)
        boot_metrics[name]["mcc"].append(mcc_b)
        iter_vals[name] = (auc_b, mcc_b)
    for name in boot_diff_vs_3b:
        boot_diff_vs_3b[name]["auc"].append(iter_vals[name][0] - iter_vals["3B-transfer alone"][0])
        boot_diff_vs_3b[name]["mcc"].append(iter_vals[name][1] - iter_vals["3B-transfer alone"][1])
print(f"bootstrap ({B} resamples, {n_chains} chains) done in {(time.time()-t0)/60:.1f}m, "
      f"{B - len(boot_metrics['3B-transfer alone']['auc'])} degenerate resamples skipped")

def ci(arr):
    a = np.array(arr)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

print(f"\n{'method':<32}{'AUC 95% CI':>22}{'MCC 95% CI':>22}")
print("-"*76)
ci_results = {}
for name in METHODS:
    auc_ci = ci(boot_metrics[name]["auc"]); mcc_ci = ci(boot_metrics[name]["mcc"])
    ci_results[name] = {"auc_ci": auc_ci, "mcc_ci": mcc_ci}
    print(f"{name:<32}[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]  [{mcc_ci[0]:.4f}, {mcc_ci[1]:.4f}]")

print(f"\n{'method vs 3B-transfer alone':<32}{'AUC diff 95% CI':>26}{'MCC diff 95% CI':>26}{'sig?':>8}")
print("-"*100)
diff_results = {}
for name in boot_diff_vs_3b:
    auc_d_ci = ci(boot_diff_vs_3b[name]["auc"]); mcc_d_ci = ci(boot_diff_vs_3b[name]["mcc"])
    mcc_sig = "YES" if (mcc_d_ci[0] > 0 or mcc_d_ci[1] < 0) else "no"
    diff_results[name] = {"auc_diff_ci": auc_d_ci, "mcc_diff_ci": mcc_d_ci, "mcc_significant": mcc_sig == "YES"}
    print(f"{name:<32}[{auc_d_ci[0]:+.4f}, {auc_d_ci[1]:+.4f}]  [{mcc_d_ci[0]:+.4f}, {mcc_d_ci[1]:+.4f}]{mcc_sig:>8}")

print(f"\n('sig?' = does the 95% CI of the MCC DIFFERENCE exclude zero -- ")
print(f" YES means that method's MCC gain over 3B-transfer-alone is statistically real at p<0.05,")
print(f" no means the apparent gain or loss is not distinguishable from chain-resampling noise)")

json.dump({
    "point_estimates": point_estimates,
    "confidence_intervals": ci_results,
    "diff_vs_3b_alone": diff_results,
    "lr_combiner_coefficients": dict(zip(SCALE_NAMES, lr.coef_[0].tolist())),
    "lr_combiner_intercept": float(lr.intercept_[0]),
    "grid_weights_mcc_optimal": best_w,
    "grid_weights_auc_optimal": best_w_auc,
    "n_bootstrap": B, "n_chains": n_chains,
}, open("/kaggle/working/phase153_final_analysis.json", "w"), indent=2)
print("\nsaved -> phase153_final_analysis.json")
''')

md(r'''
---

## Reading this

- **LR combiner beats grid search AND has a tighter, more defensible
  rationale** -- adopt it as the final recommended combination method.
- **LR combiner's MCC gain over 3B-transfer-alone is bootstrap-
  significant** -- the "tie" from Phase148 was a grid-search artifact,
  not a property of ensembling itself; report this as the real final
  result.
- **All ensemble variants remain statistically tied with 3B-transfer
  alone on MCC** -- report the honest conclusion that ensembling helps
  AUC but not demonstrably MCC on this benchmark's test set, and that
  3B-transfer alone is a legitimate, simpler choice.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase153_cryptobench_final_analysis.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
