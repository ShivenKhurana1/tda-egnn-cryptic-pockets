#!/usr/bin/env python3
"""Phase 156: CryptoBench prediction-graph export. Applies the final
recommended model (LR meta-combiner, coefficients from Phase153) to the
full TEST set and exports: (1) ROC/PR curve coordinates + raw pooled
(prob,label) arrays for a probability-distribution histogram, (2) a
per-chain diagnostic table (AUC/MCC/hit-rate where computable, positive
count) to pick a good example structure for a real predicted-vs-true
pocket visualization, (3) full per-residue predictions (resSeq, prob,
true label) for EVERY test chain, so the chosen example's structure can
be rendered locally without a second Kaggle round-trip."""
import json
from pathlib import Path

nb148 = json.load(open("/Users/shivenkhurana/bioproj/phase148_cryptobench_weighted_ensemble_alltransfer.ipynb"))
c148 = nb148["cells"]
s148 = lambda i: c148[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 156 — CryptoBench Prediction-Graph Export

Applies the final recommended model (LR meta-combiner across the 3
transfer-learned scales, coefficients from Phase153) to the full TEST
set and exports everything needed to build real prediction graphs:
ROC/PR curves, a probability-distribution histogram, and per-chain +
per-residue predictions for picking and rendering a real example
structure's predicted-vs-true cryptic pocket.
''')

md(r'''## Cell 1 — Install dependencies''')
code(s148(2))

md(r'''## Cell 2 — Model + setup (identical to Phase130/131/148)''')
code(s148(4))

md(r'''## Cell 3 — Load each scale's blob + 8 checkpoints, predict on TEST (identical to Phase148)''')
code(r'''
SCALES = {
    "35M":  dict(blob="phase42_cryptobench_graphs.pt",         ckpt_prefix="cryptobench_transfer_seed"),
    "650M": dict(blob="phase45_cryptobench_graphs_esm650m.pt", ckpt_prefix="cryptobench_esm650m_transfer_seed"),
    "3B":   dict(blob="phase48_cryptobench_graphs_esm3b.pt",   ckpt_prefix="cryptobench_esm3b_transfer_seed"),
}

scale_test_probs, scale_test_y, scale_test_keys = {}, {}, {}
t0 = time.time()
for scale, cfg in SCALES.items():
    blob = torch.load(locate(cfg["blob"]), map_location="cpu", weights_only=False)
    dataset = blob["dataset"]
    N_PRE = int(blob["n_pre_topo"]); TOPO_DIM = int(blob["topo_dim"])
    test_set = [d for d in dataset if d.split == "test"]
    test_loader = DataLoader(test_set, batch_size=16)
    scale_test_keys[scale] = [(d.pdb_id, d.chain_id, int(d.num_nodes)) for d in test_set]

    test_seed_probs = []
    for seed in range(8):
        ckpt = locate(f"{cfg['ckpt_prefix']}{seed}.pt")
        m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m.eval()
        tp, ty = predict_seed(m, test_loader)
        test_seed_probs.append(tp)
        if seed == 0:
            scale_test_y[scale] = ty
    scale_test_probs[scale] = np.mean(test_seed_probs, axis=0)
    print(f"  {scale:<6} 8-seed test predictions ready  [{(time.time()-t0)/60:.1f}m]")

ref = scale_test_keys["35M"]
for scale in ("650M", "3B"):
    assert scale_test_keys[scale] == ref, f"{scale} key order mismatch vs 35M"
y_test_ref = scale_test_y["35M"]
print(f"\nalignment verified across all 3 scales, {len(y_test_ref):,} residues, {len(ref):,} chains")
''')

md(r'''## Cell 4 — Apply the LR meta-combiner (Phase153's fitted coefficients) to get final probabilities''')
code(r'''
# Coefficients from Phase153's fit (logistic regression on logit-transformed
# per-scale probabilities, fit on VAL, this notebook only re-applies it --
# no re-fitting, so these are exactly the numbers reported in the paper).
LR_COEF = {"35M": 0.20230552393124115, "650M": 0.3401849963262731, "3B": 0.5880724521257392}
LR_INTERCEPT = -2.8981747736592864
SCALE_NAMES = ["35M", "650M", "3B"]

EPS = 1e-6
def logit(p): return np.log(np.clip(p, EPS, 1-EPS) / np.clip(1-p, EPS, 1-EPS))
def sigmoid(z): return 1 / (1 + np.exp(-z))

z = LR_INTERCEPT + sum(LR_COEF[s] * logit(scale_test_probs[s]) for s in SCALE_NAMES)
final_p = sigmoid(z)

from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
print(f"sanity check vs Phase153's reported point estimate (AUC 0.9177, MCC~0.470):")
print(f"  recomputed AUC: {roc_auc_score(y_test_ref, final_p):.4f}")
best_t, best_mcc = 0.5, -1
for t in np.arange(0.05, 0.96, 0.01):
    mcc = matthews_corrcoef(y_test_ref, (final_p >= t).astype(int))
    if mcc > best_mcc: best_mcc, best_t = mcc, t
print(f"  recomputed best-threshold MCC: {best_mcc:.4f} (threshold {best_t:.2f})")
''')

md(r'''## Cell 5 — Export ROC/PR curves + probability histogram data (pooled over all test residues)''')
code(r'''
from sklearn.metrics import roc_curve, precision_recall_curve

fpr, tpr, roc_thresh = roc_curve(y_test_ref, final_p)
prec, rec, pr_thresh = precision_recall_curve(y_test_ref, final_p)

# Downsample curves to <=300 points each for a manageable JSON/plot (keep endpoints)
def downsample(x, y, n=300):
    if len(x) <= n: return x.tolist(), y.tolist()
    idx = np.linspace(0, len(x)-1, n).astype(int)
    return x[idx].tolist(), y[idx].tolist()

fpr_ds, tpr_ds = downsample(fpr, tpr)
rec_ds, prec_ds = downsample(rec, prec)

# Probability histogram: bin into 40 bins, separately for positive/negative residues
bins = np.linspace(0, 1, 41)
hist_pos, _ = np.histogram(final_p[y_test_ref == 1], bins=bins)
hist_neg, _ = np.histogram(final_p[y_test_ref == 0], bins=bins)

json.dump({
    "roc": {"fpr": fpr_ds, "tpr": tpr_ds, "auc": float(roc_auc_score(y_test_ref, final_p))},
    "pr": {"recall": rec_ds, "precision": prec_ds, "auprc": float(average_precision_score(y_test_ref, final_p))},
    "histogram": {"bin_edges": bins.tolist(), "positive_counts": hist_pos.tolist(), "negative_counts": hist_neg.tolist()},
    "n_residues": int(len(y_test_ref)), "n_positive": int(y_test_ref.sum()),
}, open("/kaggle/working/phase156_curves.json", "w"), indent=2)
print("saved -> phase156_curves.json")
''')

md(r'''## Cell 6 — Per-chain diagnostics (to pick a good example structure) + full per-residue predictions''')
code(r'''
# Rebuild per-chain slices using num_nodes offsets (same pattern as Phase153's bootstrap)
chain_keys = scale_test_keys["35M"]   # [(pdb_id, chain_id, num_nodes), ...] in residue-array order
offsets = np.concatenate([[0], np.cumsum([k[2] for k in chain_keys])])

per_chain = []
per_residue_export = {}
for i, (pdb_id, chain_id, n) in enumerate(chain_keys):
    lo, hi = offsets[i], offsets[i+1]
    y_c = y_test_ref[lo:hi]
    p_c = final_p[lo:hi]
    n_pos = int(y_c.sum())
    entry = {"pdb_id": pdb_id, "chain_id": chain_id, "n_residues": int(n), "n_positive": n_pos}
    if n_pos > 0 and n_pos < n:   # AUC/MCC need both classes present
        entry["auc"] = float(roc_auc_score(y_c, p_c))
        pred_c = (p_c >= best_t).astype(int)
        entry["mcc"] = float(matthews_corrcoef(y_c, pred_c)) if len(set(pred_c)) > 1 else None
    else:
        entry["auc"] = None; entry["mcc"] = None
    per_chain.append(entry)
    # per-residue export for every chain (cheap: ~55k residues total across 182 chains)
    per_residue_export[f"{pdb_id}_{chain_id}"] = {
        "prob": p_c.round(4).tolist(), "true_label": y_c.astype(int).tolist(), "n_residues": int(n)
    }

per_chain_sorted = sorted([c for c in per_chain if c["auc"] is not None],
                          key=lambda c: (-c["auc"], -c["n_positive"]))
print(f"{'pdb_id':<8}{'chain':<7}{'n_res':>7}{'n_pos':>7}{'AUC':>8}{'MCC':>8}")
print("-"*50)
for c in per_chain_sorted[:15]:
    mcc_s = f"{c['mcc']:.3f}" if c['mcc'] is not None else "  --  "
    print(f"{c['pdb_id']:<8}{c['chain_id']:<7}{c['n_residues']:>7}{c['n_positive']:>7}{c['auc']:>8.3f}{mcc_s:>8}")

json.dump({"per_chain": per_chain}, open("/kaggle/working/phase156_per_chain.json", "w"), indent=2)
json.dump(per_residue_export, open("/kaggle/working/phase156_per_residue.json", "w"))
print(f"\nsaved -> phase156_per_chain.json ({len(per_chain)} chains), phase156_per_residue.json")
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase156_cryptobench_prediction_export.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
