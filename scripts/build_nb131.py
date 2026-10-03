#!/usr/bin/env python3
"""Phase 131: CryptoBench weighted cross-scale ensemble. Phase130's
combined result equal-weighted all 3 scales (1/3 each) even though
3B-transfer (val PR/MCC clearly the strongest) and 650M-scratch (clearly
the weakest) are not equally good. Grid-searches ensemble weights on VAL
ONLY (never touches test until the final evaluation), then applies the
best weights to test -- same val/test discipline as every other
threshold-tuning step in this project. Pure recombination of predictions
that already exist -- zero new training."""
import json
from pathlib import Path

nb130 = json.load(open("/Users/shivenkhurana/bioproj/phase130_cryptobench_combined.ipynb"))
c130 = nb130["cells"]
s130 = lambda i: c130[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 131 - CryptoBench: Weighted Cross-Scale Ensemble

Phase130 equal-weighted all 3 scales (1/3 each) despite 3B-transfer being
clearly the strongest (val PR/MCC) and 650M-scratch the weakest. Grid-
searches ensemble weights on VAL ONLY, applies the winning weights to
TEST once. Zero new training -- pure recombination of Phase130's
predictions.
''')

md(r'''## Cell 1 - Install dependencies''')
code(s130(2))

md(r'''## Cell 2 - Model (identical to Phase130)''')
code(s130(4))

md(r'''## Cell 3 - Load each scale''' + "'" + r'''s blob + 8 checkpoints, predict on VAL and TEST (identical to Phase130)''')
code(s130(6))

md(r'''## Cell 4 - Grid-search ensemble weights on VAL ONLY, apply winning weights + val-tuned threshold to TEST''')
code(r'''
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

y_val_ref = scale_val_y["35M"]
y_test_ref = scale_test_y["35M"]
SCALE_NAMES = ["35M", "650M", "3B"]

# grid search: weights on a 0.1 simplex (w1+w2+w3=1, each a multiple of 0.1)
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
        grid_results.append(dict(w35m=w1, w650m=w2, w3b=w3, val_mcc=val_mcc, val_auc=val_auc, thresh=t))
        if val_mcc > best_val_mcc:
            best_val_mcc, best_w, best_val_auc_at_w = val_mcc, weights, val_auc

print(f"best weights by VAL MCC: 35M={best_w['35M']:.1f}  650M={best_w['650M']:.1f}  3B={best_w['3B']:.1f}")
print(f"  (val MCC {best_val_mcc:.4f}, val AUC {best_val_auc_at_w:.4f})")

# apply winning weights + a threshold re-tuned on val with those weights, to TEST
test_p_weighted = sum(best_w[s] * scale_test_probs[s] for s in SCALE_NAMES)
val_p_weighted = sum(best_w[s] * scale_val_probs[s] for s in SCALE_NAMES)
final_thresh, _ = best_thresh(val_p_weighted, y_val_ref)
weighted_result = metrics_at(test_p_weighted, y_test_ref, final_thresh)

# also grid-search separately by VAL AUC (threshold-independent), in case it picks different weights
best_w_auc, best_val_auc = None, -1.0
for r in grid_results:
    weights = {"35M": r["w35m"], "650M": r["w650m"], "3B": r["w3b"]}
    if r["val_auc"] > best_val_auc:
        best_val_auc, best_w_auc = r["val_auc"], weights
test_p_auc = sum(best_w_auc[s] * scale_test_probs[s] for s in SCALE_NAMES)
t_auc, _ = best_thresh(sum(best_w_auc[s] * scale_val_probs[s] for s in SCALE_NAMES), y_val_ref)
weighted_result_auc = metrics_at(test_p_auc, y_test_ref, t_auc)

print(f"\n{'variant':<28}{'AUC':>8}{'AUPRC':>8}{'MCC':>8}{'F1':>8}")
print("-"*60)
print(f"{'equal-weight (Phase130)':<28}{0.9140:>8.4f}{0.4311:>8.4f}{0.4530:>8.4f}{0.4776:>8.4f}")
print(f"{'weighted (val-MCC-optimal)':<28}{weighted_result['auc']:>8.4f}{weighted_result['auprc']:>8.4f}{weighted_result['mcc']:>8.4f}{weighted_result['f1']:>8.4f}")
print(f"{'weighted (val-AUC-optimal)':<28}{weighted_result_auc['auc']:>8.4f}{weighted_result_auc['auprc']:>8.4f}{weighted_result_auc['mcc']:>8.4f}{weighted_result_auc['f1']:>8.4f}")
print(f"{'3B-transfer alone (best single)':<28}{0.9111:>8.4f}{0.4307:>8.4f}{0.4664:>8.4f}{0.4896:>8.4f}")

json.dump({
    "best_weights_by_val_mcc": best_w, "val_mcc": best_val_mcc,
    "best_weights_by_val_auc": best_w_auc, "val_auc": best_val_auc,
    "test_result_val_mcc_weights": weighted_result,
    "test_result_val_auc_weights": weighted_result_auc,
    "grid": grid_results,
}, open(WORK/"phase131_weighted_ensemble_results.json", "w"), indent=2)
print("\nsaved -> phase131_weighted_ensemble_results.json")
''')

md(r'''
---

## Reading this

- **Weighted beats both equal-weight ensemble AND 3B-transfer alone**
  -- the best possible outcome, a genuinely optimized combination worth
  adopting as the final CryptoBench number.
- **Weighted beats equal-weight but not 3B-transfer alone** -- the
  ensemble adds a little value but 650M is still net-diluting; consider
  dropping it and using only 35M+3B (or 3B alone).
- **Weighted ties equal-weight** -- the equal-weight combination was
  already close to optimal, nothing left on the table here.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase131_cryptobench_weighted_ensemble.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
