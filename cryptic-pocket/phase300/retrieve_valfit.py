"""Robustness check for retrieve.py: fit the stacker on the TEST-TIME model's own validation-chain predictions (Phase 299 stack3, 60 val
chains, templates from the other training folds) instead of on the differently-scaled CV models, then apply to test."""
import json, sys
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon
sys.argv = [sys.argv[0]]
import retrieve as R
A = torch.load("/tmp/p299_out/phase299_test_preds.pt", map_location="cpu", weights_only=False)["stack3"]["seeds"]
val = [q for q in R.INFO if R.key(q) in A[0]["val"]]
for q in val: R.PM[q] = np.mean([A[s]["val"][R.key(q)]["p"] for s in range(8)], 0)
val = [q for q in val if len(R.PM[q]) == len(R.INFO[q]["y"])]
lr = LogisticRegression(C=1.0, max_iter=2000).fit(np.vstack([R.feats(q) for q in val]), np.concatenate([R.INFO[q]["y"] for q in val]))
print(f"val chains {len(val)}; coefficients", np.round(lr.coef_[0], 3))
st = {q: lr.predict_proba(R.feats(q))[:, 1] for q in R.te_keys}
R.report("TEST (stacker fit on val)", R.te_keys, {q: R.PM[q] for q in R.te_keys}, st, {q: R.TPL[q][0] for q in R.INFO})
R.report("TEST strictly cryptic (stacker fit on val)", R.crypt, {q: R.PM[q] for q in R.crypt}, st, {q: R.TPL[q][0] for q in R.INFO})
