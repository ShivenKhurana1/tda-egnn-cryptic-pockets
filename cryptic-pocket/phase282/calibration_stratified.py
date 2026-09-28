"""Phase 282b: the per-stratum calibration follow-up flagged in paper.md's Future Work. A single isotonic fit on the pooled 60-chain
validation set fixed most of the raw probabilities' overconfidence (Phase 282) but left the small-pocket stratum (<=8 residues) the
worst-calibrated group even afterward (ECE 0.025 vs 0.0029 pooled). This fits a SEPARATE isotonic curve per pocket-size stratum on the
validation set (17 small chains / 5,246 residues / 104 positives; 43 larger chains / 14,316 residues / 742 positives -- both comfortably
enough to fit a stable curve) and checks whether that closes the residual gap on the held-out test set, against the single pooled fit as
the baseline it must beat. Cryptic/pre-formed stratified fitting is NOT attempted: those labels exist only for the test split (the external
audit relabeled test, never validation), so there is no validation-set ground truth to fit a cryptic-specific curve against -- noted here
rather than faked. The size-stratified fit's effect on the cryptic/pre-formed test breakdown is reported anyway, for context, since pocket
size and crypticity are likely correlated even though the fit itself does not use crypticity labels.

usage: python3 calibration_stratified.py <phase266_full_preds.pt> <phase276a_crypticity_labels.json> <out_prefix>
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import torch

preds_path, crypt_path, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
P = torch.load(preds_path, map_location="cpu", weights_only=False)
crypt = json.load(open(crypt_path))
label_of = {k: v["label"] for k, v in crypt.items()}
seeds = sorted(P["seeds"])


def ensemble(split):
    S0 = P["seeds"][seeds[0]][split]
    return {k: dict(p=np.mean([P["seeds"][s][split][k]["p"] for s in seeds], axis=0), y=S0[k]["y"]) for k in S0}


val, test = ensemble("val"), ensemble("test")
SIZE_CUT = 8
val_small = [k for k in val if int(val[k]["y"].sum()) <= SIZE_CUT]
val_large = [k for k in val if int(val[k]["y"].sum()) > SIZE_CUT]
test_small = [k for k in test if int(test[k]["y"].sum()) <= SIZE_CUT]
test_large = [k for k in test if int(test[k]["y"].sum()) > SIZE_CUT]
print(f"validation: {len(val_small)} small chains, {len(val_large)} larger chains")
print(f"test:       {len(test_small)} small chains, {len(test_large)} larger chains")

pv_all = np.concatenate([val[k]["p"] for k in val]); yv_all = np.concatenate([val[k]["y"] for k in val])
iso_pooled = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(pv_all, yv_all)

pv_s = np.concatenate([val[k]["p"] for k in val_small]); yv_s = np.concatenate([val[k]["y"] for k in val_small])
pv_l = np.concatenate([val[k]["p"] for k in val_large]); yv_l = np.concatenate([val[k]["y"] for k in val_large])
iso_small = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(pv_s, yv_s)
iso_large = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(pv_l, yv_l)


def ece(p, y, n_bins=15):
    order = np.argsort(p); p, y = p[order], y[order]
    err, ns = [], []
    for b in np.array_split(np.arange(len(p)), n_bins):
        if len(b) == 0: continue
        err.append(abs(p[b].mean() - y[b].mean()) * len(b)); ns.append(len(b))
    return sum(err) / sum(ns)


def report(name, p, y):
    if y.sum() == 0 or y.sum() == len(y):
        print(f"  {name:34s} n={len(y):6d}  (degenerate, skipped)"); return None
    b, ll, e = brier_score_loss(y, p), log_loss(y, p, labels=[0, 1]), ece(p, y)
    print(f"  {name:34s} n={len(y):6d}  Brier {b:.4f}  log-loss {ll:.4f}  ECE {e:.4f}")
    return dict(name=name, n=len(y), brier=b, logloss=ll, ece=e)


rows = []

print("\n=== small-pocket test residues (<=8): pooled-fit vs size-stratified-fit ===")
p_s = np.concatenate([test[k]["p"] for k in test_small]); y_s = np.concatenate([test[k]["y"] for k in test_small])
r = report("raw", p_s, y_s); rows.append(dict(stratum="small", cal="raw", **r)) if r else None
r = report("pooled isotonic (Phase 282 baseline)", iso_pooled.predict(p_s), y_s); rows.append(dict(stratum="small", cal="pooled", **r)) if r else None
r = report("size-stratified isotonic (this follow-up)", iso_small.predict(p_s), y_s); rows.append(dict(stratum="small", cal="stratified", **r)) if r else None

print("\n=== larger-pocket test residues (>8): pooled-fit vs size-stratified-fit ===")
p_l = np.concatenate([test[k]["p"] for k in test_large]); y_l = np.concatenate([test[k]["y"] for k in test_large])
r = report("raw", p_l, y_l); rows.append(dict(stratum="large", cal="raw", **r)) if r else None
r = report("pooled isotonic (Phase 282 baseline)", iso_pooled.predict(p_l), y_l); rows.append(dict(stratum="large", cal="pooled", **r)) if r else None
r = report("size-stratified isotonic (this follow-up)", iso_large.predict(p_l), y_l); rows.append(dict(stratum="large", cal="stratified", **r)) if r else None

print("\n=== pooled overall test residues: does size-stratified calibration hurt or help the headline (unstratified) number? ===")
p_all = np.concatenate([test[k]["p"] for k in test]); y_all = np.concatenate([test[k]["y"] for k in test])
p_all_strat = np.concatenate([(iso_small if k in set(test_small) else iso_large).predict(test[k]["p"]) for k in test])
r = report("raw", p_all, y_all); rows.append(dict(stratum="all", cal="raw", **r)) if r else None
r = report("pooled isotonic", iso_pooled.predict(p_all), y_all); rows.append(dict(stratum="all", cal="pooled", **r)) if r else None
r = report("size-stratified isotonic", p_all_strat, y_all); rows.append(dict(stratum="all", cal="stratified", **r)) if r else None

print("\n=== context only: effect on the cryptic / pre-formed test breakdown (fit never used crypticity labels) ===")
for stratum in ("cryptic", "preformed"):
    ks = [k for k in test if label_of.get(f"{k[0]}_{k[1]}") == stratum]
    if not ks: continue
    p_c = np.concatenate([test[k]["p"] for k in ks]); y_c = np.concatenate([test[k]["y"] for k in ks])
    p_c_strat = np.concatenate([(iso_small if k in set(test_small) else iso_large).predict(test[k]["p"]) for k in ks])
    r = report(f"{stratum} raw", p_c, y_c); rows.append(dict(stratum=stratum, cal="raw", **r)) if r else None
    r = report(f"{stratum} pooled isotonic", iso_pooled.predict(p_c), y_c); rows.append(dict(stratum=stratum, cal="pooled", **r)) if r else None
    r = report(f"{stratum} size-stratified isotonic", p_c_strat, y_c); rows.append(dict(stratum=stratum, cal="stratified", **r)) if r else None

pd.DataFrame(rows).to_csv(OUT + "_stratified_calibration.csv", index=False)
print(f"\nsaved -> {OUT}_stratified_calibration.csv")
