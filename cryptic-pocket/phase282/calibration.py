"""Phase 282: is the headline model's confidence score trustworthy? Every result so far in this project reports discrimination (PR-AUC,
ROC-AUC, MCC) -- whether the model ranks true sites above non-sites -- never calibration: whether "this residue is 70% likely to be a
pocket residue" is actually right about 70% of the time. A model can discriminate well while being badly calibrated (systematically over-
or under-confident), which matters directly for real use: a user deciding whether to trust a single prediction needs the probability itself
to mean something, not just its rank. Uses only predictions already produced by this project (no new training, no GPU).

Method: fit isotonic regression on the VALIDATION set only (60 chains, never used for any other calibration/threshold decision in this
project beyond the existing MCC threshold), apply it to the held-out TEST set, and report Brier score, log-loss, and expected calibration
error (ECE, 15 quantile bins) before and after, both pooled and stratified by the difficulty subgroups this project has already established
(cryptic vs pre-formed by the external crypticity audit; small (<=8 residue) vs larger pockets). If a single pooled calibration fit does not
generalize to these subgroups, that is itself an important, reportable finding.

usage: python3 calibration.py <phase266_full_preds.pt> <phase276a_crypticity_labels.json> <out_prefix>
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import torch

preds_path, crypt_path, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
P = torch.load(preds_path, map_location="cpu", weights_only=False)
crypt = json.load(open(crypt_path))
seeds = sorted(P["seeds"])


def ensemble(split):
    S0 = P["seeds"][seeds[0]][split]
    out = {}
    for k in S0:
        out[k] = dict(p=np.mean([P["seeds"][s][split][k]["p"] for s in seeds], axis=0), y=S0[k]["y"])
    return out


val, test = ensemble("val"), ensemble("test")
pv = np.concatenate([val[k]["p"] for k in val]); yv = np.concatenate([val[k]["y"] for k in val])
pt = np.concatenate([test[k]["p"] for k in test]); yt = np.concatenate([test[k]["y"] for k in test])
print(f"validation: {len(pv)} residues, {int(yv.sum())} positive ({100*yv.mean():.2f}%)")
print(f"test:       {len(pt)} residues, {int(yt.sum())} positive ({100*yt.mean():.2f}%)")

iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(pv, yv)
pt_cal = iso.predict(pt)


def ece(p, y, n_bins=15):
    order = np.argsort(p); p, y = p[order], y[order]
    bins = np.array_split(np.arange(len(p)), n_bins)
    err, ns = [], []
    for b in bins:
        if len(b) == 0: continue
        conf, acc = p[b].mean(), y[b].mean()
        err.append(abs(conf - acc) * len(b)); ns.append(len(b))
    return sum(err) / sum(ns)


def report(name, p, y):
    if y.sum() == 0 or y.sum() == len(y):
        print(f"  {name:28s} n={len(y):6d}  (degenerate, skipped)"); return None
    b, ll, e = brier_score_loss(y, p), log_loss(y, p, labels=[0, 1]), ece(p, y)
    print(f"  {name:28s} n={len(y):6d}  Brier {b:.4f}  log-loss {ll:.4f}  ECE {e:.4f}")
    return dict(name=name, n=len(y), brier=b, logloss=ll, ece=e)


print("\n=== pooled, all test residues ===")
rows = []
rows.append(dict(cal="raw", **report("raw probabilities", pt, yt)))
rows.append(dict(cal="isotonic(fit on val)", **report("isotonic-calibrated", pt_cal, yt)))

# reliability table: raw vs calibrated, 10 quantile bins
print("\nreliability (10 quantile bins of RAW probability): predicted vs actual positive rate")
order = np.argsort(pt); p_s, y_s, pc_s = pt[order], yt[order], pt_cal[order]
for b in np.array_split(np.arange(len(p_s)), 10):
    if len(b) == 0: continue
    print(f"  n={len(b):5d}  raw pred {p_s[b].mean():.4f}  cal pred {pc_s[b].mean():.4f}  actual {y_s[b].mean():.4f}")

# stratified: cryptic vs preformed (test chains only, from the external crypticity audit)
label_of = {k: v["label"] for k, v in crypt.items()}   # keys already "{pdb}_{chain}", matching keyfmt() below
def keyfmt(k): return f"{k[0]}_{k[1]}"
print("\n=== stratified by crypticity (external audit labels) ===")
for stratum in ("cryptic", "preformed"):
    ks = [k for k in test if label_of.get(keyfmt(k)) == stratum]
    if not ks: continue
    p_s = np.concatenate([test[k]["p"] for k in ks]); y_s = np.concatenate([test[k]["y"] for k in ks])
    pc_s = np.concatenate([iso.predict(test[k]["p"]) for k in ks])
    r1 = report(f"{stratum} raw", p_s, y_s); r2 = report(f"{stratum} isotonic(fit on val)", pc_s, y_s)
    if r1: rows.append(dict(cal="raw", **r1))
    if r2: rows.append(dict(cal="isotonic(fit on val)", **r2))

print("\n=== stratified by pocket size ===")
for lo, hi, name in [(0, 8, "small (<=8 residues)"), (9, 10**9, "larger (>8 residues)")]:
    ks = [k for k in test if lo <= int(test[k]["y"].sum()) <= hi]
    if not ks: continue
    p_s = np.concatenate([test[k]["p"] for k in ks]); y_s = np.concatenate([test[k]["y"] for k in ks])
    pc_s = np.concatenate([iso.predict(test[k]["p"]) for k in ks])
    r1 = report(f"{name} raw", p_s, y_s); r2 = report(f"{name} isotonic(fit on val)", pc_s, y_s)
    if r1: rows.append(dict(cal="raw", **r1))
    if r2: rows.append(dict(cal="isotonic(fit on val)", **r2))

pd.DataFrame(rows).to_csv(OUT + "_calibration.csv", index=False)
print(f"\nsaved -> {OUT}_calibration.csv")
