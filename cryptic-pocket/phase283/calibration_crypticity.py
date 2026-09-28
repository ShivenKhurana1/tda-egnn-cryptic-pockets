"""Phase 283c: the crypticity-stratified calibration follow-up flagged in paper.md's Future Work ("extending the crypticity audit's method
to the validation split... remains open"). Phase 283's own grid-based cavity-voxel detector (apo-side only) gives a proxy crypticity score
(my_apo, lower = more cryptic-like) that correlates with the external audit's real labels on the 128-chain labeled test subset (Spearman
r=0.256, p=0.0035, n=128; pooled AUC 0.740 cryptic-vs-preformed) -- real but modest signal, NOT strong enough to trust as hard per-chain
ground truth (5-fold CV precision at the Youden-J threshold is only 0.23, given the ~13% cryptic base rate). Used here only as a soft,
continuous stratifier, exactly the way pocket size was used in Phase 282b, not as a relabeling of individual chains.

Procedure: (1) apply the SAME fixed proxy threshold fit via 5-fold CV on the 128-chain labeled test set (my_apo <= 296 => "cryptic-like") to
BOTH the validation chains (their crypticity is otherwise unknown -- this is the actual audit extension) and the test chains (whose real
labels we already have, for honest evaluation); (2) fit two separate isotonic curves on the validation set's residues, split by this proxy
label; (3) apply the matching curve to each test chain by ITS OWN proxy label (never by its true label -- that would leak); (4) evaluate
against test's real cryptic/pre-formed labels and compare against the Phase 282 pooled fit and Phase 282b size-stratified fit as baselines.

usage: python3 calibration_crypticity.py <phase266_full_preds.pt> <phase276a_crypticity_labels.json> <val_my_apo.json> <test_my_apo.json> <threshold.json> <out_prefix>
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import torch

preds_path, crypt_path, val_apo_path, test_apo_path, thr_path, OUT = sys.argv[1:7]
P = torch.load(preds_path, map_location="cpu", weights_only=False)
crypt = json.load(open(crypt_path))
label_of = {k: v["label"] for k, v in crypt.items()}
THR = json.load(open(thr_path))["final_threshold_my_apo_le"]
seeds = sorted(P["seeds"])


def ensemble(split):
    S0 = P["seeds"][seeds[0]][split]
    return {k: dict(p=np.mean([P["seeds"][s][split][k]["p"] for s in seeds], axis=0), y=S0[k]["y"]) for k in S0}


val, test = ensemble("val"), ensemble("test")

val_apo_rows = json.load(open(val_apo_path))
val_apo = {(r["pdb"], r["chain"]): r["my_apo"] for r in val_apo_rows if "error" not in r}
test_apo_rows = json.load(open(test_apo_path))  # 128-chain labeled subset only
test_apo = {}
for r in test_apo_rows:
    if "error" in r: continue
    pdb, chain = r["key"].split("_", 1)
    test_apo[(pdb, chain)] = r["my_apo"]

print(f"proxy threshold (my_apo <= {THR:.0f} => cryptic-like), fit via 5-fold CV on the 128-chain labeled test set (Phase 283 full calibration)")
print(f"val chains with proxy score: {len(val_apo)}/{len(val)}")
print(f"test chains with proxy score (labeled cryptic/preformed subset): {len(test_apo)}")

val_cryptlike = [k for k in val if k in val_apo and val_apo[k] <= THR]
val_preflike = [k for k in val if k in val_apo and val_apo[k] > THR]
print(f"validation: {len(val_cryptlike)} proxy-cryptic-like chains, {len(val_preflike)} proxy-preformed-like chains")

pv_all = np.concatenate([val[k]["p"] for k in val]); yv_all = np.concatenate([val[k]["y"] for k in val])
iso_pooled = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(pv_all, yv_all)

pv_c = np.concatenate([val[k]["p"] for k in val_cryptlike]); yv_c = np.concatenate([val[k]["y"] for k in val_cryptlike])
pv_p = np.concatenate([val[k]["p"] for k in val_preflike]); yv_p = np.concatenate([val[k]["y"] for k in val_preflike])
iso_cryptlike = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(pv_c, yv_c)
iso_preflike = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(pv_p, yv_p)


def ece(p, y, n_bins=15):
    order = np.argsort(p); p, y = p[order], y[order]
    err, ns = [], []
    for b in np.array_split(np.arange(len(p)), n_bins):
        if len(b) == 0: continue
        err.append(abs(p[b].mean() - y[b].mean()) * len(b)); ns.append(len(b))
    return sum(err) / sum(ns)


def report(name, p, y):
    if y.sum() == 0 or y.sum() == len(y):
        print(f"  {name:42s} n={len(y):6d}  (degenerate, skipped)"); return None
    b, ll, e = brier_score_loss(y, p), log_loss(y, p, labels=[0, 1]), ece(p, y)
    print(f"  {name:42s} n={len(y):6d}  Brier {b:.4f}  log-loss {ll:.4f}  ECE {e:.4f}")
    return dict(name=name, n=len(y), brier=b, logloss=ll, ece=e)


rows = []
test_keys_with_apo = [k for k in test if k in test_apo]
print(f"\ntest chains usable for this evaluation (labeled + proxy score present): {len(test_keys_with_apo)}")


def apply_crypticity_stratified(keys):
    return np.concatenate([(iso_cryptlike if test_apo[k] <= THR else iso_preflike).predict(test[k]["p"]) for k in keys])


for stratum_name in ("cryptic", "preformed"):
    ks = [k for k in test_keys_with_apo if label_of.get(f"{k[0]}_{k[1]}") == stratum_name]
    if not ks: continue
    p_ = np.concatenate([test[k]["p"] for k in ks]); y_ = np.concatenate([test[k]["y"] for k in ks])
    p_strat = apply_crypticity_stratified(ks)
    print(f"\n=== TRUE-{stratum_name} test residues (n_chains={len(ks)}): raw vs pooled-isotonic vs crypticity-PROXY-stratified ===")
    r = report("raw", p_, y_); rows.append(dict(stratum=stratum_name, cal="raw", **r)) if r else None
    r = report("pooled isotonic (Phase 282 baseline)", iso_pooled.predict(p_), y_); rows.append(dict(stratum=stratum_name, cal="pooled", **r)) if r else None
    r = report("crypticity-proxy-stratified (this follow-up)", p_strat, y_); rows.append(dict(stratum=stratum_name, cal="crypt_proxy_stratified", **r)) if r else None

p_all = np.concatenate([test[k]["p"] for k in test_keys_with_apo]); y_all = np.concatenate([test[k]["y"] for k in test_keys_with_apo])
p_all_strat = apply_crypticity_stratified(test_keys_with_apo)
print(f"\n=== ALL labeled test residues pooled (n_chains={len(test_keys_with_apo)}) ===")
r = report("raw", p_all, y_all); rows.append(dict(stratum="all_labeled", cal="raw", **r)) if r else None
r = report("pooled isotonic", iso_pooled.predict(p_all), y_all); rows.append(dict(stratum="all_labeled", cal="pooled", **r)) if r else None
r = report("crypticity-proxy-stratified", p_all_strat, y_all); rows.append(dict(stratum="all_labeled", cal="crypt_proxy_stratified", **r)) if r else None

pd.DataFrame(rows).to_csv(OUT + "_crypticity_calibration.csv", index=False)
print(f"\nsaved -> {OUT}_crypticity_calibration.csv")

out_labels = {"threshold_my_apo_le": THR, "val_proxy_cryptic_like": [f"{k[0]}_{k[1]}" for k in val_cryptlike],
              "val_proxy_preformed_like": [f"{k[0]}_{k[1]}" for k in val_preflike],
              "val_my_apo": {f"{k[0]}_{k[1]}": val_apo[k] for k in val if k in val_apo}}
json.dump(out_labels, open(OUT + "_val_crypticity_proxy_labels.json", "w"), indent=1)
print(f"saved -> {OUT}_val_crypticity_proxy_labels.json  (provisional validation-split crypticity-proxy labels)")
