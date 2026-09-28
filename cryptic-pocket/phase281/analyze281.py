"""Phase 281 analysis: pool the 17 truly-cryptic proteins' real-MD per-residue features against the true site-residue label and check raw
predictive power (AUC), the same quick pre-commitment check used for the crypticity geometric proxy (Phase 279, AUC 0.611, rejected) and
conceptually the same question the elastic-network movement features (Phase 270-272) answered via a full retrain and came back null.
This is a cheap, single-feature check that must show real signal before any full retrain is justified.

usage: python3 analyze281.py <phase281_md_pilot_results.json>
"""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score

results = json.load(open(sys.argv[1]))
print(f"{len(results)} proteins with MD results")
feat_names = ["rmsf", "nbr_mean_delta", "nbr_std", "nbr_min_delta", "nbr_max_delta"]
rows = {f: [] for f in feat_names}; y_all = []
per_protein_auc = {f: [] for f in feat_names}
for key, r in results.items():
    y = np.zeros(r["n_res"]); y[r["site_idx"]] = 1
    if y.sum() == 0 or y.sum() == len(y):
        print(f"  {key}: skipped (degenerate, {int(y.sum())}/{len(y)} positive)"); continue
    y_all.append(y)
    for f in feat_names:
        v = np.array(r[f])
        rows[f].append(v)
        try: per_protein_auc[f].append(roc_auc_score(y, v))
        except Exception: pass
y_all = np.concatenate(y_all)
print(f"\npooled: {len(y_all)} residues, {int(y_all.sum())} true site residues ({100*y_all.mean():.1f}%)")
print(f"{'feature':16s} {'pooled AUC':>10s} {'mean per-protein AUC':>22s}")
for f in feat_names:
    v = np.concatenate(rows[f])
    pooled = roc_auc_score(y_all, v)
    per_prot = np.mean(per_protein_auc[f]) if per_protein_auc[f] else float("nan")
    print(f"{f:16s} {pooled:10.3f} {per_prot:22.3f}")
