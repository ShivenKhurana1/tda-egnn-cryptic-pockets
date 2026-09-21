"""Merge Phase 230a/230b (folds 0-4, 5-9) and run the pre-specified paired comparisons over the 10 folds.

usage: python3 analyze_230.py <dir_a> <dir_b>   (each holds phase230{a,b}_records.json)
Unit of analysis = fold (10 independent homology-clean folds); the 2 seeds are averaged within fold, as in Phase 11.
"""
import json, sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

da, db = sys.argv[1], sys.argv[2]
recs = json.load(open(f"{da}/phase230arecords.json"))["records"] + json.load(open(f"{db}/phase230brecords.json"))["records"]
df = pd.DataFrame(recs)
assert sorted(df.fold.unique()) == list(range(10)), sorted(df.fold.unique())
assert df.groupby(["arm", "fold"]).size().eq(2).all(), "expected 2 seeds per arm per fold"
fold = df.groupby(["arm", "fold"])[["test_pr", "test_roc", "val_pr"]].mean().reset_index()

ARMS = ["no-topo", "full", "geom", "geom+topo"]
print("Test PR-AUC / ROC-AUC, mean +- std over 10 folds (2 seeds averaged per fold)")
for a in ARMS:
    s = fold[fold.arm == a]
    print(f"  {a:10s} PR {s.test_pr.mean():.4f} +- {s.test_pr.std(ddof=0):.4f}   ROC {s.test_roc.mean():.4f} +- {s.test_roc.std(ddof=0):.4f}")

def vec(a, m):
    return fold[fold.arm == a].sort_values("fold")[m].values

COMPS = [("geom+topo", "geom"), ("full", "geom"), ("geom", "no-topo"), ("full", "no-topo"), ("geom+topo", "full")]
rows = []
print("\nPaired over folds (a - b)")
for a, b in COMPS:
    for m in ["test_pr", "test_roc"]:
        x, y = vec(a, m), vec(b, m); d = x - y
        t, w = ttest_rel(x, y).pvalue, wilcoxon(d).pvalue
        se = d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = d.mean() - 2.262 * se, d.mean() + 2.262 * se        # t(9, 0.975)
        rows.append(dict(a=a, b=b, metric=m, mean_a=x.mean(), mean_b=y.mean(), mean_delta=d.mean(), ci_lo=lo, ci_hi=hi,
                         wins=int((d > 0).sum()), t_p=t, wilcoxon_p=w))
        print(f"  {a:10s} - {b:9s} {m:8s} {x.mean():.4f} vs {y.mean():.4f}  delta {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  {int((d>0).sum())}/10 folds  t p={t:.1e}  W p={w:.1e}")
pd.DataFrame(rows).to_csv(f"{da}/phase230_merged_paired_stats.csv", index=False)
fold.to_csv(f"{da}/phase230_merged_fold_means.csv", index=False)
print("\nsaved phase230_merged_paired_stats.csv, phase230_merged_fold_means.csv")
