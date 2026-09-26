"""Merge the 7,513-chain classification geometry control: folds 0-3 from the GPU kernels (Phase 255a/b) and folds 4-9 from the CPU kernels
(Phase 257, one kernel per fold x arm, 2 seeds each), then run the pre-specified paired comparisons over the folds that are complete.

usage: python3 analyze_255_257.py <gpu_results_dir containing phase255{a,b}_records.json> <cpu_dir containing f{fold}{arm}/phase257_f{fold}{arm}_records.json>
Unit of analysis = fold (independent, homology-clean); the 2 seeds are averaged within fold. A comparison is reported only over folds where
BOTH arms are complete, and the number of folds is printed with every line so partial data cannot be mistaken for the full ten.
"""
import glob, json, sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, t as tdist

gpu_dir, cpu_dir = sys.argv[1], sys.argv[2]
recs = []
for tag in "ab":
    recs += json.load(open(f"{gpu_dir}/phase255{tag}_records.json"))["records"]
for f in sorted(glob.glob(f"{cpu_dir}/f*/phase257_f*_records.json")):
    recs += json.load(open(f))["records"]
df = pd.DataFrame(recs)
assert df.groupby(["arm", "fold", "seed"]).size().eq(1).all(), "duplicate (arm, fold, seed) records"
cnt = df.groupby(["arm", "fold"]).size()
assert cnt.eq(2).all(), f"an (arm, fold) does not have exactly 2 seeds:\n{cnt[cnt != 2]}"
fold = df.groupby(["arm", "fold"])[["test_pr", "test_roc", "val_pr"]].mean().reset_index()
ARMS = ["no-topo", "full", "geom", "geom+topo"]
have = {a: sorted(fold[fold.arm == a].fold.unique()) for a in ARMS}
print("folds available per arm:", {a: f"{len(v)}/10" for a, v in have.items()})

print("\nTest PR-AUC / ROC-AUC, mean +- std over each arm's own complete folds")
for a in ARMS:
    s = fold[fold.arm == a]
    print(f"  {a:10s} PR {s.test_pr.mean():.4f} +- {s.test_pr.std(ddof=0):.4f}   ROC {s.test_roc.mean():.4f} +- {s.test_roc.std(ddof=0):.4f}   ({len(s)} folds)")

COMPS = [("geom+topo", "geom"), ("full", "geom"), ("geom", "no-topo"), ("full", "no-topo"), ("geom+topo", "full")]
rows = []
print("\nPaired over folds (a - b), restricted to folds where both arms are complete")
for a, b in COMPS:
    common = sorted(set(have[a]) & set(have[b]))
    if len(common) < 3:
        print(f"  {a} - {b}: only {len(common)} common folds, skipped"); continue
    for m in ["test_pr", "test_roc"]:
        x = fold[(fold.arm == a) & fold.fold.isin(common)].sort_values("fold")[m].values
        y = fold[(fold.arm == b) & fold.fold.isin(common)].sort_values("fold")[m].values
        d = x - y; n = len(d)
        tp, wp = ttest_rel(x, y).pvalue, wilcoxon(d).pvalue
        se = d.std(ddof=1) / np.sqrt(n); tc = tdist.ppf(0.975, n - 1)
        rows.append(dict(a=a, b=b, metric=m, n_folds=n, mean_a=x.mean(), mean_b=y.mean(), mean_delta=d.mean(),
                         ci_lo=d.mean() - tc * se, ci_hi=d.mean() + tc * se, wins=int((d > 0).sum()), t_p=tp, wilcoxon_p=wp))
        print(f"  {a:10s} - {b:9s} {m:8s} {x.mean():.4f} vs {y.mean():.4f}  delta {d.mean():+.4f} [{d.mean()-tc*se:+.4f},{d.mean()+tc*se:+.4f}]  "
              f"{int((d>0).sum())}/{n} folds  t p={tp:.1e}  W p={wp:.1e}")
out = f"{cpu_dir}/phase255_257_merged"
pd.DataFrame(rows).to_csv(out + "_paired_stats.csv", index=False); fold.to_csv(out + "_fold_means.csv", index=False)
print(f"\nsaved {out}_paired_stats.csv, {out}_fold_means.csv")
