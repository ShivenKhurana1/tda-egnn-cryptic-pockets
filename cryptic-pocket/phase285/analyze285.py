"""Phase 285: which topological dimension drives the confirmed classification residual beyond geometry?
Merges the four new single-dimension arms (geom+H0/H1/H2/burial) with the EXISTING geom/geom+topo records
from Phase 240a/b (same 2,056-chain pool, same 10-fold split seed=1337, same model/training -- not rerun).

Primary, pre-specified comparisons: pairwise among the four single-dimension arms (does one dimension carry
the whole residual, or is it diffuse), each single-dimension arm against `geom` alone (does adding just this
one dimension help at all), and the four arms' sum-of-effects against `geom+topo` (is the combination
synergistic, additive, or does one dimension explain the whole thing). Two-test rule throughout (paired t
AND Wilcoxon both <0.05); also reports each gap relative to the fold-level noise floor (std of the geom-arm
per-fold test PR-AUC), matching the 1.5x-noise-floor heuristic this project has used for every prior
topology-decomposition pass (Phase 23/58/86).

usage: python3 analyze285.py <phase285a_records.json> <phase285b_records.json> <phase240_merged_fold_means.csv> <out_prefix>
"""
import json, sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ra_path, rb_path, ref_csv, OUT = sys.argv[1:5]
recs = json.load(open(ra_path))["records"] + json.load(open(rb_path))["records"]
df = pd.DataFrame(recs)
assert sorted(df.fold.unique()) == list(range(10)), sorted(df.fold.unique())
assert df.groupby(["arm", "fold"]).size().eq(2).all(), "expected 2 seeds per arm per fold"
new_fold = df.groupby(["arm", "fold"])[["test_pr", "test_roc", "val_pr"]].mean().reset_index()

ref = pd.read_csv(ref_csv)
ref_fold = ref[ref.arm.isin(["geom", "geom+topo"])][["arm", "fold", "test_pr", "test_roc", "val_pr"]]

fold = pd.concat([new_fold, ref_fold], ignore_index=True)
NEW_ARMS = ["geom+H0", "geom+H1", "geom+H2", "geom+burial"]
ALL_ARMS = NEW_ARMS + ["geom", "geom+topo"]
assert set(fold.arm.unique()) == set(ALL_ARMS), set(fold.arm.unique())

print("Test PR-AUC / ROC-AUC, mean +- std over 10 folds (2 seeds averaged per fold)")
for a in ALL_ARMS:
    s = fold[fold.arm == a]
    print(f"  {a:12s} PR {s.test_pr.mean():.4f} +- {s.test_pr.std(ddof=0):.4f}   ROC {s.test_roc.mean():.4f} +- {s.test_roc.std(ddof=0):.4f}")

NOISE_PR = fold[fold.arm == "geom"].test_pr.std(ddof=0)
NOISE_ROC = fold[fold.arm == "geom"].test_roc.std(ddof=0)
print(f"\nnoise floor (geom-arm fold-to-fold std): PR {NOISE_PR:.4f}, ROC {NOISE_ROC:.4f}")


def vec(a, m):
    return fold[fold.arm == a].sort_values("fold")[m].values


def compare(a, b, m, noise):
    x, y = vec(a, m), vec(b, m)
    d = x - y
    t, w = ttest_rel(x, y).pvalue, wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    se = d.std(ddof=1) / np.sqrt(len(d))
    lo, hi = d.mean() - 2.262 * se, d.mean() + 2.262 * se
    wins = int((d > 0).sum())
    ratio = abs(d.mean()) / noise if noise > 0 else float("nan")
    conf = "  * CONFIRMED" if (t < .05 and w < .05) else ""
    print(f"  {a:12s} - {b:12s} [{m:8s}] {d.mean():+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]  {wins}/10 folds  "
          f"t p={t:.3f} W p={w:.3f}  ({ratio:.2f}x noise){conf}")
    return dict(a=a, b=b, metric=m, delta=float(d.mean()), lo=float(lo), hi=float(hi), wins=wins,
                t_p=float(t), w_p=float(w), noise_ratio=float(ratio), confirmed=bool(t < .05 and w < .05))


rows = []
print("\n=== Primary: pairwise among the four single-dimension arms ===")
import itertools
for a, b in itertools.combinations(NEW_ARMS, 2):
    for m, noise in (("test_pr", NOISE_PR), ("test_roc", NOISE_ROC)):
        rows.append(compare(a, b, m, noise))

print("\n=== Secondary: does each single dimension help at all over geometry alone? ===")
for a in NEW_ARMS:
    for m, noise in (("test_pr", NOISE_PR), ("test_roc", NOISE_ROC)):
        rows.append(compare(a, "geom", m, noise))

print("\n=== Context: does the strongest single dimension alone recover geom+topo's full residual? ===")
for a in NEW_ARMS:
    for m, noise in (("test_pr", NOISE_PR), ("test_roc", NOISE_ROC)):
        rows.append(compare("geom+topo", a, m, noise))

pd.DataFrame(rows).to_csv(OUT + "_pairwise.csv", index=False)
fold.to_csv(OUT + "_fold_means.csv", index=False)
print(f"\nsaved -> {OUT}_pairwise.csv, {OUT}_fold_means.csv")
