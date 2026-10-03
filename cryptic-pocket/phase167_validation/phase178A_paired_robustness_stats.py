import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

INPUT = "phase167_validation/phase177A_protein_level_results.csv"

OUTPUT = "phase167_validation/phase178A_paired_robustness_stats.csv"

ARMS = [
    "esm_structure",
    "full_tda",
    "full_model",
]

TARGET = "sharpened_gamma10"

CONTROLS = [
    "top10_centroid",
    "protein_centroid",
    "random10_centroid",
    "probability_centroid_gamma1",
]

METRICS = [
    "median_center_error",
    "dcc4_rate",
    "dcc6_rate",
    "dcc8_rate",
    "dcc10_rate",
]

rng = np.random.default_rng(178)

N_PERM = 100000


def paired_permutation_p(x, y, rng, n_perm=100000):
    """
    Paired sign-flip permutation test.

    Null:
        median/mean paired difference = 0

    Uses the mean paired difference as the test statistic.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    d = x - y

    observed = np.mean(d)

    # Generate signs in chunks to avoid excessive memory.
    count = 0
    total = 0

    chunk = 5000

    while total < n_perm:
        m = min(chunk, n_perm - total)

        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(m, len(d))
        )

        stats = np.mean(
            signs * d[None, :],
            axis=1
        )

        count += np.sum(
            np.abs(stats) >= abs(observed)
        )

        total += m

    return (count + 1) / (n_perm + 1)


def bh_fdr(pvals):
    """
    Benjamini-Hochberg FDR correction.
    """
    pvals = np.asarray(pvals, dtype=float)

    n = len(pvals)
    order = np.argsort(pvals)

    ranked = pvals[order]

    adjusted = np.empty(n)

    running = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1

        value = ranked[i] * n / rank

        running = min(running, value)

        adjusted[i] = running

    result = np.empty(n)

    result[order] = adjusted

    return result


df = pd.read_csv(INPUT)

print("=" * 110)
print("PHASE 178A - PAIRED PROTEIN-LEVEL ROBUSTNESS STATISTICS")
print("=" * 110)

print(f"Input rows: {len(df):,}")

required = {
    "apo_pdb_id",
    "apo_chain",
    "arm",
    "method",
    *METRICS,
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}"
    )

results = []

for arm in ARMS:

    arm_df = df[df["arm"] == arm].copy()

    for control in CONTROLS:

        target_df = arm_df[
            arm_df["method"] == TARGET
        ].copy()

        control_df = arm_df[
            arm_df["method"] == control
        ].copy()

        key = [
            "apo_pdb_id",
            "apo_chain",
        ]

        merged = target_df.merge(
            control_df,
            on=key,
            suffixes=("_target", "_control"),
            how="inner",
        )

        if len(merged) != 185:
            print(
                f"WARNING: {arm} vs {control}: "
                f"{len(merged)} proteins instead of 185"
            )

        for metric in METRICS:

            x = merged[
                f"{metric}_target"
            ].to_numpy(dtype=float)

            y = merged[
                f"{metric}_control"
            ].to_numpy(dtype=float)

            d = x - y

            # Wilcoxon on protein-level paired summaries.
            try:
                w_stat, w_p = wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
            except Exception:
                w_stat = np.nan
                w_p = np.nan

            # Paired sign-flip permutation.
            perm_p = paired_permutation_p(
                x,
                y,
                rng,
                N_PERM,
            )

            # Direction.
            if metric == "median_center_error":
                target_better = np.sum(d < 0)
                control_better = np.sum(d > 0)
            else:
                target_better = np.sum(d > 0)
                control_better = np.sum(d < 0)

            ties = np.sum(d == 0)

            results.append({
                "arm": arm,
                "control": control,
                "metric": metric,
                "n_proteins": len(merged),

                "target_mean": np.mean(x),
                "control_mean": np.mean(y),

                "target_median": np.median(x),
                "control_median": np.median(y),

                "mean_difference_target_minus_control":
                    np.mean(d),

                "median_difference_target_minus_control":
                    np.median(d),

                "wilcoxon_p": w_p,

                "paired_permutation_p": perm_p,

                "target_better": target_better,
                "control_better": control_better,
                "ties": ties,
            })


results = pd.DataFrame(results)

# Correct all 36 tests together.
results["wilcoxon_bh_fdr"] = bh_fdr(
    results["wilcoxon_p"].fillna(1.0)
)

results["permutation_bh_fdr"] = bh_fdr(
    results["paired_permutation_p"]
)

m = len(results)

results["wilcoxon_bonferroni"] = np.minimum(
    results["wilcoxon_p"] * m,
    1.0
)

results["permutation_bonferroni"] = np.minimum(
    results["paired_permutation_p"] * m,
    1.0
)

results["wilcoxon_bh_significant"] = (
    results["wilcoxon_bh_fdr"] < 0.05
)

results["permutation_bh_significant"] = (
    results["permutation_bh_fdr"] < 0.05
)

results["permutation_bonferroni_significant"] = (
    results["permutation_bonferroni"] < 0.05
)

results.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 110)
print("ALL PAIRED RESULTS")
print("=" * 110)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

print("\n" + "=" * 110)
print("PAIRED PERMUTATION TESTS SURVIVING BH-FDR < 0.05")
print("=" * 110)

sig = results[
    results["permutation_bh_significant"]
].copy()

if len(sig) == 0:
    print("None.")
else:
    print(
        sig.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}"
        )
    )

print("\n" + "=" * 110)
print("PAIRED PERMUTATION TESTS SURVIVING BONFERRONI < 0.05")
print("=" * 110)

sig_bonf = results[
    results["permutation_bonferroni_significant"]
].copy()

if len(sig_bonf) == 0:
    print("None.")
else:
    print(
        sig_bonf.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}"
        )
    )

print("\nSaved:")
print(OUTPUT)
