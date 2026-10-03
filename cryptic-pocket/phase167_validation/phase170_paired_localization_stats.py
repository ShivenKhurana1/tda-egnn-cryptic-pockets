import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


INPUT = "phase167_validation/phase169A_localization_baselines.csv"
OUTPUT = "phase167_validation/phase170_paired_localization_stats.csv"

df = pd.read_csv(INPUT)

arms = [
    "esm_structure",
    "full_tda",
    "full_model",
]

baselines = [
    "top_10pct_centroid",
    "top_20_centroid",
    "top_10_centroid",
]

rows = []

for arm in arms:
    arm_df = df[df["arm"] == arm].copy()

    for baseline in baselines:
        learned = arm_df[
            arm_df["method"] == "learned_localization"
        ][
            ["apo_pdb_id", "apo_chain", "seed", "center_error",
             "dcc4", "dcc6", "dcc8", "dcc10"]
        ].copy()

        base = arm_df[
            arm_df["method"] == baseline
        ][
            ["apo_pdb_id", "apo_chain", "seed", "center_error",
             "dcc4", "dcc6", "dcc8", "dcc10"]
        ].copy()

        learned = learned.rename(columns={
            "center_error": "learned_center_error",
            "dcc4": "learned_dcc4",
            "dcc6": "learned_dcc6",
            "dcc8": "learned_dcc8",
            "dcc10": "learned_dcc10",
        })

        base = base.rename(columns={
            "center_error": "baseline_center_error",
            "dcc4": "baseline_dcc4",
            "dcc6": "baseline_dcc6",
            "dcc8": "baseline_dcc8",
            "dcc10": "baseline_dcc10",
        })

        merged = learned.merge(
            base,
            on=["apo_pdb_id", "apo_chain", "seed"],
            how="inner"
        )

        # Center error:
        # Positive delta means the baseline has LOWER error.
        delta_error = (
            merged["learned_center_error"]
            - merged["baseline_center_error"]
        )

        stat_error, p_error = wilcoxon(
            merged["learned_center_error"],
            merged["baseline_center_error"],
            alternative="two-sided"
        )

        # DCC metrics:
        # Positive delta means learned localization has HIGHER DCC.
        for metric in ["dcc4", "dcc6", "dcc8", "dcc10"]:
            learned_vals = merged[f"learned_{metric}"]
            base_vals = merged[f"baseline_{metric}"]

            try:
                stat, p = wilcoxon(
                    learned_vals,
                    base_vals,
                    alternative="two-sided"
                )
            except ValueError:
                stat, p = np.nan, np.nan

            rows.append({
                "arm": arm,
                "baseline": baseline,
                "metric": metric,
                "n": len(merged),
                "learned_mean": learned_vals.mean(),
                "baseline_mean": base_vals.mean(),
                "learned_median": learned_vals.median(),
                "baseline_median": base_vals.median(),
                "mean_delta_learned_minus_baseline":
                    (learned_vals - base_vals).mean(),
                "wilcoxon_stat": stat,
                "p_value": p,
            })

        rows.append({
            "arm": arm,
            "baseline": baseline,
            "metric": "center_error",
            "n": len(merged),
            "learned_mean": merged["learned_center_error"].mean(),
            "baseline_mean": merged["baseline_center_error"].mean(),
            "learned_median": merged["learned_center_error"].median(),
            "baseline_median": merged["baseline_center_error"].median(),
            "mean_delta_learned_minus_baseline":
                delta_error.mean(),
            "wilcoxon_stat": stat_error,
            "p_value": p_error,
        })

results = pd.DataFrame(rows)

# Correct all 27 tests:
# 3 arms × 3 baselines × 5 metrics
mask = results["p_value"].notna()

# Benjamini-Hochberg FDR correction
def benjamini_hochberg(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)

    order = np.argsort(pvalues)
    ranked = pvalues[order]

    adjusted = np.empty(n, dtype=float)
    running = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        running = min(running, value)
        adjusted[i] = running

    result = np.empty(n, dtype=float)
    result[order] = np.minimum(adjusted, 1.0)

    return result

pvals = results.loc[mask, "p_value"].values

results.loc[mask, "p_adjusted_bh"] = benjamini_hochberg(pvals)

# Bonferroni correction
results.loc[mask, "p_adjusted_bonferroni"] = np.minimum(
    pvals * len(pvals),
    1.0
)

results["significant_fdr_05"] = (
    results["p_adjusted_bh"] < 0.05
)

results["significant_bonferroni_05"] = (
    results["p_adjusted_bonferroni"] < 0.05
)

results.to_csv(OUTPUT, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 100)

print("=" * 100)
print("PHASE 170 - PAIRED LOCALIZATION STATISTICS")
print("=" * 100)

print("\nResults:")
print(
    results[
        [
            "arm",
            "baseline",
            "metric",
            "n",
            "learned_mean",
            "baseline_mean",
            "mean_delta_learned_minus_baseline",
            "p_value",
            "p_adjusted_bh",
            "significant_fdr_05",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(OUTPUT)
