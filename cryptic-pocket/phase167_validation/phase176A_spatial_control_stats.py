import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

INPUT = "phase167_validation/phase175A_spatial_controls.csv"
OUT = "phase167_validation/phase176A_spatial_control_stats.csv"

ARMS = ["esm_structure", "full_tda", "full_model"]

CONTROLS = [
    "protein_centroid",
    "random10_centroid",
    "probability_centroid_gamma1",
    "top10_centroid",
]

TARGET = "sharpened_gamma10"

METRICS = [
    "center_error",
    "dcc4",
    "dcc6",
    "dcc8",
    "dcc10",
]


def bh_adjust(pvalues):
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
    result[order] = adjusted

    return result


def bonferroni(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    return np.minimum(pvalues * len(pvalues), 1.0)


def exact_mcnemar_pvalue(a, b):
    """
    Exact two-sided McNemar test.

    a = target method binary outcomes
    b = control method binary outcomes

    Only discordant pairs matter:
        target=1, control=0
        target=0, control=1
    """

    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)

    target_only = np.sum((a == 1) & (b == 0))
    control_only = np.sum((a == 0) & (b == 1))

    discordant = target_only + control_only

    if discordant == 0:
        return 1.0, target_only, control_only

    result = binomtest(
        target_only,
        discordant,
        0.5,
        alternative="two-sided",
    )

    return float(result.pvalue), target_only, control_only


records = []

df = pd.read_csv(INPUT)

print("=" * 110)
print("PHASE 176A — SPATIAL CONTROL STATISTICAL VALIDATION")
print("=" * 110)

print(f"Input records: {len(df):,}")

for arm in ARMS:

    arm_df = df[df["arm"] == arm].copy()

    for control in CONTROLS:

        target_df = (
            arm_df[arm_df["method"] == TARGET]
            .sort_values(["apo_pdb_id", "apo_chain", "seed"])
        )

        control_df = (
            arm_df[arm_df["method"] == control]
            .sort_values(["apo_pdb_id", "apo_chain", "seed"])
        )

        assert len(target_df) == len(control_df)

        for metric in METRICS:

            target = target_df[metric].to_numpy()
            control_values = control_df[metric].to_numpy()

            if metric == "center_error":

                result = wilcoxon(
                    target,
                    control_values,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )

                p = float(result.pvalue)

                effect = float(
                    np.mean(target - control_values)
                )

                statistic = float(result.statistic)

                target_better = int(
                    np.sum(target < control_values)
                )

                control_better = int(
                    np.sum(target > control_values)
                )

                ties = int(
                    np.sum(target == control_values)
                )

            else:

                p, target_only, control_only = exact_mcnemar_pvalue(
                    target,
                    control_values,
                )

                effect = float(
                    np.mean(target) - np.mean(control_values)
                )

                statistic = float(
                    target_only - control_only
                )

                target_better = int(target_only)
                control_better = int(control_only)
                ties = int(len(target) - target_only - control_only)

            records.append({
                "arm": arm,
                "comparison": f"{TARGET}_vs_{control}",
                "control": control,
                "metric": metric,
                "n": len(target),
                "target_mean": float(np.mean(target)),
                "control_mean": float(np.mean(control_values)),
                "target_median": float(np.median(target)),
                "control_median": float(np.median(control_values)),
                "difference_target_minus_control": effect,
                "statistic": statistic,
                "p_value": p,
                "target_better": target_better,
                "control_better": control_better,
                "ties": ties,
            })


results = pd.DataFrame(records)

results["bh_fdr"] = bh_adjust(
    results["p_value"].to_numpy()
)

results["bonferroni"] = bonferroni(
    results["p_value"].to_numpy()
)

results["bh_significant"] = results["bh_fdr"] < 0.05
results["bonferroni_significant"] = results["bonferroni"] < 0.05

results.to_csv(OUT, index=False)

print("\n" + "=" * 110)
print("RESULTS")
print("=" * 110)

display_cols = [
    "arm",
    "control",
    "metric",
    "target_mean",
    "control_mean",
    "difference_target_minus_control",
    "p_value",
    "bh_fdr",
    "bonferroni",
    "target_better",
    "control_better",
]

print(
    results[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

print("\n" + "=" * 110)
print("SIGNIFICANT AFTER BH-FDR < 0.05")
print("=" * 110)

sig = results[results["bh_significant"]]

if len(sig) == 0:
    print("None")
else:
    print(
        sig[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}"
        )
    )

print("\n" + "=" * 110)
print("SIGNIFICANT AFTER BONFERRONI < 0.05")
print("=" * 110)

sig_bonf = results[results["bonferroni_significant"]]

if len(sig_bonf) == 0:
    print("None")
else:
    print(
        sig_bonf[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}"
        )
    )

print("\nSaved:")
print(OUT)
