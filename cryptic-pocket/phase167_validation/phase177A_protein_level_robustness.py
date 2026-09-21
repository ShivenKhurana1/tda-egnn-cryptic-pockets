import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

INPUT = "phase167_validation/phase175A_spatial_controls.csv"

OUT_PROTEIN = "phase167_validation/phase177A_protein_level_results.csv"
OUT_STATS = "phase167_validation/phase177A_protein_level_stats.csv"

ARMS = [
    "esm_structure",
    "full_tda",
    "full_model",
]

TARGET = "sharpened_gamma10"

CONTROLS = [
    "protein_centroid",
    "random10_centroid",
    "probability_centroid_gamma1",
    "top10_centroid",
]

DCCS = ["dcc4", "dcc6", "dcc8", "dcc10"]


def bh_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)

    order = np.argsort(pvalues)
    ranked = pvalues[order]

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


def exact_mcnemar(a, b):
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)

    target_only = np.sum((a == 1) & (b == 0))
    control_only = np.sum((a == 0) & (b == 1))

    discordant = target_only + control_only

    if discordant == 0:
        return 1.0, target_only, control_only

    p = binomtest(
        target_only,
        discordant,
        0.5,
        alternative="two-sided",
    ).pvalue

    return float(p), int(target_only), int(control_only)


df = pd.read_csv(INPUT)

print("=" * 110)
print("PHASE 177A — PROTEIN-LEVEL ROBUSTNESS")
print("=" * 110)

print(f"Input rows: {len(df):,}")

# ------------------------------------------------------------------
# Collapse 8 seeds into one protein-level observation
# ------------------------------------------------------------------

protein = (
    df.groupby(
        ["apo_pdb_id", "apo_chain", "arm", "method"],
        as_index=False
    )
    .agg(
        median_center_error=("center_error", "median"),
        mean_center_error=("center_error", "mean"),
        dcc4_rate=("dcc4", "mean"),
        dcc6_rate=("dcc6", "mean"),
        dcc8_rate=("dcc8", "mean"),
        dcc10_rate=("dcc10", "mean"),
        n_seeds=("seed", "nunique"),
    )
)

protein.to_csv(OUT_PROTEIN, index=False)

print(f"Protein-level rows: {len(protein):,}")

print("\nSeeds per protein/method:")
print(
    protein["n_seeds"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ------------------------------------------------------------------
# Pair target vs controls
# ------------------------------------------------------------------

records = []

for arm in ARMS:

    arm_df = protein[protein["arm"] == arm].copy()

    for control in CONTROLS:

        target_df = (
            arm_df[arm_df["method"] == TARGET]
            .sort_values(["apo_pdb_id", "apo_chain"])
        )

        control_df = (
            arm_df[arm_df["method"] == control]
            .sort_values(["apo_pdb_id", "apo_chain"])
        )

        assert len(target_df) == len(control_df)
        assert (
            target_df[["apo_pdb_id", "apo_chain"]].values
            == control_df[["apo_pdb_id", "apo_chain"]].values
        ).all()

        # ----------------------------------------------------------
        # Continuous center error
        # ----------------------------------------------------------

        target = target_df["median_center_error"].to_numpy()
        control_values = control_df["median_center_error"].to_numpy()

        result = wilcoxon(
            target,
            control_values,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        records.append({
            "arm": arm,
            "control": control,
            "metric": "median_center_error",
            "n_proteins": len(target),
            "target_mean": np.mean(target),
            "control_mean": np.mean(control_values),
            "target_median": np.median(target),
            "control_median": np.median(control_values),
            "difference_target_minus_control":
                np.mean(target - control_values),
            "p_value": float(result.pvalue),
            "target_better": int(np.sum(target < control_values)),
            "control_better": int(np.sum(target > control_values)),
            "ties": int(np.sum(target == control_values)),
        })

        # ----------------------------------------------------------
        # DCC rates
        # ----------------------------------------------------------

        for dcc in DCCS:

            target = target_df[
                dcc + "_rate"
            ].to_numpy()

            control_values = control_df[
                dcc + "_rate"
            ].to_numpy()

            # Difference in per-protein success rate
            difference = target - control_values

            # For paired protein-level robustness, test whether
            # target tends to have greater success rate.
            result = wilcoxon(
                target,
                control_values,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            records.append({
                "arm": arm,
                "control": control,
                "metric": dcc + "_rate",
                "n_proteins": len(target),
                "target_mean": np.mean(target),
                "control_mean": np.mean(control_values),
                "target_median": np.median(target),
                "control_median": np.median(control_values),
                "difference_target_minus_control":
                    np.mean(difference),
                "p_value": float(result.pvalue),
                "target_better": int(np.sum(target > control_values)),
                "control_better": int(np.sum(target < control_values)),
                "ties": int(np.sum(target == control_values)),
            })


stats = pd.DataFrame(records)

stats["bh_fdr"] = bh_adjust(
    stats["p_value"].to_numpy()
)

stats["bonferroni"] = np.minimum(
    stats["p_value"] * len(stats),
    1.0
)

stats["bh_significant"] = stats["bh_fdr"] < 0.05
stats["bonferroni_significant"] = stats["bonferroni"] < 0.05

stats.to_csv(OUT_STATS, index=False)

# ------------------------------------------------------------------
# Print summary
# ------------------------------------------------------------------

print("\n" + "=" * 110)
print("PROTEIN-LEVEL COMPARISONS")
print("=" * 110)

print(
    stats.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

print("\n" + "=" * 110)
print("BH-FDR < 0.05")
print("=" * 110)

sig = stats[stats["bh_significant"]]

if len(sig):
    print(
        sig.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}"
        )
    )
else:
    print("None")

print("\n" + "=" * 110)
print("BONFERRONI < 0.05")
print("=" * 110)

sig = stats[stats["bonferroni_significant"]]

if len(sig):
    print(
        sig.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}"
        )
    )
else:
    print("None")

print("\nSaved:")
print(OUT_PROTEIN)
print(OUT_STATS)
