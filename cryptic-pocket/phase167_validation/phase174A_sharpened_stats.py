import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PRED_PATH = "phase167_validation/phase167_residue_predictions.csv"
CENTER_PATH = "phase167_validation/phase168A_corrected_centers.csv"

ARMS = ["esm_structure", "full_tda", "full_model"]

print("=" * 100)
print("PHASE 174A - SHARPENED CENTROID STATISTICAL VALIDATION")
print("=" * 100)


def norm_pdb(x):
    return str(x).strip().lower()


def norm_chain(x):
    return str(x).strip().upper()


pred = pd.read_csv(PRED_PATH)
centers = pd.read_csv(CENTER_PATH)

pred = pred[pred["arm"].isin(ARMS)].copy()

pred["pdb_key"] = pred["apo_pdb_id"].map(norm_pdb)
pred["chain_key"] = pred["apo_chain"].map(norm_chain)

centers["pdb_key"] = centers["apo_pdb_id"].map(norm_pdb)
centers["chain_key"] = centers["apo_chain"].map(norm_chain)

pred = pred.merge(
    centers[
        [
            "pdb_key",
            "chain_key",
            "true_center_x",
            "true_center_y",
            "true_center_z",
        ]
    ],
    on=["pdb_key", "chain_key"],
    how="inner",
)

# Load graph coordinates
import torch

GRAPH_PATH = "phase159_input/phase159_cryptobench_anmmsf_graphs.pt"
obj = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
dataset = obj["dataset"]

graph_pos = {}

for g in dataset:
    key = (norm_pdb(g.pdb_id), norm_chain(g.chain_id))
    graph_pos[key] = g.pos.detach().cpu().numpy()


records = []

for (pdb, chain, arm, seed), group in pred.groupby(
    ["pdb_key", "chain_key", "arm", "seed"],
    sort=False
):

    key = (pdb, chain)

    if key not in graph_pos:
        continue

    pos = graph_pos[key]

    node_idx = group["node_index"].to_numpy(dtype=int)

    valid = (
        (node_idx >= 0) &
        (node_idx < len(pos))
    )

    group = group.loc[valid]

    if len(group) == 0:
        continue

    node_idx = group["node_index"].to_numpy(dtype=int)
    probs = np.clip(
        group["probability"].to_numpy(dtype=float),
        1e-12,
        None
    )

    coords = pos[node_idx]

    target = np.array([
        group["true_center_x"].iloc[0],
        group["true_center_y"].iloc[0],
        group["true_center_z"].iloc[0],
    ])

    def error(center):
        return float(np.linalg.norm(center - target))

    # ---------------------------------------------------------
    # 1. Learned localization
    # ---------------------------------------------------------
    learned = np.array([
        group["predicted_center_x"].iloc[0],
        group["predicted_center_y"].iloc[0],
        group["predicted_center_z"].iloc[0],
    ])

    # ---------------------------------------------------------
    # 2. Top 10% centroid
    # ---------------------------------------------------------
    n = len(group)
    k = max(1, int(np.ceil(0.10 * n)))

    order = np.argsort(-probs)
    top_idx = order[:k]

    top10_center = coords[top_idx].mean(axis=0)

    # ---------------------------------------------------------
    # 3. Sharpened centroid gamma=10
    # ---------------------------------------------------------
    weights = probs ** 10
    weights /= weights.sum()

    sharp_center = np.sum(
        coords * weights[:, None],
        axis=0
    )

    centers_dict = {
        "learned_localization": learned,
        "top10_centroid": top10_center,
        "sharpened_gamma10": sharp_center,
    }

    for method, center in centers_dict.items():

        e = error(center)

        records.append({
            "apo_pdb_id": group["apo_pdb_id"].iloc[0],
            "apo_chain": group["apo_chain"].iloc[0],
            "arm": arm,
            "seed": seed,
            "method": method,
            "center_error": e,
            "dcc4": float(e <= 4),
            "dcc6": float(e <= 6),
            "dcc8": float(e <= 8),
            "dcc10": float(e <= 10),
        })


df = pd.DataFrame(records)

print(f"Evaluated records: {len(df):,}")

# -------------------------------------------------------------
# Summary
# -------------------------------------------------------------

summary = (
    df.groupby(["arm", "method"])
      .agg(
          n=("center_error", "size"),
          mean_center_error=("center_error", "mean"),
          median_center_error=("center_error", "median"),
          dcc4=("dcc4", "mean"),
          dcc6=("dcc6", "mean"),
          dcc8=("dcc8", "mean"),
          dcc10=("dcc10", "mean"),
      )
      .reset_index()
)

summary_path = "phase167_validation/phase174A_sharpened_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n" + "=" * 100)
print("LOCALIZATION SUMMARY")
print("=" * 100)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# -------------------------------------------------------------
# Paired tests
# -------------------------------------------------------------

comparisons = [
    ("sharpened_gamma10", "learned_localization"),
    ("sharpened_gamma10", "top10_centroid"),
]

metrics = [
    "center_error",
    "dcc4",
    "dcc6",
    "dcc8",
    "dcc10",
]

test_rows = []

for arm in ARMS:

    arm_df = df[df["arm"] == arm]

    for method_a, method_b in comparisons:

        a = arm_df[arm_df["method"] == method_a]
        b = arm_df[arm_df["method"] == method_b]

        merge_cols = [
            "apo_pdb_id",
            "apo_chain",
            "seed",
        ]

        merged = a.merge(
            b,
            on=merge_cols,
            suffixes=("_a", "_b")
        )

        for metric in metrics:

            x = merged[f"{metric}_a"].to_numpy()
            y = merged[f"{metric}_b"].to_numpy()

            # For DCC metrics, test the paired binary outcomes.
            # Wilcoxon is still used consistently with the earlier analysis.
            diff = x - y

            if np.allclose(diff, 0):
                p = 1.0
                statistic = 0.0
            else:
                result = wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox"
                )

                statistic = float(result.statistic)
                p = float(result.pvalue)

            test_rows.append({
                "arm": arm,
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "n_pairs": len(merged),
                "mean_a": float(np.mean(x)),
                "mean_b": float(np.mean(y)),
                "median_a": float(np.median(x)),
                "median_b": float(np.median(y)),
                "mean_difference_a_minus_b": float(np.mean(x - y)),
                "wilcoxon_statistic": statistic,
                "p_value": p,
            })


tests = pd.DataFrame(test_rows)

# -------------------------------------------------------------
# Multiple testing correction
# Benjamini-Hochberg FDR + Bonferroni
# -------------------------------------------------------------

def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)

    order = np.argsort(pvals)
    ranked = pvals[order]

    adjusted = np.empty(m)

    running = 1.0

    for i in range(m - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * m / rank
        running = min(running, value)
        adjusted[i] = running

    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)

    return out


tests["bh_fdr"] = bh_fdr(tests["p_value"].values)
tests["bonferroni"] = np.minimum(
    tests["p_value"].values * len(tests),
    1.0
)

tests["bh_significant_0.05"] = tests["bh_fdr"] < 0.05
tests["bonferroni_significant_0.05"] = tests["bonferroni"] < 0.05

tests_path = "phase167_validation/phase174A_sharpened_stats.csv"
tests.to_csv(tests_path, index=False)

print("\n" + "=" * 100)
print("PAIRED STATISTICAL TESTS")
print("=" * 100)

print(
    tests.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

print("\n" + "=" * 100)
print("KEY COMPARISON: SHARPENED γ=10 vs LEARNED LOCALIZATION")
print("=" * 100)

print(
    tests[
        tests["method_b"] == "learned_localization"
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

print("\n" + "=" * 100)
print("KEY COMPARISON: SHARPENED γ=10 vs TOP-10% CENTROID")
print("=" * 100)

print(
    tests[
        tests["method_b"] == "top10_centroid"
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

print("\nSaved:")
print(summary_path)
print(tests_path)
