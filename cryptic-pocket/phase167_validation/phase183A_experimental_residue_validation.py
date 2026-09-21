import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from scipy.stats import wilcoxon

ROOT = Path(".")
BASE = ROOT / "phase167_validation"

PRED_FILE = BASE / "phase167_residue_predictions.csv"
MAPPING_FILE = BASE / "phase167_node_residue_mapping.csv"
GRAPH_FILE = ROOT / "phase159_input/phase159_cryptobench_anmmsf_graphs.pt"

OUT_FILE = BASE / "phase183A_experimental_residue_validation.csv"
SUMMARY_FILE = BASE / "phase183A_experimental_residue_summary.csv"
PAIRED_FILE = BASE / "phase183A_experimental_residue_paired_stats.csv"

print("=" * 110)
print("PHASE 183A — DIRECT EXPERIMENTAL RESIDUE-LEVEL VALIDATION")
print("=" * 110)

# ---------------------------------------------------------------------
# 1. LOAD PREDICTIONS
# ---------------------------------------------------------------------

pred = pd.read_csv(PRED_FILE, low_memory=False)

print("\nPredictions:", pred.shape)
print("Prediction columns:", list(pred.columns))

required_pred = [
    "apo_pdb_id",
    "apo_chain",
    "node_index",
    "probability",
    "arm",
    "seed",
]

missing = [c for c in required_pred if c not in pred.columns]

if missing:
    raise RuntimeError(f"Missing prediction columns: {missing}")

# ---------------------------------------------------------------------
# 2. LOAD EXPERIMENTAL RESIDUE MAPPING
# ---------------------------------------------------------------------

mapping = pd.read_csv(MAPPING_FILE, low_memory=False)

print("\nExperimental mapping:", mapping.shape)
print("Mapping columns:", list(mapping.columns))

required_mapping = [
    "apo_pdb_id",
    "apo_chain",
    "node_index",
    "experimental_site",
]

missing = [c for c in required_mapping if c not in mapping.columns]

if missing:
    raise RuntimeError(f"Missing mapping columns: {missing}")

# Normalize identifiers
pred["apo_pdb_id_norm"] = pred["apo_pdb_id"].astype(str).str.lower()
pred["apo_chain_norm"] = pred["apo_chain"].astype(str)

mapping["apo_pdb_id_norm"] = mapping["apo_pdb_id"].astype(str).str.lower()
mapping["apo_chain_norm"] = mapping["apo_chain"].astype(str)

# ---------------------------------------------------------------------
# 3. MERGE PREDICTIONS WITH EXPERIMENTAL SITE LABELS
# ---------------------------------------------------------------------

mapping_cols = [
    "apo_pdb_id_norm",
    "apo_chain_norm",
    "node_index",
    "experimental_site",
]

mapping_small = mapping[mapping_cols].copy()

# Ensure mapping is unique
dup = mapping_small.duplicated(
    ["apo_pdb_id_norm", "apo_chain_norm", "node_index"]
).sum()

print("\nDuplicate mapping keys:", int(dup))

if dup:
    raise RuntimeError(
        "Experimental mapping contains duplicate node keys. "
        "Do not continue until this is resolved."
    )

merged = pred.merge(
    mapping_small,
    on=[
        "apo_pdb_id_norm",
        "apo_chain_norm",
        "node_index",
    ],
    how="left",
    validate="many_to_one",
)

print("Merged:", merged.shape)

missing_labels = merged["experimental_site"].isna().sum()

print("Missing experimental labels:", int(missing_labels))

if missing_labels:
    raise RuntimeError(
        f"{missing_labels} predictions lack experimental labels. "
        "Do not silently drop them."
    )

merged["experimental_site"] = (
    merged["experimental_site"].astype(int)
)

print(
    "Experimental positives:",
    int(merged["experimental_site"].sum())
)

print(
    "Experimental positive fraction:",
    merged["experimental_site"].mean()
)

# ---------------------------------------------------------------------
# 4. SANITY CHECK
# ---------------------------------------------------------------------

print("\nExperimental label counts by arm:")

print(
    merged.groupby("arm")["experimental_site"]
    .agg(["count", "sum", "mean"])
)

print("\nExperimental label counts by seed:")

print(
    merged.groupby("seed")["experimental_site"]
    .agg(["count", "sum", "mean"])
)

# ---------------------------------------------------------------------
# 5. TOP-K EXPERIMENTAL SITE ENRICHMENT
# ---------------------------------------------------------------------

fractions = [0.01, 0.02, 0.05, 0.10, 0.20]

rows = []

print("\n" + "=" * 110)
print("TOP-K EXPERIMENTAL SITE ENRICHMENT")
print("=" * 110)

for arm, arm_df in merged.groupby("arm"):

    for seed, seed_df in arm_df.groupby("seed"):

        # Evaluate each protein independently.
        for (pdb, chain), g in seed_df.groupby(
            ["apo_pdb_id_norm", "apo_chain_norm"],
            sort=False,
        ):

            g = g.sort_values(
                "probability",
                ascending=False,
            ).copy()

            n = len(g)

            y = g["experimental_site"].to_numpy(dtype=int)

            total_positive = int(y.sum())

            if total_positive == 0:
                print(
                    "WARNING: no experimental positives:",
                    arm,
                    seed,
                    pdb,
                    chain,
                )
                continue

            background = y.mean()

            for frac in fractions:

                k = max(1, int(np.ceil(frac * n)))

                selected = y[:k]

                selected_fraction = selected.mean()

                enrichment = (
                    selected_fraction / background
                    if background > 0
                    else np.nan
                )

                tp = int(selected.sum())

                precision = tp / k

                recall = tp / total_positive

                rows.append({
                    "arm": arm,
                    "seed": seed,
                    "apo_pdb_id": pdb,
                    "apo_chain": chain,
                    "fraction": frac,
                    "k": k,
                    "n_residues": n,
                    "background_positive_fraction": background,
                    "selected_positive_fraction": selected_fraction,
                    "enrichment": enrichment,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": tp,
                    "total_experimental_positives": total_positive,
                })

topk = pd.DataFrame(rows)

print("\nTop-k rows:", topk.shape)

# ---------------------------------------------------------------------
# 6. RESIDUE-LEVEL GLOBAL METRICS
# ---------------------------------------------------------------------

print("\n" + "=" * 110)
print("RESIDUE-LEVEL PR-AUC / ROC-AUC")
print("=" * 110)

global_rows = []

for arm, arm_df in merged.groupby("arm"):

    y = arm_df["experimental_site"].to_numpy(dtype=int)
    p = arm_df["probability"].to_numpy(dtype=float)

    pr_auc = average_precision_score(y, p)
    roc_auc = roc_auc_score(y, p)

    global_rows.append({
        "arm": arm,
        "n_predictions": len(y),
        "n_positive": int(y.sum()),
        "positive_fraction": y.mean(),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    })

    print(
        f"{arm:20s} "
        f"PR-AUC={pr_auc:.6f} "
        f"ROC-AUC={roc_auc:.6f}"
    )

global_metrics = pd.DataFrame(global_rows)

# ---------------------------------------------------------------------
# 7. PROTEIN-LEVEL METRICS
# ---------------------------------------------------------------------

print("\n" + "=" * 110)
print("PROTEIN-LEVEL METRICS")
print("=" * 110)

protein_rows = []

for (arm, seed, pdb, chain), g in merged.groupby(
    ["arm", "seed", "apo_pdb_id_norm", "apo_chain_norm"],
    sort=False,
):

    y = g["experimental_site"].to_numpy(dtype=int)
    p = g["probability"].to_numpy(dtype=float)

    # A protein must contain both classes for ROC-AUC.
    if len(np.unique(y)) == 2:
        roc = roc_auc_score(y, p)
        pr = average_precision_score(y, p)
    else:
        roc = np.nan
        pr = np.nan

    protein_rows.append({
        "arm": arm,
        "seed": seed,
        "apo_pdb_id": pdb,
        "apo_chain": chain,
        "n_residues": len(y),
        "n_positive": int(y.sum()),
        "positive_fraction": y.mean(),
        "pr_auc": pr,
        "roc_auc": roc,
    })

protein = pd.DataFrame(protein_rows)

print(
    "\nProtein-level rows:",
    protein.shape
)

print("\nProtein-level summary:")

print(
    protein.groupby("arm")
    .agg(
        proteins=("apo_pdb_id", "nunique"),
        mean_pr_auc=("pr_auc", "mean"),
        median_pr_auc=("pr_auc", "median"),
        std_pr_auc=("pr_auc", "std"),
        mean_roc_auc=("roc_auc", "mean"),
        median_roc_auc=("roc_auc", "median"),
        std_roc_auc=("roc_auc", "std"),
    )
)

# ---------------------------------------------------------------------
# 8. PROTEIN-LEVEL TOP-K SUMMARY
# ---------------------------------------------------------------------

topk_summary = (
    topk.groupby(["arm", "fraction"])
    .agg(
        mean_enrichment=("enrichment", "mean"),
        median_enrichment=("enrichment", "median"),
        mean_precision=("precision", "mean"),
        median_precision=("precision", "median"),
        mean_recall=("recall", "mean"),
        median_recall=("recall", "median"),
        n_proteins=("apo_pdb_id", "nunique"),
    )
    .reset_index()
)

print("\nTop-k summary:")

for arm in sorted(topk_summary["arm"].unique()):

    print("\nARM:", arm)

    sub = topk_summary[
        topk_summary["arm"] == arm
    ].sort_values("fraction")

    print(
        sub.to_string(index=False)
    )

# ---------------------------------------------------------------------
# 9. PAIRED PROTEIN-LEVEL COMPARISONS
# ---------------------------------------------------------------------
#
# Collapse 8 seeds within each protein first.
# This prevents proteins with more observations from dominating.
# ---------------------------------------------------------------------

protein_collapsed = (
    protein.groupby(
        ["arm", "apo_pdb_id", "apo_chain"],
        as_index=False
    )
    .agg(
        pr_auc=("pr_auc", "mean"),
        roc_auc=("roc_auc", "mean"),
        positive_fraction=("positive_fraction", "mean"),
    )
)

print(
    "\nCollapsed protein-level rows:",
    protein_collapsed.shape
)

arms = sorted(
    protein_collapsed["arm"].unique()
)

pairs = [
    ("full_tda", "esm_structure"),
    ("full_model", "esm_structure"),
    ("full_model", "full_tda"),
]

paired_rows = []

for arm_a, arm_b in pairs:

    a = protein_collapsed[
        protein_collapsed["arm"] == arm_a
    ].copy()

    b = protein_collapsed[
        protein_collapsed["arm"] == arm_b
    ].copy()

    key = [
        "apo_pdb_id",
        "apo_chain",
    ]

    m = a.merge(
        b,
        on=key,
        suffixes=("_a", "_b"),
        how="inner",
    )

    print(
        f"\nPAIR: {arm_a} vs {arm_b}",
        "proteins:",
        len(m),
    )

    for metric in [
        "pr_auc",
        "roc_auc",
    ]:

        x = m[f"{metric}_a"].dropna()
        y = m.loc[x.index, f"{metric}_b"]

        if len(x) < 5:
            continue

        try:
            stat, p = wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox",
            )
        except Exception:
            stat = np.nan
            p = np.nan

        diff = float(np.mean(x - y))

        paired_rows.append({
            "arm_a": arm_a,
            "arm_b": arm_b,
            "metric": metric,
            "n_proteins": len(x),
            "mean_a": float(x.mean()),
            "mean_b": float(y.mean()),
            "median_a": float(x.median()),
            "median_b": float(y.median()),
            "mean_difference_a_minus_b": diff,
            "wilcoxon_stat": stat,
            "p_value": p,
        })

        print(
            f"{metric:10s} "
            f"{arm_a}={x.mean():.6f} "
            f"{arm_b}={y.mean():.6f} "
            f"diff={diff:+.6f} "
            f"p={p:.6g}"
        )

paired = pd.DataFrame(paired_rows)

# ---------------------------------------------------------------------
# 10. SAVE
# ---------------------------------------------------------------------

merged.to_csv(
    OUT_FILE,
    index=False,
)

summary = global_metrics.merge(
    topk_summary,
    on="arm",
    how="left",
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)

paired.to_csv(
    PAIRED_FILE,
    index=False,
)

print("\n" + "=" * 110)
print("FILES WRITTEN")
print("=" * 110)

print(OUT_FILE)
print(SUMMARY_FILE)
print(PAIRED_FILE)

print("\n" + "=" * 110)
print("PHASE 183A COMPLETE")
print("=" * 110)
