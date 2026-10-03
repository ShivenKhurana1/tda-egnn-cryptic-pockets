import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = "phase167_validation"

RESIDUE = os.path.join(
    BASE,
    "phase183A_experimental_residue_validation.csv"
)

LOC = os.path.join(
    BASE,
    "phase185_experimental_residue_localization.csv"
)

OUT = os.path.join(
    BASE,
    "phase188_pocket_size_curve.csv"
)

# ============================================================================
# LOAD
# ============================================================================

res = pd.read_csv(RESIDUE)
loc = pd.read_csv(LOC)

for df in [res, loc]:
    df["apo_pdb_id"] = (
        df["apo_pdb_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["apo_chain"] = (
        df["apo_chain"]
        .astype(str)
        .str.strip()
    )

print("=" * 100)
print("PHASE 188 - CONTINUOUS POCKET-SIZE PERFORMANCE ANALYSIS")
print("=" * 100)

# ============================================================================
# PROTEIN PROPERTIES
# ============================================================================

protein = (
    res.groupby(
        ["apo_pdb_id", "apo_chain"],
        as_index=False
    )
    .agg(
        n_nodes=("node_index", "count"),
        n_experimental=("experimental_site", "sum"),
    )
)

protein["experimental_fraction"] = (
    protein["n_experimental"] /
    protein["n_nodes"]
)

print("Proteins:", len(protein))

# ============================================================================
# RESIDUE PERFORMANCE PER PROTEIN × ARM
# ============================================================================

rows = []

for (pdb, chain, arm), g in res.groupby(
    ["apo_pdb_id", "apo_chain", "arm"]
):

    y = g["experimental_site"].to_numpy(dtype=int)
    p = g["probability"].to_numpy(dtype=float)

    order = np.argsort(-p)
    y_sorted = y[order]

    tp = np.cumsum(y_sorted)
    k = np.arange(1, len(y_sorted) + 1)

    precision = tp / k
    total_pos = y.sum()

    if total_pos > 0:
        pr_auc = (
            precision[y_sorted == 1].sum() /
            total_pos
        )
    else:
        pr_auc = np.nan

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    if len(pos) > 0 and len(neg) > 0:

        ranks = (
            pd.Series(p)
            .rank(method="average")
            .to_numpy()
        )

        rank_sum = ranks[pos].sum()

        roc_auc = (
            rank_sum -
            len(pos) * (len(pos) + 1) / 2
        ) / (len(pos) * len(neg))

    else:
        roc_auc = np.nan

    rows.append({
        "apo_pdb_id": pdb,
        "apo_chain": chain,
        "arm": arm,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    })

perf = pd.DataFrame(rows)

# ============================================================================
# GAMMA10 LOCALIZATION
# ============================================================================

gamma = loc[
    loc["method"] == "gamma10_centroid"
].copy()

gamma = (
    gamma.groupby(
        ["apo_pdb_id", "apo_chain", "arm"],
        as_index=False
    )
    .agg(
        center_error=("center_error", "mean"),
        dcc4=("dcc4", "mean"),
        dcc6=("dcc6", "mean"),
        dcc8=("dcc8", "mean"),
        dcc10=("dcc10", "mean"),
    )
)

# ============================================================================
# MERGE
# ============================================================================

df = (
    perf
    .merge(
        protein,
        on=["apo_pdb_id", "apo_chain"],
        how="inner"
    )
    .merge(
        gamma,
        on=["apo_pdb_id", "apo_chain", "arm"],
        how="inner"
    )
)

print("Merged rows:", df.shape)

if len(df) != 555:
    raise RuntimeError(
        f"Expected 555 protein × arm rows, got {len(df)}"
    )

# ============================================================================
# CONTINUOUS CORRELATIONS
# ============================================================================

print()
print("=" * 100)
print("CONTINUOUS CORRELATIONS")
print("=" * 100)

corr_rows = []

for arm in sorted(df["arm"].unique()):

    sub = df[df["arm"] == arm]

    print()
    print("ARM:", arm)

    for predictor, pcol in [
        ("experimental_pocket_size", "n_experimental"),
        ("experimental_pocket_fraction", "experimental_fraction"),
        ("protein_size", "n_nodes"),
    ]:

        for outcome, ocol in [
            ("PR_AUC", "pr_auc"),
            ("ROC_AUC", "roc_auc"),
            ("center_error", "center_error"),
            ("DCC4", "dcc4"),
            ("DCC8", "dcc8"),
            ("DCC10", "dcc10"),
        ]:

            valid = sub[
                [pcol, ocol]
            ].dropna()

            rho, p = spearmanr(
                valid[pcol],
                valid[ocol]
            )

            corr_rows.append({
                "arm": arm,
                "predictor": predictor,
                "outcome": outcome,
                "n": len(valid),
                "spearman_rho": rho,
                "p_value": p,
            })

            print(
                f"{predictor:30s} -> "
                f"{outcome:15s} "
                f"rho={rho:+.4f} "
                f"p={p:.4e}"
            )

# ============================================================================
# POCKET-SIZE BINS
# ============================================================================

# Fixed percentile bins, same 185 proteins for all arms.
protein["pocket_percentile"] = (
    protein["n_experimental"]
    .rank(method="average", pct=True)
)

protein["pocket_bin"] = pd.cut(
    protein["pocket_percentile"],
    bins=[
        0.0,
        0.20,
        0.40,
        0.60,
        0.80,
        1.0,
    ],
    labels=[
        "P0-20",
        "P20-40",
        "P40-60",
        "P60-80",
        "P80-100",
        ],
    include_lowest=True,
)

df = df.drop(
    columns=["pocket_percentile", "pocket_bin"],
    errors="ignore"
)

df = df.merge(
    protein[
        [
            "apo_pdb_id",
            "apo_chain",
            "pocket_percentile",
            "pocket_bin",
        ]
    ],
    on=["apo_pdb_id", "apo_chain"],
    how="left",
)

# ============================================================================
# BIN SUMMARY
# ============================================================================

bin_rows = []

for arm in sorted(df["arm"].unique()):

    sub = df[df["arm"] == arm]

    for bin_name, g in sub.groupby(
        "pocket_bin",
        observed=True
    ):

        bin_rows.append({
            "arm": arm,
            "pocket_bin": str(bin_name),
            "n_proteins":
                g[
                    ["apo_pdb_id", "apo_chain"]
                ].drop_duplicates().shape[0],

            "mean_pocket_size":
                g["n_experimental"].mean(),

            "median_pocket_size":
                g["n_experimental"].median(),

            "mean_pocket_fraction":
                g["experimental_fraction"].mean(),

            "mean_pr_auc":
                g["pr_auc"].mean(),

            "median_pr_auc":
                g["pr_auc"].median(),

            "mean_roc_auc":
                g["roc_auc"].mean(),

            "mean_center_error":
                g["center_error"].mean(),

            "median_center_error":
                g["center_error"].median(),

            "dcc4":
                g["dcc4"].mean(),

            "dcc8":
                g["dcc8"].mean(),

            "dcc10":
                g["dcc10"].mean(),
        })

bins = pd.DataFrame(bin_rows)

# ============================================================================
# SAVE
# ============================================================================

out = pd.concat(
    [
        pd.DataFrame(corr_rows),
        bins.assign(
            predictor="POCKET_SIZE_BIN",
            outcome="BIN_SUMMARY"
        ),
    ],
    ignore_index=True,
)

out.to_csv(
    OUT,
    index=False
)

# ============================================================================
# PRINT BIN TABLE
# ============================================================================

print()
print("=" * 100)
print("POCKET-SIZE BIN PERFORMANCE")
print("=" * 100)

print(
    bins.to_string(index=False)
)

print()
print("=" * 100)
print("PHASE 188 COMPLETE")
print("=" * 100)

print("Output:")
print(OUT)
