import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

BASE = "phase167_validation"

RESIDUE = os.path.join(
    BASE, "phase183A_experimental_residue_validation.csv"
)

LOC = os.path.join(
    BASE, "phase185_experimental_residue_localization.csv"
)

OUT_SUMMARY = os.path.join(
    BASE, "phase187_subgroup_summary.csv"
)

OUT_CORR = os.path.join(
    BASE, "phase187_correlations.csv"
)

OUT_STATS = os.path.join(
    BASE, "phase187_subgroup_tests.csv"
)

print("=" * 100)
print("PHASE 187 — FAILURE-MODE / SUBGROUP ANALYSIS")
print("=" * 100)

# ============================================================================
# LOAD
# ============================================================================

res = pd.read_csv(RESIDUE)
loc = pd.read_csv(LOC)

print("Residue data:", res.shape)
print("Localization data:", loc.shape)

# Normalize identifiers BEFORE every merge.
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

# ============================================================================
# PROTEIN-LEVEL EXPERIMENTAL PROPERTIES
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

print("Unique proteins:", len(protein))

# ============================================================================
# RESIDUE-LEVEL PERFORMANCE → ONE VALUE PER PROTEIN × ARM
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

    # Rank-based ROC-AUC.
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

    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }

    for frac in [0.01, 0.05, 0.10, 0.20]:

        n_top = max(
            1,
            int(np.ceil(len(g) * frac))
        )

        top_y = y_sorted[:n_top]

        precision_k = top_y.mean()
        baseline = y.mean()

        enrichment = (
            precision_k / baseline
            if baseline > 0
            else np.nan
        )

        metrics[
            f"enrichment_{int(frac * 100)}"
        ] = enrichment

    rows.append({
        "apo_pdb_id": pdb,
        "apo_chain": chain,
        "arm": arm,
        **metrics,
    })

perf = pd.DataFrame(rows)

# ============================================================================
# GAMMA10 LOCALIZATION
# ============================================================================

gamma = loc[
    loc["method"] == "gamma10_centroid"
].copy()

if gamma.empty:
    raise RuntimeError(
        "No gamma10_centroid rows found."
    )

# Seed collapse: one localization value per protein × arm.
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

print("Gamma10 seed-collapsed rows:", gamma.shape)

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

print(
    "Merged proteins:",
    df[
        ["apo_pdb_id", "apo_chain"]
    ].drop_duplicates().shape[0]
)

if len(df) == 0:
    raise RuntimeError(
        "Merged dataset is empty after identifier normalization."
    )

# ============================================================================
# TDA GAIN OVER ESM
# ============================================================================

esm = (
    df[df["arm"] == "esm_structure"]
    .set_index(["apo_pdb_id", "apo_chain"])
)

tda = (
    df[df["arm"] == "full_tda"]
    .set_index(["apo_pdb_id", "apo_chain"])
)

common = esm.index.intersection(tda.index)

gain = pd.DataFrame(index=common)

gain["pr_auc_gain_tda_vs_esm"] = (
    tda.loc[common, "pr_auc"] -
    esm.loc[common, "pr_auc"]
)

gain["roc_auc_gain_tda_vs_esm"] = (
    tda.loc[common, "roc_auc"] -
    esm.loc[common, "roc_auc"]
)

gain["enrichment10_gain_tda_vs_esm"] = (
    tda.loc[common, "enrichment_10"] -
    esm.loc[common, "enrichment_10"]
)

gain = gain.reset_index()

# ============================================================================
# QUARTILE DEFINITIONS
# ============================================================================

protein["protein_size_quartile"] = pd.qcut(
    protein["n_nodes"],
    q=4,
    labels=[
        "Q1_small",
        "Q2",
        "Q3",
        "Q4_large",
    ],
    duplicates="drop",
)

protein["pocket_size_quartile"] = pd.qcut(
    protein["n_experimental"],
    q=4,
    labels=[
        "Q1_small",
        "Q2",
        "Q3",
        "Q4_large",
    ],
    duplicates="drop",
)

protein["pocket_fraction_quartile"] = pd.qcut(
    protein["experimental_fraction"],
    q=4,
    labels=[
        "Q1_low",
        "Q2",
        "Q3",
        "Q4_high",
    ],
    duplicates="drop",
)

# ============================================================================
# MERGE SUBGROUP LABELS
# ============================================================================

df = df.drop(
    columns=[
        "protein_size_quartile",
        "pocket_size_quartile",
        "pocket_fraction_quartile",
    ],
    errors="ignore",
)

df = df.merge(
    protein[
        [
            "apo_pdb_id",
            "apo_chain",
            "protein_size_quartile",
            "pocket_size_quartile",
            "pocket_fraction_quartile",
        ]
    ],
    on=["apo_pdb_id", "apo_chain"],
    how="left",
)

# ============================================================================
# SUBGROUP SUMMARIES
# ============================================================================

summary_rows = []

for arm in sorted(df["arm"].unique()):

    sub = df[df["arm"] == arm]

    for grouping in [
        "protein_size_quartile",
        "pocket_size_quartile",
        "pocket_fraction_quartile",
    ]:

        for group, g in sub.groupby(
            grouping,
            observed=True
        ):

            summary_rows.append({

                "arm": arm,

                "grouping": grouping,

                "group": str(group),

                "n_proteins":
                    g[
                        ["apo_pdb_id", "apo_chain"]
                    ].drop_duplicates().shape[0],

                "mean_pr_auc":
                    g["pr_auc"].mean(),

                "median_pr_auc":
                    g["pr_auc"].median(),

                "mean_roc_auc":
                    g["roc_auc"].mean(),

                "median_roc_auc":
                    g["roc_auc"].median(),

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

summary = pd.DataFrame(summary_rows)

# ============================================================================
# SPEARMAN CORRELATIONS
# ============================================================================

corr_rows = []

for arm in sorted(df["arm"].unique()):

    sub = df[df["arm"] == arm]

    variables = {
        "protein_size": "n_nodes",
        "experimental_pocket_size": "n_experimental",
        "experimental_pocket_fraction":
            "experimental_fraction",
    }

    outcomes = {
        "pr_auc": "pr_auc",
        "roc_auc": "roc_auc",
        "center_error": "center_error",
        "dcc4": "dcc4",
        "dcc8": "dcc8",
        "dcc10": "dcc10",
    }

    for x_name, x_col in variables.items():

        for y_name, y_col in outcomes.items():

            valid = sub[
                [x_col, y_col]
            ].dropna()

            if len(valid) >= 10:

                rho, p = spearmanr(
                    valid[x_col],
                    valid[y_col]
                )

            else:
                rho, p = np.nan, np.nan

            corr_rows.append({

                "arm": arm,

                "predictor": x_name,

                "outcome": y_name,

                "n": len(valid),

                "spearman_rho": rho,

                "p_value": p,
            })

correlations = pd.DataFrame(corr_rows)

# ============================================================================
# KRUSKAL-WALLIS SUBGROUP TESTS
# ============================================================================

test_rows = []

for arm in sorted(df["arm"].unique()):

    sub = df[df["arm"] == arm]

    for grouping in [
        "protein_size_quartile",
        "pocket_size_quartile",
        "pocket_fraction_quartile",
    ]:

        for metric in [
            "pr_auc",
            "roc_auc",
            "center_error",
            "dcc4",
            "dcc8",
            "dcc10",
        ]:

            groups = []

            for _, g in sub.groupby(
                grouping,
                observed=True
            ):

                vals = g[
                    metric
                ].dropna().to_numpy()

                if len(vals) > 0:
                    groups.append(vals)

            if len(groups) >= 2:

                stat, p = kruskal(
                    *groups
                )

            else:

                stat, p = np.nan, np.nan

            test_rows.append({

                "arm": arm,

                "grouping": grouping,

                "metric": metric,

                "kruskal_stat": stat,

                "p_value": p,
            })

tests = pd.DataFrame(test_rows)

# ============================================================================
# TDA GAIN CORRELATIONS
# ============================================================================

gain2 = gain.merge(
    protein[
        [
            "apo_pdb_id",
            "apo_chain",
            "n_nodes",
            "n_experimental",
            "experimental_fraction",
        ]
    ],
    on=["apo_pdb_id", "apo_chain"],
    how="inner",
)

gain_rows = []

for predictor in [
    "n_nodes",
    "n_experimental",
    "experimental_fraction",
]:

    for outcome in [
        "pr_auc_gain_tda_vs_esm",
        "roc_auc_gain_tda_vs_esm",
        "enrichment10_gain_tda_vs_esm",
    ]:

        valid = gain2[
            [predictor, outcome]
        ].dropna()

        if len(valid) >= 10:

            rho, p = spearmanr(
                valid[predictor],
                valid[outcome]
            )

        else:

            rho, p = np.nan, np.nan

        gain_rows.append({

            "analysis": "TDA_gain",

            "predictor": predictor,

            "outcome": outcome,

            "n": len(valid),

            "spearman_rho": rho,

            "p_value": p,
        })

gain_corr = pd.DataFrame(gain_rows)

# ============================================================================
# SAVE
# ============================================================================

summary.to_csv(
    OUT_SUMMARY,
    index=False
)

correlations.to_csv(
    OUT_CORR,
    index=False
)

gain_corr.to_csv(
    OUT_STATS.replace(
        "phase187_subgroup_tests.csv",
        "phase187_tda_gain_correlations.csv"
    ),
    index=False
)

tests.to_csv(
    OUT_STATS,
    index=False
)

# ============================================================================
# OUTPUT
# ============================================================================

print()
print("=" * 100)
print("PHASE 187 COMPLETE")
print("=" * 100)

print("\nFiles:")
print(OUT_SUMMARY)
print(OUT_CORR)
print(OUT_STATS)

print("\nTop correlations by absolute Spearman rho:")

print(
    correlations
    .assign(
        abs_rho=lambda x:
            x["spearman_rho"].abs()
    )
    .sort_values(
        "abs_rho",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)

print("\nStrongest TDA-gain correlations:")

print(
    gain_corr
    .assign(
        abs_rho=lambda x:
            x["spearman_rho"].abs()
    )
    .sort_values(
        "abs_rho",
        ascending=False
    )
    .to_string(index=False)
)

print("\nSubgroup tests with p < 0.05:")

sig = tests[
    tests["p_value"].notna() &
    (tests["p_value"] < 0.05)
].sort_values("p_value")

if len(sig) == 0:
    print("None")

else:
    print(
        sig
        .head(50)
        .to_string(index=False)
    )
