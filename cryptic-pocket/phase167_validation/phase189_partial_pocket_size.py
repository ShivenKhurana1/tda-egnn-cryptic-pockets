import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

BASE = "phase167_validation"

RESIDUE_FILE = os.path.join(
    BASE, "phase183A_experimental_residue_validation.csv"
)

LOCALIZATION_FILE = os.path.join(
    BASE, "phase185_experimental_residue_localization.csv"
)

OUTPUT = os.path.join(
    BASE, "phase189_partial_pocket_size.csv"
)

ARMS = ["esm_structure", "full_model", "full_tda"]

OUTCOMES = [
    "PR_AUC",
    "ROC_AUC",
    "center_error",
    "dcc4",
    "dcc8",
    "dcc10",
]


def partial_spearman(x, y, z):
    """
    Partial Spearman correlation.

    Rank-transform x, y, and z, then regress x and y
    separately on z and correlate the residuals.
    """
    x = rankdata(np.asarray(x, dtype=float))
    y = rankdata(np.asarray(y, dtype=float))
    z = rankdata(np.asarray(z, dtype=float))

    Z = np.column_stack([
        np.ones(len(z)),
        z
    ])

    bx = np.linalg.lstsq(Z, x, rcond=None)[0]
    by = np.linalg.lstsq(Z, y, rcond=None)[0]

    rx = x - Z @ bx
    ry = y - Z @ by

    rho, p = spearmanr(rx, ry)

    return rho, p


# ============================================================================
# LOAD
# ============================================================================

print("=" * 90)
print("PHASE 189 - PARTIAL POCKET-SIZE ANALYSIS")
print("=" * 90)

print("\nLoading residue validation...")
res = pd.read_csv(RESIDUE_FILE)

print("Loading localization...")
loc = pd.read_csv(LOCALIZATION_FILE)

print("Residue data:", res.shape)
print("Localization data:", loc.shape)


# ============================================================================
# NORMALIZE IDENTIFIERS
# ============================================================================

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
# RESIDUE-LEVEL -> PROTEIN-LEVEL
# ============================================================================

print("\nCollapsing residue predictions to protein level...")

protein_rows = []

group_cols = [
    "apo_pdb_id",
    "apo_chain",
    "arm",
    "seed",
]

for keys, sub in res.groupby(group_cols):

    pdb_id, chain, arm, seed = keys

    y = sub["experimental_site"].astype(int).values
    p = sub["probability"].astype(float).values

    # PR-AUC
    from sklearn.metrics import average_precision_score, roc_auc_score

    try:
        pr_auc = average_precision_score(y, p)
    except Exception:
        pr_auc = np.nan

    try:
        roc_auc = roc_auc_score(y, p)
    except Exception:
        roc_auc = np.nan

    n_experimental = int(y.sum())
    n_nodes = len(y)

    protein_rows.append({
        "apo_pdb_id": pdb_id,
        "apo_chain": chain,
        "arm": arm,
        "seed": seed,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "n_experimental": n_experimental,
        "n_nodes": n_nodes,
        "experimental_pocket_fraction":
            n_experimental / n_nodes
            if n_nodes > 0 else np.nan,
    })


protein = pd.DataFrame(protein_rows)

print("Seed-level protein rows:", protein.shape)
print("Unique proteins:", protein[["apo_pdb_id", "apo_chain"]].drop_duplicates().shape[0])


# ============================================================================
# SEED COLLAPSE
# ============================================================================

print("\nCollapsing 8 seeds to one row per protein × arm...")

protein_metrics = (
    protein
    .groupby(
        ["apo_pdb_id", "apo_chain", "arm"],
        as_index=False
    )
    .agg({
        "PR_AUC": "mean",
        "ROC_AUC": "mean",
        "n_experimental": "first",
        "n_nodes": "first",
        "experimental_pocket_fraction": "first",
    })
)

print("Collapsed residue rows:", protein_metrics.shape)


# ============================================================================
# LOCALIZATION
# ============================================================================

print("\nProcessing gamma10 localization...")

# Phase 185 calls the method gamma10_centroid.
loc_gamma = loc[
    loc["method"].astype(str) == "gamma10_centroid"
].copy()

print("Gamma10 rows:", loc_gamma.shape)

loc_gamma = (
    loc_gamma
    .groupby(
        ["apo_pdb_id", "apo_chain", "arm"],
        as_index=False
    )
    .agg({
        "center_error": "mean",
        "dcc4": "mean",
        "dcc6": "mean",
        "dcc8": "mean",
        "dcc10": "mean",
    })
)

print("Collapsed localization rows:", loc_gamma.shape)


# ============================================================================
# MERGE
# ============================================================================

print("\nMerging...")

df = protein_metrics.merge(
    loc_gamma,
    on=["apo_pdb_id", "apo_chain", "arm"],
    how="inner",
    validate="one_to_one",
)

print("Merged:", df.shape)
print(
    "Unique proteins:",
    df[["apo_pdb_id", "apo_chain"]].drop_duplicates().shape[0]
)

# Rename for clarity
df = df.rename(columns={
    "n_nodes": "protein_size",
    "n_experimental": "experimental_pocket_size",
})

expected = 185 * 3

if len(df) != expected:
    raise RuntimeError(
        f"Expected {expected} protein×arm rows, got {len(df)}"
    )


# ============================================================================
# PARTIAL SPEARMAN ANALYSIS
# ============================================================================

rows = []

for arm in ARMS:

    sub = df[df["arm"] == arm].copy()

    print("\n" + "=" * 90)
    print(f"ARM: {arm}")
    print("=" * 90)

    for predictor in [
        "experimental_pocket_size",
        "experimental_pocket_fraction",
    ]:

        for outcome in OUTCOMES:

            temp = sub[
                [
                    predictor,
                    "protein_size",
                    outcome,
                ]
            ].dropna()

            rho, p = partial_spearman(
                temp[predictor].values,
                temp[outcome].values,
                temp["protein_size"].values,
            )

            rows.append({
                "arm": arm,
                "predictor": predictor,
                "outcome": outcome,
                "control": "protein_size",
                "n": len(temp),
                "partial_spearman_rho": rho,
                "p_value": p,
            })

            print(
                f"{predictor:30s} -> "
                f"{outcome:15s} "
                f"partial_rho={rho:+.4f} "
                f"p={p:.4e}"
            )


# ============================================================================
# SAVE
# ============================================================================

out = pd.DataFrame(rows)

out.to_csv(OUTPUT, index=False)

print("\n" + "=" * 90)
print("PHASE 189 COMPLETE")
print("=" * 90)
print(f"Rows: {len(out)}")
print(f"Saved: {OUTPUT}")
