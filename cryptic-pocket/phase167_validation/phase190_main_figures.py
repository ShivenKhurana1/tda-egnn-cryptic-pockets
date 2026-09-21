import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr

BASE = "phase167_validation"

RESIDUE_FILE = os.path.join(
    BASE, "phase183A_experimental_residue_validation.csv"
)

LOCALIZATION_FILE = os.path.join(
    BASE, "phase185_experimental_residue_localization.csv"
)

PARTIAL_FILE = os.path.join(
    BASE, "phase189_partial_pocket_size.csv"
)

OUTDIR = os.path.join(BASE, "phase190_figures")
os.makedirs(OUTDIR, exist_ok=True)

ARMS = ["esm_structure", "full_model", "full_tda"]

DISPLAY = {
    "esm_structure": "ESM + Structure",
    "full_tda": "Full TDA",
    "full_model": "Full Model",
}


# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 90)
print("PHASE 190 — MAIN PAPER FIGURES")
print("=" * 90)

print("\nLoading data...")

res = pd.read_csv(RESIDUE_FILE)
loc = pd.read_csv(LOCALIZATION_FILE)
partial = pd.read_csv(PARTIAL_FILE)

print("Residue:", res.shape)
print("Localization:", loc.shape)
print("Partial correlations:", partial.shape)


# ============================================================================
# NORMALIZE IDS
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
# BUILD PROTEIN-LEVEL RESIDUE PERFORMANCE
# ============================================================================

print("\nBuilding protein-level residue performance...")

rows = []

for keys, sub in res.groupby(
    ["apo_pdb_id", "apo_chain", "arm", "seed"]
):

    pdb_id, chain, arm, seed = keys

    y = sub["experimental_site"].astype(int).values
    p = sub["probability"].astype(float).values

    pr_auc = average_precision_score(y, p)
    roc_auc = roc_auc_score(y, p)

    rows.append({
        "apo_pdb_id": pdb_id,
        "apo_chain": chain,
        "arm": arm,
        "seed": seed,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "n_experimental": int(y.sum()),
        "n_nodes": len(y),
    })

seed_df = pd.DataFrame(rows)

# Collapse seeds
protein = (
    seed_df
    .groupby(
        ["apo_pdb_id", "apo_chain", "arm"],
        as_index=False
    )
    .agg({
        "PR_AUC": "mean",
        "ROC_AUC": "mean",
        "n_experimental": "first",
        "n_nodes": "first",
    })
)

protein["experimental_pocket_size"] = protein["n_experimental"]

protein["experimental_pocket_fraction"] = (
    protein["n_experimental"] /
    protein["n_nodes"]
)

protein["protein_size"] = protein["n_nodes"]

print("Protein-level rows:", protein.shape)
print(
    "Unique proteins:",
    protein[["apo_pdb_id", "apo_chain"]]
    .drop_duplicates()
    .shape[0]
)


# ============================================================================
# BUILD LOCALIZATION DATA
# ============================================================================

print("\nBuilding protein-level localization...")

loc_gamma = loc[
    loc["method"].astype(str) == "gamma10_centroid"
].copy()

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

print("Localization rows:", loc_gamma.shape)


# ============================================================================
# MERGE
# ============================================================================

df = protein.merge(
    loc_gamma,
    on=["apo_pdb_id", "apo_chain", "arm"],
    how="inner",
    validate="one_to_one",
)

print("Final merged rows:", df.shape)

if len(df) != 555:
    raise RuntimeError(
        f"Expected 555 protein×arm rows, got {len(df)}"
    )


# ============================================================================
# FIGURE 1
# POCKET SIZE vs PR-AUC
# ============================================================================

print("\nGenerating Figure 1...")

fig, ax = plt.subplots(figsize=(8, 6))

for arm in ARMS:

    sub = df[df["arm"] == arm].copy()

    x = sub["experimental_pocket_size"].values
    y = sub["PR_AUC"].values

    # Raw Spearman for plotted relationship
    rho, p = spearmanr(x, y)

    ax.scatter(
        x,
        y,
        s=30,
        alpha=0.45,
        label=f"{DISPLAY[arm]} ρ={rho:.3f}"
    )

    # Linear visualization trend
    coef = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 200)
    yline = coef[0] * xline + coef[1]

    ax.plot(
        xline,
        yline,
        linewidth=2
    )

ax.set_xlabel(
    "Experimental pocket size (ligand-contact residues)"
)

ax.set_ylabel(
    "Protein-level PR-AUC"
)

ax.set_title(
    "Cryptic-pocket prediction performance increases with pocket size"
)

ax.legend(
    frameon=False
)

ax.grid(
    alpha=0.2
)

fig.tight_layout()

path1 = os.path.join(
    OUTDIR,
    "figure1_pocket_size_vs_pr_auc.png"
)

fig.savefig(
    path1,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:", path1)


# ============================================================================
# FIGURE 2
# POCKET SIZE vs CENTER ERROR
# ============================================================================

print("\nGenerating Figure 2...")

fig, ax = plt.subplots(figsize=(8, 6))

for arm in ARMS:

    sub = df[df["arm"] == arm].copy()

    x = sub["experimental_pocket_size"].values
    y = sub["center_error"].values

    rho, p = spearmanr(x, y)

    ax.scatter(
        x,
        y,
        s=30,
        alpha=0.45,
        label=f"{DISPLAY[arm]} ρ={rho:.3f}"
    )

    coef = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 200)
    yline = coef[0] * xline + coef[1]

    ax.plot(
        xline,
        yline,
        linewidth=2
    )

ax.set_xlabel(
    "Experimental pocket size (ligand-contact residues)"
)

ax.set_ylabel(
    "Gamma10 localization error (Å)"
)

ax.set_title(
    "Pocket localization improves as experimental pocket size increases"
)

ax.legend(
    frameon=False
)

ax.grid(
    alpha=0.2
)

fig.tight_layout()

path2 = os.path.join(
    OUTDIR,
    "figure2_pocket_size_vs_center_error.png"
)

fig.savefig(
    path2,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:", path2)


# ============================================================================
# FIGURE 3
# PR-AUC ACROSS POCKET-SIZE QUINTILES
# ============================================================================

print("\nGenerating Figure 3...")

# Construct identical quintile boundaries using the 185 proteins
base_proteins = (
    df[
        ["apo_pdb_id", "apo_chain", "experimental_pocket_size"]
    ]
    .drop_duplicates()
    .copy()
)

base_proteins["pocket_bin"] = pd.qcut(
    base_proteins["experimental_pocket_size"],
    q=5,
    labels=[
        "P0–20",
        "P20–40",
        "P40–60",
        "P60–80",
        "P80–100",
    ],
    duplicates="drop",
)

df_bins = df.merge(
    base_proteins[
        [
            "apo_pdb_id",
            "apo_chain",
            "pocket_bin",
        ]
    ],
    on=["apo_pdb_id", "apo_chain"],
    how="left",
)

bin_order = [
    "P0–20",
    "P20–40",
    "P40–60",
    "P60–80",
    "P80–100",
]

x = np.arange(len(bin_order))

width = 0.24

fig, ax = plt.subplots(figsize=(9, 6))

offsets = {
    "esm_structure": -width,
    "full_tda": 0,
    "full_model": width,
}

for arm in ARMS:

    sub = df_bins[
        df_bins["arm"] == arm
    ]

    means = []

    for b in bin_order:
        vals = sub[
            sub["pocket_bin"].astype(str) == b
        ]["PR_AUC"]

        means.append(vals.mean())

    ax.bar(
        x + offsets[arm],
        means,
        width,
        label=DISPLAY[arm]
    )

ax.set_xticks(x)

ax.set_xticklabels(
    bin_order
)

ax.set_xlabel(
    "Experimental pocket-size percentile"
)

ax.set_ylabel(
    "Mean protein-level PR-AUC"
)

ax.set_title(
    "Prediction performance across experimental pocket-size groups"
)

ax.legend(
    frameon=False
)

ax.grid(
    axis="y",
    alpha=0.2
)

fig.tight_layout()

path3 = os.path.join(
    OUTDIR,
    "figure3_pr_auc_by_pocket_size_bin.png"
)

fig.savefig(
    path3,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:", path3)


# ============================================================================
# SAVE FIGURE DATA
# ============================================================================

figure_data = df[
    [
        "apo_pdb_id",
        "apo_chain",
        "arm",
        "experimental_pocket_size",
        "experimental_pocket_fraction",
        "protein_size",
        "PR_AUC",
        "ROC_AUC",
        "center_error",
        "dcc4",
        "dcc6",
        "dcc8",
        "dcc10",
    ]
].copy()

data_path = os.path.join(
    OUTDIR,
    "phase190_figure_data.csv"
)

figure_data.to_csv(
    data_path,
    index=False
)

print("Saved:", data_path)


# ============================================================================
# FINAL
# ============================================================================

print("\n" + "=" * 90)
print("PHASE 190 COMPLETE")
print("=" * 90)

print("\nFigures:")
print(path1)
print(path2)
print(path3)

print("\nData:")
print(data_path)
