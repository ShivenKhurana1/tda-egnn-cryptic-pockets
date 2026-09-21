import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("phase167_validation")

PRED = BASE / "phase167_residue_predictions.csv"
TRANS = BASE / "phase168A_coordinate_translations.csv"
SITE = Path("phase167_experimental_sites.csv")

pred = pd.read_csv(PRED, low_memory=False)
trans = pd.read_csv(TRANS, low_memory=False)
site = pd.read_csv(SITE, low_memory=False)

# ------------------------------------------------------------
# One center prediction per protein / arm / seed
# ------------------------------------------------------------

center_cols = [
    "apo_pdb_id",
    "apo_chain",
    "arm",
    "seed",
    "predicted_center_x",
    "predicted_center_y",
    "predicted_center_z"
]

centers = (
    pred[center_cols]
    .drop_duplicates()
)

print("Predicted centers:", len(centers))
print(
    centers.groupby("arm")
    .size()
    .to_string()
)

# ------------------------------------------------------------
# Experimental site centers
# ------------------------------------------------------------

site_cols = [
    "apo_pdb_id",
    "apo_chain",
    "site_center_x",
    "site_center_y",
    "site_center_z"
]

sites = (
    site[site_cols]
    .drop_duplicates(
        ["apo_pdb_id", "apo_chain"]
    )
)

# ------------------------------------------------------------
# Join graph-frame transformation
# ------------------------------------------------------------

centers = centers.merge(
    trans[
        [
            "apo_pdb_id",
            "apo_chain",
            "tx",
            "ty",
            "tz",
            "translation_residual_mean",
            "translation_residual_max"
        ]
    ],
    on=["apo_pdb_id", "apo_chain"],
    how="inner",
    validate="many_to_one"
)

centers = centers.merge(
    sites,
    on=["apo_pdb_id", "apo_chain"],
    how="inner",
    validate="many_to_one"
)

# ------------------------------------------------------------
# Convert experimental center PDB -> graph coordinates
# ------------------------------------------------------------

centers["true_x"] = (
    centers.site_center_x - centers.tx
)

centers["true_y"] = (
    centers.site_center_y - centers.ty
)

centers["true_z"] = (
    centers.site_center_z - centers.tz
)

# ------------------------------------------------------------
# Calculate center error
# ------------------------------------------------------------

centers["center_error"] = np.sqrt(
    (
        centers.predicted_center_x -
        centers.true_x
    ) ** 2
    +
    (
        centers.predicted_center_y -
        centers.true_y
    ) ** 2
    +
    (
        centers.predicted_center_z -
        centers.true_z
    ) ** 2
)

centers["dcc4"] = (
    centers.center_error < 4
)

centers["dcc6"] = (
    centers.center_error < 6
)

centers["dcc8"] = (
    centers.center_error < 8
)

centers["dcc10"] = (
    centers.center_error < 10
)

# ------------------------------------------------------------
# Print raw seed-level results
# ------------------------------------------------------------

print("\n=== SEED-LEVEL LOCALIZATION ===")

for arm in [
    "esm_structure",
    "full_tda",
    "full_model"
]:

    x = centers[
        centers.arm == arm
    ]

    print(f"\n{arm}")
    print(
        "Mean:",
        x.center_error.mean()
    )
    print(
        "Median:",
        x.center_error.median()
    )
    print(
        "DCC4:",
        x.dcc4.mean()
    )
    print(
        "DCC6:",
        x.dcc6.mean()
    )
    print(
        "DCC8:",
        x.dcc8.mean()
    )
    print(
        "DCC10:",
        x.dcc10.mean()
    )

# ------------------------------------------------------------
# Average seeds per protein
# ------------------------------------------------------------

protein = (
    centers
    .groupby(
        [
            "apo_pdb_id",
            "apo_chain",
            "arm"
        ],
        as_index=False
    )
    .agg(
        mean_center_error=(
            "center_error",
            "mean"
        ),
        median_center_error=(
            "center_error",
            "median"
        ),
        dcc4=(
            "dcc4",
            "mean"
        ),
        dcc6=(
            "dcc6",
            "mean"
        ),
        dcc8=(
            "dcc8",
            "mean"
        ),
        dcc10=(
            "dcc10",
            "mean"
        )
    )
)

print("\n=== PROTEIN-LEVEL LOCALIZATION ===")

for arm in [
    "esm_structure",
    "full_tda",
    "full_model"
]:

    x = protein[
        protein.arm == arm
    ]

    print(f"\n{arm}")
    print(
        "Mean error:",
        x.mean_center_error.mean()
    )
    print(
        "Median error:",
        x.median_center_error.mean()
    )
    print(
        "DCC4:",
        x.dcc4.mean()
    )
    print(
        "DCC6:",
        x.dcc6.mean()
    )
    print(
        "DCC8:",
        x.dcc8.mean()
    )
    print(
        "DCC10:",
        x.dcc10.mean()
    )

# ------------------------------------------------------------
# Paired protein comparisons
# ------------------------------------------------------------

wide = protein.pivot_table(
    index=[
        "apo_pdb_id",
        "apo_chain"
    ],
    columns="arm",
    values=[
        "mean_center_error",
        "median_center_error",
        "dcc4",
        "dcc6",
        "dcc8",
        "dcc10"
    ]
)

print("\n=== TDA - ESM+STRUCTURE ===")

for metric in [
    "mean_center_error",
    "median_center_error",
    "dcc4",
    "dcc6",
    "dcc8",
    "dcc10"
]:

    d = (
        wide[(metric, "full_tda")] -
        wide[(metric, "esm_structure")]
    ).dropna()

    print(
        f"{metric}: "
        f"mean Δ={d.mean():.6f}, "
        f"median Δ={d.median():.6f}, "
        f"improved={(d < 0).sum()}/{len(d)}"
    )

print("\n=== ANM - TDA ===")

for metric in [
    "mean_center_error",
    "median_center_error",
    "dcc4",
    "dcc6",
    "dcc8",
    "dcc10"
]:

    d = (
        wide[(metric, "full_model")] -
        wide[(metric, "full_tda")]
    ).dropna()

    print(
        f"{metric}: "
        f"mean Δ={d.mean():.6f}, "
        f"median Δ={d.median():.6f}, "
        f"improved={(d < 0).sum()}/{len(d)}"
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

centers.to_csv(
    BASE / "phase168A_final_seed_centers.csv",
    index=False
)

protein.to_csv(
    BASE / "phase168A_final_protein_metrics.csv",
    index=False
)

print("\nSaved:")
print("phase168A_final_seed_centers.csv")
print("phase168A_final_protein_metrics.csv")
