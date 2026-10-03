import pandas as pd
import numpy as np
import torch
from pathlib import Path

BASE = Path("phase167_validation")
GRAPH_PATH = Path("phase159_input/phase159_cryptobench_anmmsf_graphs.pt")

PRED_PATH = BASE / "phase167_residue_predictions.csv"
CENTER_PATH = BASE / "phase168A_corrected_centers.csv"

print("=" * 110)
print("PHASE 179A - RESIDUE CENTER-PROXIMITY ENRICHMENT")
print("=" * 110)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

pred = pd.read_csv(PRED_PATH)
centers = pd.read_csv(CENTER_PATH)

print("Predictions:", pred.shape)
print("Centers:", centers.shape)

# Normalize PDB IDs to lowercase for matching
pred["pdb_key"] = pred["apo_pdb_id"].astype(str).str.lower()
pred["chain_key"] = pred["apo_chain"].astype(str)

centers["pdb_key"] = centers["apo_pdb_id"].astype(str).str.lower()
centers["chain_key"] = centers["apo_chain"].astype(str)

# ------------------------------------------------------------
# 2. Load graph
# ------------------------------------------------------------

print("\nLoading graph file...")

obj = torch.load(
    GRAPH_PATH,
    map_location="cpu",
    weights_only=False,
)

dataset = obj["dataset"]

print("Graphs:", len(dataset))

# ------------------------------------------------------------
# 3. Build coordinate lookup
# ------------------------------------------------------------

coord_lookup = {}

for g in dataset:

    pdb_id = str(getattr(g, "pdb_id")).lower()
    chain_id = str(getattr(g, "chain_id"))

    pos = g.pos.detach().cpu().numpy()

    for node_idx in range(len(pos)):

        coord_lookup[
            (pdb_id, chain_id, node_idx)
        ] = pos[node_idx]

print(
    "Coordinate lookup entries:",
    len(coord_lookup)
)

# ------------------------------------------------------------
# 4. Map graph coordinates to prediction rows
# ------------------------------------------------------------

xyz = []

missing = 0

for row in pred.itertuples(index=False):

    key = (
        row.pdb_key,
        row.chain_key,
        int(row.node_index),
    )

    coord = coord_lookup.get(key)

    if coord is None:

        xyz.append(
            (np.nan, np.nan, np.nan)
        )

        missing += 1

    else:

        xyz.append(
            (
                float(coord[0]),
                float(coord[1]),
                float(coord[2]),
            )
        )

xyz = np.asarray(xyz)

pred["residue_x"] = xyz[:, 0]
pred["residue_y"] = xyz[:, 1]
pred["residue_z"] = xyz[:, 2]

print(
    "Prediction rows missing graph coordinates:",
    missing,
)

print(
    "Prediction rows successfully mapped:",
    len(pred) - missing,
)

# ------------------------------------------------------------
# 5. Merge experimental centers
# ------------------------------------------------------------

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

print(
    "After center merge:",
    pred.shape
)

# ------------------------------------------------------------
# 6. Remove invalid rows
# ------------------------------------------------------------

pred = pred.dropna(
    subset=[
        "residue_x",
        "residue_y",
        "residue_z",
        "true_center_x",
        "true_center_y",
        "true_center_z",
    ]
).copy()

print(
    "Rows with valid coordinates:",
    pred.shape
)

if len(pred) == 0:

    raise RuntimeError(
        "No valid rows remain. "
        "Identifier matching still failed."
    )

# ------------------------------------------------------------
# 7. TRUE residue-to-pocket-center distance
# ------------------------------------------------------------

pred["dist_to_true_center"] = np.sqrt(
    (pred["residue_x"] - pred["true_center_x"]) ** 2
    + (pred["residue_y"] - pred["true_center_y"]) ** 2
    + (pred["residue_z"] - pred["true_center_z"]) ** 2
)

print(
    "\nDistance range:",
    f"{pred['dist_to_true_center'].min():.3f}",
    "to",
    f"{pred['dist_to_true_center'].max():.3f}",
    "Å",
)

# ------------------------------------------------------------
# 8. Center-proximity enrichment
# ------------------------------------------------------------

fractions = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
]

cutoffs = [
    4,
    6,
    8,
    10,
    12,
]

records = []

for (arm, pdb, chain, seed), g in pred.groupby(
    [
        "arm",
        "apo_pdb_id",
        "apo_chain",
        "seed",
    ],
    sort=False,
):

    g = g.copy()

    if len(g) < 5:
        continue

    probabilities = (
        g["probability"].to_numpy()
    )

    distances = (
        g["dist_to_true_center"].to_numpy()
    )

    order = np.argsort(
        -probabilities
    )

    for fraction in fractions:

        k = max(
            1,
            int(
                np.ceil(
                    len(g) * fraction
                )
            ),
        )

        top_idx = order[:k]

        for cutoff in cutoffs:

            background_fraction = np.mean(
                distances <= cutoff
            )

            selected_fraction = np.mean(
                distances[top_idx] <= cutoff
            )

            if background_fraction > 0:

                enrichment = (
                    selected_fraction
                    / background_fraction
                )

            else:

                enrichment = np.nan

            records.append(
                {
                    "arm": arm,
                    "apo_pdb_id": pdb,
                    "apo_chain": chain,
                    "seed": seed,
                    "fraction": fraction,
                    "cutoff": cutoff,
                    "n_residues": len(g),
                    "n_selected": k,
                    "background_fraction":
                        background_fraction,
                    "top_fraction_fraction":
                        selected_fraction,
                    "enrichment":
                        enrichment,
                }
            )

out = pd.DataFrame(records)

if out.empty:

    raise RuntimeError(
        "No enrichment records were generated."
    )

# ------------------------------------------------------------
# 9. Save raw results
# ------------------------------------------------------------

out_path = (
    BASE /
    "phase179A_pocket_enrichment.csv"
)

out.to_csv(
    out_path,
    index=False,
)

# ------------------------------------------------------------
# 10. Summary
# ------------------------------------------------------------

summary = (
    out
    .groupby(
        [
            "arm",
            "fraction",
            "cutoff",
        ],
        as_index=False,
    )
    .agg(
        n=("enrichment", "size"),

        mean_background_fraction=(
            "background_fraction",
            "mean",
        ),

        mean_top_fraction_fraction=(
            "top_fraction_fraction",
            "mean",
        ),

        mean_enrichment=(
            "enrichment",
            "mean",
        ),

        median_enrichment=(
            "enrichment",
            "median",
        ),
    )
)

summary_path = (
    BASE /
    "phase179A_pocket_enrichment_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

# ------------------------------------------------------------
# 11. Print results
# ------------------------------------------------------------

print("\n" + "=" * 110)
print("PHASE 179A RESULTS")
print("=" * 110)

for arm in summary["arm"].unique():

    print("\n" + "-" * 100)
    print(arm)
    print("-" * 100)

    s = summary[
        (summary["arm"] == arm)
        & (
            summary["fraction"].isin(
                [0.01, 0.05, 0.10, 0.20]
            )
        )
    ]

    print(
        s.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

print("\nSaved:")
print(out_path)
print(summary_path)

print("\nInterpretation:")
print(
    "Enrichment > 1 means that the model's "
    "highest-probability residues are closer "
    "to the experimental pocket center than "
    "the average residue background."
)
