import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(".")
BASE = ROOT / "phase167_validation"

PRED_FILE = BASE / "phase183A_experimental_residue_validation.csv"
CENTER_FILE = BASE / "phase168A_corrected_centers.csv"
GRAPH_FILE = ROOT / "phase159_input/phase159_cryptobench_anmmsf_graphs.pt"

OUT_FILE = BASE / "phase185_experimental_residue_localization.csv"
OUT_SUMMARY = BASE / "phase185_experimental_residue_localization_summary.csv"

print("=" * 110)
print("PHASE 185 - EXPERIMENTAL-RESIDUE-DRIVEN POCKET LOCALIZATION")
print("=" * 110)

pred = pd.read_csv(PRED_FILE, low_memory=False)
centers = pd.read_csv(CENTER_FILE, low_memory=False)

print("\nPredictions:", pred.shape)
print("Centers:", centers.shape)

# ---------------------------------------------------------------------
# Normalize identifiers
# ---------------------------------------------------------------------

for df in [pred, centers]:
    df["apo_pdb_id_norm"] = (
        df["apo_pdb_id"]
        .astype(str)
        .str.lower()
    )

    df["apo_chain_norm"] = (
        df["apo_chain"]
        .astype(str)
    )

# ---------------------------------------------------------------------
# Load graph coordinates
# ---------------------------------------------------------------------

print("\nLoading graph coordinates...")

obj = torch.load(
    GRAPH_FILE,
    map_location="cpu",
)

graphs = obj["dataset"]

coord_lookup = {}

for g in graphs:

    pdb = str(g.pdb_id).lower()
    chain = str(g.chain_id)

    pos = g.pos.detach().cpu().numpy()

    for node_idx, xyz in enumerate(pos):

        coord_lookup[
            (pdb, chain, node_idx)
        ] = xyz

print(
    "Coordinate lookup:",
    len(coord_lookup)
)

# ---------------------------------------------------------------------
# Map every prediction to actual residue coordinates
# ---------------------------------------------------------------------

coords = []

missing = 0

for row in pred.itertuples(index=False):

    key = (
        row.apo_pdb_id_norm,
        row.apo_chain_norm,
        int(row.node_index),
    )

    xyz = coord_lookup.get(key)

    if xyz is None:
        coords.append(
            (np.nan, np.nan, np.nan)
        )
        missing += 1
    else:
        coords.append(
            tuple(xyz)
        )

pred[
    [
        "x",
        "y",
        "z",
    ]
] = np.asarray(coords)

print(
    "Missing coordinates:",
    missing
)

if missing:
    raise RuntimeError(
        "Some predictions lack graph coordinates."
    )

# ---------------------------------------------------------------------
# Merge experimental centers
# ---------------------------------------------------------------------

center_cols = [
    "apo_pdb_id_norm",
    "apo_chain_norm",
    "true_center_x",
    "true_center_y",
    "true_center_z",
]

center_small = centers[
    center_cols
].drop_duplicates(
    [
        "apo_pdb_id_norm",
        "apo_chain_norm",
    ]
)

pred = pred.merge(
    center_small,
    on=[
        "apo_pdb_id_norm",
        "apo_chain_norm",
    ],
    how="left",
    validate="many_to_one",
)

missing_centers = pred["true_center_x"].isna().sum()

print(
    "Missing experimental centers:",
    missing_centers
)

if missing_centers:
    raise RuntimeError(
        "Missing experimental centers."
    )

# ---------------------------------------------------------------------
# Localization methods
# ---------------------------------------------------------------------

methods = [
    "probability_centroid",
    "top10_centroid",
    "gamma10_centroid",
    "experimental_positive_centroid",
]

fractions = [0.01, 0.05, 0.10, 0.20]

rows = []

print("\nCalculating localization...")

for (
    arm,
    seed,
    pdb,
    chain,
), g in pred.groupby(
    [
        "arm",
        "seed",
        "apo_pdb_id_norm",
        "apo_chain_norm",
    ],
    sort=False,
):

    g = g.sort_values(
        "probability",
        ascending=False,
    ).copy()

    xyz = g[
        ["x", "y", "z"]
    ].to_numpy()

    p = g[
        "probability"
    ].to_numpy(dtype=float)

    experimental = g[
        "experimental_site"
    ].to_numpy(dtype=int)

    true_center = np.array([
        g["true_center_x"].iloc[0],
        g["true_center_y"].iloc[0],
        g["true_center_z"].iloc[0],
    ])

    # ---------------------------------------------------------------
    # Probability centroid
    # ---------------------------------------------------------------

    p_sum = p.sum()

    if p_sum > 0:
        prob_center = (
            xyz * p[:, None]
        ).sum(axis=0) / p_sum
    else:
        prob_center = xyz.mean(axis=0)

    # ---------------------------------------------------------------
    # Gamma 10
    # ---------------------------------------------------------------

    w = np.power(
        np.clip(p, 0, None),
        10,
    )

    w_sum = w.sum()

    if w_sum > 0:
        gamma_center = (
            xyz * w[:, None]
        ).sum(axis=0) / w_sum
    else:
        gamma_center = xyz.mean(axis=0)

    # ---------------------------------------------------------------
    # Experimental-positive oracle
    # ---------------------------------------------------------------

    positive_xyz = xyz[
        experimental == 1
    ]

    if len(positive_xyz) > 0:
        experimental_center = positive_xyz.mean(axis=0)
    else:
        experimental_center = np.array([
            np.nan,
            np.nan,
            np.nan,
        ])

    def distance(a, b):
        return float(
            np.linalg.norm(a - b)
        )

    rows.append({
        "arm": arm,
        "seed": seed,
        "apo_pdb_id": pdb,
        "apo_chain": chain,
        "method": "probability_centroid",
        "fraction": np.nan,
        "center_error": distance(
            prob_center,
            true_center,
        ),
    })

    rows.append({
        "arm": arm,
        "seed": seed,
        "apo_pdb_id": pdb,
        "apo_chain": chain,
        "method": "gamma10_centroid",
        "fraction": np.nan,
        "center_error": distance(
            gamma_center,
            true_center,
        ),
    })

    rows.append({
        "arm": arm,
        "seed": seed,
        "apo_pdb_id": pdb,
        "apo_chain": chain,
        "method": "experimental_positive_centroid",
        "fraction": np.nan,
        "center_error": distance(
            experimental_center,
            true_center,
        ),
    })

    # ---------------------------------------------------------------
    # Top-k centroids
    # ---------------------------------------------------------------

    for frac in fractions:

        k = max(
            1,
            int(np.ceil(frac * len(g)))
        )

        center = xyz[
            :k
        ].mean(axis=0)

        rows.append({
            "arm": arm,
            "seed": seed,
            "apo_pdb_id": pdb,
            "apo_chain": chain,
            "method": "top_centroid",
            "fraction": frac,
            "center_error": distance(
                center,
                true_center,
            ),
        })

loc = pd.DataFrame(rows)

print(
    "\nLocalization rows:",
    loc.shape
)

# ---------------------------------------------------------------------
# DCC metrics
# ---------------------------------------------------------------------

for cutoff in [4, 6, 8, 10]:

    loc[
        f"dcc{cutoff}"
    ] = (
        loc["center_error"] <= cutoff
    ).astype(int)

# ---------------------------------------------------------------------
# Seed-level summary
# ---------------------------------------------------------------------

summary = (
    loc.groupby(
        [
            "arm",
            "method",
            "fraction",
        ],
        dropna=False,
    )
    .agg(
        mean_center_error=(
            "center_error",
            "mean",
        ),
        median_center_error=(
            "center_error",
            "median",
        ),
        dcc4=(
            "dcc4",
            "mean",
        ),
        dcc6=(
            "dcc6",
            "mean",
        ),
        dcc8=(
            "dcc8",
            "mean",
        ),
        dcc10=(
            "dcc10",
            "mean",
        ),
        n=(
            "center_error",
            "count",
        ),
    )
    .reset_index()
)

print("\n" + "=" * 110)
print("LOCALIZATION SUMMARY")
print("=" * 110)

print(
    summary.to_string(index=False)
)

# ---------------------------------------------------------------------
# Protein-level seed collapse
# ---------------------------------------------------------------------

protein = (
    loc.groupby(
        [
            "arm",
            "method",
            "fraction",
            "apo_pdb_id",
            "apo_chain",
        ],
        dropna=False,
    )
    .agg(
        center_error=(
            "center_error",
            "mean",
        ),
        dcc4=(
            "dcc4",
            "mean",
        ),
        dcc6=(
            "dcc6",
            "mean",
        ),
        dcc8=(
            "dcc8",
            "mean",
        ),
        dcc10=(
            "dcc10",
            "mean",
        ),
    )
    .reset_index()
)

protein_summary = (
    protein.groupby(
        [
            "arm",
            "method",
            "fraction",
        ],
        dropna=False,
    )
    .agg(
        mean_center_error=(
            "center_error",
            "mean",
        ),
        median_center_error=(
            "center_error",
            "median",
        ),
        dcc4=("dcc4", "mean"),
        dcc6=("dcc6", "mean"),
        dcc8=("dcc8", "mean"),
        dcc10=("dcc10", "mean"),
        n_proteins=(
            "apo_pdb_id",
            "nunique",
        ),
    )
    .reset_index()
)

print("\n" + "=" * 110)
print("PROTEIN-LEVEL LOCALIZATION SUMMARY")
print("=" * 110)

print(
    protein_summary.to_string(index=False)
)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

loc.to_csv(
    OUT_FILE,
    index=False,
)

protein_summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

print("\n" + "=" * 110)
print("FILES WRITTEN")
print("=" * 110)

print(OUT_FILE)
print(OUT_SUMMARY)

print("\n" + "=" * 110)
print("PHASE 185 COMPLETE")
print("=" * 110)
