import pandas as pd
import numpy as np
import torch

PRED_FILE = "phase167_validation/phase167_residue_predictions.csv"
CENTER_FILE = "phase167_validation/phase168A_corrected_centers.csv"
GRAPH_FILE = "phase159_input/phase159_cryptobench_anmmsf_graphs.pt"
OUTPUT = "phase167_validation/phase171A_threshold_sweep.csv"

print("=" * 100)
print("PHASE 171A — CLASSIFICATION THRESHOLD LOCALIZATION SWEEP")
print("=" * 100)

# ---------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------

pred = pd.read_csv(PRED_FILE)

pred = pred[
    pred["arm"].isin([
        "esm_structure",
        "full_tda",
        "full_model"
    ])
].copy()

# Normalize identifiers
pred["pdb_norm"] = (
    pred["apo_pdb_id"]
    .astype(str)
    .str.strip()
    .str.lower()
)

pred["chain_norm"] = (
    pred["apo_chain"]
    .astype(str)
    .str.strip()
    .str.upper()
)

print(f"Prediction rows: {len(pred):,}")
print(f"Prediction proteins: {pred[['pdb_norm','chain_norm']].drop_duplicates().shape[0]}")

# ---------------------------------------------------------------------
# Load experimental centers
# ---------------------------------------------------------------------

centers = pd.read_csv(CENTER_FILE)

centers["pdb_norm"] = (
    centers["apo_pdb_id"]
    .astype(str)
    .str.strip()
    .str.lower()
)

centers["chain_norm"] = (
    centers["apo_chain"]
    .astype(str)
    .str.strip()
    .str.upper()
)

center_cols = [
    "pdb_norm",
    "chain_norm",
    "true_center_x",
    "true_center_y",
    "true_center_z",
]

pred = pred.merge(
    centers[center_cols],
    on=["pdb_norm", "chain_norm"],
    how="inner"
)

print(f"Rows after center merge: {len(pred):,}")
print(
    "Proteins after center merge:",
    pred[["pdb_norm", "chain_norm"]].drop_duplicates().shape[0]
)

# ---------------------------------------------------------------------
# Load graphs
# ---------------------------------------------------------------------

print("\nLoading graphs...")

obj = torch.load(
    GRAPH_FILE,
    map_location="cpu",
    weights_only=False
)

graphs = obj["dataset"]

lookup = {
    (str(g.pdb_id).strip().lower(), str(g.chain_id).strip().upper()): g
    for g in graphs
}

print(f"Loaded {len(graphs)} graphs.")
print(f"Graph lookup entries: {len(lookup)}")

# ---------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------

fractions = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]

rows = []

groups = pred.groupby(
    ["pdb_norm", "chain_norm", "arm", "seed"],
    sort=False
)

print(f"\nEvaluating {len(groups):,} protein/arm/seed combinations...")

for group_num, ((pdb, chain, arm, seed), g) in enumerate(groups, start=1):

    graph = lookup.get((pdb, chain))

    if graph is None:
        continue

    pos = graph.pos.detach().cpu().numpy()

    node_idx = g["node_index"].astype(int).to_numpy()
    probs = g["probability"].astype(float).to_numpy()

    true_center = np.array([
        g["true_center_x"].iloc[0],
        g["true_center_y"].iloc[0],
        g["true_center_z"].iloc[0],
    ], dtype=float)

    valid = (
        (node_idx >= 0) &
        (node_idx < len(pos)) &
        np.isfinite(probs)
    )

    node_idx = node_idx[valid]
    probs = probs[valid]

    if len(node_idx) == 0:
        continue

    coords = pos[node_idx]

    # Highest probability first
    order = np.argsort(-probs)

    for frac in fractions:

        k = max(1, int(np.ceil(frac * len(order))))

        selected = order[:k]

        selected_coords = coords[selected]

        # Simple unweighted centroid of the selected
        # highest-probability residues.
        center = selected_coords.mean(axis=0)

        error = float(np.linalg.norm(center - true_center))

        rows.append({
            "apo_pdb_id": pdb,
            "apo_chain": chain,
            "arm": arm,
            "seed": seed,
            "fraction": frac,
            "n_selected": k,
            "center_error": error,
            "dcc4": float(error <= 4.0),
            "dcc6": float(error <= 6.0),
            "dcc8": float(error <= 8.0),
            "dcc10": float(error <= 10.0),
        })

results = pd.DataFrame(rows)

if results.empty:
    raise RuntimeError(
        "No threshold results were generated. "
        "Check graph/prediction key matching."
    )

# ---------------------------------------------------------------------
# Save detailed results
# ---------------------------------------------------------------------

results.to_csv(OUTPUT, index=False)

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

summary = (
    results
    .groupby(["arm", "fraction"], as_index=False)
    .agg(
        n=("center_error", "size"),
        mean_center_error=("center_error", "mean"),
        median_center_error=("center_error", "median"),
        dcc4=("dcc4", "mean"),
        dcc6=("dcc6", "mean"),
        dcc8=("dcc8", "mean"),
        dcc10=("dcc10", "mean"),
    )
)

print("\n" + "=" * 100)
print("PHASE 171A RESULTS")
print("=" * 100)

print(summary.to_string(index=False))

print("\nSaved:")
print(f"  {OUTPUT}")

# ---------------------------------------------------------------------
# Best fraction by median center error
# ---------------------------------------------------------------------

print("\n" + "=" * 100)
print("BEST FRACTION BY MEDIAN CENTER ERROR")
print("=" * 100)

for arm in summary["arm"].unique():

    a = summary[summary["arm"] == arm]

    best = a.loc[a["median_center_error"].idxmin()]

    print(
        f"{arm:15s} "
        f"fraction={best['fraction']:.2f} "
        f"median_error={best['median_center_error']:.4f} Å "
        f"DCC10={best['dcc10']:.4f}"
    )

# ---------------------------------------------------------------------
# Best fraction by DCC10
# ---------------------------------------------------------------------

print("\n" + "=" * 100)
print("BEST FRACTION BY DCC10")
print("=" * 100)

for arm in summary["arm"].unique():

    a = summary[summary["arm"] == arm]

    best = a.loc[a["dcc10"].idxmax()]

    print(
        f"{arm:15s} "
        f"fraction={best['fraction']:.2f} "
        f"DCC10={best['dcc10']:.4f} "
        f"median_error={best['median_center_error']:.4f} Å"
    )
