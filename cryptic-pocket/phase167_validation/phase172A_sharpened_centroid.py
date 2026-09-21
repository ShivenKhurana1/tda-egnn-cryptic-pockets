import pandas as pd
import numpy as np
import torch

PRED_FILE = "phase167_validation/phase167_residue_predictions.csv"
CENTER_FILE = "phase167_validation/phase168A_corrected_centers.csv"
GRAPH_FILE = "phase159_input/phase159_cryptobench_anmmsf_graphs.pt"
OUTPUT = "phase167_validation/phase172A_sharpened_centroid.csv"

print("=" * 100)
print("PHASE 172A — SHARPENED PROBABILITY CENTROID")
print("=" * 100)

pred = pd.read_csv(PRED_FILE)

pred = pred[
    pred["arm"].isin([
        "esm_structure",
        "full_tda",
        "full_model"
    ])
].copy()

pred["pdb_norm"] = (
    pred["apo_pdb_id"].astype(str).str.strip().str.lower()
)

pred["chain_norm"] = (
    pred["apo_chain"].astype(str).str.strip().str.upper()
)

centers = pd.read_csv(CENTER_FILE)

centers["pdb_norm"] = (
    centers["apo_pdb_id"].astype(str).str.strip().str.lower()
)

centers["chain_norm"] = (
    centers["apo_chain"].astype(str).str.strip().str.upper()
)

pred = pred.merge(
    centers[
        [
            "pdb_norm",
            "chain_norm",
            "true_center_x",
            "true_center_y",
            "true_center_z"
        ]
    ],
    on=["pdb_norm", "chain_norm"],
    how="inner"
)

obj = torch.load(
    GRAPH_FILE,
    map_location="cpu",
    weights_only=False
)

graphs = obj["dataset"]

lookup = {
    (
        str(g.pdb_id).strip().lower(),
        str(g.chain_id).strip().upper()
    ): g
    for g in graphs
}

gammas = [1, 2, 3, 4, 5, 8, 10]

rows = []

groups = pred.groupby(
    ["pdb_norm", "chain_norm", "arm", "seed"],
    sort=False
)

print(f"Evaluating {len(groups):,} protein/arm/seed combinations...")

for (pdb, chain, arm, seed), g in groups:

    graph = lookup.get((pdb, chain))

    if graph is None:
        continue

    pos = graph.pos.detach().cpu().numpy()

    idx = g["node_index"].astype(int).to_numpy()
    p = g["probability"].astype(float).to_numpy()

    true_center = np.array([
        g["true_center_x"].iloc[0],
        g["true_center_y"].iloc[0],
        g["true_center_z"].iloc[0]
    ])

    valid = (
        (idx >= 0) &
        (idx < len(pos)) &
        np.isfinite(p)
    )

    idx = idx[valid]
    p = p[valid]

    if len(idx) == 0:
        continue

    coords = pos[idx]

    # Clamp to avoid numerical problems.
    p = np.clip(p, 1e-8, 1.0)

    for gamma in gammas:

        weights = p ** gamma
        weights = weights / weights.sum()

        center = (weights[:, None] * coords).sum(axis=0)

        error = float(np.linalg.norm(center - true_center))

        rows.append({
            "apo_pdb_id": pdb,
            "apo_chain": chain,
            "arm": arm,
            "seed": seed,
            "gamma": gamma,
            "center_error": error,
            "dcc4": float(error <= 4),
            "dcc6": float(error <= 6),
            "dcc8": float(error <= 8),
            "dcc10": float(error <= 10),
        })

results = pd.DataFrame(rows)

results.to_csv(OUTPUT, index=False)

summary = (
    results
    .groupby(["arm", "gamma"], as_index=False)
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
print("PHASE 172A RESULTS")
print("=" * 100)

print(summary.to_string(index=False))

print("\n" + "=" * 100)
print("BEST GAMMA BY MEDIAN CENTER ERROR")
print("=" * 100)

for arm in summary["arm"].unique():

    a = summary[summary["arm"] == arm]

    best = a.loc[a["median_center_error"].idxmin()]

    print(
        f"{arm:15s} "
        f"gamma={best['gamma']} "
        f"median_error={best['median_center_error']:.4f} Å "
        f"DCC10={best['dcc10']:.4f}"
    )

print("\n" + "=" * 100)
print("BEST GAMMA BY DCC10")
print("=" * 100)

for arm in summary["arm"].unique():

    a = summary[summary["arm"] == arm]

    best = a.loc[a["dcc10"].idxmax()]

    print(
        f"{arm:15s} "
        f"gamma={best['gamma']} "
        f"DCC10={best['dcc10']:.4f} "
        f"median_error={best['median_center_error']:.4f} Å"
    )

print("\nSaved:")
print(OUTPUT)
