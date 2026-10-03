import os
import numpy as np
import pandas as pd
import torch

PRED_PATH = "phase167_validation/phase167_residue_predictions.csv"
CENTER_PATH = "phase167_validation/phase168A_corrected_centers.csv"
GRAPH_PATH = "phase159_input/phase159_cryptobench_anmmsf_graphs.pt"
OUT_PATH = "phase167_validation/phase173A_extended_sharpening.csv"

ARMS = ["esm_structure", "full_tda", "full_model"]

# Extended sharpening range
GAMMAS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]


def normalize_pdb(x):
    return str(x).strip().lower()


def normalize_chain(x):
    return str(x).strip().upper()


def dcc(error, threshold):
    return float(np.mean(error <= threshold))


print("=" * 100)
print("PHASE 173A - EXTENDED PROBABILITY SHARPENING")
print("=" * 100)

pred = pd.read_csv(PRED_PATH)
centers = pd.read_csv(CENTER_PATH)

pred = pred[pred["arm"].isin(ARMS)].copy()

pred["pdb_key"] = pred["apo_pdb_id"].map(normalize_pdb)
pred["chain_key"] = pred["apo_chain"].map(normalize_chain)

centers["pdb_key"] = centers["apo_pdb_id"].map(normalize_pdb)
centers["chain_key"] = centers["apo_chain"].map(normalize_chain)

merge_cols = [
    "pdb_key",
    "chain_key",
    "experimental_true_center_distance",
]

pred = pred.merge(
    centers[merge_cols],
    on=["pdb_key", "chain_key"],
    how="inner",
)

if len(pred) == 0:
    raise RuntimeError("Center merge produced zero rows.")

print(f"Prediction rows after merge: {len(pred):,}")

# Load graph dataset
obj = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
dataset = obj["dataset"]

# Coordinates indexed by pdb/chain
graph_pos = {}

for g in dataset:
    pdb = normalize_pdb(g.pdb_id)
    chain = normalize_chain(g.chain_id)

    pos = g.pos.detach().cpu().numpy()

    graph_pos[(pdb, chain)] = pos


results = []

group_cols = ["pdb_key", "chain_key", "arm", "seed"]

groups = list(pred.groupby(group_cols, sort=False))

print(f"Evaluating {len(groups):,} protein/arm/seed combinations...")
print(f"Gammas: {GAMMAS}")

for idx, ((pdb, chain, arm, seed), group) in enumerate(groups):

    key = (pdb, chain)

    if key not in graph_pos:
        continue

    pos = graph_pos[key]

    # Make sure node indices are valid
    node_idx = group["node_index"].to_numpy(dtype=int)

    valid = (
        (node_idx >= 0) &
        (node_idx < len(pos))
    )

    group = group.loc[valid].copy()

    if len(group) == 0:
        continue

    node_idx = group["node_index"].to_numpy(dtype=int)
    probabilities = group["probability"].to_numpy(dtype=float)

    coordinates = pos[node_idx]

    # Experimental center is constant for this protein
    true_center = np.array([
        group["experimental_true_center_distance"].iloc[0]
    ])

    # Recover actual graph-space center from the center file.
    center_row = centers[
        (centers["pdb_key"] == pdb) &
        (centers["chain_key"] == chain)
    ]

    if len(center_row) == 0:
        continue

    center_row = center_row.iloc[0]

    target = np.array([
        center_row["true_center_x"],
        center_row["true_center_y"],
        center_row["true_center_z"],
    ], dtype=float)

    # Guard against pathological probabilities
    probabilities = np.clip(probabilities, 1e-12, None)

    for gamma in GAMMAS:

        weights = probabilities ** gamma

        weight_sum = weights.sum()

        if not np.isfinite(weight_sum) or weight_sum <= 0:
            continue

        weights = weights / weight_sum

        predicted_center = np.sum(
            coordinates * weights[:, None],
            axis=0
        )

        error = float(
            np.linalg.norm(predicted_center - target)
        )

        results.append({
            "apo_pdb_id": group["apo_pdb_id"].iloc[0],
            "apo_chain": group["apo_chain"].iloc[0],
            "arm": arm,
            "seed": seed,
            "gamma": gamma,
            "center_error": error,
        })

    if (idx + 1) % 500 == 0:
        print(f"  processed {idx + 1:,}/{len(groups):,}")


df = pd.DataFrame(results)

if len(df) == 0:
    raise RuntimeError("No results were produced.")

summary = (
    df.groupby(["arm", "gamma"])
      .agg(
          n=("center_error", "size"),
          mean_center_error=("center_error", "mean"),
          median_center_error=("center_error", "median"),
          dcc4=("center_error", lambda x: np.mean(x <= 4)),
          dcc6=("center_error", lambda x: np.mean(x <= 6)),
          dcc8=("center_error", lambda x: np.mean(x <= 8)),
          dcc10=("center_error", lambda x: np.mean(x <= 10)),
      )
      .reset_index()
)

summary.to_csv(OUT_PATH, index=False)

print("\n" + "=" * 100)
print("PHASE 173A RESULTS")
print("=" * 100)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "=" * 100)
print("BEST GAMMA BY MEDIAN CENTER ERROR")
print("=" * 100)

for arm in ARMS:
    sub = summary[summary["arm"] == arm]
    best = sub.loc[sub["median_center_error"].idxmin()]

    print(
        f"{arm:15s} "
        f"gamma={int(best['gamma'])} "
        f"median_error={best['median_center_error']:.4f} Å "
        f"DCC10={best['dcc10']:.4f}"
    )

print("\n" + "=" * 100)
print("BEST GAMMA BY DCC10")
print("=" * 100)

for arm in ARMS:
    sub = summary[summary["arm"] == arm]
    best = sub.loc[sub["dcc10"].idxmax()]

    print(
        f"{arm:15s} "
        f"gamma={int(best['gamma'])} "
        f"DCC10={best['dcc10']:.4f} "
        f"median_error={best['median_center_error']:.4f} Å"
    )

print(f"\nSaved:")
print(OUT_PATH)
