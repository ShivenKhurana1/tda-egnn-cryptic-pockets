import os
import numpy as np
import pandas as pd
import torch

BASE = "phase167_validation"

PRED_FILE = f"{BASE}/phase167_residue_predictions.csv"
CENTER_FILE = f"{BASE}/phase168A_corrected_centers.csv"
GRAPH_FILE = "phase159_input/phase159_cryptobench_anmmsf_graphs.pt"

OUT_FILE = f"{BASE}/phase169A_localization_baselines.csv"
SUMMARY_FILE = f"{BASE}/phase169A_localization_summary.csv"


def center_error(pred, true):
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(true)))


def dcc(pred, true, cutoff):
    return center_error(pred, true) <= cutoff


print("=" * 80)
print("PHASE 169A — LOCALIZATION BASELINES")
print("=" * 80)

# ------------------------------------------------------------------
# 1. Load graph dataset
# ------------------------------------------------------------------
print("\nLoading graphs...")

obj = torch.load(GRAPH_FILE, map_location="cpu")
graphs = obj["dataset"]

print(f"Loaded {len(graphs)} graphs.")

graph_lookup = {}

for g in graphs:
    key = (str(g.pdb_id).lower(), str(g.chain_id))
    graph_lookup[key] = g

print(f"Graph lookup entries: {len(graph_lookup)}")


# ------------------------------------------------------------------
# 2. Load experimental hard-holdout proteins
# ------------------------------------------------------------------
centers = pd.read_csv(CENTER_FILE)

centers["apo_pdb_id"] = centers["apo_pdb_id"].str.lower()
centers["apo_chain"] = centers["apo_chain"].astype(str)

print(f"\nExperimental proteins: {len(centers)}")


# ------------------------------------------------------------------
# 3. Load residue predictions
# ------------------------------------------------------------------
pred = pd.read_csv(PRED_FILE)

pred["apo_pdb_id"] = pred["apo_pdb_id"].str.lower()
pred["apo_chain"] = pred["apo_chain"].astype(str)

# Only the three relevant arms
arms = ["esm_structure", "full_tda", "full_model"]

pred = pred[pred["arm"].isin(arms)].copy()

print(f"Prediction rows after arm filtering: {len(pred):,}")


# ------------------------------------------------------------------
# 4. Restrict predictions to the 185 hard-holdout proteins
# ------------------------------------------------------------------
protein_keys = set(
    zip(
        centers["apo_pdb_id"],
        centers["apo_chain"],
    )
)

pred["protein_key"] = list(
    zip(pred["apo_pdb_id"], pred["apo_chain"])
)

pred = pred[pred["protein_key"].isin(protein_keys)].copy()

print(f"Hard-holdout prediction rows: {len(pred):,}")


# ------------------------------------------------------------------
# 5. Merge experimental centers
# ------------------------------------------------------------------
merge_cols = [
    "apo_pdb_id",
    "apo_chain",
    "true_center_x",
    "true_center_y",
    "true_center_z",
]

pred = pred.merge(
    centers[merge_cols],
    on=["apo_pdb_id", "apo_chain"],
    how="inner",
)

print(f"Rows after center merge: {len(pred):,}")


# ------------------------------------------------------------------
# 6. Compute localization baselines
# ------------------------------------------------------------------
results = []

group_cols = ["apo_pdb_id", "apo_chain", "arm", "seed"]

grouped = pred.groupby(group_cols, sort=False)

print(f"\nEvaluating {len(grouped):,} protein/arm/seed combinations...")


for (pdb, chain, arm, seed), df in grouped:

    key = (pdb, chain)

    if key not in graph_lookup:
        print(f"WARNING: missing graph for {pdb} {chain}")
        continue

    g = graph_lookup[key]

    pos = g.pos.detach().cpu().numpy()

    # Make sure node indices are integers
    node_idx = df["node_index"].astype(int).to_numpy()

    if node_idx.max() >= len(pos):
        print(f"WARNING: invalid node index for {pdb} {chain}")
        continue

    coords = pos[node_idx]

    probs = df["probability"].to_numpy(dtype=float)

    true = np.array([
        df["true_center_x"].iloc[0],
        df["true_center_y"].iloc[0],
        df["true_center_z"].iloc[0],
    ])

    # --------------------------------------------------------------
    # Learned localization center
    # --------------------------------------------------------------
    learned = np.array([
        df["predicted_center_x"].iloc[0],
        df["predicted_center_y"].iloc[0],
        df["predicted_center_z"].iloc[0],
    ])

    centers_to_test = {
        "learned_localization": learned
    }

    # --------------------------------------------------------------
    # Probability-weighted centroid
    # --------------------------------------------------------------
    prob_sum = probs.sum()

    if prob_sum > 0:
        prob_centroid = (coords * probs[:, None]).sum(axis=0) / prob_sum
        centers_to_test["probability_centroid"] = prob_centroid

    # --------------------------------------------------------------
    # Top-k centroids
    # --------------------------------------------------------------
    order = np.argsort(-probs)

    for k in [5, 10, 20]:
        k_actual = min(k, len(order))

        if k_actual > 0:
            top_coords = coords[order[:k_actual]]
            centers_to_test[f"top_{k}_centroid"] = top_coords.mean(axis=0)

    # --------------------------------------------------------------
    # Top 10% centroid
    # --------------------------------------------------------------
    k10 = max(1, int(np.ceil(0.10 * len(order))))
    centers_to_test["top_10pct_centroid"] = coords[order[:k10]].mean(axis=0)

    # --------------------------------------------------------------
    # Record metrics
    # --------------------------------------------------------------
    for method, predicted in centers_to_test.items():

        error = center_error(predicted, true)

        results.append({
            "apo_pdb_id": pdb,
            "apo_chain": chain,
            "arm": arm,
            "seed": int(seed),
            "method": method,

            "center_error": error,

            "dcc4": float(error <= 4.0),
            "dcc6": float(error <= 6.0),
            "dcc8": float(error <= 8.0),
            "dcc10": float(error <= 10.0),

            "pred_x": predicted[0],
            "pred_y": predicted[1],
            "pred_z": predicted[2],

            "true_x": true[0],
            "true_y": true[1],
            "true_z": true[2],

            "n_residues": len(coords),
        })


# ------------------------------------------------------------------
# 7. Save individual results
# ------------------------------------------------------------------
results_df = pd.DataFrame(results)

results_df.to_csv(OUT_FILE, index=False)

print(f"\nSaved detailed results:")
print(f"  {OUT_FILE}")

# ------------------------------------------------------------------
# 8. Aggregate summary
# ------------------------------------------------------------------
summary_rows = []

for (arm, method), df in results_df.groupby(["arm", "method"]):

    summary_rows.append({
        "arm": arm,
        "method": method,
        "n": len(df),

        "mean_center_error": df["center_error"].mean(),
        "median_center_error": df["center_error"].median(),

        "dcc4": df["dcc4"].mean(),
        "dcc6": df["dcc6"].mean(),
        "dcc8": df["dcc8"].mean(),
        "dcc10": df["dcc10"].mean(),
    })

summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    ["arm", "median_center_error", "method"]
)

summary.to_csv(SUMMARY_FILE, index=False)

print(f"Saved summary:")
print(f"  {SUMMARY_FILE}")

# ------------------------------------------------------------------
# 9. Print clean comparison
# ------------------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n" + "=" * 80)
print("PHASE 169A RESULTS")
print("=" * 80)

display_cols = [
    "arm",
    "method",
    "n",
    "mean_center_error",
    "median_center_error",
    "dcc4",
    "dcc6",
    "dcc8",
    "dcc10",
]

print(summary[display_cols].to_string(index=False))

print("\nDone.")
