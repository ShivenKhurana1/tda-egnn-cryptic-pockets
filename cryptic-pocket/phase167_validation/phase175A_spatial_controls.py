import numpy as np
import pandas as pd
import torch

PRED_PATH = "phase167_validation/phase167_residue_predictions.csv"
CENTER_PATH = "phase167_validation/phase168A_corrected_centers.csv"
GRAPH_PATH = "phase159_input/phase159_cryptobench_anmmsf_graphs.pt"

OUT_PATH = "phase167_validation/phase175A_spatial_controls.csv"
SUMMARY_PATH = "phase167_validation/phase175A_spatial_controls_summary.csv"

ARMS = ["esm_structure", "full_tda", "full_model"]

rng = np.random.default_rng(172)

def norm_pdb(x):
    return str(x).strip().lower()

def norm_chain(x):
    return str(x).strip().upper()

pred = pd.read_csv(PRED_PATH)
centers = pd.read_csv(CENTER_PATH)

pred = pred[pred["arm"].isin(ARMS)].copy()

pred["pdb_key"] = pred["apo_pdb_id"].map(norm_pdb)
pred["chain_key"] = pred["apo_chain"].map(norm_chain)

centers["pdb_key"] = centers["apo_pdb_id"].map(norm_pdb)
centers["chain_key"] = centers["apo_chain"].map(norm_chain)

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

obj = torch.load(
    GRAPH_PATH,
    map_location="cpu",
    weights_only=False
)

dataset = obj["dataset"]

graph_pos = {}

for g in dataset:
    key = (
        norm_pdb(g.pdb_id),
        norm_chain(g.chain_id)
    )
    graph_pos[key] = g.pos.detach().cpu().numpy()

records = []

for (pdb, chain, arm, seed), group in pred.groupby(
    ["pdb_key", "chain_key", "arm", "seed"],
    sort=False
):

    key = (pdb, chain)

    if key not in graph_pos:
        continue

    pos = graph_pos[key]

    node_idx = group["node_index"].to_numpy(dtype=int)

    valid = (
        (node_idx >= 0) &
        (node_idx < len(pos))
    )

    group = group.loc[valid].copy()

    if len(group) == 0:
        continue

    node_idx = group["node_index"].to_numpy(dtype=int)

    probs = np.clip(
        group["probability"].to_numpy(dtype=float),
        1e-12,
        None
    )

    coords = pos[node_idx]

    target = np.array([
        group["true_center_x"].iloc[0],
        group["true_center_y"].iloc[0],
        group["true_center_z"].iloc[0],
    ])

    n = len(coords)

    # ---------------------------------------------------------
    # Controls
    # ---------------------------------------------------------

    protein_centroid = pos.mean(axis=0)

    uniform_centroid = coords.mean(axis=0)

    # Random 10% centroid
    k = max(1, int(np.ceil(0.10 * n)))

    random_centers = []

    # Use 10 independent random draws and average the resulting
    # errors to reduce Monte Carlo noise.
    for _ in range(10):
        idx = rng.choice(
            n,
            size=k,
            replace=False
        )

        random_centers.append(
            coords[idx].mean(axis=0)
        )

    random10_center = np.mean(
        random_centers,
        axis=0
    )

    # ---------------------------------------------------------
    # Prediction methods
    # ---------------------------------------------------------

    # gamma = 1
    weights1 = probs / probs.sum()

    probability_centroid = np.sum(
        coords * weights1[:, None],
        axis=0
    )

    # gamma = 10
    weights10 = probs ** 10
    weights10 /= weights10.sum()

    sharpened_center = np.sum(
        coords * weights10[:, None],
        axis=0
    )

    # top 10%
    order = np.argsort(-probs)
    top_idx = order[:k]

    top10_center = coords[top_idx].mean(axis=0)

    methods = {
        "protein_centroid": protein_centroid,
        "uniform_node_centroid": uniform_centroid,
        "random10_centroid": random10_center,
        "probability_centroid_gamma1": probability_centroid,
        "top10_centroid": top10_center,
        "sharpened_gamma10": sharpened_center,
    }

    for method, center in methods.items():

        e = float(
            np.linalg.norm(center - target)
        )

        records.append({
            "apo_pdb_id": group["apo_pdb_id"].iloc[0],
            "apo_chain": group["apo_chain"].iloc[0],
            "arm": arm,
            "seed": seed,
            "method": method,
            "center_error": e,
            "dcc4": float(e <= 4),
            "dcc6": float(e <= 6),
            "dcc8": float(e <= 8),
            "dcc10": float(e <= 10),
        })

df = pd.DataFrame(records)

df.to_csv(OUT_PATH, index=False)

summary = (
    df.groupby(["arm", "method"])
      .agg(
          n=("center_error", "size"),
          mean_center_error=("center_error", "mean"),
          median_center_error=("center_error", "median"),
          dcc4=("dcc4", "mean"),
          dcc6=("dcc6", "mean"),
          dcc8=("dcc8", "mean"),
          dcc10=("dcc10", "mean"),
      )
      .reset_index()
)

summary.to_csv(SUMMARY_PATH, index=False)

print("=" * 100)
print("PHASE 175A - SPATIAL CONTROL ANALYSIS")
print("=" * 100)

print(f"Records: {len(df):,}")

print("\n" + "=" * 100)
print("RESULTS")
print("=" * 100)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nSaved:")
print(OUT_PATH)
print(SUMMARY_PATH)
