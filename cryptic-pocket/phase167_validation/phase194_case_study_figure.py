import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

BASE = "phase167_validation"

GRAPH_FILE = (
    "phase159_input/"
    "phase159_cryptobench_anmmsf_graphs.pt"
)

CASE_FILE = os.path.join(
    BASE,
    "phase191_case_studies.csv"
)

RESIDUE_FILE = os.path.join(
    BASE,
    "phase183A_experimental_residue_validation.csv"
)

MAPPING_FILE = os.path.join(
    BASE,
    "phase167_node_residue_mapping.csv"
)

OUT_DIR = os.path.join(
    BASE,
    "phase194_figures"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(
    OUT_DIR,
    "phase194_case_study_4panel.png"
)

OUT_DATA = os.path.join(
    OUT_DIR,
    "phase194_case_study_centers.csv"
)

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

cases = pd.read_csv(CASE_FILE)
res = pd.read_csv(RESIDUE_FILE)
mapping = pd.read_csv(MAPPING_FILE)

for df in [cases, res, mapping]:
    df["apo_pdb_id"] = (
        df["apo_pdb_id"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["apo_chain"] = (
        df["apo_chain"]
        .astype(str)
        .str.strip()
    )

# Full model only
res = res[
    res["arm"] == "full_model"
].copy()

print("=" * 100)
print("PHASE 194 - 3D CASE STUDY FIGURE")
print("=" * 100)

print("\nLoading graphs...")

obj = torch.load(
    GRAPH_FILE,
    map_location="cpu"
)

dataset = obj["dataset"]

# --------------------------------------------------------------------------
# Find graph
# --------------------------------------------------------------------------

def find_graph(pdb, chain):

    for g in dataset:

        gpdb = str(
            g.get("pdb_id", "")
        ).lower().strip()

        gchain = str(
            g.get("chain_id", "")
        ).strip()

        if gpdb == pdb and gchain == chain:
            return g

    return None


# --------------------------------------------------------------------------
# Build case data
# --------------------------------------------------------------------------

case_results = []

fig = plt.figure(
    figsize=(16, 14)
)

for panel, (_, case) in enumerate(
    cases.iterrows(),
    start=1
):

    pdb = case["apo_pdb_id"]
    chain = case["apo_chain"]

    print(
        f"\n[{panel}/4] "
        f"{case['case']}: "
        f"{pdb} chain {chain}"
    )

    graph = find_graph(pdb, chain)

    if graph is None:
        print("ERROR: graph not found")
        continue

    # Coordinates
    pos = graph["pos"].detach().cpu().numpy()

    # Experimental center
    true_center = (
        graph["true_center"]
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------------
    # Predictions
    # --------------------------------------------------------------

    pred = res[
        (res["apo_pdb_id"] == pdb) &
        (res["apo_chain"] == chain)
    ].copy()

    if len(pred) == 0:
        print("ERROR: predictions not found")
        continue

    # Average probability over the 8 seeds
    prob = (
        pred
        .groupby("node_index")["probability"]
        .mean()
    )

    prob = prob.reindex(
        range(len(pos)),
        fill_value=0
    ).to_numpy()

    # --------------------------------------------------------------
    # Gamma=10 sharpened center
    # --------------------------------------------------------------

    gamma = 10.0

    weights = prob ** gamma

    if weights.sum() == 0:
        predicted_center = np.mean(pos, axis=0)
    else:
        weights = weights / weights.sum()

        predicted_center = (
            weights[:, None] * pos
        ).sum(axis=0)

    center_error = np.linalg.norm(
        predicted_center - true_center
    )

    # --------------------------------------------------------------
    # Top 10% predicted residues
    # --------------------------------------------------------------

    n = len(pos)

    k = max(
        1,
        int(np.ceil(0.10 * n))
    )

    top_idx = np.argsort(
        prob
    )[::-1][:k]

    # --------------------------------------------------------------
    # Experimental residues
    # --------------------------------------------------------------

    exp = mapping[
        (mapping["apo_pdb_id"] == pdb) &
        (mapping["apo_chain"] == chain)
    ].copy()

    exp_idx = exp[
        exp["experimental_site"] == 1
    ]["node_index"].astype(int).to_numpy()

    exp_idx = np.unique(exp_idx)

    # --------------------------------------------------------------
    # Overlap
    # --------------------------------------------------------------

    top_set = set(top_idx.tolist())
    exp_set = set(exp_idx.tolist())

    overlap = len(
        top_set.intersection(exp_set)
    )

    precision = (
        overlap / len(top_idx)
        if len(top_idx) > 0
        else 0
    )

    print(
        f"Nodes: {n}"
    )

    print(
        f"Experimental residues: "
        f"{len(exp_idx)}"
    )

    print(
        f"Top 10% residues: "
        f"{len(top_idx)}"
    )

    print(
        f"Top-10% experimental overlap: "
        f"{overlap}"
    )

    print(
        f"Top-10% precision: "
        f"{precision:.4f}"
    )

    print(
        f"Gamma10 center error: "
        f"{center_error:.2f} Å"
    )

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------

    ax = fig.add_subplot(
        2,
        2,
        panel,
        projection="3d"
    )

    # Protein backbone
    ax.plot(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        linewidth=0.7,
        alpha=0.20
    )

    # All residues
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        s=5,
        alpha=0.12
    )

    # Experimental site
    if len(exp_idx) > 0:

        ax.scatter(
            pos[exp_idx, 0],
            pos[exp_idx, 1],
            pos[exp_idx, 2],
            s=32,
            alpha=0.85,
            label="Experimental site"
        )

    # Top predicted residues
    ax.scatter(
        pos[top_idx, 0],
        pos[top_idx, 1],
        pos[top_idx, 2],
        s=24,
        alpha=0.75,
        label="Top 10% prediction"
    )

    # Experimental center
    ax.scatter(
        true_center[0],
        true_center[1],
        true_center[2],
        s=150,
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        label="Experimental center"
    )

    # Predicted center
    ax.scatter(
        predicted_center[0],
        predicted_center[1],
        predicted_center[2],
        s=110,
        marker="X",
        edgecolors="black",
        linewidths=0.8,
        label="γ=10 predicted center"
    )

    # --------------------------------------------------------------
    # Titles
    # --------------------------------------------------------------

    case_label = case["case"].replace(
        "_",
        " "
    ).title()

    ax.set_title(
        f"{case_label}\n"
        f"{pdb.upper()} chain {chain}   "
        f"PR-AUC={case['PR_AUC']:.3f}   "
        f"Error={center_error:.1f} Å",
        fontsize=11
    )

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")

    # Equal-ish aspect ratio
    ranges = np.ptp(pos, axis=0)
    max_range = max(ranges)

    mid = np.mean(pos, axis=0)

    ax.set_xlim(
        mid[0] - max_range / 2,
        mid[0] + max_range / 2
    )

    ax.set_ylim(
        mid[1] - max_range / 2,
        mid[1] + max_range / 2
    )

    ax.set_zlim(
        mid[2] - max_range / 2,
        mid[2] + max_range / 2
    )

    ax.view_init(
        elev=20,
        azim=35
    )

    if panel == 1:
        ax.legend(
            fontsize=8,
            loc="upper right"
        )

    # --------------------------------------------------------------
    # Save numerical summary
    # --------------------------------------------------------------

    case_results.append({
        "case": case["case"],
        "apo_pdb_id": pdb,
        "apo_chain": chain,
        "protein_size": n,
        "experimental_pocket_size": len(exp_idx),
        "top10_size": len(top_idx),
        "top10_experimental_overlap": overlap,
        "top10_precision": precision,
        "experimental_center_x": true_center[0],
        "experimental_center_y": true_center[1],
        "experimental_center_z": true_center[2],
        "predicted_center_x": predicted_center[0],
        "predicted_center_y": predicted_center[1],
        "predicted_center_z": predicted_center[2],
        "gamma10_center_error": center_error,
    })

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------

plt.tight_layout()

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

pd.DataFrame(case_results).to_csv(
    OUT_DATA,
    index=False
)

print("\n" + "=" * 100)
print("PHASE 194 COMPLETE")
print("=" * 100)

print(
    f"\nFigure:\n{OUT_PNG}"
)

print(
    f"\nData:\n{OUT_DATA}"
)

print("\nCase summary:")

print(
    pd.DataFrame(case_results).to_string(
        index=False
    )
)
