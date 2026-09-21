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
    "phase195_figures"
)

os.makedirs(OUT_DIR, exist_ok=True)

cases = pd.read_csv(CASE_FILE)
res = pd.read_csv(RESIDUE_FILE)
mapping = pd.read_csv(MAPPING_FILE)

for df in [cases, res, mapping]:
    df["apo_pdb_id"] = (
        df["apo_pdb_id"].astype(str).str.lower().str.strip()
    )
    df["apo_chain"] = (
        df["apo_chain"].astype(str).str.strip()
    )

res = res[res["arm"] == "full_model"].copy()

obj = torch.load(
    GRAPH_FILE,
    map_location="cpu"
)

dataset = obj["dataset"]


def find_graph(pdb, chain):
    for g in dataset:
        gpdb = str(g.get("pdb_id", "")).lower().strip()
        gchain = str(g.get("chain_id", "")).strip()

        if gpdb == pdb and gchain == chain:
            return g
    return None


def get_case_data(case):

    pdb = case["apo_pdb_id"]
    chain = case["apo_chain"]

    graph = find_graph(pdb, chain)

    pos = graph["pos"].detach().cpu().numpy()

    true_center = (
        graph["true_center"]
        .detach()
        .cpu()
        .numpy()
    )

    pred = res[
        (res["apo_pdb_id"] == pdb) &
        (res["apo_chain"] == chain)
    ].copy()

    prob = (
        pred.groupby("node_index")["probability"]
        .mean()
        .reindex(range(len(pos)), fill_value=0)
        .to_numpy()
    )

    weights = prob ** 10

    predicted_center = (
        (weights[:, None] * pos).sum(axis=0)
        / weights.sum()
    )

    n = len(pos)

    k = max(1, int(np.ceil(0.10 * n)))

    top_idx = np.argsort(prob)[::-1][:k]

    exp = mapping[
        (mapping["apo_pdb_id"] == pdb) &
        (mapping["apo_chain"] == chain)
    ]

    exp_idx = (
        exp[exp["experimental_site"] == 1]
        ["node_index"]
        .astype(int)
        .unique()
    )

    return (
        pos,
        true_center,
        predicted_center,
        top_idx,
        exp_idx
    )


# ----------------------------------------------------------------------
# Create one zoomed figure per case
# ----------------------------------------------------------------------

for _, case in cases.iterrows():

    name = case["case"]
    pdb = case["apo_pdb_id"]
    chain = case["apo_chain"]

    (
        pos,
        true_center,
        predicted_center,
        top_idx,
        exp_idx
    ) = get_case_data(case)

    # --------------------------------------------------------------
    # Zoom around experimental center
    # --------------------------------------------------------------

    distances = np.linalg.norm(
        pos - true_center,
        axis=1
    )

    # 15 Å local neighborhood
    local = distances <= 15.0

    # Always include experimental and predicted residues
    local[exp_idx] = True
    local[top_idx] = True

    local_pos = pos[local]

    # --------------------------------------------------------------
    # Figure
    # --------------------------------------------------------------

    fig = plt.figure(
        figsize=(9, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # Local structure
    ax.scatter(
        local_pos[:, 0],
        local_pos[:, 1],
        local_pos[:, 2],
        s=12,
        alpha=0.18
    )

    # Experimental site
    ax.scatter(
        pos[exp_idx, 0],
        pos[exp_idx, 1],
        pos[exp_idx, 2],
        s=55,
        alpha=0.9,
        label="Experimental site"
    )

    # Top predictions
    ax.scatter(
        pos[top_idx, 0],
        pos[top_idx, 1],
        pos[top_idx, 2],
        s=35,
        alpha=0.75,
        label="Top 10% prediction"
    )

    # Experimental center
    ax.scatter(
        true_center[0],
        true_center[1],
        true_center[2],
        s=220,
        marker="*",
        edgecolors="black",
        linewidths=1,
        label="Experimental center"
    )

    # Predicted center
    ax.scatter(
        predicted_center[0],
        predicted_center[1],
        predicted_center[2],
        s=170,
        marker="X",
        edgecolors="black",
        linewidths=1,
        label="γ=10 predicted center"
    )

    error = np.linalg.norm(
        predicted_center - true_center
    )

    overlap = len(
        set(top_idx.tolist()) &
        set(exp_idx.tolist())
    )

    precision = overlap / len(top_idx)

    ax.set_title(
        f"{name.replace('_', ' ').title()}\n"
        f"{pdb.upper()} chain {chain} | "
        f"PR-AUC={case['PR_AUC']:.3f} | "
        f"Center error={error:.1f} Å | "
        f"Top-10% precision={precision:.3f}",
        fontsize=13
    )

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")

    ax.legend(
        fontsize=9
    )

    ax.view_init(
        elev=22,
        azim=35
    )

    plt.tight_layout()

    path = os.path.join(
        OUT_DIR,
        f"{pdb}_{chain}_{name}_zoom.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {path}"
    )

print("\nPHASE 195 COMPLETE")
