import os
import warnings
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from Bio.PDB import MMCIFParser
from Bio.Align import PairwiseAligner

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path("/Users/shivenkhurana/bioproj/cryptic-pocket")

GRAPH_FILE = ROOT / "phase159_input/phase159_cryptobench_anmmsf_graphs.pt"
APO_DIR = ROOT / "phase167_apo"
HOLO_DIR = ROOT / "phase167_holo"

SITE_FILE = ROOT / "phase167_experimental_sites.csv"
PRED_FILE = ROOT / "phase167_validation/phase167_residue_predictions.csv"

OUT_DIR = ROOT / "phase167_validation"
OUT_DIR.mkdir(exist_ok=True)

TRANSFORM_OUT = OUT_DIR / "phase168A_corrected_transforms.csv"
CENTER_OUT = OUT_DIR / "phase168A_corrected_centers.csv"
PROTEIN_OUT = OUT_DIR / "phase168A_corrected_protein_metrics.csv"
SUMMARY_OUT = OUT_DIR / "phase168A_corrected_summary.csv"

# ============================================================
# SETTINGS
# ============================================================

N_SEEDS = 8

AA3 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

parser = MMCIFParser(QUIET=True)

# ============================================================
# KABSCH
# ============================================================

def kabsch(P, Q):
    """
    Find rigid transform mapping P -> Q.

    Row-vector convention:
        Q ~= P @ R.T + t
    """

    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)

    P0 = P - pc
    Q0 = Q - qc

    H = P0.T @ Q0

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = qc - pc @ R.T

    return R, t


# ============================================================
# CHAIN DATA
# ============================================================

def chain_data(chain):

    seq = []
    coords = []
    residues = []

    for residue in chain:

        if residue.id[0].strip() != "":
            continue

        name = residue.resname.upper()

        if name not in AA3:
            continue

        if "CA" not in residue:
            continue

        seq.append(AA3[name])
        coords.append(
            np.asarray(residue["CA"].coord, dtype=float)
        )
        residues.append(residue)

    return (
        "".join(seq),
        np.asarray(coords, dtype=float),
        residues
    )


# ============================================================
# HOLO -> APO
# ============================================================

def holo_to_apo(
    apo_structure,
    apo_chain_id,
    holo_structure,
    holo_chain_id,
    site_holo
):

    apo_chain = apo_structure[0][apo_chain_id]

    # CryptoBench uses "A-B" for the two identical 2BHI copies.
    # For 1XT3 B <- 2BHI A-B, chain A has the better structural alignment.
    if holo_chain_id == "A-B":
        holo_chain_id = "A"

    holo_chain = holo_structure[0][holo_chain_id]

    apo_seq, apo_ca, apo_res = chain_data(apo_chain)
    holo_seq, holo_ca, holo_res = chain_data(holo_chain)

    aligner = PairwiseAligner()
    aligner.mode = "global"

    alignment = aligner.align(
        apo_seq,
        holo_seq
    )[0]

    pairs = []

    for (a0, a1), (h0, h1) in zip(
        alignment.aligned[0],
        alignment.aligned[1]
    ):

        length = min(
            a1 - a0,
            h1 - h0
        )

        for i in range(length):
            pairs.append(
                (
                    a0 + i,
                    h0 + i
                )
            )

    if len(pairs) < 3:
        raise RuntimeError(
            f"Only {len(pairs)} structural pairs"
        )

    P = np.asarray(
        [holo_ca[j] for i, j in pairs]
    )

    Q = np.asarray(
        [apo_ca[i] for i, j in pairs]
    )

    R, t = kabsch(P, Q)

    fitted = P @ R.T + t

    residuals = np.linalg.norm(
        fitted - Q,
        axis=1
    )

    site_apo = site_holo @ R.T + t

    return {
        "R": R,
        "t": t,
        "site_apo": site_apo,
        "n_pairs": len(pairs),
        "mean_rmsd": float(residuals.mean()),
        "median_rmsd": float(np.median(residuals)),
        "max_rmsd": float(residuals.max()),
        "apo_seq_len": len(apo_seq),
        "holo_seq_len": len(holo_seq),
        "apo_res": apo_res,
        "apo_ca": apo_ca,
    }


# ============================================================
# GRAPH / APO REGISTRATION
# ============================================================

blob = torch.load(
    GRAPH_FILE,
    map_location="cpu"
)

graphs = blob["dataset"]

graph_lookup = {}

for g in graphs:

    pid = str(g.pdb_id).lower()
    chain = str(g.chain_id).upper()

    graph_lookup[(pid, chain)] = g

print(
    f"Loaded {len(graph_lookup)} graphs"
)

mapping = pd.read_csv(
    ROOT /
    "phase167_validation/phase167_node_residue_mapping.csv"
)

# ============================================================
# APO -> GRAPH
# ============================================================

def apo_to_graph_transform(
    graph,
    apo_res,
    apo_ca,
    mapping,
    apo_id,
    apo_chain
):

    graph_pos = (
        graph.pos.detach()
        .cpu()
        .numpy()
    )

    m = mapping[
        (mapping.apo_pdb_id.str.lower() == apo_id) &
        (mapping.apo_chain.str.upper() == apo_chain)
    ].drop_duplicates("node_index")

    graph_pts = []
    apo_pts = []

    # Build residue lookup once
    residue_lookup = {}

    for idx, residue in enumerate(apo_res):

        het, seqnum, insertion = residue.id

        if het.strip() != "":
            continue

        key = (
            int(seqnum),
            str(insertion).strip()
        )

        residue_lookup[key] = idx

    for _, row in m.iterrows():

        node = int(row.node_index)

        if node >= len(graph_pos):
            continue

        if pd.isna(row.pdb_residue_number):
            continue

        resnum = int(
            row.pdb_residue_number
        )

        icode = (
            ""
            if pd.isna(row.pdb_insertion_code)
            else str(
                row.pdb_insertion_code
            ).strip()
        )

        key = (resnum, icode)

        if key not in residue_lookup:
            continue

        idx = residue_lookup[key]

        graph_pts.append(
            graph_pos[node]
        )

        apo_pts.append(
            apo_ca[idx]
        )

    graph_pts = np.asarray(
        graph_pts,
        dtype=float
    )

    apo_pts = np.asarray(
        apo_pts,
        dtype=float
    )

    if len(graph_pts) < 3:
        raise RuntimeError(
            f"Only {len(graph_pts)} apo-graph matches"
        )

    translation = np.mean(
        graph_pts - apo_pts,
        axis=0
    )

    residuals = np.linalg.norm(
        (apo_pts + translation)
        - graph_pts,
        axis=1
    )

    return {
        "translation": translation,
        "n_pairs": len(graph_pts),
        "mean_residual": float(
            residuals.mean()
        ),
        "max_residual": float(
            residuals.max()
        )
    }


# ============================================================
# EXPERIMENTAL SITES
# ============================================================

sites = pd.read_csv(
    SITE_FILE
)

# ============================================================
# PREDICTIONS
# ============================================================

pred = pd.read_csv(
    PRED_FILE
)

# Normalize IDs
pred["apo_pdb_id"] = (
    pred["apo_pdb_id"]
    .astype(str)
    .str.lower()
)

pred["apo_chain"] = (
    pred["apo_chain"]
    .astype(str)
    .str.upper()
)

# ============================================================
# ONLY HARD TEST
# ============================================================

test_ids = set(
    pred.apo_pdb_id.unique()
)

sites = sites[
    sites.apo_pdb_id
    .astype(str)
    .str.lower()
    .isin(test_ids)
].copy()

print(
    f"Experimental sites: {len(sites)}"
)

# ============================================================
# TRANSFORM ALL SITES
# ============================================================

transform_rows = []
center_rows = []

failed = []

for n, s in enumerate(
    sites.itertuples(index=False),
    start=1
):

    apo_id = str(
        s.apo_pdb_id
    ).lower()

    apo_chain_id = str(
        s.apo_chain
    ).upper()

    holo_id = str(
        s.holo_pdb_id
    ).lower()

    holo_chain_id = str(
        s.holo_chain
    ).upper()

    site_holo = np.array([
        float(s.site_center_x),
        float(s.site_center_y),
        float(s.site_center_z)
    ])

    try:

        print(
            f"[{n:3d}/{len(sites)}] "
            f"{apo_id.upper()} {apo_chain_id} "
            f"<- {holo_id.upper()} {holo_chain_id}",
            flush=True
        )

        # ----------------------------------------
        # Load structures
        # ----------------------------------------

        apo_file = (
            APO_DIR /
            f"{apo_id}.cif"
        )

        holo_file = (
            HOLO_DIR /
            f"{holo_id}.cif"
        )

        apo_structure = parser.get_structure(
            f"apo_{apo_id}",
            str(apo_file)
        )

        holo_structure = parser.get_structure(
            f"holo_{holo_id}",
            str(holo_file)
        )

        # ----------------------------------------
        # Holo -> apo
        # ----------------------------------------

        h2a = holo_to_apo(
            apo_structure,
            apo_chain_id,
            holo_structure,
            holo_chain_id,
            site_holo
        )

        # ----------------------------------------
        # Graph
        # ----------------------------------------

        graph = graph_lookup[
            (apo_id, apo_chain_id)
        ]

        # ----------------------------------------
        # Apo -> graph
        # ----------------------------------------

        a2g = apo_to_graph_transform(
            graph,
            h2a["apo_res"],
            h2a["apo_ca"],
            mapping,
            apo_id,
            apo_chain_id
        )

        # ----------------------------------------
        # Final center
        # ----------------------------------------

        site_apo = h2a["site_apo"]

        site_graph = (
            site_apo +
            a2g["translation"]
        )

        graph_pos = (
            graph.pos.detach()
            .cpu()
            .numpy()
        )

        nearest = np.min(
            np.linalg.norm(
                graph_pos - site_graph,
                axis=1
            )
        )

        inside = (
            np.all(
                site_graph >= graph_pos.min(axis=0)
            )
            and
            np.all(
                site_graph <= graph_pos.max(axis=0)
            )
        )

        true_center = (
            graph.true_center
            .detach()
            .cpu()
            .numpy()
        )

        true_dist = np.linalg.norm(
            site_graph - true_center
        )

        # ----------------------------------------
        # Transform diagnostics
        # ----------------------------------------

        transform_rows.append({

            "apo_pdb_id": apo_id,
            "apo_chain": apo_chain_id,
            "holo_pdb_id": holo_id,
            "holo_chain": holo_chain_id,

            "n_holo_apo_pairs":
                h2a["n_pairs"],

            "holo_apo_mean_rmsd":
                h2a["mean_rmsd"],

            "holo_apo_median_rmsd":
                h2a["median_rmsd"],

            "holo_apo_max_rmsd":
                h2a["max_rmsd"],

            "n_apo_graph_pairs":
                a2g["n_pairs"],

            "apo_graph_mean_residual":
                a2g["mean_residual"],

            "apo_graph_max_residual":
                a2g["max_residual"],

            "graph_tx":
                a2g["translation"][0],

            "graph_ty":
                a2g["translation"][1],

            "graph_tz":
                a2g["translation"][2],

            "inside_graph_bbox":
                bool(inside),

            "nearest_graph_node":
                float(nearest)
        })

        center_rows.append({

            "apo_pdb_id": apo_id,
            "apo_chain": apo_chain_id,

            "holo_site_x":
                site_holo[0],
            "holo_site_y":
                site_holo[1],
            "holo_site_z":
                site_holo[2],

            "apo_site_x":
                site_apo[0],
            "apo_site_y":
                site_apo[1],
            "apo_site_z":
                site_apo[2],

            "graph_site_x":
                site_graph[0],
            "graph_site_y":
                site_graph[1],
            "graph_site_z":
                site_graph[2],

            "true_center_x":
                true_center[0],
            "true_center_y":
                true_center[1],
            "true_center_z":
                true_center[2],

            "experimental_true_center_distance":
                true_dist,

            "nearest_graph_node":
                nearest,

            "inside_graph_bbox":
                bool(inside)
        })

    except Exception as e:

        print(
            "   FAILED:",
            repr(e)
        )

        failed.append({
            "apo_pdb_id": apo_id,
            "apo_chain": apo_chain_id,
            "holo_pdb_id": holo_id,
            "holo_chain": holo_chain_id,
            "error": repr(e)
        })


# ============================================================
# SAVE TRANSFORM RESULTS
# ============================================================

transform_df = pd.DataFrame(
    transform_rows
)

center_df = pd.DataFrame(
    center_rows
)

transform_df.to_csv(
    TRANSFORM_OUT,
    index=False
)

center_df.to_csv(
    CENTER_OUT,
    index=False
)

if failed:
    pd.DataFrame(failed).to_csv(
        OUT_DIR /
        "phase168A_corrected_failures.csv",
        index=False
    )

print()
print("=" * 70)
print("TRANSFORM SUMMARY")
print("=" * 70)

print(
    "successful:",
    len(center_df),
    "/",
    len(sites)
)

print(
    "failed:",
    len(failed)
)

if len(transform_df):

    print(
        "mean holo->apo RMSD:",
        transform_df.holo_apo_mean_rmsd.mean()
    )

    print(
        "median holo->apo RMSD:",
        transform_df.holo_apo_median_rmsd.median()
    )

    print(
        "max holo->apo RMSD:",
        transform_df.holo_apo_max_rmsd.max()
    )

    print(
        "mean apo->graph residual:",
        transform_df.apo_graph_mean_residual.mean()
    )

    print(
        "bbox inside:",
        transform_df.inside_graph_bbox.mean()
    )

print()
print(
    "Saved:",
    TRANSFORM_OUT
)

print(
    "Saved:",
    CENTER_OUT
)

# ============================================================
# MERGE PREDICTIONS WITH EXPERIMENTAL CENTERS
# ============================================================

pred = pred.merge(
    center_df[
        [
            "apo_pdb_id",
            "apo_chain",
            "graph_site_x",
            "graph_site_y",
            "graph_site_z"
        ]
    ],
    on=[
        "apo_pdb_id",
        "apo_chain"
    ],
    how="inner"
)

print()
print(
    "Prediction rows after center merge:",
    len(pred)
)

# ============================================================
# LOCALIZATION ERROR
# ============================================================

pred["center_error"] = np.sqrt(
    (
        pred.predicted_center_x -
        pred.graph_site_x
    ) ** 2
    +
    (
        pred.predicted_center_y -
        pred.graph_site_y
    ) ** 2
    +
    (
        pred.predicted_center_z -
        pred.graph_site_z
    ) ** 2
)

# ============================================================
# SEED-LEVEL SUMMARY
# ============================================================

seed_metrics = []

for (arm, seed), g in pred.groupby(
    ["arm", "seed"]
):

    e = g.center_error.values

    seed_metrics.append({

        "arm": arm,
        "seed": seed,

        "mean_center_error":
            float(np.mean(e)),

        "median_center_error":
            float(np.median(e)),

        "DCC4":
            float(np.mean(e < 4)),

        "DCC6":
            float(np.mean(e < 6)),

        "DCC8":
            float(np.mean(e < 8)),

        "DCC10":
            float(np.mean(e < 10))
    })

seed_df = pd.DataFrame(
    seed_metrics
)

print()
print("=" * 70)
print("SEED-LEVEL LOCALIZATION")
print("=" * 70)

print(
    seed_df.groupby("arm")[
        [
            "mean_center_error",
            "median_center_error",
            "DCC4",
            "DCC6",
            "DCC8",
            "DCC10"
        ]
    ].agg(["mean", "std"])
)

# ============================================================
# PROTEIN-LEVEL
# ============================================================

protein_rows = []

for (arm, seed, pid, chain), g in pred.groupby(
    [
        "arm",
        "seed",
        "apo_pdb_id",
        "apo_chain"
    ]
):

    e = g.center_error.values

    protein_rows.append({

        "arm": arm,
        "seed": seed,
        "apo_pdb_id": pid,
        "apo_chain": chain,

        "center_error":
            float(np.mean(e)),

        "DCC4":
            float(np.mean(e < 4)),

        "DCC6":
            float(np.mean(e < 6)),

        "DCC8":
            float(np.mean(e < 8)),

        "DCC10":
            float(np.mean(e < 10))
    })

protein_seed = pd.DataFrame(
    protein_rows
)

# Average seeds per protein
protein = (
    protein_seed
    .groupby(
        [
            "arm",
            "apo_pdb_id",
            "apo_chain"
        ]
    )[
        [
            "center_error",
            "DCC4",
            "DCC6",
            "DCC8",
            "DCC10"
        ]
    ]
    .mean()
    .reset_index()
)

protein.to_csv(
    PROTEIN_OUT,
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

summary = (
    protein
    .groupby("arm")[
        [
            "center_error",
            "DCC4",
            "DCC6",
            "DCC8",
            "DCC10"
        ]
    ]
    .agg(["mean", "std", "median"])
)

print()
print("=" * 70)
print("PROTEIN-LEVEL LOCALIZATION")
print("=" * 70)

print(summary)

summary.to_csv(
    SUMMARY_OUT
)

print()
print("Saved:")
print(PROTEIN_OUT)
print(SUMMARY_OUT)
