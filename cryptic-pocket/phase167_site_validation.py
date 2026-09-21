import os
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB import MMCIFParser
from Bio.Align import PairwiseAligner
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path("~/bioproj/cryptic-pocket").expanduser()

GRAPH_FILE = ROOT / "phase159_input/phase159_cryptobench_anmmsf_graphs.pt"
DATASET_JSON = ROOT / "cryptobench_dataset.json"
PRED_FILE = ROOT / "phase167_validation/phase167_residue_predictions.csv"

APO_DIR = ROOT / "phase167_apo"
OUT_DIR = ROOT / "phase167_validation"
APO_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Load graph dataset
# ---------------------------------------------------------------------

obj = torch.load(GRAPH_FILE, weights_only=False)
graphs = obj["dataset"]
alphabet = obj["aa_alphabet"]

graph_by_key = {
    (g.pdb_id.lower(), g.chain_id): g
    for g in graphs
}

# ---------------------------------------------------------------------
# Load official CryptoBench dataset
# ---------------------------------------------------------------------

with open(DATASET_JSON) as f:
    cb = json.load(f)

# ---------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------

pred = pd.read_csv(PRED_FILE)

# Only hard-test proteins
hard_test = pred["apo_pdb_id"].str.lower().isin(
    {
        g.pdb_id.lower()
        for g in graphs
        if g.split == "test"
    }
)

# The prediction file already contains the Phase166 hard-test proteins.
# Keep every prediction row.
pred = pred.copy()

print("Prediction rows:", len(pred))
print("Prediction proteins:", pred["apo_pdb_id"].nunique())

# ---------------------------------------------------------------------
# Amino-acid conversion
# ---------------------------------------------------------------------

aa3to1 = {
    "ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C",
    "GLN":"Q", "GLU":"E", "GLY":"G", "HIS":"H", "ILE":"I",
    "LEU":"L", "LYS":"K", "MET":"M", "PHE":"F", "PRO":"P",
    "SER":"S", "THR":"T", "TRP":"W", "TYR":"Y", "VAL":"V",
}

# ---------------------------------------------------------------------
# Download apo structure
# ---------------------------------------------------------------------

def download_apo(pdb_id):
    pdb_id = pdb_id.upper()
    out = APO_DIR / f"{pdb_id.lower()}.cif"

    if out.exists() and out.stat().st_size > 1000:
        return out

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"

    print("Downloading", pdb_id)

    try:
        urllib.request.urlretrieve(url, out)
        if out.stat().st_size < 1000:
            out.unlink(missing_ok=True)
            raise RuntimeError("Downloaded file is unexpectedly small")
        return out
    except Exception as e:
        print("DOWNLOAD FAILED:", pdb_id, e)
        out.unlink(missing_ok=True)
        return None


# ---------------------------------------------------------------------
# Extract PDB chain sequence + residue IDs
# ---------------------------------------------------------------------

def get_pdb_chain(cif_path, chain_id):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)

    model = structure[0]

    if chain_id not in model:
        raise KeyError(
            f"Chain {chain_id} not found; available chains: "
            f"{list(model.child_dict.keys())}"
        )

    chain = model[chain_id]

    residues = []
    sequence = []

    for r in chain:
        # Standard polymer residue only
        if r.id[0] != " ":
            continue

        if r.resname not in aa3to1:
            continue

        sequence.append(aa3to1[r.resname])
        residues.append(r)

    return "".join(sequence), residues


# ---------------------------------------------------------------------
# Align graph sequence to PDB sequence
# ---------------------------------------------------------------------

aligner = PairwiseAligner()
aligner.mode = "global"
aligner.match_score = 2
aligner.mismatch_score = -1
aligner.open_gap_score = -5
aligner.extend_gap_score = -0.5


def map_graph_to_pdb(graph_seq, pdb_seq, pdb_residues):
    alignment = aligner.align(graph_seq, pdb_seq)[0]

    mapping = {}
    graph_i = 0
    pdb_i = 0

    # Alignment coordinates describe blocks:
    #
    # [[graph_start, graph_end], ...]
    # [[pdb_start, pdb_end], ...]
    #
    # Walk the aligned blocks and map matching residue positions.

    for (g0, g1), (p0, p1) in zip(
        alignment.aligned[0],
        alignment.aligned[1],
    ):
        length = min(g1 - g0, p1 - p0)

        for j in range(length):
            mapping[g0 + j] = pdb_residues[p0 + j]

    return mapping, alignment


# ---------------------------------------------------------------------
# Build experimental site lookup
# ---------------------------------------------------------------------

def get_experimental_residues(apo_pdb, apo_chain):
    entries = cb.get(apo_pdb.lower(), [])

    # CryptoBench JSON keys are lowercase PDB IDs.
    # Try exact apo chain.
    matches = [
        e for e in entries
        if e.get("apo_chain") == apo_chain
    ]

    experimental = set()

    for e in matches:
        for s in e.get("apo_pocket_selection", []):
            # Expected format: B_123
            if "_" not in s:
                continue

            chain, resnum = s.split("_", 1)

            if chain != apo_chain:
                continue

            # Keep insertion codes intact if present.
            experimental.add(resnum)

    return experimental


# ---------------------------------------------------------------------
# Process every protein appearing in predictions
# ---------------------------------------------------------------------

mapping_rows = []
failures = []

protein_keys = (
    pred[["apo_pdb_id", "apo_chain"]]
    .drop_duplicates()
    .sort_values(["apo_pdb_id", "apo_chain"])
)

print("Proteins to map:", len(protein_keys))

for n, row in enumerate(protein_keys.itertuples(index=False), 1):

    pdb_id = row.apo_pdb_id
    chain_id = row.apo_chain
    key = (pdb_id.lower(), chain_id)

    print(f"[{n}/{len(protein_keys)}] {pdb_id}{chain_id}")

    graph = graph_by_key.get(key)

    if graph is None:
        failures.append({
            "apo_pdb_id": pdb_id,
            "apo_chain": chain_id,
            "reason": "graph_not_found",
        })
        continue

    # Graph sequence
    aa_idx = graph.x[:, :21].argmax(dim=1).tolist()
    graph_seq = "".join(alphabet[i] for i in aa_idx)

    # Download structure
    cif = download_apo(pdb_id)

    if cif is None:
        failures.append({
            "apo_pdb_id": pdb_id,
            "apo_chain": chain_id,
            "reason": "download_failed",
        })
        continue

    try:
        pdb_seq, pdb_residues = get_pdb_chain(cif, chain_id)

        mapping, alignment = map_graph_to_pdb(
            graph_seq,
            pdb_seq,
            pdb_residues,
        )

        experimental = get_experimental_residues(
            pdb_id,
            chain_id,
        )

        mapped = 0

        for node_index in range(graph.num_nodes):

            r = mapping.get(node_index)

            if r is None:
                mapping_rows.append({
                    "apo_pdb_id": pdb_id,
                    "apo_chain": chain_id,
                    "node_index": node_index,
                    "pdb_residue_number": None,
                    "pdb_insertion_code": None,
                    "residue_name": None,
                    "residue": None,
                    "experimental_site": 0,
                    "mapping_status": "unmapped",
                })
                continue

            resnum = str(r.id[1])
            icode = str(r.id[2]).strip()

            residue_id = resnum + icode

            mapping_rows.append({
                "apo_pdb_id": pdb_id,
                "apo_chain": chain_id,
                "node_index": node_index,
                "pdb_residue_number": r.id[1],
                "pdb_insertion_code": icode,
                "residue_name": r.resname,
                "residue": aa3to1[r.resname],
                "experimental_site": int(residue_id in experimental),
                "mapping_status": "mapped",
            })

            mapped += 1

        identity = 0.0

        # Calculate simple sequence identity over aligned matches.
        for (g0, g1), (p0, p1) in zip(
            alignment.aligned[0],
            alignment.aligned[1],
        ):
            for j in range(min(g1-g0, p1-p0)):
                if graph_seq[g0+j] == pdb_seq[p0+j]:
                    identity += 1

        identity /= max(len(graph_seq), 1)

        print(
            f"  graph={len(graph_seq)} "
            f"pdb={len(pdb_seq)} "
            f"mapped={mapped}/{len(graph_seq)} "
            f"identity={identity:.3f} "
            f"experimental={len(experimental)}"
        )

    except Exception as e:
        failures.append({
            "apo_pdb_id": pdb_id,
            "apo_chain": chain_id,
            "reason": repr(e),
        })


# ---------------------------------------------------------------------
# Save mapping
# ---------------------------------------------------------------------

mapping_df = pd.DataFrame(mapping_rows)

mapping_file = OUT_DIR / "phase167_node_residue_mapping.csv"
mapping_df.to_csv(mapping_file, index=False)

failure_file = OUT_DIR / "phase167_mapping_failures.csv"
pd.DataFrame(failures).to_csv(failure_file, index=False)

print()
print("Mapping rows:", len(mapping_df))
print("Mapping proteins:", mapping_df["apo_pdb_id"].nunique())
print("Failures:", len(failures))
print("Saved:", mapping_file)
print("Saved:", failure_file)

# ---------------------------------------------------------------------
# Join predictions
# ---------------------------------------------------------------------

merged = pred.merge(
    mapping_df,
    on=["apo_pdb_id", "apo_chain", "node_index"],
    how="left",
    validate="many_to_one",
)

merged_file = OUT_DIR / "phase167_experimental_site_predictions.csv"
merged.to_csv(merged_file, index=False)

print()
print("Merged rows:", len(merged))
print("Merged missing residue mappings:",
      merged["pdb_residue_number"].isna().sum())
print("Experimental positives:",
      merged["experimental_site"].sum())
print("Saved:", merged_file)

# ---------------------------------------------------------------------
# Overall residue-level metrics
# ---------------------------------------------------------------------

def metrics(df, label):
    df = df.dropna(subset=["experimental_site", "probability"])

    y = df["experimental_site"].astype(int).to_numpy()
    p = df["probability"].astype(float).to_numpy()
    pred_label = (p >= 0.5).astype(int)

    if y.sum() == 0:
        return None

    return {
        "arm": label,
        "n_rows": len(df),
        "n_proteins": df["apo_pdb_id"].nunique(),
        "positives": int(y.sum()),
        "positive_fraction": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "mcc": float(matthews_corrcoef(y, pred_label)),
        "f1": float(f1_score(y, pred_label, zero_division=0)),
        "precision": float(precision_score(y, pred_label, zero_division=0)),
        "recall": float(recall_score(y, pred_label, zero_division=0)),
    }


metric_rows = []

for arm in sorted(merged["arm"].unique()):
    sub = merged[merged["arm"] == arm]
    metric_rows.append(metrics(sub, arm))

metrics_df = pd.DataFrame(metric_rows)

metrics_file = OUT_DIR / "phase167_experimental_site_metrics.csv"
metrics_df.to_csv(metrics_file, index=False)

print()
print("=== EXPERIMENTAL SITE METRICS ===")
print(metrics_df.to_string(index=False))
print()
print("Saved:", metrics_file)

# ---------------------------------------------------------------------
# Per-seed metrics
# ---------------------------------------------------------------------

seed_rows = []

for (arm, seed), sub in merged.groupby(["arm", "seed"]):
    m = metrics(sub, arm)
    if m is not None:
        m["seed"] = seed
        seed_rows.append(m)

seed_df = pd.DataFrame(seed_rows)

seed_file = OUT_DIR / "phase167_experimental_site_seed_metrics.csv"
seed_df.to_csv(seed_file, index=False)

print()
print("=== SEED METRICS ===")
print(seed_df.to_string(index=False))
print()
print("Saved:", seed_file)
