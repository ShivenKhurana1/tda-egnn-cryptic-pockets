import os
import torch
import pandas as pd

BASE = "phase167_validation"

GRAPH_FILE = (
    "phase159_input/"
    "phase159_cryptobench_anmmsf_graphs.pt"
)

CASE_FILE = os.path.join(
    BASE,
    "phase191_case_studies.csv"
)

MAPPING_FILE = os.path.join(
    BASE,
    "phase167_node_residue_mapping.csv"
)

cases = pd.read_csv(CASE_FILE)
mapping = pd.read_csv(MAPPING_FILE)

cases["apo_pdb_id"] = (
    cases["apo_pdb_id"].astype(str).str.lower().str.strip()
)

cases["apo_chain"] = (
    cases["apo_chain"].astype(str).str.strip()
)

mapping["apo_pdb_id"] = (
    mapping["apo_pdb_id"].astype(str).str.lower().str.strip()
)

mapping["apo_chain"] = (
    mapping["apo_chain"].astype(str).str.strip()
)

print("=" * 100)
print("PHASE 193 - CASE STRUCTURE CHECK")
print("=" * 100)

print("\nLoading graph file...")
obj = torch.load(
    GRAPH_FILE,
    map_location="cpu"
)

dataset = obj["dataset"]

print(f"Graphs available: {len(dataset)}")

for _, case in cases.iterrows():

    pdb = case["apo_pdb_id"]
    chain = case["apo_chain"]

    print("\n" + "-" * 90)
    print(
        f"{case['case']}: {pdb} chain {chain}"
    )

    found_graph = False

    for g in dataset:

        gpdb = str(g.get("pdb_id", "")).lower().strip()
        gchain = str(g.get("chain_id", "")).strip()

        if gpdb == pdb and gchain == chain:

            found_graph = True

            print("Graph: FOUND")

            print(
                f"Nodes: {g['x'].shape[0]}"
            )

            print(
                f"Features: {g['x'].shape[1]}"
            )

            print(
                f"Coordinates: {tuple(g['pos'].shape)}"
            )

            print(
                f"true_center: "
                f"{g['true_center'].numpy()}"
            )

            break

    if not found_graph:
        print("Graph: NOT FOUND")

    m = mapping[
        (mapping["apo_pdb_id"] == pdb) &
        (mapping["apo_chain"] == chain)
    ]

    print(
        f"Residue mapping rows: {len(m)}"
    )

    if len(m) > 0:

        print(
            f"Mapping columns: "
            f"{list(m.columns)}"
        )

        if "experimental_site" in m.columns:

            print(
                f"Experimental positives: "
                f"{m['experimental_site'].sum()}"
            )

print("\n" + "=" * 100)
print("PHASE 193 CHECK COMPLETE")
print("=" * 100)
