import os
import pandas as pd
import numpy as np

BASE = "phase167_validation"

CASE_FILE = os.path.join(
    BASE,
    "phase191_case_studies.csv"
)

RESIDUE_FILE = os.path.join(
    BASE,
    "phase183A_experimental_residue_validation.csv"
)

OUT = os.path.join(
    BASE,
    "phase192_case_residue_analysis.csv"
)

cases = pd.read_csv(CASE_FILE)
res = pd.read_csv(RESIDUE_FILE)

# Normalize identifiers
for df in [cases, res]:
    df["apo_pdb_id"] = df["apo_pdb_id"].astype(str).str.lower().str.strip()
    df["apo_chain"] = df["apo_chain"].astype(str).str.strip()

# Only full model for the main case studies
res = res[
    res["arm"] == "full_model"
].copy()

# --------------------------------------------------------------------------
# Determine the top 10% predicted residues for each case/seed
# --------------------------------------------------------------------------

rows = []

for _, case in cases.iterrows():

    pdb = case["apo_pdb_id"]
    chain = case["apo_chain"]

    sub = res[
        (res["apo_pdb_id"] == pdb) &
        (res["apo_chain"] == chain)
    ].copy()

    if len(sub) == 0:
        print(f"WARNING: no predictions found for {pdb} {chain}")
        continue

    for seed in sorted(sub["seed"].unique()):

        s = sub[sub["seed"] == seed].copy()

        n = len(s)
        k = max(1, int(np.ceil(0.10 * n)))

        s = s.sort_values(
            "probability",
            ascending=False
        ).copy()

        s["rank"] = np.arange(1, len(s) + 1)

        s["top10"] = s["rank"] <= k

        # Save every residue in the top 10%
        top = s[s["top10"]].copy()

        for _, r in top.iterrows():

            rows.append({
                "case": case["case"],
                "apo_pdb_id": pdb,
                "apo_chain": chain,
                "seed": seed,
                "node_index": r["node_index"],
                "rank": r["rank"],
                "probability": r["probability"],
                "experimental_site": r["experimental_site"],
            })

out = pd.DataFrame(rows)

out.to_csv(
    OUT,
    index=False
)

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

print("=" * 100)
print("PHASE 192 - CASE RESIDUE ANALYSIS")
print("=" * 100)

for case_name in cases["case"]:

    c = out[out["case"] == case_name].copy()

    if len(c) == 0:
        continue

    print("\n" + "-" * 90)
    print(case_name)

    print(
        f"Protein: "
        f"{c['apo_pdb_id'].iloc[0]} "
        f"chain {c['apo_chain'].iloc[0]}"
    )

    print(
        f"Top-10% residue rows: {len(c)}"
    )

    print(
        f"Seeds represented: {c['seed'].nunique()}"
    )

    # Per-seed precision
    seed_stats = (
        c.groupby("seed")
        .agg(
            top10_n=("node_index", "count"),
            experimental_hits=("experimental_site", "sum"),
            mean_probability=("probability", "mean"),
            max_probability=("probability", "max")
        )
        .reset_index()
    )

    seed_stats["precision"] = (
        seed_stats["experimental_hits"] /
        seed_stats["top10_n"]
    )

    print(
        f"Mean top-10% precision: "
        f"{seed_stats['precision'].mean():.4f}"
    )

    print(
        f"Mean experimental hits: "
        f"{seed_stats['experimental_hits'].mean():.2f}"
    )

    print(
        f"Mean top-10% probability: "
        f"{seed_stats['mean_probability'].mean():.4f}"
    )

    print(
        f"Maximum predicted probability: "
        f"{seed_stats['max_probability'].max():.4f}"
    )

    # Pool all seeds and calculate experimental fraction
    print(
        f"Pooled experimental-positive fraction: "
        f"{c['experimental_site'].mean():.4f}"
    )

print("\n" + "=" * 100)
print("PHASE 192 COMPLETE")
print("=" * 100)

print(f"Saved: {OUT}")
