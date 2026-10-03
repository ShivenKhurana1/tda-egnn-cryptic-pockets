import os
import pandas as pd
import numpy as np

BASE = "phase167_validation"

DATA = os.path.join(
    BASE,
    "phase190_figures",
    "phase190_figure_data.csv"
)

OUT = os.path.join(
    BASE,
    "phase191_case_studies.csv"
)

df = pd.read_csv(DATA)

# Only full model for primary case studies
sub = df[
    df["arm"] == "full_model"
].copy()

# --------------------------------------------------------------------------
# Composite ranks
# --------------------------------------------------------------------------

# Higher PR-AUC = better
sub["pr_rank"] = sub["PR_AUC"].rank(
    ascending=False,
    method="average"
)

# Lower center error = better
sub["localization_rank"] = sub["center_error"].rank(
    ascending=True,
    method="average"
)

# Normalize to 0-1
sub["pr_norm"] = (
    sub["PR_AUC"] - sub["PR_AUC"].min()
) / (
    sub["PR_AUC"].max() - sub["PR_AUC"].min()
)

sub["loc_norm"] = (
    sub["center_error"].max() - sub["center_error"]
) / (
    sub["center_error"].max() - sub["center_error"].min()
)

# Combined performance score for selecting representative examples.
sub["combined_score"] = (
    0.5 * sub["pr_norm"] +
    0.5 * sub["loc_norm"]
)

# --------------------------------------------------------------------------
# Success
# --------------------------------------------------------------------------

success = (
    sub
    .sort_values(
        ["combined_score", "PR_AUC"],
        ascending=[False, False]
    )
    .head(1)
    .copy()
)

success["case"] = "strong_success"


# --------------------------------------------------------------------------
# Intermediate
# --------------------------------------------------------------------------

# Pick protein closest to median PR-AUC and median center error
pr_med = sub["PR_AUC"].median()
loc_med = sub["center_error"].median()

sub["median_distance"] = (
    ((sub["PR_AUC"] - pr_med) / sub["PR_AUC"].std()) ** 2
    +
    ((sub["center_error"] - loc_med) / sub["center_error"].std()) ** 2
)

intermediate = (
    sub
    .sort_values("median_distance")
    .head(1)
    .copy()
)

intermediate["case"] = "intermediate"


# --------------------------------------------------------------------------
# Difficult overall failure
# --------------------------------------------------------------------------

failure = (
    sub
    .sort_values(
        ["combined_score", "PR_AUC"],
        ascending=[True, True]
    )
    .head(1)
    .copy()
)

failure["case"] = "difficult_failure"


# --------------------------------------------------------------------------
# Small-pocket failure
# --------------------------------------------------------------------------

small_cutoff = sub["experimental_pocket_size"].quantile(0.20)

small = sub[
    sub["experimental_pocket_size"] <= small_cutoff
].copy()

small_failure = (
    small
    .sort_values(
        ["combined_score", "PR_AUC"],
        ascending=[True, True]
    )
    .head(1)
    .copy()
)

small_failure["case"] = "small_pocket_failure"


# --------------------------------------------------------------------------
# Combine
# --------------------------------------------------------------------------

cases = pd.concat(
    [
        success,
        intermediate,
        failure,
        small_failure,
    ],
    ignore_index=True
)

# Remove accidental duplicates while preserving first case
cases = cases.drop_duplicates(
    subset=["apo_pdb_id", "apo_chain"],
    keep="first"
)

columns = [
    "case",
    "apo_pdb_id",
    "apo_chain",
    "experimental_pocket_size",
    "experimental_pocket_fraction",
    "protein_size",
    "PR_AUC",
    "ROC_AUC",
    "center_error",
    "dcc4",
    "dcc6",
    "dcc8",
    "dcc10",
    "combined_score",
]

cases = cases[columns]

cases.to_csv(
    OUT,
    index=False
)

# --------------------------------------------------------------------------
# Print
# --------------------------------------------------------------------------

print("=" * 100)
print("PHASE 191 - CASE STUDY SELECTION")
print("=" * 100)

print(f"\nSmall-pocket threshold: {small_cutoff:.1f} residues")

print("\nSelected cases:\n")

for _, r in cases.iterrows():

    print("-" * 90)

    print(
        f"{r['case']}: "
        f"{r['apo_pdb_id']} chain {r['apo_chain']}"
    )

    print(
        f"Pocket size: {r['experimental_pocket_size']:.0f} residues "
        f"({r['experimental_pocket_fraction']:.3f} of protein)"
    )

    print(
        f"Protein size: {r['protein_size']:.0f} residues"
    )

    print(
        f"PR-AUC: {r['PR_AUC']:.4f}"
    )

    print(
        f"ROC-AUC: {r['ROC_AUC']:.4f}"
    )

    print(
        f"Center error: {r['center_error']:.2f} Å"
    )

    print(
        f"DCC4: {r['dcc4']:.3f}"
    )

    print(
        f"DCC8: {r['dcc8']:.3f}"
    )

    print(
        f"DCC10: {r['dcc10']:.3f}"
    )

print("\n" + "=" * 100)
print("PHASE 191 COMPLETE")
print("=" * 100)

print(f"Saved: {OUT}")
