import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("phase167_validation")

INPUT = BASE / "phase179A_pocket_enrichment.csv"

OUT = BASE / "phase181A_arm_comparison.csv"

df = pd.read_csv(INPUT)

print("=" * 110)
print("PHASE 181A - DIRECT ARM-TO-ARM CENTER-PROXIMITY COMPARISON")
print("=" * 110)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

FRACTIONS = [
    0.01,
    0.02,
    0.05,
    0.10,
]

CUTOFFS = [
    4,
    6,
    8,
    10,
]

PAIRS = [
    ("full_model", "esm_structure"),
    ("full_tda", "esm_structure"),
    ("full_model", "full_tda"),
]

N_PERM = 10000

RNG = np.random.default_rng(181)

# ------------------------------------------------------------
# Collapse 8 seeds -> one value per protein
# ------------------------------------------------------------

protein = (
    df
    .groupby(
        [
            "arm",
            "apo_pdb_id",
            "apo_chain",
            "fraction",
            "cutoff",
        ],
        as_index=False,
    )
    .agg(
        selected_fraction=(
            "top_fraction_fraction",
            "mean",
        ),
        enrichment=(
            "enrichment",
            "mean",
        ),
        n_seeds=(
            "seed",
            "nunique",
        ),
    )
)

print(
    "Protein-level rows:",
    protein.shape
)

# ------------------------------------------------------------
# Paired sign-flip permutation
# ------------------------------------------------------------

def paired_permutation(
    a,
    b,
    n_perm=10000,
    rng=None,
):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    diff = a - b

    valid = np.isfinite(diff)

    diff = diff[valid]

    if len(diff) == 0:
        return np.nan

    observed = np.mean(diff)

    nonzero = diff != 0
    diff = diff[nonzero]

    if len(diff) == 0:
        return 1.0

    signs = rng.choice(
        [-1.0, 1.0],
        size=(n_perm, len(diff)),
    )

    perm = (
        signs * diff
    ).mean(axis=1)

    p = (
        1
        + np.sum(
            np.abs(perm)
            >= abs(observed)
        )
    ) / (n_perm + 1)

    return p


# ------------------------------------------------------------
# BH correction
# ------------------------------------------------------------

def bh_fdr(pvalues):

    pvalues = np.asarray(
        pvalues,
        dtype=float,
    )

    out = np.full(
        len(pvalues),
        np.nan,
    )

    valid = np.isfinite(pvalues)

    p = pvalues[valid]

    if len(p) == 0:
        return out

    order = np.argsort(p)

    ranked = p[order]

    n = len(ranked)

    q = (
        ranked
        * n
        / np.arange(1, n + 1)
    )

    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.minimum(
        q,
        1.0,
    )

    temp = np.empty_like(q)

    temp[order] = q

    out[valid] = temp

    return out


# ------------------------------------------------------------
# Pairwise comparisons
# ------------------------------------------------------------

records = []

for better, baseline in PAIRS:

    for fraction in FRACTIONS:

        for cutoff in CUTOFFS:

            a = protein[
                (protein["arm"] == better)
                & (
                    protein["fraction"]
                    == fraction
                )
                & (
                    protein["cutoff"]
                    == cutoff
                )
            ]

            b = protein[
                (protein["arm"] == baseline)
                & (
                    protein["fraction"]
                    == fraction
                )
                & (
                    protein["cutoff"]
                    == cutoff
                )
            ]

            key_cols = [
                "apo_pdb_id",
                "apo_chain",
            ]

            merged = a.merge(
                b,
                on=key_cols,
                suffixes=(
                    "_better",
                    "_baseline",
                ),
            )

            if len(merged) == 0:
                continue

            x = merged[
                "selected_fraction_better"
            ].to_numpy()

            y = merged[
                "selected_fraction_baseline"
            ].to_numpy()

            diff = x - y

            p = paired_permutation(
                x,
                y,
                n_perm=N_PERM,
                rng=RNG,
            )

            records.append(
                {
                    "comparison":
                        f"{better} vs {baseline}",

                    "better_arm":
                        better,

                    "baseline_arm":
                        baseline,

                    "fraction":
                        fraction,

                    "cutoff":
                        cutoff,

                    "n_proteins":
                        len(merged),

                    "better_mean":
                        np.mean(x),

                    "baseline_mean":
                        np.mean(y),

                    "mean_difference":
                        np.mean(diff),

                    "median_difference":
                        np.median(diff),

                    "better_enrichment":
                        np.mean(
                            merged[
                                "enrichment_better"
                            ]
                        ),

                    "baseline_enrichment":
                        np.mean(
                            merged[
                                "enrichment_baseline"
                            ]
                        ),

                    "p_permutation":
                        p,
                }
            )

stats = pd.DataFrame(records)

stats["q_bh"] = bh_fdr(
    stats["p_permutation"]
)

stats["p_bonferroni"] = np.minimum(
    stats["p_permutation"]
    * len(stats),
    1.0,
)

stats["bh_significant"] = (
    stats["q_bh"] < 0.05
)

stats["bonferroni_significant"] = (
    stats["p_bonferroni"] < 0.05
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

stats.to_csv(
    OUT,
    index=False,
)

# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print("\n" + "=" * 110)
print("ARM-TO-ARM RESULTS")
print("=" * 110)

print(
    stats.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}",
    )
)

print("\n" + "=" * 110)
print("SIGNIFICANCE COUNTS")
print("=" * 110)

for comparison in stats["comparison"].unique():

    s = stats[
        stats["comparison"]
        == comparison
    ]

    print(
        f"{comparison}: "
        f"{s['bh_significant'].sum()}/"
        f"{len(s)} BH significant; "
        f"{s['bonferroni_significant'].sum()}/"
        f"{len(s)} Bonferroni significant"
    )

print("\nSaved:")
print(OUT)
