import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("phase167_validation")

INPUT = BASE / "phase179A_pocket_enrichment.csv"

OUT = BASE / "phase180A_enrichment_stats.csv"
SUMMARY = BASE / "phase180A_enrichment_stats_summary.csv"

df = pd.read_csv(INPUT)

print("=" * 110)
print("PHASE 180A - PROTEIN-LEVEL CENTER-PROXIMITY ENRICHMENT STATISTICS")
print("=" * 110)

print("Input:", df.shape)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

FRACTIONS = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
]

CUTOFFS = [
    4,
    6,
    8,
    10,
    12,
]

ARMS = sorted(df["arm"].unique())

N_PERM = 10000

RNG = np.random.default_rng(180)

# ------------------------------------------------------------
# Collapse seeds to protein level
# ------------------------------------------------------------

# Each protein has 8 seeds.
#
# For each protein/fraction/cutoff, average:
#   background fraction
#   selected fraction
#
# This prevents the 8 seeds from being treated as 8
# independent proteins.

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
        background_fraction=(
            "background_fraction",
            "mean",
        ),
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

print(
    "Unique proteins:",
    protein[
        ["apo_pdb_id", "apo_chain"]
    ]
    .drop_duplicates()
    .shape[0]
)

# ------------------------------------------------------------
# Paired permutation test
# ------------------------------------------------------------

def paired_permutation_pvalue(
    selected,
    background,
    n_perm=10000,
    rng=None,
):
    """
    Test whether selected_fraction > background_fraction.

    Null hypothesis:
        no directional difference between selected
        and background.

    Paired sign-flip permutation.
    """

    selected = np.asarray(
        selected,
        dtype=float,
    )

    background = np.asarray(
        background,
        dtype=float,
    )

    diff = selected - background

    diff = diff[np.isfinite(diff)]

    if len(diff) == 0:
        return np.nan

    observed = np.mean(diff)

    # Exact zero difference contributes nothing.
    nonzero = diff != 0

    diff = diff[nonzero]

    if len(diff) == 0:
        return 1.0

    # Generate random sign flips.
    signs = rng.choice(
        [-1.0, 1.0],
        size=(n_perm, len(diff)),
    )

    perm_means = (
        (signs * diff).mean(axis=1)
    )

    # One-sided test: selected > background
    p = (
        1
        + np.sum(
            perm_means >= observed
        )
    ) / (n_perm + 1)

    return p


# ------------------------------------------------------------
# Benjamini-Hochberg
# ------------------------------------------------------------

def bh_fdr(pvalues):

    pvalues = np.asarray(
        pvalues,
        dtype=float,
    )

    result = np.full(
        len(pvalues),
        np.nan,
    )

    valid = np.isfinite(pvalues)

    if not np.any(valid):
        return result

    p = pvalues[valid]

    order = np.argsort(p)

    ranked = p[order]

    n = len(ranked)

    adjusted = (
        ranked
        * n
        / np.arange(1, n + 1)
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.minimum(
        adjusted,
        1.0,
    )

    temp = np.empty_like(
        adjusted
    )

    temp[order] = adjusted

    result[valid] = temp

    return result


# ------------------------------------------------------------
# Run statistical tests
# ------------------------------------------------------------

records = []

for arm in ARMS:

    for fraction in FRACTIONS:

        for cutoff in CUTOFFS:

            g = protein[
                (protein["arm"] == arm)
                & (
                    protein["fraction"]
                    == fraction
                )
                & (
                    protein["cutoff"]
                    == cutoff
                )
            ].copy()

            if len(g) == 0:
                continue

            selected = (
                g["selected_fraction"]
                .to_numpy()
            )

            background = (
                g["background_fraction"]
                .to_numpy()
            )

            diff = (
                selected
                - background
            )

            p = paired_permutation_pvalue(
                selected,
                background,
                n_perm=N_PERM,
                rng=RNG,
            )

            records.append(
                {
                    "arm": arm,
                    "fraction": fraction,
                    "cutoff": cutoff,
                    "n_proteins": len(g),

                    "mean_background":
                        np.mean(background),

                    "mean_selected":
                        np.mean(selected),

                    "mean_difference":
                        np.mean(diff),

                    "median_difference":
                        np.median(diff),

                    "mean_enrichment":
                        np.mean(
                            g["enrichment"]
                        ),

                    "median_enrichment":
                        np.median(
                            g["enrichment"]
                        ),

                    "p_permutation": p,
                }
            )

stats = pd.DataFrame(records)

# ------------------------------------------------------------
# Multiple-testing correction
#
# 3 arms × 6 fractions × 5 cutoffs = 90 tests
# ------------------------------------------------------------

stats["q_bh"] = bh_fdr(
    stats["p_permutation"]
)

stats["p_bonferroni"] = (
    stats["p_permutation"]
    * len(stats)
)

stats["p_bonferroni"] = np.minimum(
    stats["p_bonferroni"],
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
# Print key results
# ------------------------------------------------------------

print("\n" + "=" * 110)
print("KEY RESULTS")
print("=" * 110)

for arm in ARMS:

    print("\n" + "-" * 105)
    print(arm)
    print("-" * 105)

    s = stats[
        (stats["arm"] == arm)
        & (
            stats["fraction"].isin(
                [0.01, 0.05, 0.10, 0.20]
            )
        )
        & (
            stats["cutoff"].isin(
                [4, 6, 8, 10]
            )
        )
    ].copy()

    print(
        s[
            [
                "fraction",
                "cutoff",
                "n_proteins",
                "mean_background",
                "mean_selected",
                "mean_enrichment",
                "p_permutation",
                "q_bh",
                "p_bonferroni",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

# ------------------------------------------------------------
# Count significant tests
# ------------------------------------------------------------

print("\n" + "=" * 110)
print("SIGNIFICANCE COUNTS")
print("=" * 110)

for arm in ARMS:

    s = stats[
        stats["arm"] == arm
    ]

    print(
        f"{arm}: "
        f"{s['bh_significant'].sum()}/"
        f"{len(s)} BH-FDR significant; "
        f"{s['bonferroni_significant'].sum()}/"
        f"{len(s)} Bonferroni significant"
    )

# ------------------------------------------------------------
# Best enrichment per arm
# ------------------------------------------------------------

print("\n" + "=" * 110)
print("STRONGEST ENRICHMENT RESULTS")
print("=" * 110)

for arm in ARMS:

    s = stats[
        stats["arm"] == arm
    ].sort_values(
        "mean_enrichment",
        ascending=False,
    )

    print(
        "\n",
        arm,
    )

    print(
        s.head(5)[
            [
                "fraction",
                "cutoff",
                "mean_enrichment",
                "mean_selected",
                "mean_background",
                "q_bh",
                "p_bonferroni",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

# ------------------------------------------------------------
# Save compact summary
# ------------------------------------------------------------

summary = (
    stats[
        [
            "arm",
            "fraction",
            "cutoff",
            "n_proteins",
            "mean_background",
            "mean_selected",
            "mean_difference",
            "mean_enrichment",
            "median_enrichment",
            "p_permutation",
            "q_bh",
            "p_bonferroni",
            "bh_significant",
            "bonferroni_significant",
        ]
    ]
    .sort_values(
        [
            "arm",
            "fraction",
            "cutoff",
        ]
    )
)

summary.to_csv(
    SUMMARY,
    index=False,
)

print("\nSaved:")
print(OUT)
print(SUMMARY)
