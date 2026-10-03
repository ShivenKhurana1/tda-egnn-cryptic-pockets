import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(".")
BASE = ROOT / "phase167_validation"

INPUT = BASE / "phase185_experimental_residue_localization.csv"

OUT = BASE / "phase186_localization_paired_stats.csv"

N_PERM = 10000
SEED = 186

rng = np.random.default_rng(SEED)

print("=" * 110)
print("PHASE 186 - PROTEIN-LEVEL EXPERIMENTAL LOCALIZATION STATISTICS")
print("=" * 110)

df = pd.read_csv(INPUT, low_memory=False)

print("\nInput:", df.shape)

required = [
    "arm",
    "seed",
    "apo_pdb_id",
    "apo_chain",
    "method",
    "fraction",
    "center_error",
    "dcc4",
    "dcc6",
    "dcc8",
    "dcc10",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )


# ---------------------------------------------------------------------
# Paired sign-flip permutation
# ---------------------------------------------------------------------

def paired_permutation(
    x,
    y,
    n_perm=10000,
    rng=None,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (
        np.isfinite(x) &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan, np.nan, 0

    diff = x - y

    observed = np.mean(diff)

    signs = rng.choice(
        [-1.0, 1.0],
        size=(n_perm, len(diff)),
    )

    perm = (
        signs * diff[None, :]
    ).mean(axis=1)

    extreme = np.sum(
        np.abs(perm) >= abs(observed)
    )

    p = (
        extreme + 1
    ) / (
        n_perm + 1
    )

    return observed, p, len(diff)


def bh_fdr(pvals):
    pvals = np.asarray(
        pvals,
        dtype=float,
    )

    result = np.full(
        len(pvals),
        np.nan,
    )

    valid = np.isfinite(pvals)

    if not valid.any():
        return result

    pv = pvals[valid]
    m = len(pv)

    order = np.argsort(pv)

    ranked = pv[order]

    q = (
        ranked *
        m /
        np.arange(1, m + 1)
    )

    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.clip(q, 0, 1)

    restored = np.empty(m)

    restored[order] = q

    result[valid] = restored

    return result


# ---------------------------------------------------------------------
# Collapse 8 seeds FIRST
# ---------------------------------------------------------------------

metrics = [
    "center_error",
    "dcc4",
    "dcc6",
    "dcc8",
    "dcc10",
]

protein = (
    df.groupby(
        [
            "arm",
            "method",
            "fraction",
            "apo_pdb_id",
            "apo_chain",
        ],
        dropna=False,
        as_index=False,
    )[metrics]
    .mean()
)

print(
    "\nSeed-collapsed rows:",
    protein.shape,
)

print(
    "Unique proteins:",
    protein[
        "apo_pdb_id"
    ].nunique()
)


# ---------------------------------------------------------------------
# Define comparisons
# ---------------------------------------------------------------------

comparisons = [
    (
        "gamma10_centroid",
        None,
        "probability_centroid",
        None,
    ),

    (
        "gamma10_centroid",
        None,
        "top_centroid",
        0.01,
    ),

    (
        "gamma10_centroid",
        None,
        "top_centroid",
        0.05,
    ),

    (
        "gamma10_centroid",
        None,
        "top_centroid",
        0.10,
    ),

    (
        "gamma10_centroid",
        None,
        "top_centroid",
        0.20,
    ),
]


# ---------------------------------------------------------------------
# Run paired tests
# ---------------------------------------------------------------------

rows = []

for arm in sorted(
    protein["arm"].unique()
):

    print("\n" + "=" * 100)
    print("ARM:", arm)
    print("=" * 100)

    for (
        method_a,
        frac_a,
        method_b,
        frac_b,
    ) in comparisons:

        a = protein[
            (protein["arm"] == arm) &
            (protein["method"] == method_a)
        ].copy()

        if frac_a is not None:
            a = a[
                a["fraction"] == frac_a
            ]

        b = protein[
            (protein["arm"] == arm) &
            (protein["method"] == method_b)
        ].copy()

        if frac_b is not None:
            b = b[
                b["fraction"] == frac_b
            ]

        keys = [
            "apo_pdb_id",
            "apo_chain",
        ]

        m = a.merge(
            b,
            on=keys,
            suffixes=("_a", "_b"),
            how="inner",
        )

        label_b = method_b

        if frac_b is not None:
            label_b += f"_{int(frac_b*100)}pct"

        print(
            f"\nGamma10 vs {label_b}: "
            f"{len(m)} proteins"
        )

        for metric in metrics:

            x = m[
                f"{metric}_a"
            ].to_numpy(float)

            y = m[
                f"{metric}_b"
            ].to_numpy(float)

            diff, p, n = paired_permutation(
                x,
                y,
                n_perm=N_PERM,
                rng=rng,
            )

            rows.append({
                "arm": arm,
                "method_a": method_a,
                "fraction_a": frac_a,
                "method_b": method_b,
                "fraction_b": frac_b,
                "metric": metric,
                "n_proteins": n,
                "mean_a": np.nanmean(x),
                "mean_b": np.nanmean(y),
                "median_a": np.nanmedian(x),
                "median_b": np.nanmedian(y),
                "mean_difference_a_minus_b": diff,
                "p_value": p,
            })

            print(
                f"{metric:12s} "
                f"diff={diff:+.6f} "
                f"p={p:.6g}"
            )


stats = pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------

stats["bh_fdr"] = bh_fdr(
    stats["p_value"].to_numpy()
)

stats["bonferroni"] = np.minimum(
    stats["p_value"] * len(stats),
    1.0,
)

stats["bh_significant"] = (
    stats["bh_fdr"] < 0.05
)

stats["bonferroni_significant"] = (
    stats["bonferroni"] < 0.05
)

stats["direction"] = np.where(
    stats[
        "mean_difference_a_minus_b"
    ] < 0,
    "gamma10 better",
    np.where(
        stats[
            "mean_difference_a_minus_b"
        ] > 0,
        "comparison better",
        "equal",
    ),
)


# ---------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------

print("\n" + "=" * 110)
print("MULTIPLE-TESTING SUMMARY")
print("=" * 110)

for arm in sorted(
    stats["arm"].unique()
):

    sub = stats[
        stats["arm"] == arm
    ]

    print(
        f"\n{arm}"
    )

    print(
        "BH significant:",
        int(
            sub["bh_significant"].sum()
        ),
        "/",
        len(sub),
    )

    print(
        "Bonferroni significant:",
        int(
            sub[
                "bonferroni_significant"
            ].sum()
        ),
        "/",
        len(sub),
    )

    print("\nSignificant results:")

    sig = sub[
        sub["bh_significant"]
    ]

    if len(sig) == 0:
        print("None")
    else:
        print(
            sig[
                [
                    "method_b",
                    "fraction_b",
                    "metric",
                    "mean_difference_a_minus_b",
                    "p_value",
                    "bh_fdr",
                    "bonferroni",
                    "direction",
                ]
            ].to_string(
                index=False
            )
        )


stats.to_csv(
    OUT,
    index=False,
)

print("\n" + "=" * 110)
print("FILE WRITTEN")
print("=" * 110)

print(OUT)

print("\n" + "=" * 110)
print("PHASE 186 COMPLETE")
print("=" * 110)
