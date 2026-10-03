import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(".")
BASE = ROOT / "phase167_validation"

INPUT = BASE / "phase183A_experimental_residue_validation.csv"

OUT_STATS = BASE / "phase184_experimental_residue_paired_stats.csv"
OUT_SUMMARY = BASE / "phase184_experimental_residue_summary.csv"

N_PERM = 10000
SEED = 184

rng = np.random.default_rng(SEED)

print("=" * 110)
print("PHASE 184 - PROTEIN-LEVEL PAIRED STATISTICAL VALIDATION")
print("=" * 110)

df = pd.read_csv(INPUT, low_memory=False)

print("\nInput:", df.shape)
print("Columns:", list(df.columns))

required = [
    "apo_pdb_id_norm",
    "apo_chain_norm",
    "arm",
    "seed",
    "probability",
    "experimental_site",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(f"Missing required columns: {missing}")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def bh_fdr(pvals):
    """
    Benjamini-Hochberg FDR correction.
    """
    pvals = np.asarray(pvals, dtype=float)

    q = np.full(len(pvals), np.nan)

    valid = np.isfinite(pvals)

    if not valid.any():
        return q

    pv = pvals[valid]
    m = len(pv)

    order = np.argsort(pv)
    ranked = pv[order]

    adjusted = ranked * m / np.arange(1, m + 1)

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(adjusted, 0, 1)

    result = np.empty(m)
    result[order] = adjusted

    q[valid] = result

    return q


def paired_permutation(x, y, n_perm=10000, rng=None):
    """
    Two-sided paired sign-flip permutation test.

    H0: paired differences are exchangeable around zero.
    Each protein remains paired between methods.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan, np.nan, 0

    diff = x - y

    observed = np.mean(diff)

    if rng is None:
        rng = np.random.default_rng()

    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(n_perm, len(diff)),
    )

    perm_means = (
        signs * diff[None, :]
    ).mean(axis=1)

    extreme = np.sum(
        np.abs(perm_means) >= abs(observed)
    )

    p = (extreme + 1) / (n_perm + 1)

    return observed, p, len(diff)


def safe_enrichment(selected, background):
    if background <= 0:
        return np.nan
    return selected / background


# ---------------------------------------------------------------------
# Build one prediction ranking per protein/arm/seed
# ---------------------------------------------------------------------

fractions = [0.01, 0.02, 0.05, 0.10, 0.20]

rows = []

print("\nBuilding per-protein/per-seed experimental-site metrics...")

for (arm, seed, pdb, chain), g in df.groupby(
    ["arm", "seed", "apo_pdb_id_norm", "apo_chain_norm"],
    sort=False,
):

    g = g.sort_values(
        "probability",
        ascending=False,
    ).copy()

    y = g["experimental_site"].to_numpy(dtype=int)

    n = len(y)

    total_positive = int(y.sum())

    if total_positive == 0:
        continue

    background = y.mean()

    # Full ranking metrics are calculated without seed averaging.
    # These will subsequently be averaged over the 8 seeds.
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    if len(np.unique(y)) == 2:
        pr_auc = average_precision_score(
            y,
            g["probability"].to_numpy(dtype=float),
        )

        roc_auc = roc_auc_score(
            y,
            g["probability"].to_numpy(dtype=float),
        )
    else:
        pr_auc = np.nan
        roc_auc = np.nan

    base = {
        "arm": arm,
        "seed": seed,
        "apo_pdb_id": pdb,
        "apo_chain": chain,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "n_residues": n,
        "n_positive": total_positive,
        "background_positive_fraction": background,
    }

    for frac in fractions:

        k = max(
            1,
            int(np.ceil(frac * n))
        )

        selected = y[:k]

        tp = int(selected.sum())

        precision = tp / k

        recall = tp / total_positive

        selected_fraction = selected.mean()

        enrichment = safe_enrichment(
            selected_fraction,
            background,
        )

        base[f"enrichment_{int(frac*100)}pct"] = enrichment
        base[f"precision_{int(frac*100)}pct"] = precision
        base[f"recall_{int(frac*100)}pct"] = recall

    rows.append(base)

metrics_seed = pd.DataFrame(rows)

print(
    "Protein × seed rows:",
    metrics_seed.shape
)

print(
    "Unique proteins:",
    metrics_seed["apo_pdb_id"].nunique()
)

print(
    "Seeds:",
    sorted(metrics_seed["seed"].unique())
)

print(
    "Arms:",
    sorted(metrics_seed["arm"].unique())
)


# ---------------------------------------------------------------------
# Collapse seeds FIRST
# ---------------------------------------------------------------------

metric_cols = [
    "pr_auc",
    "roc_auc",
]

for frac in fractions:
    pct = int(frac * 100)

    metric_cols.extend([
        f"enrichment_{pct}pct",
        f"precision_{pct}pct",
        f"recall_{pct}pct",
    ])

protein = (
    metrics_seed
    .groupby(
        ["apo_pdb_id", "apo_chain", "arm"],
        as_index=False,
    )[metric_cols]
    .mean()
)

print(
    "\nSeed-collapsed protein rows:",
    protein.shape
)

print(
    "Expected:",
    185 * 3
)


# ---------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------

pairs = [
    ("full_tda", "esm_structure"),
    ("full_model", "esm_structure"),
    ("full_model", "full_tda"),
]

stats_rows = []

print("\n" + "=" * 110)
print("PAIRED PERMUTATION TESTS")
print("=" * 110)

for arm_a, arm_b in pairs:

    a = protein[
        protein["arm"] == arm_a
    ].copy()

    b = protein[
        protein["arm"] == arm_b
    ].copy()

    keys = [
        "apo_pdb_id",
        "apo_chain",
    ]

    merged = a.merge(
        b,
        on=keys,
        suffixes=("_a", "_b"),
        how="inner",
    )

    print(
        f"\nPAIR: {arm_a} vs {arm_b}"
    )

    print(
        "Paired proteins:",
        len(merged)
    )

    for metric in metric_cols:

        x = merged[f"{metric}_a"].to_numpy(dtype=float)
        y = merged[f"{metric}_b"].to_numpy(dtype=float)

        diff, p, n = paired_permutation(
            x,
            y,
            n_perm=N_PERM,
            rng=rng,
        )

        mean_a = np.nanmean(x)
        mean_b = np.nanmean(y)

        median_a = np.nanmedian(x)
        median_b = np.nanmedian(y)

        stats_rows.append({
            "arm_a": arm_a,
            "arm_b": arm_b,
            "metric": metric,
            "n_proteins": n,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "median_a": median_a,
            "median_b": median_b,
            "mean_difference_a_minus_b": diff,
            "p_value": p,
        })

        print(
            f"{metric:24s} "
            f"{arm_a}={mean_a:.6f} "
            f"{arm_b}={mean_b:.6f} "
            f"diff={diff:+.6f} "
            f"p={p:.6g}"
        )


stats = pd.DataFrame(stats_rows)


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
    stats["mean_difference_a_minus_b"] > 0,
    "A > B",
    np.where(
        stats["mean_difference_a_minus_b"] < 0,
        "A < B",
        "equal",
    ),
)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

summary_rows = []

for arm in sorted(protein["arm"].unique()):

    sub = protein[
        protein["arm"] == arm
    ]

    row = {
        "arm": arm,
        "n_proteins": sub["apo_pdb_id"].nunique(),
    }

    for metric in metric_cols:

        row[f"{metric}_mean"] = sub[metric].mean()
        row[f"{metric}_median"] = sub[metric].median()

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

stats.to_csv(
    OUT_STATS,
    index=False,
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)


# ---------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------

print("\n" + "=" * 110)
print("MULTIPLE-TESTING SUMMARY")
print("=" * 110)

for arm_a, arm_b in pairs:

    sub = stats[
        (stats["arm_a"] == arm_a) &
        (stats["arm_b"] == arm_b)
    ]

    print(
        f"\n{arm_a} vs {arm_b}"
    )

    print(
        "BH significant:",
        int(sub["bh_significant"].sum()),
        "/",
        len(sub),
    )

    print(
        "Bonferroni significant:",
        int(sub["bonferroni_significant"].sum()),
        "/",
        len(sub),
    )

    print(
        "\nSignificant results:"
    )

    sig = sub[
        sub["bh_significant"]
    ].copy()

    if len(sig) == 0:
        print("None")
    else:
        print(
            sig[
                [
                    "metric",
                    "mean_difference_a_minus_b",
                    "p_value",
                    "bh_fdr",
                    "bonferroni",
                    "direction",
                ]
            ].to_string(index=False)
        )


print("\n" + "=" * 110)
print("FILES WRITTEN")
print("=" * 110)

print(OUT_STATS)
print(OUT_SUMMARY)

print("\n" + "=" * 110)
print("PHASE 184 COMPLETE")
print("=" * 110)
