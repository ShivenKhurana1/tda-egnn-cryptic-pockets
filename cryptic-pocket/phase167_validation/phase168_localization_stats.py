import pandas as pd
import numpy as np

INPUT = "phase167_validation/phase168A_corrected_protein_metrics.csv"
OUT = "phase167_validation/phase168_localization_paired_stats.csv"

df = pd.read_csv(INPUT)

# Put each protein on one row
id_cols = ["apo_pdb_id", "apo_chain"]

metrics = ["center_error", "DCC4", "DCC6", "DCC8", "DCC10"]

wide = df.pivot_table(
    index=id_cols,
    columns="arm",
    values=metrics
)

# Flatten MultiIndex columns
wide.columns = [f"{metric}_{arm}" for metric, arm in wide.columns]
wide = wide.reset_index()

print("Proteins:", len(wide))
print("Columns:", list(wide.columns))


def bootstrap_ci(x, n_boot=10000, seed=168):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)

    means = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        means[i] = sample.mean()

    return (
        float(np.mean(x)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5))
    )


def sign_flip_pvalue(x, n_perm=10000, seed=168):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)

    observed = abs(x.mean())

    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(n_perm, len(x))
    )

    perm_means = np.abs((signs * x).mean(axis=1))

    # +1 correction
    p = (np.sum(perm_means >= observed) + 1) / (n_perm + 1)

    return float(p)


comparisons = [
    ("full_tda", "esm_structure", "TDA - ESM+structure"),
    ("full_model", "full_tda", "ANM - TDA"),
]

results = []

for arm_a, arm_b, comparison_name in comparisons:

    print("\n" + "=" * 70)
    print(comparison_name)
    print("=" * 70)

    for metric in metrics:

        a = wide[f"{metric}_{arm_a}"].to_numpy(float)
        b = wide[f"{metric}_{arm_b}"].to_numpy(float)

        diff = a - b

        mean_diff, ci_low, ci_high = bootstrap_ci(diff)
        median_diff = float(np.median(diff))
        sd_diff = float(np.std(diff, ddof=1))

        p = sign_flip_pvalue(diff)

        n_improved = int(np.sum(diff > 0))
        n_worsened = int(np.sum(diff < 0))
        n_equal = int(np.sum(diff == 0))

        print(f"\n{metric}")
        print(f"  mean difference:   {mean_diff:+.6f}")
        print(f"  median difference: {median_diff:+.6f}")
        print(f"  SD:                {sd_diff:.6f}")
        print(f"  95% bootstrap CI:  [{ci_low:+.6f}, {ci_high:+.6f}]")
        print(f"  permutation p:     {p:.6g}")
        print(f"  improved:          {n_improved}/{len(diff)}")
        print(f"  worsened:          {n_worsened}/{len(diff)}")
        print(f"  unchanged:         {n_equal}/{len(diff)}")

        results.append({
            "comparison": comparison_name,
            "metric": metric,
            "n_proteins": len(diff),
            "mean_difference": mean_diff,
            "median_difference": median_diff,
            "sd_difference": sd_diff,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "permutation_p": p,
            "n_improved": n_improved,
            "n_worsened": n_worsened,
            "n_equal": n_equal,
        })

out = pd.DataFrame(results)
out.to_csv(OUT, index=False)

print("\nSaved:")
print(OUT)
