"""Phase 225 -- prevalence-normalized re-analysis of the pocket-size finding.

Phase 189 reports partial Spearman(pocket size, PR-AUC | protein size) = 0.756.
PR-AUC's chance level equals the positive fraction, so a larger pocket raises
PR-AUC even for a random scorer. Protein size alone does not remove that. Here
every protein's average precision (AP) is compared with its own exact
chance-level AP (mean AP over random permutations of the scores, which depends
only on the labels), and the size analysis is repeated on prevalence-free metrics.

Run from cryptic-pocket/:  python3 phase167_validation/phase225_prevalence_normalized.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, wilcoxon, ttest_rel
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = "phase167_validation"
RESIDUE_FILE = os.path.join(BASE, "phase183A_experimental_residue_validation.csv")
OUT_PROTEIN = os.path.join(BASE, "phase225_protein_normalized_metrics.csv")
OUT_CORR = os.path.join(BASE, "phase225_size_correlations.csv")
OUT_TDA = os.path.join(BASE, "phase225_tda_gain_normalized.csv")

ARMS = ["esm_structure", "full_model", "full_tda"]
N_PERM = 500
RNG_SEED = 225


def null_ap(y, rng, n_perm=N_PERM):
    """Exact-in-expectation chance AP for these labels: mean AP of random scores."""
    scores = rng.random((n_perm, len(y)))
    return float(np.mean([average_precision_score(y, s) for s in scores]))


def partial_spearman(x, y, controls):
    x = rankdata(np.asarray(x, float))
    y = rankdata(np.asarray(y, float))
    Z = np.column_stack([np.ones(len(x))] + [rankdata(np.asarray(c, float)) for c in controls])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return spearmanr(rx, ry)


print("loading", RESIDUE_FILE)
res = pd.read_csv(
    RESIDUE_FILE,
    usecols=["apo_pdb_id_norm", "apo_chain_norm", "node_index", "probability", "arm", "seed", "experimental_site"],
)
res = res.rename(columns={"apo_pdb_id_norm": "pdb", "apo_chain_norm": "chain"})

# ---- per-protein chance AP (labels only, identical across arms/seeds) -------
rng = np.random.default_rng(RNG_SEED)
labels = (
    res[res.arm == ARMS[0]][res.seed == 0]
    .sort_values(["pdb", "chain", "node_index"])
    .groupby(["pdb", "chain"])["experimental_site"]
    .apply(lambda s: s.astype(int).values)
)
chance = {k: null_ap(v, rng) for k, v in labels.items()}
print("proteins:", len(labels))

# ---- per protein x arm x seed metrics ---------------------------------------
rows = []
for (pdb, chain, arm, seed), sub in res.groupby(["pdb", "chain", "arm", "seed"]):
    y = sub["experimental_site"].astype(int).values
    p = sub["probability"].astype(float).values
    n, k = len(y), int(y.sum())
    if k == 0 or k == n:
        continue
    ap = average_precision_score(y, p)
    rows.append(dict(pdb=pdb, chain=chain, arm=arm, seed=seed, protein_size=n, pocket_size=k,
                     prevalence=k / n, ap=ap, ap_chance=chance[(pdb, chain)],
                     roc_auc=roc_auc_score(y, p)))
seedlvl = pd.DataFrame(rows)

prot = seedlvl.groupby(["pdb", "chain", "arm"], as_index=False).agg(
    protein_size=("protein_size", "first"), pocket_size=("pocket_size", "first"),
    prevalence=("prevalence", "first"), ap=("ap", "mean"), ap_chance=("ap_chance", "first"),
    roc_auc=("roc_auc", "mean"))
prot["ap_gain"] = prot["ap"] - prot["ap_chance"]                       # absolute AP above chance
prot["ap_lift"] = prot["ap"] / prot["ap_chance"]                       # fold over chance
prot["ap_norm"] = (prot["ap"] - prot["ap_chance"]) / (1 - prot["ap_chance"])  # 0 = chance, 1 = perfect
prot.to_csv(OUT_PROTEIN, index=False)
print("protein x arm rows:", len(prot))

# ---- size correlations, raw vs prevalence-free ------------------------------
metrics = ["ap", "ap_gain", "ap_lift", "ap_norm", "roc_auc"]
corr = []
print("\nSpearman / partial Spearman of pocket size with each metric")
for arm in ARMS:
    s = prot[prot.arm == arm]
    for m in metrics:
        r_raw, p_raw = spearmanr(s["pocket_size"], s[m])
        r_ps, p_ps = partial_spearman(s["pocket_size"], s[m], [s["protein_size"]])
        r_pv, p_pv = partial_spearman(s["pocket_size"], s[m], [s["protein_size"], s["ap_chance"]])
        corr.append(dict(arm=arm, metric=m, n=len(s),
                         rho_raw=r_raw, p_raw=p_raw,
                         rho_given_protein_size=r_ps, p_given_protein_size=p_ps,
                         rho_given_size_and_chance=r_pv, p_given_size_and_chance=p_pv))
        print(f"{arm:14s} {m:8s} raw {r_raw:+.3f} | |size {r_ps:+.3f} (p={p_ps:.1e}) | |size,chance {r_pv:+.3f} (p={p_pv:.1e})")
pd.DataFrame(corr).to_csv(OUT_CORR, index=False)

# ---- does the TDA gain survive normalization? (paired over proteins) --------
print("\nPaired protein-level comparisons (185 proteins, seed-averaged)")
tda = []
piv = {m: prot.pivot_table(index=["pdb", "chain"], columns="arm", values=m) for m in metrics}
for base, new in [("esm_structure", "full_tda"), ("esm_structure", "full_model"), ("full_tda", "full_model")]:
    for m in metrics:
        d = (piv[m][new] - piv[m][base]).dropna()
        w = wilcoxon(d)
        t = ttest_rel(piv[m][new].loc[d.index], piv[m][base].loc[d.index])
        tda.append(dict(base=base, new=new, metric=m, n=len(d), mean_base=piv[m][base].mean(),
                        mean_new=piv[m][new].mean(), mean_delta=d.mean(), median_delta=d.median(),
                        frac_proteins_improved=(d > 0).mean(), t_p=t.pvalue, wilcoxon_p=w.pvalue))
        print(f"{new:>10s} - {base:<13s} {m:8s} {piv[m][base].mean():.4f} -> {piv[m][new].mean():.4f} "
              f"delta {d.mean():+.4f} improved {100*(d>0).mean():.0f}%  t p={t.pvalue:.1e}  W p={w.pvalue:.1e}")
pd.DataFrame(tda).to_csv(OUT_TDA, index=False)

print("\nchance AP: mean %.4f  (mean prevalence %.4f)" % (
    prot[prot.arm == ARMS[0]]["ap_chance"].mean(), prot[prot.arm == ARMS[0]]["prevalence"].mean()))
