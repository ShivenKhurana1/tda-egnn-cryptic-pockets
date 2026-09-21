"""Phase 233 -- experimental-site analysis of the Phase 232 arms (Phase 166 hard split, 185 proteins).

Inputs: Phase 232 exported per-residue test probabilities (8 seeds x 5 arms), the experimental ligand-contact labels
used by the manuscript (phase167_node_residue_mapping.csv), and the P2Rank per-protein table from Phase 226.
Everything is protein-level (185 units), seed-averaged per protein, paired across arms, chance-normalized.

Run from cryptic-pocket/:  python3 phase167_validation/phase233_geometry_control_hardsplit_analysis.py <p232_dir>
"""
import sys
sys.path.insert(0, "/tmp")
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, rankdata, wilcoxon, ttest_rel
from sklearn.metrics import average_precision_score, roc_auc_score

P232 = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p232"
BASE = "phase167_validation"
ARMS = ["esm_structure", "tda", "geom", "geom_tda", "geom_persistence"]

probs = torch.load(f"{P232}/phase232_test_probabilities.pt", map_location="cpu", weights_only=False)
mapping = pd.read_csv(f"{BASE}/phase167_node_residue_mapping.csv")
mapping = mapping[mapping.mapping_status == "mapped"].copy()
mapping["key"] = list(zip(mapping.apo_pdb_id.str.lower(), mapping.apo_chain.astype(str)))
labels = {k: g.sort_values("node_index") for k, g in mapping.groupby("key")}
p2 = pd.read_csv(f"{BASE}/phase226_p2rank_protein_metrics.csv")
p2["skey"] = p2.pdb + "|" + p2.chain.astype(str)
p2 = p2.set_index("skey")

rows = []
for key, g in labels.items():
    y = g.experimental_site.astype(int).values
    skey = f"{key[0]}|{key[1]}"
    if skey not in p2.index or y.sum() == 0 or y.sum() == len(y):
        continue
    nodes = g.node_index.values
    rec = dict(key=key, n=len(y), k=int(y.sum()), ap_chance=float(p2.loc[skey, "ap_chance"]),
               p2rank_ap=float(p2.loc[skey, "p2rank_score_ap"]), p2rank_roc=float(p2.loc[skey, "p2rank_score_roc"]))
    for arm in ARMS:
        aps, rocs = [], []
        for seed in range(8):
            pr = probs[(arm, seed)].get(key)
            if pr is None:
                raise KeyError((arm, seed, key))
            p = pr[nodes]
            aps.append(average_precision_score(y, p)); rocs.append(roc_auc_score(y, p))
        rec[f"{arm}_ap"], rec[f"{arm}_roc"] = float(np.mean(aps)), float(np.mean(rocs))
    rows.append(rec)
df = pd.DataFrame(rows)
print("proteins evaluated:", len(df), "of", len(labels), "(test_hard graphs in the P232 export:", len({k for k in probs[(ARMS[0], 0)]}), ")")
for c in ["p2rank"] + ARMS:
    df[f"{c}_apnorm"] = (df[f"{c}_ap"] - df.ap_chance) / (1 - df.ap_chance)
df.drop(columns="key").to_csv(f"{BASE}/phase233_protein_metrics.csv", index=False)

print("\nMean over proteins (experimental ligand-contact labels)")
print(f"{'method':18s} {'AP':>7s} {'AP-norm':>8s} {'lift':>6s} {'ROC':>7s} {'medROC':>7s}")
for c in ["p2rank"] + ARMS:
    print(f"{c:18s} {df[c+'_ap'].mean():7.4f} {df[c+'_apnorm'].mean():8.4f} {(df[c+'_ap']/df.ap_chance).mean():6.2f} "
          f"{df[c+'_roc'].mean():7.4f} {df[c+'_roc'].median():7.4f}")

COMPS = [("geom_persistence", "geom"), ("geom_tda", "geom"), ("tda", "geom"), ("geom", "esm_structure"),
         ("tda", "esm_structure"), ("geom_tda", "tda"), ("geom", "p2rank"), ("tda", "p2rank"), ("geom_tda", "p2rank"),
         ("geom_persistence", "p2rank")]
print("\nPaired protein-level comparisons (a - b)")
out = []
for a, b in COMPS:
    for m in ["ap", "apnorm", "roc"]:
        d = df[f"{a}_{m}"] - df[f"{b}_{m}"]
        w, t = wilcoxon(d), ttest_rel(df[f"{a}_{m}"], df[f"{b}_{m}"])
        out.append(dict(a=a, b=b, metric=m, mean_a=df[f"{a}_{m}"].mean(), mean_b=df[f"{b}_{m}"].mean(),
                        mean_delta=d.mean(), frac_a_wins=(d > 0).mean(), t_p=t.pvalue, wilcoxon_p=w.pvalue))
        print(f"{a:>16s} - {b:<14s} {m:7s} {df[f'{a}_{m}'].mean():.4f} vs {df[f'{b}_{m}'].mean():.4f} "
              f"delta {d.mean():+.4f} wins {100*(d>0).mean():.0f}%  t p={t.pvalue:.1e}  W p={w.pvalue:.1e}")
pd.DataFrame(out).to_csv(f"{BASE}/phase233_paired_stats.csv", index=False)


def partial_spearman(x, y, controls):
    x = rankdata(np.asarray(x, float)); y = rankdata(np.asarray(y, float))
    Z = np.column_stack([np.ones(len(x))] + [rankdata(np.asarray(c, float)) for c in controls])
    return spearmanr(x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0], y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0])


print("\nPocket size vs performance (partial Spearman | protein size; second value also | chance AP)")
for c in ["p2rank"] + ARMS:
    r1, p1 = partial_spearman(df.k, df[f"{c}_ap"], [df.n])
    r2, p2_ = partial_spearman(df.k, df[f"{c}_roc"], [df.n])
    r3, p3 = partial_spearman(df.k, df[f"{c}_ap"], [df.n, df.ap_chance])
    print(f"{c:18s} AP {r1:+.3f} (p={p1:.0e}) | ROC {r2:+.3f} (p={p2_:.0e}) | AP|size,chance {r3:+.3f} (p={p3:.0e})")
