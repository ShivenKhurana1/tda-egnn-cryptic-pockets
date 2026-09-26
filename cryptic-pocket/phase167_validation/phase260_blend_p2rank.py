"""Phase 260 -- does blending the learned model with P2Rank improve residue-level cryptic-pocket prediction?

Data (all already on disk, nothing new fetched): Phase 232's exported per-residue probabilities (8 seeds x 5 arms, Phase 166 hard-split
test set, 185 proteins), P2Rank 2.5.1 per-residue scores for the same chains (Phase 226), and the experimental ligand-contact labels.

Pre-specified candidates (fixed BEFORE looking at results; nothing is tuned on the 185 test proteins):
  model   : the 8-seed mean probability of the best arm (geom_tda)
  p2rank  : P2Rank's residue score
  rankavg : 0.5 * within-protein rank(model) + 0.5 * within-protein rank(P2Rank)                      (no free parameters)
  stack   : logistic regression on [logit(model), P2Rank score, P2Rank z-score], fit on OTHER proteins only (5-fold GroupKFold over
            proteins, so every protein is scored by a model that never saw it)
Metric: protein-level PR-AUC and ROC-AUC (185 paired units), plus lift over each protein's exact chance PR-AUC. Paired Wilcoxon + t-test
and a bootstrap CI over proteins.

Run from cryptic-pocket/:  python3 phase167_validation/phase260_blend_p2rank.py
"""
import re, sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, wilcoxon, ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

BASE = "phase167_validation"
PROBS = "phase232/results/phase232_test_probabilities.pt"
P2 = "tools/p2rank_out"
ARM = "geom_tda"

probs = torch.load(PROBS, map_location="cpu", weights_only=False)
mapping = pd.read_csv(f"{BASE}/phase167_node_residue_mapping.csv")
mapping = mapping[mapping.mapping_status == "mapped"].copy()
mapping["key"] = list(zip(mapping.apo_pdb_id.str.lower(), mapping.apo_chain.astype(str)))
mapping["icode"] = mapping.pdb_insertion_code.fillna("").astype(str).str.strip()
mapping["resnum"] = mapping.pdb_residue_number.astype(int)
chance = pd.read_csv(f"{BASE}/phase226_p2rank_protein_metrics.csv")
chance["key"] = list(zip(chance.pdb, chance.chain.astype(str)))
chance = chance.set_index("key").ap_chance
label_re = re.compile(r"^(-?\d+)([A-Za-z]?)$")

def p2_table(pdb):
    df = pd.read_csv(f"{P2}/{pdb}.cif_residues.csv", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df["chain"] = df["chain"].astype(str).str.strip()
    lab = df["residue_label"].astype(str).str.strip().str.extract(label_re)
    df["resnum"] = pd.to_numeric(lab[0]); df["icode"] = lab[1].fillna("")
    return df

rows = []
for key, g in mapping.groupby("key"):
    pdb, chain = key
    g = g.sort_values("node_index")
    y = g.experimental_site.astype(int).values
    if y.sum() == 0 or y.sum() == len(y) or key not in chance.index:
        continue
    t = p2_table(pdb); t = t[t.chain == chain]
    j = g.merge(t[["resnum", "icode", "score", "zscore", "probability"]], on=["resnum", "icode"], how="left")
    assert not j.score.isna().any(), key
    m = np.mean([probs[(ARM, s)][key][g.node_index.values] for s in range(8)], axis=0)
    rows.append(dict(key=key, y=y, model=m, p2=j.score.values, p2z=j.zscore.values, p2p=j.probability.values))
print("proteins:", len(rows), "| residues:", sum(len(r["y"]) for r in rows))

def rk(v): return (rankdata(v) - 0.5) / len(v)
for r in rows:
    r["rankavg"] = 0.5 * rk(r["model"]) + 0.5 * rk(r["p2"])
    r["logit"] = np.log(np.clip(r["model"], 1e-6, 1 - 1e-6) / (1 - np.clip(r["model"], 1e-6, 1 - 1e-6)))

# stacking, out-of-fold by protein
X = [np.column_stack([r["logit"], r["p2"], r["p2z"]]) for r in rows]
groups = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(rows)])
Xall = np.vstack(X); yall = np.concatenate([r["y"] for r in rows])
oof = np.zeros(len(yall))
for tr, te in GroupKFold(n_splits=5).split(Xall, yall, groups):
    mu, sd = Xall[tr].mean(0), Xall[tr].std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=1000).fit((Xall[tr] - mu) / sd, yall[tr])
    oof[te] = lr.decision_function((Xall[te] - mu) / sd)
off = 0
for r in rows:
    n = len(r["y"]); r["stack"] = oof[off:off + n]; off += n

def per_protein(name):
    ap = np.array([average_precision_score(r["y"], r[name]) for r in rows])
    roc = np.array([roc_auc_score(r["y"], r[name]) for r in rows])
    ch = np.array([chance[r["key"]] for r in rows])
    return ap, roc, ch
res = {n: per_protein(n) for n in ["model", "p2", "rankavg", "stack"]}
ch = res["model"][2]
print(f"\n{'candidate':10s} {'PR-AUC':>8s} {'lift':>6s} {'ROC-AUC':>8s}   (mean over 185 proteins; chance PR-AUC {ch.mean():.3f})")
for n, (ap, roc, _) in res.items():
    print(f"{n:10s} {ap.mean():8.4f} {(ap/ch).mean():6.2f} {roc.mean():8.4f}")

rng = np.random.default_rng(260); idx = rng.integers(0, len(rows), size=(10000, len(rows)))
print("\nPaired comparisons over proteins (a - b)")
out = []
for a, b in [("rankavg", "model"), ("rankavg", "p2"), ("stack", "model"), ("stack", "p2"), ("model", "p2")]:
    for mi, mname in ((0, "PR-AUC"), (1, "ROC-AUC")):
        x, z = res[a][mi], res[b][mi]; d = x - z
        bs = d[idx].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append(dict(a=a, b=b, metric=mname, delta=d.mean(), ci_lo=lo, ci_hi=hi, frac_a_wins=(d > 0).mean(),
                        t_p=ttest_rel(x, z).pvalue, wilcoxon_p=wilcoxon(d).pvalue))
        print(f"  {a:8s} - {b:6s} {mname:8s} {x.mean():.4f} vs {z.mean():.4f}  delta {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  wins {100*(d>0).mean():.0f}%  t p={ttest_rel(x,z).pvalue:.1e}  W p={wilcoxon(d).pvalue:.1e}")
pd.DataFrame(out).to_csv(f"{BASE}/phase260_blend_stats.csv", index=False)

# pooled-residue view (all 185 proteins concatenated) for context
print("\nPooled over all residues:")
for n in ["model", "p2", "rankavg", "stack"]:
    v = np.concatenate([r[n] for r in rows])
    print(f"  {n:8s} PR-AUC {average_precision_score(yall, v):.4f}   ROC-AUC {roc_auc_score(yall, v):.4f}")
