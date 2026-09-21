"""Phase 226 -- P2Rank 2.5.1 (pretrained, default model) on the same 185 apo chains.

Run from cryptic-pocket/ after:  tools/p2rank_2.5.1/prank predict tools/apo185.ds -o tools/p2rank_out
Residue-level: P2Rank residue score vs experimental_site, same residues as our models, same metrics
(average precision vs exact chance AP, ROC-AUC). Localization: centre of the top-ranked P2Rank pocket
lying on the target chain vs the experimental centre (phase168A_corrected_centers.csv, graph frame).
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = "phase167_validation"
P2 = "tools/p2rank_out"
ARMS = ["esm_structure", "full_model", "full_tda"]
N_PERM = 500
rng = np.random.default_rng(226)

mapping = pd.read_csv(f"{BASE}/phase167_node_residue_mapping.csv")
mapping = mapping[mapping.mapping_status == "mapped"].copy()
mapping["pdb"] = mapping.apo_pdb_id.str.lower()
mapping["chain"] = mapping.apo_chain.astype(str)
mapping["icode"] = mapping.pdb_insertion_code.fillna("").astype(str).str.strip()
mapping["resnum"] = mapping.pdb_residue_number.astype(int)

res = pd.read_csv(f"{BASE}/phase183A_experimental_residue_validation.csv",
                  usecols=["apo_pdb_id_norm", "apo_chain_norm", "node_index", "probability", "arm", "seed", "experimental_site"])
res = res.rename(columns={"apo_pdb_id_norm": "pdb", "apo_chain_norm": "chain"})

label_re = re.compile(r"^(-?\d+)([A-Za-z]?)$")


def read_p2_residues(pdb):
    df = pd.read_csv(f"{P2}/{pdb}.cif_residues.csv", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df["chain"] = df["chain"].astype(str).str.strip()
    lab = df["residue_label"].astype(str).str.strip().str.extract(label_re)
    df["resnum"] = pd.to_numeric(lab[0])
    df["icode"] = lab[1].fillna("")
    return df


def chance_ap(y):
    return float(np.mean([average_precision_score(y, rng.random(len(y))) for _ in range(N_PERM)]))


rows, unmatched = [], []
for (pdb, chain), m in mapping.groupby(["pdb", "chain"]):
    p2 = read_p2_residues(pdb)
    p2 = p2[p2.chain == chain]
    j = m.merge(p2[["resnum", "icode", "score", "probability"]].rename(columns={"probability": "p2_prob"}),
                on=["resnum", "icode"], how="left")
    unmatched.append((pdb, chain, int(j.score.isna().sum()), len(j)))
    j = j.dropna(subset=["score"])
    y = j.experimental_site.astype(int).values
    if y.sum() == 0 or y.sum() == len(y):
        continue
    ch = chance_ap(y)
    rec = dict(pdb=pdb, chain=chain, n=len(y), k=int(y.sum()), ap_chance=ch,
               p2rank_score_ap=average_precision_score(y, j.score), p2rank_score_roc=roc_auc_score(y, j.score),
               p2rank_prob_ap=average_precision_score(y, j.p2_prob), p2rank_prob_roc=roc_auc_score(y, j.p2_prob))
    # our models, restricted to exactly the residues P2Rank scored, averaged over the 8 seeds
    keep = set(j.node_index)
    for arm in ARMS:
        aps, rocs = [], []
        sub = res[(res.pdb == pdb) & (res.chain == chain) & (res.arm == arm) & res.node_index.isin(keep)]
        for _, s in sub.groupby("seed"):
            s = s.sort_values("node_index")
            aps.append(average_precision_score(s.experimental_site, s.probability))
            rocs.append(roc_auc_score(s.experimental_site, s.probability))
        rec[f"{arm}_ap"], rec[f"{arm}_roc"] = np.mean(aps), np.mean(rocs)
    rows.append(rec)

um = pd.DataFrame(unmatched, columns=["pdb", "chain", "n_unmatched", "n_nodes"])
print(f"residues without a P2Rank score: {um.n_unmatched.sum()} of {um.n_nodes.sum()}; chains with any: {(um.n_unmatched>0).sum()}")

df = pd.DataFrame(rows)
for c in ["p2rank_score", "p2rank_prob"] + ARMS:
    a = "ap" if c in ("p2rank_score", "p2rank_prob") else "ap"
    apc = f"{c}_ap"
    df[f"{c}_apnorm"] = (df[apc] - df.ap_chance) / (1 - df.ap_chance)
df.to_csv(f"{BASE}/phase226_p2rank_protein_metrics.csv", index=False)
print("proteins:", len(df), " mean chance AP %.4f" % df.ap_chance.mean())

print("\nMean over proteins")
for c in ["p2rank_score", "p2rank_prob"] + ARMS:
    print(f"{c:15s} AP {df[c+'_ap'].mean():.4f}  AP-norm {df[c+'_apnorm'].mean():.4f}  ROC {df[c+'_roc'].mean():.4f}  "
          f"median ROC {df[c+'_roc'].median():.4f}  lift {(df[c+'_ap']/df.ap_chance).mean():.2f}x")

print("\nPaired over proteins (model - P2Rank score)")
out = []
for arm in ARMS:
    for m in ["ap", "apnorm", "roc"]:
        d = df[f"{arm}_{m}"] - df[f"p2rank_score_{m}"]
        w, t = wilcoxon(d), ttest_rel(df[f"{arm}_{m}"], df[f"p2rank_score_{m}"])
        out.append(dict(model=arm, metric=m, model_mean=df[f"{arm}_{m}"].mean(), p2rank_mean=df[f"p2rank_score_{m}"].mean(),
                        mean_delta=d.mean(), frac_model_wins=(d > 0).mean(), t_p=t.pvalue, wilcoxon_p=w.pvalue))
        print(f"{arm:14s} {m:7s} {df[f'{arm}_{m}'].mean():.4f} vs {df[f'p2rank_score_{m}'].mean():.4f}  "
              f"delta {d.mean():+.4f}  model wins {100*(d>0).mean():.0f}%  t p={t.pvalue:.1e} W p={w.pvalue:.1e}")
pd.DataFrame(out).to_csv(f"{BASE}/phase226_p2rank_paired_stats.csv", index=False)

# ---------------------------- localization -----------------------------------
centers = pd.read_csv(f"{BASE}/phase168A_corrected_centers.csv")
trans = pd.read_csv(f"{BASE}/phase168A_coordinate_translations.csv")
for d in (centers, trans):
    d["pdb"] = d.apo_pdb_id.str.lower()
    d["chain"] = d.apo_chain.astype(str)
loc_models = pd.read_csv(f"{BASE}/phase185_experimental_residue_localization.csv")
loc_models = loc_models[loc_models.method == "gamma10_centroid"]

lrows = []
for _, c in centers.iterrows():
    pdb, chain = c.pdb, c.chain
    t = trans[(trans.pdb == pdb) & (trans.chain == chain)].iloc[0]
    pk = pd.read_csv(f"{P2}/{pdb}.cif_predictions.csv", skipinitialspace=True)
    pk.columns = [x.strip() for x in pk.columns]
    on_chain = []
    for _, r in pk.iterrows():
        ids = str(r["residue_ids"]).split()
        frac = np.mean([i.split("_")[0] == chain for i in ids]) if ids else 0
        if frac >= 0.5:
            on_chain.append(r)
    true = c[["true_center_x", "true_center_y", "true_center_z"]].values.astype(float)
    errs = []
    for r in on_chain[:3]:
        ctr = np.array([r["center_x"], r["center_y"], r["center_z"]], float) - t[["tx", "ty", "tz"]].values.astype(float)
        errs.append(float(np.linalg.norm(ctr - true)))
    lrows.append(dict(pdb=pdb, chain=chain, n_pockets_on_chain=len(on_chain),
                      p2_top1_err=errs[0] if errs else np.nan, p2_best3_err=min(errs) if errs else np.nan))
L = pd.DataFrame(lrows)
L.to_csv(f"{BASE}/phase226_p2rank_localization.csv", index=False)
print("\nP2Rank found >=1 pocket on target chain for %d/%d chains" % (L.p2_top1_err.notna().sum(), len(L)))


def dcc(e, th):
    return float(np.mean(e[~np.isnan(e)] <= th)) if (~np.isnan(e)).any() else np.nan


print("\nLocalization (Angstrom); DCC over chains with a P2Rank pocket, models on the same chains")
have = L.dropna(subset=["p2_top1_err"])
for name, e in [("P2Rank top-1", have.p2_top1_err.values), ("P2Rank best-of-3", have.p2_best3_err.values)]:
    print(f"{name:18s} mean {e.mean():.2f} median {np.median(e):.2f} DCC4 {dcc(e,4):.3f} DCC6 {dcc(e,6):.3f} DCC8 {dcc(e,8):.3f} DCC10 {dcc(e,10):.3f}")
for arm in ARMS:
    s = loc_models[loc_models.arm == arm].groupby(["apo_pdb_id", "apo_chain"], as_index=False).center_error.mean()
    s["pdb"], s["chain"] = s.apo_pdb_id.str.lower(), s.apo_chain.astype(str)
    s = s.merge(have[["pdb", "chain"]], on=["pdb", "chain"])
    e = s.center_error.values
    print(f"{arm:18s} mean {e.mean():.2f} median {np.median(e):.2f} DCC4 {dcc(e,4):.3f} DCC6 {dcc(e,6):.3f} DCC8 {dcc(e,8):.3f} DCC10 {dcc(e,10):.3f}")
