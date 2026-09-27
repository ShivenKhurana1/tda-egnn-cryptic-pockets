"""Phase 276b: are this project's confirmed results (geometry ~ topology; transfer beats scratch) actually about CRYPTIC pockets, or mostly about the
"pre-formed" pockets that dominate CryptoBench's positive labels? An independent benchmark (pmfran4252-afk/cryptic-site-benchmark, "A benchmark selected
for crypticity") relabels the CryptoBench test split by a strict apo/holo cavity-voxel rule and reports that pooled metrics are misleading: a published
model's pooled AUC 0.831 falls to 0.539 on the 17 truly-cryptic proteins, and removing cavity/geometry features IMPROVES cryptic AUC by 0.06 for them.
This script re-scores OUR existing prediction files (no new training) stratified by their relabeling, and reproduces their headline metric
(precision@20 / recall@20 / success@10 / success@20 on the top-ranked residues per protein) so our numbers are directly comparable to their table.

usage: python3 analyze276.py <out_prefix>
Arms scored (all already-downloaded prediction files from this project): from-scratch no-topo/topology/geometry/geometry+topo (Phase 262), leaky-pool
3B transfer topology/geometry/geometry+topo (Phase 264), homology-clean 3B transfer topology-only -- our reported headline model (Phase 266c).
Primary question (pre-specified after reading their preprint, before running this script): within the CRYPTIC stratum only, does geometry (Phase 264
"geom") beat, match, or lose to topology-only (Phase 264 "full"), paired over 8 seeds -- i.e. does their "cavity features are harmful for cryptic sites"
finding replicate on our architecture and our features."""
import json, sys
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = sys.argv[1] if len(sys.argv) > 1 else "phase276"
LABELS = json.load(open("phase276a_crypticity_labels.json"))
ARMS = {
    "scratch_no-topo": "/tmp/p262/phase262_none_preds.pt", "scratch_topology": "/tmp/p262/phase262_full_preds.pt",
    "scratch_geometry": "/tmp/p262/phase262_geom_preds.pt", "scratch_geom+topo": "/tmp/p262/phase262_geom_topo_preds.pt",
    "transfer(leaky)_topology": "/tmp/p264/phase264_full_preds.pt", "transfer(leaky)_geometry": "/tmp/p264/phase264_geom_preds.pt",
    "transfer(leaky)_geom+topo": "/tmp/p264/phase264_geom_topo_preds.pt",
    "transfer(clean)_topology[HEADLINE]": "/tmp/p266c/phase266_full_preds.pt",
}
P = {a: torch.load(f, map_location="cpu", weights_only=False) for a, f in ARMS.items()}
kt0 = sorted(P["transfer(clean)_topology[HEADLINE]"]["seeds"][0]["test"])
STRATA = {"cryptic": [], "preformed": [], "neither(unlabelled)": [], "all_labelled": [], "pooled_182": kt0}
for k in kt0:
    key = f"{k[0]}_{k[1]}"
    lab = LABELS.get(key, {}).get("label")
    STRATA["all_labelled"].append(k) if lab is not None else None
    if lab == "cryptic": STRATA["cryptic"].append(k)
    elif lab == "preformed": STRATA["preformed"].append(k)
    else: STRATA["neither(unlabelled)"].append(k)
print("stratum sizes among our 182 test chains:", {s: len(v) for s, v in STRATA.items()})


def topk_metrics(y, p, k=20):
    idx = np.argsort(-p)[:k]
    hit = y[idx]
    prec = hit.mean(); rec = hit.sum() / max(y.sum(), 1)
    return prec, rec, hit[:10].sum() > 0, hit.sum() > 0


def seed_scores(preds_seed, keys):
    keys = [k for k in keys if k in preds_seed]
    if not keys: return None
    y_all = np.concatenate([preds_seed[k]["y"] for k in keys]); p_all = np.concatenate([preds_seed[k]["p"] for k in keys])
    both = [k for k in keys if 0 < preds_seed[k]["y"].sum() < len(preds_seed[k]["y"])]
    pr = np.mean([average_precision_score(preds_seed[k]["y"], preds_seed[k]["p"]) for k in both]) if both else np.nan
    roc = np.mean([roc_auc_score(preds_seed[k]["y"], preds_seed[k]["p"]) for k in both]) if both else np.nan
    p20, r20, s10, s20 = zip(*[topk_metrics(preds_seed[k]["y"], preds_seed[k]["p"]) for k in keys if preds_seed[k]["y"].sum() > 0])
    return dict(n=len(keys), pooled_pr=average_precision_score(y_all, p_all), pooled_roc=roc_auc_score(y_all, p_all),
                protein_pr=pr, protein_roc=roc, precision_at_20=np.mean(p20), recall_at_20=np.mean(r20), success_at_10=np.mean(s10), success_at_20=np.mean(s20))


rows = []
for arm, pr in P.items():
    for seed in sorted(pr["seeds"]):
        preds_seed = pr["seeds"][seed]["test"]
        for stratum, keys in STRATA.items():
            m = seed_scores(preds_seed, keys)
            if m: rows.append(dict(arm=arm, seed=seed, stratum=stratum, **m))
df = pd.DataFrame(rows); df.to_csv(OUT + "_per_seed.csv", index=False)

print(f"\n{'arm':38s} {'stratum':20s} {'n':>3s} {'pooledPR':>8s} {'pooledROC':>9s} {'protPR':>7s} {'protROC':>8s} {'prec@20':>8s} {'rec@20':>7s} {'succ@10':>8s} {'succ@20':>8s}")
for arm in ARMS:
    for stratum in STRATA:
        s = df[(df.arm == arm) & (df.stratum == stratum)]
        if len(s) == 0: continue
        print(f"{arm:38s} {stratum:20s} {int(s.n.iloc[0]):3d} {s.pooled_pr.mean():8.4f} {s.pooled_roc.mean():9.4f} {s.protein_pr.mean():7.4f} "
              f"{s.protein_roc.mean():8.4f} {s.precision_at_20.mean():8.4f} {s.recall_at_20.mean():7.4f} {s.success_at_10.mean():8.4f} {s.success_at_20.mean():8.4f}")

print("\nPRIMARY: within the CRYPTIC stratum, geometry vs topology-only, paired over seeds (leaky-pool transfer arms, Phase 264)")
a, b = df[(df.arm == "transfer(leaky)_geometry") & (df.stratum == "cryptic")].sort_values("seed"), df[(df.arm == "transfer(leaky)_topology") & (df.stratum == "cryptic")].sort_values("seed")
for m in ["pooled_pr", "pooled_roc", "protein_pr", "protein_roc", "precision_at_20", "recall_at_20"]:
    x, y = a[m].values, b[m].values; d = x - y
    tp = ttest_rel(x, y).pvalue; wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    print(f"  geometry - topology  {m:16s} {y.mean():.4f} -> {x.mean():.4f}  delta {d.mean():+.4f}  {int((d>0).sum())}/{len(d)}  t p={tp:.1e}  W p={wp:.1e}")
print("\nSame comparison on the PREFORMED stratum (where we expect geometry to help, as a within-study control)")
a, b = df[(df.arm == "transfer(leaky)_geometry") & (df.stratum == "preformed")].sort_values("seed"), df[(df.arm == "transfer(leaky)_topology") & (df.stratum == "preformed")].sort_values("seed")
for m in ["pooled_pr", "pooled_roc", "protein_pr", "protein_roc"]:
    x, y = a[m].values, b[m].values; d = x - y
    tp = ttest_rel(x, y).pvalue; wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    print(f"  geometry - topology  {m:16s} {y.mean():.4f} -> {x.mean():.4f}  delta {d.mean():+.4f}  {int((d>0).sum())}/{len(d)}  t p={tp:.1e}  W p={wp:.1e}")

print("\nSecondary: does our HEADLINE model (homology-clean topology transfer) do better than P2Rank-comparable pooled-vs-cryptic gap the preprint reports?")
for stratum in ("pooled_182", "cryptic", "preformed"):
    s = df[(df.arm == "transfer(clean)_topology[HEADLINE]") & (df.stratum == stratum)]
    print(f"  {stratum:12s} pooled ROC {s.pooled_roc.mean():.3f}  protein PR {s.protein_pr.mean():.3f}  prec@20 {s.precision_at_20.mean():.3f}  rec@20 {s.recall_at_20.mean():.3f}")
