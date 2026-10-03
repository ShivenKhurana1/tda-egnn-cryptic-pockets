"""Like-for-like comparison on CryptoBench's strictly cryptic test proteins, using the metric protocol of the external
benchmark (pmfran4252-afk/cryptic-site-benchmark, CC BY 4.0): per-protein AUC over residues averaged across proteins,
95% CI by 5,000-sample bootstrap over proteins, top-20 content (precision@20, recall@20, success@10/20). Competitor numbers are
read straight from that repository's results/ tables (copied alongside); our model is the homology-clean 3B-transfer 8-seed ensemble."""
import json, numpy as np, torch
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel, wilcoxon
rng = np.random.RandomState(0)
L = json.load(open("../phase276/phase276a_crypticity_labels.json"))
P = torch.load("/tmp/p266c_fresh/phase266_full_preds.pt", map_location="cpu", weights_only=False)
test = P["seeds"][0]["test"]
ens = {k: np.mean([P["seeds"][s]["test"][k]["p"] for s in range(8)], 0) for k in test}
y = {k: test[k]["y"].astype(int) for k in test}
uni = {tuple(k.rsplit("_", 1)): v["uniprot"] for k, v in L.items()}
lab = {tuple(k.rsplit("_", 1)): v["label"] for k, v in L.items()}

def topk(p, yy):
    o = np.argsort(-p); r = {}
    for k in (10, 20):
        hits = int(yy[o[:k]].sum()); r[f"success@{k}"] = float(hits > 0)
        r[f"precision@{k}"] = hits / k
    r["recall@20"] = int(yy[o[:20]].sum()) / max(int(yy.sum()), 1)
    return r

ours = {}   # uniprot -> metrics, cryptic proteins only
for k in test:
    if lab.get(k) == "cryptic" and 0 < y[k].sum() < len(y[k]):
        ours[uni[k]] = dict(auc=float(roc_auc_score(y[k], ens[k])), **topk(ens[k], y[k]), key=f"{k[0]}_{k[1]}")
theirs = json.load(open("topk_cryptic.json")); p2 = {r["gid"]: r for r in json.load(open("topk_p2rank.json"))}
models = {"this work (TDA-EGNN, 3B transfer)": ours, **{m: {r["gid"]: r for r in rows} for m, rows in theirs.items()}, "P2Rank 2.5": p2}
common = sorted(set.intersection(*[set(v) for v in models.values()]))
print(f"cryptic proteins scored by every model: {len(common)} (ours in test split: {len(ours)}; benchmark: {len(theirs['ESM-based'])})")

def boot(vals, n=5000):
    vals = np.asarray(vals); b = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
rows = []
for m, d in models.items():
    r = dict(model=m, n=len(common))
    for met in ("auc", "precision@20", "recall@20", "success@10", "success@20"):
        mean, lo, hi = boot([d[g][met] for g in common]); r[met] = round(mean, 3)
        if met == "auc": r["auc_ci"] = [round(lo, 3), round(hi, 3)]
    rows.append(r)
print(f"{'model':38s} {'cryptic AUC':>11s} {'95% CI':>16s} {'P@20':>6s} {'R@20':>6s} {'hit@10':>7s} {'hit@20':>7s}")
for r in sorted(rows, key=lambda r: -r["auc"]):
    print(f"{r['model']:38s} {r['auc']:11.3f} {str(r['auc_ci']):>16s} {r['precision@20']:6.3f} {r['recall@20']:6.3f} {r['success@10']:7.2f} {r['success@20']:7.2f}")
print("\npaired over the same proteins, this work minus each baseline")
for m in models:
    if m.startswith("this work (TDA"): continue
    for met in ("auc", "precision@20", "recall@20"):
        a = np.array([ours[g][met] for g in common]); b = np.array([models[m][g][met] for g in common]); d = a - b
        bm = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(5000)]
        w = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f"  vs {m:28s} {met:13s} {d.mean():+.3f} [{np.percentile(bm,2.5):+.3f},{np.percentile(bm,97.5):+.3f}] wins {int((d>0).sum())}/{len(d)}  t p={ttest_rel(a,b).pvalue:.3f} W p={w:.3f}")
# pre-formed context
pf = [roc_auc_score(y[k], ens[k]) for k in test if lab.get(k) == "preformed" and 0 < y[k].sum() < len(y[k])]
print(f"\npre-formed (n={len(pf)}) per-protein AUC mean: ours {np.mean(pf):.3f}   [benchmark paper: P2Rank 0.835, ESM-based 0.919, interpretable 0.899, their method 0.884]")
json.dump(dict(common=common, rows=rows), open("cryptic_head2head.json", "w"), indent=1)
