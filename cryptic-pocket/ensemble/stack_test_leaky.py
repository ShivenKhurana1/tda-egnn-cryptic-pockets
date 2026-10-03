"""Does the structural-ensemble channel add anything beyond the model's own output? Label-blind protocol: only WHOLE entries with no ligand near the protein
('clean') are used, plus the input apo structure; no per-residue ligand filtering, so the pattern of which residues have data carries no ligand-location
information. Per-residue features = spread (SD and range) of frame-free CA-CA distances to neighbouring residues across the conformations. A small
regularised logistic stacker on [model logit, ensemble features] is FIT ON THE 60 VALIDATION CHAINS and scored on the 182 test chains (no test labels used)."""
import json, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata, wilcoxon, ttest_rel
rng = np.random.RandomState(0)
REC = {r["key"]: r for r in json.load(open("../site/data.json"))["records"]}
COORDS = json.load(open("../site/coords.json"))
cry = {k for k, v in json.load(open("../phase276/phase276a_crypticity_labels.json")).items() if v["label"] == "cryptic"}

def feats(key):
    r = REC[key]; n = r["n"]; ref = np.array(COORDS[key]["xyz"]).reshape(-1, 3)
    z = np.load(f"cache/{key}.npz"); keep = np.ones(len(z["ids"]), bool)
    nclean = int(keep.sum())
    X = np.zeros((n, 5))
    if nclean: X[:, 4] = np.log1p(((~np.isnan(z["ca"][:, :, 0])) & (~z["lig"])).sum(0))  # per-RESIDUE usable-entry count (leaky)
    if nclean == 0: X[:, 2] = 0.5; X[:, 3] = 0.5; return X, 0
    ca = z["ca"].copy(); ca[z["lig"]] = np.nan                                       # per-residue ligand filter (leaky)
    ents = np.concatenate([ref[None], ca], 0)                         # E x n x 3, reference apo first
    R = np.linalg.norm(ref[:, None] - ref[None], axis=-1)
    ii, jj = np.where(np.triu((R > 3.5) & (R < 10.0), 1))                         # neighbour pairs in the reference structure
    if len(ii) == 0: X[:, 2] = 0.5; X[:, 3] = 0.5; return X, nclean
    d = np.linalg.norm(ents[:, ii] - ents[:, jj], axis=-1)                         # E x P (NaN where unmapped)
    cnt = (~np.isnan(d)).sum(0); ok = cnt >= 2
    sd = np.full(len(ii), np.nan); rg = np.full(len(ii), np.nan)
    sd[ok] = np.nanstd(d[:, ok], 0); rg[ok] = np.nanmax(d[:, ok], 0) - np.nanmin(d[:, ok], 0)
    fsd = np.full(n, np.nan); frg = np.full(n, np.nan)
    for arr, out in ((sd, fsd), (rg, frg)):
        s = np.zeros(n); c = np.zeros(n)
        v = ~np.isnan(arr)
        np.add.at(s, ii[v], arr[v]); np.add.at(c, ii[v], 1); np.add.at(s, jj[v], arr[v]); np.add.at(c, jj[v], 1)
        out[c > 0] = s[c > 0] / c[c > 0]
    X[:, 0] = np.where(np.isnan(fsd), 0, fsd); X[:, 1] = np.where(np.isnan(frg), 0, frg)
    for col, a in ((2, fsd), (3, frg)):                                            # within-chain percentile of valid residues
        v = ~np.isnan(a); p = np.full(n, 0.5)
        if v.sum() > 5: p[v] = (rankdata(a[v]) - 0.5) / v.sum()
        X[:, col] = p
    return X, nclean

P = torch.load("/tmp/p266c_fresh/phase266_full_preds.pt", map_location="cpu", weights_only=False)
def build(split):
    out = []
    for k, r in REC.items():
        if r["split"] != split: continue
        p = np.clip(np.mean([P["seeds"][s][split][tuple(k.rsplit("_", 1))]["p"] for s in range(8)], 0), 1e-4, 1 - 1e-4)
        X, ncl = feats(k); logit = np.log(p / (1 - p))
        out.append(dict(key=k, y=np.array(r["y"]), p=p, F=np.column_stack([logit, X]), ncl=ncl, cryptic=k in cry))
    return out
val, test = build("val"), build("test")
print("LEAKY per-residue-filter diagnostic"); print(f"val chains {len(val)}, test chains {len(test)}; test chains with >=1 clean extra entry: {sum(c['ncl']>0 for c in test)}")
def stack(cols):
    Xv = np.vstack([c["F"][:, cols] for c in val]); yv = np.concatenate([c["y"] for c in val])
    sc = StandardScaler().fit(Xv); m = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(Xv), yv)
    return lambda F: m.decision_function(sc.transform(F[:, cols]))
A = stack([0]); B = stack([0, 1, 2, 3, 4, 5])
def metrics(chains, fn):
    per = []; hits = []; pooled_s = []; pooled_y = []
    for c in chains:
        s = fn(c["F"]); pooled_s.append(s); pooled_y.append(c["y"])
        if 0 < c["y"].sum() < len(c["y"]): per.append(roc_auc_score(c["y"], s))
        else: per.append(np.nan)
        o = np.argsort(-s)[:20]; hits.append(c["y"][o].sum() / 20)
    ps = np.concatenate(pooled_s); py = np.concatenate(pooled_y)
    return dict(per=np.array(per), p20=np.array(hits), roc=roc_auc_score(py, ps), pr=average_precision_score(py, ps))
def report(name, chains):
    a, b = metrics(chains, A), metrics(chains, B)
    v = ~np.isnan(a["per"]) & ~np.isnan(b["per"]); d = (b["per"] - a["per"])[v]; d20 = b["p20"] - a["p20"]
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(5000)]
    w = wilcoxon(d).pvalue if np.any(d != 0) and len(d) > 5 else float("nan")
    print(f"{name:34s} n={len(chains):3d} | per-protein AUC {np.nanmean(a['per']):.3f} -> {np.nanmean(b['per']):.3f} ({d.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}], W p={w:.3f}) | P@20 {a['p20'].mean():.3f} -> {b['p20'].mean():.3f} | pooled ROC {a['roc']:.4f} -> {b['roc']:.4f}, PR {a['pr']:.4f} -> {b['pr']:.4f}")
report("all test chains", test)
report("test chains WITH ensemble info", [c for c in test if c["ncl"] > 0])
report("test chains without", [c for c in test if c["ncl"] == 0])
report("strictly cryptic (all 16)", [c for c in test if c["cryptic"]])
report("strictly cryptic WITH ensemble", [c for c in test if c["cryptic"] and c["ncl"] > 0])
