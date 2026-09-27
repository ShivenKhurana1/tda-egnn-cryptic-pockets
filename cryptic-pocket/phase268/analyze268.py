"""Phase 268: (#3) multi-candidate localization and (#4) P2Rank blending on the OFFICIAL CryptoBench split, using the clean-pool 3B transfer ensemble (Phase 266c).
usage: python3 analyze268.py <phase266_full_preds.pt> <phase267_p2rank_and_coords.pt> <out_prefix>
Pre-specified protocol (fixed before any test number was looked at):
  * every choice is made on the 60 VALIDATION chains only; the 182 test chains are scored once.
  * localization candidates: head = mean of the 8 seeds' predicted-centre head; sharp = probability^10-weighted centroid; cluster(top, link) = top-`top` residues linked
    at `link` A, connected components ranked by summed probability, weighted centroids, first 3; hybrid = head, then the first two cluster components >8 A from the head.
    cluster parameters (top in {0.05,0.10,0.15}, link in {6,8,10}) are chosen by validation best-of-3 DCC@8.
  * baselines: P2Rank top-1 and best-of-3 pockets lying on the target chain (>50% of pocket residues on the chain), Phase 226's definition.
  * blend: model ensemble probability, P2Rank score, rank-average (no parameters), and a logistic stack on [logit(model), P2Rank score, P2Rank z-score] FIT ON THE VALIDATION
    CHAINS then applied to test (primary), plus a 5-fold protein-grouped CV stack over test chains (secondary, no dependence on validation).
Metrics: protein-level PR-AUC / ROC-AUC (chains with both classes), pooled PR/ROC, MCC at the validation-optimal threshold; DCC@4/6/8/10 for top-1 and best-of-3."""
import sys
import numpy as np, pandas as pd, torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import rankdata, wilcoxon, ttest_rel, binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold

preds = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
aux = torch.load(sys.argv[2], map_location="cpu", weights_only=False); OUT = sys.argv[3]
rows = {"val": [], "test": []}
for sp in ("val", "test"):
    for key in sorted(preds["seeds"][0][sp]):
        if key not in aux: continue
        a = aux[key]
        if np.isnan(a["p2"]).any() or a["p2"].shape[0] != len(preds["seeds"][0][sp][key]["p"]): continue
        p = np.mean([preds["seeds"][s][sp][key]["p"] for s in range(8)], 0)
        c = np.mean([preds["seeds"][s][sp][key]["c"] for s in range(8)], 0)
        assert np.allclose(preds["seeds"][0][sp][key]["tc"].reshape(-1), a["tc"], atol=1e-3), key      # same frame as the graph blob
        pk = [q for q in a["pockets"] if q["frac_on_chain"] > 0.5]
        rows[sp].append(dict(key=key, p=p, c=c, y=a["y"], pos=a["pos"], tc=a["tc"], p2=a["p2"], pockets=pk))
print({k: len(v) for k, v in rows.items()}, "chains usable (of 60 val / 182 test)")

# ---------------- #3 localization ----------------
def sharp(x, p, g=10.0): w = p ** g; w = w / w.sum(); return [(w[:, None] * x).sum(0)]
def cluster(x, p, top, link, k=3):
    n = max(3, int(round(top * len(p)))); idx = np.argsort(-p)[:n]; xs, ps = x[idx], p[idx]
    pairs = cKDTree(xs).query_pairs(link, output_type="ndarray")
    A = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else csr_matrix((n, n))
    _, lab = connected_components(A, directed=False)
    comps = sorted(range(lab.max() + 1), key=lambda q: -ps[lab == q].sum())[:k]
    return [((ps[lab == q])[:, None] * xs[lab == q]).sum(0) / ps[lab == q].sum() for q in comps]
def hybrid(r, top, link):
    out = [r["c"]]
    for q in cluster(r["pos"], r["p"], top, link, 6):
        if len(out) >= 3: break
        if np.linalg.norm(q - r["c"]) > 8.0: out.append(q)
    return out
def errs(cands, tc): return [float(np.linalg.norm(q - tc)) for q in cands]
best = None
for top in (0.05, 0.10, 0.15):
    for link in (6.0, 8.0, 10.0):
        e = np.array([min(errs(cluster(r["pos"], r["p"], top, link), r["tc"])) for r in rows["val"]])
        score = np.mean(e <= 8.0); print(f"  val cluster top={top} link={link}: best-of-3 DCC8 {score:.3f}")
        if best is None or score > best[0]: best = (score, top, link)
_, TOP, LINK = best; print(f"chosen on validation: top={TOP} link={LINK}")

def loc_table(rs):
    rec = []
    for r in rs:
        d = {"key": r["key"], "head": errs([r["c"]], r["tc"])[0], "sharp": errs(sharp(r["pos"], r["p"]), r["tc"])[0]}
        ce = errs(cluster(r["pos"], r["p"], TOP, LINK), r["tc"]); d["cluster_top1"], d["cluster_best3"] = ce[0], min(ce)
        he = errs(hybrid(r, TOP, LINK), r["tc"]); d["hybrid_best3"] = min(he)
        pe = errs([q["center"] for q in sorted(r["pockets"], key=lambda q: q["rank"])[:3]], r["tc"]) if r["pockets"] else [np.nan]
        d["p2_top1"] = pe[0]; d["p2_best3"] = np.nanmin(pe) if r["pockets"] else np.nan
        rec.append(d)
    return pd.DataFrame(rec)
T = loc_table(rows["test"]); T.to_csv(OUT + "_localization_test.csv", index=False)
both = T.dropna(subset=["p2_top1"])
rng = np.random.default_rng(268)
def line(name, e, ref=None):
    e = np.asarray(e); bs = [np.mean(e[rng.integers(0, len(e), len(e))] <= 8) for _ in range(2000)]
    print(f"  {name:22s} n={len(e):3d} mean {e.mean():6.2f} A  median {np.median(e):6.2f}   DCC4 {np.mean(e<=4):.3f}  DCC6 {np.mean(e<=6):.3f}  DCC8 {np.mean(e<=8):.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]  DCC10 {np.mean(e<=10):.3f}")
for label, d in (("ALL TEST CHAINS", T), (f"LIKE-FOR-LIKE (P2Rank returned a pocket on the chain)", both)):
    print(f"\n{label} ({len(d)} chains)")
    for name, col in [("model head (1 guess)", "head"), ("model sharp (1 guess)", "sharp"), ("cluster top-1", "cluster_top1"), ("cluster best of 3", "cluster_best3"), ("hybrid best of 3", "hybrid_best3")]: line(name, d[col])
    if label != "ALL TEST CHAINS": line("P2Rank top-1", d.p2_top1); line("P2Rank best of 3", d.p2_best3)
print("\nPaired hit tests vs P2Rank (like-for-like chains)")
for mc, pc in [("head", "p2_top1"), ("cluster_best3", "p2_best3"), ("hybrid_best3", "p2_best3")]:
    for th in (4, 8):
        a = both[mc].values <= th; b = both[pc].values <= th; n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
        print(f"  {mc:14s} vs {pc:8s} @ {th} A: model-only {n10}, P2Rank-only {n01}, sign-test p={binomtest(n10, n10+n01, .5).pvalue if n10+n01 else 1:.3f}")

# ---------------- #4 blend ----------------
def feats(r):
    m = np.clip(r["p"], 1e-6, 1 - 1e-6); return np.column_stack([np.log(m / (1 - m)), r["p2"][:, 0], r["p2"][:, 1]])
def rk(v): return (rankdata(v) - 0.5) / len(v)
for sp in rows:
    for r in rows[sp]: r["model"] = r["p"]; r["p2rank"] = r["p2"][:, 0]; r["rankavg"] = 0.5 * rk(r["p"]) + 0.5 * rk(r["p2"][:, 0])
Xv = np.vstack([feats(r) for r in rows["val"]]); yv = np.concatenate([r["y"] for r in rows["val"]])
mu, sd = Xv.mean(0), Xv.std(0) + 1e-9
lr = LogisticRegression(C=1.0, max_iter=1000).fit((Xv - mu) / sd, yv); print("\nstack coefficients (logit model, P2Rank score, P2Rank z):", np.round(lr.coef_[0], 3))
for r in rows["test"]: r["stack_val"] = lr.decision_function((feats(r) - mu) / sd)
Xt = np.vstack([feats(r) for r in rows["test"]]); yt = np.concatenate([r["y"] for r in rows["test"]])
grp = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(rows["test"])]); oof = np.zeros(len(yt))
for tr, te in GroupKFold(5).split(Xt, yt, grp):
    m_, s_ = Xt[tr].mean(0), Xt[tr].std(0) + 1e-9; oof[te] = LogisticRegression(C=1.0, max_iter=1000).fit((Xt[tr] - m_) / s_, yt[tr]).decision_function((Xt[te] - m_) / s_)
o = 0
for r in rows["test"]: n = len(r["y"]); r["stack_cv"] = oof[o:o + n]; o += n
NAMES = ["model", "p2rank", "rankavg", "stack_val", "stack_cv"]
ok = [r for r in rows["test"] if 0 < r["y"].sum() < len(r["y"])]
def per(name, key):
    return (np.array([average_precision_score(r["y"], r[key]) for r in ok]), np.array([roc_auc_score(r["y"], r[key]) for r in ok]))
res = {n: per(n, n) for n in NAMES}
thr_v = {}
print(f"\nBLEND on {len(ok)} test chains with both classes (protein-level mean; pooled in brackets)")
allp = {n: np.concatenate([r[n] for r in rows["test"]]) for n in NAMES}
for n in NAMES:
    ap, roc = res[n]; print(f"  {n:10s} PR {ap.mean():.4f}  ROC {roc.mean():.4f}   [pooled PR {average_precision_score(yt, allp[n]):.4f}  ROC {roc_auc_score(yt, allp[n]):.4f}]")
rec = []; idx = rng.integers(0, len(ok), size=(5000, len(ok)))
print("paired over proteins (a - b)")
for a, b in [("rankavg", "model"), ("stack_val", "model"), ("stack_cv", "model"), ("stack_val", "p2rank"), ("model", "p2rank")]:
    for mi, mn in ((0, "PR"), (1, "ROC")):
        x, z = res[a][mi], res[b][mi]; dd = x - z; bs = dd[idx].mean(1)
        rec.append(dict(a=a, b=b, metric=mn, delta=dd.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), wins=(dd > 0).mean(), t_p=ttest_rel(x, z).pvalue, w_p=(wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0)))
        print(f"  {a:9s} - {b:7s} {mn:3s} delta {dd.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] wins {100*(dd>0).mean():.0f}%  t p={ttest_rel(x,z).pvalue:.1e} W p={(wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0):.1e}")
pd.DataFrame(rec).to_csv(OUT + "_blend_stats.csv", index=False)
