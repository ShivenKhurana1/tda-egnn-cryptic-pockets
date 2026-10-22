"""Pocket-level scoring of the homology-clean 3B-transfer 8-seed ensemble (Phase 266c) on the CryptoBench test split,
against the OFFICIAL CryptoBench truth: every holo pairing's apo_pocket_selection (union per apo structure, as in
CryptoBench's create-annotations.py), split into distinct pockets (sites sharing >=50% of the smaller set are merged).

Predicted pockets: residues with p >= t, single-linkage clustered on CA coordinates at cutoff d, clusters smaller than
m residues dropped, ranked by summed probability; pocket centre = probability-weighted CA centroid. (t, d, m) are
chosen on the 60 validation chains only, then frozen for test.

Metrics, per test chain with n true pockets:
  DCC top-n / top-(n+2): fraction of true pockets whose CA centroid lies within 4 / 8 A of one of the top-n (n+2)
    predicted centres (P2Rank/PUResNet convention, with residue centroids instead of ligand atoms).
  Overlap top-n: a true pocket is found if a top-n predicted pocket covers >=1/3 of its residues and >=1/3 of the
    predicted pocket lies inside it.
  Reference: the model's own centre head (one centre per chain), scored by DCC against each chain's true pockets.

usage: python3 score_pockets.py
"""
import json, itertools
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from common import chains, sites, HERE

rng = np.random.RandomState(0)


def predicted_pockets(ch, t, d, m):
    idx = np.where(ch["p"] >= t)[0]
    if len(idx) == 0:
        return []
    if len(idx) == 1:
        lab = np.array([1])
    else:
        lab = fcluster(linkage(ch["xyz"][idx], "single"), d, "distance")
    out = []
    for c in np.unique(lab):
        r = idx[lab == c]
        if len(r) < m:
            continue
        w = ch["p"][r]
        out.append(dict(res=set(r.tolist()), score=float(w.sum()), centre=(ch["xyz"][r] * w[:, None]).sum(0) / w.sum()))
    return sorted(out, key=lambda q: -q["score"])


def score_chain(ch, P, S):
    n = len(S)
    tcs = [ch["xyz"][sorted(s)].mean(0) for s in S]
    r = {}
    for tag, k in (("n", n), ("n2", n + 2)):
        top = P[:k]
        for cut in (4, 8):
            r[f"dcc{cut}_top{tag}"] = np.mean([any(np.linalg.norm(q["centre"] - tc) <= cut for q in top) for tc in tcs])
    r["overlap_topn"] = np.mean([any(len(q["res"] & s) >= len(s) / 3 and len(q["res"] & s) >= len(q["res"]) / 3
                                     for q in P[:n]) for s in S])
    r["n_pred"] = len(P)
    r["head_dcc4"] = np.mean([np.linalg.norm(ch["c"] - tc) <= 4 for tc in tcs])
    r["head_dcc8"] = np.mean([np.linalg.norm(ch["c"] - tc) <= 8 for tc in tcs])
    r["head_any8"] = float(any(np.linalg.norm(ch["c"] - tc) <= 8 for tc in tcs))
    r["clu_any8_top1"] = float(bool(P) and any(np.linalg.norm(P[0]["centre"] - tc) <= 8 for tc in tcs))
    return r


def evaluate(C, t, d, m):
    rows = {}
    for k, ch in C.items():
        S = sites(ch)
        if S:
            rows[k] = score_chain(ch, predicted_pockets(ch, t, d, m), S)
    return rows


def mean(rows, key):
    return float(np.mean([r[key] for r in rows.values()]))


def boot(v, n=5000):
    v = np.asarray(v, float)
    b = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


V, T = chains("val"), chains("test")
pv = np.concatenate([c["p"] for c in V.values()])
grid = list(itertools.product(np.quantile(pv, [0.80, 0.85, 0.90, 0.93, 0.95, 0.97]), (6.0, 8.0, 10.0), (1, 3, 5)))
# selection objective on validation: overlap top-n plus DCC8 top-n (both 0..1), ties broken by fewer predicted pockets
val_scores = []
for t, d, m in grid:
    R = evaluate(V, t, d, m)
    val_scores.append((mean(R, "overlap_topn") + mean(R, "dcc8_topn"), -mean(R, "n_pred"), t, d, m))
val_scores.sort(reverse=True)
_, _, t, d, m = val_scores[0]
print(f"chosen on val: threshold {t:.4f} (val quantile {np.mean(pv < t):.2f}), link cutoff {d} A, min size {m}")

R = evaluate(T, t, d, m)
keys = ["dcc4_topn", "dcc8_topn", "dcc4_topn2", "dcc8_topn2", "overlap_topn", "head_dcc4", "head_dcc8",
        "head_any8", "clu_any8_top1", "n_pred"]
out = dict(params=dict(threshold=float(t), link_cutoff=d, min_size=m), n_chains=len(R),
           n_true_pockets=int(sum(len(sites(T[k])) for k in R)), metrics={})
print(f"test chains {len(R)}, true pockets {out['n_true_pockets']}")
for k in keys:
    mu = mean(R, k)
    lo, hi = boot([r[k] for r in R.values()]) if k != "n_pred" else (np.nan, np.nan)
    out["metrics"][k] = dict(mean=mu, ci=[lo, hi])
    print(f"  {k:14s} {mu:.3f}  [{lo:.3f}, {hi:.3f}]")

# paired: clustered top-1 vs centre head, chain-level "hit any true pocket within 8 A"
a = np.array([R[k]["clu_any8_top1"] for k in R]); b = np.array([R[k]["head_any8"] for k in R])
print(f"top-1 within 8 A of a true pocket: clustering {a.mean():.3f} vs centre head {b.mean():.3f} "
      f"(clustering-only wins {int(((a == 1) & (b == 0)).sum())}, head-only wins {int(((a == 0) & (b == 1)).sum())})")
out["top1_any8"] = dict(clustering=float(a.mean()), head=float(b.mean()),
                        clustering_only=int(((a == 1) & (b == 0)).sum()), head_only=int(((a == 0) & (b == 1)).sum()))
json.dump(dict(out, per_chain={k: {kk: float(vv) for kk, vv in r.items()} for k, r in R.items()}),
          open(HERE / "pocket_level_test.json", "w"), indent=1)
