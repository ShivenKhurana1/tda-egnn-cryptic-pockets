"""Score Phase 264 (CryptoBench ESM2-3B TRANSFER with geometry-aware pretraining) against Phase 262 (same setup from scratch).
usage: python3 analyze264.py <dir with phase264_{full,geom,geom_topo}_preds.pt> <dir with phase262_{none,full,geom,geom_topo}_preds.pt> <out_prefix> [phase265_homology.json]
Primary pre-specified comparison: transfer geom+topo - transfer full (does geometry-aware pretraining beat the topology-only transfer of Phase 129/97?),
seed-paired over 8 seeds (paired t and Wilcoxon both < 0.05 to call confirmed). Secondary: geom vs full; transfer vs scratch; and, if the homology JSON
is given, every metric recomputed on test chains WITHOUT a >=30% identity homolog in the pretraining pool."""
import json, sys
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

d264, d262, out = sys.argv[1], sys.argv[2], sys.argv[3]
hom = json.load(open(sys.argv[4]))["cb"] if len(sys.argv) > 4 else None
LOAD = {"T-full": (d264, "full"), "T-geom": (d264, "geom"), "T-geom+topo": (d264, "geom_topo"),
        "S-no-topo": (d262, "none"), "S-full": (d262, "full"), "S-geom": (d262, "geom"), "S-geom+topo": (d262, "geom_topo")}
P = {a: torch.load(f"{d}/phase{264 if d == d264 else 262}_{f}_preds.pt", map_location="cpu", weights_only=False) for a, (d, f) in LOAD.items()}
def cat(pp, keys): return np.concatenate([pp[k]["p"] for k in keys]), np.concatenate([pp[k]["y"] for k in keys])
def dcc(pp, keys, th=4.0): return float(np.mean([np.linalg.norm(pp[k]["c"] - pp[k]["tc"].reshape(-1)) <= th for k in keys]))
def best_thr(p, y):
    ts = np.quantile(p, np.linspace(0.5, 0.999, 200)); return ts[int(np.argmax([matthews_corrcoef(y, p >= t) for t in ts]))]
def metrics(S, keys_v, keys_t):
    pv, yv = cat(S["val"], sorted(S["val"])); pt, yt = cat(S["test"], keys_t); t = best_thr(pv, yv)
    return dict(pr=average_precision_score(yt, pt), roc=roc_auc_score(yt, pt), mcc=matthews_corrcoef(yt, pt >= t), dcc4=dcc(S["test"], keys_t))
def subsets():
    kt = sorted(P["T-full"]["seeds"][0]["test"]); yield "all test chains", kt
    if hom:
        H = {(x.rsplit("_", 1)[0].lower(), x.rsplit("_", 1)[1]): v["leaked"] for x, v in hom.items()}
        assert all(k in H for k in kt), "homology JSON does not cover every test chain"
        k2 = [k for k in kt if not H[k]]
        yield "test chains with NO pool homolog", k2
rows = []
for label, keys in subsets():
    print(f"\n########## {label} ({len(keys)} chains)")
    for a, pr in P.items():
        assert sorted(pr["seeds"]) == list(range(8)), a
        for s in range(8): rows.append(dict(subset=label, arm=a, seed=s, **metrics(pr["seeds"][s], None, keys)))
    df = pd.DataFrame(rows); df = df[df.subset == label]
    for a in P:
        s = df[df.arm == a]; print(f"  {a:12s} PR {s.pr.mean():.4f}+-{s.pr.std(ddof=0):.4f}  ROC {s.roc.mean():.4f}  MCC {s.mcc.mean():.4f}  DCC4 {s.dcc4.mean():.4f}")
    print("  paired over seeds (a - b)")
    for a, b in [("T-geom+topo", "T-full"), ("T-geom", "T-full"), ("T-geom+topo", "T-geom"), ("T-full", "S-full"), ("T-geom+topo", "S-geom+topo"), ("T-geom+topo", "S-geom")]:
        for m in ["pr", "roc", "mcc", "dcc4"]:
            x = df[df.arm == a].sort_values("seed")[m].values; y = df[df.arm == b].sort_values("seed")[m].values; dd = x - y
            tp = ttest_rel(x, y).pvalue; wp = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
            print(f"    {a:12s} - {b:12s} {m:5s} {x.mean():.4f} vs {y.mean():.4f} delta {dd.mean():+.4f} {int((dd>0).sum())}/8  t p={tp:.1e} W p={wp:.1e}")
pd.DataFrame(rows).to_csv(out + "_per_seed.csv", index=False)
# ensembles
print("\n8-seed ensembles (all test chains)")
kt = sorted(P["T-full"]["seeds"][0]["test"])
for a, pr in P.items():
    E = lambda sp: {k: dict(p=np.mean([pr["seeds"][s][sp][k]["p"] for s in range(8)], 0), y=pr["seeds"][0][sp][k]["y"],
                            c=np.mean([pr["seeds"][s][sp][k]["c"] for s in range(8)], 0), tc=pr["seeds"][0][sp][k]["tc"]) for k in pr["seeds"][0][sp]}
    m = metrics(dict(val=E("val"), test=E("test")), None, kt); print(f"  {a:12s} " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))
