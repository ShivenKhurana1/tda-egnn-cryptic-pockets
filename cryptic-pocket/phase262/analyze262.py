"""Score Phase 262 (CryptoBench, ESM2-3B, official split, from scratch): does plain geometry help the strongest setup?
usage: python3 analyze262.py <dir holding phase262_{none,full,geom,geom_topo}_preds.pt> [out_prefix]
Per arm: 8 seeds; pooled-residue test PR-AUC / ROC-AUC, DCC@4A (predicted centre vs true centre), MCC at a threshold chosen on VALIDATION only.
Ensemble = mean of the 8 seeds' probabilities and centres. Paired tests over seeds (a - b) and a protein-bootstrap CI for the ensemble delta.
Pre-specified primary comparison: geom+topo - geom. Project rule: paired t AND Wilcoxon both < 0.05 (n=8 Wilcoxon floor 0.0078)."""
import sys
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

d = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else f"{d}/phase262"
FILES = {"no-topo": "none", "full": "full", "geom": "geom", "geom+topo": "geom_topo"}
P = {a: torch.load(f"{d}/phase262_{f}_preds.pt", map_location="cpu", weights_only=False) for a, f in FILES.items()}
def cat(pp, keys): return np.concatenate([pp[k]["p"] for k in keys]), np.concatenate([pp[k]["y"] for k in keys])
def dcc(pp, keys, th=4.0): return float(np.mean([np.linalg.norm(pp[k]["c"] - pp[k]["tc"].reshape(-1)) <= th for k in keys]))
def best_thr(p, y):
    ts = np.quantile(p, np.linspace(0.5, 0.999, 200)); return ts[int(np.argmax([matthews_corrcoef(y, p >= t) for t in ts]))]
def metrics(pv, yv, pt, yt, dccv):
    t = best_thr(pv, yv)
    return dict(pr=average_precision_score(yt, pt), roc=roc_auc_score(yt, pt), mcc=matthews_corrcoef(yt, pt >= t), dcc4=dccv)
rows = []; ens = {}
for a, pr in P.items():
    seeds = sorted(pr["seeds"]); assert seeds == list(range(8)), (a, seeds)
    kv = sorted(pr["seeds"][0]["val"]); kt = sorted(pr["seeds"][0]["test"])
    for s in seeds:
        S = pr["seeds"][s]; assert sorted(S["test"]) == kt
        pv, yv = cat(S["val"], kv); pt, yt = cat(S["test"], kt)
        rows.append(dict(arm=a, seed=s, **metrics(pv, yv, pt, yt, dcc(S["test"], kt)), val_pr=S["val_pr"]))
    E = lambda split, ks: {k: dict(p=np.mean([pr["seeds"][s][split][k]["p"] for s in seeds], 0), y=pr["seeds"][0][split][k]["y"],
                                   c=np.mean([pr["seeds"][s][split][k]["c"] for s in seeds], 0), tc=pr["seeds"][0][split][k]["tc"]) for k in ks}
    ens[a] = (E("val", kv), E("test", kt))
df = pd.DataFrame(rows); df.to_csv(out + "_per_seed.csv", index=False)
print("Per-seed mean +- std (8 seeds), official CryptoBench test split")
for a in FILES: 
    s = df[df.arm == a]; print(f"  {a:10s} PR {s.pr.mean():.4f}+-{s.pr.std(ddof=0):.4f}  ROC {s.roc.mean():.4f}+-{s.roc.std(ddof=0):.4f}  MCC {s.mcc.mean():.4f}+-{s.mcc.std(ddof=0):.4f}  DCC4 {s.dcc4.mean():.4f}+-{s.dcc4.std(ddof=0):.4f}")
print("\n8-seed ENSEMBLE")
erow = {}
for a, (ev, et) in ens.items():
    kv, kt = sorted(ev), sorted(et); pv, yv = cat(ev, kv); pt, yt = cat(et, kt)
    erow[a] = metrics(pv, yv, pt, yt, dcc(et, kt)); print(f"  {a:10s} " + "  ".join(f"{k} {v:.4f}" for k, v in erow[a].items()))
print("\nPaired over seeds (a - b)")
res = []
for a, b in [("geom+topo", "geom"), ("geom", "no-topo"), ("full", "no-topo"), ("geom+topo", "full"), ("geom+topo", "no-topo")]:
    for m in ["pr", "roc", "mcc", "dcc4"]:
        x = df[df.arm == a].sort_values("seed")[m].values; y = df[df.arm == b].sort_values("seed")[m].values; dd = x - y
        tp = ttest_rel(x, y).pvalue; wp = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
        res.append(dict(a=a, b=b, metric=m, mean_a=x.mean(), mean_b=y.mean(), delta=dd.mean(), wins=int((dd > 0).sum()), t_p=tp, wilcoxon_p=wp))
        print(f"  {a:9s} - {b:8s} {m:5s} {x.mean():.4f} vs {y.mean():.4f}  delta {dd.mean():+.4f}  {int((dd>0).sum())}/8 seeds  t p={tp:.1e}  W p={wp:.1e}")
pd.DataFrame(res).to_csv(out + "_paired.csv", index=False)
# protein bootstrap of the ensemble delta (PR-AUC and DCC4), geom+topo vs geom
rng = np.random.default_rng(262); kt = sorted(ens["geom"][1])
def boot(a, b):
    A, B = ens[a][1], ens[b][1]; n = len(kt); dp, dd = [], []
    for _ in range(2000):
        ix = rng.integers(0, n, n); ks = [kt[i] for i in ix]
        pa, y = cat(A, ks); pb, _ = cat(B, ks)
        dp.append(average_precision_score(y, pa) - average_precision_score(y, pb)); dd.append(dcc(A, ks) - dcc(B, ks))
    return np.percentile(dp, [2.5, 97.5]), np.percentile(dd, [2.5, 97.5])
for a, b in [("geom+topo", "geom"), ("geom", "no-topo")]:
    cp, cd = boot(a, b); print(f"\nEnsemble {a} - {b}, protein-bootstrap 95% CI: PR [{cp[0]:+.4f},{cp[1]:+.4f}]  DCC4 [{cd[0]:+.4f},{cd[1]:+.4f}]")
