"""Phase 266: homology-clean CryptoBench transfer. usage: python3 analyze266.py <dir phase266_full_preds.pt> <dir phase264_full_preds.pt> <dir phase262_{none,full}_preds.pt> <out_prefix>
Arms: C = clean-pool transfer (Phase 266), T = leaky-pool transfer topology-only (Phase 264 full), S = from scratch topology-only (Phase 262 full), N = from scratch no-topo.
Primary: C - S (does transfer still help when the pool has no homolog of any val/test chain?), seed-paired over 8 seeds; secondary C - T (how much did homolog overlap inflate transfer?)."""
import sys
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
d266, d264, d262, out = sys.argv[1:5]
P = {"C-clean-transfer": torch.load(f"{d266}/phase266_full_preds.pt", map_location="cpu", weights_only=False),
     "T-leaky-transfer": torch.load(f"{d264}/phase264_full_preds.pt", map_location="cpu", weights_only=False),
     "S-scratch-full": torch.load(f"{d262}/phase262_full_preds.pt", map_location="cpu", weights_only=False),
     "N-scratch-no-topo": torch.load(f"{d262}/phase262_none_preds.pt", map_location="cpu", weights_only=False)}
def cat(pp, keys): return np.concatenate([pp[k]["p"] for k in keys]), np.concatenate([pp[k]["y"] for k in keys])
def dcc(pp, keys, th=4.0): return float(np.mean([np.linalg.norm(pp[k]["c"] - pp[k]["tc"].reshape(-1)) <= th for k in keys]))
def best_thr(p, y):
    ts = np.quantile(p, np.linspace(0.5, 0.999, 200)); return ts[int(np.argmax([matthews_corrcoef(y, p >= t) for t in ts]))]
def metrics(S, kt):
    pv, yv = cat(S["val"], sorted(S["val"])); pt, yt = cat(S["test"], kt); t = best_thr(pv, yv)
    return dict(pr=average_precision_score(yt, pt), roc=roc_auc_score(yt, pt), mcc=matthews_corrcoef(yt, pt >= t), dcc4=dcc(S["test"], kt))
kt = sorted(P["C-clean-transfer"]["seeds"][0]["test"])
for a, pr in P.items(): assert sorted(pr["seeds"]) == list(range(8)) and sorted(pr["seeds"][0]["test"]) == kt, a
rows = [dict(arm=a, seed=s, **metrics(pr["seeds"][s], kt)) for a, pr in P.items() for s in range(8)]
df = pd.DataFrame(rows); df.to_csv(out + "_per_seed.csv", index=False)
print(f"All {len(kt)} official test chains (all homology-clean w.r.t. the clean-transfer pool)")
for a in P:
    s = df[df.arm == a]; print(f"  {a:18s} PR {s.pr.mean():.4f}+-{s.pr.std(ddof=0):.4f}  ROC {s.roc.mean():.4f}  MCC {s.mcc.mean():.4f}  DCC4 {s.dcc4.mean():.4f}")
print("paired over seeds (a - b)")
for a, b in [("C-clean-transfer", "S-scratch-full"), ("C-clean-transfer", "N-scratch-no-topo"), ("C-clean-transfer", "T-leaky-transfer")]:
    for m in ["pr", "roc", "mcc", "dcc4"]:
        x = df[df.arm == a].sort_values("seed")[m].values; y = df[df.arm == b].sort_values("seed")[m].values; dd = x - y
        print(f"  {a:17s} - {b:18s} {m:5s} {x.mean():.4f} vs {y.mean():.4f} delta {dd.mean():+.4f} {int((dd>0).sum())}/8  t p={ttest_rel(x,y).pvalue:.1e} W p={wilcoxon(dd).pvalue:.1e}")
print("8-seed ensembles")
for a, pr in P.items():
    E = lambda sp: {k: dict(p=np.mean([pr["seeds"][s][sp][k]["p"] for s in range(8)], 0), y=pr["seeds"][0][sp][k]["y"],
                            c=np.mean([pr["seeds"][s][sp][k]["c"] for s in range(8)], 0), tc=pr["seeds"][0][sp][k]["tc"]) for k in pr["seeds"][0][sp]}
    m = metrics(dict(val=E("val"), test=E("test")), kt); print(f"  {a:18s} " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))
