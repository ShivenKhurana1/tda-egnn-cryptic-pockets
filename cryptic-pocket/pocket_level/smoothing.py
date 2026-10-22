"""Neighbour smoothing of residue probabilities, p' = (1-a) p + a * mean(p over residues whose CA is within R A),
on the OFFICIAL all-site CryptoBench labels. (a, R) chosen on the 60 validation chains by pooled PR-AUC, frozen for
the 182 test chains. Reports pooled ROC/PR/MCC (val-tuned threshold) and per-protein AUC/AP paired over chains."""
import json, itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from common import chains, sites, HERE

def ally(ch):
    y = np.zeros(len(ch["p"]), int)
    for s in sites(ch): y[list(s)] = 1
    return y

def smooth(ch, a, R):
    if a == 0: return ch["p"]
    nb = cKDTree(ch["xyz"]).query_ball_point(ch["xyz"], R)
    m = np.array([ch["p"][j].mean() for j in nb])
    return (1 - a) * ch["p"] + a * m

V, T = chains("val"), chains("test")
for C in (V, T):
    for ch in C.values(): ch["ya"] = ally(ch)
def pooled(C, a, R):
    return np.concatenate([smooth(c, a, R) for c in C.values()]), np.concatenate([c["ya"] for c in C.values()])
best = max(((average_precision_score(*pooled(V, a, R)[::-1]), a, R) for a, R in itertools.product((0.25, 0.5, 0.75), (6.0, 8.0, 10.0, 12.0))))
_, a, R = best
print(f"chosen on val: a={a}, R={R} A (val PR {best[0]:.4f} vs unsmoothed {average_precision_score(*pooled(V,0,0)[::-1]):.4f})")
res = {}
for name, (aa, RR) in (("raw", (0, 0)), ("smoothed", (a, R))):
    pv, yv = pooled(V, aa, RR); pt, yt = pooled(T, aa, RR)
    ts = np.quantile(pv, np.linspace(.5, .999, 200)); thr = ts[np.argmax([matthews_corrcoef(yv, pv >= t) for t in ts])]
    keys = [k for k in T if 0 < T[k]["ya"].sum() < len(T[k]["ya"])]
    pp = {k: smooth(T[k], aa, RR) for k in keys}
    res[name] = dict(roc=roc_auc_score(yt, pt), pr=average_precision_score(yt, pt), mcc=matthews_corrcoef(yt, pt >= thr),
                     auc_pp=[roc_auc_score(T[k]["ya"], pp[k]) for k in keys], ap_pp=[average_precision_score(T[k]["ya"], pp[k]) for k in keys])
    r = res[name]; print(f"{name:9s} pooled ROC {r['roc']:.4f} PR {r['pr']:.4f} MCC {r['mcc']:.4f} | per-protein AUC {np.mean(r['auc_pp']):.4f} AP {np.mean(r['ap_pp']):.4f}")
for met in ("auc_pp", "ap_pp"):
    d = np.array(res["smoothed"][met]) - np.array(res["raw"][met])
    print(f"  {met}: delta {d.mean():+.4f}, {int((d>0).sum())}/{len(d)} chains up, Wilcoxon p={wilcoxon(d).pvalue:.3g}")
json.dump(dict(a=a, R=R, **{n: {k: (float(v) if not isinstance(v, list) else float(np.mean(v))) for k, v in r.items()} for n, r in res.items()}),
          open(HERE / "smoothing_test.json", "w"), indent=1)
