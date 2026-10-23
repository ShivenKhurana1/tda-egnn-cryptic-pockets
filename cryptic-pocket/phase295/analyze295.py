"""Phase 295 analysis (pre-registered in build_nb295.py): PRIMARY = held-out PR-AUC pooled over the 760 CV chains, each arm vs base paired
by seed, paired t-test AND Wilcoxon both < .05 after Holm over the 3 arms. Secondary: held-out ROC, per-protein AP, precision@20, and
the per-fold breakdown of the primary.

usage: python3 analyze295.py <phase295_cv_preds.pt> <out_prefix>
"""
import sys, json
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

ALL = torch.load(sys.argv[1], map_location="cpu", weights_only=False); OUT = sys.argv[2]
ARMS = ["base", "pu20", "frozen650", "frozen650_pu20"]
SEEDS = sorted(set.intersection(*[set(ALL[a]) for a in ARMS]))
K = sorted(ALL["base"][SEEDS[0]])
for a in ARMS:
    for s in SEEDS: assert sorted(ALL[a][s]) == K, (a, s)
print(f"seeds complete in every arm: {SEEDS}; held-out chains {len(K)}")


def metrics(P):
    p = np.concatenate([P[k]["p"] for k in K]); y = np.concatenate([P[k]["y"] for k in K])
    ks = [k for k in K if 0 < P[k]["y"].sum() < len(P[k]["y"])]
    r = dict(pr=average_precision_score(y, p), roc=roc_auc_score(y, p),
             ap_pp=np.mean([average_precision_score(P[k]["y"], P[k]["p"]) for k in ks]),
             p20=np.mean([P[k]["y"][np.argsort(-P[k]["p"])[:20]].mean() for k in ks]))
    for f in range(4):
        kf = [k for k in K if P[k]["fold"] == f]
        r[f"pr_fold{f}"] = average_precision_score(np.concatenate([P[k]["y"] for k in kf]), np.concatenate([P[k]["p"] for k in kf]))
    return r


df = pd.DataFrame([dict(arm=a, seed=s, **metrics(ALL[a][s])) for a in ARMS for s in SEEDS]); df.to_csv(OUT + "_per_seed.csv", index=False)
METS = ["pr", "roc", "ap_pp", "p20"] + [f"pr_fold{f}" for f in range(4)]
print(df.groupby("arm")[METS].mean().loc[ARMS].round(4).to_string())
base = df[df.arm == "base"].sort_values("seed"); S = {}
for a in ARMS[1:]:
    x = df[df.arm == a].sort_values("seed"); S[a] = {}
    for m in METS:
        d = x[m].values - base[m].values
        S[a][m] = dict(delta=float(d.mean()), up=int((d > 0).sum()), t_p=float(ttest_rel(x[m].values, base[m].values).pvalue),
                       w_p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
for key in ("t_p", "w_p"):
    order = sorted(S, key=lambda a: S[a]["pr"][key]); run = 0
    for i, a in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * S[a]["pr"][key])); S[a]["pr"][key + "_holm"] = run
n = len(SEEDS)
for a in ARMS[1:]:
    v = S[a]["pr"]; conf = "CONFIRMED" if v["t_p_holm"] < .05 and v["w_p_holm"] < .05 and v["delta"] > 0 else "not confirmed"
    S[a]["verdict"] = conf
    print(f"\n[{a}] PRIMARY held-out PR {v['delta']:+.4f} {v['up']}/{n}  t p={v['t_p']:.3g} (Holm {v['t_p_holm']:.3g})  W p={v['w_p']:.3g} (Holm {v['w_p_holm']:.3g})  -> {conf}")
    for m in METS[1:]:
        r = S[a][m]; print(f"    {m:9s} {r['delta']:+.4f}  {r['up']}/{n}  t p={r['t_p']:.3g}  W p={r['w_p']:.3g}")
json.dump(S, open(OUT + "_summary.json", "w"), indent=1, default=float)
