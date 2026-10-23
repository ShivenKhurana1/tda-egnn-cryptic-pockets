"""Phase 298 analysis (pre-registered in build_nb298.py): all arms from one kernel. PRIMARY = pooled held-out PR-AUC over 760 CV chains,
stack2_ankh vs stack2 and stack2_saprot vs stack2, paired by seed, t AND Wilcoxon < .05 after Holm over the two. Secondary: stack4 vs
stack2 (unadjusted), and ROC, per-protein AP, P@20 for every comparison.
usage: python3 analyze298.py <phase298_cv_preds.pt> <out_prefix>"""
import sys, json
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
A = torch.load(sys.argv[1], map_location="cpu", weights_only=False); OUT = sys.argv[2]
K = sorted(A["stack2"][0]); SEEDS = sorted(set.intersection(*[set(v) for v in A.values()])); print("seeds:", SEEDS)
def metrics(P):
    p = np.concatenate([P[k]["p"] for k in K]); y = np.concatenate([P[k]["y"] for k in K])
    ks = [k for k in K if 0 < P[k]["y"].sum() < len(P[k]["y"])]
    return dict(pr=average_precision_score(y, p), roc=roc_auc_score(y, p),
                ap_pp=np.mean([average_precision_score(P[k]["y"], P[k]["p"]) for k in ks]),
                p20=np.mean([P[k]["y"][np.argsort(-P[k]["p"])[:20]].mean() for k in ks]))
df = pd.DataFrame([dict(arm=a, seed=s, **metrics(A[a][s])) for a in A for s in SEEDS]); df.to_csv(OUT + "_per_seed.csv", index=False)
METS = ["pr", "roc", "ap_pp", "p20"]; print(df.groupby("arm")[METS].mean().round(4).to_string())
S = {}
for a in ("stack2_ankh", "stack2_saprot", "stack4"):
    x = df[df.arm == a].sort_values("seed"); y = df[df.arm == "stack2"].sort_values("seed"); S[a] = r = {}
    for m in METS:
        d = x[m].values - y[m].values
        r[m] = dict(delta=float(d.mean()), up=int((d > 0).sum()), t_p=float(ttest_rel(x[m].values, y[m].values).pvalue),
                    w_p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
prim = ["stack2_ankh", "stack2_saprot"]
for key in ("t_p", "w_p"):
    order = sorted(prim, key=lambda c: S[c]["pr"][key]); run = 0
    for i, c in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * S[c]["pr"][key])); S[c]["pr"][key + "_holm"] = run
for c, r in S.items():
    v = r["pr"]; tp, wp = v.get("t_p_holm", v["t_p"]), v.get("w_p_holm", v["w_p"])
    r["verdict"] = conf = ("CONFIRMED" if tp < .05 and wp < .05 and v["delta"] > 0 else "not confirmed") + ("" if c in prim else " (secondary, unadjusted)")
    print(f"\n[{c} vs stack2] held-out PR {v['delta']:+.4f} {v['up']}/{len(SEEDS)}  t p={v['t_p']:.3g}{' (Holm %.3g)' % tp if c in prim else ''}  W p={v['w_p']:.3g}{' (Holm %.3g)' % wp if c in prim else ''}  -> {conf}")
    for m in METS[1:]:
        x = r[m]; print(f"    {m:6s} {x['delta']:+.4f}  {x['up']}/{len(SEEDS)}  t p={x['t_p']:.3g}  W p={x['w_p']:.3g}")
json.dump(S, open(OUT + "_summary.json", "w"), indent=1, default=float)
