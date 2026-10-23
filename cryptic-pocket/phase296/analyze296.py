"""Phase 296 analysis (pre-registered in build_nb296.py). Comparators come from Phase 295 and are valid only if the base seed-0 re-run
reproduces Phase 295 exactly. PRIMARY: pooled held-out PR-AUC over the 760 CV chains, (a) prott5 vs base, (b) frozen650_prott5 vs
frozen650, paired by seed, t AND Wilcoxon < .05 after Holm over the two. Secondary: ROC, per-protein AP, P@20.

usage: python3 analyze296.py <phase296b_cv_preds.pt> <out_prefix>                                  (confirmatory: all arms in one kernel)
       python3 analyze296.py <phase295_cv_preds.pt> <phase296_cv_preds.pt> <out_prefix> --provisional  (cross-run comparators)
"""
import sys, json
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

args = [a for a in sys.argv[1:] if a != "--provisional"]; PROV = "--provisional" in sys.argv
if len(args) == 2:
    A = torch.load(args[0], map_location="cpu", weights_only=False); OUT = args[1]; K = sorted(A["base"][0])
    print("confirmatory mode: every arm from one kernel")
else:
    A295 = torch.load(args[0], map_location="cpu", weights_only=False); A296 = torch.load(args[1], map_location="cpu", weights_only=False); OUT = args[2]
    K = sorted(A295["base"][0])
    same = all(np.array_equal(A295["base"][0][k]["p"], A296["base_check"][0][k]["p"]) for k in K)
    print("base seed 0 re-run bit-identical to Phase 295:", same)
    assert same or PROV, "comparators void (pass --provisional to report anyway, flagged)"
    if not same: print("PROVISIONAL: cross-run comparators; run-to-run noise ~1e-4 held-out PR")
    A = {"base": A295["base"], "frozen650": A295["frozen650"], "prott5": A296["prott5"], "frozen650_prott5": A296["frozen650_prott5"]}
SEEDS = sorted(set.intersection(*[set(v) for v in A.values()])); print("seeds:", SEEDS)


def metrics(P):
    p = np.concatenate([P[k]["p"] for k in K]); y = np.concatenate([P[k]["y"] for k in K])
    ks = [k for k in K if 0 < P[k]["y"].sum() < len(P[k]["y"])]
    return dict(pr=average_precision_score(y, p), roc=roc_auc_score(y, p),
                ap_pp=np.mean([average_precision_score(P[k]["y"], P[k]["p"]) for k in ks]),
                p20=np.mean([P[k]["y"][np.argsort(-P[k]["p"])[:20]].mean() for k in ks]))


df = pd.DataFrame([dict(arm=a, seed=s, **metrics(A[a][s])) for a in A for s in SEEDS]); df.to_csv(OUT + "_per_seed.csv", index=False)
METS = ["pr", "roc", "ap_pp", "p20"]
print(df.groupby("arm")[METS].mean().round(4).to_string())
COMP = [("prott5", "base"), ("frozen650_prott5", "frozen650")]
S = {}
for a, b in COMP:
    x = df[df.arm == a].sort_values("seed"); y = df[df.arm == b].sort_values("seed"); S[f"{a}-{b}"] = r = {}
    for m in METS:
        d = x[m].values - y[m].values
        r[m] = dict(delta=float(d.mean()), up=int((d > 0).sum()), t_p=float(ttest_rel(x[m].values, y[m].values).pvalue),
                    w_p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
for key in ("t_p", "w_p"):
    order = sorted(S, key=lambda c: S[c]["pr"][key]); run = 0
    for i, c in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * S[c]["pr"][key])); S[c]["pr"][key + "_holm"] = run
for c, r in S.items():
    v = r["pr"]; r["verdict"] = conf = "CONFIRMED" if v["t_p_holm"] < .05 and v["w_p_holm"] < .05 and v["delta"] > 0 else "not confirmed"
    print(f"\n[{c}] PRIMARY held-out PR {v['delta']:+.4f} {v['up']}/{len(SEEDS)}  t p={v['t_p']:.3g} (Holm {v['t_p_holm']:.3g})  W p={v['w_p']:.3g} (Holm {v['w_p_holm']:.3g})  -> {conf}")
    for m in METS[1:]:
        x = r[m]; print(f"    {m:6s} {x['delta']:+.4f}  {x['up']}/{len(SEEDS)}  t p={x['t_p']:.3g}  W p={x['w_p']:.3g}")
json.dump(S, open(OUT + "_summary.json", "w"), indent=1, default=float)
