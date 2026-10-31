"""Phase 306 analysis: does the deployment-matched CV instrument agree with the known test outcomes? (pre-registered in build_nb306.py)
  (a) stack3 vs stack2 should be positive on pooled PR and per-protein AUC (test: pooled PR +0.012, ROC +0.011 significant)
  (b) stack3_hot_studemb vs stack3 should NOT be a confirmed per-protein AUC gain (test: +0.0003)
Paired by seed, t-test and Wilcoxon; per-protein metrics overall, cryptic-enriched (Phase 300) and rest.
usage: python3 analyze306.py <phase306_cv_preds.pt> <out_prefix>"""
import sys, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score
Z = torch.load(sys.argv[1], map_location="cpu", weights_only=False); A, EP = Z["preds"], Z["epochs"]; OUT = sys.argv[2]
C = json.load(open(Path(__file__).resolve().parents[1] / "phase300" / "crypticity_counts.json"))
SEEDS = sorted(set.intersection(*[set(v) for v in A.values()])); K = sorted(A["stack3"][SEEDS[0]])
ENR = [k for k in K if f"{k[0]}_{k[1]}" in C and C[f"{k[0]}_{k[1]}"]["apo"] <= 414]
REST = [k for k in K if f"{k[0]}_{k[1]}" in C and C[f"{k[0]}_{k[1]}"]["apo"] > 414]
print(f"seeds {SEEDS}; CV {len(K)}; early-stopping epochs used: " + ", ".join(f"{a} median {np.median(v):.0f} (range {min(v)}-{max(v)})" for a, v in EP.items()))
def sub(P, ks, tag):
    ks = [k for k in ks if 0 < P[k]["y"].sum() < len(P[k]["y"])]
    return {f"{tag}_auc": np.mean([roc_auc_score(P[k]["y"], P[k]["p"]) for k in ks]), f"{tag}_ap": np.mean([average_precision_score(P[k]["y"], P[k]["p"]) for k in ks])}
M = {a: {s: dict(pooled_pr=average_precision_score(np.concatenate([A[a][s][k]["y"] for k in K]), np.concatenate([A[a][s][k]["p"] for k in K])),
                 **sub(A[a][s], K, "all"), **sub(A[a][s], ENR, "enr"), **sub(A[a][s], REST, "rest")) for s in SEEDS} for a in A}
METS = list(M["stack3"][SEEDS[0]])
print(f"{'arm':20s} " + " ".join(f"{m:>9s}" for m in METS))
for a in A: print(f"{a:20s} " + " ".join(f"{np.mean([M[a][s][m] for s in SEEDS]):9.4f}" for m in METS))
S = {}
for a, b in (("stack3", "stack2"), ("stack3_hot_studemb", "stack3")):
    print(f"\n[{a} vs {b}]"); S[f"{a}-{b}"] = {}
    for m in METS:
        x = np.array([M[a][s][m] for s in SEEDS]); y = np.array([M[b][s][m] for s in SEEDS]); d = x - y
        tp = ttest_rel(x, y).pvalue; wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        S[f"{a}-{b}"][m] = dict(delta=float(d.mean()), up=int((d > 0).sum()), t_p=float(tp), w_p=float(wp))
        print(f"    {m:9s} {d.mean():+.4f}  {int((d>0).sum())}/{len(SEEDS)}  t p={tp:.3g}  W p={wp:.3g}")
a_ok = S["stack3-stack2"]["pooled_pr"]["delta"] > 0 and S["stack3-stack2"]["all_auc"]["delta"] > 0
hs = S["stack3_hot_studemb-stack3"]["all_auc"]; b_ok = not (hs["delta"] > 0 and hs["t_p"] < .05 and hs["w_p"] < .05)
print(f"\ncriterion (a) SaProt gain visible: {a_ok} | criterion (b) hot_studemb not a confirmed per-protein gain: {b_ok} -> instrument {'AGREES' if a_ok and b_ok else 'DISAGREES'} with deployment")
json.dump(dict(S, criteria=dict(a=a_ok, b=b_ok)), open(OUT + "_summary.json", "w"), indent=1, default=float)
