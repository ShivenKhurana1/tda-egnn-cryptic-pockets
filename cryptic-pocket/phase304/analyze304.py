"""Phase 304 analysis (pre-registered in build_nb304.py). PRIMARY: per-protein AUC over all 760 CV chains (all_auc), stack3_studemb and
stack3_hot_studemb each vs stack3, paired by seed (fresh seeds 8-15), t AND Wilcoxon < .05 after Holm over the two. Secondary: pooled
held-out PR, per-protein AP/P@20, cryptic-enriched and rest.
usage: python3 analyze304.py <phase304_cv_preds.pt> <out_prefix>"""
import sys, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score
A = torch.load(sys.argv[1], map_location="cpu", weights_only=False); OUT = sys.argv[2]
C = json.load(open(Path(__file__).resolve().parents[1] / "phase300" / "crypticity_counts.json"))
K = sorted(next(iter(A["stack3"].values()))); SEEDS = sorted(set.intersection(*[set(v) for v in A.values()]))
ENR = [k for k in K if f"{k[0]}_{k[1]}" in C and C[f"{k[0]}_{k[1]}"]["apo"] <= 414]
REST = [k for k in K if f"{k[0]}_{k[1]}" in C and C[f"{k[0]}_{k[1]}"]["apo"] > 414]
def sub(P, ks, tag):
    ks = [k for k in ks if 0 < P[k]["y"].sum() < len(P[k]["y"])]
    return {f"{tag}_auc": np.mean([roc_auc_score(P[k]["y"], P[k]["p"]) for k in ks]), f"{tag}_ap": np.mean([average_precision_score(P[k]["y"], P[k]["p"]) for k in ks]),
            f"{tag}_p20": np.mean([P[k]["y"][np.argsort(-P[k]["p"])[:20]].mean() for k in ks])}
M = {a: {s: dict(pooled_pr=average_precision_score(np.concatenate([A[a][s][k]["y"] for k in K]), np.concatenate([A[a][s][k]["p"] for k in K])),
                 **sub(A[a][s], K, "all"), **sub(A[a][s], ENR, "enr"), **sub(A[a][s], REST, "rest")) for s in SEEDS} for a in A}
METS = list(M["stack3"][SEEDS[0]])
print(f"seeds {SEEDS}; CV {len(K)}, cryptic-enriched {len(ENR)}, rest {len(REST)}\n{'arm':20s} " + " ".join(f"{m:>9s}" for m in METS))
for a in A: print(f"{a:20s} " + " ".join(f"{np.mean([M[a][s][m] for s in SEEDS]):9.4f}" for m in METS))
ARMS = [a for a in A if a != "stack3"]; S = {}
for a in ARMS:
    S[a] = {}
    for m in METS:
        x = np.array([M[a][s][m] for s in SEEDS]); y = np.array([M["stack3"][s][m] for s in SEEDS]); d = x - y
        S[a][m] = dict(delta=float(d.mean()), up=int((d > 0).sum()), t_p=float(ttest_rel(x, y).pvalue), w_p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
for key in ("t_p", "w_p"):
    order = sorted(ARMS, key=lambda a: S[a]["all_auc"][key]); run = 0
    for i, a in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * S[a]["all_auc"][key])); S[a]["all_auc"][key + "_holm"] = run
for a in ARMS:
    v = S[a]["all_auc"]; v["verdict"] = "CONFIRMED" if v["t_p_holm"] < .05 and v["w_p_holm"] < .05 and v["delta"] > 0 else "not confirmed"
    print(f"\n[{a} vs stack3] PRIMARY per-protein AUC {v['delta']:+.4f} {v['up']}/{len(SEEDS)}  t p={v['t_p']:.3g} (Holm {v['t_p_holm']:.3g})  W p={v['w_p']:.3g} (Holm {v['w_p_holm']:.3g}) -> {v['verdict']}")
    for m in [x for x in METS if x != "all_auc"]:
        r = S[a][m]; print(f"    {m:9s} {r['delta']:+.4f}  {r['up']}/{len(SEEDS)}  t p={r['t_p']:.3g}  W p={r['w_p']:.3g}")
json.dump(S, open(OUT + "_summary.json", "w"), indent=1, default=float)
