"""Phase 301 analysis (pre-registered in build_nb301.py). PRIMARY: per-protein AUC on the cryptic-enriched CV chains (Phase 300 consensus apo
count <= 414), seed-level means, stud3di and studemb each vs stack3, paired t AND Wilcoxon < .05 after Holm over the two. Secondary:
per-protein AP / P@20 on cryptic-enriched, the same three on the rest, pooled held-out PR on all CV chains. oracle_holo = headroom.
usage: python3 analyze301.py <phase301_cv_preds.pt> <out_prefix>"""
import sys, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score
A = torch.load(sys.argv[1], map_location="cpu", weights_only=False); OUT = sys.argv[2]
C = json.load(open(Path(__file__).resolve().parents[1] / "phase300" / "crypticity_counts.json"))
K = sorted(A["stack3"][0]); SEEDS = sorted(set.intersection(*[set(v) for v in A.values()]))
ENR = [k for k in K if f"{k[0]}_{k[1]}" in C and C[f"{k[0]}_{k[1]}"]["apo"] <= 414]
REST = [k for k in K if f"{k[0]}_{k[1]}" in C and C[f"{k[0]}_{k[1]}"]["apo"] > 414]
print(f"seeds {SEEDS}; cryptic-enriched {len(ENR)}, rest {len(REST)}, all {len(K)}")

def m_subset(P, ks):
    ks = [k for k in ks if 0 < P[k]["y"].sum() < len(P[k]["y"])]
    return dict(auc=np.mean([roc_auc_score(P[k]["y"], P[k]["p"]) for k in ks]), ap=np.mean([average_precision_score(P[k]["y"], P[k]["p"]) for k in ks]),
                p20=np.mean([P[k]["y"][np.argsort(-P[k]["p"])[:20]].mean() for k in ks]))
def pooled(P): return average_precision_score(np.concatenate([P[k]["y"] for k in K]), np.concatenate([P[k]["p"] for k in K]))

M = {a: {s: dict(**{f"enr_{k}": v for k, v in m_subset(A[a][s], ENR).items()}, **{f"rest_{k}": v for k, v in m_subset(A[a][s], REST).items()},
                 pooled_pr=pooled(A[a][s])) for s in SEEDS} for a in A}
METS = list(M["stack3"][SEEDS[0]])
print("\nmeans:"); print(f"{'arm':12s} " + " ".join(f"{m:>10s}" for m in METS))
for a in A: print(f"{a:12s} " + " ".join(f"{np.mean([M[a][s][m] for s in SEEDS]):10.4f}" for m in METS))
S = {}
for a in ("oracle_holo", "stud3di", "studemb"):
    S[a] = {}
    for m in METS:
        x = np.array([M[a][s][m] for s in SEEDS]); y = np.array([M["stack3"][s][m] for s in SEEDS]); d = x - y
        S[a][m] = dict(delta=float(d.mean()), up=int((d > 0).sum()), t_p=float(ttest_rel(x, y).pvalue), w_p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
prim = ["stud3di", "studemb"]
for key in ("t_p", "w_p"):
    order = sorted(prim, key=lambda a: S[a]["enr_auc"][key]); run = 0
    for i, a in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * S[a]["enr_auc"][key])); S[a]["enr_auc"][key + "_holm"] = run
for a in ("oracle_holo", "stud3di", "studemb"):
    v = S[a]["enr_auc"]
    if a in prim:
        v["verdict"] = "CONFIRMED" if v["t_p_holm"] < .05 and v["w_p_holm"] < .05 and v["delta"] > 0 else "not confirmed"
        head = f"PRIMARY cryptic-enriched AUC {v['delta']:+.4f} {v['up']}/{len(SEEDS)} t p={v['t_p']:.3g} (Holm {v['t_p_holm']:.3g}) W p={v['w_p']:.3g} (Holm {v['w_p_holm']:.3g}) -> {v['verdict']}"
    else:
        head = f"HEADROOM (not deployable) cryptic-enriched AUC {v['delta']:+.4f} {v['up']}/{len(SEEDS)} t p={v['t_p']:.3g} W p={v['w_p']:.3g}"
    print(f"\n[{a} vs stack3] {head}")
    for m in METS:
        if m == "enr_auc": continue
        r = S[a][m]; print(f"    {m:10s} {r['delta']:+.4f}  {r['up']}/{len(SEEDS)}  t p={r['t_p']:.3g}  W p={r['w_p']:.3g}")
json.dump(S, open(OUT + "_summary.json", "w"), indent=1, default=float)
