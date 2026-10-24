"""Cryptic-enriched CV evaluation. CV chains are split by our consensus apo pocket count at the threshold fitted on the calibration sample
(<= 414 voxels near the site in apo = 'cryptic-enriched'; validated on the CryptoBench test split: 88% of truly cryptic sites, 31% precision).
Every comparison is within one kernel: Phase 296b arms vs its base, Phase 298 arms vs its stack2. Per-protein AUC / AP / P@20, paired by
seed (t-test and Wilcoxon over the 8 seed-level subset means)."""
import json
import numpy as np, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score
C = json.load(open("crypticity_counts.json")); TC = 414
def key(k): return f"{k[0]}_{k[1]}"
def subset_metrics(P, keys):
    out = {m: [] for m in ("auc", "ap", "p20")}
    for s in range(8):
        ks = [k for k in keys if 0 < P[s][k]["y"].sum() < len(P[s][k]["y"])]
        out["auc"].append(np.mean([roc_auc_score(P[s][k]["y"], P[s][k]["p"]) for k in ks]))
        out["ap"].append(np.mean([average_precision_score(P[s][k]["y"], P[s][k]["p"]) for k in ks]))
        out["p20"].append(np.mean([P[s][k]["y"][np.argsort(-P[s][k]["p"])[:20]].mean() for k in ks]))
    return {m: np.array(v) for m, v in out.items()}
res = {}
for f, ref, arms in (("/tmp/p296b_out/phase296_cv_preds.pt", "base", ["frozen650", "prott5", "frozen650_prott5"]),
                     ("/tmp/p298_out/phase298_cv_preds.pt", "stack2", ["stack2_saprot", "stack2_ankh", "stack4"])):
    A = torch.load(f, map_location="cpu", weights_only=False); K = sorted(A[ref][0])
    enr = [k for k in K if key(k) in C and C[key(k)]["apo"] <= TC]; rest = [k for k in K if key(k) in C and C[key(k)]["apo"] > TC]
    print(f"\n{f.split('/')[-1]}: cryptic-enriched {len(enr)} chains, rest {len(rest)}")
    for name, ks in (("cryptic-enriched", enr), ("rest", rest)):
        r = subset_metrics(A[ref], ks); print(f"  [{ref}] {name:16s} AUC {r['auc'].mean():.4f} AP {r['ap'].mean():.4f} P@20 {r['p20'].mean():.4f}")
        for a in arms:
            x = subset_metrics(A[a], ks); line = []
            for m in ("auc", "ap", "p20"):
                d = x[m] - r[m]; line.append(f"{m} {d.mean():+.4f} {int((d>0).sum())}/8 t{ttest_rel(x[m], r[m]).pvalue:.2g} W{wilcoxon(d).pvalue:.2g}")
                res.setdefault(f"{a}-{ref}", {}).setdefault(name, {})[m] = dict(delta=float(d.mean()), up=int((d > 0).sum()), t_p=float(ttest_rel(x[m], r[m]).pvalue), w_p=float(wilcoxon(d).pvalue))
            print(f"     {a:17s} " + " | ".join(line))
json.dump(res, open("cryptic_cv.json", "w"), indent=1)
