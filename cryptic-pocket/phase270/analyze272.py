"""Phase 272: do the protein-movement (elastic-network) features help the pretrained-3B CryptoBench transfer?
usage: python3 analyze272.py <phase272_full_preds.pt (matched control)> <phase272_full_flex_preds.pt> <phase266_full_preds.pt (Phase 266c reference)> <out_prefix>
Pre-specified BEFORE any test number was seen: primary = overall pooled test PR-AUC and ROC-AUC, flex - control, seed-paired over 8 seeds (paired t AND Wilcoxon both < 0.05
to call confirmed; n=8 Wilcoxon floor 0.0078). Secondary = pooled MCC, DCC@4/@8, and the label-defined pocket-size strata (<=8, 9-15, >=16 positive residues; the movement idea
predicts the largest help where pockets are hardest, i.e. <=8). Sanity: control vs the Phase 266c reference (same recipe, 51-wide geometry-zeroed channel) should agree."""

import sys
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
base, w1, w2, OUT = sys.argv[1:5]           # base = matched control, w1 = flex arm, w2 = Phase 266c reference
P = {n: torch.load(f, map_location="cpu", weights_only=False) for n, f in [("base", base), ("flex", w1), ("ref266c", w2)]}
kt = sorted(P["base"]["seeds"][0]["test"])
for n, p in P.items(): assert sorted(p["seeds"]) == list(range(8)) and sorted(p["seeds"][0]["test"]) == kt, n
size = {k: int(P["base"]["seeds"][0]["test"][k]["y"].sum()) for k in kt}
BINS = {"<=8": lambda s: s <= 8, "9-15": lambda s: 9 <= s <= 15, ">=16": lambda s: s >= 16, "all": lambda s: True}
def best_thr(p, y):
    ts = np.quantile(p, np.linspace(0.5, 0.999, 200)); return ts[int(np.argmax([matthews_corrcoef(y, p >= t) for t in ts]))]
def seed_metrics(S):
    out = {}
    pv = np.concatenate([S["val"][k]["p"] for k in sorted(S["val"])]); yv = np.concatenate([S["val"][k]["y"] for k in sorted(S["val"])]); thr = best_thr(pv, yv)
    for b, f in BINS.items():
        ks = [k for k in kt if f(size[k])]
        both = [k for k in ks if 0 < S["test"][k]["y"].sum() < len(S["test"][k]["y"])]
        out[f"{b}|roc"] = np.mean([roc_auc_score(S["test"][k]["y"], S["test"][k]["p"]) for k in both])
        out[f"{b}|pr"] = np.mean([average_precision_score(S["test"][k]["y"], S["test"][k]["p"]) for k in both])
        e = np.array([np.linalg.norm(S["test"][k]["c"] - S["test"][k]["tc"].reshape(-1)) for k in ks])
        out[f"{b}|dcc4"] = float(np.mean(e <= 4)); out[f"{b}|dcc8"] = float(np.mean(e <= 8)); out[f"{b}|err"] = float(e.mean())
    pt = np.concatenate([S["test"][k]["p"] for k in kt]); yt = np.concatenate([S["test"][k]["y"] for k in kt])
    out["pooled|pr"] = average_precision_score(yt, pt); out["pooled|roc"] = roc_auc_score(yt, pt); out["pooled|mcc"] = matthews_corrcoef(yt, pt >= thr)
    return out
rows = [dict(arm=n, seed=s, **seed_metrics(p["seeds"][s])) for n, p in P.items() for s in range(8)]
df = pd.DataFrame(rows); df.to_csv(OUT + "_per_seed.csv", index=False)
print("test chains per stratum:", {b: sum(f(size[k]) for k in kt) for b, f in BINS.items()})
show = ["<=8|roc", "<=8|pr", "<=8|dcc4", "<=8|dcc8", "<=8|err", "9-15|roc", "9-15|dcc4", ">=16|roc", ">=16|dcc4", "all|dcc4", "all|dcc8", "pooled|pr", "pooled|roc", "pooled|mcc"]
print("\nper-seed means (8 seeds)")
print(f"{'metric':12s} " + " ".join(f"{n:>9s}" for n in P))
for m in show: print(f"{m:12s} " + " ".join(f"{df[df.arm==n][m].mean():9.4f}" for n in P))
print("\npaired over seeds, arm - matched control (ref266c row = sanity check on the control itself)")
rec = []
for n in ("flex", "ref266c"):
    for m in show:
        x = df[df.arm == n].sort_values("seed")[m].values; y = df[df.arm == "base"].sort_values("seed")[m].values; d = x - y
        tp = ttest_rel(x, y).pvalue; wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        rec.append(dict(arm=n, metric=m, base=y.mean(), arm_mean=x.mean(), delta=d.mean(), wins=int((d > 0).sum()), t_p=tp, w_p=wp))
        flag = "  *" if (tp < .05 and wp < .05) else ""
        print(f"  {n} {m:11s} {y.mean():.4f} -> {x.mean():.4f} delta {d.mean():+.4f} {int((d>0).sum())}/8  t p={tp:.1e} W p={wp:.1e}{flag}")
pd.DataFrame(rec).to_csv(OUT + "_paired.csv", index=False)
print("\n8-seed ensembles (small-pocket stratum, all)")
for n, p in P.items():
    S = dict(val={k: dict(p=np.mean([p["seeds"][s]["val"][k]["p"] for s in range(8)], 0), y=p["seeds"][0]["val"][k]["y"]) for k in p["seeds"][0]["val"]},
             test={k: dict(p=np.mean([p["seeds"][s]["test"][k]["p"] for s in range(8)], 0), y=p["seeds"][0]["test"][k]["y"],
                           c=np.mean([p["seeds"][s]["test"][k]["c"] for s in range(8)], 0), tc=p["seeds"][0]["test"][k]["tc"]) for k in kt})
    m = seed_metrics(S); print(f"  {n:5s} " + "  ".join(f"{k} {m[k]:.4f}" for k in ["<=8|roc", "<=8|dcc4", "<=8|dcc8", "all|dcc4", "all|dcc8", "pooled|pr", "pooled|roc", "pooled|mcc"]))
