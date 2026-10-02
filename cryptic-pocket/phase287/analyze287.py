"""Phase 287: does a global (long-range) self-attention block after the EGCL stack improve CryptoBench
classification/localization, trained from scratch (650M, 8 seeds/arm)? Both arms rerun fresh in the same
kernel (base = h128L4, no global attention; global_attn = h128L4 + one global self-attention block),
identical data/loss/optimiser/early-stopping. Primary, pre-specified metric: validation PR-AUC (the metric
every other CryptoBench lever in this project reports first), paired by seed. Secondary: test PR/ROC/MCC
and DCC4/DCC8/mean-center-error.

usage: python3 analyze287.py <phase287_base_preds.pt> <phase287_global_attn_preds.pt> <out_prefix>
"""
import sys
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

base_path, ga_path, OUT = sys.argv[1:4]
P = {"base": torch.load(base_path, map_location="cpu", weights_only=False),
     "global_attn": torch.load(ga_path, map_location="cpu", weights_only=False)}
kt = sorted(P["base"]["seeds"][0]["test"])
for n, p in P.items():
    assert sorted(p["seeds"]) == list(range(8)) and sorted(p["seeds"][0]["test"]) == kt, n


def best_thr(p, y):
    ts = np.quantile(p, np.linspace(0.5, 0.999, 200))
    return ts[int(np.argmax([matthews_corrcoef(y, p >= t) for t in ts]))]


def seed_metrics(S, val_pr_cached=None):
    pv = np.concatenate([S["val"][k]["p"] for k in sorted(S["val"])]); yv = np.concatenate([S["val"][k]["y"] for k in sorted(S["val"])])
    thr = best_thr(pv, yv)
    pt = np.concatenate([S["test"][k]["p"] for k in kt]); yt = np.concatenate([S["test"][k]["y"] for k in kt])
    e = np.array([np.linalg.norm(S["test"][k]["c"] - S["test"][k]["tc"].reshape(-1)) for k in kt])
    val_pr = val_pr_cached if val_pr_cached is not None else average_precision_score(yv, pv)
    return dict(val_pr=val_pr, pr=average_precision_score(yt, pt), roc=roc_auc_score(yt, pt),
                mcc=matthews_corrcoef(yt, pt >= thr), dcc4=float(np.mean(e <= 4)), dcc8=float(np.mean(e <= 8)),
                err=float(e.mean()))


rows = [dict(arm=n, seed=s, **seed_metrics(p["seeds"][s], p["seeds"][s].get("val_pr"))) for n, p in P.items() for s in range(8)]
df = pd.DataFrame(rows); df.to_csv(OUT + "_per_seed.csv", index=False)
print("per-seed means (8 seeds)")
print(f"{'metric':8s} " + " ".join(f"{n:>12s}" for n in P))
for m in ["val_pr", "pr", "roc", "mcc", "dcc4", "dcc8", "err"]:
    print(f"{m:8s} " + " ".join(f"{df[df.arm==n][m].mean():12.4f}" for n in P))

print("\npaired over seeds, global_attn - base  (PRIMARY = val_pr)")
rec = []
for m in ["val_pr", "pr", "roc", "mcc", "dcc4", "dcc8", "err"]:
    x = df[df.arm == "global_attn"].sort_values("seed")[m].values
    y = df[df.arm == "base"].sort_values("seed")[m].values
    d = x - y
    tp = ttest_rel(x, y).pvalue; wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    rec.append(dict(metric=m, base=y.mean(), global_attn=x.mean(), delta=d.mean(), wins=int((d > 0).sum()), t_p=tp, w_p=wp))
    flag = "  * CONFIRMED" if (tp < .05 and wp < .05) else ""
    primary = " [PRIMARY]" if m == "val_pr" else ""
    print(f"  {m:8s} {y.mean():.4f} -> {x.mean():.4f} delta {d.mean():+.4f} {int((d>0).sum())}/8  t p={tp:.3f} W p={wp:.3f}{flag}{primary}")
pd.DataFrame(rec).to_csv(OUT + "_paired.csv", index=False)

print("\n8-seed ensembles")
for n, p in P.items():
    S = dict(val={k: dict(p=np.mean([p["seeds"][s]["val"][k]["p"] for s in range(8)], 0), y=p["seeds"][0]["val"][k]["y"]) for k in p["seeds"][0]["val"]},
             test={k: dict(p=np.mean([p["seeds"][s]["test"][k]["p"] for s in range(8)], 0), y=p["seeds"][0]["test"][k]["y"],
                           c=np.mean([p["seeds"][s]["test"][k]["c"] for s in range(8)], 0), tc=p["seeds"][0]["test"][k]["tc"]) for k in kt})
    m = seed_metrics(S)
    print(f"  {n:12s} " + "  ".join(f"{k} {m[k]:.4f}" for k in ["val_pr", "pr", "roc", "mcc", "dcc4", "dcc8", "err"]))
