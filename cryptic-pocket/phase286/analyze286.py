"""Phase 286: does a genuinely local (spatial) attention mechanism for pocket-center prediction improve
localization over the existing chain-wide softmax? Primary, pre-specified metric: DCC@4A and DCC@8A on the
held-out TEST set, paired by seed (8 seeds) against the Phase 266c baseline (same checkpoints, same data,
only the localization readout differs -- not rerun). Secondary: mean center error. Classification metrics
(PR-AUC/ROC-AUC/MCC) were EXPECTED to be numerically identical to baseline per seed, since the classification
head's logits never read the localization readout at inference time -- this expectation was wrong (see
RESULT note in build_nb286.py): the shared EGCL trunk is trained jointly against both losses, so a different
center-prediction loss gradient shifts the trunk representations the classification head reads from too.
Reported below as a sanity-check-turned-finding, not the actual result.

usage: python3 analyze286.py <phase266_full_preds.pt (baseline)> <phase286_full_preds.pt> <out_prefix>
"""
import sys
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

base_path, new_path, OUT = sys.argv[1:4]
P = {"base": torch.load(base_path, map_location="cpu", weights_only=False),
     "local": torch.load(new_path, map_location="cpu", weights_only=False)}
kt = sorted(P["base"]["seeds"][0]["test"])
for n, p in P.items():
    assert sorted(p["seeds"]) == list(range(8)) and sorted(p["seeds"][0]["test"]) == kt, n


def best_thr(p, y):
    ts = np.quantile(p, np.linspace(0.5, 0.999, 200))
    return ts[int(np.argmax([matthews_corrcoef(y, p >= t) for t in ts]))]


def seed_metrics(S):
    pv = np.concatenate([S["val"][k]["p"] for k in sorted(S["val"])]); yv = np.concatenate([S["val"][k]["y"] for k in sorted(S["val"])])
    thr = best_thr(pv, yv)
    pt = np.concatenate([S["test"][k]["p"] for k in kt]); yt = np.concatenate([S["test"][k]["y"] for k in kt])
    e = np.array([np.linalg.norm(S["test"][k]["c"] - S["test"][k]["tc"].reshape(-1)) for k in kt])
    return dict(pr=average_precision_score(yt, pt), roc=roc_auc_score(yt, pt), mcc=matthews_corrcoef(yt, pt >= thr),
                dcc4=float(np.mean(e <= 4)), dcc8=float(np.mean(e <= 8)), err=float(e.mean()))


rows = [dict(arm=n, seed=s, **seed_metrics(p["seeds"][s])) for n, p in P.items() for s in range(8)]
df = pd.DataFrame(rows); df.to_csv(OUT + "_per_seed.csv", index=False)
print("per-seed means (8 seeds)")
print(f"{'metric':6s} " + " ".join(f"{n:>9s}" for n in P))
for m in ["dcc4", "dcc8", "err", "pr", "roc", "mcc"]:
    print(f"{m:6s} " + " ".join(f"{df[df.arm==n][m].mean():9.4f}" for n in P))

print("\nsanity check: classification metrics should be ~identical per seed (logits never touch loc readout)")
for m in ["pr", "roc", "mcc"]:
    d = (df[df.arm == "local"].sort_values("seed")[m].values - df[df.arm == "base"].sort_values("seed")[m].values)
    print(f"  {m}: max abs per-seed diff = {np.abs(d).max():.2e}")

print("\npaired over seeds, local - base  (PRIMARY = dcc4, dcc8)")
rec = []
for m in ["dcc4", "dcc8", "err", "pr", "roc", "mcc"]:
    x = df[df.arm == "local"].sort_values("seed")[m].values
    y = df[df.arm == "base"].sort_values("seed")[m].values
    d = x - y
    tp = ttest_rel(x, y).pvalue; wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    rec.append(dict(metric=m, base=y.mean(), local=x.mean(), delta=d.mean(), wins=int((d > 0).sum()), t_p=tp, w_p=wp))
    flag = "  * CONFIRMED" if (tp < .05 and wp < .05) else ""
    primary = " [PRIMARY]" if m in ("dcc4", "dcc8") else ""
    print(f"  {m:6s} {y.mean():.4f} -> {x.mean():.4f} delta {d.mean():+.4f} {int((d>0).sum())}/8  t p={tp:.3f} W p={wp:.3f}{flag}{primary}")
pd.DataFrame(rec).to_csv(OUT + "_paired.csv", index=False)

print("\n8-seed ensembles")
for n, p in P.items():
    S = dict(val={k: dict(p=np.mean([p["seeds"][s]["val"][k]["p"] for s in range(8)], 0), y=p["seeds"][0]["val"][k]["y"]) for k in p["seeds"][0]["val"]},
             test={k: dict(p=np.mean([p["seeds"][s]["test"][k]["p"] for s in range(8)], 0), y=p["seeds"][0]["test"][k]["y"],
                           c=np.mean([p["seeds"][s]["test"][k]["c"] for s in range(8)], 0), tc=p["seeds"][0]["test"][k]["tc"]) for k in kt})
    m = seed_metrics(S)
    print(f"  {n:6s} " + "  ".join(f"{k} {m[k]:.4f}" for k in ["dcc4", "dcc8", "err", "pr", "roc", "mcc"]))
