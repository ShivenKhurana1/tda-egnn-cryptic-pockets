"""Phase 284: does crypticity-proxy loss up-weighting (Phase 283's cavity-voxel score) help the true-cryptic test chains, without hurting the
rest -- and does it do anything the already-tested-negative small-pocket up-weighting (Phase 269) doesn't already do? Primary = true-cryptic
test chains (external audit's real labels, not the proxy), protein-level ROC-AUC/PR-AUC, seed-paired (8 seeds) vs the unweighted Phase 266c
baseline. Secondary = the same pocket-size bins Phase 269 used (context: is any effect here just re-deriving that already-negative result,
given the proxy correlates with pocket size at Spearman r=0.876 on validation, Phase 283). Non-inferiority = pooled overall PR/ROC/MCC.
Two arms tested (w1/w2); confirmed only if paired t AND Wilcoxon are both <0.05 (n=8 Wilcoxon floor 0.0078).

usage: python3 analyze284.py <phase266_full_preds.pt (baseline)> <phase284w1_full_preds.pt> <phase284w2_full_preds.pt> <phase276a_crypticity_labels.json> <out_prefix>
"""
import json, sys
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

base, w1, w2, crypt_path, OUT = sys.argv[1:6]
P = {n: torch.load(f, map_location="cpu", weights_only=False) for n, f in [("base", base), ("w1", w1), ("w2", w2)]}
kt = sorted(P["base"]["seeds"][0]["test"])
for n, p in P.items(): assert sorted(p["seeds"]) == list(range(8)) and sorted(p["seeds"][0]["test"]) == kt, n
size = {k: int(P["base"]["seeds"][0]["test"][k]["y"].sum()) for k in kt}

label_of = {tuple(k.rsplit("_", 1)): v["label"] for k, v in json.load(open(crypt_path)).items()}
crypt_of = {k: label_of.get(k) for k in kt}
n_cryptic = sum(1 for k in kt if crypt_of[k] == "cryptic"); n_preformed = sum(1 for k in kt if crypt_of[k] == "preformed")
print(f"true-label coverage on test: {n_cryptic} cryptic, {n_preformed} preformed, {sum(1 for k in kt if crypt_of[k] is None)} unlabelled/neither of {len(kt)}")

BINS = {"cryptic": lambda k: crypt_of[k] == "cryptic", "preformed": lambda k: crypt_of[k] == "preformed",
        "<=8": lambda k: size[k] <= 8, "9-15": lambda k: 9 <= size[k] <= 15, ">=16": lambda k: size[k] >= 16, "all": lambda k: True}


def best_thr(p, y):
    ts = np.quantile(p, np.linspace(0.5, 0.999, 200)); return ts[int(np.argmax([matthews_corrcoef(y, p >= t) for t in ts]))]


def seed_metrics(S):
    out = {}
    pv = np.concatenate([S["val"][k]["p"] for k in sorted(S["val"])]); yv = np.concatenate([S["val"][k]["y"] for k in sorted(S["val"])]); thr = best_thr(pv, yv)
    for b, f in BINS.items():
        ks = [k for k in kt if f(k)]
        both = [k for k in ks if 0 < S["test"][k]["y"].sum() < len(S["test"][k]["y"])]
        out[f"{b}|roc"] = np.mean([roc_auc_score(S["test"][k]["y"], S["test"][k]["p"]) for k in both]) if both else float("nan")
        out[f"{b}|pr"] = np.mean([average_precision_score(S["test"][k]["y"], S["test"][k]["p"]) for k in both]) if both else float("nan")
        e = np.array([np.linalg.norm(S["test"][k]["c"] - S["test"][k]["tc"].reshape(-1)) for k in ks])
        out[f"{b}|dcc4"] = float(np.mean(e <= 4)) if len(ks) else float("nan"); out[f"{b}|err"] = float(e.mean()) if len(ks) else float("nan")
    pt = np.concatenate([S["test"][k]["p"] for k in kt]); yt = np.concatenate([S["test"][k]["y"] for k in kt])
    out["pooled|pr"] = average_precision_score(yt, pt); out["pooled|roc"] = roc_auc_score(yt, pt); out["pooled|mcc"] = matthews_corrcoef(yt, pt >= thr)
    return out


rows = [dict(arm=n, seed=s, **seed_metrics(p["seeds"][s])) for n, p in P.items() for s in range(8)]
df = pd.DataFrame(rows); df.to_csv(OUT + "_per_seed.csv", index=False)
print("test chains per stratum:", {b: sum(f(k) for k in kt) for b, f in BINS.items()})
show = ["cryptic|roc", "cryptic|pr", "cryptic|dcc4", "preformed|roc", "preformed|pr", "<=8|roc", "<=8|pr", ">=16|roc",
        "all|dcc4", "pooled|pr", "pooled|roc", "pooled|mcc"]
print("\nper-seed means (8 seeds)")
print(f"{'metric':14s} " + " ".join(f"{n:>9s}" for n in P))
for m in show: print(f"{m:14s} " + " ".join(f"{df[df.arm==n][m].mean():9.4f}" for n in P))
print("\npaired over seeds, arm - base  (PRIMARY = cryptic|roc, cryptic|pr)")
rec = []
for n in ("w1", "w2"):
    for m in show:
        x = df[df.arm == n].sort_values("seed")[m].values; y = df[df.arm == "base"].sort_values("seed")[m].values; d = x - y
        tp = ttest_rel(x, y).pvalue; wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        rec.append(dict(arm=n, metric=m, base=y.mean(), arm_mean=x.mean(), delta=d.mean(), wins=int((d > 0).sum()), t_p=tp, w_p=wp))
        flag = "  * CONFIRMED" if (tp < .05 and wp < .05) else ""
        primary = " [PRIMARY]" if m.startswith("cryptic|") else ""
        print(f"  {n} {m:13s} {y.mean():.4f} -> {x.mean():.4f} delta {d.mean():+.4f} {int((d>0).sum())}/8  t p={tp:.1e} W p={wp:.1e}{flag}{primary}")
pd.DataFrame(rec).to_csv(OUT + "_paired.csv", index=False)

print("\n8-seed ensembles")
for n, p in P.items():
    S = dict(val={k: dict(p=np.mean([p["seeds"][s]["val"][k]["p"] for s in range(8)], 0), y=p["seeds"][0]["val"][k]["y"]) for k in p["seeds"][0]["val"]},
             test={k: dict(p=np.mean([p["seeds"][s]["test"][k]["p"] for s in range(8)], 0), y=p["seeds"][0]["test"][k]["y"],
                           c=np.mean([p["seeds"][s]["test"][k]["c"] for s in range(8)], 0), tc=p["seeds"][0]["test"][k]["tc"]) for k in kt})
    m = seed_metrics(S)
    print(f"  {n:5s} " + "  ".join(f"{k} {m[k]:.4f}" for k in ["cryptic|roc", "cryptic|pr", "preformed|roc", "all|dcc4", "pooled|pr", "pooled|roc", "pooled|mcc"]))
