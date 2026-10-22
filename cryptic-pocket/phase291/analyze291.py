"""Phase 291 analysis: does training on CryptoBench's official union-of-pockets labels beat training on the main pocket only?
Arms (same checkpoints, seeds, split, recipe): main = Phase 266c preds, official = Phase 291 preds. Both arms are scored on the
SAME label vectors, taken from the Phase 291 file (y = official, y_main = main-only; Phase 291 verified y_main reproduces
266c's stored labels exactly). PRIMARY, pre-specified: validation PR-AUC on official labels, paired by seed (t-test AND
Wilcoxon < 0.05 to call it confirmed). Secondary: test pooled ROC/PR/MCC (val-tuned threshold), per-protein macro AUC/AP,
DCC4 of the centre head (target unchanged: main pocket), and everything again on main-only labels.

usage: python3 analyze291.py <phase266_full_preds.pt> <phase291_official_preds.pt> <out_prefix>
"""
import sys, json
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

p266, p291, OUT = sys.argv[1:4]
A = {"main": torch.load(p266, map_location="cpu", weights_only=False),
     "official": torch.load(p291, map_location="cpu", weights_only=False)}
ref = A["official"]["seeds"][0]
KV, KT = sorted(ref["val"]), sorted(ref["test"])
LAB = {lab: {sp: {k: ref[sp][k]["y" if lab == "official" else "y_main"].astype(int) for k in ref[sp]} for sp in ("val", "test")}
       for lab in ("official", "main")}
for arm, P in A.items():
    assert sorted(P["seeds"]) == list(range(8)), arm
    assert sorted(P["seeds"][0]["val"]) == KV and sorted(P["seeds"][0]["test"]) == KT, arm
# 266c stored main labels must equal 291's y_main (identity of the label vectors both arms are scored on)
assert all(np.array_equal(A["main"]["seeds"][0]["test"][k]["y"], ref["test"][k]["y_main"]) for k in KT)


def scores(get_p, lab):
    yv, yt = LAB[lab]["val"], LAB[lab]["test"]
    pv = np.concatenate([get_p("val", k) for k in KV]); yvc = np.concatenate([yv[k] for k in KV])
    pt = np.concatenate([get_p("test", k) for k in KT]); ytc = np.concatenate([yt[k] for k in KT])
    ts = np.quantile(pv, np.linspace(.5, .999, 200)); thr = ts[int(np.argmax([matthews_corrcoef(yvc, pv >= t) for t in ts]))]
    ks = [k for k in KT if 0 < yt[k].sum() < len(yt[k])]
    return dict(val_pr=average_precision_score(yvc, pv), pr=average_precision_score(ytc, pt), roc=roc_auc_score(ytc, pt),
                mcc=matthews_corrcoef(ytc, pt >= thr),
                auc_pp=float(np.mean([roc_auc_score(yt[k], get_p("test", k)) for k in ks])),
                ap_pp=float(np.mean([average_precision_score(yt[k], get_p("test", k)) for k in ks])))


def dcc4(get_c):
    return float(np.mean([np.linalg.norm(get_c(k) - ref["test"][k]["tc"].reshape(-1)) <= 4 for k in KT]))


rows = []
for arm, P in A.items():
    for s in range(8):
        S = P["seeds"][s]
        for lab in ("official", "main"):
            r = scores(lambda sp, k: S[sp][k]["p"], lab)
            rows.append(dict(arm=arm, seed=s, labels=lab, dcc4=dcc4(lambda k: S["test"][k]["c"]), **r))
df = pd.DataFrame(rows); df.to_csv(OUT + "_per_seed.csv", index=False)
METS = ["val_pr", "pr", "roc", "mcc", "auc_pp", "ap_pp", "dcc4"]
summary = {}
for lab in ("official", "main"):
    print(f"\n=== scored on {lab.upper()} labels; paired over 8 seeds, official-trained minus main-trained ===")
    a = df[(df.arm == "main") & (df.labels == lab)].sort_values("seed"); b = df[(df.arm == "official") & (df.labels == lab)].sort_values("seed")
    summary[lab] = {}
    for m in METS:
        x, y = a[m].values, b[m].values; d = y - x
        tp = ttest_rel(y, x).pvalue; wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        tag = " [PRIMARY]" if (lab == "official" and m == "val_pr") else ""
        print(f"  {m:7s} {x.mean():.4f} -> {y.mean():.4f}  delta {d.mean():+.4f}  {int((d > 0).sum())}/8  t p={tp:.3g}  W p={wp:.3g}{tag}")
        summary[lab][m] = dict(main=float(x.mean()), official=float(y.mean()), delta=float(d.mean()), up=int((d > 0).sum()), t_p=float(tp), w_p=float(wp))
    for arm, P in A.items():
        ens = scores(lambda sp, k: np.mean([P["seeds"][s][sp][k]["p"] for s in range(8)], 0), lab)
        e_d = dcc4(lambda k: np.mean([P["seeds"][s]["test"][k]["c"] for s in range(8)], 0))
        summary[lab][f"ensemble_{arm}"] = dict(ens, dcc4=e_d)
        print(f"  8-seed ensemble, {arm:8s}-trained: val PR {ens['val_pr']:.4f} | test ROC {ens['roc']:.4f} PR {ens['pr']:.4f} MCC {ens['mcc']:.4f} "
              f"| per-protein AUC {ens['auc_pp']:.4f} AP {ens['ap_pp']:.4f} | DCC4 {e_d:.3f}")
json.dump(summary, open(OUT + "_summary.json", "w"), indent=1)
