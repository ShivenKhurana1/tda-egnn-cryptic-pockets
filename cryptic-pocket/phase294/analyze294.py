"""Phase 294 analysis: each arm (ncmask, pu15, pu20) vs the Phase 292 base (matches Phase 291 to within 8e-5 per residue), paired by seed, on the OFFICIAL labels.
PRIMARY, pre-specified: validation PR-AUC, paired t-test AND Wilcoxon, Holm-adjusted across the 4 arms (both adjusted p < .05 to
call an arm confirmed). Secondary (test): pooled ROC/PR/MCC (val-tuned threshold), per-protein AP and precision@20, per-protein
AUC on the strictly cryptic test proteins (main-site labels, the external benchmark's definition), centre-head DCC4.
Extra secondary: p20_any = top-20 precision counting known non-cryptic binders (CryptoBench noncryptic-pockets.json) as hits.

usage: python3 analyze294.py <phase292_base_preds.pt> <dir with phase294_{arm}_preds.pt> <out_prefix>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

BASE, D, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
ARMS = ["base", "ncmask", "pu15", "pu20"]
A = {a: torch.load(BASE if a == "base" else D / f"phase294_{a}_preds.pt", map_location="cpu", weights_only=False) for a in ARMS}
ref = A["base"]["seeds"][0]; KV, KT = sorted(ref["val"]), sorted(ref["test"])
for a, P in A.items():
    assert sorted(P["seeds"]) == list(range(8)), a
    assert sorted(P["seeds"][0]["test"]) == KT and sorted(P["seeds"][0]["val"]) == KV, a
YV = {k: ref["val"][k]["y"].astype(int) for k in KV}; YT = {k: ref["test"][k]["y"].astype(int) for k in KT}
YM = {k: ref["test"][k]["y_main"].astype(int) for k in KT}
NCM = {k: A["ncmask"]["seeds"][0]["test"][k]["nc"].astype(bool) for k in KT}
L276 = json.load(open(Path(__file__).resolve().parents[1] / "phase276" / "phase276a_crypticity_labels.json"))
CRYPT = [k for k in KT if L276.get(f"{k[0]}_{k[1]}", {}).get("label") == "cryptic" and 0 < YM[k].sum() < len(YM[k])]


def metrics(S):
    pv = np.concatenate([S["val"][k]["p"] for k in KV]); yv = np.concatenate([YV[k] for k in KV])
    pt = np.concatenate([S["test"][k]["p"] for k in KT]); yt = np.concatenate([YT[k] for k in KT])
    ts = np.quantile(pv, np.linspace(.5, .999, 200)); thr = ts[int(np.argmax([matthews_corrcoef(yv, pv >= t) for t in ts]))]
    ks = [k for k in KT if 0 < YT[k].sum() < len(YT[k])]
    p20 = np.mean([YT[k][np.argsort(-S["test"][k]["p"])[:20]].mean() for k in ks])
    p20_any = np.mean([(YT[k].astype(bool) | NCM[k])[np.argsort(-S["test"][k]["p"])[:20]].mean() for k in ks])
    return dict(val_pr=average_precision_score(yv, pv), pr=average_precision_score(yt, pt), roc=roc_auc_score(yt, pt),
                mcc=matthews_corrcoef(yt, pt >= thr), ap_pp=np.mean([average_precision_score(YT[k], S["test"][k]["p"]) for k in ks]),
                p20=p20, p20_any=p20_any, cryptic_auc=np.mean([roc_auc_score(YM[k], S["test"][k]["p"]) for k in CRYPT]),
                dcc4=np.mean([np.linalg.norm(S["test"][k]["c"] - S["test"][k]["tc"].reshape(-1)) <= 4 for k in KT]))


rows = [dict(arm=a, seed=s, **metrics(P["seeds"][s])) for a, P in A.items() for s in range(8)]
df = pd.DataFrame(rows); df.to_csv(OUT + "_per_seed.csv", index=False)
METS = ["val_pr", "pr", "roc", "mcc", "ap_pp", "p20", "p20_any", "cryptic_auc", "dcc4"]
print(f"test chains {len(KT)}, strictly cryptic {len(CRYPT)}")
print("per-arm means:\n" + df.groupby("arm")[METS].mean().loc[ARMS].round(4).to_string())

base = df[df.arm == "base"].sort_values("seed"); summary = {}
prim = {}
for a in ARMS[1:]:
    x = df[df.arm == a].sort_values("seed"); summary[a] = {}
    for m in METS:
        d = x[m].values - base[m].values
        tp = ttest_rel(x[m].values, base[m].values).pvalue; wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        summary[a][m] = dict(delta=float(d.mean()), up=int((d > 0).sum()), t_p=float(tp), w_p=float(wp))
    prim[a] = summary[a]["val_pr"]
# Holm across the 3 arms, separately for t and Wilcoxon p on the primary
for key in ("t_p", "w_p"):
    order = sorted(prim, key=lambda a: prim[a][key]); run = 0
    for i, a in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * prim[a][key])); prim[a][key + "_holm"] = run
print("\npaired vs base (8 seeds); PRIMARY = val_pr, Holm-adjusted over 3 arms")
for a in ARMS[1:]:
    s = summary[a]; p = prim[a]
    conf = "CONFIRMED" if (p["t_p_holm"] < .05 and p["w_p_holm"] < .05 and s["val_pr"]["delta"] > 0) else "not confirmed"
    print(f"\n[{a}] val_pr {s['val_pr']['delta']:+.4f} {s['val_pr']['up']}/8  t p={p['t_p']:.3g} (Holm {p['t_p_holm']:.3g})  W p={p['w_p']:.3g} (Holm {p['w_p_holm']:.3g})  -> {conf}")
    for m in METS[1:]:
        r = s[m]; print(f"    {m:11s} {r['delta']:+.4f}  {r['up']}/8  t p={r['t_p']:.3g}  W p={r['w_p']:.3g}")
    summary[a]["primary_verdict"] = conf

print("\n8-seed ensembles (test, official labels):")
for a, P in A.items():
    E = {sp: {k: dict(P["seeds"][0][sp][k], p=np.mean([P["seeds"][s][sp][k]["p"] for s in range(8)], 0),
                      c=np.mean([P["seeds"][s][sp][k]["c"] for s in range(8)], 0)) for k in P["seeds"][0][sp]} for sp in ("val", "test")}
    e = metrics(E); summary.setdefault("ensembles", {})[a] = {k: float(v) for k, v in e.items()}
    print(f"  {a:7s} ROC {e['roc']:.4f} PR {e['pr']:.4f} MCC {e['mcc']:.4f} | AP/protein {e['ap_pp']:.4f} P@20 {e['p20']:.4f} P@20any {e['p20_any']:.4f} | cryptic AUC {e['cryptic_auc']:.4f} | DCC4 {e['dcc4']:.3f}")

json.dump(summary, open(OUT + "_summary.json", "w"), indent=1, default=float)
