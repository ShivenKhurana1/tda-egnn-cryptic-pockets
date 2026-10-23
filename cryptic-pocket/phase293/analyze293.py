"""Phase 293 analysis: frozen650 and lora650 (ESM2-650M branch, frozen vs LoRA fine-tuned) vs the Phase 292 base (bit-identical to
Phase 291), paired by seed, official labels. PRIMARY, pre-specified: validation PR-AUC, paired t AND Wilcoxon, Holm-adjusted over the two
comparisons with base. Key secondary: lora650 vs frozen650 (the effect of fine-tuning itself). Same secondary metrics as Phase 292.

usage: python3 analyze293.py <phase292_base_preds.pt> <phase293_frozen650_preds.pt> <phase293a_lora650_preds.pt> <phase293b_lora650_preds.pt> <out_prefix>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

ld = lambda p: torch.load(p, map_location="cpu", weights_only=False)
base_p, frozen_p, la_p, lb_p, OUT = sys.argv[1:6]
lora = ld(la_p); lora["seeds"].update(ld(lb_p)["seeds"])
A = {"base": ld(base_p), "frozen650": ld(frozen_p), "lora650": lora}
SEEDS = sorted(set.intersection(*[set(P["seeds"]) for P in A.values()]))
print("seeds present in every arm:", SEEDS)
ref = A["base"]["seeds"][0]; KV, KT = sorted(ref["val"]), sorted(ref["test"])
for a, P in A.items():
    for s in SEEDS: assert sorted(P["seeds"][s]["test"]) == KT and sorted(P["seeds"][s]["val"]) == KV, a
YV = {k: ref["val"][k]["y"].astype(int) for k in KV}; YT = {k: ref["test"][k]["y"].astype(int) for k in KT}
YM = {k: ref["test"][k]["y_main"].astype(int) for k in KT}
L276 = json.load(open(Path(__file__).resolve().parents[1] / "phase276" / "phase276a_crypticity_labels.json"))
CRYPT = [k for k in KT if L276.get(f"{k[0]}_{k[1]}", {}).get("label") == "cryptic" and 0 < YM[k].sum() < len(YM[k])]


def metrics(S):
    pv = np.concatenate([S["val"][k]["p"] for k in KV]); yv = np.concatenate([YV[k] for k in KV])
    pt = np.concatenate([S["test"][k]["p"] for k in KT]); yt = np.concatenate([YT[k] for k in KT])
    ts = np.quantile(pv, np.linspace(.5, .999, 200)); thr = ts[int(np.argmax([matthews_corrcoef(yv, pv >= t) for t in ts]))]
    ks = [k for k in KT if 0 < YT[k].sum() < len(YT[k])]
    p20 = np.mean([YT[k][np.argsort(-S["test"][k]["p"])[:20]].mean() for k in ks])
    return dict(val_pr=average_precision_score(yv, pv), pr=average_precision_score(yt, pt), roc=roc_auc_score(yt, pt),
                mcc=matthews_corrcoef(yt, pt >= thr), ap_pp=np.mean([average_precision_score(YT[k], S["test"][k]["p"]) for k in ks]),
                p20=p20, cryptic_auc=np.mean([roc_auc_score(YM[k], S["test"][k]["p"]) for k in CRYPT]),
                dcc4=np.mean([np.linalg.norm(S["test"][k]["c"] - S["test"][k]["tc"].reshape(-1)) <= 4 for k in KT]))



rows = [dict(arm=a, seed=s, **metrics(P["seeds"][s])) for a, P in A.items() for s in SEEDS]
df = pd.DataFrame(rows); df.to_csv(OUT + "_per_seed.csv", index=False)
METS = ["val_pr", "pr", "roc", "mcc", "ap_pp", "p20", "cryptic_auc", "dcc4"]
print(f"test chains {len(KT)}, strictly cryptic {len(CRYPT)}")
print(df.groupby("arm")[METS].mean().round(4).to_string())
if "epochs" in A["lora650"]["seeds"][SEEDS[0]]:
    print("epochs used:", {a: [A[a]["seeds"][s].get("epochs") for s in SEEDS] for a in ("frozen650", "lora650")})

def paired(a, b):
    x = df[df.arm == a].sort_values("seed"); y = df[df.arm == b].sort_values("seed"); r = {}
    for m in METS:
        d = x[m].values - y[m].values
        r[m] = dict(delta=float(d.mean()), up=int((d > 0).sum()), t_p=float(ttest_rel(x[m].values, y[m].values).pvalue),
                    w_p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
    return r
S = {"frozen650-base": paired("frozen650", "base"), "lora650-base": paired("lora650", "base"), "lora650-frozen650": paired("lora650", "frozen650")}
prim = ["frozen650-base", "lora650-base"]
for key in ("t_p", "w_p"):
    order = sorted(prim, key=lambda c: S[c]["val_pr"][key]); run = 0
    for i, c in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * S[c]["val_pr"][key])); S[c]["val_pr"][key + "_holm"] = run
n = len(SEEDS)
for c, r in S.items():
    v = r["val_pr"]; hol = f"  Holm t {v['t_p_holm']:.3g} W {v['w_p_holm']:.3g}" if "t_p_holm" in v else ""
    conf = ("CONFIRMED" if v.get("t_p_holm", v["t_p"]) < .05 and v.get("w_p_holm", v["w_p"]) < .05 and v["delta"] > 0 else "not confirmed")
    print(f"\n[{c}] val_pr {v['delta']:+.4f} {v['up']}/{n}  t p={v['t_p']:.3g}  W p={v['w_p']:.3g}{hol}  -> {conf}")
    for m in METS[1:]:
        x = r[m]; print(f"    {m:11s} {x['delta']:+.4f}  {x['up']}/{n}  t p={x['t_p']:.3g}  W p={x['w_p']:.3g}")
    r["verdict"] = conf
print("\nseed ensembles (test, official labels):")
for a, P in A.items():
    E = {sp: {k: dict(P["seeds"][SEEDS[0]][sp][k], p=np.mean([P["seeds"][s][sp][k]["p"] for s in SEEDS], 0),
                      c=np.mean([P["seeds"][s][sp][k]["c"] for s in SEEDS], 0)) for k in (KV if sp == "val" else KT)}
         for sp in ("val", "test")}
    e = metrics(E); S.setdefault("ensembles", {})[a] = {k: float(v) for k, v in e.items()}
    print(f"  {a:9s} ROC {e['roc']:.4f} PR {e['pr']:.4f} MCC {e['mcc']:.4f} | AP/protein {e['ap_pp']:.4f} P@20 {e['p20']:.4f} | cryptic AUC {e['cryptic_auc']:.4f} | DCC4 {e['dcc4']:.3f}")
json.dump(S, open(OUT + "_summary.json", "w"), indent=1, default=float)
