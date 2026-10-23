"""Phase 297: test-set numbers for base vs frozen650_prott5 (selected on the Phase 295/296 CV instrument, not on test). Paired by seed;
8-seed ensembles on official labels, with the main-site strictly-cryptic per-protein AUC (16 proteins, external benchmark definition).
usage: python3 analyze297.py <phase297_test_preds.pt> <out_prefix>"""
import sys, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
A = torch.load(sys.argv[1], map_location="cpu", weights_only=False); OUT = sys.argv[2]
ref = A["base"]["seeds"][0]; KV, KT = sorted(ref["val"]), sorted(ref["test"])
L276 = json.load(open(Path(__file__).resolve().parents[1] / "phase276" / "phase276a_crypticity_labels.json"))
CRYPT = [k for k in KT if L276.get(f"{k[0]}_{k[1]}", {}).get("label") == "cryptic" and 0 < ref["test"][k]["y_main"].sum() < len(ref["test"][k]["y_main"])]
def metrics(S):
    pv = np.concatenate([S["val"][k]["p"] for k in KV]); yv = np.concatenate([S["val"][k]["y"] for k in KV])
    pt = np.concatenate([S["test"][k]["p"] for k in KT]); yt = np.concatenate([S["test"][k]["y"] for k in KT])
    ts = np.quantile(pv, np.linspace(.5, .999, 200)); thr = ts[int(np.argmax([matthews_corrcoef(yv, pv >= t) for t in ts]))]
    ks = [k for k in KT if 0 < S["test"][k]["y"].sum() < len(S["test"][k]["y"])]
    return dict(val_pr=average_precision_score(yv, pv), pr=average_precision_score(yt, pt), roc=roc_auc_score(yt, pt), mcc=matthews_corrcoef(yt, pt >= thr),
                ap_pp=np.mean([average_precision_score(S["test"][k]["y"], S["test"][k]["p"]) for k in ks]),
                p20=np.mean([S["test"][k]["y"][np.argsort(-S["test"][k]["p"])[:20]].mean() for k in ks]),
                cryptic_auc=np.mean([roc_auc_score(S["test"][k]["y_main"], S["test"][k]["p"]) for k in CRYPT]),
                dcc4=np.mean([np.linalg.norm(S["test"][k]["c"] - S["test"][k]["tc"].reshape(-1)) <= 4 for k in KT]))
R = {a: [metrics(A[a]["seeds"][s]) for s in range(8)] for a in A}
METS = list(R["base"][0]); out = {"paired": {}, "ensembles": {}}
print(f"{'metric':12s} {'base':>8s} {'multiPLM':>9s}  delta  up  t p    W p")
for m in METS:
    x = np.array([r[m] for r in R["base"]]); y = np.array([r[m] for r in R["frozen650_prott5"]]); d = y - x
    tp = ttest_rel(y, x).pvalue; wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    out["paired"][m] = dict(base=float(x.mean()), multiplm=float(y.mean()), delta=float(d.mean()), up=int((d > 0).sum()), t_p=float(tp), w_p=float(wp))
    print(f"{m:12s} {x.mean():8.4f} {y.mean():9.4f} {d.mean():+.4f} {int((d>0).sum())}/8 {tp:.3g} {wp:.3g}")
for a in A:
    E = {sp: {k: dict(A[a]["seeds"][0][sp][k], p=np.mean([A[a]["seeds"][s][sp][k]["p"] for s in range(8)], 0),
                      c=np.mean([A[a]["seeds"][s][sp][k]["c"] for s in range(8)], 0)) for k in (KV if sp == "val" else KT)} for sp in ("val", "test")}
    e = metrics(E); out["ensembles"][a] = {k: float(v) for k, v in e.items()}
    print(f"8-seed ensemble {a:17s} ROC {e['roc']:.4f} PR {e['pr']:.4f} MCC {e['mcc']:.4f} | AP/protein {e['ap_pp']:.4f} P@20 {e['p20']:.4f} | cryptic AUC {e['cryptic_auc']:.4f} | DCC4 {e['dcc4']:.3f}")
json.dump(out, open(OUT + "_summary.json", "w"), indent=1)
