"""Phase 310: 16-seed vs 8-seed (seeds 0-7) stack3 test ensemble. Primary: pooled PR and ROC on test, 95% paired bootstrap CI over test
proteins. Secondary: MCC (val-tuned threshold), per-protein AP/AUC, P@20, strictly-cryptic AUC, the 1..16 seed curve, and 16 vs the mean of
random 8-of-16 subsets.
usage: python3 analyze310.py <phase310_test_preds.pt> <out_prefix>"""
import sys, json
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
A = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["stack3"]["seeds"]; OUT = sys.argv[2]
SEEDS = sorted(A); assert SEEDS == list(range(16)), SEEDS
ref = A[0]; KV, KT = sorted(ref["val"]), sorted(ref["test"])
L276 = json.load(open(Path(__file__).resolve().parents[1] / "phase276" / "phase276a_crypticity_labels.json"))
CRYPT = [k for k in KT if L276.get(f"{k[0]}_{k[1]}", {}).get("label") == "cryptic" and 0 < ref["test"][k]["y_main"].sum() < len(ref["test"][k]["y_main"])]
KS = [k for k in KT if 0 < ref["test"][k]["y"].sum() < len(ref["test"][k]["y"])]
def ens(seeds, split):
    return {k: np.mean([A[s][split][k]["p"] for s in seeds], 0) for k in (KV if split == "val" else KT)}
Y = {k: ref["test"][k]["y"] for k in KT}; YV = np.concatenate([ref["val"][k]["y"] for k in KV])
def metrics(seeds):
    pv = np.concatenate(list(ens(seeds, "val").values())); P = ens(seeds, "test")
    pt = np.concatenate([P[k] for k in KT]); yt = np.concatenate([Y[k] for k in KT])
    ts = np.quantile(pv, np.linspace(.5, .999, 200)); thr = ts[int(np.argmax([matthews_corrcoef(YV, pv >= t) for t in ts]))]
    return dict(pr=average_precision_score(yt, pt), roc=roc_auc_score(yt, pt), mcc=matthews_corrcoef(yt, pt >= thr),
                ap_pp=np.mean([average_precision_score(Y[k], P[k]) for k in KS]), auc_pp=np.mean([roc_auc_score(Y[k], P[k]) for k in KS]),
                p20=np.mean([Y[k][np.argsort(-P[k])[:20]].mean() for k in KS]),
                cryptic_auc=np.mean([roc_auc_score(ref["test"][k]["y_main"], P[k]) for k in CRYPT]))
out = {"n_test": len(KT), "n_cryptic": len(CRYPT)}
print(f"test chains {len(KT)}, strictly-cryptic {len(CRYPT)}; val PR per seed:", " ".join(f"{A[s]['val_pr']:.3f}" for s in SEEDS))
print("epochs per seed:", [A[s]["epochs"] for s in SEEDS])
E8, E16 = metrics(range(8)), metrics(range(16)); E8b = metrics(range(8, 16))
print(f"\n{'ensemble':22s} " + " ".join(f"{m:>10s}" for m in E8))
for n, e in (("8 seeds (0-7)", E8), ("8 seeds (8-15)", E8b), ("16 seeds", E16)): print(f"{n:22s} " + " ".join(f"{v:10.4f}" for v in e.values()))
out["ens8"], out["ens8b"], out["ens16"] = E8, E8b, E16
rng = np.random.default_rng(0)
SUB = [metrics(sorted(rng.choice(16, 8, replace=False))) for _ in range(30)]
out["random8_mean"] = {m: float(np.mean([s[m] for s in SUB])) for m in E8}
print(f"{'random 8-of-16 (30)':22s} " + " ".join(f"{out['random8_mean'][m]:10.4f}" for m in E8))
print("\ncurve (first k seeds): k, pooled PR, ROC")
out["curve"] = []
for k in (1, 2, 4, 8, 12, 16):
    e = metrics(range(k)); out["curve"].append(dict(k=k, pr=e["pr"], roc=e["roc"])); print(f"  {k:2d}  {e['pr']:.4f}  {e['roc']:.4f}")
# PRIMARY: paired bootstrap over test proteins
P8, P16 = ens(range(8), "test"), ens(range(16), "test"); B = {"pr": [], "roc": []}
for _ in range(1000):
    ks = [KT[i] for i in rng.integers(0, len(KT), len(KT))]
    y = np.concatenate([Y[k] for k in ks]); a = np.concatenate([P8[k] for k in ks]); b = np.concatenate([P16[k] for k in ks])
    B["pr"].append(average_precision_score(y, b) - average_precision_score(y, a)); B["roc"].append(roc_auc_score(y, b) - roc_auc_score(y, a))
print("\nPRIMARY 16 vs 8 (seeds 0-7), paired bootstrap over test proteins (1000):")
out["primary"] = {}
for m in ("pr", "roc"):
    lo, hi = np.percentile(B[m], [2.5, 97.5]); d = E16[m] - E8[m]; ok = lo > 0
    out["primary"][m] = dict(delta=float(d), ci=[float(lo), float(hi)], confirmed=bool(ok))
    print(f"  {m:4s} {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {'CONFIRMED' if ok else 'not confirmed'}")
json.dump(out, open(f"{OUT}_summary.json", "w"), indent=1, default=float); print("wrote", f"{OUT}_summary.json")
