"""Phase 311: final model (stack3) on the Phase 277 external sets. Same scoring as analyze277.py: PocketMiner pooled only, over its
evaluated (non-masked) residues; cryptic-site benchmark pooled and per protein; each on all chains and on the homology-clean subset
(Phase 277c). Reports the per-seed mean and the 8-model ensemble, plus CryptoBench's strictly cryptic test proteins (main-site labels)
from the same run, and checks that val/test reproduce Phase 310 seeds 0-7.
usage: python3 analyze311.py <phase311_ext_preds.pt> <out.json> [phase310_test_preds.pt]"""
import json, sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import average_precision_score, roc_auc_score
CP = Path(__file__).resolve().parents[1]
A = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["stack3"]["seeds"]; seeds = sorted(A); assert seeds == list(range(8)), seeds
hom = json.load(open(CP / "phase277" / "results" / "phase277c_homology.json"))
clean = {"pocketminer": lambda k: not hom["pm_leak"].get(k[0] + k[1], True), "csb_cryptic": lambda k: not hom["csb_leak"].get(f"{k[0]}_{k[1]}", True)}
L276 = json.load(open(CP / "phase276" / "phase276a_crypticity_labels.json"))
out = {}
def pooled(name, keys, p_of):
    m = [A[0][name][k].get("eval_mask", np.ones(len(A[0][name][k]["y"]), bool)) for k in keys]
    y = np.concatenate([A[0][name][k]["y"][mm] for k, mm in zip(keys, m)]); p = np.concatenate([p_of(k)[mm] for k, mm in zip(keys, m)])
    return roc_auc_score(y, p), average_precision_score(y, p), len(y), int(y.sum())
for name in ("pocketminer", "csb_cryptic"):
    for subset in ("all", "clean"):
        ks = [k for k in sorted(A[0][name]) if subset == "all" or clean[name](k)]
        per = np.array([pooled(name, ks, lambda k, s=s: A[s][name][k]["p"])[:2] for s in seeds])
        roc_e, pr_e, n, npos = pooled(name, ks, lambda k: np.mean([A[s][name][k]["p"] for s in seeds], 0))
        r = dict(n_proteins=len(ks), n_res=n, n_pos=npos, roc_seed_mean=float(per[:, 0].mean()), pr_seed_mean=float(per[:, 1].mean()), roc_ens=float(roc_e), pr_ens=float(pr_e))
        if name == "csb_cryptic":
            kp = [k for k in ks if 0 < A[0][name][k]["y"].sum() < len(A[0][name][k]["y"])]
            r["protein_roc_ens"] = float(np.mean([roc_auc_score(A[0][name][k]["y"], np.mean([A[s][name][k]["p"] for s in seeds], 0)) for k in kp]))
        out[f"{name}_{subset}"] = r
        print(f"{name:12s} [{subset:5s}] {len(ks):3d} proteins {n:6d} residues {npos:5d} pos | per-seed ROC {r['roc_seed_mean']:.4f} PR {r['pr_seed_mean']:.4f} | "
              f"ensemble ROC {roc_e:.4f} PR {pr_e:.4f}" + (f" | per-protein ROC {r['protein_roc_ens']:.4f}" if "protein_roc_ens" in r else ""))
T = A[0]["test"]; KT = sorted(T)
CR = [k for k in KT if L276.get(f"{k[0]}_{k[1]}", {}).get("label") == "cryptic" and 0 < T[k]["y_main"].sum() < len(T[k]["y_main"])]
y = np.concatenate([T[k]["y_main"] for k in CR]); pe = np.concatenate([np.mean([A[s]["test"][k]["p"] for s in seeds], 0) for k in CR])
ps = [roc_auc_score(y, np.concatenate([A[s]["test"][k]["p"] for k in CR])) for s in seeds]
out["cryptobench_strict"] = dict(n_proteins=len(CR), roc_ens=float(roc_auc_score(y, pe)), roc_seed_mean=float(np.mean(ps)), pr_ens=float(average_precision_score(y, pe)))
print(f"CryptoBench strictly cryptic ({len(CR)} proteins, main-site labels): ensemble ROC {out['cryptobench_strict']['roc_ens']:.4f}, per-seed {np.mean(ps):.4f}")
yt = np.concatenate([T[k]["y"] for k in KT]); pt = np.concatenate([np.mean([A[s]["test"][k]["p"] for s in seeds], 0) for k in KT])
out["cryptobench_test"] = dict(roc_ens=float(roc_auc_score(yt, pt)), pr_ens=float(average_precision_score(yt, pt)))
print(f"CryptoBench test ensemble ROC {roc_auc_score(yt, pt):.4f} PR {average_precision_score(yt, pt):.4f} (Phase 310 seeds 0-7: 0.9079 / 0.4762)")
if len(sys.argv) > 3:
    B = torch.load(sys.argv[3], map_location="cpu", weights_only=False)["stack3"]["seeds"]
    dmax = max(float(np.abs(A[s]["test"][k]["p"] - B[s]["test"][k]["p"]).max()) for s in seeds for k in KT)
    print(f"max |p| difference vs Phase 310 on test: {dmax:.2e}")
json.dump(out, open(sys.argv[2], "w"), indent=1); print("wrote", sys.argv[2])
