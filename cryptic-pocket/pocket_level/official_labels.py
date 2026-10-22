"""Residue-level metrics of the Phase 266c 3B-transfer model on two label definitions of the same 182 CryptoBench
test chains: MAIN (the is_main_holo_structure pocket only, which this project's Phase 42 ingestion used for training
and for every number reported before this script) and OFFICIAL (union of every holo pairing's apo_pocket_selection,
as CryptoBench's create-annotations.py builds its released labels, and as Seq2Pocket and the CryptoBench paper score).
Thresholds are tuned on validation under the same label definition. Reports the 8-seed ensemble and the mean of the
8 single seeds, pooled and per-protein (macro)."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score
from common import P, chains, sites, HERE

def official(ch):
    y = np.zeros(len(ch["p"]), int)
    for s in sites(ch): y[list(s)] = 1
    return y

V, T = chains("val"), chains("test")
LAB = {"main": lambda c: c["y"], "official": official}

def seed_p(C, split, s):
    return {k: P["seeds"][s][split][(c["pdb"], c["chain"])]["p"] for k, c in C.items()}

def metrics(pv, yv, pt, yt):
    pvc, yvc = np.concatenate(list(pv.values())), np.concatenate(list(yv.values()))
    ts = np.quantile(pvc, np.linspace(.5, .999, 200)); thr = ts[int(np.argmax([matthews_corrcoef(yvc, pvc >= t) for t in ts]))]
    ptc, ytc = np.concatenate([pt[k] for k in yt]), np.concatenate([yt[k] for k in yt])
    pooled = dict(roc=roc_auc_score(ytc, ptc), pr=average_precision_score(ytc, ptc), mcc=matthews_corrcoef(ytc, ptc >= thr), f1=f1_score(ytc, ptc >= thr))
    ks = [k for k in yt if 0 < yt[k].sum() < len(yt[k])]
    macro = dict(roc=np.mean([roc_auc_score(yt[k], pt[k]) for k in ks]), pr=np.mean([average_precision_score(yt[k], pt[k]) for k in ks]),
                 mcc=np.mean([matthews_corrcoef(yt[k], pt[k] >= thr) for k in ks]), f1=np.mean([f1_score(yt[k], pt[k] >= thr, zero_division=0) for k in ks]))
    return dict(pooled=pooled, macro=macro, thr=float(thr))

out = {}
for name, lab in LAB.items():
    yv = {k: lab(c) for k, c in V.items()}; yt = {k: lab(c) for k, c in T.items()}
    ens = metrics({k: c["p"] for k, c in V.items()}, yv, {k: c["p"] for k, c in T.items()}, yt)
    singles = [metrics(seed_p(V, "val", s), yv, seed_p(T, "test", s), yt) for s in range(8)]
    mean_single = {lvl: {m: float(np.mean([r[lvl][m] for r in singles])) for m in ("roc", "pr", "mcc", "f1")} for lvl in ("pooled", "macro")}
    ytc = np.concatenate(list(yt.values()))
    out[name] = dict(n_chains=len(T), n_res=int(len(ytc)), pos_rate=float(ytc.mean()), ensemble=ens, mean_single=mean_single)
    print(f"{name:8s} {len(ytc):,} residues, {ytc.mean():.2%} positive")
    for tag, r in (("8-seed ensemble", ens), ("mean single seed", mean_single)):
        for lvl in ("pooled", "macro"):
            x = r[lvl]; print(f"   {tag:16s} {lvl:6s} ROC {x['roc']:.3f} PR {x['pr']:.3f} MCC {x['mcc']:.3f} F1 {x['f1']:.3f}")
json.dump(out, open(HERE / "official_labels.json", "w"), indent=1, default=float)
