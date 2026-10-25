"""Phase 302 evaluation (pre-registered in hotspots.py's plan). Model = stack3 (ESM2-3B + 650M + ProtT5 + SaProt): Phase 301 in-kernel CV
predictions (8-seed mean) for the 760 CV chains, Phase 299 test ensemble (with its 60 val chains) for test.
  Stage 1: each hotspot feature alone, per-protein AUC, cryptic-enriched vs rest (CV).
  Stage 2 PRIMARY: fold-wise logistic stacking [logit p_model + 11 standardised hotspot features] vs the model alone, per-protein AUC on
    the cryptic-enriched CV chains (Phase 300 consensus apo count <= 414): Wilcoxon over proteins and a 5,000-sample bootstrap CI.
  Control: the same stack with buriedness only (feature 10), to see whether probe energies add beyond cavity geometry.
  Secondary: stacker fit on the test-time model's own 60 val chains, applied to test (all, strictly cryptic 16).
usage: python3 evaluate.py"""
import json
from pathlib import Path
import numpy as np, torch
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
H = Path(__file__).resolve().parent; CP = H.parent
INFO = json.load(open(CP / "phase300" / "chains.json")); CC = json.load(open(CP / "phase300" / "crypticity_counts.json"))
FZ = np.load(H / "hotspot_feats.npz"); FT = {k: FZ[k] for k in FZ.files}
NAMES = ["site4_strength", "site4_types", "site8_strength", "site8_types", "dist_strong_site", "inv_rank_nearest", "Emin_C", "Emin_O", "Emin_N", "Emin_AR", "max_buried"]
SIGN = [1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1]
def kk(q): p, c = q.rsplit("_", 1); return (p, c)
A301 = torch.load("/tmp/p301_out/phase301_cv_preds.pt", map_location="cpu", weights_only=False)["stack3"]
A299 = torch.load("/tmp/p299_out/phase299_test_preds.pt", map_location="cpu", weights_only=False)["stack3"]["seeds"]
PM = {}
for q in INFO:
    if kk(q) in A301[0]: PM[q] = np.mean([A301[s][kk(q)]["p"] for s in range(8)], 0)
    elif kk(q) in A299[0]["test"]: PM[q] = np.mean([A299[s]["test"][kk(q)]["p"] for s in range(8)], 0)
VAL = {q: np.mean([A299[s]["val"][kk(q)]["p"] for s in range(8)], 0) for q in INFO if kk(q) in A299[0]["val"]}
ok = lambda q, P: q in FT and q in P and len(P[q]) == len(INFO[q]["y"]) == len(FT[q]) and 0 < sum(INFO[q]["y"]) < len(INFO[q]["y"])
CV = [q for q in INFO if INFO[q]["fold"] in (0, 1, 2, 3) and ok(q, PM)]; TE = [q for q in INFO if INFO[q]["fold"] == "test" and ok(q, PM)]
VA = [q for q in VAL if ok(q, VAL)]
ENR = [q for q in CV if q in CC and CC[q]["apo"] <= 414]; REST = [q for q in CV if q in CC and CC[q]["apo"] > 414]
L276 = json.load(open(CP / "phase276" / "phase276a_crypticity_labels.json")); CRY = [q for q in TE if L276.get(q, {}).get("label") == "cryptic"]
print(f"CV {len(CV)} (cryptic-enriched {len(ENR)}, rest {len(REST)}); test {len(TE)} (strictly cryptic {len(CRY)}); val {len(VA)}")
y = lambda q: np.array(INFO[q]["y"])
print("\nStage 1: feature alone, mean per-protein AUC   cryptic-enriched | rest")
for c, (n, sg) in enumerate(zip(NAMES, SIGN)):
    a = np.mean([roc_auc_score(y(q), sg * FT[q][:, c]) for q in ENR]); b = np.mean([roc_auc_score(y(q), sg * FT[q][:, c]) for q in REST])
    print(f"  {n:18s} {a:.3f} | {b:.3f}")
print(f"  {'model (stack3)':18s} {np.mean([roc_auc_score(y(q), PM[q]) for q in ENR]):.3f} | {np.mean([roc_auc_score(y(q), PM[q]) for q in REST]):.3f}")
def X(q, P, cols):
    p = np.clip(P[q], 1e-6, 1 - 1e-6); return np.c_[np.log(p / (1 - p)), FT[q][:, cols]]
ALL = list(range(11)); BUR = [10]
def foldwise(cols):
    out = {}
    for f in range(4):
        tr = [q for q in CV if INFO[q]["fold"] != f]; ev = [q for q in CV if INFO[q]["fold"] == f]
        sc = StandardScaler().fit(np.vstack([X(q, PM, cols) for q in tr]))
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(np.vstack([X(q, PM, cols) for q in tr])), np.concatenate([y(q) for q in tr]))
        for q in ev: out[q] = lr.predict_proba(sc.transform(X(q, PM, cols)))[:, 1]
    return out
rng = np.random.RandomState(0); RES = {}
def compare(name, keys, base, new):
    RES[name] = {}
    for m, fn in (("auc", roc_auc_score), ("ap", average_precision_score), ("p20", lambda t, s: t[np.argsort(-s)[:20]].mean())):
        a = np.array([fn(y(q), base[q]) for q in keys]); b = np.array([fn(y(q), new[q]) for q in keys]); d = b - a
        bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(5000)]; wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        RES[name][m] = dict(model=float(a.mean()), stacked=float(b.mean()), delta=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], w_p=float(wp))
        print(f"  {m:4s} model {a.mean():.4f} -> {b.mean():.4f}  delta {d.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]  W p={wp:.3g}")
S_all, S_bur = foldwise(ALL), foldwise(BUR)
print(f"\nStage 2 PRIMARY: cryptic-enriched CV ({len(ENR)}), stack3 + hotspot features (fold-wise)"); compare("primary_enr_hotspot", ENR, PM, S_all)
print(f"control: cryptic-enriched, stack3 + buriedness only"); compare("control_enr_buried", ENR, PM, S_bur)
print(f"rest ({len(REST)}), stack3 + hotspot features"); compare("rest_hotspot", REST, PM, S_all)
sc = StandardScaler().fit(np.vstack([X(q, VAL, ALL) for q in VA]))
lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(np.vstack([X(q, VAL, ALL) for q in VA])), np.concatenate([y(q) for q in VA]))
ST = {q: lr.predict_proba(sc.transform(X(q, PM, ALL)))[:, 1] for q in TE}
print(f"\nSecondary: TEST, stacker fit on the test-time model's {len(VA)} val chains"); compare("test_all", TE, PM, ST)
print(f"TEST strictly cryptic ({len(CRY)})"); compare("test_cryptic16", CRY, PM, ST)
json.dump(RES, open(H / "evaluate_results.json", "w"), indent=1)
