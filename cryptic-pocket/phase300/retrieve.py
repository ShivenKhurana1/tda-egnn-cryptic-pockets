"""Phase 300: structural template retrieval. Do cryptic sites recur across folds, so that a protein's pocket can be read off structurally
similar proteins with known pockets?

Foldseek (3Di+AA alignment, exhaustive) searches every CryptoBench apo chain against every other. A query may only use templates it could
legitimately see: CV chains in fold f use chains from the other three training folds, test chains use the 760 training chains. For each
query residue, template score = max over the top-K hits of (hit TM-score normalised by query length) x (aligned template residue is an
official positive). Evaluation:
  1. template score alone (per-protein AUC/AP, P@20) on CV and test
  2. stacked with the model: logistic regression on [logit p_model, template max, template weighted mean, best hit TM, residue covered],
     fit fold-wise on CV (model = Phase 298 stack2_saprot CV predictions, 8-seed mean) and, for test, fit on all 760 CV chains then applied
     to the Phase 299 stack3 8-seed ensemble. Paired bootstrap over proteins; strictly cryptic 16 reported separately.
usage: python3 retrieve.py"""
import json, subprocess
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon
H = Path(__file__).resolve().parent; CP = H.parent; FS = "/tmp/fstest/foldseek/bin/foldseek"; W = Path("/tmp/fsdb")
INFO = json.load(open(H / "chains.json")); TOPK = 20
m8 = W / "hits.m8"
if not m8.exists():
    subprocess.run([FS, "easy-search", str(W / "in"), str(W / "in"), str(m8), str(W / "tmp"), "--exhaustive-search", "1", "-e", "10",
                    "--format-output", "query,target,evalue,prob,qtmscore,qstart,qend,tstart,tend,qaln,taln", "-v", "1", "--threads", "8"], check=True)
hits = {}
for l in open(m8):
    q, t, ev, prob, qtm, qs, qe, ts, te, qa, ta = l.rstrip("\n").split("\t")
    q, t = q.replace(".pdb", ""), t.replace(".pdb", "")
    if q == t or q not in INFO or t not in INFO: continue
    hits.setdefault(q, []).append((float(qtm), t, int(qs), int(ts), qa, ta))
print(f"queries with hits {len(hits)}/{len(INFO)}")

def allowed(q, t):
    fq, ft = INFO[q]["fold"], INFO[t]["fold"]
    if ft == "test" or ft is None: return False
    return fq == "test" or fq != ft

def template(q):
    L = len(INFO[q]["y"]); smax = np.zeros(L); ssum = np.zeros(L); wsum = np.zeros(L); best = 0.0
    hs = sorted([h for h in hits.get(q, []) if allowed(q, h[1])], reverse=True)[:TOPK]
    for qtm, t, qs, ts, qa, ta in hs:
        yt = INFO[t]["y"]; i, j = qs - 1, ts - 1; best = max(best, qtm)
        for a, b in zip(qa, ta):
            if a != "-" and b != "-":
                v = qtm * yt[j]; smax[i] = max(smax[i], v); ssum[i] += qtm * yt[j]; wsum[i] += qtm
            if a != "-": i += 1
            if b != "-": j += 1
    return smax, np.divide(ssum, wsum, out=np.zeros(L), where=wsum > 0), best, (wsum > 0).astype(float)

TPL = {q: template(q) for q in INFO}
cv_keys = [q for q in INFO if INFO[q]["fold"] in (0, 1, 2, 3)]; te_keys = [q for q in INFO if INFO[q]["fold"] == "test"]
A298 = torch.load("/tmp/p298_out/phase298_cv_preds.pt", map_location="cpu", weights_only=False)["stack2_saprot"]
A299 = torch.load("/tmp/p299_out/phase299_test_preds.pt", map_location="cpu", weights_only=False)["stack3"]["seeds"]
def key(q): p, c = q.rsplit("_", 1); return (p, c)
PM = {}
for q in cv_keys:
    if key(q) in A298[0]: PM[q] = np.mean([A298[s][key(q)]["p"] for s in range(8)], 0)
for q in te_keys:
    if key(q) in A299[0]["test"]: PM[q] = np.mean([A299[s]["test"][key(q)]["p"] for s in range(8)], 0)
cv_keys = [q for q in cv_keys if q in PM and len(PM[q]) == len(INFO[q]["y"])]; te_keys = [q for q in te_keys if q in PM and len(PM[q]) == len(INFO[q]["y"])]
print(f"CV chains {len(cv_keys)}, test chains {len(te_keys)}")

def feats(q):
    p = np.clip(PM[q], 1e-6, 1 - 1e-6); smax, smean, best, cov = TPL[q]
    return np.c_[np.log(p / (1 - p)), smax, smean, np.full(len(p), best), cov]

def per_protein(keys, score):
    ks = [q for q in keys if 0 < sum(INFO[q]["y"]) < len(INFO[q]["y"])]
    y = {q: np.array(INFO[q]["y"]) for q in ks}
    return {q: dict(auc=roc_auc_score(y[q], score[q]), ap=average_precision_score(y[q], score[q]), p20=y[q][np.argsort(-score[q])[:20]].mean()) for q in ks}

# CV: fold-wise stacking
stack_cv = {}
for f in range(4):
    tr = [q for q in cv_keys if INFO[q]["fold"] != f]; ev = [q for q in cv_keys if INFO[q]["fold"] == f]
    X = np.vstack([feats(q) for q in tr]); Y = np.concatenate([INFO[q]["y"] for q in tr])
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, Y)
    for q in ev: stack_cv[q] = lr.predict_proba(feats(q))[:, 1]
lr_all = LogisticRegression(C=1.0, max_iter=2000).fit(np.vstack([feats(q) for q in cv_keys]), np.concatenate([INFO[q]["y"] for q in cv_keys]))
print("stacker coefficients [logit p, tpl max, tpl mean, best TM, covered]:", np.round(lr_all.coef_[0], 3))
stack_te = {q: lr_all.predict_proba(feats(q))[:, 1] for q in te_keys}
L276 = json.load(open(CP / "phase276" / "phase276a_crypticity_labels.json"))
crypt = [q for q in te_keys if L276.get(q, {}).get("label") == "cryptic"]

rng = np.random.RandomState(0); out = {}
def report(name, keys, base_s, new_s, tpl_s):
    a, b, t = per_protein(keys, base_s), per_protein(keys, new_s), per_protein(keys, tpl_s); ks = sorted(a); out[name] = {}
    print(f"\n{name}: {len(ks)} proteins")
    for m in ("auc", "ap", "p20"):
        x = np.array([a[q][m] for q in ks]); y = np.array([b[q][m] for q in ks]); z = np.array([t[q][m] for q in ks]); d = y - x
        bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(5000)]
        wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        out[name][m] = dict(model=float(x.mean()), template=float(z.mean()), stacked=float(y.mean()), delta=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], w_p=float(wp))
        print(f"  {m:4s} model {x.mean():.4f} | template alone {z.mean():.4f} | stacked {y.mean():.4f}  delta {d.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] W p={wp:.3g}")
tplscore = {q: TPL[q][0] for q in INFO}
report("CV (760 chains, fold-wise stacking)", cv_keys, {q: PM[q] for q in cv_keys}, stack_cv, tplscore)
report("TEST (all)", te_keys, {q: PM[q] for q in te_keys}, stack_te, tplscore)
report("TEST strictly cryptic", crypt, {q: PM[q] for q in crypt}, stack_te, tplscore)
hb = np.array([TPL[q][2] for q in te_keys]); print(f"\nbest-hit TM-score for test queries: median {np.median(hb):.3f}, share >= 0.5: {(hb >= 0.5).mean():.2f}")
json.dump(out, open(H / "retrieve_results.json", "w"), indent=1)
