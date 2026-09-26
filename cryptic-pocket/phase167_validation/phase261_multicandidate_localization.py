"""Phase 261 -- can we locate hidden pockets better by returning SEVERAL candidate centers instead of one?

Uses only data already on disk: Phase 232's exported per-residue probabilities (8-seed mean of the best arm, geom_tda, Phase 166 hard-split
test set, 185 proteins), the graph coordinates (Phase 159 blob), and the same single experimental pocket center used everywhere else in the
cryptic-pocket line (phase168A_corrected_centers.csv, graph frame). P2Rank's top-1 / best-of-3 errors on the same chains come from Phase 226.

Candidate-generation methods are fixed in advance, with no tuning on these 185 proteins (radii are round numbers):
  sharp   : the existing probability-sharpened centroid, exponent 10 (one candidate)              [current method]
  peel    : repeat 3 times: take the highest-probability remaining residue, gather all remaining residues within 10 A of it, emit their
            probability-weighted centroid as a candidate, remove them
  cluster : top 10% of residues by probability; link residues within 8 A; connected components ranked by summed probability; each
            component's probability-weighted centroid is a candidate (up to 3)
Metric: best-of-k distance to the experimental center, DCC at 4/6/8/10 A. Compared with P2Rank top-1 / best-of-3 on the SAME chains.

Run from cryptic-pocket/:  python3 phase167_validation/phase261_multicandidate_localization.py
"""
import sys
sys.path.insert(0, "/tmp")
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

BASE = "phase167_validation"
ARM = "geom_tda"

# --- load graph coordinates without torch_geometric (unpickle into stub classes) ---
import types
class _Any:
    def __init__(self, *a, **k): pass
    def __setstate__(self, st): self.__dict__.update(st if isinstance(st, dict) else {"_st": st})
for n in ["torch_geometric", "torch_geometric.data", "torch_geometric.data.data", "torch_geometric.data.storage", "torch_geometric.data.batch"]:
    m = types.ModuleType(n); m.__getattr__ = (lambda name: type(name, (_Any,), {})); sys.modules[n] = m
blob = torch.load("phase159_input/phase159_cryptobench_anmmsf_graphs.pt", map_location="cpu", weights_only=False)
pos = {(str(d._store._mapping["pdb_id"]).lower(), str(d._store._mapping["chain_id"])): d._store._mapping["pos"].numpy() for d in blob["dataset"]}
del blob

probs = torch.load("phase232/results/phase232_test_probabilities.pt", map_location="cpu", weights_only=False)
centers = pd.read_csv(f"{BASE}/phase168A_corrected_centers.csv")
centers["key"] = list(zip(centers.apo_pdb_id.str.lower(), centers.apo_chain.astype(str)))
truth = {k: np.array([r.true_center_x, r.true_center_y, r.true_center_z]) for k, r in zip(centers.key, centers.itertuples())}
p2df = pd.read_csv(f"{BASE}/phase226_p2rank_localization.csv")
p2 = {(r.pdb, str(r.chain)): (r.p2_top1_err, r.p2_best3_err) for r in p2df.itertuples()}

def sharp(x, p, g=10.0):
    w = p ** g; w = w / w.sum(); return [(w[:, None] * x).sum(0)]

def peel(x, p, k=3, R=10.0):
    x, p = x.copy(), p.copy(); out = []
    for _ in range(k):
        if len(p) == 0: break
        i = int(np.argmax(p)); near = np.linalg.norm(x - x[i], axis=1) <= R
        w = p[near]; out.append((w[:, None] * x[near]).sum(0) / w.sum())
        x, p = x[~near], p[~near]
    return out

def cluster(x, p, k=3, top=0.10, link=8.0):
    n = max(3, int(round(top * len(p)))); idx = np.argsort(-p)[:n]; xs, ps = x[idx], p[idx]
    pairs = cKDTree(xs).query_pairs(link, output_type="ndarray")
    a = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else csr_matrix((n, n))
    _, lab = connected_components(a, directed=False)
    comps = sorted(range(lab.max() + 1), key=lambda c: -ps[lab == c].sum())[:k]
    return [((ps[lab == c])[:, None] * xs[lab == c]).sum(0) / ps[lab == c].sum() for c in comps]

METHODS = {"sharp": sharp, "peel": peel, "cluster": cluster}
rows = []
for key, t in truth.items():
    if key not in pos or (ARM, 0) not in probs or key not in probs[(ARM, 0)]: continue
    x = pos[key]; p = np.mean([probs[(ARM, s)][key] for s in range(8)], axis=0)
    rec = {"key": key}
    for name, fn in METHODS.items():
        cands = fn(x, p)
        d = [float(np.linalg.norm(c - t)) for c in cands]
        rec[f"{name}_top1"] = d[0]; rec[f"{name}_best3"] = min(d)
    rec["p2_top1"], rec["p2_best3"] = p2.get(key, (np.nan, np.nan))
    rows.append(rec)
df = pd.DataFrame(rows); df.to_csv(f"{BASE}/phase261_localization_errors.csv", index=False)
both = df.dropna(subset=["p2_top1"])
print(f"proteins: {len(df)} evaluated; {len(both)} where P2Rank returned a pocket on the target chain (like-for-like subset)\n")
def line(name, col, d):
    e = d[col].values
    print(f"  {name:22s} mean {e.mean():6.2f} A  median {np.median(e):6.2f} A   DCC4 {np.mean(e<=4):.3f}  DCC6 {np.mean(e<=6):.3f}  DCC8 {np.mean(e<=8):.3f}  DCC10 {np.mean(e<=10):.3f}")
print("LIKE-FOR-LIKE SUBSET (same chains for every row)")
line("P2Rank top-1", "p2_top1", both); line("P2Rank best of 3", "p2_best3", both)
line("model sharp (1 cand.)", "sharp_top1", both)
line("model peel top-1", "peel_top1", both); line("model peel best of 3", "peel_best3", both)
line("model cluster top-1", "cluster_top1", both); line("model cluster best of 3", "cluster_best3", both)
print("\nALL 185 (model only)")
line("model sharp", "sharp_top1", df); line("model peel best of 3", "peel_best3", df); line("model cluster best of 3", "cluster_best3", df)
# paired hit comparison at 4 A and 8 A: model best-of-3 vs P2Rank best-of-3
from scipy.stats import binomtest
for col in ("peel_best3", "cluster_best3"):
    for th in (4, 8):
        a = both[col].values <= th; b = both["p2_best3"].values <= th
        n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
        pv = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        print(f"  {col:13s} vs P2Rank best-of-3 @ {th} A: model-only hits {n10}, P2Rank-only hits {n01}, sign-test p={pv:.3f}")
