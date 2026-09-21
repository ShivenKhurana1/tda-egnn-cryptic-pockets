#!/usr/bin/env python3
"""Phase 15: external validation of the localization-trained model on
COACH420/HOLO4K, using the model's own trained centroid (no post-hoc heuristic)."""
import json, pickle
from pathlib import Path

c12 = pickle.load(open("/tmp/p12_cells.pkl", "rb"))
c14 = pickle.load(open("/tmp/p14_cells.pkl", "rb"))
s12 = lambda i: c12[i]["source"]
s14 = lambda i: c14[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 15 — External Validation: Localization-Trained Model on COACH420/HOLO4K

Phase 14 showed the auxiliary localization loss beats the best prior method
(+0.0478 DCC, "aux-loss model + own centroid" vs "baseline + Phase 13's tuned
heuristic") on our own held-out split, with negligible cost to classification
(-0.0026 PR-AUC). That result earned this step: retrain the same architecture
as a deployment ensemble on all 2,056 training chains, then evaluate DCC/DCA on
the real external benchmark — COACH420 and HOLO4K, leakage-checked against our
training set exactly as in Phase 12 — using the model's own trained centroid
output directly. No clustering heuristic anywhere in this notebook.

**Reference points, both already independently verified:**

| | COACH420 (n≈174) | HOLO4K (n≈967) |
|---|---|---|
| Phase 12 model (classification-only + Phase-12-heuristic) DCC | 0.368 | 0.403 |
| P2Rank, same-run, same clean subset | 0.471 | 0.551 |

Phase 13's tuned heuristic improved the Phase-12 model to a *pooled* (not
per-benchmark) DCC of 0.435 on its own tune/test split of this same clean set.
This notebook's number is directly comparable to the Phase 12 row above (same
benchmark, same leakage filter, same scoring code) and informative against
P2Rank, with the same caveat as Phase 12: P2Rank is re-run by us on the same
structures (controlled); nothing here claims to beat literature EquiPocket
numbers, for the reason established in Phase 12 (the clean subset is
measurably easier than the full published benchmark).

**For multi-chain HOLO4K structures**, each chain's summed classification
probability is used as a per-chain confidence score, and the chain with the
highest confidence contributes its own-centroid prediction for that structure
— the same "most likely to contain the pocket" logic as Phase 12/13, just
applied to pick among trained centroids instead of clustered ones.
''')

md(r'''## Cell 1 — Setup''')
code(r'''
!pip install -q torch-geometric==2.6.1 gudhi==3.10.1 biopython==1.84 transformers==4.46.3
!apt-get -qq update > /dev/null 2>&1 && apt-get -qq install -y default-jre-headless > /dev/null 2>&1
''')

code(r'''
''' + s12(3) + r'''

# Phase12's setup cell (reused above) doesn't import these -- Phase14's reused
# model/training code (below) needs them. Add explicitly rather than relying on
# import order across two differently-assembled notebooks.
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

# Phase12's setup names this train_dataset; Phase14's reused true-center-recovery
# code (next cell) expects `dataset` -- alias so both names resolve consistently.
dataset = train_dataset
''')

md(r'''
## Cell 2 — Recover true ligand centers for the TRAINING set (Phase 14's logic)

Needed to train the auxiliary loss. This re-fetches and re-parses our own
2,056 training chains' structures — a separate PDB-ID set from the benchmark,
fetched and cached independently.
''')

_p14_true_center = s14(5)
_cut = _p14_true_center.index('dataset = [d for d in dataset if hasattr(d, "true_center")]')
_true_center_core = _p14_true_center[:_cut].rstrip()

code(r'''
''' + _true_center_core + r'''

dataset = [d for d in dataset if hasattr(d, "true_center")]
train_dataset = dataset   # keep both names in sync for cells reused from Phase 12
n_orig = len(blob10["dataset"])
print(f"training chains with recovered true centers: {len(dataset):,}/{n_orig:,}")

# Deployment split: no held-out test needed here (the external benchmark IS
# the test) -- small internal val slice purely for early stopping, matching
# the Phase 12 deployment-training pattern.
rng = np.random.RandomState(42)
perm = rng.permutation(len(dataset))
n_val = max(1, int(0.08 * len(perm)))
splits = {"val": [dataset[i] for i in perm[:n_val]], "train": [dataset[i] for i in perm[n_val:]]}
n_tr, n_va = len(splits["train"]), len(splits["val"])
print(f"deployment training: {n_tr} train / {n_va} val")
''')

md(r'''## Cell 3 — Model (identical to Phase 14)''')
code(s14(7))

md(r'''## Cell 4 — Equivariance check (re-verified on this run, same discipline as always)''')
code(s14(9))

md(r'''## Cell 5 — Training machinery (identical to Phase 14)''')
code(s14(11))

md(r'''## Cell 6 — Collation sanity check before training anything''')
_dry = s14(13)
_dry = _dry.replace('_probe_items = splits["train"][:8]', '_probe_items = splits["train"][:8]')
code(_dry)

md(r'''
## Cell 7 — Train the deployment ensemble (3 seeds, lambda=0.1 from Phase 14's sweep)

`lambda` is not re-tuned here — it was already selected on a held-out val set
in Phase 14 and re-tuning it again on this run's val split would be a second,
unnecessary bite at the same apple.
''')

code(r'''
BEST_LAMBDA = 0.1   # selected in Phase 14 via val DCC sweep; reused, not re-tuned

WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)
ensemble = []
t0 = time.time()
for seed in (0,1,2):
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    ensemble.append(m)
    torch.save(m.state_dict(), WORK/f"cache/loc_deploy_seed{seed}.pt")
    val_dcc = v.get("dcc", float("nan"))
    print(f"  seed {seed}: val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  "
          f"[{(time.time()-t0)/60:.1f}m]")
print(f"deployment ensemble of {len(ensemble)} trained on {len(dataset):,} chains")
''')

md(r'''## Cell 8 — Fetch benchmark structures (identical to Phase 12)''')
code(s12(5))

md(r'''## Cell 9 — Ground-truth ligand extraction + leakage check (identical to Phase 12)''')
code(s12(7))
code(s12(8))

md(r'''## Cell 10 — Featurize the clean benchmark chains, AND cache raw features this time

Phase 12 only cached derived outputs (probabilities); Phase 13 needed to redo
the ~170-minute featurization to test new heuristics. This time the raw
per-chain tensors are saved directly, so any future architecture iteration on
this same benchmark set is free.
''')
code(s12(12))

code(r'''
# Cache raw features (x, pos, edge_index, edge_attr) -- NOT just derived probs --
# so a future model change can be evaluated without repeating ~170 min of
# fetch+parse+ESM+topology.
import pickle as _pkl
raw_cache = {k: {kk: (vv.numpy() if torch.is_tensor(vv) else vv) for kk,vv in v.items()}
            for k,v in built.items()}
with open(WORK/"cache/phase15_raw_features.pkl", "wb") as fh:
    _pkl.dump(raw_cache, fh)
print(f"cached raw features for {len(raw_cache)} chains -> "
      f"{(WORK/'cache/phase15_raw_features.pkl').stat().st_size/1e6:.0f} MB")
''')

md(r'''## Cell 11 — Inference with the trained localization head, no heuristic''')
code(r'''
from collections import defaultdict

def dcc_dca(pred_center, ligs):
    if pred_center is None or not ligs: return None, None
    best_dcc, best_dca = np.inf, np.inf
    for lig_atoms in ligs:
        if len(lig_atoms) == 0: continue
        com = lig_atoms.mean(axis=0)
        best_dcc = min(best_dcc, float(np.linalg.norm(pred_center - com)))
        best_dca = min(best_dca, float(np.linalg.norm(lig_atoms - pred_center, axis=1).min()))
    return (best_dcc if np.isfinite(best_dcc) else None,
            best_dca if np.isfinite(best_dca) else None)


mu, sd = blob6b["topo_mean"], blob6b["topo_std"]

@torch.no_grad()
def predict_chain(entry):
    x = entry["x"].clone()
    x[:, N_PRE:] = (x[:, N_PRE:] - torch.from_numpy(mu)) / torch.from_numpy(sd)
    d = type("D", (), {})()
    d.x=x.to(DEVICE); d.pos=entry["pos"].to(DEVICE)
    d.edge_index=entry["edge_index"].to(DEVICE); d.edge_attr=entry["edge_attr"].to(DEVICE)
    d.batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
    seed_logits, seed_centers, seed_offsets = [], [], []
    for m in ensemble:
        lg, pc = m(d)
        seed_logits.append(torch.sigmoid(lg).cpu().numpy())
        seed_centers.append(pc.cpu().numpy()[0])                 # (3,) -- single graph
    probs = np.mean(seed_logits, axis=0)
    center = np.mean(seed_centers, axis=0)                       # ensemble-average predicted point
    return probs, center


t0 = time.time()
by_structure = defaultdict(list)
for (src_,pid,cid), entry in built.items():
    by_structure[(src_,pid)].append((cid, entry))

predictions = {}
for (src_, pid), chain_list in by_structure.items():
    ligs = chain_list[0][1]["ligs"]
    best_conf, best_center = -np.inf, None
    for cid, entry in chain_list:
        probs, center = predict_chain(entry)
        conf = float(probs.sum())                                # per-chain "pocket-likelihood"
        if conf > best_conf:
            best_conf, best_center = conf, center
    predictions[(src_,pid)] = (best_center, ligs)

print(f"predicted centers for {len(predictions)} structures in {(time.time()-t0)/60:.1f} min "
      f"(own-centroid, no clustering heuristic)")
''')

md(r'''## Cell 12 — Score, compare against Phase 12/13 and P2Rank''')
code(r'''
PHASE12_REFERENCE = {
    "coach420": {"dcc": 0.3678, "dca": 0.6782},
    "holo4k":   {"dcc": 0.4033, "dca": 0.7053},
}
P2RANK_REFERENCE = {
    "coach420": {"dcc": 0.4706, "dca": 0.6941},
    "holo4k":   {"dcc": 0.5509, "dca": 0.7796},
}
PHASE13_POOLED_TUNED_DCC = 0.4355   # pooled coach420+holo4k, different tune/test split

results = {"coach420": [], "holo4k": []}
for (src_, pid), (center, ligs) in predictions.items():
    dcc, dca = dcc_dca(center, ligs)
    if dcc is not None: results[src_].append((dcc, dca))

print(f"{'benchmark':<10}{'n':>6}{'DCC succ':>10}{'DCA succ':>10}")
print("-"*36)
summary = {}
all_dcc = []
for src_ in ("coach420","holo4k"):
    vals = results[src_]
    if not vals: print(f"{src_:<10}{0:>6}{'--':>10}{'--':>10}"); continue
    dcc_s = np.mean([1.0 if d<4.0 else 0.0 for d,_ in vals])
    dca_s = np.mean([1.0 if a<4.0 else 0.0 for _,a in vals])
    summary[src_] = {"n": len(vals), "dcc": float(dcc_s), "dca": float(dca_s)}
    all_dcc.extend([1.0 if d<4.0 else 0.0 for d,_ in vals])
    print(f"{src_:<10}{len(vals):>6}{dcc_s:>10.3f}{dca_s:>10.3f}")

pooled_dcc = float(np.mean(all_dcc)) if all_dcc else float("nan")

print(f"\n{'':30}{'DCC':>8}")
print("-"*40)
for src_ in ("coach420","holo4k"):
    if src_ not in summary: continue
    p12v = PHASE12_REFERENCE[src_]["dcc"]; p2v = P2RANK_REFERENCE[src_]["dcc"]
    ours = summary[src_]["dcc"]
    print(f"  {src_:<10} ours {ours:.4f}  |  Phase12(heuristic) {p12v:.4f}  ({ours-p12v:+.4f})  |  "
          f"P2Rank {p2v:.4f}  ({ours-p2v:+.4f})")

print(f"\n  pooled DCC (both benchmarks): {pooled_dcc:.4f}  "
      f"vs Phase 13 tuned-heuristic pooled: {PHASE13_POOLED_TUNED_DCC:.4f}  "
      f"({pooled_dcc-PHASE13_POOLED_TUNED_DCC:+.4f})")

json.dump({"per_benchmark": summary, "pooled_dcc": pooled_dcc,
           "phase12_reference": PHASE12_REFERENCE, "p2rank_reference": P2RANK_REFERENCE,
           "phase13_pooled_tuned_dcc": PHASE13_POOLED_TUNED_DCC},
          open(WORK/"phase15_summary.json", "w"), indent=2)
print("\nsaved -> phase15_summary.json")
''')

md(r'''
---

## Reading this

Three comparisons, in order of how much they can be trusted:

1. **vs Phase 12's classification-only model** (same benchmark, same leakage
   filter, same scoring code, only the model differs) — the cleanest
   before/after in this notebook. A clear win here means the localization
   training signal generalizes beyond our own held-out split to a genuinely
   external, harder benchmark.
2. **vs P2Rank** (same-run, same clean subset, controlled) — the number that
   answers "is this competitive with the field's standard tool", the same
   question Phase 12 first asked and answered "no" to.
3. **vs Phase 13's pooled tuned-heuristic number** — informative but on a
   different tune/test split of the same clean set, so treat this one as
   directional.

If this closes most or all of the P2Rank gap, that is a real, externally
validated result: a trained localization objective beats both a classification
-only model and a purpose-tuned post-hoc heuristic, on the field's own
benchmark. If it falls short of Phase 12's baseline despite winning cleanly on
our own held-out split in Phase 14, that is itself informative — it would mean
the auxiliary loss overfit to the training distribution's ligand-placement
patterns in a way that doesn't transfer, which is a different and equally
useful thing to know before calling this approach finished.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase15_localized_sota_validation.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
