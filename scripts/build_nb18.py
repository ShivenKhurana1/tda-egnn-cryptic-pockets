#!/usr/bin/env python3
"""Phase 18: does topology improve LOCALIZATION specifically, not just
classification? A genuinely untested question given the literature landscape -
TDA-for-binding papers use classical ML with no trained localization
objective; EquiPocket/VN-EGNN don't use topology at all."""
import json, pickle
from pathlib import Path

c14 = pickle.load(open("/tmp/p14_cells.pkl", "rb"))
s14 = lambda i: c14[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 18 - Does Topology Improve Localization, Not Just Classification?

Every prior topology ablation in this project (Phase 6c, 11) tested per-residue
**classification** (PR-AUC): does knowing about persistent homology help decide
*which residues* are binding sites. That question is now answered - yes,
properly powered, p<1e-5.

This asks a different, untested question: does topology help the model
produce a more accurate **single predicted point** - the trained localization
head added in Phase 14? This is not a re-run of the old ablation on a new
metric; it is a genuinely distinct capability, since Phase 8 already showed
these two heads decouple (training the localization head changed what the
*classification* head found useful, in an unexpected direction). There is no
guarantee topology's contribution to one transfers to the other.

**Why this hasn't been tested by anyone else, as far as the literature check
in this project found:** the persistent-homology-for-protein-binding
literature (Cang & Wei's line of work, PATH/PATH+, persistent Laplacians) uses
classical ML (gradient boosting, SVMs) for **affinity regression**, never a
trained equivariant localization head. EquiPocket and VN-EGNN use equivariant
GNNs for exactly this localization task, but neither uses topological
features at all. This sits in the gap between the two literatures.

### Design: reuse everything expensive, train only what's new

- `full` (topology present): **reuses Phase 15's already-trained checkpoints**
  - no retraining, no risk of introducing a confound from a different training
  run.
- `no_topo` (topology zeroed, same architecture/budget): a **new** 3-seed
  ensemble, trained here.
- Both are scored on **Phase 15's cached benchmark features** - no
  re-fetching, no re-running the ~170-minute featurization. This whole
  experiment costs one training run (~15 min) plus fast inference.
''')

md(r'''## Cell 1 - Setup''')
code(r'''
!pip install -q torch-geometric==2.6.1 biopython==1.84
''')

code("import pickle\n\n" + s14(3))

md(r'''## Cell 2 - Recover true ligand centers for training data (Phase 14 logic, reused)''')
_p14_true_center = s14(5)
_cut = _p14_true_center.index('dataset = [d for d in dataset if hasattr(d, "true_center")]')
_true_center_core = _p14_true_center[:_cut].rstrip()
code(_true_center_core + '''

dataset = [d for d in dataset if hasattr(d, "true_center")]
print(f"training chains with recovered true centers: {len(dataset):,}")

rng = np.random.RandomState(42)
perm = rng.permutation(len(dataset))
n_val = max(1, int(0.08 * len(perm)))
splits = {"val": [dataset[i] for i in perm[:n_val]], "train": [dataset[i] for i in perm[n_val:]]}
n_tr, n_va = len(splits["train"]), len(splits["val"])
print(f"deployment training: {n_tr} train / {n_va} val")
''')

md(r'''## Cell 3 - Model (identical to Phase 14/15/16)''')
code(s14(7))

md(r'''## Cell 4 - Training machinery (identical to Phase 14)''')
code(s14(11))

md(r'''
## Cell 5 - Train the `no_topo` ensemble (3 seeds, lambda=0.1, topology zeroed)

`lambda=0.1` reused from Phase 14's sweep, not re-tuned - this experiment
isolates the topology-vs-no-topology question, and re-tuning lambda separately
for each arm would confound that comparison with a second variable.
''')
code(r'''
import time
WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)

ORIG_X = [d.x.clone() for d in dataset]
def zero_topology():
    for d, x0 in zip(dataset, ORIG_X):
        t = x0[:, N_PRE:].clone(); t.zero_()
        d.x = torch.cat([x0[:, :N_PRE], t], dim=1)
def restore_topology():
    for d, x0 in zip(dataset, ORIG_X):
        d.x = x0.clone()

zero_topology()
BEST_LAMBDA = 0.1
notopo_ensemble = []
t0 = time.time()
for seed in (0,1,2):
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    notopo_ensemble.append(m)
    torch.save(m.state_dict(), WORK/f"cache/notopo_deploy_seed{seed}.pt")
    val_dcc = v.get("dcc", float("nan"))
    print(f"  seed {seed}: val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")
restore_topology()
print(f"no_topo ensemble of {len(notopo_ensemble)} trained on {len(dataset):,} chains (topology zeroed)")
''')

md(r'''
## Cell 6 - Load Phase 15's `full` checkpoints (no retraining) + Phase 15's cached benchmark features
''')
code(r'''
full_ensemble = []
for seed in (0,1,2):
    cands = []
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): cands.extend(r.rglob(f"loc_deploy_seed{seed}.pt"))
    state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
    m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE); m.load_state_dict(state); m.eval()
    full_ensemble.append(m)
print(f"loaded {len(full_ensemble)} 'full' checkpoints from Phase 15 (no retraining)")

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

with open(locate("phase15_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)
blob6b_bench = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)
mu_b, sd_b = blob6b_bench["topo_mean"], blob6b_bench["topo_std"]
print(f"loaded cached features for {len(raw_cache):,} benchmark chains")
''')

md(r'''## Cell 7 - Score both arms on the SAME cached benchmark chains, frame-corrected''')
code(r'''
from collections import defaultdict

@torch.no_grad()
def predict_chain(entry, model_ensemble, zero_topo=False):
    x = torch.from_numpy(entry["x"]).clone()
    if zero_topo:
        x[:, N_PRE:] = 0.0
    else:
        x[:, N_PRE:] = (x[:, N_PRE:] - torch.from_numpy(mu_b)) / torch.from_numpy(sd_b)
    d = type("D", (), {})()
    d.x=x.to(DEVICE); d.pos=torch.from_numpy(entry["pos"]).to(DEVICE)
    d.edge_index=torch.from_numpy(entry["edge_index"]).to(DEVICE)
    d.edge_attr=torch.from_numpy(entry["edge_attr"]).to(DEVICE)
    d.batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
    seed_logits, seed_centers = [], []
    for m in model_ensemble:
        lg, pc = m(d)
        seed_logits.append(torch.sigmoid(lg).cpu().numpy())
        seed_centers.append(pc.cpu().numpy()[0])
    probs = np.mean(seed_logits, axis=0)
    offset = entry["ca"].mean(axis=0)
    center_raw = np.mean(seed_centers, axis=0) + offset
    return probs, center_raw


def dcc_dca(pred_center, ligs):
    best_dcc, best_dca = np.inf, np.inf
    for lig_atoms in ligs:
        if len(lig_atoms) == 0: continue
        com = lig_atoms.mean(axis=0)
        d_ = float(np.linalg.norm(pred_center - com))
        if d_ < best_dcc:
            best_dcc = d_
            best_dca = float(np.linalg.norm(lig_atoms - pred_center, axis=1).min())
    return (best_dcc if np.isfinite(best_dcc) else None, best_dca if np.isfinite(best_dca) else None)


by_structure = defaultdict(list)
for (src_,pid,cid), entry in raw_cache.items():
    by_structure[(src_,pid)].append((cid, entry))

def score_arm(model_ensemble, zero_topo, tag):
    results = {"coach420": [], "holo4k": []}
    for (src_, pid), chain_list in by_structure.items():
        ligs = chain_list[0][1]["ligs"]
        best_conf, best_center = -np.inf, None
        for cid, entry in chain_list:
            probs, center = predict_chain(entry, model_ensemble, zero_topo)
            conf = float(probs.sum())
            if conf > best_conf:
                best_conf, best_center = conf, center
        dcc, dca = dcc_dca(best_center, ligs)
        if dcc is not None: results[src_].append((dcc, dca))

    summary = {}
    all_dcc = []
    for src_ in ("coach420","holo4k"):
        vals = results[src_]
        if not vals: continue
        dcc_s = np.mean([1.0 if d<4.0 else 0.0 for d,_ in vals])
        dca_s = np.mean([1.0 if a<4.0 else 0.0 for _,a in vals])
        summary[src_] = {"n": len(vals), "dcc": float(dcc_s), "dca": float(dca_s)}
        all_dcc.extend([1.0 if d<4.0 else 0.0 for d,_ in vals])
    pooled = float(np.mean(all_dcc)) if all_dcc else float("nan")
    print(f"[{tag}] " + " | ".join(f"{k} DCC={v['dcc']:.4f}(n={v['n']})" for k,v in summary.items())
          + f" | pooled {pooled:.4f}")
    return summary, pooled


full_summary, full_pooled = score_arm(full_ensemble, zero_topo=False, tag="full (with topology)")
notopo_summary, notopo_pooled = score_arm(notopo_ensemble, zero_topo=True, tag="no_topo")
''')

md(r'''## Cell 8 - Verdict''')
code(r'''
delta_pooled = full_pooled - notopo_pooled
print(f"\n{'='*70}")
print(f"DOES TOPOLOGY IMPROVE LOCALIZATION (not just classification)?")
print(f"{'='*70}")
print(f"  full (with topology)  pooled DCC: {full_pooled:.4f}")
print(f"  no_topo                pooled DCC: {notopo_pooled:.4f}")
print(f"  delta: {delta_pooled:+.4f}")
for src_ in ("coach420","holo4k"):
    if src_ in full_summary and src_ in notopo_summary:
        d = full_summary[src_]["dcc"] - notopo_summary[src_]["dcc"]
        print(f"  {src_}: full {full_summary[src_]['dcc']:.4f}  no_topo {notopo_summary[src_]['dcc']:.4f}  "
              f"delta {d:+.4f}")

verdict = ("YES -- topology contributes to localization accuracy specifically, "
          "beyond its already-established classification benefit" if delta_pooled > 0.02
          else "NO CLEAR BENEFIT for localization -- topology's contribution appears "
               "specific to classification (which residues), not localization (where exactly)"
          if abs(delta_pooled) <= 0.02 else
          "topology appears to HURT localization -- worth investigating why, "
          "possibly the loc_head backbone competes differently with topological features")
print(f"\nVERDICT: {verdict}")
print("\n(single run, n~1140 pooled structures, not multi-seed significance tested -- "
      "treat as a first, directional answer to a question nobody else has asked)")

json.dump({"full": {"summary": full_summary, "pooled": full_pooled},
           "no_topo": {"summary": notopo_summary, "pooled": notopo_pooled},
           "delta_pooled": delta_pooled},
          open("/kaggle/working/phase18_summary.json", "w"), indent=2)
print("\nsaved -> phase18_summary.json")
''')

md(r'''
---

## Reading this

This is a **single run**, not the multi-seed significance-tested comparison
that made the classification-side topology claim (Phase 6c/11) trustworthy -
treat the direction as a first answer, not a final one. But directionally:

- **A real gap either way is a genuinely new finding.** Nobody in the surveyed
  literature has asked whether topological features help an equivariant GNN
  localize a pocket center specifically, as distinct from classifying which
  residues are near one.
- **If topology helps localization too**, that strengthens the case for fusing
  it into the shared backbone (as this architecture does) rather than treating
  it as classification-only side information.
- **If it doesn't**, that is itself informative: it would suggest topology
  encodes *what* a pocket looks like (useful for classification) without
  encoding *precisely where its center is* (a geometric property topology, by
  construction, only partially captures) - a real, defensible, specific claim
  about what this kind of feature is and isn't good for.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase18_topology_for_localization.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
