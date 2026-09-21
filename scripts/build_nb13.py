#!/usr/bin/env python3
"""Phase 13: properly tuned pocket-center localization (tune/test split, no reuse-as-test)."""
import json, pickle
from pathlib import Path

cells12 = pickle.load(open("/tmp/p12_cells.pkl", "rb"))
def src(i): return cells12[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})
reuse=lambda i:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src(i).strip("\n")})

md(r'''
# Phase 13 — Tuning the Pocket-Center Localization Step

Phase 12 found the model loses to P2Rank on DCC/DCA (pocket-center accuracy)
despite beating it on per-residue PR-AUC (Phase 11). The suspected cause: the
per-residue-probabilities-to-single-point conversion was an untuned heuristic
(fixed thresholds, connected components, unweighted centroid) bolted onto a
model that was never trained for localization — unlike P2Rank, which is
purpose-built for exactly this output.

This notebook does not touch the trained model at all. It reuses the exact
Phase 12 ensemble checkpoints and asks a narrower question: **given the same
per-residue predictions, can a better-designed conversion step close some of
the gap?**

### Discipline: tune on one half, report on the other, never both

Re-tuning a heuristic and reporting its score on the same data it was tuned on
is the exact mistake this project has spent many phases avoiding elsewhere. The
1,149 leakage-clean benchmark structures from Phase 12 are split by a fixed
seed into a **35% tune set** (used only to pick the best heuristic) and a
**65% test set** (scored exactly once, with the heuristic already fixed). The
untuned Phase-12 heuristic is also re-scored on this same test slice, so the
before/after comparison is paired and fair — not "old number on 1,149
structures vs new number on a different 650."

### Why several candidate heuristics, not just re-tuned thresholds

The working hypothesis is that connected-components clustering at a loose
threshold can produce a large, diffuse blob whose centroid drifts away from the
true (small, tight) ligand pocket — a few residues at the "wrong end" of a big
component are enough to fail a strict 4 Å threshold. Two structurally different
families are tried, not just parameter variations on the original idea:

- **Top-K peak**: ignore spatial clustering entirely, take the K
  highest-probability residues, centroid them (weighted or unweighted). Simple,
  and immune to the diffuse-blob failure mode by construction.
- **Connected components, revised**: the Phase 12 approach, but with a
  probability-*weighted* centroid instead of a plain mean — pulling the estimate
  toward the cluster's most confident core rather than treating a fringe residue
  the same as the peak.
''')

md(r'''## Cell 1 — Setup (identical to Phase 12)''')
reuse(2)   # pip installs
reuse(3)   # imports, blob loading, DEVICE

md(r'''## Cell 2 — Fetch benchmark structures (identical to Phase 12)''')
reuse(5)   # parse_ds + fetch

md(r'''## Cell 3 — Ground-truth ligand extraction + leakage check (identical to Phase 12)''')
reuse(7)   # parse_structure, eval_units assembly
reuse(8)   # MMseqs2 leakage check -> clean_units

md(r'''
## Cell 4 — Load the Phase 12 ensemble (skip retraining)

Same three checkpoints Phase 12 trained on all 2,056 chains. Loading them here
instead of retraining saves roughly 9 minutes and, more importantly, guarantees
this is testing a heuristic change against a **fixed** model rather than
introducing fresh training-seed variance into the comparison.
''')

_full_src10 = src(10)
_cutoff = _full_src10.index("from torch_geometric.loader import DataLoader")
_classes_only = _full_src10[:_cutoff].rstrip()   # EGCL, Fus, Net, run_epoch -- NOT the training loop

code(r'''
''' + _classes_only + r'''

def locate_any(fname):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(fname))
    return c

ensemble = []
for seed in (0,1,2):
    cands = locate_any(f"deploy_seed{seed}.pt")
    assert cands, f"deploy_seed{seed}.pt not found -- attach the Phase 12 kernel output"
    state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
    m = Net(N_PRE, TOPO_DIM).to(DEVICE); m.load_state_dict(state); m.eval()
    ensemble.append(m)
print(f"loaded {len(ensemble)} ensemble members from Phase 12 (no retraining)")
''')

md(r'''## Cell 5 — Featurize the clean benchmark chains (identical to Phase 12)''')
reuse(12)   # gudhi/ESM featurizer + main loop building `built`

md(r'''
## Cell 6 — Ensemble inference, cached per chain

Unlike Phase 12, per-residue probabilities (plus the coordinates and ground
truth needed to score them) are saved to disk here — so any *future* heuristic
idea can be tested in seconds instead of paying the ~3-hour featurization cost
again.
''')

code(r'''
mu, sd = blob6b["topo_mean"], blob6b["topo_std"]
N_SEQ_DIH_ESM = N_PRE

@torch.no_grad()
def predict_chain(entry):
    x = entry["x"].clone()
    x[:, N_SEQ_DIH_ESM:] = (x[:, N_SEQ_DIH_ESM:] - torch.from_numpy(mu)) / torch.from_numpy(sd)
    d = type("D", (), {})()
    d.x=x.to(DEVICE); d.pos=entry["pos"].to(DEVICE)
    d.edge_index=entry["edge_index"].to(DEVICE); d.edge_attr=entry["edge_attr"].to(DEVICE)
    probs = torch.stack([torch.sigmoid(m(d)) for m in ensemble], dim=0).mean(0)
    return probs.cpu().numpy()

t0=time.time()
cache = {}   # (src,pid,cid) -> {ca, edge_index(np), probs, ligs}
for key, entry in built.items():
    probs = predict_chain(entry)
    cache[key] = {"ca": entry["ca"], "edge_index": entry["edge_index"].numpy(),
                 "probs": probs, "ligs": entry["ligs"]}
print(f"inference on {len(cache)} chains in {(time.time()-t0)/60:.1f} min")

import pickle as _pkl
with open(WORK/"cache/phase13_per_chain_cache.pkl", "wb") as fh:
    _pkl.dump(cache, fh)
print(f"cached per-chain probabilities -> {WORK}/cache/phase13_per_chain_cache.pkl "
      f"({(WORK/'cache/phase13_per_chain_cache.pkl').stat().st_size/1e6:.0f} MB)")
''')

md(r'''
## Cell 7 — Tune / test split, fixed before any heuristic is tried

Split at the **structure** level (not chain level, so a multi-chain HOLO4K
entry doesn't leak across the split), stratified by benchmark so both COACH420
and HOLO4K are represented in each half.
''')

code(r'''
structures = sorted({(src_, pid) for src_, pid, _cid in cache})
by_bench = defaultdict(list)
for s_, p_ in structures: by_bench[s_].append(p_)

rng = np.random.RandomState(2024)
tune_structs, test_structs = set(), set()
for s_, pids in by_bench.items():
    pids = sorted(pids); rng.shuffle(pids)
    n_tune = max(1, int(0.35 * len(pids)))
    tune_structs.update((s_, p) for p in pids[:n_tune])
    test_structs.update((s_, p) for p in pids[n_tune:])

print(f"tune structures: {len(tune_structs)}  |  test structures: {len(test_structs)}")
for s_ in by_bench:
    nt = sum(1 for x,p in tune_structs if x==s_); ns = sum(1 for x,p in test_structs if x==s_)
    print(f"  {s_:<10} tune {nt:>4}  test {ns:>4}")

def group_by_structure(struct_set):
    out = defaultdict(list)
    for (src_,pid,cid), entry in cache.items():
        if (src_,pid) in struct_set: out[(src_,pid)].append((cid, entry))
    return out

tune_groups = group_by_structure(tune_structs)
test_groups = group_by_structure(test_structs)
''')

md(r'''## Cell 8 — Candidate localization heuristics''')

code(r'''
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


def heuristic_topk(ca, probs, k, weighted):
    if len(probs) == 0: return None, -np.inf
    k = min(k, len(probs))
    top = np.argsort(-probs)[:k]
    if weighted:
        w = probs[top]; w = w / max(w.sum(), 1e-9)
        centroid = (ca[top] * w[:, None]).sum(axis=0)
    else:
        centroid = ca[top].mean(axis=0)
    return centroid, float(probs[top].sum())


def heuristic_cc(ca, probs, edge_index, top_frac, min_prob, weighted):
    n = len(ca)
    thresh = max(min_prob, np.quantile(probs, 1 - top_frac))
    hot = np.where(probs >= thresh)[0]
    if len(hot) == 0: return None, -np.inf
    hot_set = set(hot.tolist())
    adj = defaultdict(list)
    for a, b in zip(edge_index[0], edge_index[1]):
        if a in hot_set and b in hot_set: adj[a].append(b); adj[b].append(a)
    seen = set(); best_centroid, best_score = None, -np.inf
    for start in hot:
        if start in seen: continue
        stack=[start]; comp=[]
        while stack:
            u=stack.pop()
            if u in seen: continue
            seen.add(u); comp.append(u)
            stack.extend(adj[u])
        comp = np.array(comp)
        score = float(probs[comp].sum())
        if score > best_score:
            best_score = score
            if weighted:
                w = probs[comp]; w = w / max(w.sum(), 1e-9)
                best_centroid = (ca[comp] * w[:, None]).sum(axis=0)
            else:
                best_centroid = ca[comp].mean(axis=0)
    return best_centroid, best_score


CANDIDATES = {
    "topk_1_unweighted":  lambda ca,p,ei: heuristic_topk(ca,p,1,False),
    "topk_3_weighted":    lambda ca,p,ei: heuristic_topk(ca,p,3,True),
    "topk_5_weighted":    lambda ca,p,ei: heuristic_topk(ca,p,5,True),
    "topk_10_weighted":   lambda ca,p,ei: heuristic_topk(ca,p,10,True),
    "topk_20_weighted":   lambda ca,p,ei: heuristic_topk(ca,p,20,True),
    "cc_phase12_baseline": lambda ca,p,ei: heuristic_cc(ca,p,ei,0.15,0.3,False),
    "cc_weighted_loose":   lambda ca,p,ei: heuristic_cc(ca,p,ei,0.15,0.3,True),
    "cc_weighted_tight":   lambda ca,p,ei: heuristic_cc(ca,p,ei,0.08,0.5,True),
}
print(f"{len(CANDIDATES)} candidate heuristics")
''')

md(r'''## Cell 9 — Score every candidate on TUNE only, pick the winner''')

code(r'''
def score_group(groups, heuristic_fn):
    dcc_hits, dca_hits, n = 0, 0, 0
    for (src_, pid), chain_list in groups.items():
        all_cands = []
        ligs = chain_list[0][1]["ligs"]
        for cid, entry in chain_list:
            center, score = heuristic_fn(entry["ca"], entry["probs"], entry["edge_index"])
            if center is not None: all_cands.append((center, score))
        if not all_cands: continue
        all_cands.sort(key=lambda c: -c[1])
        dcc, dca = dcc_dca(all_cands[0][0], ligs)
        if dcc is None: continue
        n += 1
        dcc_hits += int(dcc < 4.0); dca_hits += int(dca < 4.0)
    return (dcc_hits/max(n,1), dca_hits/max(n,1), n)

print(f"{'heuristic':<24}{'tune DCC':>10}{'tune DCA':>10}{'n':>6}")
print("-"*50)
tune_scores = {}
for name, fn in CANDIDATES.items():
    dcc, dca, n = score_group(tune_groups, fn)
    tune_scores[name] = (dcc, dca, n)
    print(f"{name:<24}{dcc:>10.3f}{dca:>10.3f}{n:>6}")

best_name = max(tune_scores, key=lambda k: tune_scores[k][0])
print(f"\nselected on TUNE (best DCC): {best_name}  (DCC {tune_scores[best_name][0]:.3f})")
''')

md(r'''
## Cell 10 — Final, one-shot evaluation on TEST

The winning heuristic and the Phase-12 baseline heuristic are both scored here,
**for the first time**, on the test structures neither ever influenced.
''')

code(r'''
baseline_fn = CANDIDATES["cc_phase12_baseline"]
winner_fn = CANDIDATES[best_name]

base_dcc, base_dca, base_n = score_group(test_groups, baseline_fn)
new_dcc, new_dca, new_n = score_group(test_groups, winner_fn)

# P2Rank reference: re-use Phase 12's already-computed numbers where the test
# structures overlap (P2Rank does not depend on our model, so re-running it is
# unnecessary -- report its full-clean-set number as the fixed comparison point).
P2RANK_REF = {
    "coach420": {"dcc": 0.4706, "dca": 0.6941},
    "holo4k":   {"dcc": 0.5509, "dca": 0.7796},
}

print(f"{'':<28}{'DCC':>8}{'DCA':>8}{'n':>6}")
print("-"*50)
print(f"{'Phase 12 baseline (test)':<28}{base_dcc:>8.3f}{base_dca:>8.3f}{base_n:>6}")
print(f"{'tuned: '+best_name:<28}{new_dcc:>8.3f}{new_dca:>8.3f}{new_n:>6}")
print(f"{'P2Rank (Phase 12, full clean set)':<28}{'--':>8}{'--':>8}{'':>6}  "
      f"(coach420 DCC {P2RANK_REF['coach420']['dcc']:.3f} / holo4k DCC {P2RANK_REF['holo4k']['dcc']:.3f})")

print(f"\nimprovement from tuning: DCC {new_dcc-base_dcc:+.3f}  DCA {new_dca-base_dca:+.3f}")
gap_before = base_dcc - np.mean([P2RANK_REF['coach420']['dcc'], P2RANK_REF['holo4k']['dcc']])
gap_after = new_dcc - np.mean([P2RANK_REF['coach420']['dcc'], P2RANK_REF['holo4k']['dcc']])
print(f"gap to P2Rank (rough, mixed benchmarks): before {gap_before:+.3f}  after {gap_after:+.3f}")

json.dump({"tune_scores": {k: list(v) for k,v in tune_scores.items()}, "selected": best_name,
           "test_baseline": {"dcc": base_dcc, "dca": base_dca, "n": base_n},
           "test_tuned": {"dcc": new_dcc, "dca": new_dca, "n": new_n},
           "p2rank_reference": P2RANK_REF},
          open(WORK/"phase13_summary.json", "w"), indent=2)
print("\nsaved -> phase13_summary.json")
''')

md(r'''
---

## Reading this

**The comparison that matters is `Phase 12 baseline (test)` vs `tuned` — both
scored on the identical held-out structures, neither having influenced the
tuning choice.** That isolates exactly what changed: the conversion heuristic,
nothing else.

- **A clear DCC improvement** confirms the diagnosis: the per-residue signal was
  fine, the conversion step was the weak link, and this recovers real ground
  against P2Rank.
- **Little or no improvement** means the problem is deeper than the conversion
  heuristic — most likely that the per-residue probability map itself is not
  precise enough for tight single-point localization, which weighted-centroid
  tricks cannot fix. In that case the honest next step is training a dedicated
  pocket-center head end-to-end (how EquiPocket/VN-EGNN are actually built),
  not further heuristic tweaking.

Either way, `phase13_per_chain_cache.pkl` now holds every benchmark chain's
per-residue probabilities, so testing a ninth or tenth heuristic idea later
costs seconds, not another 3-hour run.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase13_localization_tuning.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
