#!/usr/bin/env python3
"""Phase 23: WHICH part of topology drives the localization benefit?
Decomposes the topology block into H0 (components) / H1 (loops) / H2 (voids)
/ burial, isolates each (all else zeroed), trains 3 seeds per arm, and scores
against the already-cached full/no_topo baselines from Phase 15/18/22."""
import json, pickle
from pathlib import Path

c14 = pickle.load(open("/tmp/p14_cells.pkl", "rb"))
s14 = lambda i: c14[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 23 — Which Part of Topology Drives the Localization Benefit?

Phase 22 confirmed (8/8 seed-paired wins, p~0.008) that persistent homology
improves pocket-CENTER localization, not just classification. That is a
"that" finding. This notebook asks "why" — which specific piece of the
topology block is doing the work.

The topology feature block decomposes into four pieces with distinct
mathematical meanings:

- **H0** (connected components) — how fragmented / clustered the local point
  cloud is.
- **H1** (loops / tunnels) — 1-dimensional cycles, e.g. access channels.
- **H2** (enclosed voids) — 2-dimensional cavities that trap volume. This is
  the mathematically direct description of what a binding pocket *is*.
- **burial** (`nbhd_n_atoms`, `nbhd_gyration`) — density/depth, already shown
  in Phase 2b to be a real but separate signal from topology, for
  classification. Never tested for localization.

**Hypothesis, stated before running anything (so this is a real test, not a
post-hoc story):** if the "cavity" intuition for why topology helps is
correct, H2 alone should recover most of `full`'s localization advantage over
`no_topo`. If instead H1 or burial alone does the same, that points to
pockets being found via access channels or simple depth, not cavity shape —
a different and equally interesting mechanistic answer.

**Scope:** 3 seeds per new arm (H0-only, H1-only, H2-only, burial-only) — a
first-pass screen, same discipline as Phase 18's original single-pass topology
test before Phase 22 validated it further with 8 seeds. If a clear winner
emerges here, it earns the same multi-seed treatment next.
''')

md(r'''## Cell 1 — Setup, rebuild training data (Phase 14 logic, reused)''')
code(r'''
!pip install -q torch-geometric==2.6.1 biopython==1.84
''')

code("import pickle\n\n" + s14(3))

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
print(f"deployment training: {n_tr} train / {n_va} val  (matches Phase 15/18/22's setup)")
''')

md(r'''## Cell 2 — Model (identical to Phase 14/15/18/22)''')
code(s14(7))

md(r'''## Cell 3 — Training machinery (identical to Phase 14)''')
code(s14(11))

md(r'''
## Cell 4 — Build H0 / H1 / H2 / burial column masks

Reconstructs the same 35-name -> 34-kept mapping used in every phase that
touches this topology block, then further splits by homology dimension.
Matching by NAME (not a hardcoded index) so this self-verifies via the assert
rather than silently drifting.
''')
code(r'''
blob6b = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)
TOPO_NAMES = blob6b["topo_names"]
BURIAL_NAMES = set(blob6b["burial_names"])
print(f"topo columns: {len(TOPO_NAMES)} -- {TOPO_NAMES[:3]} ... {TOPO_NAMES[-2:]}")

MASKS = {
    "H0_only":     np.array([n.startswith("H0_") for n in TOPO_NAMES]),
    "H1_only":     np.array([n.startswith("H1_") for n in TOPO_NAMES]),
    "H2_only":     np.array([n.startswith("H2_") for n in TOPO_NAMES]),
    "burial_only": np.array([n in BURIAL_NAMES for n in TOPO_NAMES]),
}
for name, mask in MASKS.items():
    kept = [n for n,k in zip(TOPO_NAMES, mask) if k]
    print(f"  {name:<14} {mask.sum()} columns: {kept}")
assert sum(m.sum() for m in MASKS.values()) == len(TOPO_NAMES), \
    "masks should exactly partition the topo block (no overlap, no gaps)"
print("masks partition the full topo block exactly -- no overlap, no gaps")
''')

md(r'''## Cell 5 — Train 3 seeds per arm (H0/H1/H2/burial, all else zeroed)''')
code(r'''
import time
WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)

ORIG_X = [d.x.clone() for d in dataset]
def apply_mask(mask):
    for d, x0 in zip(dataset, ORIG_X):
        t = x0[:, N_PRE:].clone()
        keep = torch.from_numpy(mask)
        t[:, ~keep] = 0.0
        d.x = torch.cat([x0[:, :N_PRE], t], dim=1)
def restore():
    for d, x0 in zip(dataset, ORIG_X):
        d.x = x0.clone()

BEST_LAMBDA = 0.1
arm_models = {name: {} for name in MASKS}
t0 = time.time()
for arm_name, mask in MASKS.items():
    apply_mask(mask)
    for seed in range(8):
        m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
        v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
        arm_models[arm_name][seed] = m
        torch.save(m.state_dict(), WORK/f"cache/{arm_name}_seed{seed}.pt")
        print(f"  [{arm_name}] seed {seed}: val PR {val_pr:.4f}  "
              f"val DCC {v.get('dcc',float('nan')):.4f}  [{(time.time()-t0)/60:.1f}m]")
    restore()

print(f"\ntrained {sum(len(v) for v in arm_models.values())} models across {len(MASKS)} arms")
''')

md(r'''## Cell 6 — Also load Phase 15/18's full and no_topo checkpoints (no retraining)''')
code(r'''
def load_existing(prefix, seeds):
    out = {}
    for seed in seeds:
        cands = []
        for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
            if r.exists(): cands.extend(r.rglob(f"{prefix}{seed}.pt"))
        if not cands: continue
        state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
        m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE); m.load_state_dict(state); m.eval()
        out[seed] = m
    return out

arm_models["full"] = load_existing("loc_deploy_seed", range(8))
arm_models["no_topo"] = load_existing("notopo_deploy_seed", range(8))
print(f"loaded {len(arm_models['full'])} 'full' + {len(arm_models['no_topo'])} 'no_topo' "
      f"reference checkpoints (up to 8 seeds each, from Phase 21/22's already-trained pool)")
''')

md(r'''## Cell 7 — Load Phase 15's cached benchmark features (no re-fetch)''')
code(r'''
with open(locate("phase15_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)
print(f"loaded cached features for {len(raw_cache):,} chains (COACH420 + HOLO4K)")
mu_b, sd_b = blob6b["topo_mean"], blob6b["topo_std"]
''')

md(r'''## Cell 8 — Score every arm on the cached benchmark''')
code(r'''
from collections import defaultdict

@torch.no_grad()
def predict(entry, model_dict, mask=None):
    x = torch.from_numpy(entry["x"]).clone()
    if mask is None:
        x[:, N_PRE:] = 0.0                              # no_topo
    else:
        t = x[:, N_PRE:].clone()
        keep = torch.from_numpy(mask) if mask is not True else torch.ones(t.size(1), dtype=torch.bool)
        t = (t - torch.from_numpy(mu_b)) / torch.from_numpy(sd_b)   # standardize first
        t[:, ~keep] = 0.0    # THEN zero masked columns -- matches training-time apply_mask,
                              # which zeroed an ALREADY-standardized d.x (see Cell 5)
        x[:, N_PRE:] = t
    d = type("D", (), {})()
    d.x=x.to(DEVICE); d.pos=torch.from_numpy(entry["pos"]).to(DEVICE)
    d.edge_index=torch.from_numpy(entry["edge_index"]).to(DEVICE)
    d.edge_attr=torch.from_numpy(entry["edge_attr"]).to(DEVICE)
    d.batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
    offset = entry["ca"].mean(axis=0)
    seed_logits, seed_centers = [], []
    for m in model_dict.values():
        lg, pc = m(d)
        seed_logits.append(torch.sigmoid(lg).cpu().numpy())
        seed_centers.append(pc.cpu().numpy()[0])
    return float(np.mean(seed_logits, axis=0).sum()), np.mean(seed_centers, axis=0) + offset


def dcc(pred_center, ligs):
    if pred_center is None or not ligs: return None
    best = np.inf
    for lig_atoms in ligs:
        if len(lig_atoms) == 0: continue
        best = min(best, float(np.linalg.norm(pred_center - lig_atoms.mean(axis=0))))
    return best if np.isfinite(best) else None


by_structure = defaultdict(list)
for key, entry in raw_cache.items():
    by_structure[key[:2]].append((key[2], entry))

ARM_MASK_FOR_SCORING = {**{k: v for k, v in MASKS.items()}, "full": True, "no_topo": None}

t0 = time.time()
print(f"{'arm':<14}{'DCC':>10}{'n':>6}")
print("-"*30)
summary = {}
for arm_name, mask in ARM_MASK_FOR_SCORING.items():
    models = arm_models[arm_name]
    hits, n = 0, 0
    for key, chain_list in by_structure.items():
        ligs = raw_cache[(key[0], key[1], chain_list[0][0])]["ligs"]
        best_conf, best_center = -np.inf, None
        for cid, entry in chain_list:
            conf, center = predict(entry, models, mask)
            if conf > best_conf: best_conf, best_center = conf, center
        d_ = dcc(best_center, ligs)
        if d_ is None: continue
        n += 1; hits += int(d_ < 4.0)
    rate = hits / max(n, 1)
    summary[arm_name] = {"dcc": rate, "n": n}
    print(f"{arm_name:<14}{rate:>10.4f}{n:>6}")
print(f"\nscored all {len(ARM_MASK_FOR_SCORING)} arms in {(time.time()-t0)/60:.1f} min")
''')

md(r'''## Cell 9 — Which piece recovers the benefit?''')
code(r'''
base = summary["no_topo"]["dcc"]
full = summary["full"]["dcc"]
total_gain = full - base

print(f"no_topo baseline: {base:.4f}")
print(f"full (all topo):  {full:.4f}  (total gain: {total_gain:+.4f})\n")

print(f"{'arm':<14}{'DCC':>10}{'gain over no_topo':>20}{'% of full gain':>18}")
print("-"*62)
for name in ("H0_only", "H1_only", "H2_only", "burial_only"):
    g = summary[name]["dcc"] - base
    pct = 100 * g / total_gain if total_gain > 1e-9 else float("nan")
    print(f"{name:<14}{summary[name]['dcc']:>10.4f}{g:>+20.4f}{pct:>17.1f}%")

winner = max(("H0_only","H1_only","H2_only","burial_only"), key=lambda n: summary[n]["dcc"])
print(f"\nlargest single-piece recovery (8-seed ensemble): {winner} ({summary[winner]['dcc']:.4f})")

# --- per-seed breakdown for the two closest arms: is the gap real, or noise? ---
print(f"\n{'='*70}\nPER-SEED CHECK: is the top-vs-runner-up gap real, or seed noise?\n{'='*70}")
ranked = sorted(("H0_only","H1_only","H2_only","burial_only"), key=lambda n: -summary[n]["dcc"])
top_name, second_name = ranked[0], ranked[1]

def per_seed_dcc(arm_name):
    models_all = arm_models[arm_name]
    out = []
    for seed in sorted(models_all):
        single = {seed: models_all[seed]}
        hits, n = 0, 0
        for key, chain_list in by_structure.items():
            ligs = raw_cache[(key[0], key[1], chain_list[0][0])]["ligs"]
            best_conf, best_center = -np.inf, None
            for cid, entry in chain_list:
                conf, center = predict(entry, single, ARM_MASK_FOR_SCORING[arm_name])
                if conf > best_conf: best_conf, best_center = conf, center
            d_ = dcc(best_center, ligs)
            if d_ is None: continue
            n += 1; hits += int(d_ < 4.0)
        out.append(hits / max(n, 1))
    return np.array(out)

top_seeds = per_seed_dcc(top_name)
second_seeds = per_seed_dcc(second_name)
print(f"{top_name}: per-seed {np.round(top_seeds,4).tolist()}")
print(f"  mean {top_seeds.mean():.4f}  std {top_seeds.std():.4f}")
print(f"{second_name}: per-seed {np.round(second_seeds,4).tolist()}")
print(f"  mean {second_seeds.mean():.4f}  std {second_seeds.std():.4f}")
diff = top_seeds.mean() - second_seeds.mean()
noise = max(top_seeds.std(), second_seeds.std())
print(f"\ndiff: {diff:+.4f}  vs noise scale {noise:.4f}  ({abs(diff)/max(noise,1e-9):.2f}x)")
verdict = (f"{top_name} CONFIRMED as the leading contributor (beyond noise)" if abs(diff) > 1.5*noise
          else f"{top_name} vs {second_name} gap is WITHIN plausible seed noise -- not confidently resolved")
print(f"VERDICT: {verdict}")

json.dump({"summary": summary, "base": base, "full": full, "total_gain": total_gain,
           "winner": winner, "top_name": top_name, "second_name": second_name,
           "top_seeds": top_seeds.tolist(), "second_seeds": second_seeds.tolist(),
           "verdict": verdict},
          open(WORK/"phase23_summary.json", "w"), indent=2)
print("\nsaved -> phase23_summary.json")
''')

md(r'''
---

## Reading this

This is a **3-seed first pass**, not yet validated at the rigor Phase 22 gave
the headline finding — treat the winner as a hypothesis to confirm with more
seeds, not a final answer, exactly as Phase 18's single-pass result was
treated before Phase 22 checked it properly.

- **If H2 wins clearly**: the cavity-shape intuition is right, and "the model
  benefits from features that directly describe enclosed voids, consistent
  with what a binding pocket physically is" is a mechanistic, defensible claim
  — not just a correlation.
- **If H1 or burial wins instead**: equally interesting, and worth saying
  plainly rather than forcing the tidier H2 story — it would mean the benefit
  comes from access-channel topology or simple depth, not cavity enclosure.
- **If no single piece recovers much of the gain**: the benefit may be
  distributed/synergistic across all three homology dimensions together,
  which is itself a finding (topology's value here isn't reducible to one
  "most important" feature).
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase23_topology_decomposition.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
