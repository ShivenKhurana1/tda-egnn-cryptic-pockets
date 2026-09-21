#!/usr/bin/env python3
"""Phase 58: Thread 1, redone at round-3 scale (7,513 clusters, 3.65x
original). At the original scale, the proper 8-seed matched comparison
(Phase26) found H0/H1/H2/burial statistically indistinguishable -- the
topology benefit is diffuse, not attributable to one homology dimension.
This retests that at much larger scale, using the SAME leakage-clean
COACH420 training methodology Phase53/56 already validated (avoids the
original Thread 1's leakage-cleanliness problem: Phase15's cached
benchmark was only guaranteed clean against the ORIGINAL 2,056-chain pool,
not the larger round-3 pool). Reuses Phase56's already-trained 'full' arm
checkpoints directly (seeds 0-2, matched) -- only 5 new arms need training
(no_topo, H0_only, H1_only, H2_only, burial_only), avoiding the
ensemble-size-mismatch confound from the start by using matched 3-seed
counts throughout, unlike the original Phase23's mistake."""
import json, pickle
from pathlib import Path

C53 = pickle.load(open("/tmp/p53_cells.pkl", "rb"))
C23 = pickle.load(open("/tmp/p23_cells.pkl", "rb"))
s53 = lambda i: (C53[i]["source"] if isinstance(C53[i]["source"], str) else "".join(C53[i]["source"]))
s23 = lambda i: (C23[i]["source"] if isinstance(C23[i]["source"], str) else "".join(C23[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 58 — Thread 1 Redone at Round-3 Scale: Does More Data Resolve the Diffuse Topology Signal?

At the original pool scale (2,056 clusters), the proper 8-seed matched
comparison (Phase26) found H0/H1/H2/burial statistically indistinguishable
from each other -- every pairwise gap sat under ~1x the seed noise floor,
nowhere near the 3x+ threshold that made the topology-vs-no-topology
finding (Phase22) credible. **Standing conclusion at that scale: the
topology benefit is diffuse across all four pieces, not concentrated in
one.**

This redoes the test at round-3 scale (7,513 clusters) using the
leakage-clean-training-against-COACH420 methodology Phase53/56 already
validated -- necessary because Thread 1's original benchmark
(`phase15_raw_features.pkl`) was only leakage-verified against the
ORIGINAL 2,056-chain pool, and that guarantee does not automatically
extend to the much larger round-3 training pool. Phase56's already-trained
`full` arm (COACH420-clean, round-3 pool) is reused directly -- only the
five other arms (no_topo, H0/H1/H2/burial-only) need fresh training, at a
MATCHED 3 seeds each from the start, avoiding the ensemble-size-mismatch
confound that undermined the original Phase23.

**Question:** does more training data sharpen the topology-decomposition
signal into something attributable, or does the diffuse-signal finding
hold at scale too?
''')

md(r'''## Cell 1 — Setup (loads Phase 52/51's round-3 scaled blobs, identical to Phase53/56)''')
code(s53(2))
code(s53(3))

md(r'''## Cell 2 — Fetch ALL 420 COACH420 structures (identical to Phase 53/56)''')
code(s53(5))

md(r'''## Cell 3 — Clean the TRAINING side: exclude any chain homologous to COACH420 (identical to Phase 53/56; reproduces the same 7,181 chains)''')
code(s53(7))

md(r'''## Cell 4 — Recover true ligand centers for the cleaned training set (identical to Phase 53/56)''')
code(s53(9))

md(r'''## Cell 5 — Model (identical to every prior localization phase)''')
code(s53(11))

md(r'''## Cell 6 — Training machinery (identical to Phase 14/20/53/56)''')
code(s53(15))

md(r'''## Cell 7 — Build H0 / H1 / H2 / burial column masks (identical to Phase 23, applied to the round-3 topology block already loaded in Cell 1 as blob6b/TOPO_NAMES -- NOT reloading phase6b_graphs.pt, that would silently swap back to the original-scale topology block)''')
code(s23(10).replace(
    'blob6b = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)\n'
    'TOPO_NAMES = blob6b["topo_names"]\n'
    'BURIAL_NAMES = set(blob6b["burial_names"])',
    'BURIAL_NAMES = set(blob6b["burial_names"])   # blob6b/TOPO_NAMES already loaded in Cell 1 (phase51_graphs.pt)'
))

md(r'''## Cell 8 — Load Phase 56's already-trained 'full' arm (seeds 0-2, matched), train the 5 OTHER arms fresh at matched 3 seeds''')
code(r'''
BEST_LAMBDA = 0.1
WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)

arm_models = {"full": {}}
for seed in (0,1,2):
    cands = []
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): cands.extend(r.rglob(f"clean_deploy_scaled3_seed{seed}.pt"))
    assert cands, f"clean_deploy_scaled3_seed{seed}.pt not found -- attach Phase 56's kernel output"
    state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
    m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE); m.load_state_dict(state); m.eval()
    arm_models["full"][seed] = m
print(f"loaded {len(arm_models['full'])} 'full' checkpoints from Phase 56 (no retraining)")

ORIG_X = [d.x.clone() for d in dataset]
def apply_mask(mask):
    for d, x0 in zip(dataset, ORIG_X):
        if mask is None:
            d.x = torch.cat([x0[:, :N_PRE], torch.zeros_like(x0[:, N_PRE:])], dim=1)
        else:
            t = x0[:, N_PRE:].clone()
            keep = torch.from_numpy(mask)
            t[:, ~keep] = 0.0
            d.x = torch.cat([x0[:, :N_PRE], t], dim=1)
def restore():
    for d, x0 in zip(dataset, ORIG_X):
        d.x = x0.clone()

ARMS_TO_TRAIN = {"no_topo": None, **MASKS}   # None = zero the whole topo block
t0 = time.time()
for arm_name, mask in ARMS_TO_TRAIN.items():
    apply_mask(mask)
    arm_models[arm_name] = {}
    for seed in (0,1,2):
        m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
        v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
        arm_models[arm_name][seed] = m
        torch.save(m.state_dict(), WORK/f"cache/{arm_name}_scaled3_seed{seed}.pt")
        print(f"  [{arm_name}] seed {seed}: val PR {val_pr:.4f}  "
              f"val DCC {v.get('dcc',float('nan')):.4f}  [{(time.time()-t0)/60:.1f}m]")
    restore()

print(f"\ntrained {sum(len(v) for k,v in arm_models.items() if k!='full')} new models "
      f"across {len(ARMS_TO_TRAIN)} arms, reused 3 for 'full' -- all six arms now matched at 3 seeds")
''')

md(r'''## Cell 9 — Ground-truth ligands for ALL 420 COACH420 structures (identical to Phase 53/56)''')
code(s53(19))

md(r'''## Cell 10 — Featurize (identical pipeline: dihedral + ESM2 + topology, matched to the round-3 topology block; cache filename updated)''')
code(s53(21).replace("cache/phase53_raw_features.pkl", "cache/phase58_raw_features.pkl"))

md(r'''## Cell 11 — P2Rank on the FULL 420 (same-run controlled reference, identical to Phase 53/56)''')
code(s53(23).replace('Path("/tmp/p2_20")', 'Path("/tmp/p2_58")')
            .replace('Path("/tmp/p2_20_out")', 'Path("/tmp/p2_58_out")'))

md(r'''## Cell 12 — Score all six arms (matched 3 seeds each), per-seed pairwise significance table''')
code(r'''
mu, sd = blob6b["topo_mean"], blob6b["topo_std"]

@torch.no_grad()
def predict_chain(entry, model_dict):
    x = entry["x"].clone()
    x[:, N_PRE:] = (x[:, N_PRE:] - torch.from_numpy(mu)) / torch.from_numpy(sd)
    d = type("D", (), {})()
    d.x=x.to(DEVICE); d.pos=entry["pos"].to(DEVICE)
    d.edge_index=entry["edge_index"].to(DEVICE); d.edge_attr=entry["edge_attr"].to(DEVICE)
    d.batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
    seed_logits, seed_centers = [], []
    for m in model_dict.values():
        lg, pc = m(d)
        seed_logits.append(torch.sigmoid(lg).cpu().numpy())
        seed_centers.append(pc.cpu().numpy()[0])
    offset = entry["ca"].mean(axis=0)
    return float(np.mean(seed_logits, axis=0).sum()), np.mean(seed_centers, axis=0) + offset


def dcc(pred_center, ligs):
    if pred_center is None or not ligs: return None
    best = np.inf
    for lig_atoms in ligs:
        if len(lig_atoms) == 0: continue
        best = min(best, float(np.linalg.norm(pred_center - lig_atoms.mean(axis=0))))
    return best if np.isfinite(best) else None


by_structure = defaultdict(list)
for (pid, cid), entry in built.items():
    by_structure[pid].append((cid, entry))

def score_arm(model_dict):
    hits, n = 0, 0
    for pid, chain_list in by_structure.items():
        ligs = chain_list[0][1]["ligs"]
        best_conf, best_center = -np.inf, None
        for cid, entry in chain_list:
            conf, center = predict_chain(entry, model_dict)
            if conf > best_conf: best_conf, best_center = conf, center
        d_ = dcc(best_center, ligs)
        if d_ is None: continue
        n += 1; hits += int(d_ < 4.0)
    return hits / max(n, 1), n

print(f"{'arm':<14}{'DCC (3-seed matched)':>22}{'n':>6}")
print("-"*42)
summary = {}
for arm_name, model_dict in arm_models.items():
    rate, n = score_arm(model_dict)
    summary[arm_name] = {"dcc": rate, "n": n}
    print(f"{arm_name:<14}{rate:>22.4f}{n:>6}")

base = summary["no_topo"]["dcc"]; full = summary["full"]["dcc"]; total_gain = full - base
print(f"\nno_topo baseline: {base:.4f}  full: {full:.4f}  (total gain: {total_gain:+.4f})")

decomp_arms = ["H0_only", "H1_only", "H2_only", "burial_only"]
ranked = sorted(decomp_arms, key=lambda n: -summary[n]["dcc"])
print(f"\n{'arm':<14}{'DCC':>10}{'gain over no_topo':>20}{'% of full gain':>18}")
print("-"*62)
for name in ranked:
    g = summary[name]["dcc"] - base
    pct = 100 * g / total_gain if total_gain > 1e-9 else float("nan")
    print(f"{name:<14}{summary[name]['dcc']:>10.4f}{g:>+20.4f}{pct:>17.1f}%")
print(f"\nmatched 3-seed ranking: {' > '.join(ranked)}")
''')

md(r'''## Cell 13 — Per-seed pairwise significance table across all four decomposition arms''')
code(r'''
def per_seed_dcc(arm_name):
    rates = []
    for seed, m in sorted(arm_models[arm_name].items()):
        rate, n = score_arm({seed: m})
        rates.append(rate)
    return np.array(rates)

per_seed = {name: per_seed_dcc(name) for name in ["full", "no_topo"] + decomp_arms}
for name in per_seed:
    r = per_seed[name]
    print(f"{name:<14} per-seed {np.round(r,4).tolist()}  mean {r.mean():.4f}  std {r.std():.4f}")

from itertools import combinations
print(f"\n{'pair':<28}{'mean diff':>12}{'noise (max std)':>18}{'wins/3':>10}{'diff/noise':>12}")
print("-"*82)
pairwise = {}
for a, b in combinations(decomp_arms, 2):
    ra, rb = per_seed[a], per_seed[b]
    diff = ra.mean() - rb.mean()
    noise = max(ra.std(), rb.std())
    wins = int((ra > rb).sum())
    ratio = abs(diff) / noise if noise > 1e-9 else float("inf")
    pairwise[f"{a}_vs_{b}"] = {"diff": float(diff), "noise": float(noise), "wins": wins, "ratio": float(ratio)}
    print(f"{a+' vs '+b:<28}{diff:>+12.4f}{noise:>18.4f}{wins:>7}/3{ratio:>12.2f}x")

verdict = ("ATTRIBUTABLE -- at least one pair clears 1.5x noise" if
           max(v["ratio"] for v in pairwise.values()) > 1.5 else
           "STILL DIFFUSE -- no pair clears 1.5x noise, matches the original-scale finding")
print(f"\nVERDICT: {verdict}")

json.dump({"summary": summary, "per_seed": {k: v.tolist() for k,v in per_seed.items()},
           "pairwise": pairwise, "ranked": ranked, "verdict": verdict},
          open(WORK/"phase58_summary.json", "w"), indent=2)
print("\nsaved -> phase58_summary.json")
''')

md(r'''
---

## Reading this

- **Still diffuse (no pair clears 1.5x noise)** -- confirms the original
  finding is scale-independent: this decomposition genuinely cannot
  attribute the topology benefit to one homology dimension, with 3.65x
  more data. A real, stable negative result.
- **One or more pairs now separate clearly** -- more data sharpened a
  signal that was too noisy to see at the original scale; worth an 8-seed
  confirmation before fully trusting which arm(s) actually separated.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase58_topology_decomposition_scale3.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
