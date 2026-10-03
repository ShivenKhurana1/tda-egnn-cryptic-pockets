#!/usr/bin/env python3
"""Phase 34: does the scaled pool's HOLO4K gap (Phase33: -0.0297, roughly
half of Phase27b's -0.0668 on the original pool) close further or fully
resolve with more seeds -- mirroring Phase32's treatment of the COACH420
result, which revealed the win GREW from 3 to 8 seeds. Trains 5 more seeds
(3-7) on the scaled pool's leakage-clean HOLO4K training set, rescores
against Phase33's cached benchmark features (no re-fetch/re-featurize).
Uses the FIXED (pid,cid) 2-tuple key schema for by_structure grouping
(Phase27's original crash was exactly this bug on a single-benchmark cache)."""
import json, pickle
from pathlib import Path

C33 = pickle.load(open("/tmp/p33_cells.pkl", "rb"))
s33 = lambda i: (C33[i]["source"] if isinstance(C33[i]["source"], str) else "".join(C33[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 34 - Does the Scaled Pool's HOLO4K Gap Close With More Seeds?

Phase 33 (3-seed, scaled pool) found DCC 0.6283 vs live P2Rank's 0.6580 --
gap -0.0297, roughly HALF of Phase27b's -0.0668 (8-seed, original pool).
Phase 32 showed the analogous COACH420 gap didn't just shrink with more
seeds, it grew into a bigger win (+0.0336 -> +0.0433). This notebook applies
the identical stress test here: train 5 more seeds (8 total) on Phase 33's
leakage-clean training set, rescore against Phase 33's cached benchmark
features -- zero re-fetch, zero re-featurization.
''')

md(r'''## Cell 1 - Setup (loads Phase 30/29's scaled blobs)''')
code(s33(2))
code(s33(3))

md(r'''## Cell 2 - Sample the SAME ~1,000 HOLO4K structures (seed=2024, identical to Phase 33)''')
code(s33(5))

md(r'''## Cell 3 - Clean the TRAINING side (identical logic, should reproduce Phase 33's 3,187 chains)''')
code(s33(7))

md(r'''## Cell 4 - Recover true ligand centers (identical to Phase 33/24/14)''')
code(s33(9))

md(r'''## Cell 5 - Model (identical to every prior localization phase)''')
code(s33(11))

md(r'''## Cell 6 - Training machinery (identical to Phase 33/24/14)''')
code(s33(13))

md(r'''## Cell 7 - Train 5 MORE seeds (3-7), reusing seeds 0-2 from Phase 33''')
code(r'''
import time
WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)

models = {}
for seed in (0,1,2):
    cands = []
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): cands.extend(r.rglob(f"holo_clean_deploy_scaled_seed{seed}.pt"))
    assert cands, f"holo_clean_deploy_scaled_seed{seed}.pt not found -- attach Phase 33's kernel output"
    state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
    m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE); m.load_state_dict(state); m.eval()
    models[seed] = m
print(f"loaded {len(models)} checkpoints from Phase 33 (no retraining)")

BEST_LAMBDA = 0.1
t0 = time.time()
for seed in (3,4,5,6,7):
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    models[seed] = m
    torch.save(m.state_dict(), WORK/f"cache/holo_clean_deploy_scaled_seed{seed}.pt")
    val_dcc = v.get("dcc", float("nan"))
    print(f"  seed {seed}: val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")
print(f"\n{len(models)} total seeds available (0-7)")
''')

md(r'''## Cell 8 - Load Phase 33's cached benchmark features (no re-fetch)''')
code(r'''
with open(locate("phase33_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)
print(f"loaded cached (scaled-pool) HOLO4K features for {len(raw_cache):,} chains")
mu, sd = blob6b["topo_mean"], blob6b["topo_std"]
''')

md(r'''## Cell 9 - Per-seed and per-ensemble-size DCC on the cached benchmark (FIXED (pid,cid) key schema)''')
code(r'''
from collections import defaultdict

def dcc_dca(pred_center, ligs):
    if pred_center is None or not ligs: return None, None
    best_dcc, best_dca = np.inf, np.inf
    for lig_atoms in ligs:
        if len(lig_atoms) == 0: continue
        com = lig_atoms.mean(axis=0)
        d_ = float(np.linalg.norm(pred_center - com))
        if d_ < best_dcc:
            best_dcc = d_
            best_dca = float(np.linalg.norm(lig_atoms - pred_center, axis=1).min())
    return (best_dcc if np.isfinite(best_dcc) else None, best_dca if np.isfinite(best_dca) else None)


@torch.no_grad()
def predict_conf_and_centers_all_seeds(entry):
    """Returns {seed: (confidence, center_raw_frame)} for every trained seed
    on one chain. Confidence is the classification-head sigmoid sum -- the
    SAME cross-chain arbitration signal (`sum_probs`) Phase19/24/27 used to
    pick which chain represents a multi-chain HOLO4K structure."""
    x = torch.from_numpy(entry["x"]).clone()
    x[:, N_PRE:] = (x[:, N_PRE:] - torch.from_numpy(mu)) / torch.from_numpy(sd)
    d = type("D", (), {})()
    d.x=x.to(DEVICE); d.pos=torch.from_numpy(entry["pos"]).to(DEVICE)
    d.edge_index=torch.from_numpy(entry["edge_index"]).to(DEVICE)
    d.edge_attr=torch.from_numpy(entry["edge_attr"]).to(DEVICE)
    d.batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
    offset = entry["ca"].mean(axis=0)
    out = {}
    for seed, m in models.items():
        lg, pc = m(d)
        conf = float(torch.sigmoid(lg).sum())
        out[seed] = (conf, pc.cpu().numpy()[0] + offset)
    return out


by_structure = defaultdict(list)
for (pid, cid), entry in raw_cache.items():
    by_structure[pid].append((cid, entry))
print(f"grouped into {len(by_structure):,} structures")

t0 = time.time()
per_chain_conf_centers = {}   # key -> {seed: (conf, center)}
per_chain_ligs = {}
for pid, chain_list in by_structure.items():
    ligs = chain_list[0][1]["ligs"]
    for cid, entry in chain_list:
        key = (pid, cid)
        per_chain_conf_centers[key] = predict_conf_and_centers_all_seeds(entry)
        per_chain_ligs[key] = ligs
print(f"inference (8 seeds x {len(per_chain_conf_centers)} chains) in {(time.time()-t0)/60:.1f} min")

structures = sorted(by_structure)

def dcc_rate(conf_center_fn):
    """conf_center_fn(key) -> (confidence, center) or None. Picks the
    highest-confidence chain per structure (matches Phase24/27's sum_probs
    arbitration), then scores that chain's center against the ligands."""
    hits, n = 0, 0
    for pid in structures:
        chain_keys = [(pid, cid) for cid, _ in by_structure[pid]]
        best_conf, best_center = -np.inf, None
        for key in chain_keys:
            r = conf_center_fn(key)
            if r is None: continue
            conf, center = r
            if conf > best_conf: best_conf, best_center = conf, center
        if best_center is None: continue
        ligs = per_chain_ligs[chain_keys[0]]
        d_, _ = dcc_dca(best_center, ligs)
        if d_ is None: continue
        n += 1; hits += int(d_ < 4.0)
    return hits / max(n, 1), n


print(f"\n{'per-seed DCC (individual models, no ensembling, own-confidence chain arbitration)':<70}")
print("-"*70)
seed_dccs = []
for seed in range(8):
    dcc, n = dcc_rate(lambda k, s=seed: per_chain_conf_centers[k].get(s))
    seed_dccs.append(dcc)
    print(f"  seed {seed}: DCC {dcc:.4f}  (n={n})")
seed_dccs = np.array(seed_dccs)
print(f"\nseed-to-seed: mean {seed_dccs.mean():.4f}  std {seed_dccs.std():.4f}  "
      f"range [{seed_dccs.min():.4f}, {seed_dccs.max():.4f}]")
''')

md(r'''## Cell 10 - Ensemble size sweep: does averaging more seeds close the gap?''')
code(r'''
P2RANK_THIS_RUN = 0.657974   # Phase 33's same-run, same-benchmark P2Rank number

def ensemble_dcc(seed_subset):
    def conf_center_fn(key, seeds=seed_subset):
        pairs = [per_chain_conf_centers[key][s] for s in seeds if s in per_chain_conf_centers[key]]
        if not pairs: return None
        confs, centers = zip(*pairs)
        return float(np.mean(confs)), np.mean(centers, axis=0)
    return dcc_rate(conf_center_fn)


rng = np.random.RandomState(0)
print(f"{'ensemble size':<16}{'mean DCC':>10}{'std (5 draws)':>16}{'gap to P2Rank':>16}")
print("-"*58)
results = {}
for size in (1, 2, 3, 5, 8):
    draws = []
    n_draws = 5 if size < 8 else 1
    for _ in range(n_draws):
        subset = tuple(sorted(rng.choice(8, size=size, replace=False))) if size < 8 else tuple(range(8))
        dcc, n = ensemble_dcc(subset)
        draws.append(dcc)
    m, s = np.mean(draws), np.std(draws)
    results[size] = {"mean": float(m), "std": float(s), "draws": draws}
    print(f"{size:<16}{m:>10.4f}{s:>16.4f}{m-P2RANK_THIS_RUN:>16.4f}")

full8_dcc, full8_n = ensemble_dcc(tuple(range(8)))
print(f"\nfull 8-seed ensemble: DCC {full8_dcc:.4f} (n={full8_n}) vs P2Rank {P2RANK_THIS_RUN:.4f} "
      f"= {full8_dcc-P2RANK_THIS_RUN:+.4f}")
p33_dcc, p33_n = ensemble_dcc((0,1,2))
print(f"Phase 33's 3-seed (0,1,2 only) for direct comparison:")
print(f"  seeds (0,1,2): DCC {p33_dcc:.4f} (n={p33_n})  [Phase 33 reported 0.6283]")
''')

md(r'''## Cell 11 - Verdict''')
code(r'''
gap_3seed = p33_dcc - P2RANK_THIS_RUN
gap_8seed = full8_dcc - P2RANK_THIS_RUN
noise_scale = seed_dccs.std()

print(f"{'='*66}")
print(f"DOES THE SCALED-POOL HOLO4K GAP CLOSE WITH MORE SEEDS?")
print(f"{'='*66}")
print(f"single-seed DCC std across 8 independent runs: {noise_scale:.4f}")
print(f"3-seed ensemble gap to P2Rank:  {gap_3seed:+.4f}")
print(f"8-seed ensemble gap to P2Rank:  {gap_8seed:+.4f}")
print(f"gap magnitude vs single-seed noise: {abs(gap_8seed)/max(noise_scale,1e-9):.2f}x")

if gap_8seed > 0.01:
    verdict = "gap CLOSED and REVERSED with more seeds -- matches the COACH420 pattern"
elif gap_8seed > -0.015:
    verdict = "gap CLOSED to within noise with more seeds -- essentially parity"
elif abs(gap_8seed) > 1.5*noise_scale:
    verdict = "gap PERSISTS beyond noise -- HOLO4K genuinely responds differently to scale than COACH420"
else:
    verdict = "gap is WITHIN plausible seed noise -- not confidently resolved either way at this n"

print(f"\nVERDICT: {verdict}")

json.dump({"seed_dccs": seed_dccs.tolist(), "ensemble_sweep": results,
           "phase33_3seed_dcc": p33_dcc, "full8_dcc": full8_dcc,
           "p2rank_this_run": P2RANK_THIS_RUN, "verdict": verdict},
          open(WORK/"phase34_summary.json", "w"), indent=2)
print("\nsaved -> phase34_summary.json")
''')

md(r'''
---

## Reading this

Compare directly to Phase 32's COACH420 outcome (gap widened from +0.034 to
+0.043 with more seeds). If HOLO4K follows the same pattern, Thread 2 closes
too. If it doesn't, that's a genuinely interesting divergence worth
understanding (dataset composition, sample size, or something about
multi-chain HOLO4K structures behaving differently from COACH420's cleaner
single-target-chain setup).
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase34_scaled_holo4k_seed_variance.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
