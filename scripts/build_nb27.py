#!/usr/bin/env python3
"""Phase 27: seed-validate Thread 2 (HOLO4K legitimate comparison). Trains 5
more seeds (total 8) for the HOLO4K clean ensemble, reusing Phase 24's cached
benchmark features (phase24_raw_features.pkl) -- no re-fetch/re-featurize of
the 1000-structure HOLO4K sample. Mirrors exactly what Phase 21 did for
COACH420. CPU-mode per explicit instruction; training-set reconstruction
(leakage filter + true-center recovery) is spliced verbatim from Phase 24."""
import json, pickle
from pathlib import Path

C24 = pickle.load(open("/tmp/p24_cells.pkl", "rb"))
s24 = lambda i: C24[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 27 — Is the HOLO4K P2Rank Gap Real, or Seed Noise?

Phase 24 found the model beats every literature baseline on HOLO4K (EquiPocket
0.337, P2Rank-2018 0.314) but loses to this-run's live P2Rank by −0.0979 DCC
(0.5601 vs 0.6580) -- a bigger gap than COACH420's, and only a 3-seed first
pass. Phase 21 showed COACH420's analogous −0.044 gap shrank to −0.0052 (mostly
noise) once the ensemble grew from 3 to 8 seeds. This notebook applies the
same treatment here: train 5 more seeds (8 total), rescore against Phase 24's
already-cached benchmark features (no re-fetch, no re-run of the ~1hr+
featurization pipeline), and see whether the HOLO4K gap similarly shrinks.

CPU-mode (explicit instruction) -- training is slower here than the GPU runs
that produced seeds 0-2, so this budgets more wall-clock time than Phase 21
did, but no new topology computation or benchmark featurization is needed.
''')

md(r'''## Cell 1 — Setup''')
code(s24(2))

code("import pickle\n\n" + s24(3))

md(r'''## Cell 2 — Sample the same ~1,000 HOLO4K structures (seed=2024, identical to Phase 24)''')
code(s24(5))

md(r'''## Cell 3 — Rebuild the identical leakage-clean training set (Phase 24 logic, reused verbatim)''')
code(s24(7))

md(r'''## Cell 4 — Recover true ligand centers (Phase 24/14 logic, reused verbatim)''')
code(s24(9))

md(r'''## Cell 5 — Model (identical to every prior localization phase)''')
code(s24(11))

md(r'''## Cell 6 — Training machinery (identical to Phase 24/14)''')
code(s24(13))

md(r'''## Cell 7 — Train 5 more seeds (3-7), reusing seeds 0-2 from Phase 24''')
code(r'''
import time
BEST_LAMBDA = 0.1
WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)

t0 = time.time()
for seed in (3,4,5,6,7):
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    torch.save(m.state_dict(), WORK/f"cache/holo_clean_deploy_seed{seed}.pt")
    val_dcc = v.get("dcc", float("nan"))
    print(f"  seed {seed}: val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m elapsed]")
print(f"\ntrained seeds 3-7 in {(time.time()-t0)/60:.1f} min (CPU)")
''')

md(r'''## Cell 8 — Load ALL 8 seeds + Phase 24's cached benchmark features (no re-fetch)''')
code(r'''
def load_seeds(prefix, seeds):
    out = {}
    for seed in seeds:
        cands = []
        for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
            if r.exists(): cands.extend(r.rglob(f"{prefix}{seed}.pt"))
        if not cands: continue
        state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
        mm = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE); mm.load_state_dict(state); mm.eval()
        out[seed] = mm
    return out

ensemble8 = load_seeds("holo_clean_deploy_seed", range(8))
print(f"loaded {len(ensemble8)} 'holo_clean_deploy' checkpoints (seeds {sorted(ensemble8)})")
assert len(ensemble8) == 8, "expected all 8 seeds -- check that seeds 0-2 attached from Phase 24"

with open(locate("phase24_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)
print(f"loaded cached HOLO4K benchmark features for {len(raw_cache):,} chains")
''')

md(r'''## Cell 9 — Score pooled 8-seed ensemble + per-seed breakdown''')
code(r'''
from collections import defaultdict

@torch.no_grad()
def predict(entry, model_dict):
    x = torch.from_numpy(entry["x"]).clone()
    t = (x[:, N_PRE:] - torch.from_numpy(blob6b["topo_mean"])) / torch.from_numpy(blob6b["topo_std"])
    x[:, N_PRE:] = t
    d = type("D", (), {})()
    d.x=x.to(DEVICE); d.pos=torch.from_numpy(entry["pos"]).to(DEVICE)
    d.edge_index=torch.from_numpy(entry["edge_index"]).to(DEVICE)
    d.edge_attr=torch.from_numpy(entry["edge_attr"]).to(DEVICE)
    d.batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
    offset = entry["ca"].mean(axis=0)
    seed_logits, seed_centers = [], []
    for mm in model_dict.values():
        lg, pc = mm(d)
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
for (pid, cid), entry in raw_cache.items():
    by_structure[pid].append((cid, entry))

mu_b, sd_b = blob6b["topo_mean"], blob6b["topo_std"]

def score_ensemble(model_dict):
    hits, n = 0, 0
    for pid, chain_list in by_structure.items():
        ligs = chain_list[0][1]["ligs"]
        best_conf, best_center = -np.inf, None
        for cid, entry in chain_list:
            conf, center = predict(entry, model_dict)
            if conf > best_conf: best_conf, best_center = conf, center
        d_ = dcc(best_center, ligs)
        if d_ is None: continue
        n += 1; hits += int(d_ < 4.0)
    return hits / max(n, 1), n

for k in (3, 8):
    seeds = tuple(range(k))
    subset = {s: ensemble8[s] for s in seeds if s in ensemble8}
    rate, n = score_ensemble(subset)
    print(f"{k}-seed ensemble DCC: {rate:.4f}  (n={n})")

per_seed = []
for seed in range(8):
    rate, n = score_ensemble({seed: ensemble8[seed]})
    per_seed.append(rate)
    print(f"  seed {seed}: DCC {rate:.4f}")
per_seed = np.array(per_seed)
print(f"\nper-seed mean {per_seed.mean():.4f}  std {per_seed.std():.4f}")
''')

md(r'''## Cell 10 — Compare to the live P2Rank number from Phase 24 (same-run, same 1000 structures)''')
code(r'''
# Phase 24 already ran P2Rank on this exact 1000-structure sample in a fully
# controlled, same-run comparison and reported DCC 0.6580 -- reusing that
# number rather than re-running P2Rank (deterministic tool, same input set).
LIVE_P2RANK_DCC = 0.6580
ours_3seed = score_ensemble({s: ensemble8[s] for s in (0,1,2)})[0]
ours_8seed = score_ensemble({s: ensemble8[s] for s in range(8)})[0]

gap_3 = ours_3seed - LIVE_P2RANK_DCC
gap_8 = ours_8seed - LIVE_P2RANK_DCC
noise = per_seed.std()

print(f"ours (3-seed, Phase24 original): {ours_3seed:.4f}   gap to P2Rank: {gap_3:+.4f}")
print(f"ours (8-seed, this notebook):    {ours_8seed:.4f}   gap to P2Rank: {gap_8:+.4f}")
print(f"per-seed noise (std): {noise:.4f}")
print(f"8-seed gap / noise ratio: {abs(gap_8)/noise:.2f}x" if noise > 1e-9 else "noise ~0")

json.dump({"ours_3seed": ours_3seed, "ours_8seed": ours_8seed, "live_p2rank": LIVE_P2RANK_DCC,
           "gap_3seed": gap_3, "gap_8seed": gap_8, "per_seed": per_seed.tolist(),
           "noise_std": float(noise)},
          open("/kaggle/working/phase27_summary.json", "w"), indent=2)
print("\nsaved -> phase27_summary.json")
''')

md(r'''
---

This is the direct HOLO4K analogue of Phase 21's COACH420 seed-variance check
-- same methodology, same discipline: does the 3-seed gap shrink materially
once judged against a proper 8-seed noise floor, the way COACH420's did?
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase27_holo4k_seed_variance.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
