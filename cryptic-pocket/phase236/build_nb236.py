#!/usr/bin/env python3
"""Phase 236: geometry control for the LOCALIZATION claim (Phase 22 protocol: original 2,056-chain pool,
COACH420 + HOLO4K leakage-clean benchmark, pooled DCC@4A, 8 seeds). 4 arms x 8 seeds = 32 trainings."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
nb22 = json.load(open(ROOT / "phase22_topology_localization_variance.ipynb"))
src22 = lambda i: "".join(nb22["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r'''
# Phase 236 - Does topology's localization gain survive a plain-geometry control?

Phase 22 reported that topology improves pocket-center localization on COACH420 + HOLO4K: 8-seed pooled DCC@4A,
`full` vs `no_topo`, +0.0528 (the paper's headline localization claim, novel because no prior work combines trained
equivariant localization with topology). That comparison never had a geometry arm. Phases 228/230/232 showed that
17 plain per-residue geometry features match or beat the topology block for classification. This notebook runs the same
localization experiment with geometry arms.

**Protocol (identical to Phase 22).** Training pool: Phase 10's 2,056 chains with recovered true ligand centers, 92/8
train/val (RandomState 42), model `LocalizingEGNN` (lambda 0.1, 40 epochs, patience 8, batch 16). Scoring: Phase 15's cached
1,149 leakage-clean benchmark units, one chain per structure chosen by summed predicted probability, predicted center =
mean over seeds, hit if within 4 A of any ligand center of mass; pooled COACH420 + HOLO4K.

**Only changes.** (1) Every arm uses an identical 51-wide side channel (34 topology + 17 geometry columns; unselected columns
zeroed) so architecture is identical; (2) all four arms are retrained (8 seeds each), because widening the side channel
makes the old `full`/`no_topo` checkpoints incompatible; (3) chains that fail the geometry alignment check are dropped from
every arm.

Arms: `no-topo`, `full` (34 topology), `geom` (17), `geom+topo` (51).
**Primary comparison (fixed before any result): `geom+topo` vs `geom`, seed-paired pooled DCC.**
Secondary: `full` vs `no-topo` (does Phase 22 replicate?), `geom` vs `no-topo`, `geom` vs `full`.
''')
md("## Cell 1 - Install")
code("!pip install -q torch-geometric==2.6.1 biopython==1.84")
md("## Cell 2 - Load Phase 10 graphs and the training-pool geometry (Phase 229); append 17 standardized geometry columns")
code(src22(3) + r'''

geo_tr = torch.load(locate("phase229_geometry_features.pt"), map_location="cpu", weights_only=False)
GEOM_DIM = len(geo_tr["names"]); assert GEOM_DIM == 17
SIDE_DIM = TOPO_DIM + GEOM_DIM
g_mu = torch.from_numpy(np.asarray(geo_tr["mean"], np.float32)); g_sd = torch.from_numpy(np.asarray(geo_tr["std"], np.float32))
_all = dataset; dataset = []; _dropped = 0
for d in _all:
    k = (str(d.pdb_id), str(d.chain_id))
    if k not in geo_tr["feats"]:
        _dropped += 1; continue
    f = geo_tr["feats"][k]; assert f.shape == (d.x.size(0), GEOM_DIM)
    z = ((torch.from_numpy(f) - g_mu) / g_sd).clamp(-5, 5)      # label-free statistics
    d.x = torch.cat([d.x, z], dim=1); dataset.append(d)
print(f"training chains with geometry: {len(dataset):,} (dropped {_dropped}); node dim {dataset[0].x.size(1)} = {N_PRE} + {TOPO_DIM} + {GEOM_DIM}")
''')
md("## Cell 3 - Recover true ligand centers for the training set and split 92/8 (Phase 22, verbatim)")
code(src22(4))
md("## Cell 4 - Model (Phase 22, verbatim)")
code(src22(6))
md("## Cell 5 - Training machinery (Phase 22, verbatim except side-channel width)")
code(src22(8).replace("LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head)", "LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=use_loc_head)"))
md("## Cell 6 - Arms; train 4 arms x 8 seeds")
code(r'''
import time
WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)
ARMS = ["no-topo", "full", "geom", "geom+topo"]
TOPO_COLS = list(range(N_PRE, N_PRE + TOPO_DIM))
GEOM_COLS = list(range(N_PRE + TOPO_DIM, N_PRE + SIDE_DIM))
KEEP = {"no-topo": [], "full": TOPO_COLS, "geom": GEOM_COLS, "geom+topo": TOPO_COLS + GEOM_COLS}
ORIG_X = [d.x.clone() for d in dataset]
def apply_arm(a):
    for d, x0 in zip(dataset, ORIG_X):
        x = x0.clone(); mask = torch.ones(x.size(1), dtype=torch.bool)
        mask[N_PRE:] = False; mask[KEEP[a]] = True
        x[:, ~mask] = 0.0
        d.x = x

BEST_LAMBDA = 0.1
models = {a: {} for a in ARMS}; train_log = []
t0 = time.time()
for arm in ARMS:
    apply_arm(arm)
    for seed in range(8):
        m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
        v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
        models[arm][seed] = m
        torch.save(m.state_dict(), WORK/f"cache/loc236_{arm.replace('+','_')}_seed{seed}.pt")
        train_log.append({"arm": arm, "seed": seed, "val_pr": float(val_pr), "val_dcc": float(v.get("dcc", float("nan")))})
        print(f"  [{arm:9s}] seed {seed}: val PR {val_pr:.4f}  val DCC {v.get('dcc',float('nan')):.4f}  [{(time.time()-t0)/60:.1f}m]")
json.dump(train_log, open(WORK/"phase236_train_log.json", "w"), indent=2)
''')
md("## Cell 7 - Load the cached benchmark features (Phase 15) and their geometry (Phase 235)")
code(r'''
import pickle
with open(locate("phase15_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)
geo_bm = torch.load(locate("phase235_benchmark_geometry.pt"), map_location="cpu", weights_only=False)
blob6b_bench = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)
mu_b, sd_b = blob6b_bench["topo_mean"], blob6b_bench["topo_std"]
del blob6b_bench
valid_keys = {k for k in raw_cache if k in geo_bm["feats"]}
print(f"benchmark chains: {len(raw_cache):,} cached, {len(valid_keys):,} with geometry (dropped {len(raw_cache)-len(valid_keys)} for every arm)")
''')
md("## Cell 8 - Score all arms and seeds (Phase 22 scoring, arm masks applied to the benchmark features)")
code(r'''
from collections import defaultdict

def bench_x(entry, key, arm):
    x = torch.from_numpy(entry["x"]).clone()
    topo = (x[:, N_PRE:] - torch.from_numpy(mu_b)) / torch.from_numpy(sd_b)
    g = ((torch.from_numpy(geo_bm["feats"][key]) - g_mu) / g_sd).clamp(-5, 5)
    x = torch.cat([x[:, :N_PRE], topo, g], dim=1)
    mask = torch.ones(x.size(1), dtype=torch.bool); mask[N_PRE:] = False; mask[KEEP[arm]] = True
    x[:, ~mask] = 0.0
    return x

@torch.no_grad()
def predict_all_seeds(entry, key, arm):
    d = type("D", (), {})()
    d.x = bench_x(entry, key, arm).to(DEVICE); d.pos = torch.from_numpy(entry["pos"]).to(DEVICE)
    d.edge_index = torch.from_numpy(entry["edge_index"]).to(DEVICE); d.edge_attr = torch.from_numpy(entry["edge_attr"]).to(DEVICE)
    d.batch = torch.zeros(d.x.size(0), dtype=torch.long, device=DEVICE)
    offset = entry["ca"].mean(axis=0); out = {}
    for seed, m in models[arm].items():
        lg, pc = m(d)
        out[seed] = {"prob_sum": float(torch.sigmoid(lg).sum().cpu()), "center": pc.cpu().numpy()[0] + offset}
    return out

def dcc_dist(pred_center, ligs):
    if pred_center is None or not ligs: return None
    best = np.inf
    for lig_atoms in ligs:
        if len(lig_atoms) == 0: continue
        best = min(best, float(np.linalg.norm(pred_center - lig_atoms.mean(axis=0))))
    return best if np.isfinite(best) else None

t0 = time.time()
by_structure = defaultdict(list)
for key in valid_keys: by_structure[key[:2]].append((key[2], key))
out = {a: {} for a in ARMS}; ligs_of = {}
for skey, chain_list in by_structure.items():
    ligs_of[skey] = raw_cache[chain_list[0][1]]["ligs"]
    for arm in ARMS:
        out[arm][skey] = [(cid, predict_all_seeds(raw_cache[k], k, arm)) for cid, k in chain_list]
print(f"inference (8 seeds x {len(ARMS)} arms x {len(by_structure)} structures) in {(time.time()-t0)/60:.1f} min")

def hits_for(arm, seeds, src=None):
    """dict structure -> 0/1 hit, ensemble over `seeds`, best chain by mean summed probability."""
    res = {}
    for skey, chain_list in out[arm].items():
        if src is not None and skey[0] != src: continue
        best_conf, best_center = -np.inf, None
        for cid, per_seed in chain_list:
            cs = [per_seed[s]["center"] for s in seeds if s in per_seed]; fs = [per_seed[s]["prob_sum"] for s in seeds if s in per_seed]
            if not cs: continue
            c, f = np.mean(cs, axis=0), np.mean(fs)
            if f > best_conf: best_conf, best_center = f, c
        dd = dcc_dist(best_center, ligs_of[skey])
        if dd is None: continue
        res[skey] = int(dd < 4.0)
    return res

def rate(h): return float(np.mean(list(h.values()))) if h else float("nan")
''')
md("## Cell 9 - Per-seed pooled DCC, seed-paired statistics (primary comparison first), structure-level bootstrap on the 8-seed ensembles")
code(r'''
from scipy.stats import ttest_rel, wilcoxon
per_seed = {a: np.array([rate(hits_for(a, (s,))) for s in range(8)]) for a in ARMS}
per_seed_bm = {(a, src): np.array([rate(hits_for(a, (s,), src)) for s in range(8)]) for a in ARMS for src in ("coach420", "holo4k")}
print(f"{'arm':<10}{'pooled DCC mean':>17}{'std':>8}   {'COACH420':>9}{'HOLO4K':>9}")
for a in ARMS:
    print(f"{a:<10}{per_seed[a].mean():>17.4f}{per_seed[a].std():>8.4f}   {per_seed_bm[(a,'coach420')].mean():>9.4f}{per_seed_bm[(a,'holo4k')].mean():>9.4f}")

COMPS = [("geom+topo", "geom"), ("full", "no-topo"), ("geom", "no-topo"), ("geom", "full"), ("geom+topo", "full")]
rng = np.random.default_rng(236)
ens = {a: hits_for(a, tuple(range(8))) for a in ARMS}
rows = []
print(f"\n{'comparison (a - b)':<26}{'per-seed delta':>15}{'wins':>6}{'t p':>9}{'W p':>9}   {'8-seed ensemble delta [95% bootstrap over structures]':>58}")
for a, b in COMPS:
    d = per_seed[a] - per_seed[b]
    tp, wp = ttest_rel(per_seed[a], per_seed[b]).pvalue, wilcoxon(d).pvalue
    keys = sorted(set(ens[a]) & set(ens[b])); ha = np.array([ens[a][k] for k in keys]); hb = np.array([ens[b][k] for k in keys])
    idx = rng.integers(0, len(keys), size=(5000, len(keys)))
    bs = (ha[idx].mean(1) - hb[idx].mean(1))
    lo, hi = np.percentile(bs, [2.5, 97.5]); dens = float(ha.mean() - hb.mean())
    rows.append(dict(a=a, b=b, mean_seed_delta=float(d.mean()), wins=int((d > 0).sum()), t_p=float(tp), wilcoxon_p=float(wp),
                     ens_delta=dens, ens_ci_lo=float(lo), ens_ci_hi=float(hi), n_structures=len(keys)))
    print(f"{a+' - '+b:<26}{d.mean():>+15.4f}{int((d>0).sum()):>4}/8{tp:>9.1e}{wp:>9.1e}   {dens:>+8.4f} [{lo:+.4f}, {hi:+.4f}]  n={len(keys)}")
print(f"\nPhase 22 reference (old checkpoints, no geometry arm): full - no_topo = +0.0528 pooled DCC")
import pandas as pd
pd.DataFrame(rows).to_csv(WORK/"phase236_paired_stats.csv", index=False)
json.dump({"per_seed": {a: per_seed[a].tolist() for a in ARMS},
           "per_seed_by_benchmark": {f"{a}|{s}": per_seed_bm[(a, s)].tolist() for (a, s) in per_seed_bm},
           "ensemble_rate": {a: rate(ens[a]) for a in ARMS}, "n_structures": len(by_structure),
           "n_bench_chains_valid": len(valid_keys), "n_bench_chains_total": len(raw_cache), "n_train_chains": len(dataset)},
          open(WORK/"phase236_summary.json", "w"), indent=2)
print("saved phase236_paired_stats.csv, phase236_summary.json, phase236_train_log.json")
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
o = HERE / "phase236_localization_geometry_control.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
