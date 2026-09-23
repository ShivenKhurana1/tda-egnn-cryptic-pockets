#!/usr/bin/env python3
"""Phase 254: score the 32 Phase 253 checkpoints on the full COACH420 benchmark (scale-3 pipeline) and run the pre-specified
geometry-control statistics: per-seed pooled DCC paired tests, 8-seed-ensemble structure bootstrap, ensemble-size sweep over all
seed subsets, and a two-way (seeds x structures) bootstrap."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
nb53 = json.load(open(HERE.parents[1] / "phase53_scaled3_coach420.ipynb"))
s53 = lambda i: "".join(nb53["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r'''
# Phase 254 — Localization geometry control at the 7,513-chain pool: scoring and statistics

Loads the 32 checkpoints from Phase 253a/b (arms no-topo, full, geom, geom+topo x 8 seeds) and scores them on the full 420-structure
COACH420 benchmark (Phase 53's cached features; Phase 252's geometry for the same chains). Hit = predicted pocket center within 4 A
of any ligand center of mass. **Primary comparison (fixed before any result): `geom+topo` vs `geom`, seed-paired pooled DCC.**
Then the same follow-ups that resolved the 35M case: the 8-seed-ensemble difference, its dependence on ensemble size, and a
two-way bootstrap that resamples seeds and structures together.
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1")
md("## Cell 2 — Load benchmark cache, geometry, topology standardization, and the trained checkpoints")
code(r'''
import json, os, pickle, time, warnings
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax
warnings.filterwarnings("ignore")

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

WORK = Path("/kaggle/working")
N_PRE, TOPO_DIM, GEOM_DIM = 507, 34, 17; SIDE_DIM = TOPO_DIM + GEOM_DIM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(locate("phase53_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)
geo = torch.load(locate("phase252_coach_geometry.pt"), map_location="cpu", weights_only=False)
b51 = torch.load(locate("phase51_graphs.pt"), map_location="cpu", weights_only=False)
mu_b, sd_b = b51["topo_mean"], b51["topo_std"]; del b51
sa = torch.load(locate("phase253a_geom_stats.pt"), map_location="cpu", weights_only=False)
sb = torch.load(locate("phase253b_geom_stats.pt"), map_location="cpu", weights_only=False)
assert torch.allclose(sa["g_mu"], sb["g_mu"]) and torch.allclose(sa["g_sd"], sb["g_sd"]) and sa["n_train"] == sb["n_train"], \
    "the two training kernels used different training sets / z-scoring statistics"
g_mu, g_sd = sa["g_mu"], sa["g_sd"]
valid_keys = sorted(k for k in raw_cache if k in geo["feats"])
print(f"benchmark chains: {len(raw_cache)} cached, {len(valid_keys)} with geometry; training chains used: {sa['n_train']}")
k0 = valid_keys[0]; print("x width", raw_cache[k0]["x"].shape[1], "(expect", N_PRE + TOPO_DIM, ")")
assert raw_cache[k0]["x"].shape[1] == N_PRE + TOPO_DIM
''')
md("## Cell 3 — Model (Phase 53, verbatim)")
code(s53(11))
md("## Cell 4 — Load all 32 checkpoints")
code(r'''
ARMS = ["no-topo", "full", "geom", "geom+topo"]
models = {a: {} for a in ARMS}
for arm in ARMS:
    for seed in range(8):
        st = torch.load(locate(f"loc253_{arm.replace('+','_')}_seed{seed}.pt"), map_location=DEVICE)
        m = LocalizingEGNN(N_PRE, SIDE_DIM).to(DEVICE); m.load_state_dict(st); m.eval(); models[arm][seed] = m
print({a: len(v) for a, v in models.items()}, "checkpoints loaded")
KEEP_SIDE = {"no-topo": [], "full": list(range(TOPO_DIM)), "geom": list(range(TOPO_DIM, SIDE_DIM)), "geom+topo": list(range(SIDE_DIM))}
''')
md("## Cell 5 — Score every arm and seed on COACH420; centers kept per (structure, seed)")
code(r'''
def bench_x(entry, key, arm):
    x = torch.from_numpy(np.asarray(entry["x"], np.float32)).clone()
    topo = (x[:, N_PRE:] - torch.from_numpy(mu_b)) / torch.from_numpy(sd_b)
    g = ((torch.from_numpy(geo["feats"][key]) - g_mu) / g_sd).clamp(-5, 5)
    side = torch.cat([topo, g], dim=1)
    m = torch.zeros(SIDE_DIM); m[KEEP_SIDE[arm]] = 1.0
    return torch.cat([x[:, :N_PRE], side * m], dim=1)

@torch.no_grad()
def centers_for(entry, key, arm):
    d = type("D", (), {})()
    d.x = bench_x(entry, key, arm).to(DEVICE); d.pos = torch.as_tensor(entry["pos"]).float().to(DEVICE)
    d.edge_index = torch.as_tensor(entry["edge_index"]).long().to(DEVICE); d.edge_attr = torch.as_tensor(entry["edge_attr"]).float().to(DEVICE)
    d.batch = torch.zeros(d.x.size(0), dtype=torch.long, device=DEVICE)
    off = np.asarray(entry["ca"]).mean(axis=0)
    return np.stack([models[arm][s](d)[1].cpu().numpy()[0] + off for s in range(8)])          # (8, 3)

t0 = time.time()
CENT = {a: np.zeros((len(valid_keys), 8, 3)) for a in ARMS}
for i, key in enumerate(valid_keys):
    for arm in ARMS: CENT[arm][i] = centers_for(raw_cache[key], key, arm)
print(f"scored {len(valid_keys)} structures x {len(ARMS)} arms x 8 seeds in {(time.time()-t0)/60:.1f} min")

# ligand centers of mass, padded to a common count so hit tests vectorise
LMAX = max(len(raw_cache[k]["ligs"]) for k in valid_keys)
COM = np.full((len(valid_keys), LMAX, 3), np.nan)
for i, k in enumerate(valid_keys):
    for j, lig in enumerate(raw_cache[k]["ligs"]):
        if len(lig): COM[i, j] = np.asarray(lig).mean(axis=0)
has_lig = ~np.isnan(COM).all(axis=(1, 2)); print("structures with a ligand:", int(has_lig.sum()), "of", len(valid_keys))

def hits(arm, seeds):
    """0/1 per structure (NaN where no ligand): mean predicted center over `seeds` within 4 A of any ligand COM."""
    c = CENT[arm][:, list(seeds), :].mean(axis=1)                         # (K, 3)
    dmin = np.nanmin(np.linalg.norm(COM - c[:, None, :], axis=2), axis=1)
    h = (dmin < 4.0).astype(float); h[~has_lig] = np.nan
    return h
rate = lambda h: float(np.nanmean(h))
''')
md("## Cell 6 — Per-seed pooled DCC and the pre-specified paired comparisons")
code(r'''
from scipy.stats import ttest_rel, wilcoxon
import pandas as pd
per_seed = {a: np.array([rate(hits(a, (s,))) for s in range(8)]) for a in ARMS}
ens = {a: hits(a, range(8)) for a in ARMS}
print(f"{'arm':<10}{'DCC mean(8 seeds)':>19}{'std':>8}{'8-seed ensemble':>17}")
for a in ARMS: print(f"{a:<10}{per_seed[a].mean():>19.4f}{per_seed[a].std():>8.4f}{rate(ens[a]):>17.4f}")
COMPS = [("geom+topo", "geom"), ("full", "no-topo"), ("geom", "no-topo"), ("geom", "full"), ("geom+topo", "full")]
rng = np.random.default_rng(254); rows = []
print(f"\n{'comparison (a - b)':<24}{'per-seed delta':>15}{'wins':>6}{'t p':>9}{'W p':>9}   ensemble delta [95% bootstrap over structures]")
for a, b in COMPS:
    d = per_seed[a] - per_seed[b]; tp, wp = ttest_rel(per_seed[a], per_seed[b]).pvalue, wilcoxon(d).pvalue
    ok = has_lig; ha, hb = ens[a][ok], ens[b][ok]; idx = rng.integers(0, len(ha), size=(5000, len(ha)))
    bs = ha[idx].mean(1) - hb[idx].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append(dict(a=a, b=b, mean_seed_delta=float(d.mean()), wins=int((d > 0).sum()), t_p=float(tp), wilcoxon_p=float(wp),
                     ens_delta=float(ha.mean() - hb.mean()), ens_lo=float(lo), ens_hi=float(hi), n=int(ok.sum())))
    print(f"{a+' - '+b:<24}{d.mean():>+15.4f}{int((d>0).sum()):>4}/8{tp:>9.1e}{wp:>9.1e}   {ha.mean()-hb.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] n={int(ok.sum())}")
pd.DataFrame(rows).to_csv(WORK/"phase254_paired_stats.csv", index=False)
print("Phase 86 reference (8 seeds, COACH420 pooled DCC): full 0.5932, no_topo 0.4286")
''')
md("## Cell 7 — Ensemble-size sweep and two-way (seeds x structures) bootstrap")
code(r'''
rows = []
for a, b in COMPS:
    for k in (1, 2, 3, 4, 6, 8):
        ds = np.array([rate(hits(a, S)) - rate(hits(b, S)) for S in combinations(range(8), k)])
        rows.append(dict(a=a, b=b, k=k, n_subsets=len(ds), mean_delta=float(ds.mean()), sd=float(ds.std()), frac_positive=float((ds > 0).mean())))
df_k = pd.DataFrame(rows); df_k.to_csv(WORK/"phase254_ensemble_size.csv", index=False); print(df_k.round(4).to_string(index=False))
halves = []
for S in combinations(range(8), 4):
    if 0 not in S: continue
    T = tuple(x for x in range(8) if x not in S)
    for H in (S, T): halves.append(rate(hits("geom+topo", H)) - rate(hits("geom", H)))
halves = np.array(halves)
print(f"\nDisjoint 4+4 splits, geom+topo - geom: {len(halves)} half-ensemble deltas, mean {halves.mean():+.4f}, sd {halves.std():.4f}, positive {int((halves>0).sum())}/{len(halves)}")
res = {}; B = 1000
for a, b in COMPS:
    dl = []
    for _ in range(B):
        seeds = [int(x) for x in rng.integers(0, 8, size=8)]
        ha, hb = hits(a, seeds), hits(b, seeds); idx = rng.integers(0, len(ha), size=len(ha))
        dl.append(np.nanmean(ha[idx]) - np.nanmean(hb[idx]))
    dl = np.array(dl); lo, hi = np.percentile(dl, [2.5, 97.5])
    res[f"{a} - {b}"] = dict(mean=float(dl.mean()), lo=float(lo), hi=float(hi), frac_pos=float((dl > 0).mean()))
    print(f"{a+' - '+b:<22} two-way bootstrap ensemble delta {dl.mean():+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  P(delta>0) = {(dl>0).mean():.3f}")
json.dump(dict(per_seed={a: per_seed[a].tolist() for a in ARMS}, ensemble_rate={a: rate(ens[a]) for a in ARMS},
               two_way=res, n_structures=len(valid_keys), n_with_ligand=int(has_lig.sum()), n_train=int(sa["n_train"])),
          open(WORK/"phase254_summary.json", "w"), indent=2)
print("saved phase254_paired_stats.csv, phase254_ensemble_size.csv, phase254_summary.json")
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
o = HERE / "phase254_loc_score_scale3.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
meta = {"id": "blackpearls1/phase254-loc-score-scale3", "title": "Phase254 Loc Score Scale3", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/phase253a-loc-train-scale3", "blackpearls1/phase253b-loc-train-scale3",
                           "blackpearls1/tda-egnn-phase53-scaled3-coach420", "blackpearls1/phase252-coach-geometry",
                           "blackpearls1/tda-egnn-phase51-topology-scale3"]}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
