#!/usr/bin/env python3
"""Phase 86: FINAL combine and score, all six Thread-1 round-3 arms at
full matched 8 seeds each. This is the definitive answer to whether the
round-3 preliminary 3-seed finding (Phase63: H0_only significantly weaker
than H1_only, ratio 2.20x, but H1 didn't separate from H2/burial) holds up
at the same 8-seed statistical power that made every other finding in this
project trustworthy."""
import json, pickle
from pathlib import Path

C58 = pickle.load(open("/tmp/p58_cells.pkl", "rb"))
s58 = lambda i: (C58[i]["source"] if isinstance(C58[i]["source"], str) else "".join(C58[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 86 — FINAL: Combine and Score All Six Thread-1 Round-3 Arms at Full 8 Seeds

Definitive 8-seed answer to Thread 1 at round-3 scale (7,513 clusters,
3.65x original). Phase63's preliminary 3-seed pass found H0_only
significantly weaker than H1_only (ratio 2.20x) but H1 didn't separate
from H2_only or burial_only -- this notebook re-scores with the full
statistical power (8 seeds/arm) that made every other finding in this
project trustworthy, assembled from checkpoints spread across ~30
notebooks (Phase53/56/59-85) and 3 salvage datasets after multiple
Kaggle 12hr-session-limit recoveries.
''')

md(r'''## Cell 1 — Setup (loads Phase 52/51's round-3 scaled blobs, identical to Phase53/56/58-85)''')
code(s58(2))
code(s58(3))

md(r'''## Cell 2 — Fetch ALL 420 COACH420 structures (identical to Phase 53/56/58-85)''')
code(s58(5))

md(r'''## Cell 3 — Rebuild the leakage-clean chain list (identical logic; needed only for the AA3_TO_1 dict this cell defines, not for training)''')
code(s58(7))

md(r'''## Cell 4 — Model (identical to every prior localization phase)''')
code(s58(11))

md(r'''## Cell 5 — Build H0 / H1 / H2 / burial column masks (identical to Phase 23/58-85)''')
code(s58(15))

md(r'''## Cell 6 — Load all six arms'' checkpoints at FULL 8 seeds each (no retraining -- assembled from ~30 notebooks + 3 salvage datasets)''')
code(r'''
def load_arm(prefix, seeds=range(8)):
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

arm_models = {
    "full":         load_arm("clean_deploy_scaled3_seed"),
    "no_topo":      load_arm("no_topo_scaled3_seed"),
    "H0_only":      load_arm("H0_only_scaled3_seed"),
    "H1_only":      load_arm("H1_only_scaled3_seed"),
    "H2_only":      load_arm("H2_only_scaled3_seed"),
    "burial_only":  load_arm("burial_only_scaled3_seed"),
}
for name, models in arm_models.items():
    print(f"  {name:<14} {len(models)} checkpoints (seeds {sorted(models)})")
assert all(len(m) == 8 for m in arm_models.values()), \
    "expected exactly 8 matched seeds per arm -- check kernel_sources/dataset_sources for gaps"
print("\nall six arms matched at FULL 8 seeds")
''')

md(r'''## Cell 7 — Ground-truth ligands for ALL 420 COACH420 structures (identical to Phase 53/56/58-85)''')
code(s58(19))

md(r'''## Cell 8 — Featurize (identical pipeline: dihedral + ESM2 + topology; cache filename updated)''')
code(s58(21).replace("cache/phase53_raw_features.pkl", "cache/phase86_raw_features.pkl"))

md(r'''## Cell 9 — P2Rank on the FULL 420 (same-run controlled reference, identical to Phase 53/56/58-85)''')
code(s58(23).replace('Path("/tmp/p2_53")', 'Path("/tmp/p2_86")')
            .replace('Path("/tmp/p2_53_out")', 'Path("/tmp/p2_86_out")'))

md(r'''## Cell 10 — Score all six arms at full 8 seeds, per-seed pairwise significance table''')
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

print(f"{'arm':<14}{'DCC (8-seed matched)':>22}{'n':>6}")
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
print(f"\nmatched 8-seed ranking: {' > '.join(ranked)}")
''')

md(r'''## Cell 11 — Per-seed pairwise significance table across all four decomposition arms (the definitive test)''')
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
print(f"\n{'pair':<28}{'mean diff':>12}{'noise (max std)':>18}{'wins/8':>10}{'diff/noise':>12}")
print("-"*82)
pairwise = {}
for a, b in combinations(decomp_arms, 2):
    ra, rb = per_seed[a], per_seed[b]
    diff = ra.mean() - rb.mean()
    noise = max(ra.std(), rb.std())
    wins = int((ra > rb).sum())
    ratio = abs(diff) / noise if noise > 1e-9 else float("inf")
    pairwise[f"{a}_vs_{b}"] = {"diff": float(diff), "noise": float(noise), "wins": wins, "ratio": float(ratio)}
    print(f"{a+' vs '+b:<28}{diff:>+12.4f}{noise:>18.4f}{wins:>7}/8{ratio:>12.2f}x")

max_ratio = max(v["ratio"] for v in pairwise.values())
verdict = ("ATTRIBUTABLE -- at least one pair clears 1.5x noise at full 8-seed power" if max_ratio > 1.5 else
           "STILL DIFFUSE at 8-seed power -- Phase63's 3-seed 'attributable' reading does not survive "
           "the same statistical rigor that validated every other finding in this project")
print(f"\nVERDICT: {verdict}")
print(f"(max pairwise ratio: {max_ratio:.2f}x)")

json.dump({"summary": summary, "per_seed": {k: v.tolist() for k,v in per_seed.items()},
           "pairwise": pairwise, "ranked": ranked, "verdict": verdict,
           "phase63_3seed_verdict": "ATTRIBUTABLE (H0 vs H1 ratio 2.20x) -- pending 8-seed confirmation"},
          open(WORK/"phase86_summary.json", "w"), indent=2)
print("\nsaved -> phase86_summary.json")
''')

md(r'''
---

## Reading this

- **Still diffuse (no pair clears 1.5x at 8 seeds)** -- confirms Thread 1's
  original-scale finding is scale-independent after all; Phase63's 3-seed
  "attributable" reading was itself noise that resolved once given proper
  statistical power, mirroring exactly what happened with Phase23 v1's
  false H0-dominant reading. A genuinely stable, final negative result.
- **H0 vs H1 (or another pair) still clears 1.5x at 8 seeds** -- Phase63's
  finding holds up under full rigor; this would be the first genuinely
  attributable topology-decomposition result in the whole project, worth
  the significant compute this thread has cost.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase86_combine_score_8seed_scale3.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
