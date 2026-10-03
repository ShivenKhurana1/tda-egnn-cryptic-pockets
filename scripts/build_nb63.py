#!/usr/bin/env python3
"""Phase 63: combine and score all six Thread-1 round-3 arms. No training
here -- loads full (Phase53/56), no_topo (salvaged Phase58 checkpoints, now
a dataset), and H0/H1/H2/burial (Phase59-62) checkpoints, featurizes the
COACH420 benchmark fresh (needed since Phase58's featurization never
completed before its cancellation), scores all six arms at matched 3
seeds, reports the full pairwise significance table."""
import json, pickle
from pathlib import Path

C58 = pickle.load(open("/tmp/p58_cells.pkl", "rb"))
s58 = lambda i: (C58[i]["source"] if isinstance(C58[i]["source"], str) else "".join(C58[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 63 - Combine and Score All Six Thread-1 Round-3 Arms

Loads all six already-trained arms (`full` from Phase53/56, `no_topo`
salvaged from Phase58's cancelled run, `H0/H1/H2/burial_only` from
Phase59-62) and scores them on the COACH420 benchmark at matched 3 seeds
each. No training in this notebook -- just featurization (needed fresh,
since Phase58 was cancelled before reaching this stage) and scoring, so it
should be much cheaper than the training notebooks.
''')

md(r'''## Cell 1 - Setup (loads Phase 52/51's round-3 scaled blobs, identical to Phase53/56/58-62)''')
code(s58(2))
code(s58(3))

md(r'''## Cell 2 - Fetch ALL 420 COACH420 structures (identical to Phase 53/56/58-62)''')
code(s58(5))

md(r'''## Cell 3 - Rebuild the leakage-clean chain list (identical logic; needed only for the AA3_TO_1 dict this cell defines, not for training)''')
code(s58(7))

md(r'''## Cell 4 - Model (identical to every prior localization phase)''')
code(s58(11))

md(r'''## Cell 5 - Build H0 / H1 / H2 / burial column masks (identical to Phase 23/58-62)''')
code(s58(15))

md(r'''## Cell 6 - Load all six arms' checkpoints (matched 3 seeds each, no retraining)''')
code(r'''
def load_arm(prefix, seeds=(0,1,2)):
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
assert all(len(m) == 3 for m in arm_models.values()), \
    "expected exactly 3 matched seeds per arm -- check kernel_sources/dataset_sources"
print("\nall six arms matched at 3 seeds")
''')

md(r'''## Cell 7 - Ground-truth ligands for ALL 420 COACH420 structures (identical to Phase 53/56/58-62)''')
code(s58(19))

md(r'''## Cell 8 - Featurize (identical pipeline: dihedral + ESM2 + topology; cache filename updated)''')
code(s58(21).replace("cache/phase53_raw_features.pkl", "cache/phase63_raw_features.pkl"))

md(r'''## Cell 9 - P2Rank on the FULL 420 (same-run controlled reference, identical to Phase 53/56/58-62)''')
code(s58(23).replace('Path("/tmp/p2_53")', 'Path("/tmp/p2_63")')
            .replace('Path("/tmp/p2_53_out")', 'Path("/tmp/p2_63_out")'))

md(r'''## Cell 10 - Score all six arms, per-seed pairwise significance table''')
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

md(r'''## Cell 11 - Per-seed pairwise significance table across all four decomposition arms''')
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
          open(WORK/"phase63_summary.json", "w"), indent=2)
print("\nsaved -> phase63_summary.json")
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
o=Path("/Users/shivenkhurana/bioproj/phase63_combine_score_scale3.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
