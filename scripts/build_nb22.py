#!/usr/bin/env python3
"""Phase 22: multi-seed validation of Phase 18's finding (topology improves
localization, not just classification). Same rigor just applied to the
P2Rank gap in Phase 21 -- reuses Phase 15/18's cached checkpoints and
benchmark features, trains 5 more seeds per arm (8 total), measures whether
the +0.029 pooled DCC gap survives proper seed-variance scrutiny."""
import json, pickle
from pathlib import Path

c14 = pickle.load(open("/tmp/p14_cells.pkl", "rb"))
s14 = lambda i: c14[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 22 — Is "Topology Helps Localization" Real, or Also Seed Noise?

Phase 18 found topology improves pocket-center localization specifically
(+0.0289 pooled DCC), not just classification — a genuinely novel claim, since
no prior work combines trained equivariant localization with topological
features. But it was a **single run**, exactly the kind of number Phase 21
just showed can be dominated by seed variance (a 0.027 single-seed std against
a 0.044 "finding" in that case). This project's actual novel contribution
should not get an exemption from the scrutiny just applied to a less
interesting result.

**Reuses everything expensive:** Phase 15's 3 `full` (with-topology) and Phase
18's 3 `no_topo` checkpoints are loaded, not retrained. 5 more seeds are
trained per arm (8 total each), and all scoring happens on Phase 15's already
-cached benchmark features (COACH420 + HOLO4K, the full 1,149-unit
leakage-clean set) — no re-fetching, no re-featurization.
''')

md(r'''## Cell 1 — Setup, rebuild training data with true centers (Phase 14 logic)''')
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
print(f"deployment training: {n_tr} train / {n_va} val  (matches Phase 15/18's 2,056-chain setup)")
''')

md(r'''## Cell 2 — Model (identical to Phase 14/15/18)''')
code(s14(7))

md(r'''## Cell 3 — Training machinery (identical to Phase 14)''')
code(s14(11))

md(r'''
## Cell 4 — Load existing checkpoints, train 5 more seeds per arm (8 total)

Topology is zeroed for the `no_topo` arm during training, exactly as Phase 18
did it -- same architecture, same budget, only the input differs.
''')
code(r'''
import time
WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)

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

full_models = load_existing("loc_deploy_seed", (0,1,2))
notopo_models = load_existing("notopo_deploy_seed", (0,1,2))
print(f"loaded {len(full_models)} 'full' + {len(notopo_models)} 'no_topo' existing checkpoints")

ORIG_X = [d.x.clone() for d in dataset]
def zero_topology():
    for d, x0 in zip(dataset, ORIG_X):
        t = x0[:, N_PRE:].clone(); t.zero_()
        d.x = torch.cat([x0[:, :N_PRE], t], dim=1)
def restore_topology():
    for d, x0 in zip(dataset, ORIG_X):
        d.x = x0.clone()

BEST_LAMBDA = 0.1
t0 = time.time()

for seed in (3,4,5,6,7):
    restore_topology()
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    full_models[seed] = m
    torch.save(m.state_dict(), WORK/f"cache/loc_deploy_seed{seed}.pt")
    print(f"  [full] seed {seed}: val PR {val_pr:.4f}  val DCC {v.get('dcc',float('nan')):.4f}  "
          f"[{(time.time()-t0)/60:.1f}m]")

for seed in (3,4,5,6,7):
    zero_topology()
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    notopo_models[seed] = m
    torch.save(m.state_dict(), WORK/f"cache/notopo_deploy_seed{seed}.pt")
    print(f"  [no_topo] seed {seed}: val PR {val_pr:.4f}  val DCC {v.get('dcc',float('nan')):.4f}  "
          f"[{(time.time()-t0)/60:.1f}m]")

restore_topology()
print(f"\n{len(full_models)} 'full' + {len(notopo_models)} 'no_topo' seeds available (0-7 each)")
''')

md(r'''## Cell 5 — Load Phase 15's cached benchmark features (COACH420 + HOLO4K, no re-fetch)''')
code(r'''
def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

with open(locate("phase15_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)
print(f"loaded cached features for {len(raw_cache):,} chains (COACH420 + HOLO4K, no re-featurization)")

blob6b_bench = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)
mu_b, sd_b = blob6b_bench["topo_mean"], blob6b_bench["topo_std"]
''')

md(r'''## Cell 6 — Per-seed and per-ensemble-size DCC, both arms''')
code(r'''
from collections import defaultdict

@torch.no_grad()
def predict_all_seeds(entry, model_dict, zero_topo=False):
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
    offset = entry["ca"].mean(axis=0)
    out = {}
    for seed, m in model_dict.items():
        lg, pc = m(d)
        out[seed] = {"prob_sum": float(torch.sigmoid(lg).sum().cpu()),
                     "center": pc.cpu().numpy()[0] + offset}
    return out


def dcc(pred_center, ligs):
    if pred_center is None or not ligs: return None
    best = np.inf
    for lig_atoms in ligs:
        if len(lig_atoms) == 0: continue
        d_ = float(np.linalg.norm(pred_center - lig_atoms.mean(axis=0)))
        best = min(best, d_)
    return best if np.isfinite(best) else None


t0 = time.time()
by_structure = defaultdict(list)
for key, entry in raw_cache.items():
    by_structure[key[:2]].append((key[2], entry))

full_out, notopo_out, ligs_of = {}, {}, {}
for (src_, pid), chain_list in by_structure.items():
    ligs_of[(src_,pid)] = chain_list[0][1]["ligs"]
    full_out[(src_,pid)] = [(cid, predict_all_seeds(e, full_models, zero_topo=False)) for cid, e in chain_list]
    notopo_out[(src_,pid)] = [(cid, predict_all_seeds(e, notopo_models, zero_topo=True)) for cid, e in chain_list]
print(f"inference (8 seeds x 2 arms x {len(by_structure)} structures) in {(time.time()-t0)/60:.1f} min")


def dcc_rate_for_seedset(chain_output_dict, seeds):
    hits, n = 0, 0
    for key, chain_list in chain_output_dict.items():
        best_conf, best_center = -np.inf, None
        for cid, per_seed in chain_list:
            centers = [per_seed[s]["center"] for s in seeds if s in per_seed]
            confs = [per_seed[s]["prob_sum"] for s in seeds if s in per_seed]
            if not centers: continue
            c = np.mean(centers, axis=0); conf = np.mean(confs)
            if conf > best_conf: best_conf, best_center = conf, c
        d_ = dcc(best_center, ligs_of[key])
        if d_ is None: continue
        n += 1; hits += int(d_ < 4.0)
    return hits / max(n, 1), n


print(f"\n{'per-seed DCC':<20}{'full (topo)':>14}{'no_topo':>14}{'delta':>10}")
print("-"*58)
full_seed_dccs, notopo_seed_dccs = [], []
for seed in range(8):
    fd, _ = dcc_rate_for_seedset(full_out, (seed,))
    nd, _ = dcc_rate_for_seedset(notopo_out, (seed,))
    full_seed_dccs.append(fd); notopo_seed_dccs.append(nd)
    print(f"seed {seed:<15}{fd:>14.4f}{nd:>14.4f}{fd-nd:>10.4f}")

full_seed_dccs = np.array(full_seed_dccs); notopo_seed_dccs = np.array(notopo_seed_dccs)
print(f"\nfull:    mean {full_seed_dccs.mean():.4f}  std {full_seed_dccs.std():.4f}")
print(f"no_topo: mean {notopo_seed_dccs.mean():.4f}  std {notopo_seed_dccs.std():.4f}")
print(f"per-seed paired delta: mean {(full_seed_dccs-notopo_seed_dccs).mean():+.4f}  "
      f"std {(full_seed_dccs-notopo_seed_dccs).std():.4f}  "
      f"wins {(full_seed_dccs>notopo_seed_dccs).sum()}/8")
''')

md(r'''## Cell 7 — Ensemble-size sweep for both arms''')
code(r'''
print(f"{'size':<8}{'full mean':>12}{'no_topo mean':>14}{'delta':>10}")
print("-"*44)
sweep = {}
for size in (1, 3, 5, 8):
    seeds = tuple(range(size)) if size < 8 else tuple(range(8))
    fd, fn = dcc_rate_for_seedset(full_out, seeds)
    nd, nn = dcc_rate_for_seedset(notopo_out, seeds)
    sweep[size] = {"full": fd, "no_topo": nd, "delta": fd - nd, "n_full": fn, "n_notopo": nn}
    print(f"{size:<8}{fd:>12.4f}{nd:>14.4f}{fd-nd:>10.4f}")

full8, n8f = dcc_rate_for_seedset(full_out, tuple(range(8)))
notopo8, n8n = dcc_rate_for_seedset(notopo_out, tuple(range(8)))
delta8 = full8 - notopo8
noise = full_seed_dccs.std()

print(f"\n8-seed delta: {delta8:+.4f}  vs single-seed noise ({noise:.4f}): {abs(delta8)/max(noise,1e-9):.2f}x")
print(f"Phase 18's single-run (3 seed) result was: +0.0289 pooled DCC")

verdict = ("CONFIRMED: topology's localization benefit survives proper seed scrutiny"
          if delta8 > 0.5*noise else
          "NOT CONFIRMED at this n: the effect is within plausible seed noise, "
          "same caveat as the P2Rank gap before Phase 21")
print(f"\nVERDICT: {verdict}")

json.dump({"per_seed": {"full": full_seed_dccs.tolist(), "no_topo": notopo_seed_dccs.tolist()},
           "sweep": sweep, "delta_8seed": delta8, "noise_scale": float(noise), "verdict": verdict},
          open(WORK/"phase22_summary.json", "w"), indent=2)
print("\nsaved -> phase22_summary.json")
''')

md(r'''
---

## Reading this

Same standard as Phase 21: the 8-seed delta judged against the single-seed
noise floor, not the raw number alone. If the effect survives at several
times the noise scale, "topology helps localization" is a properly validated,
citable finding — the actual unique contribution of this project, not just a
plausible-sounding one. If it doesn't survive, that is equally worth knowing
before it goes into a writeup as a headline claim.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase22_topology_localization_variance.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
