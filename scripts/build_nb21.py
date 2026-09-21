#!/usr/bin/env python3
"""Phase 21: is the -0.044 DCC gap to current P2Rank real, or seed noise?
Trains 5 more seeds (total 8), reuses Phase 20's cached benchmark features
(no re-fetch/re-featurize) to measure seed-to-seed variance and whether
larger ensembles close the gap."""
import json, pickle
from pathlib import Path

c14 = pickle.load(open("/tmp/p14_cells.pkl", "rb"))
s14 = lambda i: c14[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 21 — Is the P2Rank Gap Real, or Seed Noise?

Phase 20 found a legitimate, apples-to-apples result: the model beats every
published literature baseline (fpocket, P2Rank-2018, EquiPocket-2023) on the
full COACH420 benchmark, but **loses to P2Rank's current tool version** by
−0.044 DCC (0.477 vs 0.521), the one fully contemporaneous comparison
available. That single number decided the "are we actually the best" verdict
— and this project has repeatedly learned that single-run numbers on small
test sets are not to be trusted at face value (Phase 5's sign flip between one
split and 5-fold CV; Phase 19's tune-set "winner" that evaporated on held-out
data). A 3-seed ensemble on 413 structures deserves the same skepticism before
the −0.044 gets treated as a settled fact.

**This is deliberately cheap.** Phase 20 cached the benchmark's raw features
(`phase20_raw_features.pkl`) and its 3 trained checkpoints. Training 5 more
seeds (8 total) costs about 15-20 minutes; rescoring against the cached
features costs seconds. No re-fetching, no re-running the ~50-minute
featurization pipeline.

**What this answers:**
1. How much does individual-seed DCC vary? That sets the noise floor the
   −0.044 gap needs to be judged against.
2. Does ensembling more seeds (3 → 5 → 8) systematically improve DCC, and if
   so, does it close the gap?
''')

md(r'''## Cell 1 — Setup, rebuild the clean training set (Phase 20 logic, reused)''')
code(r'''
!pip install -q torch-geometric==2.6.1 biopython==1.84
''')

code("import pickle\n\n" + s14(3))

code(r'''
!apt-get -qq update > /dev/null 2>&1
''')

code(r'''
import csv as _csv
import gzip, shutil, subprocess, os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

def parse_ds(url):
    lines = urllib.request.urlopen(url, timeout=60).read().decode().splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        name = ln.split("/")[-1].replace(".pdb", "")
        out.append((name[:4].upper(), name[4:].upper() or None))
    return out

coach = parse_ds("https://raw.githubusercontent.com/rdk/p2rank-datasets/master/coach420.ds")

from Bio.PDB import PDBParser
AA3_TO_1={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
RAW_C = Path("/tmp/coach_raw"); RAW_C.mkdir(parents=True, exist_ok=True)

def fetch_c(pid):
    for ext in ("pdb","cif"):
        p = RAW_C / f"{pid}.{ext}.gz"
        if p.exists() and p.stat().st_size > 500: return pid, p
    for ext in ("pdb","cif"):
        try:
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.{ext}.gz", timeout=60) as r:
                b = r.read()
            if len(b) > 500: p = RAW_C/f"{pid}.{ext}.gz"; p.write_bytes(b); return pid, p
        except Exception: continue
    return pid, None

need_c = sorted({pid for pid, _ in coach})
with ThreadPoolExecutor(max_workers=24) as ex: got_c = list(ex.map(fetch_c, need_c))
coach_paths = {p: f for p, f in got_c if f}
print(f"COACH420 fetched: {len(coach_paths)}/{len(need_c)}")

def get_seq(pid, path, chain_id):
    try:
        with gzip.open(path, "rt") as fh:
            model = next(PDBParser(QUIET=True, PERMISSIVE=True).get_structure(pid, fh).get_models())
        ch = model[chain_id] if chain_id else next(iter(model))
        residues = [r for r in ch if r.get_resname().strip().upper() in AA3_TO_1 and "CA" in r]
        return "".join(AA3_TO_1.get(r.get_resname().strip().upper(),"X") for r in residues)
    except Exception:
        return None

coach_seqs = {}
for pid, ch in coach:
    if pid not in coach_paths: continue
    seq = get_seq(pid, coach_paths[pid], ch)
    if seq: coach_seqs[f"{pid}_{ch or 'A'}"] = seq
print(f"COACH420 sequences: {len(coach_seqs)}")

mm = None
try:
    subprocess.run("curl -sL https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -o /tmp/mm.tar.gz "
                   "&& tar xzf /tmp/mm.tar.gz -C /tmp", shell=True, check=True, timeout=600)
    c = Path("/tmp/mmseqs/bin/mmseqs")
    if c.exists(): os.chmod(c, 0o755); mm = str(c)
except Exception as e:
    print("mmseqs2 unavailable:", type(e).__name__)

W = Path("/tmp/rev_leak21"); shutil.rmtree(W, ignore_errors=True); W.mkdir()
fasta_lines = [f">TRAIN__{d.pdb_id}_{d.chain_id}\n{d.seq}\n" for d in dataset]
fasta_lines += [f">BENCH__{k}\n{v}\n" for k, v in coach_seqs.items()]
(W/"all.fasta").write_text("".join(fasta_lines))

subprocess.run([mm, "easy-cluster", str(W/"all.fasta"), str(W/"res"), str(W/"tmp"),
                "--min-seq-id", "0.30", "-c", "0.8", "--cov-mode", "1", "-v", "1"],
               check=True, capture_output=True, timeout=3600)
rep = defaultdict(list)
for line in (W/"res_cluster.tsv").read_text().splitlines():
    a, b = line.split("\t"); rep[a].append(b)
for r_, members in rep.items():
    if r_ not in members: members.append(r_)

leaked_train_keys = set()
for members in rep.values():
    has_train = any(m.startswith("TRAIN__") for m in members)
    has_bench = any(m.startswith("BENCH__") for m in members)
    if has_train and has_bench:
        leaked_train_keys.update(m for m in members if m.startswith("TRAIN__"))

dataset = [d for d in dataset if f"TRAIN__{d.pdb_id}_{d.chain_id}" not in leaked_train_keys]
print(f"clean training set: {len(dataset):,} chains (should match Phase 20's 1,858)")
''')

md(r'''## Cell 2 — Recover true ligand centers (Phase 14 logic, reused)''')
_p14_true_center = s14(5)
_cut = _p14_true_center.index('dataset = [d for d in dataset if hasattr(d, "true_center")]')
_true_center_core = _p14_true_center[:_cut].rstrip()
code(_true_center_core + '''

dataset = [d for d in dataset if hasattr(d, "true_center")]
print(f"clean training chains with recovered true centers: {len(dataset):,}")

rng = np.random.RandomState(42)
perm = rng.permutation(len(dataset))
n_val = max(1, int(0.08 * len(perm)))
splits = {"val": [dataset[i] for i in perm[:n_val]], "train": [dataset[i] for i in perm[n_val:]]}
n_tr, n_va = len(splits["train"]), len(splits["val"])
print(f"deployment training: {n_tr} train / {n_va} val")
''')

md(r'''## Cell 3 — Model (identical to Phase 14/15/16/18/20)''')
code(s14(7))

md(r'''## Cell 4 — Training machinery (identical to Phase 14)''')
code(s14(11))

md(r'''
## Cell 5 — Load Phase 20's 3 checkpoints, train 5 MORE seeds (8 total)

Seeds 0/1/2 are loaded, not retrained — a different random init would make
this a different experiment, not a variance check on the same one.
''')
code(r'''
import time
WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)

models = {}
for seed in (0,1,2):
    cands = []
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): cands.extend(r.rglob(f"clean_deploy_seed{seed}.pt"))
    assert cands, f"clean_deploy_seed{seed}.pt not found -- attach Phase 20's kernel output"
    state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
    m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE); m.load_state_dict(state); m.eval()
    models[seed] = m
print(f"loaded {len(models)} checkpoints from Phase 20 (no retraining)")

BEST_LAMBDA = 0.1
t0 = time.time()
for seed in (3,4,5,6,7):
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    models[seed] = m
    torch.save(m.state_dict(), WORK/f"cache/clean_deploy_seed{seed}.pt")
    val_dcc = v.get("dcc", float("nan"))
    print(f"  seed {seed}: val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")
print(f"\n{len(models)} total seeds available (0-7)")
''')

md(r'''## Cell 6 — Load Phase 20's cached benchmark features (no re-fetch)''')
code(r'''
def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

with open(locate("phase20_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)
print(f"loaded cached COACH420 features for {len(raw_cache):,} chains (no re-featurization)")

blob6b = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)
mu, sd = blob6b["topo_mean"], blob6b["topo_std"]
''')

md(r'''## Cell 7 — Per-seed and per-ensemble-size DCC on the cached benchmark''')
code(r'''
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
def predict_centers_all_seeds(entry):
    """Returns {seed: center_raw_frame} for every trained seed on one chain."""
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
        _, pc = m(d)
        out[seed] = pc.cpu().numpy()[0] + offset
    return out


t0 = time.time()
per_chain_centers = {}   # key -> {seed: center}
per_chain_ligs = {}
for key, entry in raw_cache.items():
    per_chain_centers[key] = predict_centers_all_seeds(entry)
    per_chain_ligs[key] = entry["ligs"]
print(f"inference (8 seeds x {len(raw_cache)} chains) in {(time.time()-t0)/60:.1f} min")


def dcc_rate(center_fn):
    hits, n = 0, 0
    for key in raw_cache:
        c = center_fn(key)
        if c is None: continue
        d_, _ = dcc_dca(c, per_chain_ligs[key])
        if d_ is None: continue
        n += 1; hits += int(d_ < 4.0)
    return hits / max(n, 1), n


print(f"\n{'per-seed DCC (individual models, no ensembling)':<55}")
print("-"*55)
seed_dccs = []
for seed in range(8):
    dcc, n = dcc_rate(lambda k, s=seed: per_chain_centers[k][s])
    seed_dccs.append(dcc)
    print(f"  seed {seed}: DCC {dcc:.4f}  (n={n})")

seed_dccs = np.array(seed_dccs)
print(f"\nseed-to-seed: mean {seed_dccs.mean():.4f}  std {seed_dccs.std():.4f}  "
      f"range [{seed_dccs.min():.4f}, {seed_dccs.max():.4f}]")
print(f"Phase 20's 3-seed ensemble reported 0.4770 -- for reference against this spread")
''')

md(r'''## Cell 8 — Ensemble size sweep: does averaging more seeds close the gap?''')
code(r'''
import itertools

P2RANK_THIS_RUN = 0.5209   # Phase 20's same-run, same-benchmark P2Rank number

def ensemble_dcc(seed_subset):
    def center_fn(key, seeds=seed_subset):
        cs = [per_chain_centers[key][s] for s in seeds]
        return np.mean(cs, axis=0)
    return dcc_rate(center_fn)


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
print(f"Phase 20's 3-seed (0,1,2 only) for direct comparison:")
p20_dcc, p20_n = ensemble_dcc((0,1,2))
print(f"  seeds (0,1,2): DCC {p20_dcc:.4f} (n={p20_n})  [Phase 20 reported 0.4770]")
''')

md(r'''## Cell 9 — Verdict''')
code(r'''
gap_3seed = p20_dcc - P2RANK_THIS_RUN
gap_8seed = full8_dcc - P2RANK_THIS_RUN
noise_scale = seed_dccs.std()

print(f"{'='*66}")
print(f"IS THE P2RANK GAP REAL, OR SEED NOISE?")
print(f"{'='*66}")
print(f"single-seed DCC std across 8 independent runs: {noise_scale:.4f}")
print(f"3-seed ensemble gap to P2Rank:  {gap_3seed:+.4f}")
print(f"8-seed ensemble gap to P2Rank:  {gap_8seed:+.4f}")
print(f"gap magnitude vs single-seed noise: {abs(gap_8seed)/max(noise_scale,1e-9):.2f}x")

if gap_8seed > 0.01:
    verdict = "gap CLOSED with more seeds -- the 3-seed result understated the model"
elif gap_8seed < -0.03 and abs(gap_8seed) > 1.5*noise_scale:
    verdict = "gap PERSISTS beyond noise -- current P2Rank is genuinely ahead on DCC, confirmed"
else:
    verdict = "gap is WITHIN plausible seed noise -- not confidently resolved either way at this n"

print(f"\nVERDICT: {verdict}")

json.dump({"seed_dccs": seed_dccs.tolist(), "ensemble_sweep": results,
           "phase20_3seed_dcc": p20_dcc, "full8_dcc": full8_dcc,
           "p2rank_this_run": P2RANK_THIS_RUN, "verdict": verdict},
          open(WORK/"phase21_summary.json", "w"), indent=2)
print("\nsaved -> phase21_summary.json")
''')

md(r'''
---

## Reading this

The single-seed DCC spread is the noise floor. If the 8-seed-ensemble gap to
P2Rank is small relative to that spread, the honest conclusion is "not
resolved yet" — not "we lose" and not "we win." If it's large relative to the
spread, the Phase 20 verdict is confirmed with actual statistical grounding
behind it, rather than resting on one 3-seed run.

Either way, this notebook's real output is `models/clean_deploy_seed{3..7}.pt`
— five more trained checkpoints, saved for any future ensemble use.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase21_seed_variance.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
