#!/usr/bin/env python3
"""Phase 57: does Phase 54's round-3 HOLO4K result (-0.0047 DCC gap,
3-seed, 3.65x pool -- smaller in magnitude than round-2's own 3-seed gap
of -0.0097, and round-2's win flipped fully positive at 8 seeds) hold or
grow further at round-3 scale? Mirrors Phase34/41's exact treatment
(confidence-based multi-chain arbitration, proven fix) but pointed at
Phase 54's round-3 scaled pool."""
import json, pickle
from pathlib import Path

# self-contained, hand-written cells -- no splicing needed for this notebook

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 57 — Does the Round-3 HOLO4K Result Hold or Grow at 8 Seeds?

Phase 54 (3-seed, 3.65x pool) found DCC 0.6533 vs live P2Rank's 0.6580 --
gap only -0.0047, smaller in magnitude than round-2's own 3-seed gap
(-0.0097), which itself flipped fully positive (+0.0124) at 8 seeds
(Phase41). This notebook applies the identical stress test: trains 5 more
seeds (8 total) on Phase 54's leakage-clean training set, rescores against
Phase54's cached benchmark features -- using the same confidence-based
multi-chain arbitration fix validated in Phase34/41.
''')

md(r'''## Cell 1 — Setup (loads Phase 52/51's round-3 scaled blobs)''')
code(r'''
!pip install -q torch-geometric==2.6.1 gudhi==3.10.1 biopython==1.84 transformers==4.46.3
''')
code(r'''
import csv as _csv
import gzip, json, os, pickle, shutil, subprocess, time, urllib.request, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

warnings.filterwarnings("ignore")

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

blob10 = torch.load(locate("phase52_graphs.pt"), map_location="cpu", weights_only=False)
blob6b = torch.load(locate("phase51_graphs.pt"), map_location="cpu", weights_only=False)
dataset = blob10["dataset"]
N_PRE = int(blob10["n_pre_topo"]); TOPO_DIM = int(blob10["topo_dim"])
POS_WEIGHT = float(blob10["pos_weight"])
TOPO_NAMES = blob6b["topo_names"]

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
print(f"training set: {len(dataset):,} chains")

WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)
RAW = Path("/tmp/pdb_raw"); RAW.mkdir(parents=True, exist_ok=True)
''')

md(r'''## Cell 2 — Sample the SAME ~1,000 HOLO4K structures (seed=2024, identical to Phase 24/33/39/54)''')
code(r'''
def parse_ds(url):
    lines = urllib.request.urlopen(url, timeout=60).read().decode().splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        name = ln.split("/")[-1].replace(".pdb", "")
        out.append(name[:4].upper())
    return out

holo_all = parse_ds("https://raw.githubusercontent.com/rdk/p2rank-datasets/master/holo4k.ds")
print(f"HOLO4K full: {len(holo_all)} entries")

rng = np.random.RandomState(2024)
holo_ids = sorted(holo_all); rng.shuffle(holo_ids)
holo_sample = sorted(holo_ids[:1000])
print(f"sampled {len(holo_sample)} structures (seed=2024, reproducible)")

need = holo_sample
def fetch(pid):
    for ext in ("pdb","cif"):
        p = RAW / f"{pid}.{ext}.gz"
        if p.exists() and p.stat().st_size > 500: return pid, p
    for ext in ("pdb","cif"):
        try:
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.{ext}.gz", timeout=60) as r:
                b = r.read()
            if len(b) > 500: p = RAW/f"{pid}.{ext}.gz"; p.write_bytes(b); return pid, p
        except Exception: continue
    return pid, None

t0=time.time()
with ThreadPoolExecutor(max_workers=24) as ex: got = list(ex.map(fetch, need))
holo_paths = {p: f for p, f in got if f}
print(f"fetched {len(holo_paths):,}/{len(need):,} in {(time.time()-t0)/60:.1f} min")
''')

md(r'''## Cell 3 — Clean the TRAINING side (identical logic; pool size 7513, should reproduce Phase 54's clean chain count)''')
code(r'''
from Bio.PDB import PDBParser
AA3_TO_1={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}

def get_all_seqs(pid, path):
    out = {}
    try:
        with gzip.open(path, "rt") as fh:
            model = next(PDBParser(QUIET=True, PERMISSIVE=True).get_structure(pid, fh).get_models())
        for ch in model:
            residues = [r for r in ch if r.get_resname().strip().upper() in AA3_TO_1 and "CA" in r]
            if len(residues) < 10: continue
            out[ch.id] = "".join(AA3_TO_1.get(r.get_resname().strip().upper(),"X") for r in residues)
    except Exception:
        pass
    return out

holo_seqs = {}
for pid in sorted(holo_paths):
    for cid, seq in get_all_seqs(pid, holo_paths[pid]).items():
        holo_seqs[f"{pid}_{cid}"] = seq
print(f"HOLO4K sample sequences (all chains): {len(holo_seqs)}")

mm = None
try:
    subprocess.run("curl -sL https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -o /tmp/mm.tar.gz "
                   "&& tar xzf /tmp/mm.tar.gz -C /tmp", shell=True, check=True, timeout=600)
    c = Path("/tmp/mmseqs/bin/mmseqs")
    if c.exists(): os.chmod(c, 0o755); mm = str(c)
except Exception as e:
    print("mmseqs2 unavailable:", type(e).__name__)

W = Path("/tmp/rev_leak41"); shutil.rmtree(W, ignore_errors=True); W.mkdir()
fasta_lines = [f">TRAIN__{d.pdb_id}_{d.chain_id}\n{d.seq}\n" for d in dataset]
fasta_lines += [f">BENCH__{k}\n{v}\n" for k, v in holo_seqs.items()]
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
print(f"training chains homologous to the HOLO4K sample (excluded): "
      f"{5339-len(dataset)}/5339 -- clean training set: {len(dataset):,} chains")
assert len(dataset) > 0.7 * 5339, "too much training data excluded -- investigate before proceeding"
''')

md(r'''## Cell 4 — Recover true ligand centers (Phase 54/33/24/14 logic, reused)''')
code(r'''
NON_LIGAND={"HOH","DOD","WAT","SO4","PO4","GOL","EDO","PEG","PG4","PGE","1PE","MPD","TRS",
            "ACT","ACY","FMT","CIT","MES","EPE","DMS","IMD","BME","NO3","AZI","CO3","IOD",
            "BR","CL","F","NA","K","CS","RB","LI","UNX","UNL","UNK"}
CATALYTIC_METALS={"ZN","MG","MN","FE","FE2","CU","CU1","CA","NI","CO","CD","W","MO"}
MIN_LIG_ATOMS = 6

need = sorted({d.pdb_id for d in dataset})
def fetch2(pid):
    for ext in ("pdb","cif"):
        p = RAW / f"{pid}.{ext}.gz"
        if p.exists() and p.stat().st_size > 500: return pid, p
    for ext in ("pdb","cif"):
        try:
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.{ext}.gz", timeout=60) as r:
                b = r.read()
            if len(b) > 500: p = RAW/f"{pid}.{ext}.gz"; p.write_bytes(b); return pid, p
        except Exception: continue
    return pid, None

t0=time.time()
with ThreadPoolExecutor(max_workers=16) as ex: got = list(ex.map(fetch2, need))
paths = {p: f for p, f in got if f}
print(f"fetched {len(paths):,}/{len(need):,} in {(time.time()-t0)/60:.1f} min")

def ligands_and_ca(pid, path, chain_id, n_expect):
    from Bio.PDB import MMCIFParser
    try:
        parser = MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True, PERMISSIVE=True)
        with gzip.open(path,"rt") as fh:
            model = next(parser.get_structure(pid, fh).get_models())
        ch = model[chain_id]
    except Exception:
        return None
    residues=[r for r in ch if r.get_resname().strip().upper() in AA3_TO_1 and "CA" in r]
    if len(residues) != n_expect: return None
    raw_ca = np.asarray([r["CA"].coord for r in residues], dtype=np.float32)

    ligs = []
    for c2 in model:
        for res in c2:
            if res.id[0] == " ": continue
            nm = res.get_resname().strip().upper()
            if nm in AA3_TO_1 or nm in NON_LIGAND: continue
            xyz = np.asarray([a.coord for a in res.get_unpacked_list() if a.element not in ("H","D")],
                             dtype=np.float32).reshape(-1,3)
            is_metal = nm in CATALYTIC_METALS
            if is_metal or len(xyz) >= MIN_LIG_ATOMS: ligs.append(xyz)
    return raw_ca, ligs

t0=time.time(); n_ok=0; n_geom_fail=0; n_no_ligand=0
for d in dataset:
    if d.pdb_id not in paths: continue
    r = ligands_and_ca(d.pdb_id, paths[d.pdb_id], d.chain_id, int(d.num_nodes))
    if r is None: continue
    raw_ca, ligs = r
    if not ligs: n_no_ligand += 1; continue
    k = min(50, len(raw_ca))
    a = np.linalg.norm(raw_ca[:k,None]-raw_ca[None,:k], axis=-1)
    b = np.linalg.norm(d.pos.numpy()[:k,None]-d.pos.numpy()[None,:k], axis=-1)
    if float(np.abs(a-b).max()) > 1e-2:
        n_geom_fail += 1; continue
    offset = raw_ca.mean(axis=0)
    y_np = d.y.numpy().astype(bool)
    contact_centroid = raw_ca[y_np].mean(axis=0) if y_np.any() else raw_ca.mean(axis=0)
    coms = [lig.mean(axis=0) for lig in ligs]
    best = int(np.argmin([np.linalg.norm(c - contact_centroid) for c in coms]))
    d.true_center = torch.from_numpy((coms[best] - offset).astype(np.float32))
    n_ok += 1

print(f"\ntrue centers recovered: {n_ok:,}/{len(dataset):,} "
      f"({(time.time()-t0)/60:.1f} min) | geometry mismatches: {n_geom_fail} | no ligand: {n_no_ligand}")
dataset = [d for d in dataset if hasattr(d, "true_center")]
print(f"clean training chains with recovered true centers: {len(dataset):,}")

rng2 = np.random.RandomState(42)
perm = rng2.permutation(len(dataset))
n_val = max(1, int(0.08 * len(perm)))
splits = {"val": [dataset[i] for i in perm[:n_val]], "train": [dataset[i] for i in perm[n_val:]]}
n_tr, n_va = len(splits["train"]), len(splits["val"])
print(f"deployment training: {n_tr} train / {n_va} val")
''')

md(r'''## Cell 5 — Model (identical to every prior localization phase)''')
code(r'''
class EGCL(nn.Module):
    def __init__(s,h,ea=2,act=nn.SiLU,uc=True):
        super().__init__(); s.uc=uc
        s.e=nn.Sequential(nn.Linear(2*h+1+ea,h),act(),nn.Linear(h,h),act())
        s.n=nn.Sequential(nn.Linear(2*h,h),act(),nn.Linear(h,h))
        s.a=nn.Sequential(nn.Linear(h,1),nn.Sigmoid())
        if uc:
            s.c=nn.Sequential(nn.Linear(h,h),act(),nn.Linear(h,1,bias=False))
            nn.init.xavier_uniform_(s.c[-1].weight,gain=0.001)
        s.norm=nn.LayerNorm(h)
    def forward(s,h,x,ei,ea):
        src,dst=ei; rel=x[src]-x[dst]; d2=(rel**2).sum(-1,keepdim=True)
        m=s.e(torch.cat([h[src],h[dst],d2,ea],-1)); m=m*s.a(m)
        agg=torch.zeros_like(h).index_add_(0,dst,m)
        deg=torch.zeros(h.size(0),1,device=h.device).index_add_(0,dst,torch.ones_like(d2)).clamp(min=1)
        h=s.norm(h+s.n(torch.cat([h,agg/deg],-1)))
        if s.uc:
            w=s.c(m)*rel/(d2.sqrt()+1.0); x=x+torch.zeros_like(x).index_add_(0,dst,w)/deg
        return h,x

class Fus(nn.Module):
    def __init__(s,h,t):
        super().__init__(); s.p=nn.Linear(t,h); s.g=nn.Sequential(nn.Linear(h+t,h),nn.Sigmoid())
    def forward(s,h,t): return h+s.g(torch.cat([h,t],-1))*s.p(t)

class LocalizingEGNN(nn.Module):
    def __init__(s, n_pre, td, h=128, L=4, dr=0.1, use_loc_head=True):
        super().__init__(); s.n_pre=n_pre; s.use_loc_head=use_loc_head
        s.embed=nn.Sequential(nn.Linear(n_pre+td,h),nn.SiLU(),nn.Linear(h,h))
        s.ls=nn.ModuleList([EGCL(h,uc=(i<L-1)) for i in range(L)])
        s.fs=nn.ModuleList([Fus(h,td) for _ in range(L)])
        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))
        if use_loc_head:
            s.loc_head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Linear(h//2,1))
    def forward(s, d):
        h=s.embed(d.x); pos=d.pos; t=d.x[:,s.n_pre:]
        for l,f in zip(s.ls,s.fs): h,pos=l(h,pos,d.edge_index,d.edge_attr); h=f(h,t)
        logits = s.head(h).squeeze(-1)
        pred_center = None
        if s.use_loc_head:
            batch = d.batch if hasattr(d, "batch") and d.batch is not None \
                    else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            loc_logit = s.loc_head(h).squeeze(-1)
            attn = pyg_softmax(loc_logit, batch)
            weighted = attn.unsqueeze(-1) * pos
            n_graphs = int(batch.max().item()) + 1
            pred_center = torch.zeros(n_graphs, 3, device=h.device).index_add_(0, batch, weighted)
        return logits, pred_center
''')

md(r'''## Cell 6 — Training machinery (identical to Phase 54/33/24/14)''')
code(r'''
def run_epoch(model, loader, opt=None, lam=0.0):
    tr = opt is not None; model.train(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    huber = nn.SmoothL1Loss(beta=4.0)
    ps, ys, cd = [], [], []
    for b in loader:
        b = b.to(DEVICE)
        with torch.set_grad_enabled(tr):
            logits, pred_c = model(b)
            loss = crit(logits, b.y)
            if lam > 0 and pred_c is not None:
                true_c = b.true_center.view(-1, 3)
                loss = loss + lam * huber(pred_c, true_c)
                cd.append((pred_c - true_c).norm(dim=1).detach().cpu().numpy())
        if tr:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        ps.append(torch.sigmoid(logits).detach().cpu().numpy()); ys.append(b.y.cpu().numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)
    out = {"pr": average_precision_score(y, p), "roc": roc_auc_score(y, p)}
    if cd:
        d_ = np.concatenate(cd)
        out["dcc"] = float((d_ < 4.0).mean()); out["mean_center_err"] = float(d_.mean())
    return out


def fit(use_loc_head, lam, seed, epochs=40, patience=8, batch=16):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl = DataLoader(splits["train"], batch_size=batch, shuffle=True)
    vl = DataLoader(splits["val"], batch_size=batch)
    best, state, bad = -1, None, 0
    for ep in range(epochs):
        run_epoch(model, tl, opt, lam)
        v = run_epoch(model, vl, lam=lam)
        sch.step()
        if v["pr"] > best:
            best, state, bad = v["pr"], {k:t.clone() for k,t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval()
    return model, best
''')

md(r'''## Cell 7 — Train 5 MORE seeds (3-7), reusing seeds 0-2 from Phase 54''')
code(r'''
models = {}
for seed in (0,1,2):
    cands = []
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): cands.extend(r.rglob(f"holo_clean_deploy_scaled3_seed{seed}.pt"))
    assert cands, f"holo_clean_deploy_scaled3_seed{seed}.pt not found -- attach Phase 54's kernel output"
    state = torch.load(max(cands, key=lambda p: p.stat().st_size), map_location=DEVICE)
    m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE); m.load_state_dict(state); m.eval()
    models[seed] = m
print(f"loaded {len(models)} checkpoints from Phase 54 (no retraining)")

BEST_LAMBDA = 0.1
t0 = time.time()
for seed in (3,4,5,6,7):
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    models[seed] = m
    torch.save(m.state_dict(), WORK/f"cache/holo_clean_deploy_scaled3_seed{seed}.pt")
    val_dcc = v.get("dcc", float("nan"))
    print(f"  seed {seed}: val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")
print(f"\n{len(models)} total seeds available (0-7)")
''')

md(r'''## Cell 8 — Load Phase 54's cached benchmark features (no re-fetch)''')
code(r'''
with open(locate("phase54_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)
print(f"loaded cached (round-3 pool) HOLO4K features for {len(raw_cache):,} chains")
mu, sd = blob6b["topo_mean"], blob6b["topo_std"]
''')

md(r'''## Cell 9 — Per-seed and per-ensemble-size DCC on the cached benchmark (confidence-based multi-chain arbitration)''')
code(r'''
@torch.no_grad()
def predict_conf_and_centers_all_seeds(entry):
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


by_structure = defaultdict(list)
for (pid, cid), entry in raw_cache.items():
    by_structure[pid].append((cid, entry))
print(f"grouped into {len(by_structure):,} structures")

t0 = time.time()
per_chain_conf_centers = {}
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


print(f"\n{'per-seed DCC (individual models, own-confidence chain arbitration)':<70}")
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

md(r'''## Cell 10 — Ensemble size sweep: does averaging more seeds close/hold the near-parity result?''')
code(r'''
P2RANK_THIS_RUN = 0.657974   # matches Phase24/27b/33/34/39's same-run, same-benchmark number

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
p39_dcc, p39_n = ensemble_dcc((0,1,2))
print(f"Phase 54's 3-seed (0,1,2 only) for direct comparison:")
print(f"  seeds (0,1,2): DCC {p39_dcc:.4f} (n={p39_n})  [Phase 54 reported 0.6533]")
''')

md(r'''## Cell 11 — Verdict''')
code(r'''
gap_3seed = p39_dcc - P2RANK_THIS_RUN
gap_8seed = full8_dcc - P2RANK_THIS_RUN
noise_scale = seed_dccs.std()

print(f"{'='*66}")
print(f"DOES THE ROUND-2 HOLO4K NEAR-PARITY HOLD WITH MORE SEEDS?")
print(f"{'='*66}")
print(f"single-seed DCC std across 8 independent runs: {noise_scale:.4f}")
print(f"3-seed ensemble gap to P2Rank:  {gap_3seed:+.4f}")
print(f"8-seed ensemble gap to P2Rank:  {gap_8seed:+.4f}")
print(f"gap magnitude vs single-seed noise: {abs(gap_8seed)/max(noise_scale,1e-9):.2f}x")

if gap_8seed > 0.01:
    verdict = "gap CLOSED and REVERSED with more seeds -- matches COACH420's pattern exactly"
elif gap_8seed > -0.015:
    verdict = "gap CLOSED to within noise with more seeds -- essentially confirmed parity"
elif abs(gap_8seed) > 1.5*noise_scale:
    verdict = "gap PERSISTS beyond noise -- surprising given the 3-seed near-parity result"
else:
    verdict = "gap is WITHIN plausible seed noise -- not confidently resolved either way at this n"

print(f"\nVERDICT: {verdict}")

print(f"\n--- HOLO4K 8-seed DCC across training-pool scale ---")
print(f"{'pool':<28}{'clusters':>10}{'DCC (8-seed)':>14}{'gap to P2Rank':>16}")
print("-"*68)
print(f"{'original (Phase27b)':<28}{2056:>10}{0.5912:>14.4f}{-0.0668:>16.4f}")
print(f"{'1.74x scaled (Phase34)':<28}{3577:>10}{0.6313:>14.4f}{-0.0267:>16.4f}")
print(f"{'2.6x scaled (this notebook)':<28}{5339:>10}{full8_dcc:>14.4f}{gap_8seed:>16.4f}")

json.dump({"seed_dccs": seed_dccs.tolist(), "ensemble_sweep": results,
           "phase39_3seed_dcc": p39_dcc, "full8_dcc": full8_dcc,
           "p2rank_this_run": P2RANK_THIS_RUN, "verdict": verdict},
          open(WORK/"phase57_summary.json", "w"), indent=2)
print("\nsaved -> phase57_summary.json")
''')

md(r'''
---

## Reading this

Compare directly to Phase 34's HOLO4K outcome at 1.74x scale (gap only
narrowed marginally, +0.003, stayed ambiguous) and Phase 32's COACH420
outcome (gap widened cleanly). If this round shows the gap fully closing or
flipping, that completes the "scale was the answer for both benchmarks"
story. If it stays ambiguous/negative, HOLO4K may need even more scale or a
genuinely different fix.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase57_scaled3_holo4k_seed_variance.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
