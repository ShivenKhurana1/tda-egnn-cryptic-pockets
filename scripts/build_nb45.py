#!/usr/bin/env python3
"""Phase 45 -- CryptoBench re-featurization with a bigger ESM2 (650M vs the
project-standard 35M). Phase 43/44 found AUC 0.872 (8-seed) essentially
matching CryptoBench's own published SOTA (0.87), but trailed their SOTA on
AUPRC (0.329 vs 0.40) and MCC (0.367 vs 0.42) -- their baseline uses
ESM2-3B, ~85x more params than this project's usual ESM2-35M. This tests
whether a bigger embedding model closes that specific gap. Otherwise
identical to Phase 42 (same topology/dihedral/radius-graph logic, same
CryptoBench data source) -- only the ESM2 model and output filename differ.
"""
import json, pickle
from pathlib import Path

C6a = pickle.load(open("/tmp/p6a_cells.pkl", "rb"))
C6b = pickle.load(open("/tmp/p6b_cells.pkl", "rb"))
C10 = pickle.load(open("/tmp/p10_cells.pkl", "rb"))
s6a = lambda i: (C6a[i]["source"] if isinstance(C6a[i]["source"], str) else "".join(C6a[i]["source"]))
s6b = lambda i: (C6b[i]["source"] if isinstance(C6b[i]["source"], str) else "".join(C6b[i]["source"]))
s10 = lambda i: (C10[i]["source"] if isinstance(C10[i]["source"], str) else "".join(C10[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 45 - CryptoBench Re-Featurization with ESM2-650M

Phase 43/44 (ESM2-35M) found AUC 0.872 (8-seed), matching CryptoBench's own
published SOTA (0.87), but trailed on AUPRC (0.329 vs 0.40) and MCC (0.367
vs 0.42) -- their baseline uses ESM2-3B. This notebook is identical to
Phase 42 except the ESM2 model is swapped 35M -> 650M (18.6x bigger, hidden
dim 480 -> 1280) to test whether embedding capacity closes that gap. Loaded
in fp16 (inference-only, no gradients needed) and with a smaller batch size
(16 -> 4) to fit safely on a T4's 16GB VRAM given attention cost at up to
1502 tokens/sequence.

**Data:** CryptoBench (Skrhak et al. 2025) -- 1,107 apo structures, each
paired with the holo structure whose ligand-induced pocket has the highest
structural deviation (`is_main_holo_structure`). Residue-level pocket labels
(`apo_pocket_selection`, auth-numbered `Chain_ResNum` strings) are provided
directly by the authors (computed via TM-align + pocket-RMSD >=2.0A) --
no apo/holo superposition work needed on our side. Official train
(4 CV folds)/test split from `folds.json`, clustered at 10% sequence
identity -- replaces the MMseqs2 step used elsewhere in this project;
CryptoBench's own split already IS leakage-clean.

**Reused verbatim from this project's proven pipeline:** the topology
featurizer (byte-for-byte, Phase 6b), `radius_graph_capped` (Phase 6a/1),
ESM2 embedding + dihedral recovery logic (Phase 10) -- only the ESM2 model
itself is swapped (35M -> 650M). Node schema grows to 1341-dim (21 onehot +
6 dihedral + 1280 ESM + 34 topo, vs Phase 42's 541), but the SAME
`LocalizingEGNN` architecture trains on this data unchanged since N_PRE and
TOPO_DIM are read from the blob dynamically -- only the label source and
task framing differ from the main pipeline (crypticity, not simple
ligand-proximity on a bound structure).
''')

md(r'''## Cell 1 - Install dependencies''')
code(r'''
!pip install -q torch-geometric==2.6.1 gudhi==3.10.1 biopython==1.84 transformers==4.46.3
''')

md(r'''## Cell 2 - Setup, fetch CryptoBench's dataset.json + folds.json from OSF''')
code(r'''
import gzip, json, os, time, urllib.request, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gudhi
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from Bio.PDB import MMCIFParser, PDBParser, PPBuilder

warnings.filterwarnings("ignore")

WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)
RAW = Path("/tmp/pdb_raw"); RAW.mkdir(parents=True, exist_ok=True)

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")

def fetch_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode())

cb_dataset = fetch_json("https://osf.io/download/ta2ju/")
cb_folds = fetch_json("https://osf.io/download/5s93p/")
print(f"CryptoBench: {len(cb_dataset):,} apo entries | folds: "
      f"{ {k: len(v) for k,v in cb_folds.items()} }")

# --- pick the MAIN holo pairing (unique per apo entry, verified: exactly
# one is_main_holo_structure==True per apo_id, 0 zero-main / 0 multi-main) --
units = []
for apo_id, entries in cb_dataset.items():
    main = next((e for e in entries if e.get("is_main_holo_structure")), None)
    if main is None: continue
    # keep as string, not int: residue numbers can carry a PDB insertion
    # code suffix (e.g. "60A"), which int() rejects
    pocket_resnums = {sel.split("_", 1)[1] for sel in main["apo_pocket_selection"]}
    units.append({"apo_id": apo_id, "apo_chain": main["apo_chain"],
                   "pocket_resnums": pocket_resnums, "pRMSD": main.get("pRMSD")})
print(f"training units (one main pocket per apo entry): {len(units):,}")

split_of = {}
for pid in cb_folds.get("test", []): split_of[pid] = "test"
for fold_key in ("train-0","train-1","train-2","train-3"):
    for pid in cb_folds.get(fold_key, []): split_of[pid] = "train"
for u in units:
    u["split"] = split_of.get(u["apo_id"], "unassigned")
n_unassigned = sum(1 for u in units if u["split"] == "unassigned")
print(f"split coverage: {sum(1 for u in units if u['split']!='unassigned')}/{len(units)} "
      f"({n_unassigned} unassigned, dropped)")
units = [u for u in units if u["split"] != "unassigned"]
''')

md(r'''## Cell 3 - Fetch apo structures (mmCIF preferred, per CryptoBench's own README)''')
code(r'''
need = sorted({u["apo_id"] for u in units})
def fetch(pid):
    for ext in ("cif","pdb"):
        p = RAW / f"{pid}.{ext}.gz"
        if p.exists() and p.stat().st_size > 500: return pid, p
    for ext in ("cif","pdb"):
        try:
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.{ext}.gz", timeout=60) as r:
                b = r.read()
            if len(b) > 500: p = RAW/f"{pid}.{ext}.gz"; p.write_bytes(b); return pid, p
        except Exception: continue
    return pid, None

t0=time.time()
with ThreadPoolExecutor(max_workers=16) as ex: got = list(ex.map(fetch, need))
paths = {p: f for p, f in got if f}
print(f"fetched {len(paths):,}/{len(need):,} in {(time.time()-t0)/60:.1f} min")
''')

md(r'''## Cell 4 - Parse apo chains, build per-residue crypticity labels''')
code(r'''
AA3_TO_1={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
MIN_RES, MAX_RES = 30, 1500

def parse_unit(u):
    pid, cid = u["apo_id"], u["apo_chain"]
    if pid not in paths: return None
    path = paths[pid]
    try:
        parser = MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True, PERMISSIVE=True)
        with gzip.open(path, "rt") as fh:
            model = next(parser.get_structure(pid, fh).get_models())
        ch = model[cid]
    except Exception:
        return None
    residues = [r for r in ch if r.get_resname().strip().upper() in AA3_TO_1 and "CA" in r]
    if not (MIN_RES <= len(residues) <= MAX_RES): return None
    seq, ca, per_res, resnums = [], [], [], []
    for r in residues:
        seq.append(AA3_TO_1.get(r.get_resname().strip().upper(), "X"))
        ca.append(r["CA"].coord)
        atoms = [a.coord for a in r.get_unpacked_list() if a.element not in ("H","D")]
        per_res.append(np.asarray(atoms, dtype=np.float32).reshape(-1,3))
        resnums.append(str(r.id[1]) + r.id[2].strip())   # e.g. "60" or "60A" (insertion code)
    y = np.array([1.0 if rn in u["pocket_resnums"] else 0.0 for rn in resnums], dtype=np.float32)
    n_found = int(y.sum())
    if n_found < 3: return None   # too few pocket residues actually present in this chain -- skip
    return dict(pdb_id=pid, chain_id=cid, seq="".join(seq),
                ca=np.asarray(ca, dtype=np.float32), per_res=per_res, y=y,
                split=u["split"], pRMSD=u["pRMSD"], n_pocket_labeled=len(u["pocket_resnums"]),
                n_pocket_found=n_found)

from joblib import Parallel, delayed
t0 = time.time()
parsed = Parallel(n_jobs=max(1, os.cpu_count() or 2), verbose=1)(
    delayed(parse_unit)(u) for u in units)
records = [r for r in parsed if r is not None]
print(f"\nparsed in {(time.time()-t0)/60:.1f} min")
print(f"usable chains: {len(records):,}/{len(units):,}")
n_res = sum(len(r["y"]) for r in records)
n_pos = sum(int(r["y"].sum()) for r in records)
print(f"residues {n_res:,} | cryptic-pocket residues {n_pos:,} ({100*n_pos/max(n_res,1):.2f}%)")
for s in ("train","test"):
    rs = [r for r in records if r["split"] == s]
    print(f"  {s:<6} {len(rs):>5} chains")
''')

md(r'''## Cell 5 - Featuriser (byte-for-byte the Phase 2b/6b/29/36 definition)''')
code(s6b(7))

md(r'''## Cell 6 - Compute topology for every chain''')
code(r'''
n_res_total = sum(len(r["ca"]) for r in records)
print(f"computing persistent homology on {n_res_total:,} residues "
      f"({os.cpu_count()} workers, ~6.4 ms/residue -> ETA {n_res_total*0.0064/60:.0f} min)")
t0 = time.time()
topo_all = Parallel(n_jobs=max(1, os.cpu_count() or 2), verbose=5)(
    delayed(topo_for_chain)(r["ca"], r["per_res"]) for r in records)
el = time.time() - t0
print(f"\ndone in {el/60:.1f} min ({1000*el/max(n_res_total,1):.2f} ms/residue)")

topo_cat = np.concatenate(topo_all, axis=0)
keep = topo_cat.std(axis=0) > 1e-9
dropped = [TOPO_NAMES[i] for i in range(TOPO_DIM) if not keep[i]]
names = [TOPO_NAMES[i] for i in range(TOPO_DIM) if keep[i]]
print(f"pruned {len(dropped)} constant column(s): {dropped}")
mu, sd = topo_cat[:, keep].mean(0), topo_cat[:, keep].std(0); sd[sd < 1e-6] = 1.0

g = torch.Generator().manual_seed(0)
Q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
Q = (Q * torch.sign(torch.det(Q))).numpy().astype(np.float32)
shift = np.array([31.7, -12.4, 88.1], dtype=np.float32)
ca0, pr0 = records[0]["ca"][:40], records[0]["per_res"][:40]
f_ref = topo_for_chain(ca0, pr0)
f_rot = topo_for_chain(ca0 @ Q.T + shift, [a @ Q.T + shift for a in pr0])
rel = float((np.abs(f_ref - f_rot) / np.maximum(np.abs(f_ref).max(0), 1e-6)).max())
print(f"rotation invariance: {rel:.2e}")
assert rel < 1e-3
''')

md(r'''## Cell 7 - Recompute backbone dihedrals (identical to Phase 10/30/37)''')
code(r'''
AA3 = set(AA3_TO_1)

def dihedrals_for(pid, path, cid, n_expect):
    try:
        parser = MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True, PERMISSIVE=True)
        with gzip.open(path,"rt") as fh:
            model = next(parser.get_structure(pid, fh).get_models())
        ch = model[cid]
    except Exception:
        return None
    residues=[r for r in ch if r.get_resname().strip().upper() in AA3 and "CA" in r]
    if len(residues) != n_expect:
        return None
    ppmap={}
    for pp in PPBuilder().build_peptides(ch, aa_only=False):
        for res,(phi,psi) in zip(pp, pp.get_phi_psi_list()):
            ppmap[res.id]=(phi,psi)
    phis,psis=[],[]
    for r in residues:
        phi,psi = ppmap.get(r.id, (None,None))
        phis.append(phi); psis.append(psi)
    def enc(angles):
        v=torch.tensor([a if a is not None else 0.0 for a in angles], dtype=torch.float32)
        m=torch.tensor([a is not None for a in angles], dtype=torch.float32)
        return torch.stack([torch.sin(v)*m, torch.cos(v)*m],dim=1), m.unsqueeze(1)
    pf,pm = enc(phis); sf,sm = enc(psis)
    return torch.cat([pf,sf,pm,sm],dim=1)

t0=time.time()
dihed = Parallel(n_jobs=max(1, os.cpu_count() or 2), verbose=1)(
    delayed(dihedrals_for)(r["pdb_id"], paths[r["pdb_id"]], r["chain_id"], len(r["ca"]))
    for r in records)
n_ok = sum(1 for x in dihed if x is not None)
print(f"\ndihedrals recovered for {n_ok:,}/{len(records):,} chains in {(time.time()-t0)/60:.1f} min")
''')

md(r'''## Cell 8 - ESM2 embeddings (identical to Phase 10/30/37, GPU-accelerated)''')
code(r'''
from transformers import AutoTokenizer, AutoModel
ESM_NAME = "facebook/esm2_t33_650M_UR50D"   # 650M params, hidden 1280 -- 18.6x
                                             # bigger than the 35M model used
                                             # everywhere else in this project.
                                             # CryptoBench's own SOTA baseline
                                             # uses ESM2-3B; 650M is a safer
                                             # first step given T4's 16GB VRAM
                                             # and attention cost at up to 1502
                                             # tokens/sequence.
tok = AutoTokenizer.from_pretrained(ESM_NAME)
esm = AutoModel.from_pretrained(ESM_NAME).half().to(DEVICE).eval()   # fp16: inference-only, no grads needed
ESM_DIM = esm.config.hidden_size
print(f"{ESM_NAME}: hidden {ESM_DIM}, device {DEVICE}, dtype {next(esm.parameters()).dtype}")

@torch.no_grad()
def embed_batch(seqs, max_len=1502):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = esm(**enc).last_hidden_state
    embs = []
    for i, s in enumerate(seqs):
        n = len(s)
        embs.append(out[i, 1:1+n, :].float().cpu())
    return embs

order = sorted(range(len(records)), key=lambda i: len(records[i]["seq"]))
BATCH = 4   # smaller than the 35M-model batch of 16: 650M params + longer
            # sequences means attention memory adds up fast on a T4
t0=time.time()
esm_feats = [None]*len(records)
for bi in range(0, len(order), BATCH):
    idx = order[bi:bi+BATCH]
    seqs = [records[i]["seq"] for i in idx]
    embs = embed_batch(seqs)
    for i, e in zip(idx, embs):
        esm_feats[i] = e
    if bi % (BATCH*20) == 0:
        print(f"  {bi}/{len(order)}  [{(time.time()-t0)/60:.1f}m]")

n_res = sum(len(r["ca"]) for r in records)
print(f"\nESM2 embeddings for {len(records):,} chains / {n_res:,} residues "
      f"in {(time.time()-t0)/60:.1f} min ({1000*(time.time()-t0)/max(n_res,1):.2f} ms/residue)")
del esm; torch.cuda.empty_cache()
''')

md(r'''## Cell 9 - Assemble PyG graphs (1341-dim schema: 21 onehot + 6 dihedral + 1280 ESM + 34 topo)''')
code(r'''
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ALPHABET)}

def radius_graph_capped(pos, r, max_neighbors=32):
    n = pos.size(0)
    pos = pos - pos.mean(dim=0, keepdim=True)
    dist = torch.cdist(pos, pos).clone()
    dist.fill_diagonal_(float("inf"))
    mask = dist <= r
    if max_neighbors is not None and max_neighbors < n:
        _, knn = torch.topk(dist, max_neighbors, dim=1, largest=False)
        keep_ = torch.zeros_like(mask); keep_.scatter_(1, knn, True); mask &= keep_
    mask = mask | mask.t()
    tgt, src = mask.nonzero(as_tuple=True)
    return torch.stack([src, tgt], dim=0).long()

N_PRE = 21 + 6 + ESM_DIM
dataset = []
n_dropped = 0
for i, r in enumerate(records):
    if dihed[i] is None or esm_feats[i] is None:
        n_dropped += 1; continue
    n = len(r["ca"])
    if dihed[i].shape[0] != n or esm_feats[i].shape[0] != n:
        n_dropped += 1; continue
    onehot = torch.zeros(n, len(AA_ALPHABET))
    onehot[torch.arange(n), torch.tensor([AA_TO_IDX.get(c, 20) for c in r["seq"]])] = 1.0
    topo_i = topo_all[i][:, keep]
    topo_z = torch.from_numpy(((topo_i - mu) / sd).astype(np.float32))
    x = torch.cat([onehot, dihed[i], esm_feats[i], topo_z], dim=1)

    pos = torch.from_numpy(r["ca"]); pos = pos - pos.mean(0, keepdim=True)
    ei = radius_graph_capped(pos, 8.0, 32)
    d_ = (pos[ei[0]]-pos[ei[1]]).norm(dim=1,keepdim=True)
    sep = (ei[0]-ei[1]).abs().float().unsqueeze(1)

    y_np = r["y"].astype(bool)
    true_center = torch.from_numpy(r["ca"][y_np].mean(axis=0) - r["ca"].mean(axis=0)) if y_np.any() else None
    if true_center is None:
        n_dropped += 1; continue

    d = Data(x=x, pos=pos, edge_index=ei, edge_attr=torch.cat([d_, torch.log1p(sep)], 1),
             y=torch.from_numpy(r["y"]))
    d.true_center = true_center.float()
    d.pdb_id, d.chain_id, d.split = r["pdb_id"], r["chain_id"], r["split"]
    d.num_nodes = n
    assert torch.isfinite(d.x).all()
    dataset.append(d)

print(f"built {len(dataset):,} graphs ({n_dropped} dropped: alignment/labeling failures)")
n_res_ = sum(d.num_nodes for d in dataset); n_pos_ = sum(int(d.y.sum()) for d in dataset)
POS_WEIGHT = (n_res_ - n_pos_) / max(n_pos_, 1)
print(f"residues {n_res_:,} | cryptic positives {n_pos_:,} ({100*n_pos_/max(n_res_,1):.2f}%) "
      f"| pos_weight {POS_WEIGHT:.2f}")
for s in ("train","test"):
    ds = [d for d in dataset if d.split == s]
    print(f"  {s:<6} {len(ds):>5} chains")
''')

md(r'''## Cell 10 - Carve a val subset from train (identical convention to every other phase), integrity checks, save''')
code(r'''
rng = np.random.RandomState(42)
train_idx = [i for i,d in enumerate(dataset) if d.split == "train"]
perm = rng.permutation(len(train_idx))
n_val = max(1, int(0.08 * len(perm)))
val_set = {train_idx[i] for i in perm[:n_val]}
for i, d in enumerate(dataset):
    if i in val_set: d.split = "val"

for s in ("train","val","test"):
    ds = [d for d in dataset if d.split == s]
    r_ = sum(d.num_nodes for d in ds); p_ = sum(int(d.y.sum()) for d in ds)
    print(f"{s:<6} chains {len(ds):>5}  residues {r_:>7,}  positives {p_:>5}  rate {100*p_/max(r_,1):>5.2f}%")

for d in dataset:
    assert torch.isfinite(d.x).all() and torch.isfinite(d.pos).all()
    assert d.edge_index.max() < d.num_nodes and d.edge_index.min() >= 0
print("integrity OK")

p = WORK / "phase45_cryptobench_graphs_esm650m.pt"
torch.save({"dataset": dataset, "n_pre_topo": N_PRE, "topo_dim": int(keep.sum()),
            "topo_names": names, "burial_names": BURIAL_NAMES, "dropped_names": dropped,
            "topo_mean": mu, "topo_std": sd, "esm_dim": ESM_DIM, "pos_weight": POS_WEIGHT,
            "aa_alphabet": AA_ALPHABET, "source": "CryptoBench (Skrhak et al. 2025, btae745)"}, p)
print(f"saved -> {p} ({p.stat().st_size/1e6:.0f} MB)")
''')

md(r'''
---

## What this produces

`phase45_cryptobench_graphs_esm650m.pt` -- same schema as
`phase42_cryptobench_graphs.pt` except node dim is 1341 (21+6+1280+34)
instead of 541 (the ESM block grows from 480 to 1280 dims), same
`true_center`/`y` label convention, so the exact same `LocalizingEGNN`
architecture and training machinery (Phase 14/20/24's `fit`/`run_epoch`)
apply unchanged -- N_PRE/TOPO_DIM are read from the blob dynamically.
Official CryptoBench train/test split preserved via `d.split`, with a val
subset carved from train using the same 8%/seed=42 convention used
everywhere else in this project.

**Next:** Phase 46 -- train + evaluate on this ESM2-650M blob exactly like
Phase 43 did for the 35M version, and compare AUPRC/MCC directly against
Phase 43/44's numbers to see whether embedding capacity was the gap.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase45_cryptobench_ingestion_esm650m.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
