#!/usr/bin/env python3
"""Phase 117 -- CryptoBench ingestion, with a NEW auxiliary "any-pocket"
label added. Mirrors Phase42 exactly (same 35M-ESM 541-dim schema, same
topology/dihedral/ESM logic, byte-for-byte reused) but additionally
fetches CryptoBench's own noncryptic-pockets.json (auxiliary-data,
OSF file 37ztm, 665 apo structures -- confirmed via direct inspection to
be a STRICT SUBSET of the main 1,107 apo structures, same
apo_pocket_selection schema as dataset.json) and computes a second
per-residue label y_aux = 1 if a residue is part of EITHER the main
cryptic pocket OR a noncryptic pocket (i.e. "is this residue part of ANY
real, ligand-binding-capable pocket at all", cryptic or not).

Rationale: the CryptoBench authors' OWN use of this file (in their
conservation-statistics notebook, src/F-statistics/conservation) gives
cryptic labels priority when a residue is both -- confirming "noncryptic
minus cryptic" is the intended, non-conflicting way to read this data.
Using y_aux as an auxiliary multi-task target (not the main label) mirrors
this project's own PROVEN successful pattern: Phase 14's auxiliary
localization loss was a genuine, seed-validated win for the main
pipeline. The hypothesis: an easier "is this pocket-like at all" signal
may help the model learn better intermediate features for the harder
"is it specifically CRYPTIC" classification, since ~665/1107 apo
structures have real, labeled non-cryptic pocket sites the model
currently has zero signal about (they're silently lumped in with true
negatives).

Not yet known whether this actually helps -- prototyping on the cheap
35M-ESM pipeline first (fast iteration), matching this project's
established discipline of testing a lever cheaply before committing to
the expensive 3B version."""
import json, pickle
from pathlib import Path

C6b = pickle.load(open("/tmp/p6b_cells.pkl", "rb"))
s6b = lambda i: (C6b[i]["source"] if isinstance(C6b[i]["source"], str) else "".join(C6b[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 117 - CryptoBench Ingestion + Auxiliary "Any-Pocket" Label

Adds a second per-residue label (`y_aux`) to Phase42's proven CryptoBench
blob: 1 if a residue is part of EITHER the main cryptic pocket OR one of
CryptoBench's own labeled non-cryptic pockets on the same apo structure
(665/1107 apo structures have at least one). Everything else (topology,
dihedral, ESM2-35M, node schema, split) is byte-for-byte identical to
Phase42 -- this is purely an additive label, no change to existing
features or the main `y` target.
''')

md(r'''## Cell 1 - Install dependencies''')
code(r'''
!pip install -q torch-geometric==2.6.1 gudhi==3.10.1 biopython==1.84 transformers==4.46.3
''')

md(r'''## Cell 2 - Setup, fetch CryptoBench's dataset.json + folds.json + noncryptic-pockets.json from OSF''')
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
cb_noncryptic = fetch_json("https://osf.io/download/37ztm/")
print(f"CryptoBench: {len(cb_dataset):,} apo entries | folds: "
      f"{ {k: len(v) for k,v in cb_folds.items()} } | "
      f"noncryptic-pocket apo entries: {len(cb_noncryptic):,}")

# --- pick the MAIN holo pairing (unique per apo entry, verified: exactly
# one is_main_holo_structure==True per apo_id, 0 zero-main / 0 multi-main) --
units = []
for apo_id, entries in cb_dataset.items():
    main = next((e for e in entries if e.get("is_main_holo_structure")), None)
    if main is None: continue
    # keep as string, not int: residue numbers can carry a PDB insertion
    # code suffix (e.g. "60A"), which int() rejects
    pocket_resnums = {sel.split("_", 1)[1] for sel in main["apo_pocket_selection"]}
    apo_chain = main["apo_chain"]

    # noncryptic-pockets.json is keyed by 4-char PDB ID only; each entry
    # carries its own apo_chain -- filter to entries matching THIS unit's
    # apo_chain, union all their apo_pocket_selection residues. Multiple
    # entries per PDB ID are common (up to ~12, different ligand/holo
    # pairings at the same or nearby sites) -- union captures all of them.
    noncryptic_resnums = set()
    for nc in cb_noncryptic.get(apo_id, []):
        if nc.get("apo_chain") != apo_chain: continue
        noncryptic_resnums |= {sel.split("_", 1)[1] for sel in nc["apo_pocket_selection"]}

    units.append({"apo_id": apo_id, "apo_chain": apo_chain,
                   "pocket_resnums": pocket_resnums,
                   "noncryptic_resnums": noncryptic_resnums, "pRMSD": main.get("pRMSD")})
n_with_noncryptic = sum(1 for u in units if u["noncryptic_resnums"])
print(f"training units (one main pocket per apo entry): {len(units):,} "
      f"({n_with_noncryptic:,} have >=1 noncryptic-pocket residue on the same chain)")

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

md(r'''## Cell 3 - Fetch apo structures (mmCIF preferred, per CryptoBench's own README; identical to Phase42)''')
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

md(r'''## Cell 4 - Parse apo chains, build per-residue crypticity label (y) AND the new any-pocket label (y_aux)''')
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
    # any-pocket aux label: cryptic OR noncryptic (union) -- a strictly
    # easier/broader task than the main cryptic-only label
    any_pocket_resnums = u["pocket_resnums"] | u["noncryptic_resnums"]
    y_aux = np.array([1.0 if rn in any_pocket_resnums else 0.0 for rn in resnums], dtype=np.float32)
    return dict(pdb_id=pid, chain_id=cid, seq="".join(seq),
                ca=np.asarray(ca, dtype=np.float32), per_res=per_res, y=y, y_aux=y_aux,
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
n_pos_aux = sum(int(r["y_aux"].sum()) for r in records)
print(f"residues {n_res:,} | cryptic-pocket residues {n_pos:,} ({100*n_pos/max(n_res,1):.2f}%) | "
      f"any-pocket (aux) residues {n_pos_aux:,} ({100*n_pos_aux/max(n_res,1):.2f}%)")
for s in ("train","test"):
    rs = [r for r in records if r["split"] == s]
    print(f"  {s:<6} {len(rs):>5} chains")
''')

md(r'''## Cell 5 - Featuriser (byte-for-byte the Phase 2b/6b/29/36 definition)''')
code(s6b(7))

md(r'''## Cell 6 - Compute topology for every chain (identical to Phase42)''')
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

md(r'''## Cell 7 - Recompute backbone dihedrals (identical to Phase 10/30/37/42)''')
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

md(r'''## Cell 8 - ESM2 embeddings (35M -- cheap, fast iteration; identical to Phase 10/30/37/42)''')
code(r'''
from transformers import AutoTokenizer, AutoModel
ESM_NAME = "facebook/esm2_t12_35M_UR50D"
tok = AutoTokenizer.from_pretrained(ESM_NAME)
esm = AutoModel.from_pretrained(ESM_NAME).to(DEVICE).eval()
ESM_DIM = esm.config.hidden_size
print(f"{ESM_NAME}: hidden {ESM_DIM}, device {DEVICE}")

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
BATCH = 16
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

md(r'''## Cell 9 - Assemble PyG graphs (identical 541-dim schema; each graph now ALSO carries d.y_aux)''')
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
             y=torch.from_numpy(r["y"]), y_aux=torch.from_numpy(r["y_aux"]))
    d.true_center = true_center.float()
    d.pdb_id, d.chain_id, d.split = r["pdb_id"], r["chain_id"], r["split"]
    d.num_nodes = n
    assert torch.isfinite(d.x).all()
    dataset.append(d)

print(f"built {len(dataset):,} graphs ({n_dropped} dropped: alignment/labeling failures)")
n_res_ = sum(d.num_nodes for d in dataset); n_pos_ = sum(int(d.y.sum()) for d in dataset)
n_pos_aux_ = sum(int(d.y_aux.sum()) for d in dataset)
POS_WEIGHT = (n_res_ - n_pos_) / max(n_pos_, 1)
POS_WEIGHT_AUX = (n_res_ - n_pos_aux_) / max(n_pos_aux_, 1)
print(f"residues {n_res_:,} | cryptic positives {n_pos_:,} ({100*n_pos_/max(n_res_,1):.2f}%) "
      f"| pos_weight {POS_WEIGHT:.2f}")
print(f"any-pocket (aux) positives {n_pos_aux_:,} ({100*n_pos_aux_/max(n_res_,1):.2f}%) "
      f"| pos_weight_aux {POS_WEIGHT_AUX:.2f}")
for s in ("train","test"):
    ds = [d for d in dataset if d.split == s]
    print(f"  {s:<6} {len(ds):>5} chains")
''')

md(r'''## Cell 10 - Carve a val subset from train (identical convention), integrity checks, save''')
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
    pa_ = sum(int(d.y_aux.sum()) for d in ds)
    print(f"{s:<6} chains {len(ds):>5}  residues {r_:>7,}  cryptic-pos {p_:>5} ({100*p_/max(r_,1):>5.2f}%)  "
          f"any-pocket-pos {pa_:>5} ({100*pa_/max(r_,1):>5.2f}%)")

for d in dataset:
    assert torch.isfinite(d.x).all() and torch.isfinite(d.pos).all()
    assert d.edge_index.max() < d.num_nodes and d.edge_index.min() >= 0
    assert (d.y_aux >= d.y).all(), "y_aux must be a superset of y (cryptic implies any-pocket)"
print("integrity OK (including y_aux >= y superset check)")

p = WORK / "phase117_cryptobench_graphs_aux.pt"
torch.save({"dataset": dataset, "n_pre_topo": N_PRE, "topo_dim": int(keep.sum()),
            "topo_names": names, "dropped_names": dropped,
            "topo_mean": mu, "topo_std": sd, "esm_dim": ESM_DIM,
            "pos_weight": POS_WEIGHT, "pos_weight_aux": POS_WEIGHT_AUX,
            "aa_alphabet": AA_ALPHABET,
            "source": "CryptoBench (Skrhak et al. 2025, btae745) + noncryptic-pockets aux label"}, p)
print(f"saved -> {p} ({p.stat().st_size/1e6:.0f} MB)")
''')

md(r'''
---

## What this produces

`phase117_cryptobench_graphs_aux.pt` -- same 541-dim node schema as
Phase42, PLUS `d.y_aux` on every graph (1 = residue is part of the main
cryptic pocket OR a labeled non-cryptic pocket on the same apo chain).
`y_aux >= y` always holds (verified) -- it's a strict superset label.

**Next:** Phase 118 -- train with a THIRD head (`aux_head`, structurally
identical to the main classification head) predicting `y_aux`, added to
the loss with its own weight `lam_aux`. Sweep `lam_aux` (including 0.0 as
the exact Phase42/43 baseline control) on a single seed first, then
validate any apparent winner with more seeds before trusting it -- this
project's standing discipline, learned expensively this session.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase117_cryptobench_ingestion_aux.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
