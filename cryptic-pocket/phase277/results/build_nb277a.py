#!/usr/bin/env python3
"""Phase 277a: featurize PocketMiner's own held-out test set (Meller et al. 2023, Nat Commun; data at github.com/Mickdub/gvp, pocket_pred branch,
data/pm-dataset/) into our exact node format (21 one-hot + 6 dihedral + 2560 ESM2-3B + 34 topology = 2621), splicing Phase 48's featurizer cells
verbatim (same GUDHI persistent-homology definition, same dihedral code, same ESM2-3B embedding code) so a structure built here is indistinguishable,
feature-for-feature, from one in the training blob. GPU kernel (ESM2-3B).

Labels: PocketMiner's own residue-level ground truth (0 = negative, 1 = cryptic-pocket residue, 2 = masked/not evaluated -- exactly their own
`test_label_dictionary.npy` encoding, confirmed against their evaluation script src/test_performance_on_xtal_residues.py). We use THEIR pre-cleaned
apo structures (data/pm-dataset/apo-structures, loops remodeled in) because the label arrays are indexed against those specific files, not fresh
RCSB structures. Uses the ORIGINAL Phase 48 blob's topology mu/sd/keep-mask (loaded here) -- never refit on this tiny external set."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
nb48 = json.load(open(HERE.parents[1] / "phase48_cryptobench_ingestion_esm3b.ipynb"))
s48 = lambda i: "".join(nb48["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md('''
# Phase 277a — Featurize PocketMiner's held-out test set (external validation, never used anywhere in this project)

35 proteins, PocketMiner's own residue-level cryptic-pocket labels (0/1/2 = negative/positive/masked), same node format as every model
in this project (21 one-hot + 6 dihedral + 2560 ESM2-3B + 34 topology). Structures and labels: Meller, Ward, Kokh et al. 2023, Nat Commun,
github.com/Mickdub/gvp (pocket_pred branch).
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1 gudhi==3.10.1 biopython==1.84 transformers==4.46.3")
md("## Cell 2 — Fetch PocketMiner's test ids, labels, and the file listing; download every matching apo structure")
code(r'''
import gzip, json, os, re, time, urllib.request, warnings
from pathlib import Path
import numpy as np
import torch
import gudhi
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")

WORK = Path("/kaggle/working"); RAW = Path("/tmp/pm_raw"); RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://raw.githubusercontent.com/Mickdub/gvp/pocket_pred/data/pm-dataset"

GPU_OK = False
if torch.cuda.is_available():
    try:
        _p = torch.randn(256, 256, device="cuda"); (_p @ _p).sum().item(); torch.cuda.synchronize(); GPU_OK = True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")

def fetch_bytes(url, tries=4):
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r: return r.read()
        except Exception:
            time.sleep(2 * (a + 1))
    raise RuntimeError(f"failed: {url}")

test_ids = np.load(WORK / "test_apo_ids_with_chainids.npy") if (WORK / "test_apo_ids_with_chainids.npy").exists() else None
if test_ids is None:
    (WORK / "test_apo_ids_with_chainids.npy").write_bytes(fetch_bytes(f"{BASE}/test_apo_ids_with_chainids.npy"))
    (WORK / "test_label_dictionary.npy").write_bytes(fetch_bytes(f"{BASE}/test_label_dictionary.npy"))
test_ids = np.load(WORK / "test_apo_ids_with_chainids.npy")
labels = np.load(WORK / "test_label_dictionary.npy", allow_pickle=True).item()
print(f"test proteins: {len(test_ids)}  |  label dict entries: {len(labels)}")

import urllib.request as ur
api = json.loads(ur.urlopen("https://api.github.com/repos/Mickdub/gvp/contents/data/pm-dataset/apo-structures?ref=pocket_pred", timeout=30).read())
files = [f["name"] for f in api]
print(f"apo-structures files available: {len(files)}")

def match_file(idc):                                # idc like "6hb0A" (4-char pdb id + 1-char chain)
    pid, ch = idc[:4], idc[4:]
    cands = [f for f in files if f.lower().startswith(pid.lower())]
    exact = [f for f in cands if f[:len(pid) + len(ch)].lower() == (pid + ch).lower()]
    return exact[0] if exact else (cands[0] if len(cands) == 1 else None)

paths = {}
for idc in test_ids:
    idc = str(idc)
    fn = match_file(idc)
    assert fn is not None, f"no unique apo-structure file for {idc} (candidates: {[f for f in files if f.lower().startswith(idc[:4].lower())]})"
    dst = RAW / fn
    if not dst.exists() or dst.stat().st_size < 200: dst.write_bytes(fetch_bytes(f"{BASE}/apo-structures/{fn}"))
    paths[idc] = dst
print(f"resolved and fetched {len(paths)}/{len(test_ids)} structures")
''')
md("## Cell 3 — Parse residues (same AA3/CA filter as Phase 48), align to PocketMiner's label array length exactly")
code(r'''
AA3_TO_1={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
from Bio.PDB import PDBParser

def parse_one(idc, path):
    pid, ch_hint = idc[:4], idc[4:]
    model = next(PDBParser(QUIET=True, PERMISSIVE=True).get_structure(pid, str(path)).get_models())
    chains = list(model)
    chain = None
    if ch_hint:
        chain = next((c for c in chains if c.id == ch_hint), None)
    if chain is None:
        chain = max(chains, key=lambda c: sum(1 for r in c if r.get_resname().strip().upper() in AA3_TO_1 and "CA" in r))
    residues = [r for r in chain if r.get_resname().strip().upper() in AA3_TO_1 and "CA" in r]
    seq = "".join(AA3_TO_1.get(r.get_resname().strip().upper(), "X") for r in residues)
    ca = np.asarray([r["CA"].coord for r in residues], np.float32)
    per_res = [np.asarray([a.coord for a in r.get_unpacked_list() if a.element not in ("H", "D")], np.float32).reshape(-1, 3) for r in residues]
    return dict(pdb_id=pid, chain_id=chain.id, seq=seq, ca=ca, per_res=per_res)

records, fails = [], []
for idc, path in paths.items():
    r = parse_one(idc, path)
    lab = labels.get(idc[:4], labels.get(idc))     # one entry ("5x8ua") keys its label by the FULL id, not the 4-char pdb code
    assert lab is not None, f"no label entry for {idc} (tried '{idc[:4]}' and '{idc}')"
    if len(r["seq"]) != len(lab):
        fails.append((idc, len(r["seq"]), len(lab))); continue
    r["label_raw"] = lab.astype(np.int64); r["key"] = idc
    records.append(r)
print(f"aligned {len(records)}/{len(paths)} chains to their exact label-array length; length mismatches: {fails}")
assert len(fails) <= max(1, int(0.05 * len(paths))), f"too many residue-count mismatches to proceed silently: {fails}"
for r in records:
    y = (r["label_raw"] == 1).astype(np.float32); mask = (r["label_raw"] != 2)
    print(f"  {r['key']:6s} n={len(r['seq']):4d}  masked_out={int((~mask).sum()):3d}  positives={int(y[mask].sum()):3d} / evaluated={int(mask.sum()):3d}")
''')
md("## Cell 4 — Topology features (Phase 48's featurizer, verbatim), then apply the ORIGINAL blob's mu/sd/keep-mask")
code(s48(10))
code(r'''
def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

t0 = time.time()
topo_all = [topo_for_chain(r["ca"], r["per_res"]) for r in records]
print(f"topology computed for {len(records)} chains in {(time.time()-t0)/60:.1f} min")

blob48 = torch.load(locate("phase48_cryptobench_graphs_esm3b.pt"), map_location="cpu", weights_only=False)
tr_names, tr_mu, tr_sd = blob48["topo_names"], blob48["topo_mean"], blob48["topo_std"]
name_to_col = {n: i for i, n in enumerate(TOPO_NAMES)}
keep_idx = [name_to_col[n] for n in tr_names]                       # exact training-time column selection, in training order
assert len(keep_idx) == len(tr_names) == tr_mu.shape[0]
N_PRE = int(blob48["n_pre_topo"]); ESM_DIM = N_PRE - 21 - 6
GEOM_PAD_DIM = 17   # the model's "full" arm was trained with a 34+17=51-wide side channel; geometry always zeroed for this arm
print(f"reusing training-time topology stats: {len(keep_idx)} columns, N_PRE={N_PRE}, ESM_DIM={ESM_DIM}")
for r, t in zip(records, topo_all):
    z = (t[:, keep_idx] - tr_mu) / tr_sd
    r["topo_z"] = torch.from_numpy(z.astype(np.float32))
''')
md("## Cell 5 — Backbone dihedrals (Phase 48's code, verbatim)")
code(r'''
AA3 = set(AA3_TO_1)
from Bio.PDB import PPBuilder

def dihedrals_for_path(path, chain_id, n_expect):
    model = next(PDBParser(QUIET=True, PERMISSIVE=True).get_structure("x", str(path)).get_models())
    ch = model[chain_id]
    residues=[r for r in ch if r.get_resname().strip().upper() in AA3 and "CA" in r]
    if len(residues) != n_expect: return None
    ppmap={}
    for pp in PPBuilder().build_peptides(ch, aa_only=False):
        for res,(phi,psi) in zip(pp, pp.get_phi_psi_list()): ppmap[res.id]=(phi,psi)
    phis,psis=[],[]
    for r in residues:
        phi,psi = ppmap.get(r.id, (None,None)); phis.append(phi); psis.append(psi)
    def enc(angles):
        v=torch.tensor([a if a is not None else 0.0 for a in angles], dtype=torch.float32)
        m=torch.tensor([a is not None for a in angles], dtype=torch.float32)
        return torch.stack([torch.sin(v)*m, torch.cos(v)*m],dim=1), m.unsqueeze(1)
    pf,pm = enc(phis); sf,sm = enc(psis)
    return torch.cat([pf,sf,pm,sm],dim=1)

for r in records:
    r["dihed"] = dihedrals_for_path(paths[r["key"]], r["chain_id"], len(r["seq"]))
n_ok = sum(1 for r in records if r["dihed"] is not None)
print(f"dihedrals recovered for {n_ok}/{len(records)} chains")
''')
md("## Cell 6 — ESM2-3B embeddings (Phase 48's code, verbatim)")
code(s48(16).replace('order = sorted(range(len(records)), key=lambda i: len(records[i]["seq"]))',
                      'order = sorted(range(len(records)), key=lambda i: len(records[i]["seq"]))', 1))
md("## Cell 7 — Assemble graphs in the exact training schema; save")
code(r'''
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ALPHABET)}

def radius_graph_capped(pos, r, max_neighbors=32):
    n = pos.size(0); pos = pos - pos.mean(dim=0, keepdim=True)
    dist = torch.cdist(pos, pos).clone(); dist.fill_diagonal_(float("inf"))
    mask = dist <= r
    if max_neighbors is not None and max_neighbors < n:
        _, knn = torch.topk(dist, max_neighbors, dim=1, largest=False)
        keep_ = torch.zeros_like(mask); keep_.scatter_(1, knn, True); mask &= keep_
    mask = mask | mask.t(); tgt, src = mask.nonzero(as_tuple=True)
    return torch.stack([src, tgt], dim=0).long()

from torch_geometric.data import Data
dataset, n_dropped = [], 0
for i, r in enumerate(records):
    if r["dihed"] is None or esm_feats[i] is None: n_dropped += 1; continue
    n = len(r["seq"])
    onehot = torch.zeros(n, len(AA_ALPHABET))
    onehot[torch.arange(n), torch.tensor([AA_TO_IDX.get(c, 20) for c in r["seq"]])] = 1.0
    geom_zeros = torch.zeros(n, GEOM_PAD_DIM)   # the "full" arm always zeroes these 17 geometry columns in training; padding with
                                            # true zeros is exactly equivalent, not an approximation (0 * anything = 0)
    x = torch.cat([onehot, r["dihed"], esm_feats[i], r["topo_z"], geom_zeros], dim=1)
    assert x.shape[1] == N_PRE + len(keep_idx) + GEOM_PAD_DIM, (x.shape, N_PRE, len(keep_idx), GEOM_PAD_DIM)
    pos = torch.from_numpy(r["ca"]); pos = pos - pos.mean(0, keepdim=True)
    ei = radius_graph_capped(pos, 8.0, 32)
    d_ = (pos[ei[0]] - pos[ei[1]]).norm(dim=1, keepdim=True); sep = (ei[0] - ei[1]).abs().float().unsqueeze(1)
    y = torch.from_numpy((r["label_raw"] == 1).astype(np.float32))
    mask = torch.from_numpy((r["label_raw"] != 2))
    ypos = r["ca"][(r["label_raw"] == 1)]
    true_center = torch.from_numpy(ypos.mean(0) - r["ca"].mean(0)).float() if len(ypos) else torch.zeros(3)
    d = Data(x=x, pos=pos, edge_index=ei, edge_attr=torch.cat([d_, torch.log1p(sep)], 1), y=y)
    d.eval_mask = mask; d.true_center = true_center; d.pdb_id, d.chain_id = r["pdb_id"], r["key"][4:] or "A"; d.num_nodes = n
    assert torch.isfinite(d.x).all()
    dataset.append(d)
print(f"built {len(dataset)} graphs ({n_dropped} dropped)")
n_res = sum(int(d.eval_mask.sum()) for d in dataset); n_pos = sum(int(d.y[d.eval_mask].sum()) for d in dataset)
print(f"evaluated residues {n_res} | positives {n_pos} ({100*n_pos/max(n_res,1):.2f}%)")
torch.save({"dataset": dataset, "source": "PocketMiner test set (Meller et al. 2023, Nat Commun; github.com/Mickdub/gvp pocket_pred)"},
           WORK / "phase277_pocketminer_graphs.pt")
print("saved -> phase277_pocketminer_graphs.pt")
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "277a"; d.mkdir(exist_ok=True)
(d / "phase277a_pocketminer_featurize.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "blackpearls1/phase277a-pocketminer-featurize", "title": "Phase277a Pocketminer Featurize",
    "code_file": "phase277a_pocketminer_featurize.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "machine_shape": "NvidiaTeslaT4", "enable_tpu": False, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": ["blackpearls1/tda-egnn-phase48-cryptobench-esm3b"]}, indent=2) + "\n")
print("wrote", d)
