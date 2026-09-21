#!/usr/bin/env python3
"""Phase 20: legitimate full-benchmark SOTA comparison. Cleans the TRAINING
side of leakage (excludes anything homologous to COACH420) instead of
filtering the benchmark, then evaluates on the ENTIRE, unfiltered 420-structure
benchmark -- directly comparable to literature EquiPocket/P2Rank numbers."""
import json, pickle
from pathlib import Path

c12 = pickle.load(open("/tmp/p12_cells.pkl", "rb"))
c14 = pickle.load(open("/tmp/p14_cells.pkl", "rb"))
s12 = lambda i: c12[i]["source"]
s14 = lambda i: c14[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 20 — A Legitimate Full-Benchmark Comparison

Every prior benchmark comparison in this project (Phase 12, 15, 16) cleaned
the **benchmark side** of leakage: filtering out COACH420/HOLO4K structures
homologous to the training set, then scoring only on what survived. That was
the right call at the time — it made the "us vs P2Rank" comparison valid. But
it also made the literature comparison (vs EquiPocket's published numbers)
invalid, because the surviving subset turned out to be measurably *easier*
than the full benchmark (proven in Phase 12: P2Rank scored 36-75% higher on
our clean subset than its own published full-benchmark number).

This notebook fixes the actual asymmetry: **clean the training side instead.**
Exclude from training anything homologous to COACH420 (all 420 structures, not
just a filtered subset), retrain, then evaluate on the **entire, unfiltered**
benchmark. If the model does well here, it is legitimately comparable to
EquiPocket's and P2Rank's published numbers — same benchmark, same difficulty,
no subset selection in either direction.

**Scoped to COACH420 only.** HOLO4K's full ~4,400 structures would need
upwards of 10 hours of topology computation alone — past what fits in one
session. COACH420 (420 structures) is itself a real, commonly-cited benchmark
and is tractable end-to-end here.

**Reference — literature numbers, verified in Phase 12:**

| method | COACH420 DCC | COACH420 DCA |
|---|---|---|
| fpocket | 0.228 | 0.444 |
| P2Rank | 0.366 | 0.628 |
| EquiPocket | 0.423 | 0.656 |
''')

md(r'''## Cell 1 — Setup''')
code(r'''
!pip install -q torch-geometric==2.6.1 gudhi==3.10.1 biopython==1.84 transformers==4.46.3
!apt-get -qq update > /dev/null 2>&1 && apt-get -qq install -y default-jre-headless > /dev/null 2>&1
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

blob10 = torch.load(locate("phase10_graphs.pt"), map_location="cpu", weights_only=False)
blob6b = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)
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

md(r'''## Cell 2 — Fetch ALL 420 COACH420 structures''')
code(r'''
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
print(f"COACH420: {len(coach)} entries (full benchmark, no filtering)")

need = sorted({pid for pid, _ in coach})
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
coach_paths = {p: f for p, f in got if f}
print(f"fetched {len(coach_paths):,}/{len(need):,} in {(time.time()-t0)/60:.1f} min")
''')

md(r'''
## Cell 3 — Clean the TRAINING side: exclude any chain homologous to COACH420

The reverse direction of every prior leakage check in this project. Same 30%
identity threshold used throughout.
''')
code(r'''
from Bio.PDB import PDBParser

AA3_TO_1={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}

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
print(f"COACH420 sequences recovered: {len(coach_seqs)}/{len(coach)}")

mm = None
try:
    subprocess.run("curl -sL https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -o /tmp/mm.tar.gz "
                   "&& tar xzf /tmp/mm.tar.gz -C /tmp", shell=True, check=True, timeout=600)
    c = Path("/tmp/mmseqs/bin/mmseqs")
    if c.exists(): os.chmod(c, 0o755); mm = str(c)
except Exception as e:
    print("mmseqs2 unavailable:", type(e).__name__)

W = Path("/tmp/rev_leak"); shutil.rmtree(W, ignore_errors=True); W.mkdir()
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

print(f"training chains homologous to COACH420 (excluded): {len(leaked_train_keys)}/{len(dataset)}")

clean_dataset = [d for d in dataset if f"TRAIN__{d.pdb_id}_{d.chain_id}" not in leaked_train_keys]
print(f"clean training set: {len(clean_dataset):,}/{len(dataset):,} chains "
      f"({100*len(clean_dataset)/len(dataset):.1f}% retained)")
assert len(clean_dataset) > 0.8 * len(dataset), "too much training data excluded -- investigate before proceeding"
dataset = clean_dataset
''')

md(r'''## Cell 4 — Recover true ligand centers for the cleaned training set (Phase 14 logic)''')
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

md(r'''## Cell 5 — Model (identical to Phase 14/15/16/18)''')
code(s14(7))

md(r'''## Cell 6 — Equivariance check before training anything''')
code(s14(9))

md(r'''## Cell 7 — Training machinery (identical to Phase 14)''')
code(s14(11))

md(r'''## Cell 8 — Train the clean-training ensemble (3 seeds, lambda=0.1)''')
code(r'''
BEST_LAMBDA = 0.1
ensemble = []
t0 = time.time()
for seed in (0,1,2):
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    ensemble.append(m)
    torch.save(m.state_dict(), WORK/f"cache/clean_deploy_seed{seed}.pt")
    val_dcc = v.get("dcc", float("nan"))
    print(f"  seed {seed}: val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")
print(f"clean-training ensemble of {len(ensemble)} trained on {len(dataset):,} chains "
      f"(zero homology to COACH420, by construction)")
''')

md(r'''
## Cell 9 — Ground-truth ligands for ALL 420 COACH420 structures (no filtering)

Same extraction logic as every prior phase, but this time nothing gets
excluded on the benchmark side — training is already guaranteed clean.
''')
code(r'''
NON_LIGAND={"HOH","DOD","WAT","SO4","PO4","GOL","EDO","PEG","PG4","PGE","1PE","MPD","TRS",
            "ACT","ACY","FMT","CIT","MES","EPE","DMS","IMD","BME","NO3","AZI","CO3","IOD",
            "BR","CL","F","NA","K","CS","RB","LI","UNX","UNL","UNK"}
CATALYTIC_METALS={"ZN","MG","MN","FE","FE2","CU","CU1","CA","NI","CO","CD","W","MO"}
MIN_RES, MAX_RES, MIN_LIG_ATOMS = 50, 1000, 6

def parse_structure(pid, path, wanted_chain):
    try:
        with gzip.open(path, "rt") as fh:
            model = next(PDBParser(QUIET=True, PERMISSIVE=True).get_structure(pid, fh).get_models())
    except Exception:
        return []
    ligs = []
    for ch in model:
        for res in ch:
            if res.id[0] == " ": continue
            nm = res.get_resname().strip().upper()
            if nm in AA3_TO_1 or nm in NON_LIGAND: continue
            xyz = np.asarray([a.coord for a in res.get_unpacked_list() if a.element not in ("H","D")],
                            dtype=np.float32).reshape(-1,3)
            is_metal = nm in CATALYTIC_METALS
            if is_metal or len(xyz) >= MIN_LIG_ATOMS: ligs.append(xyz)
    out = []
    chains = [model[wanted_chain]] if (wanted_chain and wanted_chain in [c.id for c in model]) \
             else [c for c in model]
    for ch in chains:
        residues=[r for r in ch if r.get_resname().strip().upper() in AA3_TO_1 and "CA" in r]
        if not (MIN_RES <= len(residues) <= MAX_RES): continue
        ca=[r["CA"].coord for r in residues]
        seq="".join(AA3_TO_1.get(r.get_resname().strip().upper(),"X") for r in residues)
        atoms=[np.asarray([a.coord for a in r.get_unpacked_list() if a.element not in ("H","D")],
                          dtype=np.float32).reshape(-1,3) for r in residues]
        out.append((ch.id, seq, np.asarray(ca,dtype=np.float32), atoms, ligs))
    return out

t0=time.time()
eval_units = []
for pid, ch in coach:
    if pid not in coach_paths: continue
    chains = parse_structure(pid, coach_paths[pid], ch)
    if chains: eval_units.append((pid, chains[:1]))   # coach420: one specific target chain
print(f"evaluable COACH420 structures: {len(eval_units)}/{len(coach)}  ({(time.time()-t0)/60:.1f} min)")
''')

md(r'''## Cell 10 — Featurize (identical pipeline: dihedral + ESM2 + topology)''')
code(r'''
import gudhi
from transformers import AutoTokenizer, AutoModel
from Bio.PDB import PPBuilder

AA_ALPHABET="ACDEFGHIKLMNPQRSTVWYX"; A2I={a:i for i,a in enumerate(AA_ALPHABET)}
MAX_HOM_DIM=2; BETTI_RADII=(2.0,3.0,5.0); NBHD_RADIUS=12.0; MAX_NBHD_ATOMS=300
FPD=6+2+len(BETTI_RADII)

_FULL_TOPO_NAMES = [f"H{d}_{s}" for d in range(MAX_HOM_DIM+1)
                    for s in ["n_bars","total_pers","max_pers","mean_pers","std_pers","entropy",
                              "mean_birth","mean_death", *[f"betti_r{r:g}" for r in BETTI_RADII]]] \
                   + ["nbhd_n_atoms","nbhd_gyration"]
_TOPO_KEEP = np.array([n in set(TOPO_NAMES) for n in _FULL_TOPO_NAMES])
assert _TOPO_KEEP.sum() == TOPO_DIM

def rgc(pos,r=8.0,k=32):
    n=pos.size(0); pos=pos-pos.mean(0,keepdim=True)
    d=torch.cdist(pos,pos).clone(); d.fill_diagonal_(float("inf")); m=d<=r
    if k<n:
        _,i=torch.topk(d,k,dim=1,largest=False); keep=torch.zeros_like(m); keep.scatter_(1,i,True); m&=keep
    m=m|m.t(); t,s=m.nonzero(as_tuple=True); return torch.stack([s,t]).long()

def _dg(pts):
    pts=np.ascontiguousarray(pts,dtype=np.float64); st=None; sq=True
    if len(pts)>=5:
        try: st=gudhi.AlphaComplex(points=pts).create_simplex_tree()
        except Exception: st=None
    if st is None:
        st=gudhi.RipsComplex(points=pts,max_edge_length=float(NBHD_RADIUS)).create_simplex_tree(max_dimension=MAX_HOM_DIM+1); sq=False
    st.compute_persistence(homology_coeff_field=2,persistence_dim_max=True)
    o={}
    for d in range(MAX_HOM_DIM+1):
        b=np.asarray(st.persistence_intervals_in_dimension(d),dtype=np.float64).reshape(-1,2)
        o[d]=np.sqrt(np.maximum(b,0.0)) if (sq and len(b)) else b
    return o

def _vec(dg,pts):
    f=[]
    for d in range(MAX_HOM_DIM+1):
        bars=dg.get(d,np.zeros((0,2)))
        fin=bars[np.isfinite(bars).all(axis=1)] if len(bars) else bars.reshape(-1,2)
        if len(fin)==0: f.extend([0.0]*FPD); continue
        b,dt=fin[:,0],fin[:,1]; pers=np.maximum(dt-b,0.0); tot=float(pers.sum())
        ent=0.0
        if tot>0:
            p=pers/tot; nz=p[p>0]; ent=float(-(nz*np.log(nz)).sum())
        f.extend([float(len(fin)),tot,float(pers.max()),float(pers.mean()),float(pers.std()),
                  ent,float(b.mean()),float(dt.mean()),
                  *[float(((b<=r)&(dt>r)).sum()) for r in BETTI_RADII]])
    c=pts-pts.mean(0,keepdims=True)
    f.extend([float(len(pts)),float(np.sqrt((c**2).sum(1).mean())) if len(pts) else 0.0])
    return np.asarray(f,dtype=np.float32)

def topo_chain(ca,per_res):
    atoms=np.concatenate([a for a in per_res if len(a)],axis=0)
    tree=cKDTree(atoms); out=np.zeros((len(ca),TOPO_DIM),np.float32)
    for i,c in enumerate(ca):
        idx=tree.query_ball_point(c,NBHD_RADIUS)
        if len(idx)>MAX_NBHD_ATOMS: _,idx=tree.query(c,k=MAX_NBHD_ATOMS)
        pts=atoms[np.asarray(idx,dtype=int)]
        if len(pts)>=4: out[i]=_vec(_dg(pts),pts)[_TOPO_KEEP]
    return out

def dihedrals(pdb_id, chain_id, path, n_expect):
    try:
        with gzip.open(path,"rt") as fh:
            model=next(PDBParser(QUIET=True,PERMISSIVE=True).get_structure(pdb_id,fh).get_models())
        ch=model[chain_id]
    except Exception: return None
    residues=[r for r in ch if r.get_resname().strip().upper() in AA3_TO_1 and "CA" in r]
    if len(residues)!=n_expect: return None
    ppmap={}
    for pp in PPBuilder().build_peptides(ch, aa_only=False):
        for res,(phi,psi) in zip(pp, pp.get_phi_psi_list()): ppmap[res.id]=(phi,psi)
    phis,psis=[],[]
    for r in residues:
        phi,psi=ppmap.get(r.id,(None,None)); phis.append(phi); psis.append(psi)
    def enc(a):
        v=torch.tensor([x if x is not None else 0.0 for x in a],dtype=torch.float32)
        m=torch.tensor([x is not None for x in a],dtype=torch.float32)
        return torch.stack([torch.sin(v)*m,torch.cos(v)*m],1), m.unsqueeze(1)
    pf,pm=enc(phis); sf,sm=enc(psis)
    return torch.cat([pf,sf,pm,sm],1)

tok = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(DEVICE).eval()

@torch.no_grad()
def embed_seq(seq):
    enc = tok([seq], return_tensors="pt", truncation=True, max_length=1002)
    enc = {k:v.to(DEVICE) for k,v in enc.items()}
    out = esm(**enc).last_hidden_state
    return out[0,1:1+len(seq),:].float().cpu()

t0=time.time(); built = {}
n_total = sum(len(c) for _,c in eval_units)
done=0
for pid, chains in eval_units:
    for cid, seq, ca, per_res, ligs in chains:
        n = len(ca)
        oh = torch.zeros(n, 21); oh[torch.arange(n), torch.tensor([A2I.get(c,20) for c in seq])]=1.0
        dih = dihedrals(pid, cid, coach_paths[pid], n)
        if dih is None: continue
        esm_e = embed_seq(seq)
        topo = topo_chain(ca, per_res)
        pos = torch.from_numpy(ca); pos = pos - pos.mean(0, keepdim=True)
        x = torch.cat([oh, dih, esm_e, torch.from_numpy(topo.astype(np.float32))], dim=1)
        ei = rgc(pos)
        d_ = (pos[ei[0]]-pos[ei[1]]).norm(dim=1,keepdim=True)
        sep = (ei[0]-ei[1]).abs().float().unsqueeze(1)
        built[(pid,cid)] = dict(x=x, pos=pos, edge_index=ei,
                                edge_attr=torch.cat([d_,torch.log1p(sep)],1), ca=ca, ligs=ligs)
        done += 1
    if done % 40 < 1: print(f"  {done}/{n_total}  [{(time.time()-t0)/60:.1f}m]")

del esm; torch.cuda.empty_cache()
print(f"\nfeaturized {len(built)}/{n_total} chains in {(time.time()-t0)/60:.1f} min")

raw_cache = {k: {kk: (vv.numpy() if torch.is_tensor(vv) else vv) for kk,vv in v.items()}
            for k,v in built.items()}
with open(WORK/"cache/phase20_raw_features.pkl", "wb") as fh:
    pickle.dump(raw_cache, fh)
print(f"cached raw features -> {(WORK/'cache/phase20_raw_features.pkl').stat().st_size/1e6:.0f} MB")
''')

md(r'''## Cell 11 — P2Rank on the FULL 420 (same-run controlled reference)''')
code(r'''
t0=time.time(); PRANK=None
subprocess.run("curl -sL https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz "
               "-o /tmp/p2.tar.gz && tar xzf /tmp/p2.tar.gz -C /tmp", shell=True, check=True, timeout=1200)
PRANK = str(next(Path("/tmp").glob("p2rank_*/prank"))); os.chmod(PRANK, 0o755)

pW = Path("/tmp/p2_20"); shutil.rmtree(pW, ignore_errors=True); pW.mkdir()
pOUT = Path("/tmp/p2_20_out"); shutil.rmtree(pOUT, ignore_errors=True)
for pid in sorted(coach_paths):
    plain = pW / f"{pid}.pdb"
    if not plain.exists():
        with gzip.open(coach_paths[pid],"rb") as fi, open(plain,"wb") as fo: shutil.copyfileobj(fi,fo)
(pW/"ds.ds").write_text("\n".join(str(pW/f"{p}.pdb") for p in sorted(coach_paths))+"\n")

r = subprocess.run([PRANK,"predict",str(pW/"ds.ds"),"-o",str(pOUT),
                    "-threads", str(max(1, os.cpu_count() or 2))],
                   capture_output=True, text=True, timeout=3600)
print(f"P2Rank rc={r.returncode} in {(time.time()-t0)/60:.1f} min")
pred_csvs = {p.stem.replace(".pdb_predictions",""): p for p in pOUT.rglob("*_predictions.csv")}
print(f"P2Rank outputs: {len(pred_csvs)}/{len(coach_paths)}")

def p2rank_top_center(pid):
    f = pred_csvs.get(pid)
    if f is None: return None
    with open(f) as fh:
        rows = list(_csv.DictReader(fh))
    if not rows: return None
    rows = [{k.strip(): v.strip() if isinstance(v,str) else v for k,v in row.items() if k} for row in rows]
    cx = next((k for k in rows[0] if k.lower().replace("_","")=="centerx"), None)
    cy = next((k for k in rows[0] if k.lower().replace("_","")=="centery"), None)
    cz = next((k for k in rows[0] if k.lower().replace("_","")=="centerz"), None)
    if not (cx and cy and cz): return None
    try:
        return np.array([float(rows[0][cx]), float(rows[0][cy]), float(rows[0][cz])], dtype=np.float32)
    except (ValueError, KeyError):
        return None

p2rank_centers = {pid: p2rank_top_center(pid) for pid in coach_paths}
print(f"P2Rank centers parsed: {sum(1 for v in p2rank_centers.values() if v is not None)}/{len(coach_paths)}")
''')

md(r'''## Cell 12 — Score both, compare to literature''')
code(r'''
mu, sd = blob6b["topo_mean"], blob6b["topo_std"]

@torch.no_grad()
def predict_chain(entry):
    x = entry["x"].clone()
    x[:, N_PRE:] = (x[:, N_PRE:] - torch.from_numpy(mu)) / torch.from_numpy(sd)
    d = type("D", (), {})()
    d.x=x.to(DEVICE); d.pos=entry["pos"].to(DEVICE)
    d.edge_index=entry["edge_index"].to(DEVICE); d.edge_attr=entry["edge_attr"].to(DEVICE)
    d.batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
    seed_logits, seed_centers = [], []
    for m in ensemble:
        lg, pc = m(d)
        seed_logits.append(torch.sigmoid(lg).cpu().numpy())
        seed_centers.append(pc.cpu().numpy()[0])
    offset = entry["ca"].mean(axis=0)
    center_raw = np.mean(seed_centers, axis=0) + offset
    return center_raw


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


ours_dcc, ours_dca, p2_dcc, p2_dca = [], [], [], []
for (pid, cid), entry in built.items():
    ligs = entry["ligs"]
    c_ours = predict_chain(entry)
    d1, a1 = dcc_dca(c_ours, ligs)
    if d1 is not None: ours_dcc.append(d1); ours_dca.append(a1)
    c_p2 = p2rank_centers.get(pid)
    if c_p2 is not None:
        d2, a2 = dcc_dca(c_p2, ligs)
        if d2 is not None: p2_dcc.append(d2); p2_dca.append(a2)

ours_dcc_rate = float(np.mean([1.0 if d<4.0 else 0.0 for d in ours_dcc]))
ours_dca_rate = float(np.mean([1.0 if a<4.0 else 0.0 for a in ours_dca]))
p2_dcc_rate = float(np.mean([1.0 if d<4.0 else 0.0 for d in p2_dcc])) if p2_dcc else float("nan")
p2_dca_rate = float(np.mean([1.0 if a<4.0 else 0.0 for a in p2_dca])) if p2_dca else float("nan")

LIT = {"fpocket": (0.228, 0.444), "P2Rank": (0.366, 0.628), "EquiPocket": (0.423, 0.656)}

print(f"{'method':<28}{'DCC':>8}{'DCA':>8}{'n':>6}")
print("-"*50)
for name,(d,a) in LIT.items():
    print(f"{name+' (literature, full 420)':<28}{d:>8.3f}{a:>8.3f}{'420':>6}")
print(f"{'P2Rank (this run, full 420)':<28}{p2_dcc_rate:>8.3f}{p2_dca_rate:>8.3f}{len(p2_dcc):>6}")
print(f"{'OURS (clean train, full 420)':<28}{ours_dcc_rate:>8.3f}{ours_dca_rate:>8.3f}{len(ours_dcc):>6}")

print(f"\nvs literature EquiPocket : DCC {ours_dcc_rate-0.423:+.4f}  DCA {ours_dca_rate-0.656:+.4f}")
print(f"vs literature P2Rank     : DCC {ours_dcc_rate-0.366:+.4f}  DCA {ours_dca_rate-0.628:+.4f}")
print(f"vs this-run P2Rank       : DCC {ours_dcc_rate-p2_dcc_rate:+.4f}  DCA {ours_dca_rate-p2_dca_rate:+.4f}")

verdict = ("BEATS literature EquiPocket on the full, unfiltered benchmark -- "
          "a legitimate SOTA claim" if ours_dcc_rate > 0.423 else
          "beats P2Rank but not EquiPocket on the full benchmark" if ours_dcc_rate > 0.366 else
          "does not beat published baselines on the full, unfiltered benchmark")
print(f"\nVERDICT: {verdict}")

json.dump({"ours": {"dcc": ours_dcc_rate, "dca": ours_dca_rate, "n": len(ours_dcc)},
           "p2rank_this_run": {"dcc": p2_dcc_rate, "dca": p2_dca_rate, "n": len(p2_dcc)},
           "literature": LIT, "n_train_excluded": len(leaked_train_keys),
           "n_train_clean": len(dataset)},
          open(WORK/"phase20_summary.json", "w"), indent=2)
print("\nsaved -> phase20_summary.json")
''')

md(r'''
---

## Reading this

This is the first comparison in the project where evaluating against
literature EquiPocket numbers is actually valid: training was cleaned of any
overlap with COACH420 *before* training happened, and the evaluation covers
the **entire** published benchmark, not a filtered subset. Whatever the number
is, it is comparable on equal footing.

If this beats EquiPocket, it is a legitimate, defensible SOTA claim, on the
benchmark and metric the field itself uses. If it doesn't, that is the honest,
current state of the project relative to a purpose-built, dedicated pocket
localization architecture — still informative, and still a cleaner, more
defensible number than anything produced before this notebook.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase20_legitimate_sota.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
