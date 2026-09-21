#!/usr/bin/env python3
"""Phase 12: benchmark against real SOTA (EquiPocket/P2Rank literature numbers)
on the standard COACH420 + HOLO4K pocket-prediction benchmarks, with an explicit
train/test leakage check against our own 2,056-cluster training set."""
import json
from pathlib import Path

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 12 — Benchmarking Against Real SOTA: COACH420 + HOLO4K

A literature check (see project discussion) found that E(3)-equivariant GNNs for
binding-site prediction are an active, published field — **EquiPocket** (2023)
explicitly claims to be first-to-market on this exact task, and **VN-EGNN**
(2025) is current state-of-the-art. Our earlier win over P2Rank was real but
P2Rank is a 2018 tool; the question that actually matters is whether this system
is competitive with 2023–2025 SOTA, on the benchmarks that field actually uses.

**COACH420** and **HOLO4K** are the standard pocket-prediction benchmarks. Every
paper above reports on them with the same metric:

- **DCC** — distance from the predicted pocket center to the true ligand's
  center of mass. Success if < 4 Å.
- **DCA** — distance from the predicted pocket center to the *nearest atom* of
  the true ligand. Looser than DCC (ligands are not points), success if < 4 Å.

**Published numbers (from the EquiPocket paper, verified before writing this notebook):**

| method | COACH420 DCC | COACH420 DCA | HOLO4K DCC | HOLO4K DCA |
|---|---|---|---|---|
| fpocket | 0.228 | 0.444 | 0.192 | 0.457 |
| P2Rank | 0.366 | 0.628 | 0.314 | 0.621 |
| EquiPocket | 0.423 | 0.656 | 0.337 | 0.662 |

VN-EGNN reports beating EquiPocket on the same benchmarks but the exact table
could not be reliably extracted from the paper; it is not included in the
numeric comparison below, only named as context.

---

### Why this is not a fully controlled comparison, stated up front

1. **Different training data.** EquiPocket/P2Rank's numbers come from models
   trained on their own (undisclosed-to-us) training sets, evaluated on the full
   420 / ~4,360 structures. Our model is trained on the 2,056-cluster set built
   in Phases 6–11. If any COACH420/HOLO4K structure — or a close homolog of one —
   is in our training set, our number on it is not a fair test. **This notebook
   checks for that explicitly** (exact PDB-ID overlap, and MMseqs2 homology at
   the same 30% threshold used throughout this project) and reports metrics
   **only on the leakage-clean subset**, with the excluded count stated plainly.
2. **We could not run EquiPocket or VN-EGNN's own code** in this environment —
   both are research repos with nontrivial dependency stacks, and reproducing
   them reliably in one session was judged too high-risk to gate this analysis
   on. Their numbers here are the published ones, on the full benchmark, not a
   number we generated ourselves. P2Rank, by contrast, **is** re-run here by us,
   on the same clean subset as our own model — that comparison is fully
   apples-to-apples.
3. Because of (1), our DCC/DCA is computed on however many clean structures
   survive the leakage filter — likely fewer than the full 420 / ~4,360 the
   literature numbers use. A smaller, cleaner test set is the more honest
   choice, but it does mean the two numbers are not on identical Ns.

### One more design decision: converting per-residue scores into a pocket center

Our model outputs a probability per residue, not a ranked list of discrete
pockets. To compute DCC/DCA at all, predicted-positive residues are spatially
clustered (connected components at the same 8 Å radius used for the model's own
message-passing graph), each cluster is scored by its total predicted
probability, and the highest-scoring cluster's centroid becomes the "predicted
pocket center" — the standard way this conversion is done in the literature.
''')

md(r'''## Cell 1 — Setup''')

code(r'''
!pip install -q torch-geometric==2.6.1 gudhi==3.10.1 biopython==1.84 transformers==4.46.3
!apt-get -qq update > /dev/null 2>&1 && apt-get -qq install -y default-jre-headless > /dev/null 2>&1
''')

code(r'''
import csv as _csv
import gzip, json, os, shutil, subprocess, time, urllib.request, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

blob10 = torch.load(locate("phase10_graphs.pt"), map_location="cpu", weights_only=False)
blob6b = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)
train_dataset = blob10["dataset"]
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
print(f"training set: {len(train_dataset):,} chains, node dim {train_dataset[0].x.size(1)}")

WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)
RAW = Path("/tmp/pdb_raw"); RAW.mkdir(parents=True, exist_ok=True)
''')

md(r'''
## Cell 2 — Fetch the benchmark ID lists and structures

`coach420.ds` lines are `coach420/<4-char-PDB-ID><chain-letter>.pdb` — a specific
target chain per entry. `holo4k.ds` lines are `holo4k/<4-char-PDB-ID>.pdb` — the
whole assembly is evaluated, so at prediction time every eligible protein chain
in the structure is scored and the best pocket across all of them is taken.
''')

code(r'''
def parse_ds(url, prefix):
    lines = urllib.request.urlopen(url, timeout=60).read().decode().splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        name = ln.split("/")[-1].replace(".pdb", "")
        if prefix == "coach420":
            out.append((name[:4].upper(), name[4:].upper() or None))   # (pdb_id, chain or None)
        else:
            out.append((name[:4].upper(), None))
    return out

coach = parse_ds("https://raw.githubusercontent.com/rdk/p2rank-datasets/master/coach420.ds", "coach420")
holo  = parse_ds("https://raw.githubusercontent.com/rdk/p2rank-datasets/master/holo4k.ds", "holo4k")
print(f"COACH420: {len(coach)} entries | HOLO4K: {len(holo)} entries")

bench = [("coach420", pid, ch) for pid, ch in coach] + [("holo4k", pid, ch) for pid, ch in holo]
need = sorted({pid for _, pid, _ in bench})
print(f"unique PDB IDs to fetch: {len(need):,}")

def fetch(pid):
    p = RAW / f"{pid}.pdb.gz"
    if p.exists() and p.stat().st_size > 500: return pid, p
    try:
        with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.pdb.gz", timeout=60) as r:
            b = r.read()
        if len(b) > 500: p.write_bytes(b); return pid, p
    except Exception: pass
    return pid, None

t0=time.time()
with ThreadPoolExecutor(max_workers=24) as ex: got = list(ex.map(fetch, need))
paths = {p: f for p, f in got if f}
print(f"fetched {len(paths):,}/{len(need):,} in {(time.time()-t0)/60:.1f} min")
''')

md(r'''
## Cell 3 — Leakage check: exact ID overlap + sequence-homology overlap

Uses the same MMseqs2 threshold (30% identity) as every clustering step in this
project, so "leaked" here means exactly what "same cluster" has meant
throughout — not a stricter or looser bar invented just for this comparison.
''')

code(r'''
from Bio.PDB import PDBParser

AA3_TO_1={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
NON_LIGAND={"HOH","DOD","WAT","SO4","PO4","GOL","EDO","PEG","PG4","PGE","1PE","MPD","TRS",
            "ACT","ACY","FMT","CIT","MES","EPE","DMS","IMD","BME","NO3","AZI","CO3","IOD",
            "BR","CL","F","NA","K","CS","RB","LI","UNX","UNL","UNK"}
CATALYTIC_METALS={"ZN","MG","MN","FE","FE2","CU","CU1","CA","NI","CO","CD","W","MO"}
MIN_RES, MAX_RES, MIN_LIG_ATOMS = 50, 1000, 6

train_pdb_ids = {d.pdb_id for d in train_dataset}
overlap_exact = {pid for _, pid, _ in bench if pid in train_pdb_ids}
print(f"exact PDB-ID overlap with training set: {len(overlap_exact)}/{len(need)} unique IDs")

def parse_structure(pid, path, wanted_chain):
    """Returns list of (chain_id, seq, ca_xyz, per_res_atoms, ligand_list)."""
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
            if is_metal or len(xyz) >= MIN_LIG_ATOMS:
                ligs.append(xyz)
    out = []
    chains = [c for c in model] if wanted_chain is None else \
             [model[wanted_chain]] if wanted_chain in [c.id for c in model] else []
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
parsed = {}   # pdb_id -> list of chain tuples, ALL chains parsed once per structure
for pid in sorted(paths):
    parsed[pid] = parse_structure(pid, paths[pid], None)
print(f"parsed {len(parsed)} structures in {(time.time()-t0)/60:.1f} min")

# assemble the evaluation unit list: one entry per (benchmark, pdb_id, target_chain_or_None)
eval_units = []
for src, pid, ch in bench:
    chains = parsed.get(pid, [])
    if not chains: continue
    if src == "coach420":
        match = [c for c in chains if c[0] == ch]
        if match: eval_units.append((src, pid, [match[0]]))
    else:
        # HOLO4K evaluates the whole assembly, but some entries are large
        # multi-chain complexes (viral capsids etc.) -- cap to the 6 longest
        # chains so one pathological structure cannot dominate runtime. This
        # covers the large majority of biologically relevant subunits; the
        # count actually capped is reported below rather than silently dropped.
        if chains:
            capped = sorted(chains, key=lambda c: -len(c[1]))[:6]
            eval_units.append((src, pid, capped))
n_capped = sum(1 for src,pid,ch in eval_units if src=="holo4k"
              and len(parsed.get(pid,[])) > len(ch))
print(f"evaluable units (structure parsed, target chain present): {len(eval_units)}/{len(bench)}")
print(f"HOLO4K structures with >6 chains, capped to 6 longest: {n_capped}")
''')

code(r'''
# --- MMseqs2 homology check against the training set --------------------------
mm = None
try:
    subprocess.run("curl -sL https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -o /tmp/mm.tar.gz "
                   "&& tar xzf /tmp/mm.tar.gz -C /tmp", shell=True, check=True, timeout=600)
    c = Path("/tmp/mmseqs/bin/mmseqs")
    if c.exists(): os.chmod(c, 0o755); mm = str(c)
except Exception as e:
    print("mmseqs2 unavailable:", type(e).__name__)

W = Path("/tmp/leak_mm"); shutil.rmtree(W, ignore_errors=True); W.mkdir()
fasta_lines = []
for d in train_dataset:
    fasta_lines.append(f">TRAIN__{d.pdb_id}_{d.chain_id}\n{d.seq}\n")
bench_seq_map = {}
for src, pid, chains in eval_units:
    for cid, seq, *_ in chains:
        key = f"BENCH__{src}__{pid}_{cid}"
        bench_seq_map[key] = (src, pid, cid)
        fasta_lines.append(f">{key}\n{seq}\n")
(W/"all.fasta").write_text("".join(fasta_lines))

subprocess.run([mm, "easy-cluster", str(W/"all.fasta"), str(W/"res"), str(W/"tmp"),
                "--min-seq-id", "0.30", "-c", "0.8", "--cov-mode", "1", "-v", "1"],
               check=True, capture_output=True, timeout=3600)
rep = defaultdict(list)
for line in (W/"res_cluster.tsv").read_text().splitlines():
    a, b = line.split("\t"); rep[a].append(b)
clusters = list(rep.values()) if rep else []
# merge singleton reps (mmseqs always includes the rep itself in its own cluster list)
for r_, members in rep.items():
    if r_ not in members: members.append(r_)

leaked_keys = set()
for members in rep.values():
    has_train = any(m.startswith("TRAIN__") for m in members)
    has_bench = any(m.startswith("BENCH__") for m in members)
    if has_train and has_bench:
        leaked_keys.update(m for m in members if m.startswith("BENCH__"))

print(f"benchmark chains flagged as homologous (>=30% identity) to a training chain: "
      f"{len(leaked_keys)}/{len(bench_seq_map)}")

clean_units = []
for src, pid, chains in eval_units:
    clean_chains = [c for c in chains if f"BENCH__{src}__{pid}_{c[0]}" not in leaked_keys
                    and pid not in overlap_exact]
    if clean_chains:
        clean_units.append((src, pid, clean_chains))
n_dropped = len(eval_units) - len(clean_units)
print(f"\nclean (non-leaked) evaluation units: {len(clean_units)}/{len(eval_units)} "
      f"({n_dropped} dropped to leakage)")
for name in ("coach420","holo4k"):
    tot = sum(1 for s,_,_ in eval_units if s==name)
    cln = sum(1 for s,_,_ in clean_units if s==name)
    print(f"  {name:<10} {cln}/{tot} clean")
''')

md(r'''
## Cell 4 — Train the deployment ensemble on the FULL 2,056-chain set

No internal held-out test here — the external benchmark is the test set now.
Reuses the winning Phase 11b recipe (weighted BCE, 3 seeds, average predicted
probabilities). A small internal validation slice is still carved out purely
for early stopping.
''')

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
class Net(nn.Module):
    def __init__(s,n_pre,td,h=128,L=4,dr=0.1):
        super().__init__(); s.n_pre=n_pre
        s.embed=nn.Sequential(nn.Linear(n_pre+td,h),nn.SiLU(),nn.Linear(h,h))
        s.ls=nn.ModuleList([EGCL(h,uc=(i<L-1)) for i in range(L)])
        s.fs=nn.ModuleList([Fus(h,td) for _ in range(L)])
        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))
    def forward(s,d):
        h=s.embed(d.x); pos=d.pos; t=d.x[:,s.n_pre:]
        for l,f in zip(s.ls,s.fs): h,pos=l(h,pos,d.edge_index,d.edge_attr); h=f(h,t)
        return s.head(h).squeeze(-1)

def run_epoch(model, loader, opt=None):
    tr = opt is not None; model.train(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    ps,ys=[],[]
    for b in loader:
        b=b.to(DEVICE)
        with torch.set_grad_enabled(tr):
            lg=model(b); loss=crit(lg,b.y)
        if tr:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        ps.append(torch.sigmoid(lg).detach().cpu().numpy()); ys.append(b.y.cpu().numpy())
    from sklearn.metrics import average_precision_score
    p,y=np.concatenate(ps),np.concatenate(ys)
    return {"pr": average_precision_score(y,p) if y.sum() else float("nan")}

from torch_geometric.loader import DataLoader
rng = np.random.RandomState(42)
perm = rng.permutation(len(train_dataset))
n_val = max(1, int(0.08*len(perm)))
val_ds = [train_dataset[i] for i in perm[:n_val]]
tr_ds  = [train_dataset[i] for i in perm[n_val:]]
print(f"deployment training: {len(tr_ds)} train / {len(val_ds)} val (early stop only)")

ensemble = []
t0=time.time()
for seed in (0,1,2):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net(N_PRE, TOPO_DIM).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    tl = DataLoader(tr_ds, batch_size=16, shuffle=True); vl = DataLoader(val_ds, batch_size=16)
    best, state, bad = -1, None, 0
    for ep in range(40):
        run_epoch(m, tl, opt); v = run_epoch(m, vl); sch.step()
        if v["pr"] > best: best, state, bad = v["pr"], {k:t.clone() for k,t in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8: break
    m.load_state_dict(state); m.eval()
    ensemble.append(m)
    torch.save(state, WORK/f"cache/deploy_seed{seed}.pt")
    print(f"  seed {seed}: val PR {best:.4f}  [{(time.time()-t0)/60:.1f}m]")
print(f"ensemble of {len(ensemble)} models trained on all {len(train_dataset):,} chains")
''')

md(r'''
## Cell 5 — Featurize the clean benchmark chains (identical pipeline throughout)

Same radius graph, same dihedral encoding, same ESM2 model, same topology
parameters as every training phase — a benchmark evaluated on a different
feature pipeline than the one the model was trained on would not be a fair test.
''')

code(r'''
import gudhi
from transformers import AutoTokenizer, AutoModel
from Bio.PDB import PPBuilder

AA_ALPHABET="ACDEFGHIKLMNPQRSTVWYX"; A2I={a:i for i,a in enumerate(AA_ALPHABET)}
MAX_HOM_DIM=2; BETTI_RADII=(2.0,3.0,5.0); NBHD_RADIUS=12.0; MAX_NBHD_ATOMS=300
FPD=6+2+len(BETTI_RADII)

# Phase 6b pruned one constant column (35 raw features -> 34) before the model
# ever saw them. Reconstruct the full 35-name list with the SAME generation
# logic used in every prior phase, then match by NAME against blob6b's saved
# 34-name list to recover which column was dropped -- matching by name (not a
# hardcoded index) means this self-verifies via the assert rather than silently
# drifting if the pruned column ever changes.
_FULL_TOPO_NAMES = [f"H{d}_{s}" for d in range(MAX_HOM_DIM+1)
                    for s in ["n_bars","total_pers","max_pers","mean_pers","std_pers","entropy",
                              "mean_birth","mean_death", *[f"betti_r{r:g}" for r in BETTI_RADII]]]                    + ["nbhd_n_atoms","nbhd_gyration"]
_TOPO_KEEP = np.array([n in set(TOPO_NAMES) for n in _FULL_TOPO_NAMES])
assert _TOPO_KEEP.sum() == TOPO_DIM,     f"expected {TOPO_DIM} surviving topo columns, matched {_TOPO_KEEP.sum()} by name"
print(f"topo pruning reconstructed: {len(_FULL_TOPO_NAMES)} raw -> {_TOPO_KEEP.sum()} kept "
      f"(dropped: {[n for n,k in zip(_FULL_TOPO_NAMES,_TOPO_KEEP) if not k]})")

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

t0=time.time(); built = {}   # key -> Data-like dict
n_chains_total = sum(len(c) for _,_,c in clean_units)
done=0
for src, pid, chains in clean_units:
    for cid, seq, ca, per_res, ligs in chains:
        n = len(ca)
        oh = torch.zeros(n, 21); oh[torch.arange(n), torch.tensor([A2I.get(c,20) for c in seq])]=1.0
        dih = dihedrals(pid, cid, paths[pid], n)
        if dih is None: continue
        esm_e = embed_seq(seq)
        topo = topo_chain(ca, per_res)
        pos = torch.from_numpy(ca); pos = pos - pos.mean(0, keepdim=True)
        x = torch.cat([oh, dih, esm_e, torch.from_numpy(topo.astype(np.float32))], dim=1)
        ei = rgc(pos)
        d_ = (pos[ei[0]]-pos[ei[1]]).norm(dim=1,keepdim=True)
        sep = (ei[0]-ei[1]).abs().float().unsqueeze(1)
        built[(src,pid,cid)] = dict(x=x, pos=pos, edge_index=ei,
                                    edge_attr=torch.cat([d_,torch.log1p(sep)],1),
                                    ca=ca, ligs=ligs)
        done += 1
    if done % 100 < len(chains): print(f"  {done}/{n_chains_total}  [{(time.time()-t0)/60:.1f}m]")

del esm; torch.cuda.empty_cache()
print(f"\nfeaturized {len(built)}/{n_chains_total} clean chains in {(time.time()-t0)/60:.1f} min")
torch.save({"built_keys": list(built.keys())}, WORK/"cache/built_index.pt")
''')

md(r'''
## Cell 6 — Ensemble inference, then convert per-residue scores into pocket centers

Standardization uses the training set's own mean/std (from Phase 6b), never
statistics computed on the benchmark itself.
''')

code(r'''
mu, sd = blob6b["topo_mean"], blob6b["topo_std"]
N_SEQ_DIH_ESM = N_PRE   # 21 + 6 + 480

@torch.no_grad()
def predict_chain(entry):
    x = entry["x"].clone()
    x[:, N_SEQ_DIH_ESM:] = (x[:, N_SEQ_DIH_ESM:] - torch.from_numpy(mu)) / torch.from_numpy(sd)
    d = type("D", (), {})()
    d.x=x.to(DEVICE); d.pos=entry["pos"].to(DEVICE)
    d.edge_index=entry["edge_index"].to(DEVICE); d.edge_attr=entry["edge_attr"].to(DEVICE)
    probs = torch.stack([torch.sigmoid(m(d)) for m in ensemble], dim=0).mean(0)
    return probs.cpu().numpy()

def cluster_pockets(ca, probs, edge_index, top_frac=0.15, min_prob=0.3):
    """Connected components (via the model's own 8A graph) among high-scoring
    residues; returns list of (centroid_xyz, score) sorted best-first."""
    n = len(ca)
    thresh = max(min_prob, np.quantile(probs, 1-top_frac))
    hot = np.where(probs >= thresh)[0]
    if len(hot) == 0: return []
    hot_set = set(hot.tolist())
    adj = defaultdict(list)
    ei = edge_index.numpy()
    for a,b in zip(ei[0], ei[1]):
        if a in hot_set and b in hot_set: adj[a].append(b); adj[b].append(a)
    seen=set(); clusters=[]
    for start in hot:
        if start in seen: continue
        stack=[start]; comp=[]
        while stack:
            u=stack.pop()
            if u in seen: continue
            seen.add(u); comp.append(u)
            stack.extend(adj[u])
        comp=np.array(comp)
        centroid = ca[comp].mean(axis=0)
        score = float(probs[comp].sum())
        clusters.append((centroid, score))
    clusters.sort(key=lambda c: -c[1])
    return clusters

t0=time.time()
predictions = {}   # (src,pid) -> (top1_center, all_ligs)
by_structure = defaultdict(list)
for (src,pid,cid), entry in built.items():
    by_structure[(src,pid)].append((cid, entry))

for (src,pid), chain_list in by_structure.items():
    all_clusters = []
    ligs = chain_list[0][1]["ligs"]
    for cid, entry in chain_list:
        probs = predict_chain(entry)
        cl = cluster_pockets(entry["ca"], probs, entry["edge_index"])
        all_clusters.extend(cl)
    if not all_clusters: continue
    all_clusters.sort(key=lambda c: -c[1])
    predictions[(src,pid)] = (all_clusters[0][0], ligs)

print(f"predicted pocket centers for {len(predictions)} structures in {(time.time()-t0)/60:.1f} min")
''')

md(r'''
## Cell 7 — P2Rank on the same clean structures (same-run, apples-to-apples)

P2Rank's own top-ranked pocket center is read directly from `*_predictions.csv`
— no proxy needed here, unlike our own residue-clustering step, since P2Rank
already outputs discrete ranked pockets with coordinates.
''')

code(r'''
t0=time.time(); PRANK=None
subprocess.run("curl -sL https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz "
               "-o /tmp/p2.tar.gz && tar xzf /tmp/p2.tar.gz -C /tmp", shell=True, check=True, timeout=1200)
PRANK = str(next(Path("/tmp").glob("p2rank_*/prank"))); os.chmod(PRANK, 0o755)
print(f"P2Rank ready in {time.time()-t0:.0f}s")

pW = Path("/tmp/p2bench_in"); shutil.rmtree(pW, ignore_errors=True); pW.mkdir()
pOUT = Path("/tmp/p2bench_out"); shutil.rmtree(pOUT, ignore_errors=True)
clean_pids = sorted({pid for _,pid in predictions})
for pid in clean_pids:
    plain = pW / f"{pid}.pdb"
    if not plain.exists():
        with gzip.open(paths[pid],"rb") as fi, open(plain,"wb") as fo: shutil.copyfileobj(fi,fo)
(pW/"ds.ds").write_text("\n".join(str(pW/f"{p}.pdb") for p in clean_pids)+"\n")

t0=time.time()
r = subprocess.run([PRANK,"predict",str(pW/"ds.ds"),"-o",str(pOUT),
                    "-threads", str(max(1, os.cpu_count() or 2))],
                   capture_output=True, text=True, timeout=10800)
print(f"P2Rank rc={r.returncode} in {(time.time()-t0)/60:.1f} min")

pred_csvs = {p.stem.replace(".pdb_predictions",""): p for p in pOUT.rglob("*_predictions.csv")}
print(f"prediction files: {len(pred_csvs)}/{len(clean_pids)}")

# probe the header ONCE, defensively, rather than assuming column names
if pred_csvs:
    probe_f = next(iter(pred_csvs.values()))
    with open(probe_f) as fh:
        header = fh.readline()
    print("predictions.csv header:", header.strip())

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
    r0 = rows[0]   # already rank-ordered by P2Rank
    try:
        return np.array([float(r0[cx]), float(r0[cy]), float(r0[cz])], dtype=np.float32)
    except (ValueError, KeyError):
        return None

p2rank_centers = {pid: p2rank_top_center(pid) for pid in clean_pids}
n_ok = sum(1 for v in p2rank_centers.values() if v is not None)
print(f"P2Rank top-pocket centers parsed: {n_ok}/{len(clean_pids)}")
''')

md(r'''## Cell 8 — DCC / DCA scoring, matched to the literature definition''')

code(r'''
def dcc_dca(pred_center, ligs):
    """Best-case over all true ligands in the structure, as the literature does."""
    if pred_center is None or not ligs: return None, None
    best_dcc, best_dca = np.inf, np.inf
    for lig_atoms in ligs:
        if len(lig_atoms) == 0: continue
        com = lig_atoms.mean(axis=0)
        best_dcc = min(best_dcc, float(np.linalg.norm(pred_center - com)))
        best_dca = min(best_dca, float(np.linalg.norm(lig_atoms - pred_center, axis=1).min()))
    return (best_dcc if np.isfinite(best_dcc) else None,
            best_dca if np.isfinite(best_dca) else None)

results = {"coach420": {"ours": [], "p2rank": []}, "holo4k": {"ours": [], "p2rank": []}}
for (src, pid), (center, ligs) in predictions.items():
    dcc, dca = dcc_dca(center, ligs)
    if dcc is not None:
        results[src]["ours"].append((dcc, dca))
    p2c = p2rank_centers.get(pid)
    if p2c is not None:
        dcc2, dca2 = dcc_dca(p2c, ligs)
        if dcc2 is not None:
            results[src]["p2rank"].append((dcc2, dca2))

THRESH = 4.0
print(f"{'benchmark':<10}{'method':<10}{'n':>6}{'DCC succ':>10}{'DCA succ':>10}")
print("-"*48)
summary = {}
for src in ("coach420","holo4k"):
    for method in ("ours","p2rank"):
        vals = results[src][method]
        if not vals:
            print(f"{src:<10}{method:<10}{0:>6}{'--':>10}{'--':>10}"); continue
        dcc_succ = np.mean([1.0 if d<THRESH else 0.0 for d,_ in vals])
        dca_succ = np.mean([1.0 if a<THRESH else 0.0 for _,a in vals])
        summary[f"{src}_{method}"] = {"n": len(vals), "dcc": float(dcc_succ), "dca": float(dca_succ)}
        print(f"{src:<10}{method:<10}{len(vals):>6}{dcc_succ:>10.3f}{dca_succ:>10.3f}")

print("\n--- literature reference (published, FULL benchmark, different training data) ---")
LIT = {
    "coach420_fpocket": (0.228, 0.444), "coach420_p2rank_lit": (0.366, 0.628), "coach420_equipocket": (0.423, 0.656),
    "holo4k_fpocket": (0.192, 0.457), "holo4k_p2rank_lit": (0.314, 0.621), "holo4k_equipocket": (0.337, 0.662),
}
for k,(dcc,dca) in LIT.items():
    print(f"  {k:<26} DCC {dcc:.3f}  DCA {dca:.3f}")

json.dump({
    "leakage": {"exact_overlap": len(overlap_exact), "homology_flagged": len(leaked_keys),
               "clean_units": len(clean_units), "total_units": len(eval_units)},
    "our_results": summary,
    "literature_reference": LIT,
    "featurized": len(built),
}, open(WORK/"phase12_summary.json", "w"), indent=2)
print("\nsaved -> phase12_summary.json")
''')

md(r'''
---

## Reading this

Three numbers to weigh together, not one:

1. **Our DCC/DCA vs P2Rank, same-run, same clean subset** — the only fully
   controlled comparison in this notebook. If we beat P2Rank here, it confirms
   the Phase 11 result on an external, standard benchmark rather than our own
   held-out split.
2. **Our DCC/DCA vs the literature's P2Rank/EquiPocket numbers** — informative
   but not controlled: different N (our clean subset vs their full benchmark),
   different training data, and their numbers are not something we reran
   ourselves.
3. **The leakage count itself** — if a large fraction of COACH420/HOLO4K turned
   out to be exact-ID or homologous to our training set, that is worth reporting
   on its own: it says something about how much of a "standard benchmark"
   already overlaps a broad RCSB-sampled training set like ours.

If our numbers on the clean subset are close to or above EquiPocket's on the
full benchmark, that is a genuinely strong, reportable claim — made carefully,
with its own limitations stated rather than glossed over. If they fall short,
that is the honest state of the project relative to 2023–2025 SOTA, and is worth
knowing before, not after, presenting this as competitive.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase12_sota_benchmark.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
