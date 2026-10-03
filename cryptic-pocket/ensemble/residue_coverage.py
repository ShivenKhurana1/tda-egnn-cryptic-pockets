"""Per-residue availability of OTHER deposited unliganded X-ray structures for each CryptoBench test chain.
For chain C (apo chain scored by the model): take up to MAXE other X-ray entries of the same UniProt (best resolution first; never CryptoBench's own
apo/holo entries), pick the polymer chain that best aligns to C's sequence, map residues through the alignment (>=90% identity), and mark a residue
'unliganded' in that entry if no heavy atom of any non-water, non-additive HETATM lies within 4.5 A of it. Cached per chain in cache/<key>.npz
(entry ids, CA coordinates NaN where unmapped, ligand flags) so the same work serves the later feature build."""
import gzip, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from Bio.Align import PairwiseAligner
sys.path.insert(0, ".")
from count_entries import search_sorted
HERE = Path(__file__).resolve().parent; RAW = HERE / "raw"; CACHE = HERE / "cache"; RAW.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
MAXE, LIG_R = 30, 4.5
AA = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
NOT_LIG = {"HOH","DOD","WAT","SO4","PO4","GOL","EDO","PEG","PG4","PGE","1PE","MPD","TRS","ACT","ACY","FMT","CIT","MES","EPE","DMS","IMD","BME","NO3","AZI","CO3","IOD","BR","CL","F","NA","K","CS","RB","LI","UNX","UNL","UNK","P6G","PGO","NH4","MG","CA","ZN"}
DATA = json.load(open("../site/data.json"))["records"]; COORDS = json.load(open("../site/coords.json"))
DS = json.load(open("../cryptobench_dataset.json"))

def fetch(pid):
    p = RAW / f"{pid}.cif.gz"
    if p.exists() and p.stat().st_size > 500: return p
    for t in range(3):
        try:
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid.upper()}.cif.gz", timeout=90) as r: p.write_bytes(r.read()); return p
        except Exception: time.sleep(2)
    return None

def parse(path):
    """-> chains: {chain: list of (resnum, ins, aa1, CAxyz, heavy_xyz array)}, lig_xyz"""
    txt = gzip.open(path, "rt").read().splitlines()
    cols = [l.split(".")[1].strip() for l in txt if l.startswith("_atom_site.")]; ix = {c: i for i, c in enumerate(cols)}
    g, an, alt, comp, ach, seq, ins, mdl, sym = (ix[k] for k in ("group_PDB","label_atom_id","label_alt_id","label_comp_id","auth_asym_id","auth_seq_id","pdbx_PDB_ins_code","pdbx_PDB_model_num","type_symbol"))
    X, Y, Z = ix["Cartn_x"], ix["Cartn_y"], ix["Cartn_z"]
    res = {}; order = {}; lig = []
    for l in txt:
        if not l.startswith(("ATOM", "HETATM")): continue
        f = l.split()
        if len(f) <= max(ix.values()) or f[mdl] != "1" or f[sym] in ("H", "D"): continue
        if f[alt] not in (".", "?", "A"): continue
        c = f[comp].upper(); xyz = (float(f[X]), float(f[Y]), float(f[Z]))
        if f[g] == "ATOM" or (f[g] == "HETATM" and c in AA):
            key = (f[ach], f[seq], f[ins])
            if key not in res: res[key] = dict(aa=AA.get(c, "X"), ca=None, atoms=[]); order.setdefault(f[ach], []).append(key)
            res[key]["atoms"].append(xyz)
            if f[an] == "CA": res[key]["ca"] = xyz
        elif c not in NOT_LIG: lig.append(xyz)
    chains = {ch: [res[k] for k in ks if res[k]["ca"] is not None] for ch, ks in order.items()}
    return chains, np.array(lig, float).reshape(-1, 3)

def process(args):
    key, target_seq, entries = args
    out = HERE / "cache" / f"{key}.npz"
    al = PairwiseAligner(); al.mode = "local"; al.match_score = 2; al.mismatch_score = -1; al.open_gap_score = -6; al.extend_gap_score = -1
    n = len(target_seq); ids, CA, LG = [], [], []
    for pid in entries:
        p = RAW / f"{pid}.cif.gz"
        if not p.exists(): continue
        try: chains, lig = parse(p)
        except Exception: continue
        best = None
        for ch, rs in chains.items():
            if len(rs) < 30: continue
            q = "".join(r["aa"] for r in rs); a = al.align(target_seq, q)[0]
            blocks = list(zip(a.aligned[0], a.aligned[1])); ali = sum(int(tp[1] - tp[0]) for tp, qp in blocks)
            if ali < 30: continue
            ident = sum(target_seq[i] == q[j] for (t0, t1), (q0, q1) in blocks for i, j in zip(range(t0, t1), range(q0, q1))) / ali
            if ident >= 0.9 and (best is None or ali > best[0]): best = (ali, ident, rs, blocks)
        if best is None: continue
        _, _, rs, blocks = best
        ca = np.full((n, 3), np.nan); lg = np.zeros(n, bool)
        tree = cKDTree(lig) if len(lig) else None
        for (t0, t1), (q0, q1) in blocks:
            for i, j in zip(range(t0, t1), range(q0, q1)):
                ca[i] = rs[j]["ca"]
                if tree is not None and tree.query(np.array(rs[j]["atoms"]), distance_upper_bound=LIG_R)[0].min() <= LIG_R: lg[i] = True
        ids.append(pid); CA.append(ca); LG.append(lg)
    np.savez_compressed(out, ids=np.array(ids), ca=np.array(CA), lig=np.array(LG))
    return key, len(ids)

if __name__ == "__main__":
    SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
    test = [r for r in DATA if r["split"] == SPLIT]
    jobs, need = [], set()
    for r in test:
        k = r["key"]; ent = DS[r["pdb"]]; main = [e for e in ent if e.get("is_main_holo_structure")][0]; acc = main["uniprot_id"]
        own = {r["pdb"].lower()} | {e["holo_pdb_id"].lower() for e in ent if e["uniprot_id"] == acc}
        ids = [i for i in (search_sorted(acc) or []) if i not in own][:MAXE]
        jobs.append((k, COORDS[k]["aa"], ids)); need |= set(ids)
    print(f"{len(test)} test chains, {len(need)} distinct entries to fetch", flush=True)
    with ThreadPoolExecutor(16) as ex: list(ex.map(fetch, sorted(need)))
    print("fetched; processing", flush=True)
    with Pool(8) as pool: res = pool.map(process, jobs, chunksize=2)
    print("done", sum(1 for _, n in res if n > 0), "chains with >=1 mapped entry")
