"""Phase 292 motion targets: how far each apo residue actually moves in its CryptoBench main holo partner.

For every CryptoBench apo entry (main pairing, single-chain apo_chain as Phase 48 used it):
  1. parse apo and holo chains with Phase 48's residue rule (standard amino acid or MSE, has CA; cif first, then pdb);
  2. map residues by global sequence alignment (identical aligned residues only);
  3. superpose holo onto apo on a structural CORE: iterative Kabsch on CA, keeping residues that deviate < 3 A
     (at least half the mapped residues), so a moving pocket does not drag the frame;
  4. target per apo residue = heavy-atom RMSD over atom names present in both residues (backbone + side chain).
Output: motion_targets.json  {apo_id: {"chain": c, "resnums": [...], "rmsd": [... or null], "n_core": k, "n_map": m}}
keyed by Phase 48 resnum strings (number + insertion code), so the training kernel maps them with its own parser.
Targets are used ONLY as a training-time auxiliary loss; test/val targets are kept for diagnostics only.

usage: python3 compute_motion.py
"""
import gzip, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio import Align

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"; RAW.mkdir(exist_ok=True)
DS = json.load(open(HERE.parent / "cryptobench_dataset.json"))
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}


def fetch(pid):
    for ext in ("cif", "pdb"):
        p = RAW / f"{pid}.{ext}.gz"
        if p.exists() and p.stat().st_size > 500: return pid, p
    for ext in ("cif", "pdb"):
        for attempt in range(3):
            try:
                with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.{ext}.gz", timeout=60) as r: b = r.read()
                if len(b) > 500: p = RAW / f"{pid}.{ext}.gz"; p.write_bytes(b); return pid, p
                break
            except Exception: time.sleep(2 * (attempt + 1))
    return pid, None


def residues(path, pid, cid):
    parser = MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True, PERMISSIVE=True)
    with gzip.open(path, "rt") as fh: model = next(parser.get_structure(pid, fh).get_models())
    return [r for r in model[cid] if r.get_resname().strip().upper() in AA3 and "CA" in r]


def kabsch(P, Q):
    """R, t minimising |(Q @ R + t) - P|."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (Q - qc).T @ (P - pc); U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); D = np.diag([1, 1, d])
    R = U @ D @ Vt
    return R, pc - qc @ R


def process(args):
    apo_id, apo_path, holo_id, holo_path, achain, hchain = args
    try:
        ra = residues(apo_path, apo_id, achain); rh = residues(holo_path, holo_id, hchain)
    except Exception as e:
        return apo_id, dict(error=f"parse:{type(e).__name__}")
    sa = "".join(AA3[r.get_resname().strip().upper()] for r in ra); sh = "".join(AA3[r.get_resname().strip().upper()] for r in rh)
    al = Align.PairwiseAligner(); al.mode = "global"; al.match_score = 2; al.mismatch_score = -1
    al.open_gap_score = -6; al.extend_gap_score = -1
    aln = al.align(sa, sh)[0]
    pairs = []
    for (a0, a1), (h0, h1) in zip(*aln.aligned):
        for i, j in zip(range(a0, a1), range(h0, h1)):
            if sa[i] == sh[j]: pairs.append((i, j))
    if len(pairs) < 30:
        return apo_id, dict(error=f"few_mapped:{len(pairs)}")
    ia = np.array([p[0] for p in pairs]); ih = np.array([p[1] for p in pairs])
    A = np.array([ra[i]["CA"].coord for i in ia], float); B = np.array([rh[j]["CA"].coord for j in ih], float)
    keep = np.ones(len(ia), bool)
    for _ in range(10):
        R, t = kabsch(A[keep], B[keep]); dev = np.linalg.norm(B @ R + t - A, axis=1)
        new = dev < 3.0
        if new.sum() < len(ia) // 2: new = dev <= np.quantile(dev, 0.5)
        if np.array_equal(new, keep): break
        keep = new
    R, t = kabsch(A[keep], B[keep])
    rmsd = [None] * len(ra)
    for i, j in pairs:
        aa = {a.get_id(): a.coord for a in ra[i].get_unpacked_list() if a.element not in ("H", "D")}
        hh = {a.get_id(): a.coord for a in rh[j].get_unpacked_list() if a.element not in ("H", "D")}
        common = sorted(set(aa) & set(hh))
        if not common: continue
        X = np.array([aa[n] for n in common], float); Y = np.array([hh[n] for n in common], float) @ R + t
        rmsd[i] = round(float(np.sqrt(((X - Y) ** 2).sum(1).mean())), 3)
    return apo_id, dict(chain=achain, resnums=[str(r.id[1]) + r.id[2].strip() for r in ra], rmsd=rmsd,
                        n_core=int(keep.sum()), n_map=len(pairs), seq_id=len(pairs) / max(1, min(len(sa), len(sh))))


if __name__ == "__main__":
    jobs = []
    for apo_id, ents in DS.items():
        m = next(e for e in ents if e.get("is_main_holo_structure"))
        if "-" in m["apo_chain"] or "-" in m["holo_chain"]: continue
        jobs.append((apo_id, m["holo_pdb_id"].lower(), m["apo_chain"], m["holo_chain"]))
    need = sorted({j[0] for j in jobs} | {j[1] for j in jobs})
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex: paths = {p: f for p, f in ex.map(fetch, need) if f}
    print(f"fetched {len(paths)}/{len(need)} structures in {(time.time()-t0)/60:.1f} min", flush=True)
    args = [(a, paths[a], h, paths[h], ac, hc) for a, h, ac, hc in jobs if a in paths and h in paths]
    with ProcessPoolExecutor(max_workers=8) as ex: res = dict(ex.map(process, args, chunksize=8))
    ok = {k: v for k, v in res.items() if "error" not in v}
    err = {k: v["error"] for k, v in res.items() if "error" in v}
    print(f"targets for {len(ok)}/{len(jobs)} apo entries; errors {len(err)}: {list(err.items())[:8]}")
    json.dump(ok, open(HERE / "motion_targets.json", "w"), separators=(",", ":"))
