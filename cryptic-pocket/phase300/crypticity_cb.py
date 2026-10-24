"""Crypticity for every CryptoBench chain with the consensus voxel definition chosen in calibrate_v2.py (cons2: fpocket alpha-sphere
voxels within 2 A of a P2Rank pocket point). For each apo entry's MAIN pairing: apo count = voxels within 4 A of the site residues
(apo_pocket_selection) in the apo chain; holo count = voxels within 4 A of the ligand (resname + ligand chain + index) in the holo chain.
Thresholds on our scale are fitted on the calibration sample only (calibrate_v2.json), then checked against the external benchmark's
independent labels for the CryptoBench test split (17 cryptic / 111 pre-formed / 52 neither) before any training chain is labelled.
usage: python3 crypticity_cb.py"""
import gzip, json, subprocess, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select
H = Path(__file__).resolve().parent; CP = H.parent; RAW = CP / "phase292" / "raw"; W = Path("/tmp/cbvox"); W.mkdir(exist_ok=True)
sys.path.insert(0, str(H)); from calibrate_v2 import spheres, grid_in
FP = "/tmp/fpocket/bin/fpocket"; PRANK = str(CP / "tools" / "p2rank_2.5.1" / "prank")
DS = json.load(open(CP / "cryptobench_dataset.json")); INFO = json.load(open(H / "chains.json"))

def parse(pid):
    p = next(RAW.glob(f"{pid}.*.gz")); par = MMCIFParser(QUIET=True) if ".cif" in p.name else PDBParser(QUIET=True, PERMISSIVE=True)
    with gzip.open(p, "rt") as fh: return par.get_structure(pid, fh)

def write_protein(st, cid, dest):
    m0 = next(st.get_models()); chain = m0[cid]
    class Sel(Select):
        def accept_model(s, mo): return mo is m0
        def accept_chain(s, ch): return ch is chain
        def accept_residue(s, r): return r.id[0] == " "
    if len(cid) > 1: chain.id = next(x for x in "zyxwvutsrqponmlkjihgfedcba" if x not in {c.id for c in m0})
    io = PDBIO(); io.set_structure(st); io.save(str(dest), Sel()); chain.id = cid

def prep(key):
    pid, c = key.rsplit("_", 1); main = next(e for e in DS[pid] if e.get("is_main_holo_structure"))
    try:
        sa, sh = parse(pid), parse(main["holo_pdb_id"].lower())
        site = {s.split("_", 1)[1] for s in main["apo_pocket_selection"]}
        site_xyz = np.array([a.coord for r in next(sa.get_models())[c] if r.id[0] == " " and f"{r.id[1]}{r.id[2].strip()}" in site for a in r if a.element != "H"])
        mh = next(sh.get_models())
        lig = [r for ch in mh for r in ch if r.id[0].startswith("H_") and r.get_resname() == main["ligand"] and ch.id == main["ligand_chain"] and str(r.id[1]) == str(main["ligand_index"])]
        if not lig: lig = [r for ch in mh for r in ch if r.id[0].startswith("H_") and r.get_resname() == main["ligand"]]
        if not lig or not len(site_xyz): return key, None
        lig_xyz = np.array([a.coord for a in lig[0] if a.element != "H"])
        write_protein(sa, c, W / f"{key}_apo.pdb"); write_protein(sh, main["holo_chain"], W / f"{key}_holo.pdb")
        np.savez(W / f"{key}_ref.npz", site=site_xyz, lig=lig_xyz)
        if not (W / f"{key}_apo_out").exists(): subprocess.run([FP, "-f", str(W / f"{key}_apo.pdb")], capture_output=True)
        if not (W / f"{key}_holo_out").exists(): subprocess.run([FP, "-f", str(W / f"{key}_holo.pdb")], capture_output=True)
        return key, True
    except Exception as e:
        return key, f"{type(e).__name__}"

def counts(key):
    z = np.load(W / f"{key}_ref.npz"); out = {}
    for side, ref in (("apo", z["site"]), ("holo", z["lig"])):
        pdb = W / f"{key}_{side}.pdb"; allv = grid_in(spheres(pdb)); P = []
        for l in gzip.open(W / "p2r" / "visualizations" / "data" / f"{pdb.name}_points.pdb.gz", "rt"):
            if l.startswith("HETATM") and int(l[22:26]) > 0: P.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
        P = np.array(P).reshape(-1, 3)
        V = allv[cKDTree(P).query(allv)[0] <= 2] if len(P) and len(allv) else np.zeros((0, 3))
        out[side] = int((cKDTree(ref).query(V)[0] <= 4).sum()) if len(V) else 0
    return key, out

if __name__ == "__main__":
    keys = sorted(INFO)
    with ProcessPoolExecutor(8) as ex: st = dict(ex.map(prep, keys, chunksize=4))
    ok = [k for k, v in st.items() if v is True]; print(f"prepared {len(ok)}/{len(keys)}; failures {[(k, v) for k, v in st.items() if v is not True][:8]}", flush=True)
    (W / "list.ds").write_text("\n".join(f"{k}_{s}.pdb" for k in ok for s in ("apo", "holo")) + "\n")
    subprocess.run([PRANK, "predict", str(W / "list.ds"), "-o", str(W / "p2r"), "-threads", "8", "-visualizations", "1"], capture_output=True, cwd=str(W))
    with ProcessPoolExecutor(8) as ex: C = dict(ex.map(counts, ok, chunksize=4))
    json.dump(C, open(H / "crypticity_counts.json", "w"))
    print("counts for", len(C), "chains")
