"""Can we reproduce the external benchmark's crypticity rule? For a stratified sample of its labelled apo/holo pairs, run fpocket on the
apo chain and on the holo chain, call every 1 A grid point inside any pocket alpha sphere a voxel, and count voxels within 4 A of the site
residues (apo) / of the ligand (holo). Compare with the benchmark's published voxel counts (Spearman) and labels (cryptic: apo <= 30 and
holo >= 30; pre-formed: apo >= 60 and holo >= 30).
usage: python3 calibrate_voxels.py [n_per_class]"""
import gzip, json, subprocess, sys, shutil, urllib.request, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from Bio.PDB import MMCIFParser, PDBIO, Select
FP = "/tmp/fpocket/bin/fpocket"; W = Path("/tmp/vox"); RAW = W / "raw"; RAW.mkdir(parents=True, exist_ok=True)
B = json.load(open("/tmp/csb/benchmark/cryptic_benchmark.json"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rng = np.random.RandomState(0)
cry = [r for r in B if r["label"] == "cryptic"]; pre = [r for r in B if r["label"] == "preformed"]
rows = [cry[i] for i in rng.permutation(len(cry))[:N]] + [pre[i] for i in rng.permutation(len(pre))[:N]]

def fetch(pid):
    p = RAW / f"{pid.lower()}.cif.gz"
    if p.exists() and p.stat().st_size > 500: return p
    for t in range(3):
        try:
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid.upper()}.cif.gz", timeout=90) as r: p.write_bytes(r.read()); return p
        except Exception: time.sleep(2)
    return None

def voxels(pdb_path):
    """fpocket alpha spheres of all pockets -> 1 A grid points inside any sphere."""
    out = pdb_path.with_name(pdb_path.stem + "_out")
    if not out.exists(): subprocess.run([FP, "-f", str(pdb_path)], capture_output=True)
    sph = []
    for f in sorted((out / "pockets").glob("pocket*_vert.pqr")):
        for l in open(f):
            if l.startswith("ATOM"):
                sph.append((float(l[30:38]), float(l[38:46]), float(l[46:54]), float(l[66:71])))
    if not sph: return np.zeros((0, 3))
    S = np.array(sph); lo = np.floor(S[:, :3].min(0) - S[:, 3].max()); hi = np.ceil(S[:, :3].max(0) + S[:, 3].max())
    g = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    t = cKDTree(S[:, :3]); keep = np.zeros(len(g), bool)
    for j, idx in enumerate(t.query_ball_point(g, S[:, 3].max())):
        if idx: keep[j] = np.any(np.linalg.norm(S[idx, :3] - g[j], axis=1) <= S[idx, 3])
    return g[keep]

def one(r):
    try:
        pa, ph = fetch(r["apo"]), fetch(r["holo"])
        if pa is None or ph is None: return None
        par = MMCIFParser(QUIET=True)
        with gzip.open(pa, "rt") as fh: sa = par.get_structure("a", fh)
        with gzip.open(ph, "rt") as fh: sh = par.get_structure("h", fh)
        ma, mh = next(sa.get_models()), next(sh.get_models())
        ca, ch = ma[r["apo_chain"]], mh[r["holo_chain"]]
        tag = f"{r['apo']}_{r['apo_chain']}__{r['holo']}_{r['holo_chain']}_{r['ligand']}"
        fa, fh_ = W / f"{tag}_apo.pdb", W / f"{tag}_holo.pdb"
        class SelC(Select):
            def __init__(s, m, c): s.m, s.c = m, c
            def accept_model(s, mo): return mo is s.m
            def accept_chain(s, c_): return c_ is s.c
            def accept_residue(s, res): return res.id[0] == " "          # protein residues only (fpocket input)
        io = PDBIO(); io.set_structure(sa); io.save(str(fa), SelC(ma, ca))
        io = PDBIO(); io.set_structure(sh); io.save(str(fh_), SelC(mh, ch))
        site = {int(x) for x in r["site_residues"]}
        site_xyz = np.array([a.coord for res in ca if res.id[0] == " " and res.id[1] in site for a in res if a.element != "H"])
        lig = [res for c_ in mh for res in c_ if res.id[0].startswith("H_") and res.get_resname() == r["ligand"]]
        if not len(site_xyz) or not lig: return None
        # the ligand copy closest to the holo chain
        hxyz = np.array([a.coord for a in ch.get_atoms()]); th = cKDTree(hxyz)
        lig = min(lig, key=lambda L: th.query(np.array([a.coord for a in L]))[0].min())
        lig_xyz = np.array([a.coord for a in lig if a.element != "H"])
        va, vh = voxels(fa), voxels(fh_)
        na = int((cKDTree(site_xyz).query(va)[0] <= 4).sum()) if len(va) else 0
        nh = int((cKDTree(lig_xyz).query(vh)[0] <= 4).sum()) if len(vh) else 0
        return dict(apo=r["apo"], label=r["label"], their_apo=r["apo_voxels"], their_holo=r["holo_voxels"], our_apo=na, our_holo=nh)
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}"[:120], apo=r["apo"])

if __name__ == "__main__":
    with ThreadPoolExecutor(16) as ex: list(ex.map(lambda r: (fetch(r["apo"]), fetch(r["holo"])), rows))
    with ProcessPoolExecutor(8) as ex: res = list(ex.map(one, rows))
    ok = [r for r in res if r and "error" not in r]; err = [r for r in res if r and "error" in r]
    print(f"pairs scored {len(ok)}/{len(rows)}; errors {len(err)} {err[:3]}")
    ta, oa = np.array([r["their_apo"] for r in ok]), np.array([r["our_apo"] for r in ok])
    th, oh = np.array([r["their_holo"] for r in ok]), np.array([r["our_holo"] for r in ok])
    print(f"Spearman apo voxels {spearmanr(ta, oa).correlation:.3f} | holo voxels {spearmanr(th, oh).correlation:.3f}")
    lab = np.array([r["label"] == "cryptic" for r in ok])
    from sklearn.metrics import roc_auc_score
    print(f"cryptic-vs-preformed AUC of -our_apo {roc_auc_score(lab, -oa):.3f} (theirs by construction 1.0)")
    for thr in (10, 20, 30, 45, 60):
        pred = oa <= thr; print(f"  our apo <= {thr:2d}: sens {pred[lab].mean():.2f} spec {(~pred[~lab]).mean():.2f}")
    json.dump(ok, open(Path(__file__).resolve().parent / "calibrate_voxels.json", "w"), indent=1)
