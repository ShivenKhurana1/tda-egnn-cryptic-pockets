"""Voxel-definition variants against the external benchmark's labels/counts, same pairs as calibrate_voxels.py (fpocket + P2Rank outputs
already in /tmp/vox). Variants: fp_all (every fpocket alpha-sphere voxel), fp_top3 / fp_top5 (only fpocket's highest-ranked pockets),
p2r (P2Rank pocket points, rank > 0), cons2 / cons3 (fpocket voxels within 2 / 3 A of a P2Rank pocket point = a two-detector consensus).
Each variant: Spearman vs their apo/holo counts, cryptic-vs-preformed AUC of the apo count, best single-threshold accuracy.
usage: python3 calibrate_v2.py [pairs.json] (default: the calibration sample)"""
import gzip, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from Bio.PDB import PDBParser
W = Path("/tmp/vox")

def spheres(pdb, top=None):
    out = pdb.with_name(pdb.stem + "_out"); S = []
    files = sorted((out / "pockets").glob("pocket*_vert.pqr"), key=lambda f: int(f.name[6:].split("_")[0]))
    for f in (files[:top] if top else files):
        for l in open(f):
            if l.startswith("ATOM"): S.append((float(l[30:38]), float(l[38:46]), float(l[46:54]), float(l[66:71])))
    return np.array(S)

def grid_in(S):
    if not len(S): return np.zeros((0, 3))
    lo = np.floor(S[:, :3].min(0) - S[:, 3].max()); hi = np.ceil(S[:, :3].max(0) + S[:, 3].max())
    g = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    t = cKDTree(S[:, :3]); keep = np.zeros(len(g), bool)
    for j, idx in enumerate(t.query_ball_point(g, S[:, 3].max())):
        if idx: keep[j] = np.any(np.linalg.norm(S[idx, :3] - g[j], axis=1) <= S[idx, 3])
    return g[keep]

def p2r_points(pdb):
    f = W / "p2r" / "visualizations" / "data" / f"{pdb.name}_points.pdb.gz"; P = []
    for l in gzip.open(f, "rt"):
        if l.startswith("HETATM") and int(l[22:26]) > 0: P.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return np.array(P).reshape(-1, 3)

def variants(pdb):
    allv = grid_in(spheres(pdb)); P = p2r_points(pdb)
    v = dict(fp_all=allv, fp_top3=grid_in(spheres(pdb, 3)), fp_top5=grid_in(spheres(pdb, 5)), p2r=P)
    for r in (2, 3):
        v[f"cons{r}"] = allv[cKDTree(P).query(allv)[0] <= r] if len(P) and len(allv) else np.zeros((0, 3))
    return v

def count(V, ref_xyz): return int((cKDTree(ref_xyz).query(V)[0] <= 4).sum()) if len(V) else 0

if __name__ == "__main__":
    rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "calibrate_voxels.json"))
    B = {(r["apo"], r["label"], r["apo_voxels"]): r for r in json.load(open("/tmp/csb/benchmark/cryptic_benchmark.json"))}
    par = PDBParser(QUIET=True); res = []
    for r in rows:
        src = B[(r["apo"], r["label"], r["their_apo"])]
        tag = f"{src['apo']}_{src['apo_chain']}__{src['holo']}_{src['holo_chain']}_{src['ligand']}"
        fa, fh = W / f"{tag}_apo.pdb", W / f"{tag}_holo.pdb"
        site = {int(x) for x in src["site_residues"]}
        sa = par.get_structure("a", fa); site_xyz = np.array([a.coord for res_ in sa.get_residues() if res_.id[1] in site for a in res_])
        # ligand atoms come from the original holo entry (the written holo file has protein only)
        import gzip as gz
        from Bio.PDB import MMCIFParser
        with gz.open(W / "raw" / f"{src['holo'].lower()}.cif.gz", "rt") as fh_: sh = MMCIFParser(QUIET=True).get_structure("h", fh_)
        mh = next(sh.get_models()); ch = mh[src["holo_chain"]]; th = cKDTree(np.array([a.coord for a in ch.get_atoms()]))
        lig = [x for c_ in mh for x in c_ if x.id[0].startswith("H_") and x.get_resname() == src["ligand"]]
        lig = min(lig, key=lambda L: th.query(np.array([a.coord for a in L]))[0].min()); lig_xyz = np.array([a.coord for a in lig if a.element != "H"])
        va, vh = variants(fa), variants(fh)
        res.append(dict(label=r["label"], their_apo=r["their_apo"], their_holo=r["their_holo"],
                        **{f"apo_{k}": count(v, site_xyz) for k, v in va.items()}, **{f"holo_{k}": count(v, lig_xyz) for k, v in vh.items()}))
    y = np.array([r["label"] == "cryptic" for r in res]); ta = np.array([r["their_apo"] for r in res]); th_ = np.array([r["their_holo"] for r in res])
    print(f"{len(res)} pairs ({y.sum()} cryptic)")
    print(f"{'variant':8s} {'rho apo':>8s} {'rho holo':>9s} {'AUC':>6s} {'best acc':>9s} {'thr':>6s}")
    for k in ("fp_all", "fp_top3", "fp_top5", "p2r", "cons2", "cons3"):
        a = np.array([r[f"apo_{k}"] for r in res]); h = np.array([r[f"holo_{k}"] for r in res])
        acc, t = max((((a <= t) == y).mean(), t) for t in np.unique(a))
        print(f"{k:8s} {spearmanr(ta, a).correlation:8.3f} {spearmanr(th_, h).correlation:9.3f} {roc_auc_score(y, -a):6.3f} {acc:9.3f} {t:6d}")
    json.dump(res, open(Path(__file__).resolve().parent / "calibrate_v2.json", "w"), indent=1)
