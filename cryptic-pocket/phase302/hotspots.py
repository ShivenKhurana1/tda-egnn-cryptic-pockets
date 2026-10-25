"""Phase 302: pseudo-FTMap probe hotspot mapping on apo structures (cryptic sites tend to sit next to strong binding hot spots even when
closed; Vajda lab). Full FTMap (FFT docking of 16 probe molecules + minimisation) is too slow for ~935 proteins on CPU, so this keeps its
core idea, consensus clusters of several probe TYPES, with single-site probes on a grid. All parameters fixed a priori (not tuned).

Per apo chain (the Phase 300 files in /tmp/fsdb/in, residue order = model node order):
  grid 1 A, shell points 3-6 A from the nearest heavy atom; buriedness = fraction of 30 rays hitting protein within 12 A.
  probe energies (arbitrary units): soft LJ (eps 0.1, sigma = R_atom + R_probe, r clamped >= 0.8 sigma, 8 A cutoff) plus
    C    hydrophobic   -0.2 per carbon within 4.5 A, +0.3 per N/O within 3.5 A
    O    hydroxyl      -1.0 per N/O within 2.5-3.4 A (cap 3)
    N    amide donor   -1.0 per O within 2.5-3.4 A (cap 3)
    AR   aromatic      -0.2 per carbon within 4.5 A, -0.5 per aromatic ring atom within 4.5 A
  favourable points per probe: buriedness >= 0.5 and energy in that probe's lowest 5% over the chain; single-linkage clusters at 1.5 A,
  clusters >= 5 points kept. Consensus sites: probe-cluster centroids single-linked at 4 A; strength = number of probe clusters (and
  number of distinct probe types) in the site.
Per-residue features: strongest site within 4 A / 8 A of any residue atom (strength, distinct types), 1/rank of the nearest site, distance
to the nearest site with strength >= 3, per-probe minimum energy over shell points within 4 A, max buriedness within 4 A.
usage: python3 hotspots.py [one_key]     (no argument: all chains, parallel; writes hotspot_feats.npz)"""
import sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import fcluster, linkage
IN = Path("/tmp/fsdb/in"); H = Path(__file__).resolve().parent
RADII = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}
AROM = {("PHE", n) for n in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")} | {("TYR", n) for n in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")} | \
       {("TRP", n) for n in ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "CG", "CD1", "NE1")} | {("HIS", n) for n in ("CG", "ND1", "CD2", "CE1", "NE2")}
PROBES = {"C": 1.9, "O": 1.6, "N": 1.7, "AR": 2.2}
FIB = None

def fib(n=30):
    i = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]

def read(path):
    X, el, res_idx, arom = [], [], [], []; keys = []; cur = None
    for l in open(path):
        if not l.startswith(("ATOM", "HETATM")): continue
        e = (l[76:78].strip() or l[12:16].strip()[0]).upper()
        if e == "H": continue
        rk = (l[21], l[22:27]); rn = l[17:20]; an = l[12:16].strip()
        if rk != cur: keys.append(rk); cur = rk
        X.append((float(l[30:38]), float(l[38:46]), float(l[46:54]))); el.append(e if e in RADII else "C")
        res_idx.append(len(keys) - 1); arom.append((rn, an) in AROM)
    return np.array(X), np.array(el), np.array(res_idx), np.array(arom), len(keys)

def features(path):
    X, el, ri, arom, nres = read(path)
    tree = cKDTree(X)
    lo, hi = X.min(0) - 6, X.max(0) + 6
    g = np.stack(np.meshgrid(*[np.arange(a, b, 1.0) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    dn = tree.query(g)[0]; g = g[(dn >= 3.0) & (dn <= 6.0)]
    # buriedness by ray marching on an occupancy grid
    occ_lo = lo; shape = np.ceil((hi - lo) / 1.0).astype(int) + 1; occ = np.zeros(shape, bool)
    cells = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    occ.reshape(-1)[tree.query(cells * 1.0 + occ_lo)[0] <= 1.8] = True
    rays = fib(30); bur = np.zeros(len(g))
    for d in rays:
        hit = np.zeros(len(g), bool)
        for step in np.arange(1.0, 12.01, 1.0):
            idx = np.floor((g + d * step - occ_lo) / 1.0).astype(int)
            ok = np.all((idx >= 0) & (idx < shape), 1); h = np.zeros(len(g), bool); h[ok] = occ[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
            hit |= h
        bur += hit
    bur /= len(rays)
    # probe energies
    nb = tree.query_ball_point(g, 8.0); R = np.array([RADII[e] for e in el]); polar = np.isin(el, ["N", "O"]); carbon = el == "C"; oxy = el == "O"
    E = {p: np.zeros(len(g)) for p in PROBES}
    for j, idx in enumerate(nb):
        if not idx: continue
        idx = np.array(idx); r = np.linalg.norm(X[idx] - g[j], axis=1)
        for p, Rp in PROBES.items():
            s = R[idx] + Rp; rr = np.maximum(r, 0.8 * s); q = (s / rr) ** 6
            E[p][j] = (0.1 * (q * q - 2 * q)).sum()
        c45 = carbon[idx] & (r <= 4.5); p35 = polar[idx] & (r <= 3.5); hb = (r >= 2.5) & (r <= 3.4)
        E["C"][j] += -0.2 * c45.sum() + 0.3 * p35.sum()
        E["O"][j] += -1.0 * min(3, (polar[idx] & hb).sum())
        E["N"][j] += -1.0 * min(3, (oxy[idx] & hb).sum())
        E["AR"][j] += -0.2 * c45.sum() - 0.5 * (arom[idx] & (r <= 4.5)).sum()
    # probe clusters -> consensus sites
    pcl = []
    for p in PROBES:
        cand = np.where(bur >= 0.5)[0]
        if len(cand) < 5: continue
        thr = np.quantile(E[p], 0.05); sel = cand[E[p][cand] <= thr]
        if len(sel) < 5: continue
        lab = fcluster(linkage(g[sel], "single"), 1.5, "distance") if len(sel) > 1 else np.array([1])
        for c in np.unique(lab):
            m = sel[lab == c]
            if len(m) >= 5: pcl.append((p, g[m].mean(0), len(m)))
    sites = []
    if pcl:
        cen = np.array([c[1] for c in pcl])
        lab = fcluster(linkage(cen, "single"), 4.0, "distance") if len(cen) > 1 else np.array([1])
        for c in np.unique(lab):
            mem = [pcl[i] for i in np.where(lab == c)[0]]
            sites.append(dict(center=np.mean([m[1] for m in mem], 0), strength=len(mem), types=len({m[0] for m in mem}), pts=sum(m[2] for m in mem)))
        sites.sort(key=lambda s: (-s["strength"], -s["types"], -s["pts"]))
    # per-residue features
    F = np.zeros((nres, 11), np.float32); F[:, 4] = 30.0
    gt = cKDTree(g)
    for k in range(nres):
        A = X[ri == k]
        if sites:
            dmin = np.array([np.linalg.norm(A - s["center"], axis=1).min() for s in sites])
            for col, cut in ((0, 4.0), (2, 8.0)):
                w = np.where(dmin <= cut)[0]
                if len(w): F[k, col] = sites[w[0]]["strength"]; F[k, col + 1] = sites[w[0]]["types"]
            F[k, 5] = 1.0 / (1 + int(np.argmin(dmin)))
            strong = [i for i, s in enumerate(sites) if s["strength"] >= 3]
            if strong: F[k, 4] = min(30.0, dmin[strong].min())
        near = sorted({i for a in A for i in gt.query_ball_point(a, 4.0)})
        if near:
            near = np.array(near)
            for c, p in enumerate(PROBES): F[k, 6 + c] = E[p][near].min()
            F[k, 10] = bur[near].max()
    return F, len(sites)

def run(key):
    try:
        F, ns = features(IN / f"{key}.pdb"); return key, F, ns
    except Exception as e:
        return key, None, f"{type(e).__name__}: {e}"[:100]

if __name__ == "__main__":
    import json
    if len(sys.argv) > 1:
        t = time.time(); key, F, ns = run(sys.argv[1]); print(key, F.shape if F is not None else ns, "sites", ns, f"{time.time()-t:.1f}s"); sys.exit()
    INFO = json.load(open(H.parent / "phase300" / "chains.json")); keys = sorted(INFO)
    out, bad = {}, []
    with ProcessPoolExecutor(8) as ex:
        for key, F, ns in ex.map(run, keys, chunksize=2):
            if F is None: bad.append((key, ns))
            elif len(F) != len(INFO[key]["y"]): bad.append((key, f"len {len(F)} vs {len(INFO[key]['y'])}"))
            else: out[key] = F
    np.savez_compressed(H / "hotspot_feats.npz", **out)
    print(f"features for {len(out)}/{len(keys)} chains; failures {len(bad)}: {bad[:5]}")
