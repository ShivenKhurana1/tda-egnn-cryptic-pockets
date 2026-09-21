"""Phase 227 -- hand-crafted per-residue geometry features, the control arm for the TDA claim.

Question: does topology add anything beyond the burial / shape / exposure descriptors a structural
biologist would compute with no TDA at all? Phase 164 already showed a 2-feature burial baseline
(heavy-atom count within 12 A + gyration) recovers most of the TDA gain, so the control here is a
much richer, deliberately strong plain-geometry set.

All features are functions of the same input the TDA block sees: heavy atoms of the standard residues
of ONE chain, first model, residue filter identical to the ingestion (resname in AA3, has CA).
Every feature is invariant to rigid motion except for Shrake-Rupley sampling noise in rel_sasa and
residue_depth (lab-frame sphere points); that noise is measured in the test, not assumed away.
"""
import gzip
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.SASA import ShrakeRupley

AA3 = {"MSE": "M", "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
# Tien et al. 2013 theoretical maximum residue ASA (A^2), used to make SASA relative
MAX_ASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223, "GLY": 104,
           "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159, "SER": 155,
           "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}
COUNT_RADII = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0)
HSE_RADIUS = 13.0
SHAPE_RADIUS = 12.0
N_RAYS = 100
RAY_LENGTHS = (10.0, 20.0)
RAY_STEP = 1.0
RAY_START = 2.0
SASA_POINTS = 200
EXPOSED_ATOM_MIN_SASA = 5.0   # A^2; an atom counts as solvent exposed above this
RAY_CLEARANCE = 1.7          # a ray sample within this distance of a heavy atom counts as blocked
NAMES = ([f"n_atoms_{int(r)}A" for r in COUNT_RADII]
         + ["rel_sasa", "hse_up", "hse_down", "shape_linear", "shape_planar", "shape_spherical",
            "offset_from_local_centroid", "ray_blocked_10A", "ray_blocked_20A", "hull_depth",
            "residue_depth"])
GEOM_DIM = len(NAMES)


def _fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


RAYS = _fib_sphere(N_RAYS)


def _local_frame(pts, c):
    """Orthonormal frame (rows = axes) from the local atom cloud; sign of each axis from the skew of the
    projections so a rigid rotation of the input rotates the frame with it."""
    if len(pts) < 4:
        return np.eye(3)
    q = pts - pts.mean(0)
    w, v = np.linalg.eigh(np.cov(q.T))
    v = v[:, ::-1].copy()
    for k in range(3):
        if (q @ v[:, k]).__pow__(3).sum() < 0:
            v[:, k] = -v[:, k]
    if np.linalg.det(v) < 0:
        v[:, 2] = -v[:, 2]
    return v


def load_chain(path, chain_id, pid="x"):
    """Same filter as the ingestion: standard residues with a CA, first model, one chain, no hetero."""
    op = gzip.open if str(path).endswith(".gz") else open
    parser = MMCIFParser(QUIET=True) if ".cif" in str(path) else PDBParser(QUIET=True, PERMISSIVE=True)
    with op(path, "rt") as fh:
        model = next(parser.get_structure(pid, fh).get_models())
    chain = model[chain_id]
    for r in list(chain):
        # same residue filter as the ingestion: resname in AA3_TO_1 (which includes selenomethionine,
        # a HETATM record) and a CA atom -- no hetero-flag test, or MSE residues would be dropped
        if r.get_resname().strip().upper() not in AA3 or "CA" not in r:
            chain.detach_child(r.id)
    for r in chain:
        for a in list(r):
            if (a.element or "").strip().upper() in ("H", "D"):
                r.detach_child(a.id)
    return chain


def geometry_for_chain(chain):
    residues = list(chain)
    n = len(residues)
    ca = np.array([r["CA"].coord for r in residues], float)
    atoms, owner = [], []
    for i, r in enumerate(residues):
        for a in r:
            atoms.append(a.coord)
            owner.append(i)
    atoms = np.asarray(atoms, float)
    tree = cKDTree(atoms)

    # SASA (atom level, then residue sum) on this chain alone
    ShrakeRupley(n_points=SASA_POINTS).compute(chain, level="A")
    atom_sasa = np.array([a.sasa for r in residues for a in r], float)
    res_sasa = np.bincount(owner, weights=atom_sasa, minlength=n)
    rel_sasa = np.array([min(res_sasa[i] / MAX_ASA["MET" if residues[i].get_resname().strip().upper() == "MSE" else residues[i].get_resname().strip().upper()], 1.5)
                         for i in range(n)])
    exposed_tree = cKDTree(atoms[atom_sasa > EXPOSED_ATOM_MIN_SASA]) if (atom_sasa > EXPOSED_ATOM_MIN_SASA).any() else None

    out = np.zeros((n, GEOM_DIM), np.float32)
    ca_tree = cKDTree(ca)
    hull = None
    try:
        hull = ConvexHull(atoms)
    except Exception:
        pass

    for i in range(n):
        c = ca[i]
        col = 0
        for r in COUNT_RADII:
            out[i, col] = len(tree.query_ball_point(c, r)); col += 1
        out[i, col] = rel_sasa[i]; col += 1

        # half-sphere exposure (CA-based): direction bisecting the two CA-CA vectors
        prev_v = ca[i - 1] - c if i > 0 else None
        next_v = ca[i + 1] - c if i < n - 1 else None
        vs = [v for v in (prev_v, next_v) if v is not None]
        u = np.sum(vs, axis=0) if vs else np.zeros(3)
        nu = np.linalg.norm(u)
        u = u / nu if nu > 1e-6 else u
        nb = [j for j in ca_tree.query_ball_point(c, HSE_RADIUS) if j != i]
        d = ca[nb] - c if nb else np.zeros((0, 3))
        proj = d @ u if len(nb) else np.zeros(0)
        out[i, col] = float((proj > 0).sum()); out[i, col + 1] = float((proj <= 0).sum()); col += 2

        # local shape of the heavy-atom cloud
        idx = tree.query_ball_point(c, SHAPE_RADIUS)
        pts = atoms[idx]
        if len(pts) >= 4:
            cen = pts.mean(0)
            ev = np.sort(np.linalg.eigvalsh(np.cov((pts - cen).T)))[::-1]
            ev = np.maximum(ev, 0.0)
            l1 = max(ev[0], 1e-9)
            out[i, col:col + 3] = [(ev[0] - ev[1]) / l1, (ev[1] - ev[2]) / l1, ev[2] / l1]
            out[i, col + 3] = float(np.linalg.norm(c - cen))
        col += 4

        # ray buriedness, with rays expressed in a canonical local frame (PCA axes of the local atom
        # cloud, signs fixed by the third moment) so the feature is rotation invariant
        R = _local_frame(pts, c)
        rays = RAYS @ R.T
        ts = np.arange(RAY_START, max(RAY_LENGTHS) + 1e-9, RAY_STEP)
        samples = c[None, None, :] + rays[:, None, :] * ts[None, :, None]
        dist, _ = tree.query(samples.reshape(-1, 3), distance_upper_bound=RAY_CLEARANCE)
        hit = np.isfinite(dist).reshape(N_RAYS, len(ts))
        for L in RAY_LENGTHS:
            out[i, col] = float(hit[:, ts <= L + 1e-9].any(1).mean()); col += 1

        # depth below the convex hull surface, and residue depth (mean distance to the 3 nearest exposed atoms)
        if hull is not None:
            out[i, col] = float(np.min(-(hull.equations[:, :3] @ c + hull.equations[:, 3])))
        col += 1
        if exposed_tree is not None:
            dd = exposed_tree.query(c, k=min(3, exposed_tree.n))[0]
            out[i, col] = float(np.mean(dd))
        col += 1
    return out, ca
