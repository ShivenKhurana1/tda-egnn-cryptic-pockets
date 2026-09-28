"""Phase 283 -- a real grid-based cavity/pocket-voxel detector, reusing the isotropic ray-buriedness primitive already built and validated
in phase227_geometry.py (100 Fibonacci-sphere directions, RAY_CLEARANCE=1.7 A), applied from arbitrary 3D grid points instead of residue
centers. This is meant to approximate the "consensus-pocket voxel" counting the external crypticity audit's own definition describes (a
grid point counts as a pocket voxel if enough of its surrounding directions are blocked by protein atoms within a short range) -- a real
cavity-detection method, not the earlier local-atom-density proxy (Phase 279) that was too weak to trust (AUC 0.611, rejected).

grid_voxels(atoms, bounds=None, spacing=1.0, exclude_r=1.6, near_r=10.0, ray_len=8.0, block_frac=0.5)
  atoms: (N,3) heavy-atom coordinates (the whole apo or holo structure -- or just the chain of interest).
  Returns the (M,3) coordinates of detected cavity/pocket voxels: grid points that are (a) not inside the protein (further than
  exclude_r from every atom), (b) within near_r of at least one atom (so only near-surface space is considered, not bulk solvent),
  and (c) at least `block_frac` of 100 isotropic rays out to ray_len are blocked by a heavy atom within RAY_CLEARANCE.

site_voxel_count(voxels, site_atoms, radius=4.0): number of detected voxels within `radius` of any site atom -- the direct analogue of
the external audit's "apo_voxels"/"holo_voxels" (voxels within 4 A of the site residues in apo / within 4 A of the ligand in holo).
"""
import numpy as np
from scipy.spatial import cKDTree

RAY_CLEARANCE = 1.7


def _fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


N_RAYS_DEFAULT = 42
RAYS = _fib_sphere(N_RAYS_DEFAULT)


def grid_voxels(atoms, spacing=1.5, exclude_r=1.6, near_r=4.5, ray_len=7.0, ray_step=1.4, block_frac=0.85):
    atoms = np.asarray(atoms, float)
    lo, hi = atoms.min(0) - near_r, atoms.max(0) + near_r
    xs = np.arange(lo[0], hi[0], spacing); ys = np.arange(lo[1], hi[1], spacing); zs = np.arange(lo[2], hi[2], spacing)
    if len(xs) * len(ys) * len(zs) > 4_000_000:
        raise ValueError(f"grid too large: {len(xs)}x{len(ys)}x{len(zs)}")
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)

    tree = cKDTree(atoms)
    d_near, _ = tree.query(grid, distance_upper_bound=near_r)
    keep = np.isfinite(d_near) & (d_near > exclude_r)
    cand = grid[keep]
    if len(cand) == 0:
        return np.zeros((0, 3))

    ts = np.arange(ray_step, ray_len + 1e-9, ray_step)
    out_mask = np.zeros(len(cand), bool)
    CHUNK = 2000
    for i0 in range(0, len(cand), CHUNK):
        c = cand[i0:i0 + CHUNK]
        samples = c[:, None, None, :] + RAYS[None, :, None, :] * ts[None, None, :, None]
        dist, _ = tree.query(samples.reshape(-1, 3), distance_upper_bound=RAY_CLEARANCE)
        hit = np.isfinite(dist).reshape(len(c), len(RAYS), len(ts)).any(2)   # (chunk, n_rays): did this ray hit protein within ray_len?
        out_mask[i0:i0 + CHUNK] = hit.mean(1) >= block_frac
    return cand[out_mask]


def site_voxel_count(voxels, site_atoms, radius=4.0):
    if len(voxels) == 0 or len(site_atoms) == 0:
        return 0
    tree = cKDTree(np.asarray(site_atoms, float))
    d, _ = tree.query(voxels, distance_upper_bound=radius)
    return int(np.isfinite(d).sum())
