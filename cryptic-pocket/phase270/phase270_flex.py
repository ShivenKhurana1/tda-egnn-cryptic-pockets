"""Phase 270 -- protein-movement (elastic-network) features from C-alpha coordinates alone.

Earlier flexibility tests in this project (Phases 130-160 era) were single columns (one ANM mean-square-fluctuation value, or a B-factor) or a static-vs-perturbed
topology block, all trained from scratch at 35M. This block is richer and aimed at the actual cryptic-pocket question -- "could this spot open up?" -- and is meant to be
tested in the strong (pretrained 3B) setup. Everything is computed from the (centred) C-alpha coordinates, so it is identical for pool chains and CryptoBench chains and is
rotation/translation invariant.

Per residue (NAMES below):
  gnm_msf_rank      within-chain percentile of the GNM (10 A cutoff) mean-square fluctuation over the slowest K modes           (log/rank scaling, the untested fix flagged in the paper)
  anm_msf_rank      within-chain percentile of the ANM (15 A cutoff) mean-square fluctuation over the slowest K modes
  slow_share        fraction of the residue's ANM fluctuation carried by the 5 slowest non-trivial modes (how collective its motion is)
  msf_nbr_rank      neighbour-averaged (10 A) anm_msf_rank
  hinge_dist        distance (A/10, clipped at 2) to the nearest 'hinge' residue (smallest |amplitude| of the slowest GNM mode)
  open_max          largest loosening of the residue's 10 A CA neighbourhood (drop in neighbour count) over +/- displacement along each of the 5 slowest ANM modes (chain RMSD 1.5 A)
  close_max         largest tightening of the same neighbourhood
  nbr_std           standard deviation of the neighbour count across the 10 displaced conformations
  strain            mean |change| of pairwise distances between the residue and its 10 A neighbours, averaged over the displaced conformations
  open_mean         mean loosening over the 10 displaced conformations (signed: positive = opens on average)
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

NAMES = ["gnm_msf_rank", "anm_msf_rank", "slow_share", "msf_nbr_rank", "hinge_dist", "open_max", "close_max", "nbr_std", "strain", "open_mean"]
FLEX_DIM = len(NAMES)
K_MODES, N_SLOW, TARGET_RMSD, NBR_R = 40, 5, 1.5, 10.0
GNM_CUT, ANM_CUT = 10.0, 15.0


def _rank(v):
    r = np.argsort(np.argsort(v)); return (r + 0.5) / len(v)


def _lowest_modes(H, k):
    """k lowest non-trivial eigenpairs of a PSD matrix with a known null space (translation/rotation)."""
    n = H.shape[0]; k = min(k, n - 7)
    if n <= 900:
        w, v = np.linalg.eigh(H.toarray() if hasattr(H, "toarray") else H)
        return w, v
    w, v = eigsh(H, k=k + 6, sigma=-0.05, which="LM"); o = np.argsort(w); return w[o], v[:, o]


def _gnm(pos):
    n = len(pos); tree = cKDTree(pos); pr = tree.query_pairs(GNM_CUT, output_type="ndarray")
    i, j = pr[:, 0], pr[:, 1]
    G = coo_matrix((-np.ones(len(i)), (i, j)), shape=(n, n)); G = G + G.T
    G = G.tocsr(); deg = -np.asarray(G.sum(1)).ravel()
    from scipy.sparse import diags
    G = (G + diags(deg)).tocsc()
    w, v = _lowest_modes(G, K_MODES)
    keep = np.where(w > 1e-6)[0][:K_MODES]
    msf = (v[:, keep] ** 2 / w[keep]).sum(1)
    return msf, v[:, keep[0]]


def _anm(pos):
    n = len(pos); tree = cKDTree(pos); pr = tree.query_pairs(ANM_CUT, output_type="ndarray")
    i, j = pr[:, 0], pr[:, 1]; d = pos[j] - pos[i]; d2 = (d ** 2).sum(1); sup = d[:, :, None] * d[:, None, :] / d2[:, None, None]      # (m,3,3)
    rows, cols, vals = [], [], []
    for a in range(3):
        for b in range(3):
            s = sup[:, a, b]                                                # 3x3 super-element block between residues i and j
            rows += [3 * i + a, 3 * j + b, 3 * i + a, 3 * j + a]; cols += [3 * j + b, 3 * i + a, 3 * i + b, 3 * j + b]; vals += [-s, -s, s, s]
    H = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * n, 3 * n)).tocsc()
    w, v = _lowest_modes(H, K_MODES)
    keep = np.where(w > 1e-6)[0][:K_MODES]
    return w[keep], v[:, keep]


def flex_features(pos):
    pos = np.asarray(pos, np.float64); n = len(pos)
    out = np.zeros((n, FLEX_DIM), np.float32)
    if n < 20: return out
    gmsf, u1 = _gnm(pos)
    w, V = _anm(pos)                                                        # V: (3n, K)
    mode_msf = (V ** 2 / w).reshape(n, 3, -1).sum(1)                        # (n, K) per-residue contribution of each mode
    amsf = mode_msf.sum(1)
    tree = cKDTree(pos); nbr = tree.query_ball_point(pos, NBR_R)
    out[:, 0] = _rank(np.log(gmsf + 1e-12)); out[:, 1] = _rank(np.log(amsf + 1e-12))
    out[:, 2] = mode_msf[:, :N_SLOW].sum(1) / (amsf + 1e-12)
    out[:, 3] = _rank(np.array([out[nb, 1].mean() for nb in nbr]))
    hinge = np.argsort(np.abs(u1))[: max(3, n // 10)]; dh, _ = cKDTree(pos[hinge]).query(pos)
    out[:, 4] = np.clip(dh / 10.0, 0, 2)
    # displaced conformations along the slowest modes
    base_cnt = np.array([len(nb) - 1 for nb in nbr], float)
    d0 = [np.linalg.norm(pos[nb] - pos[i], axis=1) for i, nb in enumerate(nbr)]
    cnts, strains = [], []
    for m in range(min(N_SLOW, V.shape[1])):
        disp = V[:, m].reshape(n, 3); rms = np.sqrt((disp ** 2).sum(1).mean()); disp = disp * (TARGET_RMSD / rms)
        for sgn in (1.0, -1.0):
            p2 = pos + sgn * disp
            c = np.empty(n); s = np.empty(n)
            t2 = cKDTree(p2)
            for i, nb in enumerate(nbr):                                        # neighbourhood of the ORIGINAL structure
                dnew = np.linalg.norm(p2[nb] - p2[i], axis=1)
                c[i] = len(t2.query_ball_point(p2[i], NBR_R)) - 1
                s[i] = np.abs(dnew - d0[i]).mean()
            cnts.append(c); strains.append(s)
    cnts = np.array(cnts); loosen = base_cnt[None, :] - cnts                    # positive = fewer neighbours = opening
    out[:, 5] = loosen.max(0); out[:, 6] = (-loosen).max(0); out[:, 7] = cnts.std(0); out[:, 8] = np.mean(strains, 0); out[:, 9] = loosen.mean(0)
    return out
