"""Phase 273 -- per-residue chemistry features, from amino-acid identity + Cα positions alone (no ligand, no leakage: nothing here depends on
where anything actually binds). Rotation/translation invariant by construction (every quantity is either a per-residue lookup value or a
cKDTree radius query on relative positions).

Base per-residue chemistry (standard tabulated values; unknown/non-standard residues get the 20-residue mean for continuous properties, 0 for flags):
  kd      Kyte-Doolittle hydropathy
  vol     residue volume (Zamyatnin, A^3)
  charge  formal charge at pH 7 (D/E -1, K/R +1, H +0.1)
  arom    aromatic flag (F/W/Y = 1, H = 0.5)
  hbd     side-chain H-bond donor count (simplified standard table)
  hba     side-chain H-bond acceptor count (simplified standard table)
  small   flexibility-enabling small residue flag (G/A = 1)

Per-node output (NAMES below), the self value plus neighbourhood (10 A) aggregates of the base properties, plus one composite feature aimed
at cryptic-pocket-relevant local chemistry: the size of the nearest hydrophobic patch (connected component of KD>0 residues linked at 6 A),
taken as the max over this residue's own 10 A neighbourhood so pocket-LINING polar residues next to a hydrophobic patch also get signal.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
          L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7,
           L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
CHARGE = dict(D=-1.0, E=-1.0, K=1.0, R=1.0, H=0.1)
AROM = dict(F=1.0, W=1.0, Y=1.0, H=0.5)
HBD = dict(R=5, K=1, W=1, N=1, Q=1, H=1, S=1, T=1, Y=1, C=1)
HBA = dict(D=2, E=2, N=1, Q=1, H=1, S=1, T=1, Y=1)
SMALL = dict(G=1.0, A=1.0)
AA20 = "ACDEFGHIKLMNPQRSTVWY"
_KD_MEAN, _VOL_MEAN = float(np.mean([KD[a] for a in AA20])), float(np.mean([VOL[a] for a in AA20]))

NAMES = ["self_kd", "self_vol", "self_charge", "self_arom", "self_hbd", "self_hba", "self_small",
         "nbr_kd_mean", "nbr_kd_max", "nbr_charge_net", "nbr_charge_abs_mean", "nbr_charge_hetero",
         "nbr_arom_frac", "nbr_hbd_mean", "nbr_hba_mean", "hydrophobic_patch_size"]
CHEM_DIM = len(NAMES)
NBR_R, PATCH_LINK = 10.0, 6.0


def chem_features(seq, pos):
    n = len(seq); assert len(pos) == n
    kd = np.array([KD.get(a, _KD_MEAN) for a in seq])
    vol = np.array([VOL.get(a, _VOL_MEAN) for a in seq])
    charge = np.array([CHARGE.get(a, 0.0) for a in seq])
    arom = np.array([AROM.get(a, 0.0) for a in seq])
    hbd = np.array([HBD.get(a, 0) for a in seq], float)
    hba = np.array([HBA.get(a, 0) for a in seq], float)
    small = np.array([SMALL.get(a, 0.0) for a in seq])

    out = np.zeros((n, CHEM_DIM), np.float32)
    if n == 0: return out
    out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4], out[:, 5], out[:, 6] = kd, vol, charge, arom, hbd, hba, small
    tree = cKDTree(pos)
    nbr = tree.query_ball_point(pos, NBR_R)
    pos_charge, neg_charge = (charge > 0).astype(float), (charge < 0).astype(float)
    for i, idx in enumerate(nbr):
        idx = np.asarray(idx)
        out[i, 7] = kd[idx].mean(); out[i, 8] = kd[idx].max()
        out[i, 9] = charge[idx].mean(); out[i, 10] = np.abs(charge[idx]).mean()
        out[i, 11] = pos_charge[idx].sum() * neg_charge[idx].sum() / max(len(idx), 1)
        out[i, 12] = arom[idx].mean(); out[i, 13] = hbd[idx].mean(); out[i, 14] = hba[idx].mean()

    hydrophobic = kd > 0
    if hydrophobic.sum() > 1:
        hi = np.where(hydrophobic)[0]
        pairs = cKDTree(pos[hi]).query_pairs(PATCH_LINK, output_type="ndarray")
        m = len(hi)
        A = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m)) if len(pairs) else csr_matrix((m, m))
        n_comp, lab = connected_components(A, directed=False)
        comp_size = np.zeros(n)
        sizes = np.bincount(lab, minlength=n_comp)
        for j, node in enumerate(hi): comp_size[node] = sizes[lab[j]]
    else:
        comp_size = np.zeros(n)
    for i, idx in enumerate(nbr):
        out[i, 15] = comp_size[np.asarray(idx)].max() if len(idx) else 0.0
    return out
