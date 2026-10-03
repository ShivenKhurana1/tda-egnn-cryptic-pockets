#!/usr/bin/env python3
"""Phase 159: CryptoBench ingestion + ANM-derived flexibility feature.
Extends Phase42's exact ingestion pipeline (same fetch/parse/featurizer/
dihedral/ESM cells, reused verbatim) with ONE new per-residue feature:
mean-square-fluctuation (MSF) predicted by an Anisotropic Network Model
(ANM) from the apo structure alone -- a physics-based, first-principles
flexibility estimate, distinct from (and cheaper than) Phase157/158's
topological-variability ensemble (no perturbed conformations or
persistent-homology recomputation needed here, just one eigendecomposition
per chain) and distinct from the earlier B-factor experiment (predicted
from structure, not an experimental/crystallographic observation)."""
import json
from pathlib import Path

nb42 = json.load(open("/Users/shivenkhurana/bioproj/phase42_cryptobench_ingestion.ipynb"))
c42 = nb42["cells"]
s42 = lambda i: (c42[i]["source"] if isinstance(c42[i]["source"], str) else "".join(c42[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 159 - CryptoBench Ingestion + ANM Flexibility Feature

Extends Phase42's exact ingestion (same fetch/parse/featurizer/dihedral/
ESM cells) with ONE new per-residue feature: Anisotropic Network Model
(ANM) predicted mean-square-fluctuation (MSF) -- a physics-based
flexibility estimate from the apo structure's low-frequency collective
modes alone. Distinct hypothesis from Phase157/158 (topological
*variability* across a perturbed ensemble): this tests whether simply
knowing *which residues are predicted to move a lot* helps, independent
of any topology recomputation. Much cheaper: one eigendecomposition per
chain, no perturbed conformations, no persistent homology recomputed.
''')

md(r'''## Cell 1 - Install dependencies (identical to Phase42)''')
code(s42(2))

md(r'''## Cell 2 - Setup, fetch CryptoBench's dataset.json + folds.json from OSF (identical to Phase42)''')
code(s42(4))

md(r'''## Cell 3 - Fetch apo structures (identical to Phase42)''')
code(s42(6))

md(r'''## Cell 4 - Parse apo chains, build per-residue crypticity labels (identical to Phase42)''')
code(s42(8))

md(r'''## Cell 5 - Featuriser (identical to Phase42 -- byte-for-byte the Phase 2b/6b/29/36 definition)''')
code(s42(10))

md(r'''## Cell 6 - Compute topology for every chain (identical to Phase42)''')
code(s42(12))

md(r'''## Cell 7 - ANM mean-square-fluctuation: one eigendecomposition per chain, no perturbation needed''')
code(r'''
N_MODES = 10
ANM_CUTOFF = 13.0


def anm_msf(ca, cutoff=ANM_CUTOFF, n_modes=N_MODES, gamma=1.0):
    """Standard ANM Hessian (same construction as Phase157), but here we
    only need the predicted fluctuation MAGNITUDE per residue, not
    displacement directions to perturb coordinates -- so this is a
    single eigendecomposition, no ensemble, no topology recomputation."""
    n = len(ca)
    if n < 4:
        return np.zeros(n)
    diff = ca[:, None, :] - ca[None, :, :]
    dist2 = (diff ** 2).sum(-1)
    np.fill_diagonal(dist2, np.inf)
    contact = dist2 < cutoff ** 2
    outer = diff[:, :, :, None] * diff[:, :, None, :]
    blocks = np.where(contact[:, :, None, None], -gamma * outer / dist2[:, :, None, None], 0.0)
    diag = -blocks.sum(axis=1)
    idxr = np.arange(n)
    blocks[idxr, idxr] = diag
    H = blocks.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    eigvals, eigvecs = np.linalg.eigh(H)
    nz = np.where(eigvals > 1e-6)[0][:n_modes]
    if len(nz) == 0:
        return np.zeros(n)
    modes = eigvecs[:, nz].T.reshape(len(nz), n, 3)   # (n_modes, n, 3)
    msf = np.zeros(n)
    for k, ev in enumerate(eigvals[nz]):
        msf += (modes[k] ** 2).sum(axis=1) / ev
    return msf


t0 = time.time()
msf_all = Parallel(n_jobs=max(1, os.cpu_count() or 2), verbose=5)(
    delayed(anm_msf)(r["ca"]) for r in records)
print(f"ANM-MSF computed for {len(msf_all):,} chains in {(time.time()-t0)/60:.1f} min")

msf_cat = np.concatenate(msf_all, axis=0)
msf_mu, msf_sd = float(msf_cat.mean()), float(msf_cat.std())
if msf_sd < 1e-6: msf_sd = 1.0
print(f"MSF stats: mean {msf_mu:.4g}  std {msf_sd:.4g}  "
      f"range [{msf_cat.min():.4g}, {msf_cat.max():.4g}]")

g = torch.Generator().manual_seed(0)
Q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
Q = (Q * torch.sign(torch.det(Q))).numpy().astype(np.float32)
shift = np.array([31.7, -12.4, 88.1], dtype=np.float32)
ca0 = records[0]["ca"][:40]
f_ref = anm_msf(ca0)
f_rot = anm_msf(ca0 @ Q.T + shift)
rel = float((np.abs(f_ref - f_rot) / np.maximum(np.abs(f_ref).max(), 1e-6)).max())
print(f"ANM-MSF rotation invariance: {rel:.2e}")
assert rel < 1e-3
''')

md(r'''## Cell 8 - Recompute backbone dihedrals (identical to Phase42)''')
code(s42(14))

md(r'''## Cell 9 - ESM2 embeddings (identical to Phase42)''')
code(s42(16))

md(r'''## Cell 10 - Assemble PyG graphs (identical to Phase42's schema, PLUS one ANM-MSF column appended to the pre-topology features)''')
code(r'''
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ALPHABET)}

def radius_graph_capped(pos, r, max_neighbors=32):
    n = pos.size(0)
    pos = pos - pos.mean(dim=0, keepdim=True)
    dist = torch.cdist(pos, pos).clone()
    dist.fill_diagonal_(float("inf"))
    mask = dist <= r
    if max_neighbors is not None and max_neighbors < n:
        _, knn = torch.topk(dist, max_neighbors, dim=1, largest=False)
        keep_ = torch.zeros_like(mask); keep_.scatter_(1, knn, True); mask &= keep_
    mask = mask | mask.t()
    tgt, src = mask.nonzero(as_tuple=True)
    return torch.stack([src, tgt], dim=0).long()

N_PRE = 21 + 6 + ESM_DIM + 1   # +1 for the new ANM-MSF column
dataset = []
n_dropped = 0
for i, r in enumerate(records):
    if dihed[i] is None or esm_feats[i] is None:
        n_dropped += 1; continue
    n = len(r["ca"])
    if dihed[i].shape[0] != n or esm_feats[i].shape[0] != n:
        n_dropped += 1; continue
    onehot = torch.zeros(n, len(AA_ALPHABET))
    onehot[torch.arange(n), torch.tensor([AA_TO_IDX.get(c, 20) for c in r["seq"]])] = 1.0
    msf_z = torch.from_numpy(((msf_all[i] - msf_mu) / msf_sd).astype(np.float32)).unsqueeze(1)
    topo_i = topo_all[i][:, keep]
    topo_z = torch.from_numpy(((topo_i - mu) / sd).astype(np.float32))
    x = torch.cat([onehot, dihed[i], esm_feats[i], msf_z, topo_z], dim=1)

    pos = torch.from_numpy(r["ca"]); pos = pos - pos.mean(0, keepdim=True)
    ei = radius_graph_capped(pos, 8.0, 32)
    d_ = (pos[ei[0]]-pos[ei[1]]).norm(dim=1,keepdim=True)
    sep = (ei[0]-ei[1]).abs().float().unsqueeze(1)

    y_np = r["y"].astype(bool)
    true_center = torch.from_numpy(r["ca"][y_np].mean(axis=0) - r["ca"].mean(axis=0)) if y_np.any() else None
    if true_center is None:
        n_dropped += 1; continue

    d = Data(x=x, pos=pos, edge_index=ei, edge_attr=torch.cat([d_, torch.log1p(sep)], 1),
             y=torch.from_numpy(r["y"]))
    d.true_center = true_center.float()
    d.pdb_id, d.chain_id, d.split = r["pdb_id"], r["chain_id"], r["split"]
    d.num_nodes = n
    assert torch.isfinite(d.x).all()
    dataset.append(d)

print(f"built {len(dataset):,} graphs ({n_dropped} dropped: alignment/labeling failures)")
n_res_ = sum(d.num_nodes for d in dataset); n_pos_ = sum(int(d.y.sum()) for d in dataset)
POS_WEIGHT = (n_res_ - n_pos_) / max(n_pos_, 1)
print(f"residues {n_res_:,} | cryptic positives {n_pos_:,} ({100*n_pos_/max(n_res_,1):.2f}%) "
      f"| pos_weight {POS_WEIGHT:.2f} | N_PRE {N_PRE} (msf column at index {N_PRE-1})")
for s in ("train","test"):
    ds = [d for d in dataset if d.split == s]
    print(f"  {s:<6} {len(ds):>5} chains")
''')

md(r'''## Cell 11 - Carve a val subset from train, save (identical convention to Phase42, plus MSF metadata)''')
code(r'''
rng = np.random.RandomState(42)
train_idx = [i for i,d in enumerate(dataset) if d.split == "train"]
perm = rng.permutation(len(train_idx))
n_val = max(1, int(0.08 * len(perm)))
val_set = {train_idx[i] for i in perm[:n_val]}
for i, d in enumerate(dataset):
    if i in val_set: d.split = "val"

for s in ("train","val","test"):
    ds = [d for d in dataset if d.split == s]
    r_ = sum(d.num_nodes for d in ds); p_ = sum(int(d.y.sum()) for d in ds)
    print(f"{s:<6} chains {len(ds):>5}  residues {r_:>7,}  positives {p_:>5}  rate {100*p_/max(r_,1):>5.2f}%")

for d in dataset:
    assert torch.isfinite(d.x).all() and torch.isfinite(d.pos).all()
    assert d.edge_index.max() < d.num_nodes and d.edge_index.min() >= 0
print("integrity OK")

p = WORK / "phase159_cryptobench_anmmsf_graphs.pt"
torch.save({"dataset": dataset, "n_pre_topo": N_PRE, "topo_dim": int(keep.sum()),
            "msf_col_idx": N_PRE - 1, "msf_mean": msf_mu, "msf_std": msf_sd,
            "topo_names": names, "burial_names": BURIAL_NAMES, "dropped_names": dropped,
            "topo_mean": mu, "topo_std": sd, "esm_dim": ESM_DIM, "pos_weight": POS_WEIGHT,
            "anm_cutoff": ANM_CUTOFF, "anm_n_modes": N_MODES,
            "aa_alphabet": AA_ALPHABET, "source": "CryptoBench (Skrhak et al. 2025, btae745)"}, p)
print(f"saved -> {p} ({p.stat().st_size/1e6:.0f} MB)")
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase159_cryptobench_anmmsf_ingestion.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
