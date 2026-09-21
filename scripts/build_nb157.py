#!/usr/bin/env python3
"""Phase 157: CryptoBench ingestion + topological VARIABILITY features.
Extends Phase42's exact ingestion pipeline (same fetch/parse/featurizer/
dihedral/ESM cells, reused verbatim) with a new per-residue feature
block: the STD of the same persistent-homology feature vector across
an 8-conformer ANM (Anisotropic Network Model)-perturbed ensemble,
capturing "this local topology is unstable across plausible nearby
conformations" -- a proxy for cryptic-pocket-relevant flexibility that
the existing STATIC (single-structure) topology block cannot see.
Doubles the topology-derived feature width (34 static + ~34 variability
columns) with zero change to the model architecture (the fusion module
just sees a wider topology block)."""
import json
from pathlib import Path

nb42 = json.load(open("/Users/shivenkhurana/bioproj/phase42_cryptobench_ingestion.ipynb"))
c42 = nb42["cells"]
s42 = lambda i: (c42[i]["source"] if isinstance(c42[i]["source"], str) else "".join(c42[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 157 — CryptoBench Ingestion + Topological Variability

Extends Phase42's exact ingestion (same fetch/parse/featurizer/dihedral/
ESM cells) with a new feature block: the per-residue STD of the
persistent-homology feature vector across an 8-conformer ANM-perturbed
ensemble, on top of the existing STATIC (single apo-structure) topology
block. Hypothesis: a cryptic pocket is, by definition, a cavity that is
not present in the static apo structure but exists in some nearby,
biologically plausible conformation -- so the *variability* of local
topology across plausible small conformational changes should carry
signal the static snapshot cannot. Zero architecture change: the
fusion module (`Fus`) just sees a wider topology block (34 static + 34
variability columns instead of 34).
''')

md(r'''## Cell 1 — Install dependencies (identical to Phase42)''')
code(s42(2))

md(r'''## Cell 2 — Setup, fetch CryptoBench's dataset.json + folds.json from OSF (identical to Phase42)''')
code(s42(4))

md(r'''## Cell 3 — Fetch apo structures (identical to Phase42)''')
code(s42(6))

md(r'''## Cell 4 — Parse apo chains, build per-residue crypticity labels (identical to Phase42)''')
code(s42(8))

md(r'''## Cell 5 — Featuriser (identical to Phase42 -- byte-for-byte the Phase 2b/6b/29/36 definition)''')
code(s42(10))

md(r'''## Cell 6 — ANM (Anisotropic Network Model): compute low-frequency modes and generate perturbed conformations''')
code(r'''
N_CONFORMERS = 8
TARGET_RMSD = 1.2   # Angstrom, typical moderate ANM thermal-fluctuation scale
N_MODES = 10
ANM_CUTOFF = 13.0


def anm_modes(ca, cutoff=ANM_CUTOFF, n_modes=N_MODES, gamma=1.0):
    """Standard ANM Hessian (Cα-only elastic network): eigenvectors give
    directional (x,y,z) per-residue displacement for each collective mode,
    unlike GNM's scalar/isotropic modes -- needed here because we want to
    actually DISPLACE atoms, not just estimate fluctuation magnitude."""
    n = len(ca)
    if n < 4:
        return np.zeros((0,)), np.zeros((0, n, 3))
    diff = ca[:, None, :] - ca[None, :, :]           # diff[i,j] = ca[i]-ca[j]
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
        return np.zeros((0,)), np.zeros((0, n, 3))
    modes = eigvecs[:, nz].T.reshape(len(nz), n, 3)
    return eigvals[nz], modes


def perturb_conformation(ca, modes, eigvals, target_rmsd, rng):
    if len(eigvals) == 0:
        return ca.copy()
    coeffs = rng.normal(0, 1, size=len(eigvals)) / np.sqrt(eigvals)
    disp = np.tensordot(coeffs, modes, axes=(0, 0))
    cur = np.sqrt((disp ** 2).sum(1).mean())
    if cur > 1e-8:
        disp = disp * (target_rmsd / cur)
    return ca + disp


def topo_variability_for_chain(ca_xyz, per_res, seed):
    """Returns (static_topo, variability_topo): static is byte-identical
    to Phase42's topo_for_chain output; variability is the per-residue
    STD of that same feature vector across the original + 8 ANM-
    perturbed conformations."""
    rng = np.random.default_rng(seed)
    static = topo_for_chain(ca_xyz, per_res)
    eigvals, modes = anm_modes(ca_xyz)
    ensemble = [static]
    for _ in range(N_CONFORMERS):
        ca_pert = perturb_conformation(ca_xyz, modes, eigvals, TARGET_RMSD, rng)
        disp = ca_pert - ca_xyz
        per_res_pert = [atoms + disp[i] for i, atoms in enumerate(per_res)]
        ensemble.append(topo_for_chain(ca_pert, per_res_pert))
    stacked = np.stack(ensemble, axis=0)
    var = stacked.std(axis=0)
    return static, var
''')

md(r'''## Cell 7 — Compute static + variability topology for every chain (9 conformers each: 1 original + 8 ANM-perturbed)''')
code(r'''
n_res_total = sum(len(r["ca"]) for r in records)
print(f"computing persistent homology on {n_res_total:,} residues x 9 conformers "
      f"({os.cpu_count()} workers, ~6.4 ms/residue/conformer -> ETA {n_res_total*0.0064*9/60:.0f} min)")
t0 = time.time()
results = Parallel(n_jobs=max(1, os.cpu_count() or 2), verbose=5)(
    delayed(topo_variability_for_chain)(r["ca"], r["per_res"], seed=i) for i, r in enumerate(records))
el = time.time() - t0
print(f"\ndone in {el/60:.1f} min ({1000*el/max(n_res_total,1)/9:.2f} ms/residue/conformer)")

topo_all = [r[0] for r in results]
topovar_all = [r[1] for r in results]

topo_cat = np.concatenate(topo_all, axis=0)
keep = topo_cat.std(axis=0) > 1e-9
dropped = [TOPO_NAMES[i] for i in range(TOPO_DIM) if not keep[i]]
names = [TOPO_NAMES[i] for i in range(TOPO_DIM) if keep[i]]
print(f"static: pruned {len(dropped)} constant column(s): {dropped}")
mu, sd = topo_cat[:, keep].mean(0), topo_cat[:, keep].std(0); sd[sd < 1e-6] = 1.0

topovar_cat = np.concatenate(topovar_all, axis=0)
keep_var = topovar_cat.std(axis=0) > 1e-9
dropped_var = [TOPO_NAMES[i] for i in range(TOPO_DIM) if not keep_var[i]]
names_var = [f"var_{TOPO_NAMES[i]}" for i in range(TOPO_DIM) if keep_var[i]]
print(f"variability: pruned {len(dropped_var)} constant column(s): {dropped_var}")
mu_var, sd_var = topovar_cat[:, keep_var].mean(0), topovar_cat[:, keep_var].std(0); sd_var[sd_var < 1e-6] = 1.0

g = torch.Generator().manual_seed(0)
Q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
Q = (Q * torch.sign(torch.det(Q))).numpy().astype(np.float32)
shift = np.array([31.7, -12.4, 88.1], dtype=np.float32)
ca0, pr0 = records[0]["ca"][:40], records[0]["per_res"][:40]
f_ref = topo_for_chain(ca0, pr0)
f_rot = topo_for_chain(ca0 @ Q.T + shift, [a @ Q.T + shift for a in pr0])
rel = float((np.abs(f_ref - f_rot) / np.maximum(np.abs(f_ref).max(0), 1e-6)).max())
print(f"static topology rotation invariance: {rel:.2e}")
assert rel < 1e-3
''')

md(r'''## Cell 8 — Recompute backbone dihedrals (identical to Phase42)''')
code(s42(14))

md(r'''## Cell 9 — ESM2 embeddings (identical to Phase42)''')
code(s42(16))

md(r'''## Cell 10 — Assemble PyG graphs (identical to Phase42's schema, PLUS the new variability block appended)''')
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

N_PRE = 21 + 6 + ESM_DIM
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
    topo_i = topo_all[i][:, keep]
    topo_z = torch.from_numpy(((topo_i - mu) / sd).astype(np.float32))
    topovar_i = topovar_all[i][:, keep_var]
    topovar_z = torch.from_numpy(((topovar_i - mu_var) / sd_var).astype(np.float32))
    x = torch.cat([onehot, dihed[i], esm_feats[i], topo_z, topovar_z], dim=1)

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
      f"| pos_weight {POS_WEIGHT:.2f}")
print(f"static topo dim {int(keep.sum())} + variability topo dim {int(keep_var.sum())} "
      f"= total topo dim {int(keep.sum())+int(keep_var.sum())}")
for s in ("train","test"):
    ds = [d for d in dataset if d.split == s]
    print(f"  {s:<6} {len(ds):>5} chains")
''')

md(r'''## Cell 11 — Carve a val subset from train, save (identical convention to Phase42, plus variability metadata)''')
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

STATIC_TOPO_DIM = int(keep.sum())
VAR_TOPO_DIM = int(keep_var.sum())
p = WORK / "phase157_cryptobench_topovar_graphs.pt"
torch.save({"dataset": dataset, "n_pre_topo": N_PRE,
            "topo_dim": STATIC_TOPO_DIM + VAR_TOPO_DIM,
            "static_topo_dim": STATIC_TOPO_DIM, "var_topo_dim": VAR_TOPO_DIM,
            "topo_names": names + names_var, "burial_names": BURIAL_NAMES, "dropped_names": dropped + dropped_var,
            "topo_mean": mu, "topo_std": sd, "topovar_mean": mu_var, "topovar_std": sd_var,
            "esm_dim": ESM_DIM, "pos_weight": POS_WEIGHT,
            "n_conformers": N_CONFORMERS, "target_rmsd": TARGET_RMSD, "anm_cutoff": ANM_CUTOFF,
            "aa_alphabet": AA_ALPHABET, "source": "CryptoBench (Skrhak et al. 2025, btae745)"}, p)
print(f"saved -> {p} ({p.stat().st_size/1e6:.0f} MB)")
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase157_cryptobench_topovar_ingestion.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
