#!/usr/bin/env python3
"""Phase 286: a genuinely LOCAL (spatial) attention mechanism for pocket-center localization -- the one untried
direction flagged in paper.md's Future Work ("A dedicated cryptic-pocket localization benchmark and stronger
localization architecture... a genuinely local (rather than chain-wide) attention mechanism for the center
prediction are the two most promising untried directions"). The only prior attempt at this (Phase 161/162,
top-K masking) restricted the softmax to the K highest-SCORING residues -- not a spatial restriction at all,
so it could still pool over residues scattered across the whole chain if they all happened to score highly,
and it came back a null result on DCC, the metric it was built to move.

This is mechanistically different: a two-pass, purely SPATIAL local-attention head. Pass 1 computes the
existing chain-wide softmax-weighted centroid (c0), unchanged. Pass 2 adds a smooth penalty to each residue's
localization logit based on its 3D distance from c0 -- `logit' = logit - (dist_to_c0 / R)^2` -- before a
second softmax produces the FINAL predicted center. This directly forces the final prediction to draw only
from residues physically near the model's own preliminary guess, rather than residues merely scoring highly.
R=10 A is not tuned: it is the same edge-cutoff radius already used elsewhere in this exact architecture
(`radius cutoff (~8-10 A)`, Methods), reused here as an already-established physical scale rather than a new
number chosen to fit this specific result.

Equivariance is preserved by construction, not just tested after the fact: c0 is an attention-weighted
combination of the equivariant `pos` tensor (the SAME construction already used for the original center
prediction), and dist_to_c0 = ||pos_i - c0[batch_i]|| is an SE(3)-INVARIANT scalar (a norm of a difference of
two equivariant quantities, the identical pattern EGCL itself already uses for its own edge features). Adding
an invariant scalar to an already-invariant logit keeps every downstream quantity exactly as equivariant as
the unmodified model -- verified numerically below before any training, the same discipline used for every
other architecture change in this project.

This changes ZERO parameter shapes (R is a fixed constant, not a learned parameter; loc_head itself is
untouched), so the SAME 8 Phase 266b homology-clean pretrained checkpoints load and fine-tune with zero shape
mismatch -- identical practical advantage to what made the top-K test possible, and what makes a direct,
paired, 1:1 comparison against the existing Phase 266c baseline (not rerun) possible here too.

RESULT (see cryptic-pocket/phase286/results/): null on the target metric, like top-K before it -- DCC@4A/8A
and mean center error all moved within noise, no confirmed direction. Classification metrics (PR/ROC/MCC)
were NOT bit-identical to baseline as this docstring originally (incorrectly) predicted they'd be -- they
shifted by up to 0.026 per seed, unconfirmed either direction. That prediction was wrong: the classification
head's READOUT never touches pred_center, but the shared EGCL trunk is trained jointly against both the
classification and center-prediction losses, so changing the center-prediction loss's gradient changes the
learned trunk representations the classification head also reads from. Noted here rather than silently
corrected, since the original (wrong) reasoning is exactly the kind of assumption worth catching explicitly.

usage: build_nb286.py <output_root>"""
import json, sys
from pathlib import Path

ARM = "full"; ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "286"
LOCAL_RADIUS = 10.0  # Angstroms; matches this project's existing EGCL edge-cutoff radius (Methods), not tuned
HERE = Path(__file__).resolve().parent
nb129 = json.load(open(HERE.parents[1] / "phase129_cryptobench_esm3b_transfer.ipynb"))
s129 = lambda i: "".join(nb129["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase 286 - A genuinely local (spatial) attention mechanism for pocket-center localization

Baseline = Phase 266c (unweighted, homology-clean 3B transfer, same 8 seeds, same checkpoints, not rerun).
This notebook only changes the localization head's *readout*: a second, spatially-biased softmax pass
(radius {LOCAL_RADIUS} A, not tuned -- see module docstring) replaces the single chain-wide softmax. Same
data, loss, optimiser, early stopping, and pretrained-checkpoint initialisation as every other arm in this
lineage (Phase 266c/269/284).
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1")
md("## Cell 2 — Setup: Phase 48's CryptoBench-3B blob + Phase 227's geometry (identical to Phase 266c)")
code(s129(4) + r'''

geo = torch.load(locate("phase227_geometry_features.pt"), map_location="cpu", weights_only=False)
GEOM_DIM = len(geo["names"]); assert GEOM_DIM == 17
SIDE_DIM = TOPO_DIM + GEOM_DIM
_st = torch.load(locate("phase266_geom_stats.pt"), map_location="cpu", weights_only=False)     # the statistics the pretrained weights expect
g_mu, g_sd = _st["g_mu"].float(), _st["g_sd"].float(); print("pool geometry stats loaded; pretrain n_train", _st["n_train"])
missing = 0
for d in dataset:
    k = (str(d.pdb_id).lower(), str(d.chain_id))
    f = geo["feats"].get(k)
    if f is None or f.shape[0] != d.x.size(0): missing += 1; continue
    d.x = torch.cat([d.x, ((torch.from_numpy(f) - g_mu) / g_sd).clamp(-5, 5)], dim=1)
assert missing == 0, f"{missing} chains could not be matched to geometry"
splits = {s: [d for d in dataset if d.split == s] for s in ("train", "val", "test")}
print(f"node dim {dataset[0].x.size(1)} = {N_PRE} + {TOPO_DIM} + {GEOM_DIM}; splits", {k: len(v) for k, v in splits.items()})
assert dataset[0].x.size(1) == N_PRE + SIDE_DIM
''' + f'\nLOCAL_RADIUS = {LOCAL_RADIUS}  # Angstroms; defined once, early, so every later cell (incl. the equivariance check) sees it\n')
md("## Cell 3 — Model, with an optional spatial local-attention pass on the localization readout (parameter shapes unchanged)")
code(s129(6).replace(
    '''class LocalizingEGNN(nn.Module):
    """Same TopologicalEGNN as every prior phase, plus a second head that
    predicts one pocket-center point per chain via a batch-aware softmax over
    a dedicated per-residue localization logit -- decoupled from the
    classification probability on purpose (see markdown above)."""
    def __init__(s, n_pre, td, h=128, L=4, dr=0.1, use_loc_head=True):
        super().__init__(); s.n_pre=n_pre; s.use_loc_head=use_loc_head''',
    '''class LocalizingEGNN(nn.Module):
    """Same TopologicalEGNN as every prior phase. The localization head's PARAMETERS are unchanged -- only the
    readout is new: `local_radius` (None = original single-pass chain-wide softmax; a float R = a second,
    spatially-biased softmax pass, restricting the final center estimate to residues near the first pass's own
    preliminary center). Existing pretrained checkpoints load into this class with zero shape mismatch regardless
    of `local_radius`."""
    def __init__(s, n_pre, td, h=128, L=4, dr=0.1, use_loc_head=True, local_radius=None):
        super().__init__(); s.n_pre=n_pre; s.use_loc_head=use_loc_head; s.local_radius=local_radius'''
).replace(
    '''            loc_logit = s.loc_head(h).squeeze(-1)
            attn = pyg_softmax(loc_logit, batch)                    # sums to 1 per graph
            weighted = attn.unsqueeze(-1) * pos                     # (N,3)
            n_graphs = int(batch.max().item()) + 1
            pred_center = torch.zeros(n_graphs, 3, device=h.device).index_add_(0, batch, weighted)''',
    '''            loc_logit = s.loc_head(h).squeeze(-1)
            n_graphs = int(batch.max().item()) + 1
            attn0 = pyg_softmax(loc_logit, batch)                   # pass 1: original chain-wide softmax
            weighted0 = attn0.unsqueeze(-1) * pos
            c0 = torch.zeros(n_graphs, 3, device=h.device).index_add_(0, batch, weighted0)
            if s.local_radius is not None:
                dist = (pos - c0[batch]).norm(dim=-1)               # SE(3)-invariant: norm of a difference of
                final_logit = loc_logit - (dist / s.local_radius) ** 2   # two equivariant quantities
            else:
                final_logit = loc_logit
            attn = pyg_softmax(final_logit, batch)                  # pass 2 (or pass 1 again if local_radius=None)
            weighted = attn.unsqueeze(-1) * pos                     # (N,3)
            pred_center = torch.zeros(n_graphs, 3, device=h.device).index_add_(0, batch, weighted)'''
))
md("## Cell 4 — Equivariance check (rotate a batch, confirm both centers rotate correspondingly, before any training)")
code(r'''
torch.manual_seed(0)
_probe = splits["val"][0].clone()
_m = LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=True, local_radius=LOCAL_RADIUS).to(DEVICE).eval()
_probe = _probe.to(DEVICE)
with torch.no_grad():
    _, c_ref = _m(_probe)
Q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(1)))
Q = (Q * torch.sign(torch.det(Q))).to(DEVICE)
shift = torch.tensor([17.3, -4.1, 62.0], device=DEVICE)
_rot = _probe.clone(); _rot.pos = _probe.pos @ Q.T + shift
with torch.no_grad():
    _, c_rot = _m(_rot)
c_ref_transformed = c_ref @ Q.T + shift
err = (c_rot - c_ref_transformed).norm().item()
print(f"equivariance check (local_radius={LOCAL_RADIUS}): max error {err:.2e} (expect ~1e-5, matching every other equivariance check in this project)")
assert err < 1e-3, "equivariance broken -- do not proceed to training"
del _m, _probe, _rot, c_ref, c_rot
''')
md("## Cell 5 — Training machinery (Phase 129, unchanged)")
code(s129(8).replace("LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head)",
                     "LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=use_loc_head, local_radius=LOCAL_RADIUS)"))
md(f"## Cell 6 - Fine-tune arm `{ARM}` x 8 seeds from Phase 266b seed-matched checkpoints; save probabilities and predicted centers")
code(f'''
ARM = "{ARM}"
''' + r'''
KEEP_SIDE = {"full": list(range(TOPO_DIM)), "geom": list(range(TOPO_DIM, SIDE_DIM)), "geom+topo": list(range(SIDE_DIM))}
m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE[ARM]] = 1.0
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_

@torch.no_grad()
def predict(model, split):
    out = {}
    for d in splits[split]:
        b = d.clone().to(DEVICE); logits, pc = model(b)
        out[(str(d.pdb_id).lower(), str(d.chain_id))] = dict(p=torch.sigmoid(logits).cpu().numpy().astype(np.float32),
                                                             c=pc.cpu().numpy()[0], y=d.y.numpy().astype(np.int8),
                                                             tc=d.true_center.numpy())
    return out

BEST_LAMBDA = 0.1
PREDS = {"arm": ARM, "local_radius": LOCAL_RADIUS, "seeds": {}}; t0 = time.time()
for seed in range(8):
    ck = locate(f"pretrain266_{ARM.replace('+','_')}_seed{seed}.pt")
    model, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, pretrained_path=ck)
    PREDS["seeds"][seed] = {"val": predict(model, "val"), "test": predict(model, "test"), "val_pr": float(val_pr)}
    torch.save(PREDS, WORK / f"phase286_{ARM.replace('+','_')}_preds.pt")          # incremental: a late crash loses one seed, not all
    print(f"  [{ARM}] seed {seed}: val PR {val_pr:.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)
print("done")
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase286_cb3b_local_attention.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = {"id": "blackpearls1/phase286-cb3b-local-attention", "title": "Phase286 Cb3b Local Attention", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        # CPU, not GPU: weekly GPU quota was exhausted by Phase 285. The underlying blob (phase48) is already
        # ESM-embedded, so this kernel only trains the small EGNN (h=128, L=4) -- feasible on CPU, just slower.
        "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase48-cryptobench-esm3b", "blackpearls1/phase227-geometry-ingestion", "blackpearls1/phase266b-pretrain-3b-clean"]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
