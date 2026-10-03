#!/usr/bin/env python3
"""Phase 284: crypticity-aware loss weight, the up-weighting idea flagged in paper.md's Future Work, now testable because Phase 283 gave every
train chain a proxy crypticity score. CryptoBench ESM2-3B transfer from the Phase 266b homology-clean checkpoints (seed s <- pretrained seed
s), IDENTICAL to Phase 266c/269 except the per-protein loss weight is now w = clip((median_train_my_apo / my_apo)^beta, 1, max_w) -- LOWER
my_apo (more cryptic-like, per Phase 283's apo-only cavity-voxel proxy) gets MORE weight, applied to the residue BCE and pocket-centre Huber
term exactly the way Phase 269 up-weighted small pockets. Same two arms (w1: beta 0.5 max 3; w2: beta 1.0 max 6 -- the Phase 224/269 settings,
reused rather than newly tuned, to keep this a fair apples-to-apples comparison against that established ablation). Baseline = Phase 266c
(unweighted, same seeds, same checkpoints). Chains without a Phase 283 proxy score (not every training chain's structure/site parsed cleanly
under the proxy's own CIF pipeline) get weight 1.0 (neutral, no up/down-weight) -- logged, not silently assumed complete.

Caveat carried over from Phase 283's write-up: this proxy is highly correlated with pocket size on validation (Spearman r=0.876), so if this
finds a gain it must be checked against Phase 269's small-pocket weighting (already NEGATIVE) before crediting crypticity specifically.

usage: build_nb284.py <w1|w2> <output_root>"""
import json, sys
from pathlib import Path

ARM = "full"; WARM = sys.argv[1]; ROOT_OUT = Path(sys.argv[2]).resolve(); SLUG = "284" + WARM
BETA, MAXW = {"w1": (0.5, 3.0), "w2": (1.0, 6.0)}[WARM]
HERE = Path(__file__).resolve().parent
nb129 = json.load(open(HERE.parents[1] / "phase129_cryptobench_esm3b_transfer.ipynb"))
s129 = lambda i: "".join(nb129["cells"][i]["source"])

CRYPT_SCORES = json.load(open(HERE.parent / "phase283" / "results" / "phase283_train_crypticity_proxy_scores.json"))
MY_APO_BY_KEY = {k: v["my_apo"] for k, v in CRYPT_SCORES.items()}  # "pdbid_chain" (lowercase pdb) -> my_apo voxel count

C = []
CRYPT_LOSS_A = """nG = int(b.batch.max().item()) + 1
            yb = b.y.float(); gw = b.crypt_w if (tr and CRYPT_BETA > 0) else torch.ones(nG, device=DEVICE)
            rw = gw[b.batch]; loss = (crit(logits, yb) * rw).sum() / rw.sum()
"""
CRYPT_LOSS_B = "loss = loss + lam * (huber(pred_c, true_c).mean(1) * gw).sum() / gw.sum()"
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase 284{WARM} - CryptoBench + ESM2-3B transfer, crypticity-proxy loss weight (beta={BETA}, max_w={MAXW})

Phase 283 built a grid-based cavity-voxel detector as an apo-only proxy for the external crypticity audit's tool (validated: Spearman
r=0.256 p=0.0035 vs the audit's own published counts on 128 labelled test chains, pooled AUC 0.740 cryptic-vs-preformed) and ran it over
CryptoBench's full training pool, giving 704 training chains a proxy score. This up-weights training chains the proxy scores as more
cryptic-like (lower my_apo) in the loss, exactly the way Phase 269 up-weighted small pockets, and compares against the Phase 266c
(unweighted) baseline -- same checkpoints, same seeds, same everything else.
''')
md("## Cell 1 - Install")
code("!pip install -q torch-geometric==2.6.1")
md("## Cell 2 - Setup: Phase 48's CryptoBench-3B blob + Phase 227's geometry + Phase 283's crypticity-proxy scores")
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

''' + f"MY_APO_BY_KEY = {repr(MY_APO_BY_KEY)}\n" + r'''
n_matched = sum(1 for d in dataset if d.split == "train" and f"{str(d.pdb_id).lower()}_{str(d.chain_id)}" in MY_APO_BY_KEY)
n_train = sum(1 for d in dataset if d.split == "train")
print(f"crypticity proxy coverage on train: {n_matched}/{n_train} chains matched (rest default to neutral weight 1.0)")
for d in dataset: d.crypt_w = torch.tensor([1.0], dtype=torch.float32)   # every graph gets the attribute (train/val/test alike): avoids the
                                                                          # Phase 280c-style PyG batch-collation KeyError from uneven attrs

splits = {s: [d for d in dataset if d.split == s] for s in ("train", "val", "test")}
print(f"node dim {dataset[0].x.size(1)} = {N_PRE} + {TOPO_DIM} + {GEOM_DIM}; splits", {k: len(v) for k, v in splits.items()})
assert dataset[0].x.size(1) == N_PRE + SIDE_DIM
''')
md("## Cell 3 - Model (Phase 129, verbatim)")
code(s129(6))
md("## Cell 4 - Training machinery (Phase 129, width-parameterised; fit() loads the seed-matched pretrained state; crypticity-weighted loss)")
code(s129(8).replace("LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head)", "LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=use_loc_head)")
     .replace("crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))", "crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE), reduction='none')")
     .replace("huber = nn.SmoothL1Loss(beta=4.0)", "huber = nn.SmoothL1Loss(beta=4.0, reduction='none')")
     .replace("loss = crit(logits, b.y)\n", CRYPT_LOSS_A)
     .replace("loss = loss + lam * huber(pred_c, true_c)", CRYPT_LOSS_B))
md(f"## Cell 5 - Fine-tune arm `{ARM}` x 8 seeds from Phase 263 seed-matched checkpoints; crypticity-proxy loss weight; save probabilities and centers")
code(f'''
ARM = "{ARM}"; WARM = "{WARM}"
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

CRYPT_BETA = __BETA__; CRYPT_MAXW = __MAXW__
_matched_apo = [MY_APO_BY_KEY[f"{str(d.pdb_id).lower()}_{str(d.chain_id)}"] for d in splits["train"]
                if f"{str(d.pdb_id).lower()}_{str(d.chain_id)}" in MY_APO_BY_KEY]
CRYPT_MEDIAN = float(np.median(_matched_apo))
for d in splits["train"]:
    key = f"{str(d.pdb_id).lower()}_{str(d.chain_id)}"
    if key in MY_APO_BY_KEY:
        w = float(np.clip((CRYPT_MEDIAN / max(MY_APO_BY_KEY[key], 1)) ** CRYPT_BETA, 1.0, CRYPT_MAXW))
        d.crypt_w = torch.tensor([w], dtype=torch.float32)
_w = np.array([float(d.crypt_w.item()) for d in splits["train"]])
print(f"crypticity weights: median train my_apo {CRYPT_MEDIAN:.0f}; weight mean {_w.mean():.2f}, max {_w.max():.2f}, share up-weighted {np.mean(_w > 1.0):.2f}")

BEST_LAMBDA = 0.1
PREDS = {"arm": ARM, "seeds": {}}; t0 = time.time()
for seed in range(8):
    ck = locate(f"pretrain266_{ARM.replace('+','_')}_seed{seed}.pt")
    model, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, pretrained_path=ck)
    PREDS["seeds"][seed] = {"val": predict(model, "val"), "test": predict(model, "test"), "val_pr": float(val_pr)}
    torch.save(PREDS, WORK / f"phase284{WARM}_{ARM.replace('+','_')}_preds.pt")          # incremental: a late crash loses one seed, not all
    print(f"  [{ARM}] seed {seed}: val PR {val_pr:.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)
print("done")
'''.replace('__BETA__', repr(BETA)).replace('__MAXW__', repr(MAXW)))
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / f"phase284{WARM}_cb3b_crypt_weight.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = {"id": f"blackpearls1/phase284{WARM}-cb3b-crypt-weight", "title": f"Phase284{WARM} Cb3b Crypt Weight", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase48-cryptobench-esm3b", "blackpearls1/phase227-geometry-ingestion", "blackpearls1/phase266b-pretrain-3b-clean"]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
