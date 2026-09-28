#!/usr/bin/env python3
"""Phase 280c: Phase 266c's EXACT fine-tune recipe (homology-clean 3B transfer, topology-only, seed s <- pretrained seed s) with one change:
CryptoBench's own 700-chain training set is extended with the homology-clean "preformed"/"neither" extra training chains Phase 280b featurized
(same node format, verified identical width). Val/test unchanged, so results are directly comparable to Phase 266c's original numbers --
the only variable that differs is how much training data the model saw. usage: build_nb280c.py <output_root>"""
import json, sys
from pathlib import Path

ARM = "full"; ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "280c"
HERE = Path(__file__).resolve().parent
nb129 = json.load(open(HERE.parents[1] / "phase129_cryptobench_esm3b_transfer.ipynb"))
s129 = lambda i: "".join(nb129["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase 280c - CryptoBench transfer (Phase 266c recipe) with extra homology-clean training data

Same checkpoints, same 8 seeds, same official val/test as Phase 266c. Training data is CryptoBench's official 700 chains PLUS the
homology-clean "preformed"/"neither" chains from an independent benchmark (Phase 280a/280b) -- real labelled apo/holo pairs never used
anywhere in this project before. Validation-only selection; test predictions are saved, not used for any choice.
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1")
md("## Cell 2 — Setup: Phase 48's CryptoBench-3B blob + Phase 227's geometry")
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

extra = torch.load(locate("phase280b_extra_train_graphs.pt"), map_location="cpu", weights_only=False)["dataset"]
for d in extra:
    assert d.x.size(1) == N_PRE + SIDE_DIM, (d.x.size(1), N_PRE + SIDE_DIM)
    for extra_attr in ("eval_mask", "accession"):   # present on these graphs (PocketMiner-era / provenance fields), absent on the official
        if hasattr(d, extra_attr): delattr(d, extra_attr)   # CryptoBench graphs -- PyG's batch collation requires the same keys on every Data object
n_official = len(splits["train"]); splits["train"] = splits["train"] + extra
print(f"extra homology-clean training chains added: {len(extra)}  (official {n_official} -> total {len(splits['train'])})")
''')
md("## Cell 3 — Model (Phase 129, verbatim)")
code(s129(6))
md("## Cell 4 — Training machinery (Phase 129, width-parameterised; fit() loads the seed-matched pretrained state)")
code(s129(8).replace("LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head)", "LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=use_loc_head)"))
md(f"## Cell 5 - Fine-tune arm `{ARM}` x 8 seeds from Phase 263 seed-matched checkpoints; save per-residue probabilities and predicted centers")
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
PREDS = {"arm": ARM, "seeds": {}}; t0 = time.time()
for seed in range(8):
    ck = locate(f"pretrain266_{ARM.replace('+','_')}_seed{seed}.pt")
    model, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, pretrained_path=ck)
    PREDS["seeds"][seed] = {"val": predict(model, "val"), "test": predict(model, "test"), "val_pr": float(val_pr)}
    torch.save(PREDS, WORK / f"phase280c_{ARM.replace('+','_')}_preds.pt")          # incremental: a late crash loses one seed, not all
    print(f"  [{ARM}] seed {seed}: val PR {val_pr:.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)
print("done")
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase280c_cb3b_transfer_extradata.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = {"id": "blackpearls1/phase280c-cb3b-transfer-extradata", "title": "Phase280c Cb3b Transfer Extradata", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase48-cryptobench-esm3b", "blackpearls1/phase227-geometry-ingestion", "blackpearls1/phase266b-pretrain-3b-clean", "blackpearls1/phase280b-extra-train-featurize"]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
