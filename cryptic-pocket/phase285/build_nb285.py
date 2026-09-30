#!/usr/bin/env python3
"""Phase 285: which topological property drives the CONFIRMED classification residual beyond geometry?
(The Future Work item this project flagged as unidentified: "a systematic per-dimension ablation at the
scale where the residual is largest and most consistent (650M/3B main-pipeline classification), which this
project ran the necessary controls for but did not fully decompose." The existing H0/H1/H2/burial
decomposition, Phase 58-86, answered this question for LOCALIZATION at 35M-embedding/round-3-pool scale
(found: H0 measurably weaker than H1, everything else tied) -- it never touched classification, and never
ran at the embedding scale (650M/3B) where the classification residual is actually confirmed.

This reuses Phase 240a/b's exact data (phase238_graphs_esm650m.pt + phase229 geometry), model, training loop,
and 10-fold KFold(seed=1337) x 2-seed design UNCHANGED -- only the arms differ. Phase 240a/b already ran and
cached `geom` and `geom+topo` (all 34 topology columns) at all 10 folds; this notebook adds four NEW arms,
each combining the full 17-feature geometry block with exactly ONE topology dimension (H0/H1/H2/burial,
using Phase 58's exact column-partition rule: name-prefix on TOPO_NAMES, loaded from phase6b_graphs.pt's
metadata only -- topology column ORDER is invariant across embedding scales, since every re-embedded blob
descends from the same Phase 6b topology block with new ESM concatenated after it, never recomputed).
Every arm keeps an identical 51-wide side channel (unselected columns zeroed) so architecture and parameter
count stay equal across arms, exactly as every other geometry-control arm in this paper.

Primary, pre-specified comparison: pairwise among {geom+H0, geom+H1, geom+H2, geom+burial}, paired over the
10 folds (2 seeds averaged per fold), on test PR-AUC and ROC-AUC -- the same two-test rule (paired t AND
Wilcoxon both <0.05) used throughout this project. geom+topo (all four) and geom alone are NOT rerun; the
existing Phase 240a/b records are reused as-is as reference points on the same 10-fold split.

usage: build_nb285.py <a|b> <output_dir>
  a = folds 0-4, b = folds 5-9 (matching Phase 240a/b's split, two GPU kernels in parallel)
"""
import json, sys
from pathlib import Path

TAG = sys.argv[1]
OUT_DIR = Path(sys.argv[2]).resolve()
FOLDS = {"a": [0, 1, 2, 3, 4], "b": [5, 6, 7, 8, 9]}[TAG]
ROOT = Path(__file__).resolve().parents[2]

nb240a = json.load(open(ROOT / "cryptic-pocket" / "phase240a" / "phase240a_geometry_control_650m.ipynb"))
src240 = lambda i: "".join(nb240a["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase 285{TAG} — Which topological dimension drives the classification residual? (ESM2-650M, folds {FOLDS[0]}-{FOLDS[-1]})

Phase 240a/b confirmed a small but real classification residual, geom+topo minus geom alone, at 650M scale
(+0.0091 PR-AUC, 8/10 folds; +0.0032 ROC-AUC, 9/10 folds). This decomposes that residual by topological
dimension: four new arms, `geom+H0`, `geom+H1`, `geom+H2`, `geom+burial`, each the full 17-feature geometry
block plus exactly one topology dimension. Same 2,056-chain pool, same 10-fold split (seed 1337), same
model/loss, as every other geometry-control notebook in this project; `geom` and `geom+topo` are not rerun
here, since Phase 240a/b already produced them on this identical split.
''')
md("## Cell 1 — Setup")
code("!pip install -q torch-geometric==2.6.1 biopython==1.84")
md("## Cell 2 — Load the ESM2-650M graphs, the Phase 229 geometry, and Phase 6b's topology column names")
code(src240(4) + r'''

# Phase 6b's original topology blob (2,056-chain pool) carries topo_names/burial_names -- the same 34
# columns, in the same order, that flow unchanged into every re-embedded blob (phase238/239), since topology
# is computed once from 3D structure and never recomputed when only the ESM scale changes. We only need this
# blob's metadata, not its dataset (which uses 35M embeddings and is otherwise identical/unused here).
blob6b = torch.load(locate("phase6b_graphs.pt"), map_location="cpu", weights_only=False)
TOPO_NAMES = blob6b["topo_names"]; BURIAL_NAMES = set(blob6b["burial_names"])
assert len(TOPO_NAMES) == TOPO_DIM, (len(TOPO_NAMES), TOPO_DIM)
del blob6b
DIM_MASKS = {
    "H0": np.array([n.startswith("H0_") for n in TOPO_NAMES]),
    "H1": np.array([n.startswith("H1_") for n in TOPO_NAMES]),
    "H2": np.array([n.startswith("H2_") for n in TOPO_NAMES]),
    "burial": np.array([n in BURIAL_NAMES for n in TOPO_NAMES]),
}
for name, mask in DIM_MASKS.items():
    kept = [n for n, k in zip(TOPO_NAMES, mask) if k]
    print(f"  {name:<8} {mask.sum()} columns: {kept}")
assert sum(m.sum() for m in DIM_MASKS.values()) == TOPO_DIM, "masks should exactly partition the topo block"
print("masks partition the full topo block exactly -- no overlap, no gaps")
''')
md("## Cell 3 — Model and training machinery (Phase 11, unchanged)")
code(src240(6))
md("## Cell 4 — Four single-dimension arms x the nested CV loop")
code(f'''
FOLDS_TO_RUN = {FOLDS}
''' + r'''
N_FOLDS, SEEDS, EPOCHS, PATIENCE, BATCH = 10, [0, 1], 40, 8, 16
TOPO_COLS = list(range(N_PRE, N_PRE + TOPO_DIM))
GEOM_COLS = list(range(N_PRE + TOPO_DIM, N_PRE + SIDE_DIM))
DIM_COLS = {dim: [TOPO_COLS[i] for i, keep in enumerate(mask) if keep] for dim, mask in DIM_MASKS.items()}
ARMS = ["geom+H0", "geom+H1", "geom+H2", "geom+burial"]
KEEP = {f"geom+{dim}": GEOM_COLS + DIM_COLS[dim] for dim in DIM_MASKS}
for a in ARMS: print(f"  {a:<12} {len(KEEP[a])} kept side columns (17 geometry + {len(KEEP[a])-17} topology)")

ORIG_X = [d.x.clone() for d in dataset]

def apply_arm(a):
    for d, x0 in zip(dataset, ORIG_X):
        x = x0.clone(); mask = torch.ones(x.size(1), dtype=torch.bool)
        mask[N_PRE:] = False; mask[KEEP[a]] = True
        x[:, ~mask] = 0.0
        d.x = x

folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=1337).split(np.arange(len(dataset))))
records = []; t0 = time.time()
for arm in ARMS:
    apply_arm(arm)
    for fi in FOLDS_TO_RUN:
        tr_idx, te_idx = folds[fi]
        for seed in SEEDS:
            rng = np.random.RandomState(500 + seed)
            perm = rng.permutation(tr_idx)
            nv = max(1, int(0.15 * len(perm)))
            va_ds = [dataset[i] for i in perm[:nv]]; tr_ds = [dataset[i] for i in perm[nv:]]
            te_ds = [dataset[i] for i in te_idx]
            torch.manual_seed(seed); np.random.seed(seed)
            m = Net(N_PRE, SIDE_DIM).to(DEVICE)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
            tl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True); vl = DataLoader(va_ds, batch_size=BATCH)
            best, state, bad = -1, None, 0
            for ep in range(EPOCHS):
                run_epoch(m, tl, opt); v = run_epoch(m, vl); sch.step()
                if v["pr"] > best: best, state, bad = v["pr"], {k: t.clone() for k, t in m.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= PATIENCE: break
            m.load_state_dict(state)
            te = run_epoch(m, DataLoader(te_ds, batch_size=BATCH))
            records.append({"arm": arm, "fold": fi, "seed": seed, "val_pr": best, "test_pr": float(te["pr"]),
                            "test_roc": float(te["roc"]), "n_test": int(len(te["y"])), "n_pos": int(te["y"].sum())})
        r = records[-1]
        print(f"  {arm:<12} fold {fi:<2} | val PR {r['val_pr']:.3f} | test PR {r['test_pr']:.3f} ROC {r['test_roc']:.3f} [{(time.time()-t0)/60:.0f}m]", flush=True)
print(f"\n{len(records)} runs in {(time.time()-t0)/60:.1f} min")
''' + f'json.dump({{"records": records, "folds": FOLDS_TO_RUN}}, open("/kaggle/working/phase285{TAG}_records.json","w"), indent=2, default=float)')
md("## Cell 5 — Partial summary")
code(r'''
import pandas as pd
df = pd.DataFrame(records)
print(df.groupby("arm")[["test_pr", "test_roc"]].agg(["mean", "std"]))
''')

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT_DIR.mkdir(parents=True, exist_ok=True)
o = OUT_DIR / f"phase285{TAG}_topo_decomposition_650m.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")

meta = {
  "id": f"blackpearls1/phase285{TAG}-topo-decomposition-650m",
  "title": f"Phase285{TAG} Topo Decomposition 650M",
  "code_file": o.name,
  "language": "python", "kernel_type": "notebook", "is_private": True,
  "enable_gpu": True, "machine_shape": "NvidiaTeslaT4", "enable_tpu": False, "enable_internet": True,
  "dataset_sources": [], "competition_sources": [],
  "kernel_sources": ["blackpearls1/phase238-esm650m-ingestion", "blackpearls1/phase229-geometry-main-pool",
                      "blackpearls1/tda-egnn-phase6b-topology-scale"],
}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {OUT_DIR/'kernel-metadata.json'}")
