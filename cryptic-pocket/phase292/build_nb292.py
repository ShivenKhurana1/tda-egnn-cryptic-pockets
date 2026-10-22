#!/usr/bin/env python3
"""Phase 292: two never-tried training signals on the Phase 291 recipe (3B transfer, official CryptoBench labels, 8 seeds/arm).

  motion: an auxiliary head regresses each residue's REAL apo->holo movement (log1p heavy-atom RMSD after core superposition
          onto the main holo partner; compute_motion.py). Training-time only: nothing extra is needed at prediction time.
  rank:   a per-protein listwise loss, -mean over a chain's positives of log softmax(logits over that chain's residues),
          added to the BCE, so the model is rewarded for putting each protein's true pocket at the top of ITS list.
Arms: base (rerun, for clean pairing), mot02, mot05 (MU = 0.2 / 0.5), rank05, rank10 (NU = 0.5 / 1.0).
Everything else is Phase 291 verbatim (setup, relabel with the exact-reproduction assert, model, fit, checkpoints, seeds).
usage: build_nb292.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path

ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "292"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase291" / "build_nb291.py"), str(tmp)], check=True, capture_output=True)
nb291 = json.load(open(tmp / "291" / "phase291_cb3b_official_labels.ipynb"))
src = lambda i: nb291["cells"][i]["source"] if isinstance(nb291["cells"][i]["source"], str) else "".join(nb291["cells"][i]["source"])
assert "Relabel" in src(5) and "resnums_of" in src(6) and "class EGCL" in src(8) and "def run_epoch" in src(10)

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)

md('''
# Phase 292 - real apo->holo motion as an auxiliary target, and a per-protein ranking loss (Phase 291 recipe)

Arms: `base`, `mot02`, `mot05`, `rank05`, `rank10`, 8 seeds each, seed s fine-tuned from pretrained seed s. Official union labels.
''')
for i in range(1, 7):                                    # install, setup, relabel (Phase 291 verbatim)
    (md if nb291["cells"][i]["cell_type"] == "markdown" else code)(src(i))

md("## Cell 4 - Attach real apo->holo motion targets (training-time auxiliary target only)")
code(r'''
import math
MT = json.load(open(locate("motion_targets.json")))
cov = {s: [0, 0] for s in ("train", "val", "test")}; order_mismatch = 0
for d in dataset:
    pid, cid = str(d.pdb_id), str(d.chain_id)
    mot = torch.zeros(d.num_nodes); mask = torch.zeros(d.num_nodes, dtype=torch.bool)
    rec = MT.get(pid)
    if rec is not None and rec["chain"] == cid:
        rn = resnums_of(pid, cid)
        order_mismatch += int(rn != rec["resnums"])
        lut = {r: v for r, v in zip(rec["resnums"], rec["rmsd"])}
        for i, r in enumerate(rn):
            v = lut.get(r)
            if v is not None: mot[i] = math.log1p(min(v, 20.0)); mask[i] = True     # cap: floppy termini / domain swaps
    d.mot, d.mot_mask = mot, mask
    cov[d.split][0] += int(mask.sum()); cov[d.split][1] += d.num_nodes
print("chains whose residue list differs from the target file's:", order_mismatch)
for s, (a, b) in cov.items(): print(f"  {s:5s} residues with a motion target: {a:,}/{b:,} ({a/max(b,1):.1%})")
tr = splits["train"]
mp = torch.cat([d.mot[d.mot_mask & (d.y > 0)] for d in tr]); mn = torch.cat([d.mot[d.mot_mask & (d.y == 0)] for d in tr])
print(f"train mean log1p(RMSD): pocket {mp.mean():.3f} vs rest {mn.mean():.3f}")
''')
md("## Cell 5 - Model (Phase 129) + optional motion head")
m = src(8)
m = rep(m, "def __init__(s, n_pre, td, h=128, L=4, dr=0.1, use_loc_head=True):", "def __init__(s, n_pre, td, h=128, L=4, dr=0.1, use_loc_head=True, mot=False):")
m = rep(m, "        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))\n",
        "        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))\n"
        "        s.mot=mot; s.mot_out=None\n"
        "        if mot: s.mot_head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Linear(h//2,1))\n")
m = rep(m, "        logits = s.head(h).squeeze(-1)\n", "        logits = s.head(h).squeeze(-1)\n        s.mot_out = s.mot_head(h).squeeze(-1) if s.mot else None\n")
code(m)
md("## Cell 6 - Training machinery (Phase 129/266c) + motion and ranking loss terms")
t = src(10)
t = rep(t, "    huber = nn.SmoothL1Loss(beta=4.0)\n", "    huber = nn.SmoothL1Loss(beta=4.0)\n    mot_crit = nn.SmoothL1Loss(beta=0.5)\n")
t = rep(t, "            loss = crit(logits, b.y)\n", r'''            loss = crit(logits, b.y)
            if MU > 0:
                mk = b.mot_mask
                if mk.any(): loss = loss + MU * mot_crit(model.mot_out[mk], b.mot[mk])
            if NU > 0:
                bt = b.batch
                logp = torch.log(pyg_softmax(logits, bt) + 1e-12)
                npos = torch.zeros(int(bt.max()) + 1, device=logits.device).index_add_(0, bt, b.y)
                w = b.y / npos[bt].clamp(min=1)
                loss = loss + NU * (-(w * logp).sum() / (npos > 0).sum().clamp(min=1))
''')
t = rep(t, "LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=use_loc_head)", "LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=use_loc_head, mot=MU > 0)")
t = rep(t, "        assert not missing and not unexpected", "        missing = [k for k in missing if not k.startswith(\"mot_head\")]\n        assert not missing and not unexpected")
code(t)
md("## Cell 7 - Arms x 8 seeds; save probabilities, centres, labels, motion targets and predicted motion")
code(r'''
ARM = "full"
KEEP_SIDE = {"full": list(range(TOPO_DIM)), "geom": list(range(TOPO_DIM, SIDE_DIM)), "geom+topo": list(range(SIDE_DIM))}
m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE[ARM]] = 1.0
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_

@torch.no_grad()
def predict(model, split):
    out = {}
    for d in splits[split]:
        b = d.clone().to(DEVICE); logits, pc = model(b)
        r = dict(p=torch.sigmoid(logits).cpu().numpy().astype(np.float32), c=pc.cpu().numpy()[0], y=d.y.numpy().astype(np.int8),
                 y_main=d.y_main.numpy().astype(np.int8), tc=d.true_center.numpy(),
                 mot=d.mot.numpy().astype(np.float32), mot_mask=d.mot_mask.numpy())
        if model.mot: r["mot_pred"] = model.mot_out.cpu().numpy().astype(np.float32)
        out[(str(d.pdb_id).lower(), str(d.chain_id))] = r
    return out

ARMS = [("base", 0.0, 0.0), ("mot02", 0.2, 0.0), ("mot05", 0.5, 0.0), ("rank05", 0.0, 0.5), ("rank10", 0.0, 1.0)]
BEST_LAMBDA = 0.1; t0 = time.time()
for name, MU, NU in ARMS:
    PREDS = {"arm": name, "MU": MU, "NU": NU, "labels": "official_union", "pos_weight": POS_WEIGHT, "seeds": {}}
    for seed in range(8):
        ck = locate(f"pretrain266_full_seed{seed}.pt")
        model, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, pretrained_path=ck)
        PREDS["seeds"][seed] = {"val": predict(model, "val"), "test": predict(model, "test"), "val_pr": float(val_pr)}
        torch.save(PREDS, WORK / f"phase292_{name}_preds.pt")
        print(f"  [{name}] seed {seed}: val PR {val_pr:.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)
print("done")
''')
nb = {"cells": C, "metadata": nb291["metadata"], "nbformat": 4, "nbformat_minor": 5}
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase292_motion_rank.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = {"id": "blackpearls1/phase292-motion-rank", "title": "Phase292 Motion Rank", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False, "enable_internet": True, "dataset_sources": ["blackpearls1/phase292-motion-targets"], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase48-cryptobench-esm3b", "blackpearls1/phase227-geometry-ingestion", "blackpearls1/phase266b-pretrain-3b-clean"]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
