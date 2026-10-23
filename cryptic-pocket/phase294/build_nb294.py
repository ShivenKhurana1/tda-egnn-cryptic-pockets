#!/usr/bin/env python3
"""Phase 294: treat the label set as incomplete (Phase 291 recipe, official labels, 8 seeds per arm).

CryptoBench negatives are really "unlabelled": a pocket only becomes a label if someone crystallised a ligand there. Two ways to stop
punishing the model for finding real sites:
  ncmask: residues in CryptoBench's published NON-cryptic binding sites (noncryptic-pockets.json, OSF) that are not cryptic positives
          are removed from the loss (8,144 such residues = half the positive count; 16% of the current model's top-20 "false positives").
  pu15 / pu20: non-negative positive-unlabelled risk with every residue as the unlabelled sample and an assumed true positive rate of
          KAPPA x the observed one:  loss = PW * sum_P softplus(-z)/n + max(0, (sum_all softplus(z) - KAPPA * sum_P softplus(z))/n).
          KAPPA = 1 reduces EXACTLY to the current BCE with pos_weight PW, so the arms differ from base only by the PU assumption.
Base = Phase 292's in-kernel base (matches Phase 291 to within 8e-5 per residue). Everything else Phase 291 verbatim.
usage: build_nb294.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path

ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "294"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase291" / "build_nb291.py"), str(tmp)], check=True, capture_output=True)
nb291 = json.load(open(tmp / "291" / "phase291_cb3b_official_labels.ipynb"))
src = lambda i: nb291["cells"][i]["source"] if isinstance(nb291["cells"][i]["source"], str) else "".join(nb291["cells"][i]["source"])
assert "resnums_of" in src(6) and "class EGCL" in src(8) and "def run_epoch" in src(10)

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)

md('''
# Phase 294 - incomplete labels: mask known non-cryptic binding sites, or positive-unlabelled learning (Phase 291 recipe)

Arms `ncmask`, `pu15`, `pu20`, 8 seeds each, seed s fine-tuned from pretrained seed s. Official union labels for training targets and evaluation.
''')
for i in range(1, 7):
    (md if nb291["cells"][i]["cell_type"] == "markdown" else code)(src(i))
md("## Cell 4 - Known non-cryptic binding residues (CryptoBench noncryptic-pockets.json)")
code(r'''
NC = fetch_json("https://osf.io/download/37ztm/")
n_nc = 0
for d in dataset:
    pid, cid = str(d.pdb_id), str(d.chain_id)
    sel = {s.split("_", 1)[1] for e in NC.get(pid, []) if e["apo_chain"] == cid for s in e["apo_pocket_selection"] if s.split("_", 1)[0] == cid}
    rn = resnums_of(pid, cid)
    d.nc = torch.tensor([r in sel for r in rn], dtype=torch.bool) & (d.y == 0)
    n_nc += int(d.nc.sum())
for s in ("train", "val", "test"):
    print(f"  {s:5s} non-cryptic binder residues labelled negative: {sum(int(d.nc.sum()) for d in splits[s]):,}")
print("total", n_nc)
''')
md("## Cell 5 - Model (Phase 129, verbatim)")
code(src(8))
md("## Cell 6 - Training machinery (Phase 129/266c) with the label-incompleteness losses")
t = src(10)
t = rep(t, "            loss = crit(logits, b.y)\n", r'''            if MODE == "ncmask":
                keep = ~b.nc
                loss = nn.functional.binary_cross_entropy_with_logits(logits[keep], b.y[keep], pos_weight=crit.pos_weight)
            elif MODE == "pu":
                P_ = b.y > 0; n_ = logits.numel()
                rp = nn.functional.softplus(-logits)[P_].sum() / n_
                ln_ = nn.functional.softplus(logits)
                loss = POS_WEIGHT * rp + torch.clamp((ln_.sum() - KAPPA * ln_[P_].sum()) / n_, min=0.0)
            else:
                loss = crit(logits, b.y)
''')
code(t)
md("## Cell 7 - Arms x 8 seeds")
code(r'''
ARM = "full"
KEEP_SIDE = {"full": list(range(TOPO_DIM))}
m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE[ARM]] = 1.0
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_

@torch.no_grad()
def predict(model, split):
    out = {}
    for d in splits[split]:
        b = d.clone().to(DEVICE); logits, pc = model(b)
        out[(str(d.pdb_id).lower(), str(d.chain_id))] = dict(p=torch.sigmoid(logits).cpu().numpy().astype(np.float32), c=pc.cpu().numpy()[0],
            y=d.y.numpy().astype(np.int8), y_main=d.y_main.numpy().astype(np.int8), tc=d.true_center.numpy(), nc=d.nc.numpy())
    return out

# sanity: KAPPA = 1 PU loss equals the BCE loss on a real batch
_b = next(iter(DataLoader(splits["train"], batch_size=16))).to(DEVICE); _z = torch.randn(_b.y.numel(), device=DEVICE)
_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
_P = _b.y > 0; _ln = nn.functional.softplus(_z)
_pu = POS_WEIGHT * nn.functional.softplus(-_z)[_P].sum() / _z.numel() + (_ln.sum() - _ln[_P].sum()) / _z.numel()
print("PU(kappa=1) vs BCE:", float(_pu), float(_crit(_z, _b.y))); assert abs(float(_pu) - float(_crit(_z, _b.y))) < 1e-4

ARMS = [("ncmask", "ncmask", 1.0), ("pu15", "pu", 1.5), ("pu20", "pu", 2.0)]
BEST_LAMBDA = 0.1; t0 = time.time()
for name, MODE, KAPPA in ARMS:
    PREDS = {"arm": name, "MODE": MODE, "KAPPA": KAPPA, "seeds": {}}
    for seed in range(8):
        model, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, pretrained_path=locate(f"pretrain266_full_seed{seed}.pt"))
        PREDS["seeds"][seed] = {"val": predict(model, "val"), "test": predict(model, "test"), "val_pr": float(val_pr)}
        torch.save(PREDS, WORK / f"phase294_{name}_preds.pt")
        print(f"  [{name}] seed {seed}: val PR {val_pr:.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)
print("done")
''')
nb = {"cells": C, "metadata": nb291["metadata"], "nbformat": 4, "nbformat_minor": 5}
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase294_incomplete_labels.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = {"id": "blackpearls1/phase294-incomplete-labels", "title": "Phase294 Incomplete Labels", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase48-cryptobench-esm3b", "blackpearls1/phase227-geometry-ingestion", "blackpearls1/phase266b-pretrain-3b-clean"]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
