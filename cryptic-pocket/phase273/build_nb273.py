#!/usr/bin/env python3
"""Phase 273: compute the per-residue chemistry feature block (Phase 273_chem) for every pretraining-pool chain (Phase 96 blob, uses .seq) and every CryptoBench chain
(Phase 48 blob, sequence decoded losslessly from the one-hot block, Phase 265's method). CPU kernel."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = (HERE / "phase273_chem.py").read_text()
C = []
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})
code("!pip install -q torch-geometric==2.6.1 joblib")
code(SRC)
code(r'''
import os, time
from pathlib import Path
import numpy as np, torch
from joblib import Parallel, delayed

def locate(f):
    c = []
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

def load(blob_name, key_fn, seq_fn):
    blob = torch.load(locate(blob_name), map_location="cpu", weights_only=False)
    out = {key_fn(d): (seq_fn(d), d.pos.numpy().astype(np.float64)) for d in blob["dataset"]}
    del blob; return out

def cb_seq(d):
    idx = d.x[:, :21].float().argmax(1); assert bool((d.x[:, :21].float().sum(1) == 1).all())
    return "".join(AA_ALPHABET[i] for i in idx.tolist())

pool = load("phase96_graphs_esm3b.pt", lambda d: (str(d.pdb_id), str(d.chain_id)), lambda d: d.seq)
cb = load("phase48_cryptobench_graphs_esm3b.pt", lambda d: (str(d.pdb_id).lower(), str(d.chain_id)), cb_seq)
print("pool chains", len(pool), "CryptoBench chains", len(cb), "| residues", sum(len(p) for _, p in pool.values()), sum(len(p) for _, p in cb.values()))

def run(item):
    k, (seq, p) = item
    try:
        assert len(seq) == len(p), f"seq/pos length mismatch {len(seq)} vs {len(p)}"
        return k, chem_features(seq, p - p.mean(0)), None
    except Exception as e: return k, None, f"{type(e).__name__}: {e}"

out = {}
for name, D in (("cb", cb), ("pool", pool)):
    t0 = time.time()
    res = Parallel(n_jobs=os.cpu_count() or 2, verbose=0, batch_size=16)(delayed(run)(it) for it in D.items())
    ok = {k: f for k, f, e in res if f is not None}; bad = [(k, e) for k, f, e in res if f is None]
    print(f"{name}: {len(ok)}/{len(D)} chains in {(time.time()-t0)/60:.1f} min on {os.cpu_count()} cores; failures {len(bad)}", bad[:5])
    assert len(bad) == 0
    for k, f in ok.items(): assert f.shape == (len(D[k][1]), CHEM_DIM) and np.isfinite(f).all(), k
    out[name] = ok
allf = np.concatenate(list(out["pool"].values()), 0)
import pandas as pd
print(pd.DataFrame(allf, columns=NAMES).describe().T[["mean", "std", "min", "max"]].round(3).to_string())
torch.save(dict(names=NAMES, pool=out["pool"], cb=out["cb"]), "/kaggle/working/phase273_chem_features.pt")
print("saved")
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "cpu"; d.mkdir(exist_ok=True)
(d / "phase273_chem_features.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "blackpearls1/phase273-chem-features", "title": "Phase273 Chem Features", "code_file": "phase273_chem_features.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [],
    "kernel_sources": ["blackpearls1/tda-egnn-phase96-esm-dihedral-esm3b-v3", "blackpearls1/tda-egnn-phase48-cryptobench-esm3b"]}, indent=2) + "\n")
print("wrote", d)
