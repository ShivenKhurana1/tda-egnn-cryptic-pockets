#!/usr/bin/env python3
"""Phase 270: compute the protein-movement (elastic-network) feature block for every pretraining-pool chain (Phase 96 blob) and every CryptoBench chain (Phase 48 blob).
CPU kernel; features come from the graphs' stored C-alpha positions only (see phase270_flex.py)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = (HERE / "phase270_flex.py").read_text()
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

def positions(blob_name, key_fn):
    blob = torch.load(locate(blob_name), map_location="cpu", weights_only=False)
    out = {key_fn(d): d.pos.numpy().astype(np.float64) for d in blob["dataset"]}
    del blob; return out

pool = positions("phase96_graphs_esm3b.pt", lambda d: (str(d.pdb_id), str(d.chain_id)))
cb = positions("phase48_cryptobench_graphs_esm3b.pt", lambda d: (str(d.pdb_id).lower(), str(d.chain_id)))
print("pool chains", len(pool), "CryptoBench chains", len(cb), "| residues", sum(len(v) for v in pool.values()), sum(len(v) for v in cb.values()))

def run(item):
    k, p = item
    try: return k, flex_features(p - p.mean(0)), None
    except Exception as e: return k, None, f"{type(e).__name__}: {e}"

out = {}
for name, D in (("cb", cb), ("pool", pool)):
    t0 = time.time()
    res = Parallel(n_jobs=os.cpu_count() or 2, verbose=0, batch_size=16)(delayed(run)(it) for it in D.items())
    ok = {k: f for k, f, e in res if f is not None}; bad = [(k, e) for k, f, e in res if f is None]
    print(f"{name}: {len(ok)}/{len(D)} chains in {(time.time()-t0)/60:.1f} min on {os.cpu_count()} cores; failures {len(bad)}", bad[:5])
    assert len(bad) == 0
    for k, f in ok.items(): assert f.shape == (len(D[k]), FLEX_DIM) and np.isfinite(f).all(), k
    out[name] = ok
allf = np.concatenate(list(out["pool"].values()), 0)
import pandas as pd
print(pd.DataFrame(allf, columns=NAMES).describe().T[["mean", "std", "min", "max"]].round(3).to_string())
torch.save(dict(names=NAMES, pool=out["pool"], cb=out["cb"]), "/kaggle/working/phase270_flex_features.pt")
print("saved")
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "cpu"; d.mkdir(exist_ok=True)
(d / "phase270_flex_features.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "blackpearls1/phase270-flex-features", "title": "Phase270 Flex Features", "code_file": "phase270_flex_features.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [],
    "kernel_sources": ["blackpearls1/tda-egnn-phase96-esm-dihedral-esm3b-v3", "blackpearls1/tda-egnn-phase48-cryptobench-esm3b"]}, indent=2) + "\n")
print("wrote", d)
