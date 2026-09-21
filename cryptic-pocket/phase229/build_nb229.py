#!/usr/bin/env python3
"""Phase 229: geometry features for the 2,056 main-pipeline chains (the Phase 10/11 pool). Kaggle CPU."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEOM_SRC = (HERE.parent / "phase227_geometry.py").read_text()

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r'''
# Phase 229 — Plain-geometry features for the main-pipeline pool (2,056 chains)

CryptoBench Phase 228 found topology adds nothing measurable beyond 17 plain geometry features. The project's
*headline* topology result (Phase 6c / Phase 11: +0.032 to +0.042 PR-AUC beyond crude burial, 10/10 folds) was
tested only against a 2-feature burial baseline. This notebook builds the same 17 geometry features for the
2,056-chain pool so Phase 230 can run the same control on the main pipeline.

Each chain's structure is re-fetched from RCSB (`.pdb.gz`, as in Phase 11) and accepted only if the residue
count equals the graph's node count AND the centred CA coordinates match the graph's stored positions.
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1 biopython scipy")
md("## Cell 2 — Geometry feature code (identical to cryptic-pocket/phase227_geometry.py)")
code(GEOM_SRC)
md("## Cell 3 — Load the Phase 10 graphs (ids, node counts, positions only)")
code(r'''
import os, time, json, urllib.request, urllib.error
from pathlib import Path
import numpy as np
import torch
from joblib import Parallel, delayed

def locate(f):
    hits = []
    for root in (Path('/kaggle/input'), Path('/kaggle/working'), Path('.')):
        if root.exists(): hits.extend(root.rglob(f))
    if not hits: raise FileNotFoundError(f'Could not find {f}')
    return max(hits, key=lambda p: p.stat().st_size)

blob = torch.load(locate('phase10_graphs.pt'), map_location='cpu', weights_only=False)
graphs = {}
for d in blob['dataset']:
    k = (str(d.pdb_id), str(d.chain_id))
    assert k not in graphs, f'duplicate chain {k}'
    graphs[k] = dict(n=int(d.num_nodes), pos=d.pos.numpy())
print(len(graphs), 'chains')
del blob
''')
md("## Cell 4 — Fetch structures (.pdb.gz)")
code(r'''
WORK = Path('/kaggle/working'); RAW = WORK / 'raw'; RAW.mkdir(exist_ok=True)

def fetch(pid):
    # legacy PDB format first (as Phase 11 did); large / newer entries only exist as mmCIF
    for name, url in ((f'{pid}.pdb.gz', f'https://files.rcsb.org/download/{pid}.pdb.gz'),
                      (f'{pid}.cif', f'https://files.rcsb.org/download/{pid}.cif')):
        dst = RAW / name
        if dst.exists() and dst.stat().st_size > 500: return pid, name
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    b = r.read()
                if len(b) > 500:
                    dst.write_bytes(b); return pid, name
                break
            except urllib.error.HTTPError as e:
                if e.code == 404: break
                time.sleep(2 * (attempt + 1))
            except Exception:
                time.sleep(2 * (attempt + 1))
    return pid, None

pids = sorted({k[0] for k in graphs})
t0 = time.time()
res = Parallel(n_jobs=16, backend='threading')(delayed(fetch)(p) for p in pids)
FILES = {p: n for p, n in res if n}
missing = [p for p, n in res if not n]
print(f'fetched {len(FILES)}/{len(pids)} in {(time.time()-t0)/60:.1f} min ({sum(n.endswith(".cif") for n in FILES.values())} as mmCIF); missing: {missing[:20]}')
''')
md("## Cell 5 — Compute geometry and verify alignment")
code(r'''
def one(key):
    pid, cid = key
    if pid not in FILES:
        return key, None, 'structure could not be fetched'
    try:
        chain = load_chain(RAW / FILES[pid], cid, pid)
        feats, ca = geometry_for_chain(chain)
    except Exception as e:
        return key, None, f'error: {type(e).__name__}: {e}'
    g = graphs[key]
    if len(ca) != g['n']:
        return key, None, f'node count mismatch: structure {len(ca)} vs graph {g["n"]}'
    err = float(np.abs((ca - ca.mean(0)) - g['pos']).max())
    if err > 1e-3:
        return key, None, f'CA mismatch max {err:.4f} A'
    if not np.isfinite(feats).all():
        return key, None, 'non-finite features'
    return key, feats.astype(np.float32), 'ok'

keys = sorted(graphs)
t0 = time.time()
out = Parallel(n_jobs=os.cpu_count() or 2, verbose=5)(delayed(one)(k) for k in keys)
print(f'done in {(time.time()-t0)/60:.1f} min on {os.cpu_count()} cores')
feats = {k: f for k, f, msg in out if f is not None}
fail = {k: msg for k, f, msg in out if f is None}
print(f'accepted {len(feats)}/{len(keys)}; failed {len(fail)}')
for k, m in list(fail.items())[:40]: print('  FAIL', k, m)
import collections
print(collections.Counter(m.split(':')[0][:30] for m in fail.values()))
''')
md("## Cell 6 — Save (chains that failed alignment are recorded, not silently dropped)")
code(r'''
import pandas as pd
allrows = np.concatenate([feats[k] for k in keys if k in feats], 0)
torch.save(dict(names=NAMES, feats=feats, fail={k: v for k, v in fail.items()},
                mean=allrows.mean(0), std=np.maximum(allrows.std(0), 1e-6), n_chains=len(feats), n_total=len(keys)),
           WORK / 'phase229_geometry_features.pt')
print(pd.DataFrame(allrows, columns=NAMES).describe().T[['mean', 'std', 'min', 'max']].round(3).to_string())
print('saved -> phase229_geometry_features.pt')
import shutil; shutil.rmtree(RAW, ignore_errors=True)
assert len(fail) <= 0.02 * len(keys), 'more than 2% of chains failed alignment -- investigate before Phase 230'
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
o = HERE / "phase229_geometry_main_pool.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
