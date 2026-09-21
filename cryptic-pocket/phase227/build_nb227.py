#!/usr/bin/env python3
"""Phase 227: compute plain-geometry per-residue features for all 942 CryptoBench apo chains (Kaggle CPU)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEOM_SRC = (HERE.parent / "phase227_geometry.py").read_text()

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r'''
# Phase 227 — Plain-geometry control features (all 942 CryptoBench apo chains)

Phase 164 showed a 2-feature burial baseline (heavy-atom count within 12 Å + gyration) recovers most of the
TDA gain. This notebook builds a much richer, deliberately strong **non-topological** per-residue feature set
(neighbour counts at 6 radii, relative SASA, half-sphere exposure, local shape eigen-ratios, ray buriedness,
convex-hull depth, residue depth) so the next notebook (Phase 228) can ask whether topology adds anything
on top of it.

Every chain's apo structure is re-fetched from RCSB and re-parsed with the ingestion's residue filter; a chain
is accepted only if its residue count equals the graph's node count AND its centred CA coordinates match the
graph's stored positions (max abs error < 1e-3 A). Any mismatch is reported, never silently kept.
''')

md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1 biopython scipy")

md("## Cell 2 — Geometry feature code (identical to cryptic-pocket/phase227_geometry.py)")
code(GEOM_SRC)

md("## Cell 3 — Load graph blob (only ids, node counts and positions are used)")
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

blob_path = locate('phase159_cryptobench_anmmsf_graphs.pt')
blob = torch.load(blob_path, map_location='cpu', weights_only=False)
graphs = {(str(d.pdb_id).lower(), str(d.chain_id)): dict(n=int(d.num_nodes), pos=d.pos.numpy(), split=d.split) for d in blob['dataset']}
print(len(graphs), 'graphs;', {s: sum(g['split'] == s for g in graphs.values()) for s in ('train', 'val', 'test')})
del blob
''')

md("## Cell 4 — Fetch apo structures from RCSB")
code(r'''
WORK = Path('/kaggle/working'); CIF = WORK / 'apo_cifs'; CIF.mkdir(exist_ok=True)

def fetch(pid):
    dst = CIF / f'{pid}.cif'
    if dst.exists() and dst.stat().st_size > 1000: return pid, True
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f'https://files.rcsb.org/download/{pid.upper()}.cif', timeout=60) as r:
                dst.write_bytes(r.read())
            return pid, True
        except Exception:
            time.sleep(2 * (attempt + 1))
    return pid, False

pids = sorted({k[0] for k in graphs})
t0 = time.time()
res = Parallel(n_jobs=8, backend='threading')(delayed(fetch)(p) for p in pids)
missing = [p for p, ok in res if not ok]
print(f'fetched {len(pids) - len(missing)}/{len(pids)} structures in {(time.time()-t0)/60:.1f} min; missing: {missing}')
assert not missing, 'some structures could not be fetched'
''')

md("## Cell 5 — Compute geometry, verify alignment to the graph nodes")
code(r'''
def one(key):
    pid, cid = key
    try:
        chain = load_chain(CIF / f'{pid}.cif', cid, pid)
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
for k, m in list(fail.items())[:30]: print('  FAIL', k, m)
assert not fail, 'alignment/feature failures -- do not proceed to Phase 228 until explained'
''')

md("## Cell 6 — Save")
code(r'''
train_rows = np.concatenate([feats[k] for k in keys if graphs[k]['split'] == 'train'], 0)
stats = dict(mean=train_rows.mean(0), std=np.maximum(train_rows.std(0), 1e-6))
torch.save(dict(names=NAMES, feats=feats, train_mean=stats['mean'], train_std=stats['std'], n_chains=len(feats)),
           WORK / 'phase227_geometry_features.pt')
import pandas as pd
allrows = np.concatenate([feats[k] for k in keys], 0)
print(pd.DataFrame(allrows, columns=NAMES).describe().T[['mean', 'std', 'min', 'max']].round(3).to_string())
print('saved -> phase227_geometry_features.pt')
import shutil; shutil.rmtree(CIF, ignore_errors=True)
''')

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
o = HERE / "phase227_geometry_ingestion.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
