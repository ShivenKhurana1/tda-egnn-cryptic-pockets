#!/usr/bin/env python3
"""Phase 250: plain-geometry features for the 7,513-chain scale-3 pool (Phase 52 blob), one third per kernel.

usage: build_nb250.py <chunk 0|1|2> <output_dir>     (keys sorted, chunk = keys[chunk::3])"""
import json, sys
from pathlib import Path

CHUNK = int(sys.argv[1]); OUT_DIR = Path(sys.argv[2]).resolve()
TAG = "abc"[CHUNK]
HERE = Path(__file__).resolve().parent
GEOM_SRC = (HERE.parent / "phase227_geometry.py").read_text()

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase 250{TAG} — Plain-geometry features for the 7,513-chain scale-3 pool (chunk {CHUNK} of 3)

The geometry control has been run on the 2,055-chain pool (Phases 229-247). This builds the same 17 geometry features
for the round-3 pool (Phase 52's 7,513 chains) so the control can be run at that data scale. The pool is split three ways
(sorted chain keys, `keys[{CHUNK}::3]`) across parallel CPU kernels; the consumers merge the three outputs. Each chain is
re-fetched from RCSB and accepted only if its residue count and centred CA coordinates match the graph (1e-3 A).
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1 biopython scipy")
md("## Cell 2 — Geometry feature code (identical to cryptic-pocket/phase227_geometry.py)")
code(GEOM_SRC)
md("## Cell 3 — Load the Phase 52 graphs (ids, node counts, positions only) and select this chunk")
code(f'''
import os, time, json, urllib.request, urllib.error, collections
from pathlib import Path
import numpy as np
import torch
from joblib import Parallel, delayed

def locate(f):
    hits = []
    for root in (Path('/kaggle/input'), Path('/kaggle/working'), Path('.')):
        if root.exists(): hits.extend(root.rglob(f))
    if not hits: raise FileNotFoundError(f'Could not find {{f}}')
    return max(hits, key=lambda p: p.stat().st_size)

blob = torch.load(locate('phase52_graphs.pt'), map_location='cpu', weights_only=False)
all_graphs = {{}}
for d in blob['dataset']:
    k = (str(d.pdb_id), str(d.chain_id))
    assert k not in all_graphs, f'duplicate chain {{k}}'
    all_graphs[k] = dict(n=int(d.num_nodes), pos=d.pos.numpy())
del blob
keys = sorted(all_graphs)[{CHUNK}::3]
graphs = {{k: all_graphs[k] for k in keys}}
print(len(all_graphs), 'chains in the pool;', len(keys), 'in this chunk;', sum(g['n'] for g in graphs.values()), 'residues')
''')
md("## Cell 4 — Fetch structures (.pdb.gz, mmCIF fallback for newer/larger entries)")
code(r'''
WORK = Path('/kaggle/working'); RAW = WORK / 'raw'; RAW.mkdir(exist_ok=True)

def fetch(pid):
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

# longest chains first so the slowest tasks start early and finish together
order = sorted(keys, key=lambda k: -graphs[k]['n'])
t0 = time.time()
out = Parallel(n_jobs=os.cpu_count() or 2, verbose=5)(delayed(one)(k) for k in order)
print(f'done in {(time.time()-t0)/60:.1f} min on {os.cpu_count()} cores')
feats = {k: f for k, f, msg in out if f is not None}
fail = {k: msg for k, f, msg in out if f is None}
print(f'accepted {len(feats)}/{len(keys)}; failed {len(fail)}')
for k, m in list(fail.items())[:40]: print('  FAIL', k, m)
print(collections.Counter(m.split(':')[0][:30] for m in fail.values()))
''')
md("## Cell 6 — Save (failures recorded, not silently dropped)")
code(f'''
torch.save(dict(names=NAMES, feats=feats, fail=dict(fail), n_chains=len(feats), n_total=len(keys), chunk={CHUNK}),
           WORK / 'phase250{TAG}_geometry_scale3.pt')
print('saved -> phase250{TAG}_geometry_scale3.pt')
import shutil; shutil.rmtree(RAW, ignore_errors=True)
assert len(fail) <= 0.02 * len(keys), 'more than 2% of chains failed alignment -- investigate before consuming this'
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT_DIR.mkdir(parents=True, exist_ok=True)
o = OUT_DIR / f"phase250{TAG}_geometry_scale3.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
meta = {"id": f"blackpearls1/phase250{TAG}-geometry-scale3", "title": f"Phase250{TAG} Geometry Scale3",
        "code_file": o.name, "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase52-esm-dihedral-scale3"]}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
