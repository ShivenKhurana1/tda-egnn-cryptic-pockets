#!/usr/bin/env python3
"""Phase 252: plain-geometry features for the COACH420 benchmark chains cached by Phase 53 (scale-3 pipeline)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEOM_SRC = (HERE.parent / "phase227_geometry.py").read_text()

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r'''
# Phase 252 — Plain-geometry features for the scale-3 COACH420 benchmark chains

Phase 53 featurized the full COACH420 benchmark for the scale-3 pipeline and cached the raw features
(`phase53_raw_features.pkl`, keyed `(pdb_id, chain)`). This computes the 17 geometry features for each cached chain,
accepting a chain only if its residue count matches the cached rows AND its CA coordinates equal the cached raw `ca`
(original frame, uncentred; max abs error < 1e-3 A).
''')
md("## Cell 1 — Install")
code("!pip install -q biopython scipy")
md("## Cell 2 — Geometry feature code (identical to cryptic-pocket/phase227_geometry.py)")
code(GEOM_SRC)
md("## Cell 3 — Load Phase 53's benchmark cache (keys and CA coordinates only)")
code(r'''
import os, time, json, pickle, urllib.request, urllib.error, collections
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

with open(locate('phase53_raw_features.pkl'), 'rb') as fh:
    raw = pickle.load(fh)
entries = {k: dict(n=int(v['x'].shape[0]), ca=np.asarray(v['ca'], np.float32)) for k, v in raw.items()}
print('sample key:', next(iter(raw)), '| x width', next(iter(raw.values()))['x'].shape[1])
del raw
by_pid = collections.defaultdict(list)
for key in entries: by_pid[key[0]].append(key)
print(len(entries), 'cached chains in', len(by_pid), 'structures')
''')
md("## Cell 4 — Fetch structures (.pdb.gz, mmCIF fallback)")
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

t0 = time.time()
res = Parallel(n_jobs=16, backend='threading')(delayed(fetch)(p) for p in sorted(by_pid))
FILES = {p: n for p, n in res if n}
print(f'fetched {len(FILES)}/{len(by_pid)} in {(time.time()-t0)/60:.1f} min; missing: {[p for p, n in res if not n][:20]}')
''')
md("## Cell 5 — Compute geometry and verify alignment")
code(r'''
def one_structure(pid):
    out = {}
    for key in by_pid[pid]:
        e = entries[key]; cid = key[1]
        if pid not in FILES:
            out[key] = (None, 'structure could not be fetched'); continue
        try:
            chain = load_chain(RAW / FILES[pid], cid, pid)
            feats, ca = geometry_for_chain(chain)
        except Exception as ex:
            out[key] = (None, f'error: {type(ex).__name__}: {ex}'); continue
        if len(ca) != e['n']:
            out[key] = (None, f'node count mismatch: structure {len(ca)} vs cache {e["n"]}'); continue
        err = float(np.abs(ca.astype(np.float32) - e['ca']).max())
        if err > 1e-3:
            out[key] = (None, f'CA mismatch max {err:.4f} A'); continue
        if not np.isfinite(feats).all():
            out[key] = (None, 'non-finite features'); continue
        out[key] = (feats.astype(np.float32), 'ok')
    return out

t0 = time.time()
results = Parallel(n_jobs=os.cpu_count() or 2, verbose=5)(delayed(one_structure)(p) for p in sorted(by_pid))
feats, fail = {}, {}
for r in results:
    for k, (f, msg) in r.items():
        if f is None: fail[k] = msg
        else: feats[k] = f
print(f'done in {(time.time()-t0)/60:.1f} min; accepted {len(feats)}/{len(entries)}; failed {len(fail)}')
print(collections.Counter(m.split(":")[0][:40] for m in fail.values()))
for k, m in list(fail.items())[:25]: print('  FAIL', k, m)
''')
md("## Cell 6 — Save")
code(r'''
torch.save(dict(names=NAMES, feats=feats, fail=fail, n_total=len(entries)), WORK / 'phase252_coach_geometry.pt')
import shutil; shutil.rmtree(RAW, ignore_errors=True)
print('saved -> phase252_coach_geometry.pt')
assert len(fail) <= 0.02 * len(entries), 'more than 2% of benchmark chains failed alignment -- investigate'
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
o = HERE / "phase252_coach_geometry.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
meta = {"id": "blackpearls1/phase252-coach-geometry", "title": "Phase252 Coach Geometry", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase53-scaled3-coach420"]}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
