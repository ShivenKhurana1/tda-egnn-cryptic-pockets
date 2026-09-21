#!/usr/bin/env python3
"""Phase 235: plain-geometry features for the COACH420/HOLO4K benchmark chains cached by Phase 15 (Kaggle CPU)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEOM_SRC = (HERE.parent / "phase227_geometry.py").read_text()

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r'''
# Phase 235 — Plain-geometry features for the COACH420 / HOLO4K benchmark chains

Prerequisite for the localization control (Phase 236). Phase 22 scored topology-vs-no-topology localization on the
1,149 leakage-clean benchmark units whose raw features Phase 15 cached (`phase15_raw_features.pkl`). To run the same
comparison with a geometry arm, each cached chain needs the same 17 geometry features used for training (Phase 229).

Each chain is re-fetched from RCSB (`.pdb.gz`, exactly as Phase 15 did), parsed with the identical residue filter, and
accepted only if its residue count equals the cached feature rows AND its CA coordinates equal the cached raw CA
coordinates (the cache stores `ca` in the original frame, so this is a direct, uncentred comparison, max abs error < 1e-3 A).
''')
md("## Cell 1 — Install")
code("!pip install -q biopython scipy")
md("## Cell 2 — Geometry feature code (identical to cryptic-pocket/phase227_geometry.py)")
code(GEOM_SRC)
md("## Cell 3 — Load Phase 15's raw feature cache (keys and CA coordinates only are used)")
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

with open(locate('phase15_raw_features.pkl'), 'rb') as fh:
    raw = pickle.load(fh)
entries = {k: dict(n=int(v['x'].shape[0]), ca=np.asarray(v['ca'], np.float32)) for k, v in raw.items()}
del raw
print(len(entries), 'cached chains;', collections.Counter(k[0] for k in entries), 'by benchmark')
by_pid = collections.defaultdict(list)
for (src, pid, cid) in entries: by_pid[pid].append((src, pid, cid))
print(len(by_pid), 'structures')
''')
md("## Cell 4 — Fetch structures")
code(r'''
WORK = Path('/kaggle/working'); RAW = WORK / 'raw'; RAW.mkdir(exist_ok=True)

def fetch(pid):
    dst = RAW / f'{pid}.pdb.gz'
    if dst.exists() and dst.stat().st_size > 500: return pid, True
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f'https://files.rcsb.org/download/{pid}.pdb.gz', timeout=60) as r:
                b = r.read()
            if len(b) > 500:
                dst.write_bytes(b); return pid, True
        except Exception:
            time.sleep(2 * (attempt + 1))
    return pid, False

t0 = time.time()
res = Parallel(n_jobs=16, backend='threading')(delayed(fetch)(p) for p in sorted(by_pid))
missing = [p for p, ok in res if not ok]
print(f'fetched {len(by_pid) - len(missing)}/{len(by_pid)} in {(time.time()-t0)/60:.1f} min; missing: {missing[:20]}')
''')
md("## Cell 5 — Compute geometry per chain and verify alignment")
code(r'''
def one_structure(pid):
    out = {}
    for key in by_pid[pid]:
        src, _, cid = key
        e = entries[key]
        path = RAW / f'{pid}.pdb.gz'
        if not path.exists():
            out[key] = (None, 'structure could not be fetched'); continue
        try:
            chain = load_chain(path, cid, pid)
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

pids = sorted(by_pid)
t0 = time.time()
results = Parallel(n_jobs=os.cpu_count() or 2, verbose=5)(delayed(one_structure)(p) for p in pids)
feats, fail = {}, {}
for r in results:
    for k, (f, msg) in r.items():
        if f is None: fail[k] = msg
        else: feats[k] = f
print(f'done in {(time.time()-t0)/60:.1f} min on {os.cpu_count()} cores; accepted {len(feats)}/{len(entries)}; failed {len(fail)}')
print(collections.Counter(m.split(":")[0][:40] for m in fail.values()))
for k, m in list(fail.items())[:25]: print('  FAIL', k, m)
''')
md("## Cell 6 — Save")
code(r'''
torch.save(dict(names=NAMES, feats=feats, fail=fail, n_total=len(entries)), WORK / 'phase235_benchmark_geometry.pt')
import shutil; shutil.rmtree(RAW, ignore_errors=True)
print('saved -> phase235_benchmark_geometry.pt')
assert len(fail) <= 0.02 * len(entries), 'more than 2% of benchmark chains failed alignment -- investigate before Phase 236'
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
o = HERE / "phase235_benchmark_geometry.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
