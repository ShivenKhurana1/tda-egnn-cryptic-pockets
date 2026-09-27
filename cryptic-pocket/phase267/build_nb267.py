#!/usr/bin/env python3
"""Phase 267: inputs for localization-candidate validation (#3) and the P2Rank blend (#4) on the OFFICIAL CryptoBench split (Kaggle CPU).
For every val+test chain (60 + 182): fetch the apo structure from RCSB, verify it aligns to the Phase 48 graph (node count and centred CA positions),
run P2Rank 2.5.1 (default model, whole structure, as in Phase 226), and align P2Rank's per-residue score/z-score/probability to graph node order and
its pockets on the target chain (centres converted to the graph frame). Saves everything the local analysis needs."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
GEOM_SRC = (HERE.parent / "phase227_geometry.py").read_text()
C = []
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})
code("!pip install -q torch-geometric==2.6.1 biopython scipy\n!apt-get -qq update > /dev/null 2>&1 && apt-get -qq install -y default-jre-headless > /dev/null 2>&1; java -version 2>&1 | head -1")
code(GEOM_SRC)
code(r'''
import os, re, time, subprocess, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, torch
from joblib import Parallel, delayed

def locate(f):
    hits = []
    for root in (Path('/kaggle/input'), Path('/kaggle/working'), Path('.')):
        if root.exists(): hits.extend(root.rglob(f))
    if not hits: raise FileNotFoundError(f)
    return max(hits, key=lambda p: p.stat().st_size)

blob = torch.load(locate('phase48_cryptobench_graphs_esm3b.pt'), map_location='cpu', weights_only=False)
graphs = {}
for d in blob['dataset']:
    if str(d.split) in ('val', 'test'):
        graphs[(str(d.pdb_id).lower(), str(d.chain_id))] = dict(n=int(d.num_nodes), pos=d.pos.numpy(), split=str(d.split), tc=d.true_center.numpy().reshape(-1), y=d.y.numpy().astype(np.int8))
del blob
print(len(graphs), 'val+test chains:', {s: sum(g['split'] == s for g in graphs.values()) for s in ('val', 'test')})

WORK = Path('/kaggle/working'); CIF = WORK / 'apo_cifs'; CIF.mkdir(exist_ok=True)
def fetch(pid):
    dst = CIF / f'{pid}.cif'
    if dst.exists() and dst.stat().st_size > 1000: return pid, True
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f'https://files.rcsb.org/download/{pid.upper()}.cif', timeout=60) as r: dst.write_bytes(r.read())
            return pid, True
        except Exception: time.sleep(2 * (attempt + 1))
    return pid, False
pids = sorted({k[0] for k in graphs})
res = Parallel(n_jobs=8, backend='threading')(delayed(fetch)(p) for p in pids)
missing = [p for p, ok in res if not ok]; print(f'fetched {len(pids)-len(missing)}/{len(pids)}; missing {missing}'); assert not missing

# verify alignment + record residue ids in node order
info = {}
for key, g in sorted(graphs.items()):
    try:
        ch = load_chain(CIF / f'{key[0]}.cif', key[1], key[0]); rs = list(ch)
        ca = np.array([r['CA'].coord for r in rs], float)
        assert len(rs) == g['n'], f"node count {len(rs)} vs {g['n']}"
        err = float(np.abs((ca - ca.mean(0)) - g['pos']).max()); assert err < 1e-3, f'CA mismatch {err:.4f}'
        info[key] = dict(shift=ca.mean(0), ids=[(r.id[1], (r.id[2] or '').strip()) for r in rs])
    except Exception as e:
        print('ALIGN FAIL', key, type(e).__name__, e)
print(f'aligned {len(info)}/{len(graphs)} chains')
assert len(info) >= 0.95 * len(graphs)
''')
code(r'''
subprocess.run("curl -sL https://github.com/rdk/p2rank/releases/download/2.5.1/p2rank_2.5.1.tar.gz -o /tmp/p2.tgz && tar xzf /tmp/p2.tgz -C /tmp", shell=True, check=True, timeout=1200)
prank = next(Path('/tmp').glob('p2rank_*/prank')); os.chmod(prank, 0o755)
ds = WORK / 'val_test.ds'
ds.write_text('HEADER: protein\n' + ''.join(f'{CIF}/{p}.cif\n' for p in sorted({k[0] for k in info})))
OUT = WORK / 'p2rank_out'
t0 = time.time(); r = subprocess.run([str(prank), 'predict', str(ds), '-o', str(OUT), '-threads', str(os.cpu_count() or 2)], capture_output=True, text=True, timeout=10800)
print('prank rc', r.returncode, f'{(time.time()-t0)/60:.1f} min'); print(r.stdout[-600:]); print(r.stderr[-600:])
''')
code(r'''
label_re = re.compile(r"^(-?\d+)([A-Za-z]?)$")
out = {}; fails = []
for key, inf in info.items():
    pdb, chain = key; g = graphs[key]
    try:
        rd = pd.read_csv(OUT / f'{pdb}.cif_residues.csv', skipinitialspace=True); rd.columns = [c.strip() for c in rd.columns]
        rd['chain'] = rd['chain'].astype(str).str.strip(); rd = rd[rd.chain == chain]
        lab = rd['residue_label'].astype(str).str.strip().str.extract(label_re)
        rd = rd.assign(resnum=pd.to_numeric(lab[0]), icode=lab[1].fillna(''))
        idx = {(int(a), b): i for i, (a, b) in enumerate(zip(rd.resnum, rd.icode))}
        rows = [idx.get((int(a), b)) for a, b in inf['ids']]
        miss = sum(x is None for x in rows)
        arr = np.full((g['n'], 3), np.nan, np.float32)
        for i, x in enumerate(rows):
            if x is not None: arr[i] = rd.iloc[x][['score', 'zscore', 'probability']].astype(float).values
        pk = pd.read_csv(OUT / f'{pdb}.cif_predictions.csv', skipinitialspace=True); pk.columns = [c.strip() for c in pk.columns]
        pockets = []
        for _, q in pk.iterrows():
            ids = str(q['residue_ids']).split(); frac = np.mean([i.split('_')[0] == chain for i in ids]) if ids else 0.0
            pockets.append(dict(rank=int(q['rank']), score=float(q['score']), prob=float(q['probability']), frac_on_chain=float(frac),
                                center=np.array([q['center_x'], q['center_y'], q['center_z']], float) - inf['shift']))
        out[key] = dict(split=g['split'], pos=g['pos'], tc=g['tc'], y=g['y'], p2=arr, n_unmatched=miss, pockets=pockets)
    except Exception as e:
        fails.append((key, f'{type(e).__name__}: {e}'))
print(f'P2Rank aligned for {len(out)}/{len(info)} chains; failures {len(fails)}; chains with unmatched residues {sum(v["n_unmatched"]>0 for v in out.values())}; total unmatched nodes {sum(v["n_unmatched"] for v in out.values())} of {sum(v["p2"].shape[0] for v in out.values())}')
for f in fails[:20]: print('  FAIL', f)
assert len(out) >= 0.9 * len(graphs)
torch.save(out, WORK / 'phase267_p2rank_and_coords.pt')
import shutil; shutil.rmtree(CIF, ignore_errors=True); shutil.rmtree(OUT, ignore_errors=True)
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "cpu"; d.mkdir(exist_ok=True)
(d / "phase267_p2rank_coords.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "blackpearls1/phase267-p2rank-coords", "title": "Phase267 P2rank Coords", "code_file": "phase267_p2rank_coords.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": ["blackpearls1/tda-egnn-phase48-cryptobench-esm3b"]}, indent=2) + "\n")
print("wrote", d)
