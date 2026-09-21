#!/usr/bin/env python3
"""Generates one HOLO4K featurization CHUNK notebook. Phase100 got
CANCEL_ACKNOWLEDGED at 12hr with only 1005/1709 chains featurized (no
partial save -- the pickle only writes at the very end of the loop), so
the whole ~11-hour ESM2-3B featurization pass was lost. Splits it into
N_CHUNKS notebooks by eval_units index, each handling ~1/N_CHUNKS of the
1709 chains and saving its OWN partial raw-features pickle -- mirrors
the single-seed-per-notebook pattern already proven for CPU-session-limit
risk, applied here to structure count instead of seed count.
Usage: python3 build_nb100_chunk.py <CHUNK_IDX> <N_CHUNKS> <PHASE_NUM>"""
import json, pickle, sys
from pathlib import Path

CHUNK_IDX = int(sys.argv[1])
N_CHUNKS = int(sys.argv[2])
PHASE_NUM = sys.argv[3]

C24 = pickle.load(open("/tmp/p24_cells.pkl", "rb"))
s24 = lambda i: (C24[i]["source"] if isinstance(C24[i]["source"], str) else "".join(C24[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(f'''
# Phase {PHASE_NUM} — HOLO4K ESM2-3B Featurization: Chunk {CHUNK_IDX}/{N_CHUNKS}

Phase100 got CANCEL_ACKNOWLEDGED at the 12hr CPU cap with 1005/1709
chains featurized and nothing saved (the cache pickle only writes once,
at the end). This handles chunk {CHUNK_IDX} of {N_CHUNKS} (~1/{N_CHUNKS} of
the 1709 chains, well under the ~6.2hr this chunk size needs at the
observed ~0.65min/chain rate), saving its own partial pickle.
''')

md(r'''## Cell 0 — Install dependencies (a fresh kernel needs this too -- forgotten in v1, ModuleNotFoundError on Bio.PDB)''')
code('!pip install -q torch-geometric==2.6.1 gudhi==3.10.1 biopython==1.84 transformers==4.46.3 accelerate')

md(r'''## Cell 1 — Lightweight setup: Phase51's small topology blob only (no training blob, no ensemble checkpoints needed for featurization)''')
code(r'''
import csv as _csv
import gzip, json, os, pickle, shutil, subprocess, time, urllib.request, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

warnings.filterwarnings("ignore")

AA3_TO_1={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

blob6b = torch.load(locate("phase51_graphs.pt"), map_location="cpu", weights_only=False)
TOPO_NAMES = blob6b["topo_names"]
TOPO_DIM = len(TOPO_NAMES)
assert TOPO_DIM == 34, f"expected TOPO_DIM=34, got {TOPO_DIM}"
if "dataset" in blob6b: del blob6b["dataset"]

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
print(f"TOPO_DIM={TOPO_DIM}  DEVICE={DEVICE}")

WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)
RAW = Path("/tmp/pdb_raw"); RAW.mkdir(parents=True, exist_ok=True)
''')

md(r'''## Cell 2 — Sample the SAME ~1,000 HOLO4K structures (seed=2024, identical to every prior phase -- reproducible)''')
code(s24(5))

md(r'''## Cell 3 — Ground truth for the sampled structures (identical to Phase 24/33/39/54/94/99/100)''')
code(s24(17))

md(f'''## Cell 4 — Slice eval_units to chunk {CHUNK_IDX}/{N_CHUNKS}''')
code(f'''
_n = len(eval_units)
_lo = ({CHUNK_IDX} * _n) // {N_CHUNKS}
_hi = (({CHUNK_IDX}+1) * _n) // {N_CHUNKS}
eval_units = eval_units[_lo:_hi]
print(f"chunk {CHUNK_IDX}/{N_CHUNKS}: structures [{{_lo}}:{{_hi}}] = {{len(eval_units)}}/{{_n}}")
''')

md(f'''## Cell 5 — Featurize this chunk (dihedral + ESM2-3B + topology; saves phase{PHASE_NUM}_raw_features_chunk{CHUNK_IDX}.pkl)''')
code(s24(19).replace('cache/phase24_raw_features.pkl', f'cache/phase{PHASE_NUM}_raw_features_chunk{CHUNK_IDX}.pkl')
            .replace('AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")',
                      'AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")')
            .replace('AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(DEVICE).eval()',
                      '(AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(DEVICE)\n'
                      '       if DEVICE.type == "cuda" else\n'
                      '       AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D")).eval()')
            .replace('out = esm(**enc).last_hidden_state\n    return out[0,1:1+len(seq),:].float().cpu()',
                      'out = esm(**enc).last_hidden_state\n    return out[0,1:1+len(seq),:].float().cpu()  # 3B model is fp16; upcast here'))

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path(f"/Users/shivenkhurana/bioproj/phase{PHASE_NUM}_holo4k_featurize_chunk{CHUNK_IDX}.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
