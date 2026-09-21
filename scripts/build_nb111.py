#!/usr/bin/env python3
"""Phase 111: HOLO4K ESM2-3B COMBINE + SCORE. Merges Phase108/109/110's
3 partial raw-features pickles (each numpy-serialized) back into a torch
`built` dict, loads Phase99's 3 trained checkpoints, runs P2Rank on the
same 1000-structure sample, and scores -- the final step of the
featurization-chunking fix for Phase100's 12hr cancellation."""
import json, pickle
from pathlib import Path

C24 = pickle.load(open("/tmp/p24_cells.pkl", "rb"))
s24 = lambda i: (C24[i]["source"] if isinstance(C24[i]["source"], str) else "".join(C24[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 111 — HOLO4K ESM2-3B: Combine Chunks + Score

Merges Phase108/109/110's 3 partial featurization pickles, loads
Phase99's trained ensemble, runs P2Rank on the same 1000-structure
sample, and scores. Final step of the chunked-featurization fix.
''')

md(r'''## Cell 0 — Install dependencies (including java, needed for P2Rank)''')
code('!pip install -q torch-geometric==2.6.1 gudhi==3.10.1 biopython==1.84 transformers==4.46.3 accelerate\n'
     '!apt-get -qq update > /dev/null 2>&1 && apt-get -qq install -y default-jre-headless > /dev/null 2>&1')

md(r'''## Cell 1 — Lightweight setup: Phase51's small topology blob only''')
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

def locate_all(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    return c

blob6b = torch.load(locate("phase51_graphs.pt"), map_location="cpu", weights_only=False)
TOPO_NAMES = blob6b["topo_names"]
TOPO_DIM = len(TOPO_NAMES)
assert TOPO_DIM == 34, f"expected TOPO_DIM=34, got {TOPO_DIM}"
N_PRE = 2587
ESM_DIM = 2560
if "dataset" in blob6b: del blob6b["dataset"]

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
print(f"N_PRE={N_PRE}  TOPO_DIM={TOPO_DIM}  DEVICE={DEVICE}")

WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)
RAW = Path("/tmp/pdb_raw"); RAW.mkdir(parents=True, exist_ok=True)
''')

md(r'''## Cell 2 — Sample the SAME ~1,000 HOLO4K structures (seed=2024 -- reproducible, needed for holo_paths in P2Rank)''')
code(s24(5))

md(r'''## Cell 3 — Model''')
code(s24(11))

md(r'''## Cell 4 — Load Phase99's 3 trained checkpoints''')
code(r'''
ensemble = []
for seed in (0, 1, 2):
    m = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=True).to(DEVICE)
    ckpt = locate(f"holo_clean_deploy_esm3b_seed{seed}.pt")
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval()
    ensemble.append(m)
    print(f"  loaded seed {seed} from {ckpt}")
print(f"ensemble of {len(ensemble)} trained models ready")
''')

md(r'''## Cell 5 — Merge Phase108/109/110's 3 partial featurization pickles into `built` (numpy -> torch)''')
code(r'''
TENSOR_KEYS = {"x", "pos", "edge_index", "edge_attr"}
built = {}
n_total = 0
for chunk_idx, phase_num in enumerate((108, 109, 110)):
    matches = [p for p in locate_all(f"phase{phase_num}_raw_features_chunk{chunk_idx}.pkl")]
    if not matches:
        raise FileNotFoundError(f"chunk {chunk_idx} pickle not found -- is Phase{phase_num} attached as a kernel_source?")
    p = max(matches, key=lambda x: x.stat().st_size)
    with open(p, "rb") as fh:
        chunk = pickle.load(fh)
    for k, v in chunk.items():
        built[k] = {kk: (torch.from_numpy(vv) if kk in TENSOR_KEYS else vv) for kk, vv in v.items()}
    print(f"  chunk {chunk_idx} ({p}): {len(chunk)} chains")
    n_total += len(chunk)
print(f"merged {len(built)}/{n_total} chains total")
''')

md(r'''## Cell 6 — P2Rank on the same 1,000 structures (identical to Phase 24/33/39/54/94/100)''')
code(s24(21).replace('Path("/tmp/p2_24")', 'Path("/tmp/p2_111")')
            .replace('Path("/tmp/p2_24_out")', 'Path("/tmp/p2_111_out")'))

md(r'''## Cell 7 — Score, compare to literature AND to the 35M-vs-3B comparison''')
code(s24(23).replace("phase24_summary.json", "phase111_summary.json")
            .replace('"n_train_excluded": 2056-len(dataset),\n           "n_sample": len(holo_sample), "verdict": verdict},',
                      '"n_train_excluded": 7513-7061, "n_sample": len(holo_sample), "verdict": verdict},  # from Phase99\'s own log'))

code(r'''
print(f"\n--- HOLO4K DCC: ESM2 capacity lever at round-3 scale ---")
print(f"{'embedding':<28}{'hidden':>8}{'DCC (3-seed)':>14}{'DCC (8-seed)':>14}{'gap to P2Rank':>16}")
print("-"*82)
print(f"{'ESM2-35M (Phase54/56)':<28}{480:>8}{0.6483:>14.4f}{0.6453:>14.4f}{-0.0127:>16.4f}")
print(f"{'ESM2-3B (this notebook)':<28}{2560:>8}{ours_dcc_rate:>14.4f}{'pending':>14}"
      f"{ours_dcc_rate-p2_dcc_rate:>16.4f}")
print(f"\ntraining pool size: 7,061 clean chains (from Phase99)")
print(f"this-run P2Rank DCC: {p2_dcc_rate:.4f}")
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase111_holo4k_combine_score.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
