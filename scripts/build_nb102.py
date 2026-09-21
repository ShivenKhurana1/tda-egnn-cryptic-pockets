#!/usr/bin/env python3
"""Phase 102: COACH420 ESM2-3B 8-SEED RESCORE. Mirrors Phase98 exactly
but loads the full 8-seed ensemble (Phase97's seeds 0-2 + Phase101's
seeds 3-7) instead of just 3. This is the properly-powered final number
for the ESM2 capacity lever on COACH420, following this project's
established 3-seed-then-8-seed discipline."""
import json, pickle
from pathlib import Path

C20 = pickle.load(open("/tmp/p20_cells.pkl", "rb"))
s20 = lambda i: (C20[i]["source"] if isinstance(C20[i]["source"], str) else "".join(C20[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 102 — COACH420 ESM2-3B: 8-Seed Rescore

Phase98's 3-seed result was DCC 0.6053, already beating the 8-seed 35M
baseline (0.5932). This loads all 8 seeds (Phase97's 0-2 + Phase101's
3-7) for the properly-powered final number -- identical methodology to
Phase98 otherwise (same lightweight setup, same fresh ESM2-3B load).
''')

md(r'''## Cell 0 — Install dependencies''')
code(s20(2).replace('transformers==4.46.3', 'transformers==4.46.3 accelerate'))

md(r'''## Cell 1 — Lightweight setup: Phase51's small topology blob only, hardcoded N_PRE/TOPO_DIM/ESM_DIM (identical to Phase98)''')
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
assert TOPO_DIM == 34, f"expected TOPO_DIM=34 (matches Phase97/101's trained checkpoints), got {TOPO_DIM}"
N_PRE = 2587    # 21 onehot + 6 dihedral + 2560 ESM
ESM_DIM = 2560
if "dataset" in blob6b: del blob6b["dataset"]

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
print(f"N_PRE={N_PRE}  TOPO_DIM={TOPO_DIM}  ESM_DIM={ESM_DIM}  DEVICE={DEVICE}")

WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)
RAW = Path("/tmp/pdb_raw"); RAW.mkdir(parents=True, exist_ok=True)
''')

md(r'''## Cell 2 — Fetch ALL 420 COACH420 structures''')
code(s20(5))

md(r'''## Cell 3 — Model''')
code(s20(11))

md(r'''## Cell 4 — Load all 8 trained checkpoints (Phase97 seeds 0-2 + Phase101 seeds 3-7)''')
code(r'''
ensemble = []
for seed in range(8):
    m = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=True).to(DEVICE)
    ckpt = locate(f"clean_deploy_esm3b_seed{seed}.pt")
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval()
    ensemble.append(m)
    print(f"  loaded seed {seed} from {ckpt}")
print(f"ensemble of {len(ensemble)} trained models ready (full 8-seed)")
''')

md(r'''## Cell 5 — Ground-truth ligands for ALL 420 COACH420 structures''')
code(s20(19))

md(r'''## Cell 6 — Featurize (dihedral + ESM2-3B + topology; identical to Phase98)''')
code(s20(21).replace("cache/phase20_raw_features.pkl", "cache/phase102_raw_features.pkl")
            .replace('AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")',
                      'AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")')
            .replace('AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(DEVICE).eval()',
                      '(AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(DEVICE)\n'
                      '       if DEVICE.type == "cuda" else\n'
                      '       AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D")).eval()')
            .replace('out = esm(**enc).last_hidden_state\n    return out[0,1:1+len(seq),:].float().cpu()',
                      'out = esm(**enc).last_hidden_state\n    return out[0,1:1+len(seq),:].float().cpu()  # 3B model is fp16; upcast here'))

md(r'''## Cell 7 — P2Rank on the FULL 420''')
code(s20(23).replace('Path("/tmp/p2_20")', 'Path("/tmp/p2_102")')
            .replace('Path("/tmp/p2_20_out")', 'Path("/tmp/p2_102_out")'))

md(r'''## Cell 8 — Score, compare to literature AND the 3-seed/8-seed/35M comparison''')
code(s20(25).replace("phase20_summary.json", "phase102_summary.json")
            .replace('"n_train_excluded": len(leaked_train_keys),\n           "n_train_clean": len(dataset)},',
                      '"n_train_excluded": 332, "n_train_clean": 7181},  # from Phase97\'s own log'))

code(r'''
print(f"\n--- COACH420 DCC: ESM2 capacity lever, full comparison ---")
print(f"{'embedding':<28}{'hidden':>8}{'DCC (3-seed)':>14}{'DCC (8-seed)':>14}")
print("-"*64)
print(f"{'ESM2-35M (Phase53/56)':<28}{480:>8}{0.5763:>14.4f}{0.5932:>14.4f}")
print(f"{'ESM2-3B (Phase98/102)':<28}{2560:>8}{0.6053:>14.4f}{ours_dcc_rate:>14.4f}")
print(f"\ntraining pool size: 7,181 clean chains (round-3, unchanged, from Phase97/101)")
print(f"this-run P2Rank DCC: {p2_dcc_rate:.4f}")
noise_floor_estimate = 0.03   # typical single-seed std seen elsewhere in this project
print(f"\ngain vs 35M 8-seed: {ours_dcc_rate-0.5932:+.4f}  ({(ours_dcc_rate-0.5932)/noise_floor_estimate:.1f}x a ~0.03 noise floor)")
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase102_esm3b_coach420_rescore8seed.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
