#!/usr/bin/env python3
"""Phase 98: COACH420 SCORE-ONLY, the second half of Phase93's split (see
Phase97's docstring for the full diagnosis). This kernel NEVER loads
Phase96's 23.6GB training blob -- it only needs Phase51's small topology
blob (for topo_mean/topo_std/topo_names) and Phase97's 3 trained
checkpoints, then does its own lightweight setup, fetches the 420
COACH420 structures, featurizes them (dihedral + a FRESH ESM2-3B load,
which now has the kernel's full ~32.9GB budget to itself since training
never touched this process), runs P2Rank, and scores. N_PRE/TOPO_DIM are
hardcoded from Phase96/97's own confirmed log output (2587 = 21+6+2560,
34) rather than loaded, avoiding the big blob entirely."""
import json, pickle
from pathlib import Path

C20 = pickle.load(open("/tmp/p20_cells.pkl", "rb"))
s20 = lambda i: (C20[i]["source"] if isinstance(C20[i]["source"], str) else "".join(C20[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 98 - COACH420 ESM2-3B: SCORE-ONLY

Second half of the Phase93 split (see Phase97). Loads Phase97's 3 trained
checkpoints fresh (rebuilding `LocalizingEGNN` from the same N_PRE=2587/
TOPO_DIM=34 the checkpoints were trained with) into a kernel that has
NEVER held the 7,513-chain training pool -- so the live ESM2-3B load for
featurizing the 420 benchmark structures has the full RAM budget to
itself, unlike Phase93 v2-v6 which all crashed doing this in the same
process as training.
''')

md(r'''## Cell 0 - Install dependencies (a fresh kernel needs this too -- v1 of this notebook forgot it entirely, ModuleNotFoundError on torch_geometric)''')
code(s20(2).replace('transformers==4.46.3', 'transformers==4.46.3 accelerate'))

md(r'''## Cell 1 - Lightweight setup: Phase51's small topology blob only (NOT Phase96's big training blob), hardcoded N_PRE/TOPO_DIM/ESM_DIM (confirmed via Phase96/97's own log: node dim 2621 = 21+6+2560+34)''')
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
assert TOPO_DIM == 34, f"expected TOPO_DIM=34 (matches Phase97's trained checkpoints), got {TOPO_DIM}"
N_PRE = 2587    # 21 onehot + 6 dihedral + 2560 ESM -- must match what Phase97 trained with
ESM_DIM = 2560
if "dataset" in blob6b: del blob6b["dataset"]   # only topo_mean/topo_std/topo_names needed later

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

md(r'''## Cell 2 - Fetch ALL 420 COACH420 structures (identical to Phase 20/31/38/53/93/97)''')
code(s20(5))

md(r'''## Cell 3 - Model (identical to every prior localization phase -- needed here to reconstruct the trained architecture for loading checkpoints)''')
code(s20(11))

md(r'''## Cell 4 - Load Phase97's 3 trained checkpoints into fresh model instances''')
code(r'''
ensemble = []
for seed in (0, 1, 2):
    m = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=True).to(DEVICE)
    ckpt = locate(f"clean_deploy_esm3b_seed{seed}.pt")
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval()
    ensemble.append(m)
    print(f"  loaded seed {seed} from {ckpt}")
print(f"ensemble of {len(ensemble)} trained models ready")
''')

md(r'''## Cell 5 - Ground-truth ligands for ALL 420 COACH420 structures (identical to Phase 20/31/38/53/93)''')
code(s20(19))

md(r'''## Cell 6 - Featurize (dihedral + ESM2-3B + topology; ESM2-3B loaded fresh here, full RAM budget available since training never ran in this process)''')
code(s20(21).replace("cache/phase20_raw_features.pkl", "cache/phase98_raw_features.pkl")
            .replace('AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")',
                      'AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")')
            .replace('AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(DEVICE).eval()',
                      '(AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(DEVICE)\n'
                      '       if DEVICE.type == "cuda" else\n'
                      '       AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D")).eval()  '
                      '# low_cpu_mem_usage+.to() leaves a meta tensor on CPU-only runs (NotImplementedError); only use it for GPU')
            .replace('out = esm(**enc).last_hidden_state\n    return out[0,1:1+len(seq),:].float().cpu()',
                      'out = esm(**enc).last_hidden_state\n    return out[0,1:1+len(seq),:].float().cpu()  # 3B model is fp16; upcast here'))

md(r'''## Cell 7 - P2Rank on the FULL 420 (same-run controlled reference, identical to Phase 20/31/38/53/93)''')
code(s20(23).replace('Path("/tmp/p2_20")', 'Path("/tmp/p2_98")')
            .replace('Path("/tmp/p2_20_out")', 'Path("/tmp/p2_98_out")'))

md(r'''## Cell 8 - Score both, compare to literature AND to the 35M-vs-3B comparison''')
code(s20(25).replace("phase20_summary.json", "phase98_summary.json")
            .replace('"n_train_excluded": len(leaked_train_keys),\n           "n_train_clean": len(dataset)},',
                      '"n_train_excluded": 332, "n_train_clean": 7181},  # from Phase97\'s own log'))

code(r'''
print(f"\n--- COACH420 DCC: ESM2 capacity lever at round-3 scale ---")
print(f"{'embedding':<28}{'hidden':>8}{'DCC (3-seed)':>14}{'DCC (8-seed)':>14}")
print("-"*64)
print(f"{'ESM2-35M (Phase53/56)':<28}{480:>8}{0.5763:>14.4f}{0.5932:>14.4f}")
print(f"{'ESM2-3B (this notebook)':<28}{2560:>8}{ours_dcc_rate:>14.4f}{'pending':>14}")
print(f"\ntraining pool size: 7,181 clean chains (round-3, unchanged, from Phase97)")
print(f"this-run P2Rank DCC: {p2_dcc_rate:.4f}  (deterministic tool, should match prior rounds' ~0.5209)")
''')

md(r'''
---

## Reading this

- **DCC beats 0.5932** -- embedding capacity is a real, addressable lever
  here too (mirrors CryptoBench), worth extending to 8 seeds and then
  applying to HOLO4K (Phase99/100).
- **DCC roughly matches 0.5932 (within noise)** -- capacity doesn't help
  once topology+dihedral+scale have already saturated what COACH420's
  training signal can support.
- **DCC drops** -- would be surprising and worth real scrutiny before
  accepting.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase98_esm3b_coach420_score.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
