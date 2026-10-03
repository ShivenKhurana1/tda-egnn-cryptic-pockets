#!/usr/bin/env python3
"""Phase 100: HOLO4K SCORE-ONLY, second half of the Phase94 split (see
Phase99's docstring). Never loads Phase96's big training blob -- reads
Phase99's checkpoints + phase99_train_meta.json (avoids hardcoding/
guessing the training-pool size, unlike Phase98's COACH420 version which
hardcoded a value already independently confirmed across 5 runs)."""
import json, pickle
from pathlib import Path

C24 = pickle.load(open("/tmp/p24_cells.pkl", "rb"))
s24 = lambda i: (C24[i]["source"] if isinstance(C24[i]["source"], str) else "".join(C24[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 100 - HOLO4K ESM2-3B: SCORE-ONLY

Second half of the Phase94 split (see Phase99). Loads Phase99's 3 trained
checkpoints fresh into a kernel that has never held the 7,513-chain
training pool, so the live ESM2-3B load for featurizing the ~1,000
sampled HOLO4K structures has the full RAM budget to itself.
''')

md(r'''## Cell 0 - Install dependencies (a fresh kernel needs this too)''')
code(s24(2).replace('transformers==4.46.3', 'transformers==4.46.3 accelerate'))

md(r'''## Cell 1 - Lightweight setup: Phase51's small topology blob only, hardcoded N_PRE/ESM_DIM (same values as Phase98 -- the node-feature schema is identical across COACH420/HOLO4K)''')
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
assert TOPO_DIM == 34, f"expected TOPO_DIM=34 (matches Phase99's trained checkpoints), got {TOPO_DIM}"
N_PRE = 2587    # 21 onehot + 6 dihedral + 2560 ESM -- must match what Phase99 trained with
ESM_DIM = 2560
if "dataset" in blob6b: del blob6b["dataset"]

train_meta = json.load(open(locate("phase99_train_meta.json")))
N_TRAIN_CHAINS = train_meta["n_train_chains"]

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
print(f"N_PRE={N_PRE}  TOPO_DIM={TOPO_DIM}  ESM_DIM={ESM_DIM}  N_TRAIN_CHAINS={N_TRAIN_CHAINS:,}  DEVICE={DEVICE}")

WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)
RAW = Path("/tmp/pdb_raw"); RAW.mkdir(parents=True, exist_ok=True)
''')

md(r'''## Cell 2 - Sample the SAME ~1,000 HOLO4K structures (seed=2024, identical to Phase 24/33/39/54/94/99)''')
code(s24(5))

md(r'''## Cell 3 - Model (identical to every prior localization phase -- needed here to reconstruct the trained architecture for loading checkpoints)''')
code(s24(11))

md(r'''## Cell 4 - Load Phase99's 3 trained checkpoints into fresh model instances''')
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

md(r'''## Cell 5 - Ground truth for the sampled 1,000 structures (identical to Phase 24/33/39/54/94)''')
code(s24(17))

md(r'''## Cell 6 - Featurize (dihedral + ESM2-3B + topology; ESM2-3B loaded fresh here, full RAM budget available since training never ran in this process)''')
code(s24(19).replace("phase24_raw_features.pkl", "phase100_raw_features.pkl")
            .replace('AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")',
                      'AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")')
            .replace('AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(DEVICE).eval()',
                      '(AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(DEVICE)\n'
                      '       if DEVICE.type == "cuda" else\n'
                      '       AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D")).eval()  '
                      '# low_cpu_mem_usage+.to() leaves a meta tensor on CPU-only runs (NotImplementedError); only use it for GPU')
            .replace('out = esm(**enc).last_hidden_state\n    return out[0,1:1+len(seq),:].float().cpu()',
                      'out = esm(**enc).last_hidden_state\n    return out[0,1:1+len(seq),:].float().cpu()  # 3B model is fp16; upcast here'))

md(r'''## Cell 7 - P2Rank on the same 1,000 structures (identical to Phase 24/33/39/54/94)''')
code(s24(21))

md(r'''## Cell 8 - Score, compare to literature AND to the 35M-vs-3B comparison''')
code(s24(23).replace("2056", "7513").replace("phase24_summary.json", "phase100_summary.json")
            .replace("7513-len(dataset)", "7513-N_TRAIN_CHAINS"))

code(r'''
print(f"\n--- HOLO4K DCC: ESM2 capacity lever at round-3 scale ---")
print(f"{'embedding':<28}{'hidden':>8}{'DCC (3-seed)':>14}{'DCC (8-seed)':>14}{'gap to P2Rank':>16}")
print("-"*82)
print(f"{'ESM2-35M (Phase54/56)':<28}{480:>8}{0.6483:>14.4f}{0.6453:>14.4f}{-0.0127:>16.4f}")
print(f"{'ESM2-3B (this notebook)':<28}{2560:>8}{ours_dcc_rate:>14.4f}{'pending':>14}"
      f"{ours_dcc_rate-p2_dcc_rate:>16.4f}")
print(f"\ntraining pool size: {N_TRAIN_CHAINS:,} clean chains (round-3, unchanged, from Phase99)")
print(f"this-run P2Rank DCC: {p2_dcc_rate:.4f}  (deterministic tool, same 1000-structure sample, "
      f"should match prior rounds' ~0.6580)")
''')

md(r'''
---

## Reading this

- **Gap improves / turns clearly positive** -- capacity is the lever that
  closes HOLO4K's remaining gap, same story as COACH420 and CryptoBench.
- **Gap holds flat (within noise)** -- capacity doesn't move HOLO4K either.
- **Gap worsens** -- worth scrutiny before accepting.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase100_esm3b_holo4k_score.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
