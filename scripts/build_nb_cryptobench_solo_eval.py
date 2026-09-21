#!/usr/bin/env python3
"""Generic score-only eval for a CryptoBench 8-seed transfer-learned
ensemble at one scale -- gives full TEST metrics (AUC/AUPRC/MCC/F1/DCC)
with a val-tuned threshold, matching Phase147's pattern. Usage:
python3 build_nb_cryptobench_solo_eval.py <SCALE_TAG> <BLOB_FILE> <CKPT_PREFIX> <PHASE_NUM>"""
import json
import sys
from pathlib import Path

SCALE_TAG = sys.argv[1]      # e.g. "35M" or "3B", used only in print statements
BLOB_FILE = sys.argv[2]      # e.g. "phase42_cryptobench_graphs.pt"
CKPT_PREFIX = sys.argv[3]    # e.g. "cryptobench_transfer_seed"
PHASE_NUM = sys.argv[4]

nb46 = json.load(open("/Users/shivenkhurana/bioproj/phase46_cryptobench_train_eval_esm650m.ipynb"))
c46 = nb46["cells"]
s46 = lambda i: (c46[i]["source"] if isinstance(c46[i]["source"], str) else "".join(c46[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(f'''
# Phase {PHASE_NUM} — CryptoBench {SCALE_TAG}-Transfer: Score-Only Eval

Loads the 8 transfer-learned {SCALE_TAG} checkpoints, builds the ensemble,
tunes threshold on VAL only, evaluates on TEST -- fills in the AUPRC/F1/DCC
metrics that were not computed when this scale was only used inside the
cross-scale ensemble (Phase130/148 only tracked AUC/MCC for the ensemble
math, not the full per-scale metric set).
''')

md(r'''## Cell 1 — Install dependencies''')
code(s46(2))

md(f'''## Cell 2 — Setup, load the {SCALE_TAG} CryptoBench blob''')
code(s46(3).replace('phase45_cryptobench_graphs_esm650m.pt', BLOB_FILE))

md(r'''## Cell 3 — Model (identical to every prior localization phase)''')
code(s46(5))

md(f'''## Cell 4 — Load the 8 transfer-learned {SCALE_TAG} checkpoints into an ensemble''')
code(f'''
ensemble = []
for seed in range(8):
    m = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=True).to(DEVICE)
    state = torch.load(locate(f"{CKPT_PREFIX}{{seed}}.pt"), map_location=DEVICE)
    m.load_state_dict(state)
    m.eval()
    ensemble.append(m)
print(f"loaded {{len(ensemble)}} {SCALE_TAG}-transfer checkpoints into ensemble")
''')

md(r'''## Cell 5 — Pick a decision threshold on VAL ONLY (never touch test for this), then evaluate on TEST''')
code(s46(11))

md(r'''## Cell 6 — Field-standard residue classification metrics on TEST (pooled)''')
code(s46(13))

md(r'''## Cell 7 — Our own localization-head metric (novel -- no prior baseline exists for this exact task)''')
code(s46(15).replace('phase43_summary.json', f'phase{PHASE_NUM}_summary.json')
            .replace('saved -> phase43_summary.json', f'saved -> phase{PHASE_NUM}_summary.json'))

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path(f"/Users/shivenkhurana/bioproj/phase{PHASE_NUM}_cryptobench_{SCALE_TAG.lower()}_transfer_eval.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
