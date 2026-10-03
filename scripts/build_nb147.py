#!/usr/bin/env python3
"""Phase 147: score-only eval of Phase146's 650M-transfer 8-seed ensemble
on CryptoBench TEST, val-tuned threshold -- gives the apples-to-apples
TEST metrics needed to compare against Phase47's from-scratch 650M
ensemble (test AUPRC 0.3627, AUC 0.8873, MCC 0.3986)."""
import json
from pathlib import Path

nb46 = json.load(open("/Users/shivenkhurana/bioproj/phase46_cryptobench_train_eval_esm650m.ipynb"))
c46 = nb46["cells"]
s46 = lambda i: (c46[i]["source"] if isinstance(c46[i]["source"], str) else "".join(c46[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 147 - CryptoBench 650M-Transfer: Score-Only Eval

Loads Phase146's 8 transfer-learned checkpoints, builds the ensemble,
tunes threshold on VAL only, evaluates on TEST -- directly comparable
to Phase47's from-scratch 650M ensemble (test AUPRC 0.3627, AUC 0.8873,
MCC 0.3986).
''')

md(r'''## Cell 1 - Install dependencies''')
code(s46(2))

md(r'''## Cell 2 - Setup, load Phase 45's CryptoBench-650M blob (identical to Phase46/47/146)''')
code(s46(3))

md(r'''## Cell 3 - Model (identical to every prior localization phase)''')
code(s46(5))

md(r'''## Cell 4 - Load Phase 146's 8 transfer-learned checkpoints into an ensemble''')
code(r'''
ensemble = []
for seed in range(8):
    m = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=True).to(DEVICE)
    state = torch.load(locate(f"cryptobench_esm650m_transfer_seed{seed}.pt"), map_location=DEVICE)
    m.load_state_dict(state)
    m.eval()
    ensemble.append(m)
print(f"loaded {len(ensemble)} 650M-transfer checkpoints into ensemble")
''')

md(r'''## Cell 5 - Pick a decision threshold on VAL ONLY (never touch test for this), then evaluate on TEST''')
code(s46(11))

md(r'''## Cell 6 - Field-standard residue classification metrics on TEST (pooled)''')
code(s46(13))

md(r'''## Cell 7 - Our own localization-head metric (novel -- no prior baseline exists for this exact task)''')
code(s46(15).replace('phase43_summary.json', 'phase147_esm650m_transfer_summary.json')
            .replace('saved -> phase43_summary.json', 'saved -> phase147_esm650m_transfer_summary.json'))

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase147_cryptobench_esm650m_transfer_eval.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
