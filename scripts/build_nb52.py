#!/usr/bin/env python3
"""Phase 52: Thread 3 round 3, step 3 of 3 (dihedrals + ESM2, final
trainable blob). Mirrors Phase 10/30/37's cells exactly (proven code,
spliced verbatim) but loads Phase 51's larger pool instead of Phase 36's.
GPU-accelerated for ESM2 (the expensive stage)."""
import json, pickle
from pathlib import Path

C10 = pickle.load(open("/tmp/p10_cells.pkl", "rb"))
s10 = lambda i: (C10[i]["source"] if isinstance(C10[i]["source"], str) else "".join(C10[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 52 - Thread 3 Round 3, Step 3: Dihedrals + ESM2 at Scale

Same dihedral-recovery and ESM2-embedding logic as Phase 10/30/37
(byte-for-byte, spliced verbatim), pointed at Phase 51's larger
topology-fused pool (7,513 clusters, 3.65x original). Produces
`phase52_graphs.pt` -- the round-3 scaled equivalent of
`phase10/30/37_graphs.pt`. GPU-accelerated for the ESM2 stage.

This notebook only builds the blob -- it does NOT retrain/re-validate
anything. Follow-up: rerun the Phase20/31/38-style (COACH420) and
Phase24/33/39-style (HOLO4K) clean-training + benchmark pipelines on this
pool to see whether COACH420's still-climbing win keeps growing and
whether HOLO4K holds its win from round 2.
''')

md(r'''## Cell 0 - Install dependencies''')
code(s10(2))

md(r'''## Cell 1 - Setup, load Phase 51's topology-fused pool''')
code(s10(3).replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")'))

md(r'''## Cell 2 - Recompute real backbone dihedrals (identical to Phase 10/30/37)''')
code(s10(5))

md(r'''## Cell 3 - ESM2 embeddings (identical to Phase 10/30/37, GPU-accelerated)''')
code(s10(7))

md(r'''## Cell 4 - Assemble augmented features, verify alignment, save (new output filename)''')
code(s10(9).replace('"/kaggle/working/phase10_graphs.pt"', '"/kaggle/working/phase52_graphs.pt"')
           .replace('saved -> /kaggle/working/phase10_graphs.pt', 'saved -> /kaggle/working/phase52_graphs.pt'))

md(r'''
---

`phase52_graphs.pt` -- same schema as `phase10/30/37_graphs.pt`, just the
round-3 scaled pool. Any future retraining/re-validation notebook should
`locate("phase52_graphs.pt")` in place of `phase37_graphs.pt`.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase52_esm_dihedral_scale3.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
