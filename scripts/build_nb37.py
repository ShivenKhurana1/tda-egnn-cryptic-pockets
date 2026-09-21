#!/usr/bin/env python3
"""Phase 37: Thread 3 round 2, step 3 of 3 (dihedrals + ESM2, final
trainable blob). Mirrors Phase 10/30's cells exactly (proven code, spliced
verbatim) but loads Phase 36's larger pool instead of Phase 29's. GPU per
standing instruction (this stage genuinely benefits from it, unlike
ingestion/topology)."""
import json, pickle
from pathlib import Path

C10 = pickle.load(open("/tmp/p10_cells.pkl", "rb"))
s10 = lambda i: (C10[i]["source"] if isinstance(C10[i]["source"], str) else "".join(C10[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 37 — Thread 3 Round 2, Step 3: Dihedrals + ESM2 at Scale

Same dihedral-recovery and ESM2-embedding logic as Phase 10/30 (byte-for-byte,
spliced verbatim), pointed at Phase 36's larger topology-fused pool.
Produces `phase37_graphs.pt` -- the round-2 scaled equivalent of
`phase10_graphs.pt`/`phase30_graphs.pt`. GPU-accelerated for the ESM2 stage.

This notebook only builds the blob -- it does NOT retrain/re-validate
anything. Follow-up: rerun the Phase20/21/31/32-style (COACH420) and
Phase24/27/33/34-style (HOLO4K) clean-training + benchmark pipelines on
this pool to see whether pushing scale further extends COACH420's win and/or
resolves HOLO4K's still-ambiguous gap.
''')

md(r'''## Cell 0 — Install dependencies''')
code(s10(2))

md(r'''## Cell 1 — Setup, load Phase 36's topology-fused pool''')
code(s10(3).replace('locate("phase6b_graphs.pt")', 'locate("phase36_graphs.pt")'))

md(r'''## Cell 2 — Recompute real backbone dihedrals (identical to Phase 10/30)''')
code(s10(5))

md(r'''## Cell 3 — ESM2 embeddings (identical to Phase 10/30, GPU-accelerated)''')
code(s10(7))

md(r'''## Cell 4 — Assemble augmented features, verify alignment, save (new output filename)''')
code(s10(9).replace('"/kaggle/working/phase10_graphs.pt"', '"/kaggle/working/phase37_graphs.pt"')
           .replace('saved -> /kaggle/working/phase10_graphs.pt', 'saved -> /kaggle/working/phase37_graphs.pt'))

md(r'''
---

`phase37_graphs.pt` -- same schema as `phase10_graphs.pt`/`phase30_graphs.pt`,
just the round-2 scaled pool. Any future retraining/re-validation notebook
should `locate("phase37_graphs.pt")` in place of `phase30_graphs.pt`.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase37_esm_dihedral_scale2.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
