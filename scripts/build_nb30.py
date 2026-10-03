#!/usr/bin/env python3
"""Phase 30: Thread 3, step 3 of 3 (dihedrals + ESM2, final trainable blob).
Mirrors Phase 10's cells 3/5/7/9 exactly (identical dihedral recovery,
identical ESM2 batching, identical assembly/alignment logic -- proven code,
spliced verbatim) but loads Phase 29's ~15,000-entry topology-fused pool
instead of Phase 6b's original 6,000-entry one. GPU-accelerated for ESM2
(the expensive stage); dihedral recovery is CPU-bound but fast."""
import json, pickle
from pathlib import Path

C10 = pickle.load(open("/tmp/p10_cells.pkl", "rb"))
s10 = lambda i: (C10[i]["source"] if isinstance(C10[i]["source"], str) else "".join(C10[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 30 - Thread 3, Step 3: Dihedrals + ESM2 at Scale (Final Trainable Blob)

Same dihedral-recovery and ESM2-embedding logic as Phase 10 (byte-for-byte,
spliced verbatim), just pointed at Phase 29's ~15,000-entry topology-fused
pool instead of Phase 6b's 6,000. Produces `phase30_graphs.pt` -- the scaled
equivalent of `phase10_graphs.pt`, the blob every localization phase (14
onward) actually trains on. GPU-accelerated for the ESM2 stage.

This notebook only builds the blob -- it does NOT retrain/re-validate
anything. That's the necessary follow-up once this exists: rerun the
Phase20/21-style "clean training + full COACH420/HOLO4K benchmark" pipeline
on this larger pool to see whether more data narrows the HOLO4K gap to live
P2Rank (Phase 27b) or changes anything about the topology-decomposition
finding (Phase 26).
''')

md(r'''## Cell 0 - Install dependencies''')
code(s10(2))

md(r'''## Cell 1 - Setup, load Phase 29's topology-fused pool''')
code(s10(3).replace('locate("phase6b_graphs.pt")', 'locate("phase29_graphs.pt")'))

md(r'''## Cell 2 - Recompute real backbone dihedrals (identical to Phase 10)''')
code(s10(5))

md(r'''## Cell 3 - ESM2 embeddings (identical to Phase 10, GPU-accelerated)''')
code(s10(7))

md(r'''## Cell 4 - Assemble augmented features, verify alignment, save (new output filename)''')
code(s10(9).replace('"/kaggle/working/phase10_graphs.pt"', '"/kaggle/working/phase30_graphs.pt"')
           .replace('saved -> /kaggle/working/phase10_graphs.pt', 'saved -> /kaggle/working/phase30_graphs.pt'))

md(r'''
---

`phase30_graphs.pt` -- same schema as `phase10_graphs.pt` (dataset, n_pre_topo,
topo_dim, esm_dim, pos_weight), just the ~2.5x pool. This is the blob any
future retraining/re-validation notebook should load via
`locate("phase30_graphs.pt")` in place of `phase10_graphs.pt`.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase30_esm_dihedral_scale.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
