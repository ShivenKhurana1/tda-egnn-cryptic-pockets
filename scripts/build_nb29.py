#!/usr/bin/env python3
"""Phase 29: Thread 3, step 2 of 3 (topology at scale). Mirrors Phase 6b's
pipeline exactly (identical featurizer, identical rotation-invariance check,
identical fuse/prune/save logic -- proven code, spliced verbatim) but loads
Phase 28's ~15,000-entry pool instead of Phase 6a's original 6,000-entry one.
CPU-bound (gudhi has no GPU path) -- expect ~2.5x Phase 6b's 66 min, i.e.
~2.5-3 hours for persistent homology over ~1.5M residues."""
import json, pickle
from pathlib import Path

C6b = pickle.load(open("/tmp/p6b_cells.pkl", "rb"))
s6b = lambda i: (C6b[i]["source"] if isinstance(C6b[i]["source"], str) else "".join(C6b[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 29 — Thread 3, Step 2: Topology at Scale (2.5x pool)

Same persistent-homology featurizer as Phase 6b (byte-for-byte, spliced
verbatim -- alpha complexes on 12A neighborhoods, H0/H1/H2, Betti radii
2/3/5A, burial features, rotation-invariance re-verified below), just
pointed at Phase 28's ~15,000-entry pool instead of Phase 6a's 6,000. CPU
only (gudhi has no GPU path) -- expect roughly 2.5x Phase 6b's 66 minutes.
''')

md(r'''## Cell 0 — Install dependencies''')
code(s6b(2))

md(r'''## Cell 1 — Load Phase 28's pool and re-acquire heavy atoms''')
code(s6b(3).replace('locate("phase6_graphs.pt")', 'locate("phase28_graphs.pt")'))

md(r'''## Cell 2 — Heavy-atom clouds with per-chain alignment verification (identical to Phase 6b)''')
code(s6b(5))

md(r'''## Cell 3 — Featuriser (byte-for-byte the Phase 2b/6b definition)''')
code(s6b(7))

md(r'''## Cell 4 — Compute (the long one)''')
code(s6b(9))

md(r'''## Cell 5 — Fuse, validate, save (new output filename)''')
code(s6b(11).replace('p = out / "phase6b_graphs.pt"', 'p = out / "phase29_graphs.pt"'))

md(r'''
---

`phase29_graphs.pt` -- same schema as `phase6b_graphs.pt` (dataset with
fused x_topo/x, topo_dim, topo_names, burial_names, topo_mean/std, topo_auc,
pos_weight, aa_alphabet), just the bigger pool. Next: Phase 30 (dihedrals +
ESM2, mirrors Phase 9/10) to reach the final trainable schema.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase29_topology_scale.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
