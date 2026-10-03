#!/usr/bin/env python3
"""Phase 36: Thread 3 round 2, step 2 of 3 (topology at scale). Mirrors
Phase 6b/29's pipeline exactly (proven code, spliced verbatim) but loads
Phase 35's ~30,000-entry pool instead of Phase 28's ~15,000-entry one.
CPU-bound (gudhi has no GPU path) regardless of the enable_gpu flag."""
import json, pickle
from pathlib import Path

C6b = pickle.load(open("/tmp/p6b_cells.pkl", "rb"))
s6b = lambda i: (C6b[i]["source"] if isinstance(C6b[i]["source"], str) else "".join(C6b[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 36 - Thread 3 Round 2, Step 2: Topology at Scale

Same persistent-homology featurizer as Phase 6b/29 (byte-for-byte, spliced
verbatim), pointed at Phase 35's larger pool. CPU only (gudhi has no GPU
path) -- expect roughly proportional to Phase 29's 80.5 min, scaled by the
residue-count ratio between the two pools.
''')

md(r'''## Cell 0 - Install dependencies''')
code(s6b(2))

md(r'''## Cell 1 - Load Phase 35's pool and re-acquire heavy atoms''')
code(s6b(3).replace('locate("phase6_graphs.pt")', 'locate("phase35_graphs.pt")'))

md(r'''## Cell 2 - Heavy-atom clouds with per-chain alignment verification (identical to Phase 6b/29)''')
code(s6b(5))

md(r'''## Cell 3 - Featuriser (byte-for-byte the Phase 2b/6b/29 definition)''')
code(s6b(7))

md(r'''## Cell 4 - Compute (the long one)''')
code(s6b(9))

md(r'''## Cell 5 - Fuse, validate, save (new output filename)''')
code(s6b(11).replace('p = out / "phase6b_graphs.pt"', 'p = out / "phase36_graphs.pt"'))

md(r'''
---

`phase36_graphs.pt` -- same schema as `phase6b_graphs.pt`/`phase29_graphs.pt`,
just the bigger pool. Next: Phase 37 (ESM2 + dihedrals, mirrors Phase 10/30).
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase36_topology_scale2.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
