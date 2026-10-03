#!/usr/bin/env python3
"""Phase 49: Thread 3 round 3, step 1 of 3 (ingestion). Phase31/32/38/40
showed COACH420's win over live P2Rank kept growing at every scale tested
so far (2,056 -> 3,577 -> 5,339 clusters, DCC 0.516 -> 0.564 -> 0.596,
8-seed each) with NO sign of saturating. This pushes further: n_target
30000 -> 60000 (10x the ORIGINAL 6000, 2x Phase35's 30000). Same seed=1337
shuffle-then-slice means Phase6a's 6000, Phase28's 15000, AND Phase35's
30000 are all strict subsets of this 60000 -- still a clean superset."""
import json, pickle
from pathlib import Path

C6a = pickle.load(open("/tmp/p6a_cells.pkl", "rb"))
s6a = lambda i: (C6a[i]["source"] if isinstance(C6a[i]["source"], str) else "".join(C6a[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 49 - Thread 3 Round 3, Step 1: Scale-Up Ingestion (10x original, 2x Phase 35)

COACH420's win over live P2Rank has grown at every scale tested so far with
no sign of saturating: 0.516 (2,056 clusters) -> 0.564 (3,577) -> **0.596
(5,339)**, all 8-seed validated. This pushes further: `n_target` 30000 ->
60000 PDB entries, same `seed=1337` shuffle -- so Phase 6a's original 6,000,
Phase 28's 15,000, AND Phase 35's 30,000 are all strict subsets of this
60,000-entry sample.

Redundancy has grown sub-linearly at every step so far (15K entries -> 3,577
clusters, 1.74x not 2.5x; 30K entries -> 5,339 clusters, 2.6x not 5x) --
expect the same diminishing-but-real pattern here too.
''')

md(r'''## Cell 0 - Install dependencies''')
code(s6a(2))

md(r'''## Cell 1 - Config (n_target raised 30000 -> 60000, everything else identical)''')
code(s6a(3).replace("n_target:         int   = 6000      # entries to download (of ~127k available)",
                     "n_target:         int   = 60000     # entries to download (of ~127k available) -- 10x Phase6a, 2x Phase35"))

md(r'''## Cell 2 - Select entries via the RCSB Search API (identical query to Phase 6a/28/35)''')
code(s6a(5))

md(r'''## Cell 3 - Download (identical to Phase 6a/28/35)''')
code(s6a(7))

md(r'''## Cell 4 - Parse and label (identical predicates to Phase 6a/28/35)''')
code(s6a(9))

md(r'''## Cell 5 - MMseqs2 clustering and representative selection (identical to Phase 6a/28/35)''')
code(s6a(11))

md(r'''## Cell 6 - Build graphs (identical schema to Phase 6a/28/35/Phase 1)''')
code(s6a(13))

md(r'''## Cell 7 - Splits, QC, save (identical logic, new output filename)''')
code(s6a(15).replace('p = OUT / "phase6_graphs.pt"', 'p = OUT / "phase49_graphs.pt"'))

md(r'''
---

## What this produces

`phase49_graphs.pt` -- same schema as `phase6_graphs.pt`/`phase28_graphs.pt`/
`phase35_graphs.pt`, just a bigger pool. Next: Phase 50 (topology at this
scale, mirrors Phase 6b/29/36) and Phase 51 (ESM2 + dihedrals, mirrors
Phase 10/30/37).
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase49_scaleup3_ingestion.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
