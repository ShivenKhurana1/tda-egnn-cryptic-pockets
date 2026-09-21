#!/usr/bin/env python3
"""Phase 35: Thread 3 round 2, step 1 of 3 (ingestion). Phase31/32/33/34
showed scaling the training pool 1.74x (2,056 -> 3,577 clusters) gave a
robust, 8-seed-validated win on COACH420 and substantial-but-incomplete
progress on HOLO4K. This pushes further: n_target 15000 -> 30000 (5x the
ORIGINAL 6000, 2x Phase28's 15000). Same seed=1337 shuffle-then-slice means
Phase6a's 6000 AND Phase28's 15000 are both strict subsets of this 30000 --
still a clean superset, not an unrelated resample."""
import json, pickle
from pathlib import Path

C6a = pickle.load(open("/tmp/p6a_cells.pkl", "rb"))
s6a = lambda i: (C6a[i]["source"] if isinstance(C6a[i]["source"], str) else "".join(C6a[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 35 — Thread 3 Round 2, Step 1: Scale-Up Ingestion (5x original, 2x Phase 28)

Phase 31/32 showed the 1.74x-scaled pool (3,577 clusters) gives a robust,
8-seed-validated WIN over live P2Rank on COACH420 (+0.043 DCC). Phase 33/34
showed the same pool substantially narrows (but doesn't fully close) the
HOLO4K gap. Since training-data scale is the one lever that's demonstrably
worked, this pushes further: `n_target` 15000 -> 30000 PDB entries, same
`seed=1337` shuffle -- so Phase 6a's original 6,000 AND Phase 28's 15,000
are both strict subsets of this 30,000-entry sample.

Redundancy grows faster than linearly with sample size (Phase28: 15,000
entries only gave 3,577 clusters, 1.74x not the naively-expected 2.5x), so
expect diminishing but still real cluster growth here too.
''')

md(r'''## Cell 0 — Install dependencies''')
code(s6a(2))

md(r'''## Cell 1 — Config (n_target raised 15000 -> 30000, everything else identical)''')
code(s6a(3).replace("n_target:         int   = 6000      # entries to download (of ~127k available)",
                     "n_target:         int   = 30000     # entries to download (of ~127k available) -- 5x Phase6a, 2x Phase28"))

md(r'''## Cell 2 — Select entries via the RCSB Search API (identical query to Phase 6a/28)''')
code(s6a(5))

md(r'''## Cell 3 — Download (identical to Phase 6a/28)''')
code(s6a(7))

md(r'''## Cell 4 — Parse and label (identical predicates to Phase 6a/28)''')
code(s6a(9))

md(r'''## Cell 5 — MMseqs2 clustering and representative selection (identical to Phase 6a/28)''')
code(s6a(11))

md(r'''## Cell 6 — Build graphs (identical schema to Phase 6a/28/Phase 1)''')
code(s6a(13))

md(r'''## Cell 7 — Splits, QC, save (identical logic, new output filename)''')
code(s6a(15).replace('p = OUT / "phase6_graphs.pt"', 'p = OUT / "phase35_graphs.pt"'))

md(r'''
---

## What this produces

`phase35_graphs.pt` -- same schema as `phase6_graphs.pt`/`phase28_graphs.pt`,
just a bigger pool. Next: Phase 36 (topology at this scale, mirrors
Phase 6b/29) and Phase 37 (ESM2 + dihedrals, mirrors Phase 10/30).
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase35_scaleup2_ingestion.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
