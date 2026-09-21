#!/usr/bin/env python3
"""Phase 28: Thread 3, training-data scale-up, step 1 of 3 (ingestion).
Mirrors Phase 6a's pipeline exactly (same RCSB query, same seed=1337 shuffle,
same parse/cluster/graph-build logic -- all proven code, spliced verbatim)
but with n_target raised 6000 -> 15000 (2.5x). Because Phase 6a used
`rng.shuffle(all_ids); target_ids = sorted(all_ids[:n_target])` with the same
seed, the original 6000-entry sample is a STRICT SUBSET of this 15000-entry
sample -- this is a genuine superset scale-up, not an unrelated resample."""
import json, pickle
from pathlib import Path

C6a = pickle.load(open("/tmp/p6a_cells.pkl", "rb"))
s6a = lambda i: (C6a[i]["source"] if isinstance(C6a[i]["source"], str) else "".join(C6a[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 28 — Thread 3, Step 1: Scale-Up Ingestion (2.5x)

Every downstream localization phase (14 onward) trains on the 2,056-cluster
pool Phase 6a/6b/10 built. This notebook is step 1 of scaling that pool
2-3x: raises `n_target` from 6000 to 15000 PDB entries in Phase 6a's exact
RCSB Search API query (same resolution/ligand/size filters), same
`seed=1337` shuffle -- so the original 6000-entry sample is a **strict
subset** of this one, making the scale-up additive rather than a fresh,
unrelated resample. Same parse/label/MMseqs2-cluster/graph-build logic,
spliced verbatim from Phase 6a (proven code, no changes except the target
count and output filename).

Expect roughly 2.5x the chains and clusters (2,056 -> ~5,100), and
correspondingly ~2.5x the download/parse/cluster time (Phase 6a: well under
an hour for this stage; topology and ESM2 embedding are the expensive stages
and are separate follow-up notebooks, mirroring Phase 6b and Phase 9/10).
''')

md(r'''## Cell 0 — Install dependencies''')
code(s6a(2))

md(r'''## Cell 1 — Config (n_target raised 6000 -> 15000, everything else identical)''')
code(s6a(3).replace("n_target:         int   = 6000      # entries to download (of ~127k available)",
                     "n_target:         int   = 15000     # entries to download (of ~127k available) -- 2.5x Phase6a"))

md(r'''## Cell 2 — Select entries via the RCSB Search API (identical query to Phase 6a)''')
code(s6a(5))

md(r'''## Cell 3 — Download (identical to Phase 6a)''')
code(s6a(7))

md(r'''## Cell 4 — Parse and label (identical predicates to Phase 6a)''')
code(s6a(9))

md(r'''## Cell 5 — MMseqs2 clustering and representative selection (identical to Phase 6a)''')
code(s6a(11))

md(r'''## Cell 6 — Build graphs (identical schema to Phase 6a/Phase 1)''')
code(s6a(13))

md(r'''## Cell 7 — Splits, QC, save (identical logic, new output filename)''')
code(s6a(15).replace('p = OUT / "phase6_graphs.pt"', 'p = OUT / "phase28_graphs.pt"'))

md(r'''
---

## What this produces

`phase28_graphs.pt` -- same schema as `phase6_graphs.pt` (dataset, clusters,
reps, node/edge feat dims, pos_weight, aa_alphabet, config), just a bigger
pool. Next: Phase 29 (topology at this scale, mirrors Phase 6b) and Phase 30
(ESM2 + dihedrals, mirrors Phase 9/10) before this pool is trainable end to
end the way `phase10_graphs.pt` is today.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase28_scaleup_ingestion.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
