"""Phase 283 (training-split extension): compute the apo-side cavity-voxel crypticity-proxy score for every chain in CryptoBench's official
training pool (folds.json train-0..3 combined, 885 pdbs -- this pool is what the ~700 actual training chains and the 60 validation chains
were both drawn from; see Phase 283's validation run for how the val subset's identity was separately recovered from cached predictions).
No labels exist for any of this (that's the whole point of the extension); this only produces our own proxy score per (pdb,chain), meant to
be joined by a future training run against whatever chains it actually uses, to test the crypticity-aware up-weighting idea flagged in
paper.md's Future Work. Reuses calibrate_cavity.py's process()/main_entries_for()/fetch() unchanged (require_label=False, same as its
existing "apply to a fold" path) -- this script only merges all 4 train folds' pdbs into one job so fetching parallelizes across all of them
at once instead of four separate serial runs.

usage: python3 run_train.py <out.json>
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_cavity import folds, main_entries_for, fetch, process

OUT = sys.argv[1]
fold_pdbs = sorted(set(folds["train-0"]) | set(folds["train-1"]) | set(folds["train-2"]) | set(folds["train-3"]))
entries = main_entries_for(fold_pdbs)
targets = [(pdb, e, f"{pdb}_{e['apo_chain']}") for pdb, e in entries]
print(f"[train] processing {len(targets)} chains from {len(fold_pdbs)} pdbs (train-0..3 union)")

need_pdbs = sorted({t[0] for t in targets})
t0 = time.time()
with ThreadPoolExecutor(16) as ex:
    list(ex.map(fetch, need_pdbs))
print(f"fetched {len(need_pdbs)} structures in {(time.time()-t0)/60:.1f} min")

rows = []
t0 = time.time()
for i, (pdb, e, key) in enumerate(targets):
    r = process(pdb, e, key, require_label=False)
    if r:
        rows.append(r)
    if (i + 1) % 25 == 0:
        json.dump(rows, open(OUT, "w"), indent=1)  # incremental save, this is a long-running job
        print(f"  {i+1}/{len(targets)}  [{(time.time()-t0)/60:.1f}m]", flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
ok = [r for r in rows if "error" not in r]
print(f"done: {len(ok)}/{len(targets)} succeeded, {len(rows)-len(ok)} errors, in {(time.time()-t0)/60:.1f} min")
for r in rows:
    if "error" in r:
        print("  FAIL", r["key"], r["error"])
