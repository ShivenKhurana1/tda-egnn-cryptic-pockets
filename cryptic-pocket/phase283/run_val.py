"""Phase 283 (validation extension): compute the apo-side cavity-voxel crypticity-proxy score for the EXACT 60 validation chains used by
the headline model's cached predictions (phase266_full_preds.pt, seeds[*]['val'].keys()) -- not a re-derivation of the val split, the actual
identity list, dumped to /tmp/p283_val_keys.json as [{"pdb":..., "chain":...}, ...]. No labels exist for these chains (that's the whole
point -- the external audit never covered them); this only produces our own proxy score, to be used for crypticity-proxy-stratified
calibration, not as ground truth.

usage: python3 run_val.py <val_keys.json> <out.json>
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_cavity import d as dataset, fetch, parse_cif
from cavity_voxels import grid_voxels, site_voxel_count

keys_path, OUT = sys.argv[1], sys.argv[2]
targets = json.load(open(keys_path))
print(f"processing {len(targets)} validation chains")

need_pdbs = sorted({t["pdb"] for t in targets})
t0 = time.time()
with ThreadPoolExecutor(16) as ex: list(ex.map(fetch, need_pdbs))
print(f"fetched {len(need_pdbs)} structures in {(time.time()-t0)/60:.1f} min")


def entry_for(pdb, chain):
    for e in dataset.get(pdb, []):
        if e.get("apo_chain") == chain and e.get("is_main_holo_structure"):
            return e
    for e in dataset.get(pdb, []):
        if e.get("apo_chain") == chain:
            return e
    return None


def process_val(pdb, chain):
    e = entry_for(pdb, chain)
    if e is None:
        return dict(key=f"{pdb}_{chain}", error="no matching dataset entry")
    _, apo_path = fetch(pdb)
    if apo_path is None:
        return dict(key=f"{pdb}_{chain}", error="fetch failed")
    try:
        apo_atoms, apo_res = parse_cif(apo_path, chain)
        if apo_atoms is None or len(apo_atoms) < 20:
            return dict(key=f"{pdb}_{chain}", error="too few atoms")
        site_resnums = [s.split("_", 1)[1] for s in e["apo_pocket_selection"]]
        site_pts = np.concatenate([apo_res[rn] for rn in site_resnums if rn in apo_res]) if any(rn in apo_res for rn in site_resnums) else None
        if site_pts is None or len(site_pts) == 0:
            return dict(key=f"{pdb}_{chain}", error="no site atoms found")
        apo_voxels = grid_voxels(apo_atoms)
        my_apo = site_voxel_count(apo_voxels, site_pts, radius=4.0)
        return dict(key=f"{pdb}_{chain}", pdb=pdb, chain=chain, my_apo=my_apo, n_site_atoms=len(site_pts))
    except Exception as ex:
        return dict(key=f"{pdb}_{chain}", error=f"{type(ex).__name__}: {ex}")


rows = []
t0 = time.time()
for i, t in enumerate(targets):
    r = process_val(t["pdb"], t["chain"])
    rows.append(r)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(targets)}  [{(time.time()-t0)/60:.1f}m]", flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
ok = [r for r in rows if "error" not in r]
print(f"done: {len(ok)}/{len(targets)} succeeded, {len(rows)-len(ok)} errors, in {(time.time()-t0)/60:.1f} min")
for r in rows:
    if "error" in r:
        print("  FAIL", r["key"], r["error"])
