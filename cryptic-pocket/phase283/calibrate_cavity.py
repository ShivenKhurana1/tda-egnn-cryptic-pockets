"""Phase 283: build apo/holo cavity-voxel counts with our own grid-based detector for every CryptoBench chain the external crypticity audit
DID label (test split, ~180 chains, via cryptobench_dataset.json + phase276a_crypticity_labels.json), and check how well they correlate with
the external tool's own published apo_vox/holo_vox counts -- before trusting this method on the (unlabelled) validation or training splits.
Nothing here is applied to validation until this calibration step passes a real check (Spearman correlation with their counts; how well a
threshold or classifier fit on our own counts reproduces their published cryptic/preformed/neither labels, via k-fold cross-validation
within the labelled set so the check is not trivially overfit).

usage: python3 calibrate_cavity.py <n_chains|all> <out_prefix>
"""
import gzip, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cavity_voxels import grid_voxels, site_voxel_count

BASE = Path(__file__).resolve().parents[1] / "phase276"
RAW = Path(__file__).resolve().parent / "raw"; RAW.mkdir(exist_ok=True)
AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE"}

d = json.load(open(BASE / "cryptobench_dataset.json"))
labels = json.load(open(BASE / "phase276a_crypticity_labels.json"))
folds = json.load(open(BASE / "cryptobench_folds.json"))


def main_entries_for(pdbs):
    out = []
    for pdb in pdbs:
        for e in d.get(pdb, []):
            if e.get("is_main_holo_structure"): out.append((pdb, e))
    return out


def fetch(pid):
    for ext in ("cif", "pdb"):
        p = RAW / f"{pid}.{ext}.gz"
        if p.exists() and p.stat().st_size > 500: return pid, p
        try:
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid.upper()}.{ext}.gz", timeout=60) as r: b = r.read()
            if len(b) > 500: p.write_bytes(b); return pid, p
        except Exception: continue
    return pid, None


def parse_cif(path, chain_id):
    with gzip.open(path, "rt") as fh: text = fh.read()
    lines = text.splitlines()
    cols = [l.split(".")[1].strip() for l in lines if l.startswith("_atom_site.")]
    ix = {c: i for i, c in enumerate(cols)}
    need = ["group_PDB", "label_comp_id", "auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code", "pdbx_PDB_model_num", "type_symbol", "Cartn_x", "Cartn_y", "Cartn_z"]
    if not all(k in ix for k in need): return None, {}
    atoms, res_atoms = [], {}
    for l in lines:
        if not l.startswith(("ATOM", "HETATM")): continue
        f = l.split()
        if len(f) <= max(ix.values()): continue
        if f[ix["auth_asym_id"]] != chain_id or f[ix["pdbx_PDB_model_num"]] != "1": continue
        if f[ix["type_symbol"]].strip().upper() in ("H", "D"): continue
        resname = f[ix["label_comp_id"]].strip().upper()
        if resname not in AA3: continue
        xyz = (float(f[ix["Cartn_x"]]), float(f[ix["Cartn_y"]]), float(f[ix["Cartn_z"]]))
        ic = f[ix["pdbx_PDB_ins_code"]].strip(); ic = "" if ic in ("?", ".") else ic
        rn = f[ix["auth_seq_id"]].strip() + ic
        atoms.append(xyz); res_atoms.setdefault(rn, []).append(xyz)
    return (np.array(atoms) if atoms else None), res_atoms


def parse_ligand(path, chain_id, resname, resnum):
    with gzip.open(path, "rt") as fh: text = fh.read()
    lines = text.splitlines()
    cols = [l.split(".")[1].strip() for l in lines if l.startswith("_atom_site.")]
    ix = {c: i for i, c in enumerate(cols)}
    if "label_comp_id" not in ix: return None
    out = []
    for l in lines:
        if not l.startswith("HETATM"): continue
        f = l.split()
        if len(f) <= max(ix.values()): continue
        if f[ix["auth_asym_id"]] != chain_id: continue
        if f[ix["label_comp_id"]].strip().upper() != resname.upper(): continue
        if f[ix["auth_seq_id"]].strip() != str(resnum): continue
        if f[ix["type_symbol"]].strip().upper() in ("H", "D"): continue
        out.append((float(f[ix["Cartn_x"]]), float(f[ix["Cartn_y"]]), float(f[ix["Cartn_z"]])))
    return np.array(out) if out else None


def process(pdb, e, apo_key, require_label=True):
    lab = labels.get(apo_key)
    if require_label and (lab is None or lab["label"] not in ("cryptic", "preformed")): return None
    apo_ch = e["apo_chain"]
    _, apo_path = fetch(pdb)
    if apo_path is None: return None
    try:
        apo_atoms, apo_res = parse_cif(apo_path, apo_ch)
        if apo_atoms is None or len(apo_atoms) < 20: return None
        site_resnums = [s.split("_", 1)[1] for s in e["apo_pocket_selection"]]
        site_pts = np.concatenate([apo_res[rn] for rn in site_resnums if rn in apo_res]) if any(rn in apo_res for rn in site_resnums) else None
        if site_pts is None or len(site_pts) == 0: return None
        apo_voxels = grid_voxels(apo_atoms)
        my_apo = site_voxel_count(apo_voxels, site_pts, radius=4.0)
        out = dict(key=apo_key, my_apo=my_apo, n_site_atoms=len(site_pts))
        if lab is not None: out.update(label=lab["label"], their_apo=lab["apo_vox"], their_holo=lab["holo_vox"])
        return out
    except Exception as ex:
        return dict(key=apo_key, error=f"{type(ex).__name__}: {ex}")


if __name__ == "__main__":
    mode, n_arg, OUT = sys.argv[1], sys.argv[2], sys.argv[3]     # mode: "calibrate" (labelled test chains) or "apply" (a fold name, e.g. "val")
    if mode == "calibrate":
        fold_pdbs = folds["test"]
        entries = main_entries_for(fold_pdbs)
        targets = []
        for pdb, e in entries:
            key = f"{pdb}_{e['apo_chain']}"
            if key in labels and labels[key]["label"] in ("cryptic", "preformed"):
                targets.append((pdb, e, key))
        require_label = True
    else:
        fold_pdbs = folds[mode]
        entries = main_entries_for(fold_pdbs)
        targets = [(pdb, e, f"{pdb}_{e['apo_chain']}") for pdb, e in entries]
        require_label = False
    if n_arg != "all": targets = targets[: int(n_arg)]
    print(f"[{mode}] processing {len(targets)} chains")

    need_pdbs = sorted({t[0] for t in targets})
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex: list(ex.map(fetch, need_pdbs))
    print(f"fetched {len(need_pdbs)} structures in {(time.time()-t0)/60:.1f} min")

    rows = []
    t0 = time.time()
    for i, (pdb, e, key) in enumerate(targets):
        r = process(pdb, e, key, require_label=require_label)
        if r: rows.append(r)
        if (i + 1) % 10 == 0: print(f"  {i+1}/{len(targets)}  [{(time.time()-t0)/60:.1f}m]", flush=True)
    json.dump(rows, open(f"{OUT}_{mode}_rows.json", "w"), indent=1)
    ok = [r for r in rows if "error" not in r]
    print(f"done: {len(ok)}/{len(targets)} succeeded, {len(rows)-len(ok)} errors, in {(time.time()-t0)/60:.1f} min")
    for r in rows:
        if "error" in r: print("  FAIL", r["key"], r["error"])
