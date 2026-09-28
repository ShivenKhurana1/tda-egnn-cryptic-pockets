"""Phase 279: is there unused, already-in-hand training data, and can we tell which of CryptoBench's OWN 700 training chains are genuinely
cryptic vs already-open (pre-formed), which nobody -- including this project until now -- has ever labeled?

An independent benchmark strictly relabeled CryptoBench's 182-chain TEST split by an apo/holo cavity-voxel rule (Phase 276) but never touched
the 700-chain TRAIN split, so this project's CryptoBench fine-tune has always been trained on an unknown mix of genuinely-cryptic and
already-open sites. This script builds a cheap geometric proxy for that same distinction -- no cavity-detection software needed -- and
calibrates it against the 180 test chains we DO have ground-truth labels for, before ever applying it to the (unlabeled) training chains.

Proxy: for each apo/holo pair's site residues, count heavy atoms within 8 A of each site residue's CA, averaged over site residues, computed
independently in the apo structure and in the holo structure (no cross-structure alignment needed, since this is a same-structure local density,
not a displacement). A genuinely cryptic site should be relatively packed/buried in apo (higher atom count) and relatively open in holo (lower
atom count, since the ligand -- not counted -- has displaced protein atoms to form the pocket); a pre-formed site should look similarly open in
both. proxy_score = apo_density - holo_density (larger = more consistent with "closed in apo, open in holo").

Run from cryptic-pocket/: python3 phase279/crypticity_proxy.py
"""
import gzip, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1] / "phase276"
RAW = Path(__file__).resolve().parent / "raw"; RAW.mkdir(exist_ok=True)
R = 8.0
AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE"}

d = json.load(open(BASE / "cryptobench_dataset.json"))
labels = json.load(open(BASE / "phase276a_crypticity_labels.json"))          # key "{pdb}_{chain}" -> {label, apo_vox, holo_vox, uniprot}
folds = json.load(open(BASE / "cryptobench_folds.json"))
train_pdbs = set(folds["train-0"]) | set(folds["train-1"]) | set(folds["train-2"]) | set(folds["train-3"])

def main_entries(pdbs):
    out = []
    for pdb in pdbs:
        for e in d.get(pdb, []):
            if e.get("is_main_holo_structure"):
                out.append((pdb, e))
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

def _parse_cif(text, chain_id):
    lines = text.splitlines()
    cols = [l.split(".")[1].strip() for l in lines if l.startswith("_atom_site.")]
    ix = {c: i for i, c in enumerate(cols)}
    need = ["group_PDB", "label_atom_id", "label_comp_id", "label_asym_id", "auth_asym_id", "auth_seq_id",
            "pdbx_PDB_ins_code", "pdbx_PDB_model_num", "type_symbol", "Cartn_x", "Cartn_y", "Cartn_z"]
    if not all(k in ix for k in need): return {}, np.zeros((0, 3))
    by_res = {}          # (resnum) -> list of (atom_name, element, xyz)
    for l in lines:
        if not l.startswith(("ATOM", "HETATM")): continue
        f = l.split()
        if len(f) <= max(ix.values()): continue
        chain = f[ix["auth_asym_id"]]
        if chain != chain_id: continue
        if f[ix["pdbx_PDB_model_num"]] != "1": continue
        resname = f[ix["label_comp_id"]].strip().upper()
        if resname not in AA3: continue
        ic = f[ix["pdbx_PDB_ins_code"]].strip(); ic = "" if ic == "?" or ic == "." else ic
        rn = f[ix["auth_seq_id"]].strip() + ic
        xyz = (float(f[ix["Cartn_x"]]), float(f[ix["Cartn_y"]]), float(f[ix["Cartn_z"]]))
        by_res.setdefault(rn, []).append((f[ix["label_atom_id"]].strip(), f[ix["type_symbol"]].strip().upper(), xyz))
    ca_by_res, heavy = {}, []
    for rn, atoms in by_res.items():
        ca = next((xyz for name, el, xyz in atoms if name == "CA"), None)
        if ca is None: continue
        ca_by_res[rn] = np.array(ca)
        heavy.extend(xyz for name, el, xyz in atoms if el not in ("H", "D"))
    return ca_by_res, np.asarray(heavy, float) if heavy else np.zeros((0, 3))

def parse_ca_and_heavy(path, chain_id):
    with gzip.open(path, "rt") as fh: text = fh.read()
    return _parse_cif(text, chain_id)

def local_density(ca_by_res, heavy, resnums, radius=R):
    if len(heavy) == 0 or not resnums: return np.nan
    tree = cKDTree(heavy)
    counts = []
    for rn in resnums:
        ca = ca_by_res.get(rn)
        if ca is None: continue
        counts.append(len(tree.query_ball_point(ca, radius)))
    return float(np.mean(counts)) if counts else np.nan

def score_entry(pdb, e):
    apo_ch, holo_pdb, holo_ch = e["apo_chain"], e["holo_pdb_id"].lower(), e["holo_chain"]
    site_apo = [s.split("_", 1)[1] for s in e["apo_pocket_selection"]]
    site_holo = [s.split("_", 1)[1] for s in e["holo_pocket_selection"]]
    _, apo_path = fetch(pdb); _, holo_path = fetch(holo_pdb)
    if apo_path is None or holo_path is None: return None
    try:
        ca_a, heavy_a = parse_ca_and_heavy(apo_path, apo_ch)
        ca_h, heavy_h = parse_ca_and_heavy(holo_path, holo_ch)
        da = local_density(ca_a, heavy_a, site_apo)
        dh = local_density(ca_h, heavy_h, site_holo)
        if np.isnan(da) or np.isnan(dh): return None
        return da - dh, da, dh
    except Exception:
        return None

if __name__ == "__main__":
    # --- calibration: score every labelled TEST chain, check AUC against the known cryptic/preformed ground truth ---
    test_pdbs = folds["test"]
    test_entries = main_entries(test_pdbs)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda pe: (fetch(pe[0]), fetch(pe[1]["holo_pdb_id"].lower())), test_entries))
    rows = []
    for pdb, e in test_entries:
        key = f"{pdb}_{e['apo_chain']}"
        lab = labels.get(key, {}).get("label")
        if lab not in ("cryptic", "preformed"): continue
        r = score_entry(pdb, e)
        if r is None: continue
        rows.append(dict(key=key, label=lab, score=r[0], apo_density=r[1], holo_density=r[2]))
    y = np.array([1 if r["label"] == "cryptic" else 0 for r in rows])
    s = np.array([r["score"] for r in rows])
    print(f"calibration set: {len(rows)} labelled chains ({int(y.sum())} cryptic, {int((1-y).sum())} preformed)")
    print(f"proxy AUC (cryptic vs preformed), higher score = more cryptic-like: {roc_auc_score(y, s):.3f}")
    print(f"mean score  cryptic={s[y==1].mean():.2f}  preformed={s[y==0].mean():.2f}")
    json.dump(rows, open("phase279_calibration.json", "w"), indent=1)
