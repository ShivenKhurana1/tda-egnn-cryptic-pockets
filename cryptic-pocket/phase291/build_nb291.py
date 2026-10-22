#!/usr/bin/env python3
"""Phase 291: Phase 266c (homology-clean 3B transfer, 8 seeds) retrained on CryptoBench's OFFICIAL residue labels.

Phase 42/45/48 ingestion labelled each apo chain with the pocket of its is_main_holo_structure pairing only. CryptoBench's
released labels (create-annotations.py in the CryptoBench repo), and the published CryptoBench and Seq2Pocket numbers,
use the UNION of every holo pairing's apo_pocket_selection. 20% of official positives (507/1107 apo entries) were
therefore trained as negatives. This kernel changes only y:
  1. re-parse every apo chain with Phase 48's exact residue rule (parse_unit), rebuild the main-pocket labels and ASSERT
     they equal the stored d.y bit-for-bit (proves the residue order and numbering are identical);
  2. replace d.y with the union over every holo pairing on the same apo chain (keep d.y_main for analysis);
  3. recompute POS_WEIGHT with Phase 48's formula on the new labels.
true_center (the localization target, main pocket) is unchanged. Same checkpoints, seeds, split, loss, optimiser and
early stopping as 266c. Predictions save y (official) and y_main for val and test.
usage: build_nb291.py <output_root>"""
import json, sys
from pathlib import Path

ARM = "full"; ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "291"
HERE = Path(__file__).resolve().parent
src266 = (HERE.parent / "phase266" / "build_nb266c.py").read_text()
nb129 = json.load(open(HERE.parents[1] / "archived scripts" / "phase129_cryptobench_esm3b_transfer.ipynb"))
s129 = lambda i: "".join(nb129["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md('''
# Phase 291 - CryptoBench 3B transfer (Phase 266c recipe) on the OFFICIAL union-of-pockets residue labels

Everything is Phase 266c except the labels: each apo chain's positives become the union of every holo pairing's
`apo_pocket_selection` (CryptoBench's released annotation), instead of the main pairing only. The relabel cell first
proves residue identity by rebuilding the old main-only labels and requiring an exact match with the stored ones.
''')
md("## Cell 1 - Install")
code("!pip install -q torch-geometric==2.6.1 biopython==1.84")
md("## Cell 2 - Setup: Phase 48's CryptoBench-3B blob + Phase 227's geometry (Phase 266c, verbatim)")
cell2 = src266[src266.index("code(s129(4) + r'''") + len("code(s129(4) + r'''"):]
cell2 = cell2[:cell2.index("''')")]
code(s129(4) + cell2)
md("## Cell 3 - Relabel with the official union-of-pockets labels (verified against the stored main-pocket labels first)")
code(r'''
import gzip, urllib.request
from concurrent.futures import ThreadPoolExecutor
from Bio.PDB import MMCIFParser, PDBParser

def fetch_json(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r: return json.loads(r.read().decode())
        except Exception as e:
            print("retry", url, e); time.sleep(10 * (i + 1))
    raise RuntimeError(url)
cb_dataset = fetch_json("https://osf.io/download/ta2ju/")

RAW = Path("/tmp/pdb_raw"); RAW.mkdir(parents=True, exist_ok=True)
def fetch(pid):                                   # Phase 48 cell 6, verbatim order: cif first, then pdb
    for ext in ("cif", "pdb"):
        p = RAW / f"{pid}.{ext}.gz"
        if p.exists() and p.stat().st_size > 500: return pid, p
    for ext in ("cif", "pdb"):
        for attempt in range(3):
            try:
                with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.{ext}.gz", timeout=60) as r: b = r.read()
                if len(b) > 500: p = RAW / f"{pid}.{ext}.gz"; p.write_bytes(b); return pid, p
                break
            except Exception: time.sleep(2 * (attempt + 1))
    return pid, None
need = sorted({str(d.pdb_id) for d in dataset})
with ThreadPoolExecutor(max_workers=16) as ex: paths = {p: f for p, f in ex.map(fetch, need) if f}
print(f"fetched {len(paths)}/{len(need)} structures")

AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE"}
def resnums_of(pid, cid):                         # Phase 48 parse_unit residue rule, verbatim
    path = paths[pid]
    parser = MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True, PERMISSIVE=True)
    with gzip.open(path, "rt") as fh: model = next(parser.get_structure(pid, fh).get_models())
    return [str(r.id[1]) + r.id[2].strip() for r in model[cid] if r.get_resname().strip().upper() in AA3 and "CA" in r]

mismatch, n_main, n_off, changed = [], 0, 0, 0
for d in dataset:
    pid, cid = str(d.pdb_id), str(d.chain_id)
    ents = cb_dataset[pid]
    main = next(e for e in ents if e.get("is_main_holo_structure"))
    assert main["apo_chain"] == cid
    rn = resnums_of(pid, cid)
    main_set = {s.split("_", 1)[1] for s in main["apo_pocket_selection"]}
    y_main = torch.tensor([1.0 if r in main_set else 0.0 for r in rn])
    if len(rn) != d.num_nodes or not torch.equal(y_main, d.y.float()):
        mismatch.append(pid); continue
    off_set = set(main_set)
    for e in ents:
        if e["apo_chain"] != cid: continue
        off_set |= {s.split("_", 1)[1] for s in e["apo_pocket_selection"] if s.split("_", 1)[0] == cid}
    d.y_main = d.y.clone()
    d.y = torch.tensor([1.0 if r in off_set else 0.0 for r in rn], dtype=d.y.dtype)
    n_main += int(d.y_main.sum()); n_off += int(d.y.sum()); changed += int(not torch.equal(d.y, d.y_main))
assert not mismatch, f"{len(mismatch)} chains failed the main-label reproduction check: {mismatch[:20]}"
print(f"all {len(dataset)} chains reproduce the stored main labels exactly")
print(f"positives main {n_main:,} -> official {n_off:,} (+{n_off/n_main-1:.1%}); chains changed {changed}")
for s in ("train", "val", "test"):
    a = sum(int(d.y_main.sum()) for d in splits[s]); b = sum(int(d.y.sum()) for d in splits[s])
    print(f"  {s:5s} {len(splits[s])} chains: positives {a:,} -> {b:,}")
n_res_ = sum(d.num_nodes for d in dataset); n_pos_ = sum(int(d.y.sum()) for d in dataset)
print(f"POS_WEIGHT {POS_WEIGHT:.3f} -> ", end="")
POS_WEIGHT = (n_res_ - n_pos_) / max(n_pos_, 1)     # Phase 48 formula, on the new labels
print(f"{POS_WEIGHT:.3f}")
''')
md("## Cell 4 - Model (Phase 129, verbatim)")
code(s129(6))
md("## Cell 5 - Training machinery (Phase 129, width-parameterised; fit() loads the seed-matched pretrained state)")
code(s129(8).replace("LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head)", "LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=use_loc_head)"))
md("## Cell 6 - Fine-tune x 8 seeds from the Phase 266b seed-matched checkpoints; save probabilities, centres, both label sets")
code(f'''
ARM = "{ARM}"
''' + r'''
KEEP_SIDE = {"full": list(range(TOPO_DIM)), "geom": list(range(TOPO_DIM, SIDE_DIM)), "geom+topo": list(range(SIDE_DIM))}
m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE[ARM]] = 1.0
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_

@torch.no_grad()
def predict(model, split):
    out = {}
    for d in splits[split]:
        b = d.clone().to(DEVICE); logits, pc = model(b)
        out[(str(d.pdb_id).lower(), str(d.chain_id))] = dict(p=torch.sigmoid(logits).cpu().numpy().astype(np.float32),
                                                             c=pc.cpu().numpy()[0], y=d.y.numpy().astype(np.int8),
                                                             y_main=d.y_main.numpy().astype(np.int8), tc=d.true_center.numpy())
    return out

BEST_LAMBDA = 0.1
PREDS = {"arm": ARM, "labels": "official_union", "pos_weight": POS_WEIGHT, "seeds": {}}; t0 = time.time()
for seed in range(8):
    ck = locate(f"pretrain266_{ARM.replace('+','_')}_seed{seed}.pt")
    model, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, pretrained_path=ck)
    PREDS["seeds"][seed] = {"val": predict(model, "val"), "test": predict(model, "test"), "val_pr": float(val_pr)}
    torch.save(PREDS, WORK / "phase291_official_preds.pt")
    print(f"  seed {seed}: val PR (official labels) {val_pr:.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)
print("done")
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase291_cb3b_official_labels.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = {"id": "blackpearls1/phase291-cb3b-official-labels", "title": "Phase291 Cb3b Official Labels", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase48-cryptobench-esm3b", "blackpearls1/phase227-geometry-ingestion", "blackpearls1/phase266b-pretrain-3b-clean"]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
