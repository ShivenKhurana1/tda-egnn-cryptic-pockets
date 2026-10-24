"""Write every CryptoBench apo chain (main pairing, single-letter chain) as a PDB file with the Phase 48 residue rule, so Foldseek's residue
order equals the model's node order; also write per-chain official labels, fold and pRMSD to chains.json."""
import gzip, json
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select
H = Path(__file__).resolve().parent; CP = H.parent; RAW = CP / "phase292" / "raw"; OUT = Path("/tmp/fsdb/in"); OUT.mkdir(parents=True, exist_ok=True)
DS = json.load(open(CP / "cryptobench_dataset.json")); MT = json.load(open(CP / "phase292" / "motion_targets.json")); F = json.load(open(H / "folds.json"))
fold = {p: ("test" if k == "test" else int(k.split("-")[1])) for k, v in F.items() for p in v}
AA1 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE"}
info = {}
for pid, rec in MT.items():
    c = rec["chain"]; p = next(RAW.glob(f"{pid}.*.gz"))
    parser = MMCIFParser(QUIET=True) if ".cif" in p.name else PDBParser(QUIET=True, PERMISSIVE=True)
    with gzip.open(p, "rt") as fh: st = parser.get_structure(pid, fh)
    m0 = next(st.get_models()); res = [r for r in m0[c] if r.get_resname().strip().upper() in AA1 and "CA" in r]; keep = {id(r) for r in res}
    assert [str(r.id[1]) + r.id[2].strip() for r in res] == rec["resnums"]
    chain = m0[c]
    class Sel(Select):
        def accept_model(self, mo): return mo is m0
        def accept_chain(self, ch): return ch is chain
        def accept_residue(self, r): return id(r) in keep
    dest = OUT / f"{pid}_{c}.pdb"
    if not dest.exists():
        if len(c) > 1:                                   # PDB format allows one-letter chain ids; residues are unchanged
            used = {ch.id for ch in m0}; chain.id = next(x for x in "zyxwvutsrqponmlkjihgfedcba" if x not in used)
        io = PDBIO(); io.set_structure(st); io.save(str(dest), Sel())
    ents = DS[pid]; main = next(e for e in ents if e.get("is_main_holo_structure"))
    sel = {s.split("_", 1)[1] for e in ents if e["apo_chain"] == c for s in e["apo_pocket_selection"] if s.split("_", 1)[0] == c}
    info[f"{pid}_{c}"] = dict(fold=fold.get(pid), prmsd=float(main["pRMSD"]), uniprot=main["uniprot_id"],
                              y=[int(r in sel) for r in rec["resnums"]])
json.dump(info, open(H / "chains.json", "w"))
from collections import Counter; print(len(info), "chains written;", Counter(str(v["fold"]) for v in info.values()))
