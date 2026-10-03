"""Rebuild each chain's CA trace in the SAME residue order the model scored (phase42 parse_unit rule: residues of
the chain whose name is a standard AA/MSE and that have a CA atom), centred on the CA mean exactly as the graphs
were (pos = ca - ca.mean). Predicted/true pocket centres in the cached predictions live in that same centred frame.
Chains whose residue count does not equal the model's output length are dropped from the 3D view, not forced."""
import gzip, json, sys, urllib.request, io
from pathlib import Path
import numpy as np
from Bio.PDB import MMCIFParser
RAW = Path("/Users/shivenkhurana/bioproj/cryptic-pocket/phase283/raw")
AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L",
       "LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
D = json.load(open("data.json"))
recs = {r["key"]: r for r in D["records"]}
import torch
P = torch.load("/tmp/p266c_fresh/phase266_full_preds.pt", map_location="cpu", weights_only=False)
cen = {}
for split in ("val","test"):
    for k,v in P["seeds"][0][split].items():
        key=f"{k[0]}_{k[1]}"
        cen[key]=dict(tc=np.asarray(v["tc"]).reshape(-1), c=np.mean([P["seeds"][s][split][k]["c"] for s in range(8)],0))
parser = MMCIFParser(QUIET=True)
def cif(pid):
    p = RAW/f"{pid}.cif.gz"
    if p.exists() and p.stat().st_size>500: return p
    with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid.upper()}.cif.gz", timeout=60) as r: p.write_bytes(r.read())
    return p
out, bad = {}, []
for key, r in recs.items():
    try:
        with gzip.open(cif(r["pdb"]),"rt") as fh: model = next(parser.get_structure(r["pdb"], fh).get_models())
        ch = model[r["chain"]]
        res = [x for x in ch if x.get_resname().strip().upper() in AA3 and "CA" in x]
    except Exception as e:
        bad.append((key, f"parse:{type(e).__name__}")); continue
    if len(res) != r["n"]:
        bad.append((key, f"len {len(res)} vs {r['n']}")); continue
    ca = np.asarray([x["CA"].coord for x in res], dtype=np.float64)
    mean = ca.mean(0); pos = ca - mean
    out[key] = dict(xyz=[round(float(v),1) for v in pos.reshape(-1)],
                    aa="".join(AA3[x.get_resname().strip().upper()] for x in res),
                    resi=[int(x.id[1]) for x in res],
                    c=[round(float(v),1) for v in cen[key]["c"]], tc=[round(float(v),1) for v in cen[key]["tc"]])
print(f"{len(out)}/{len(recs)} chains reconstructed; {len(bad)} dropped")
for b in bad[:12]: print("  ", b)
json.dump(out, open("coords.json","w"), separators=(",",":"))
import os; print("coords.json MB:", round(os.path.getsize("coords.json")/1e6,2))
