import json, numpy as np
from pathlib import Path
DATA = {r["key"]: r for r in json.load(open("../site/data.json"))["records"] if r["split"] == "test"}
cry = {k for k, v in json.load(open("../phase276/phase276a_crypticity_labels.json")).items() if v["label"] == "cryptic"}
rows = []
for k, r in DATA.items():
    f = Path("cache") / f"{k}.npz"
    if not f.exists(): continue
    z = np.load(f); ca, lig = z["ca"], z["lig"]
    y = np.array(r["y"], bool)
    if len(z["ids"]) == 0: rows.append(dict(key=k, cryptic=k in cry, entries=0, usable_all=0.0, usable_pocket=0.0, pocket_ge2=False, pocket_ge3=False)); continue
    ok = (~np.isnan(ca[:, :, 0])) & (~lig)                        # entries x residues: mapped and unliganded at that residue
    per_res = ok.sum(0)                                            # usable entries per residue
    pk = per_res[y] if y.any() else per_res
    rows.append(dict(key=k, cryptic=k in cry, entries=int(len(z["ids"])), usable_all=float(np.median(per_res)), usable_pocket=float(np.median(pk)),
                     frac_pocket_ge2=float((pk >= 2).mean()), pocket_ge2=bool(np.median(pk) >= 2), pocket_ge3=bool(np.median(pk) >= 3)))
json.dump(rows, open("coverage_summary.json", "w"), indent=1)
def s(name, rs):
    if not rs: return
    a = np.array([r["usable_pocket"] for r in rs]); e = np.array([r["entries"] for r in rs])
    print(f"{name:24s} n={len(rs):3d} | mapped entries: median {np.median(e):.0f} | usable unliganded entries at the POCKET residues: median {np.median(a):.1f}, "
          f">=2 {np.mean(a>=2)*100:.0f}%, >=3 {np.mean(a>=3)*100:.0f}%, >=6 {np.mean(a>=6)*100:.0f}%, zero {np.mean(a==0)*100:.0f}%")
s("all test chains", rows); s("strictly cryptic", [r for r in rows if r["cryptic"]]); s("not cryptic", [r for r in rows if not r["cryptic"]])
