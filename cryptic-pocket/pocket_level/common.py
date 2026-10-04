"""Shared loaders for pocket-level scoring: per-residue 8-seed ensemble probabilities, CA coordinates (model frame),
and per-site truth rebuilt from CryptoBench's dataset JSON (apo_pocket_selection), checked against the stored labels."""
import json, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
CP = HERE.parent
P = torch.load("/tmp/p266c_fresh/phase266_full_preds.pt", map_location="cpu", weights_only=False)
COORDS = json.load(open(CP / "site/coords.json"))
DS = json.load(open(CP / "cryptobench_dataset.json"))

def chains(split):
    out = {}
    for k in P["seeds"][0][split]:
        key = f"{k[0]}_{k[1]}"
        if key not in COORDS: continue
        c = COORDS[key]; xyz = np.asarray(c["xyz"]).reshape(-1, 3)
        p = np.mean([P["seeds"][s][split][k]["p"] for s in range(8)], 0)
        y = P["seeds"][0][split][k]["y"].astype(int)
        if len(p) != len(xyz): continue
        cc = np.mean([P["seeds"][s][split][k]["c"] for s in range(8)], 0)
        out[key] = dict(pdb=k[0], chain=k[1], xyz=xyz, resi=c["resi"], p=p, y=y, c=np.asarray(cc), tc=np.asarray(c["tc"]))
    return out

def sites(ch):
    """True pockets: one residue-index set per CryptoBench ligand site on this apo chain, merged when they share
    >=50% of the smaller set (same physical pocket, different ligands)."""
    idx = {r: i for i, r in enumerate(ch["resi"])}
    raw = []
    for e in DS.get(ch["pdb"], []):
        if e.get("apo_chain") != ch["chain"]: continue
        s = set()
        for t in e["apo_pocket_selection"]:
            c, n = t.split("_", 1)
            try: n = int(n)
            except ValueError: continue
            if c == ch["chain"] and n in idx: s.add(idx[n])
        if s: raw.append(s)
    merged = []
    for s in raw:
        for m in merged:
            if len(s & m) >= 0.5 * min(len(s), len(m)): m |= s; break
        else: merged.append(set(s))
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if len(merged[i] & merged[j]) >= 0.5 * min(len(merged[i]), len(merged[j])):
                    merged[i] |= merged.pop(j); changed = True; break
            if changed: break
    return merged
