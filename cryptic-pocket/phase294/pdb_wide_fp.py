"""Are the model's 'false positives' real binding sites that CryptoBench never labelled? For every CryptoBench test chain, each of the
8-seed ensemble's (Phase 291, official labels) top-20 residues is classed as: cryptic positive (official label) / known non-cryptic binder
(CryptoBench noncryptic-pockets.json) / contacts a ligand (non-additive HETATM within 4.5 A) in ANY OTHER deposited X-ray structure of the
same protein (Phase 290 cache: up to 30 entries per chain, CryptoBench's own apo/holo entries excluded) / no binding evidence.
Enrichment is against all non-positive residues of the same chains."""
import json, numpy as np, torch
from pathlib import Path
CP = Path(__file__).resolve().parents[1]
NC = json.load(open(CP / "phase294" / "noncryptic-pockets.json")); MT = json.load(open(CP / "phase292" / "motion_targets.json"))
P = torch.load("/tmp/p291_out/phase291_official_preds.pt", map_location="cpu", weights_only=False)
T = P["seeds"][0]["test"]
cnt = dict(top=0, cryptic=0, nc=0, pdb=0, none=0); base = dict(n=0, nc=0, pdb=0); per_chain = []; nchain = 0
for k in sorted(T):
    pid, c = k; f = CP / "ensemble" / "cache" / f"{pid}_{c}.npz"
    y = T[k]["y"].astype(bool)
    if not f.exists() or pid not in MT or len(MT[pid]["resnums"]) != len(y): continue
    z = np.load(f, allow_pickle=True)
    lig = z["lig"] if z["lig"].ndim == 2 else np.zeros((0, len(y)), bool)
    if lig.shape[1] != len(y): continue
    nchain += 1
    sel = {s.split("_", 1)[1] for e in NC.get(pid, []) if e["apo_chain"] == c for s in e["apo_pocket_selection"] if s.split("_", 1)[0] == c}
    nc = np.array([r in sel for r in MT[pid]["resnums"]]) & ~y
    pdb = lig.any(0) & ~y & ~nc
    p = np.mean([P["seeds"][s]["test"][k]["p"] for s in range(8)], 0); top = np.argsort(-p)[:20]
    cnt["top"] += 20; cnt["cryptic"] += int(y[top].sum()); cnt["nc"] += int(nc[top].sum()); cnt["pdb"] += int(pdb[top].sum())
    cnt["none"] += int((~y[top] & ~nc[top] & ~pdb[top]).sum())
    neg = ~y; base["n"] += int(neg.sum()); base["nc"] += int(nc.sum()); base["pdb"] += int(pdb.sum())
    per_chain.append(dict(key=f"{pid}_{c}", n_entries=int(lig.shape[0]), cryptic=int(y[top].sum()), nc=int(nc[top].sum()), pdb=int(pdb[top].sum())))
fp = cnt["top"] - cnt["cryptic"]
print(f"chains {nchain} | top-20 residues {cnt['top']}")
print(f"  cryptic positive (official label)        {cnt['cryptic']:5d}  {cnt['cryptic']/cnt['top']:.1%}")
print(f"  known non-cryptic binder (CryptoBench)   {cnt['nc']:5d}  {cnt['nc']/cnt['top']:.1%}   = {cnt['nc']/fp:.1%} of 'false positives' (base rate among negatives {base['nc']/base['n']:.1%}, x{cnt['nc']/fp/(base['nc']/base['n']):.1f})")
print(f"  ligand contact in another PDB structure  {cnt['pdb']:5d}  {cnt['pdb']/cnt['top']:.1%}   = {cnt['pdb']/fp:.1%} of 'false positives' (base rate {base['pdb']/base['n']:.1%}, x{cnt['pdb']/fp/(base['pdb']/base['n']):.1f})")
print(f"  no binding evidence anywhere             {cnt['none']:5d}  {cnt['none']/cnt['top']:.1%}   = {cnt['none']/fp:.1%} of 'false positives'")
print(f"top-20 precision: CryptoBench labels {cnt['cryptic']/cnt['top']:.1%} -> any binding evidence {(cnt['top']-cnt['none'])/cnt['top']:.1%}")
json.dump(dict(counts=cnt, base=base, per_chain=per_chain), open(CP / "phase294" / "pdb_wide_fp.json", "w"), indent=1)
