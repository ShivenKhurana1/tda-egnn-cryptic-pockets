"""Confounder check for pdb_wide_fp.py: are binding-evidence 'false positives' just the rim of the labelled pocket? Split every top-20
false positive by CA distance to the nearest labelled (official) pocket residue: rim (<8 A), near (8-12 A), distant (>12 A, a different
site). Within each band, compare the share with binding evidence (known non-cryptic binder or ligand contact in another PDB structure)
to the share among ALL negative residues in that band (the chance rate for that band)."""
import json, numpy as np, torch
from pathlib import Path
from scipy.spatial import cKDTree
CP = Path(__file__).resolve().parents[1]
NC = json.load(open(CP / "phase294" / "noncryptic-pockets.json")); MT = json.load(open(CP / "phase292" / "motion_targets.json"))
CO = json.load(open(CP / "site" / "coords.json"))
P = torch.load("/tmp/p291_out/phase291_official_preds.pt", map_location="cpu", weights_only=False); T = P["seeds"][0]["test"]
bands = [("rim <8 A", 0, 8), ("near 8-12 A", 8, 12), ("distant >12 A", 12, 1e9)]
fp = {b[0]: [0, 0] for b in bands}; allneg = {b[0]: [0, 0] for b in bands}; distant_chains = set()
for k in sorted(T):
    pid, c = k; key = f"{pid}_{c}"; f = CP / "ensemble" / "cache" / f"{key}.npz"; y = T[k]["y"].astype(bool)
    if not f.exists() or pid not in MT or key not in CO or len(MT[pid]["resnums"]) != len(y): continue
    z = np.load(f, allow_pickle=True); lig = z["lig"] if z["lig"].ndim == 2 else np.zeros((0, len(y)), bool)
    xyz = np.array(CO[key]["xyz"]).reshape(-1, 3)
    if lig.shape[1] != len(y) or len(xyz) != len(y) or not y.any(): continue
    sel = {s.split("_", 1)[1] for e in NC.get(pid, []) if e["apo_chain"] == c for s in e["apo_pocket_selection"] if s.split("_", 1)[0] == c}
    ev = (np.array([r in sel for r in MT[pid]["resnums"]]) | lig.any(0)) & ~y
    dist = cKDTree(xyz[y]).query(xyz)[0]
    p = np.mean([P["seeds"][s]["test"][k]["p"] for s in range(8)], 0); top = np.zeros(len(y), bool); top[np.argsort(-p)[:20]] = True
    for name, lo, hi in bands:
        m = (dist >= lo) & (dist < hi) & ~y
        fp[name][0] += int((top & m & ev).sum()); fp[name][1] += int((top & m).sum())
        allneg[name][0] += int((m & ev).sum()); allneg[name][1] += int(m.sum())
        if name.startswith("distant") and (top & m & ev).any(): distant_chains.add(key)
print(f"{'band':15s} {'top-20 FPs':>10s} {'with evidence':>14s} {'chance rate':>12s} {'enrichment':>11s}")
for name, _, _ in bands:
    a, n = fp[name]; b, N = allneg[name]
    print(f"{name:15s} {n:10d} {a/max(n,1):14.1%} {b/max(N,1):12.1%} {a/max(n,1)/max(b/max(N,1),1e-9):10.1f}x")
print(f"chains where the model's top 20 hits a DIFFERENT (>12 A away) site with binding evidence: {len(distant_chains)}")
json.dump(dict(fp=fp, allneg=allneg, distant_chains=sorted(distant_chains)), open(CP / "phase294" / "fp_distance.json", "w"), indent=1)
