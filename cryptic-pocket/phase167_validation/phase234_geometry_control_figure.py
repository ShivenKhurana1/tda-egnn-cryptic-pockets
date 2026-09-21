"""Phase 234 figure: geometry control on the 185-protein experimental holdout (Phase 232/233 data)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("phase167_validation/phase233_protein_metrics.csv")
rng = np.random.default_rng(234)
B = 10000
idx = rng.integers(0, len(df), size=(B, len(df)))            # resample proteins

def ci(v):
    m = v.values[idx].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

INK, MUTED, ACC, GRID = "#1f2328", "#6b7280", "#2a6fbb", "#e5e7eb"
methods = [("P2Rank (pretrained)", "p2rank_ap", MUTED), ("ESM + structure", "esm_structure_ap", ACC),
           ("+ TDA", "tda_ap", ACC), ("+ geometry", "geom_ap", ACC),
           ("+ geometry + persistence", "geom_persistence_ap", ACC), ("+ geometry + TDA", "geom_tda_ap", ACC)]
diffs = [("TDA vs ESM + structure", df.tda_ap - df.esm_structure_ap),
         ("geometry vs ESM + structure", df.geom_ap - df.esm_structure_ap),
         ("geometry vs TDA", df.geom_ap - df.tda_ap),
         ("geometry + persistence vs geometry\n(pre-specified test of topology)", df.geom_persistence_ap - df.geom_ap),
         ("geometry + TDA vs geometry", df.geom_tda_ap - df.geom_ap),
         ("geometry vs P2Rank", df.geom_ap - df.p2rank_ap)]

fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
plt.rcParams.update({"font.size": 10})
a = ax[0]
for i, (lab, col, c) in enumerate(methods[::-1]):
    m, lo, hi = ci(df[col])
    a.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
    a.plot(m, i, "o", color=c, ms=8, mec="white", mew=1.5)
    a.text(hi + 0.004, i, f"{m:.3f}", va="center", color=INK, fontsize=9)
a.set_yticks(range(len(methods))); a.set_yticklabels([m[0] for m in methods[::-1]], color=INK)
a.set_xlabel("mean protein-level PR-AUC (95% bootstrap CI over 185 proteins)", color=MUTED)
a.set_title("A  Residue ranking on experimental ligand-contact labels", loc="left", color=INK, fontsize=11)
a.set_xlim(0.26, 0.40)

b = ax[1]
for i, (lab, v) in enumerate(diffs[::-1]):
    m, lo, hi = ci(v)
    prim = "pre-specified" in lab
    c = ACC if prim else MUTED
    b.plot([lo, hi], [i, i], color=c, lw=2.4 if prim else 2, solid_capstyle="round")
    b.plot(m, i, "o", color=c, ms=8, mec="white", mew=1.5)
    b.text(hi + 0.004, i, f"{m:+.3f}", va="center", color=INK, fontsize=9)
b.axvline(0, color=INK, lw=1)
b.set_yticks(range(len(diffs))); b.set_yticklabels([d[0] for d in diffs[::-1]], color=INK)
b.set_xlabel("paired difference in PR-AUC (95% bootstrap CI)", color=MUTED)
b.set_title("B  Paired differences", loc="left", color=INK, fontsize=11)
b.set_xlim(-0.03, 0.10)
for x in ax:
    x.grid(axis="x", color=GRID, lw=0.8); x.set_axisbelow(True)
    for s in ("top", "right", "left"): x.spines[s].set_visible(False)
    x.spines["bottom"].set_color(GRID); x.tick_params(length=0, colors=MUTED); x.tick_params(axis="y", colors=INK)
plt.tight_layout()
plt.savefig("phase234_figures/geometry_control.png", dpi=180, facecolor="white")
print("saved")
for lab, v in diffs: print(f"{lab.splitlines()[0]:40s} {ci(v)[0]:+.4f} [{ci(v)[1]:+.4f}, {ci(v)[2]:+.4f}]")
