"""Figure: what the model's top-20 'false positives' really are (Phase 294 binding-evidence check)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
H = Path(__file__).resolve().parent; P = H.parent / "phase294"
a = json.load(open(P / "pdb_wide_fp.json"))["counts"]; d = json.load(open(P / "fp_distance.json"))
INK, INK2, GRID, SURF, GRAY = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#b4b2aa"
C1, C2, C3, C7 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 5.4), gridspec_kw={"height_ratios": [0.8, 1.6], "hspace": 0.55})
segs = [("Cryptic pocket\n(CryptoBench label)", a["cryptic"], C1), ("Known non-cryptic\nbinding site", a["nc"], C2),
        ("Ligand bound in another\nPDB structure", a["pdb"], C3), ("No binding\nevidence", a["none"], GRAY)]
left = 0
for name, n, col in segs:
    w = 100 * n / a["top"]
    ax1.barh(0, w - 0.4, left=left + 0.2, color=col, height=0.55)
    ax1.text(left + w / 2, 0, f"{w:.0f}%", ha="center", va="center", color="white" if col != GRAY else INK, fontsize=11, fontweight="bold")
    left += w
ax1.set_xlim(0, 100); ax1.set_ylim(-0.5, 0.45); ax1.axis("off")
ax1.legend(handles=[Patch(color=c, label=n.replace("\n", " ")) for n, _, c in segs], ncol=2, frameon=False, fontsize=8.5,
           loc="upper left", bbox_to_anchor=(0, -0.02), labelcolor=INK2, handlelength=1.2, columnspacing=1.5)
ax1.set_title(f"The model's top 20 residues per protein ({a['top']:,} residues, 180 test proteins)", loc="left", color=INK, fontsize=11)
bands = ["rim <8 A", "near 8-12 A", "distant >12 A"]; labels = ["< 8 Å from the\nlabelled pocket", "8–12 Å", "> 12 Å\n(a different site)"]
ev = [100 * d["fp"][b][0] / d["fp"][b][1] for b in bands]; ch = [100 * d["allneg"][b][0] / d["allneg"][b][1] for b in bands]
x = range(3); w = 0.36
b1 = ax2.bar([i - w / 2 - 0.01 for i in x], ev, w, color=C7, label="Model's false positives")
b2 = ax2.bar([i + w / 2 + 0.01 for i in x], ch, w, color=GRAY, label="All residues in that band (chance)")
for r, v in zip(b1, ev): ax2.text(r.get_x() + r.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", color=INK, fontsize=9)
for r, v in zip(b2, ch): ax2.text(r.get_x() + r.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", color=INK2, fontsize=9)
ax2.set_xticks(list(x)); ax2.set_xticklabels(labels); ax2.set_ylim(0, 70); ax2.set_ylabel("share with binding evidence")
ax2.yaxis.grid(True, color=GRID, lw=0.8); ax2.set_axisbelow(True)
for s in ("top", "right", "left"): ax2.spines[s].set_visible(False)
ax2.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK2)
ax2.set_title("False positives carry binding evidence far above chance, most of all away from the labelled pocket", loc="left", color=INK, fontsize=11)
fig.savefig(H / "fig_fp_binding_evidence.png", dpi=200, bbox_inches="tight"); print("saved")
