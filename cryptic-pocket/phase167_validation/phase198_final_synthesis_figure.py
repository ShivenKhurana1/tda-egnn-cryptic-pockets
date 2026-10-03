import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "phase167_validation"

OUT_DIR = os.path.join(
    BASE,
    "phase198_final_figure"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT = os.path.join(
    OUT_DIR,
    "phase198_final_synthesis.png"
)

# ----------------------------------------------------------------------
# Load existing results
# ----------------------------------------------------------------------

figdata = pd.read_csv(
    os.path.join(
        BASE,
        "phase190_figures",
        "phase190_figure_data.csv"
    )
)

cases = pd.read_csv(
    os.path.join(
        BASE,
        "phase191_case_studies.csv"
    )
)

case_centers = pd.read_csv(
    os.path.join(
        BASE,
        "phase194_figures",
        "phase194_case_study_centers.csv"
    )
)

# Normalize names where necessary
figdata["arm"] = figdata["arm"].astype(str)

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------

fig = plt.figure(
    figsize=(16, 12)
)

# ======================================================================
# PANEL A - Experimental residue detection
# ======================================================================

ax1 = fig.add_subplot(2, 2, 1)

arms = [
    "esm_structure",
    "full_tda",
    "full_model"
]

labels = [
    "ESM + structure",
    "Full TDA",
    "Full model"
]

values = []

for arm in arms:

    x = figdata[
        figdata["arm"] == arm
    ]["PR_AUC"]

    values.append(
        x.mean()
    )

bars = ax1.bar(
    labels,
    values
)

ax1.set_ylabel(
    "Protein-level mean PR-AUC"
)

ax1.set_title(
    "A. Experimental ligand-contact residue detection"
)

ax1.set_ylim(
    0,
    max(values) * 1.25
)

for bar, value in zip(bars, values):

    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

ax1.tick_params(
    axis="x",
    rotation=15
)

# ======================================================================
# PANEL B - Localization
# ======================================================================

ax2 = fig.add_subplot(2, 2, 2)

loc_labels = [
    "Probability\ncentroid",
    "γ=10\nsharpened"
]

# Values from Phase 185
prob_centroid = [
    11.768457,
    11.735689,
    11.671319
]

gamma10 = [
    10.158503,
    10.052842,
    10.066239
]

x = np.arange(3)
width = 0.34

ax2.bar(
    x - width / 2,
    prob_centroid,
    width,
    label="Probability centroid"
)

ax2.bar(
    x + width / 2,
    gamma10,
    width,
    label="γ=10 sharpening"
)

ax2.set_xticks(x)
ax2.set_xticklabels(
    [
        "ESM + structure",
        "Full model",
        "Full TDA"
    ],
    rotation=15
)

ax2.set_ylabel(
    "Mean center error (Å)"
)

ax2.set_title(
    "B. Probability sharpening improves localization"
)

ax2.legend(
    fontsize=9
)

# ======================================================================
# PANEL C - Pocket-size failure mode
# ======================================================================

ax3 = fig.add_subplot(2, 2, 3)

full = figdata[
    figdata["arm"] == "full_model"
].copy()

ax3.scatter(
    full["experimental_pocket_size"],
    full["PR_AUC"],
    s=28,
    alpha=0.65
)

# Linear trend
if len(full) > 1:

    z = np.polyfit(
        full["experimental_pocket_size"],
        full["PR_AUC"],
        1
    )

    xx = np.linspace(
        full["experimental_pocket_size"].min(),
        full["experimental_pocket_size"].max(),
        200
    )

    yy = np.polyval(
        z,
        xx
    )

    ax3.plot(
        xx,
        yy,
        linewidth=2
    )

# Highlight case-study proteins
case_full = cases[
    cases["apo_pdb_id"].isin(
        ["3q6u", "5ytb", "7yjc"]
    )
]

for _, r in case_full.iterrows():

    ax3.scatter(
        r["experimental_pocket_size"],
        r["PR_AUC"],
        s=90,
        edgecolors="black",
        linewidths=1.2
    )

    ax3.annotate(
        r["apo_pdb_id"].upper(),
        (
            r["experimental_pocket_size"],
            r["PR_AUC"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9
    )

ax3.set_xlabel(
    "Experimental pocket size (residues)"
)

ax3.set_ylabel(
    "Protein-level PR-AUC"
)

ax3.set_title(
    "C. Smaller experimental pockets are harder to detect"
)

ax3.text(
    0.03,
    0.95,
    "Partial Spearman ρ = 0.756\n"
    "controlling for protein size",
    transform=ax3.transAxes,
    va="top",
    fontsize=9
)

# ======================================================================
# PANEL D - Case-study performance
# ======================================================================

ax4 = fig.add_subplot(2, 2, 4)

order = [
    "strong_success",
    "intermediate",
    "difficult_failure",
    "small_pocket_failure"
]

display_names = [
    "3q6u A\nsuccess",
    "1xt3 B\nintermediate",
    "5ytb C\nfailure",
    "7yjc A\nsmall-pocket failure"
]

case_order = []

for name in order:

    row = cases[
        cases["case"] == name
    ]

    if len(row) > 0:
        case_order.append(
            row.iloc[0]
        )

pr_values = [
    r["PR_AUC"]
    for r in case_order
]

bars = ax4.bar(
    np.arange(len(pr_values)),
    pr_values
)

ax4.set_xticks(
    np.arange(len(pr_values))
)

ax4.set_xticklabels(
    display_names,
    fontsize=9
)

ax4.set_ylabel(
    "PR-AUC"
)

ax4.set_title(
    "D. Representative success-to-failure cases"
)

ax4.set_ylim(
    0,
    max(pr_values) * 1.25
)

for bar, value in zip(
    bars,
    pr_values
):

    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value:.3f}",
        ha="center",
        fontsize=9
    )

# ----------------------------------------------------------------------
# Overall formatting
# ----------------------------------------------------------------------

fig.suptitle(
    "Cryptic Pocket Prediction: Detection, Localization, and Failure Modes",
    fontsize=17,
    y=0.995
)

plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)

fig.savefig(
    OUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("=" * 100)
print("PHASE 198 - FINAL SYNTHESIS FIGURE")
print("=" * 100)

print(f"\nSaved:")
print(OUT)

print("\nPanel A values:")
for label, value in zip(labels, values):
    print(f"  {label}: {value:.4f}")

print("\nPanel B:")
for i, label in enumerate(
    ["ESM + structure", "Full model", "Full TDA"]
):
    print(
        f"  {label}: "
        f"{prob_centroid[i]:.3f} Å -> "
        f"{gamma10[i]:.3f} Å"
    )

print("\nPanel C:")
print(
    "  Partial Spearman rho = 0.756 "
    "(p = 1.7e-35), controlling for protein size"
)

print("\nPanel D:")
for r, label in zip(case_order, display_names):
    print(
        f"  {label.replace(chr(10), ' ')}: "
        f"PR-AUC={r['PR_AUC']:.4f}"
    )

print("\n" + "=" * 100)
print("PHASE 198 COMPLETE")
print("=" * 100)
