import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = json.load(open("benchmark_data.json"))["cryptic"]["models"]
col = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
names = ["TDA-EGNN (this work)", "Benchmark authors' ESM-free\nmodel (preprint)", "P2Rank 2.5", "ESM-based model", "Cavity-feature model"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
for i, (m, c) in enumerate(zip(D, col)):
    y = len(D) - 1 - i
    ax[0].plot(m["ci"], [y, y], color=c, lw=2); ax[0].scatter(m["auc"], y, s=55, color=c, zorder=3, edgecolor="white")
    ax[0].text(m["ci"][1] + 0.015, y, f"{m['auc']:.3f}", va="center", fontsize=9, fontweight="bold")
    ax[1].barh(y, m["p20"], color=c, height=0.55); ax[1].text(m["p20"] + 0.006, y, f"{m['p20']:.3f}", va="center", fontsize=9, fontweight="bold")
for a in ax: a.set_yticks(range(len(D))); a.set_yticklabels(names[::-1] if a is ax[0] else [], fontsize=9)
ax[0].axvline(0.5, ls="--", color="gray", lw=1); ax[0].text(0.5, len(D) - 0.35, "chance", ha="center", fontsize=8, color="gray")
ax[0].set_xlim(0.3, 0.88); ax[0].set_xlabel("Per-protein AUC (mean, 95% interval)"); ax[0].set_title("Discrimination: three-way tie at the top", fontsize=10)
ax[1].set_xlim(0, 0.27); ax[1].set_xlabel("Precision in the top 20 residues"); ax[1].set_title("Top-of-list content: clear lead over P2Rank", fontsize=10)
fig.tight_layout(); fig.savefig("../figures/fig_cryptic_benchmark.png", dpi=150, bbox_inches="tight"); print("ok")
