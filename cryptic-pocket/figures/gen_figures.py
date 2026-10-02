"""Generate the paper's figures from already-cached results. No new training or Kaggle compute --
purely re-plotting data this project already produced (phase240/241 geometry control, phase266c headline
predictions, phase276 crypticity labels, phase282 calibration, phase285 decomposition)."""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 150, "font.family": "DejaVu Sans",
})
C_TOPO = "#4C72B0"; C_GEOM = "#DD8452"; C_BOTH = "#55A868"; C_NONE = "#8C8C8C"
C_RAW = "#C44E52"; C_CAL = "#4C72B0"; C_CRYPTIC = "#C44E52"; C_PREFORMED = "#4C72B0"

ROOT = "/Users/shivenkhurana/bioproj/cryptic-pocket"
OUT = "/Users/shivenkhurana/bioproj/cryptic-pocket/figures"

# ---------------------------------------------------------------------------
# Figure 1: Geometry vs topology, main-pipeline classification, across scales
# ---------------------------------------------------------------------------
def fig_geometry_control():
    rows = []
    # 35M scale numbers are reported directly in paper.md's table; hardcode (already-published values)
    rows.append(dict(scale="35M", arm="No feature", pr=0.4404, roc=0.8478))
    rows.append(dict(scale="35M", arm="Topology", pr=0.4947, roc=0.8733))
    rows.append(dict(scale="35M", arm="Geometry", pr=0.5076, roc=0.8797))
    rows.append(dict(scale="35M", arm="Geometry+Topology", pr=0.5173, roc=0.8812))
    for scale, prefix, path in [("650M", "phase240", f"{ROOT}/phase240a/results/phase240_merged_fold_means.csv"),
                                 ("3B", "phase241", f"{ROOT}/phase241a/results/phase241_merged_fold_means.csv")]:
        df = pd.read_csv(path)
        name_map = {"no-topo": "No feature", "full": "Topology", "geom": "Geometry", "geom+topo": "Geometry+Topology"}
        for arm, nm in name_map.items():
            sub = df[df.arm == arm]
            rows.append(dict(scale=scale, arm=nm, pr=sub.test_pr.mean(), roc=sub.test_roc.mean()))
    df = pd.DataFrame(rows)
    scales = ["35M", "650M", "3B"]
    arms = ["No feature", "Topology", "Geometry", "Geometry+Topology"]
    colors = {"No feature": C_NONE, "Topology": C_TOPO, "Geometry": C_GEOM, "Geometry+Topology": C_BOTH}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, title in zip(axes, ["pr", "roc"], ["Test PR-AUC", "Test ROC-AUC"]):
        x = np.arange(len(scales)); width = 0.2
        for i, arm in enumerate(arms):
            vals = [df[(df.scale == s) & (df.arm == arm)][metric].values[0] for s in scales]
            ax.bar(x + (i - 1.5) * width, vals, width, label=arm, color=colors[arm])
        ax.set_xticks(x); ax.set_xticklabels(scales)
        ax.set_xlabel("ESM2 embedding scale"); ax.set_title(title)
        ax.set_ylim(min(df[metric].min() - 0.05, 0.4), df[metric].max() + 0.03)
    axes[0].set_ylabel("Score")
    axes[1].legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.suptitle("Geometry alone matches or beats topology alone; a small confirmed\nresidual from topology-beyond-geometry grows with embedding scale",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_geometry_control.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_geometry_control.png")


# ---------------------------------------------------------------------------
# Figure 2: Calibration reliability diagram (raw vs isotonic), headline model
# ---------------------------------------------------------------------------
def fig_calibration():
    P = torch.load("/tmp/p266c_fresh/phase266_full_preds.pt", map_location="cpu", weights_only=False)
    seeds = sorted(P["seeds"])

    def ensemble(split):
        S0 = P["seeds"][seeds[0]][split]
        return {k: dict(p=np.mean([P["seeds"][s][split][k]["p"] for s in seeds], axis=0), y=S0[k]["y"]) for k in S0}

    val, test = ensemble("val"), ensemble("test")
    pv = np.concatenate([val[k]["p"] for k in val]); yv = np.concatenate([val[k]["y"] for k in val])
    pt = np.concatenate([test[k]["p"] for k in test]); yt = np.concatenate([test[k]["y"] for k in test])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(pv, yv)
    pt_cal = iso.predict(pt)

    def reliability(p, y, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.digitize(p, bins) - 1; idx = np.clip(idx, 0, n_bins - 1)
        mean_p, frac_pos, counts = [], [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() == 0: continue
            mean_p.append(p[m].mean()); frac_pos.append(y[m].mean()); counts.append(m.sum())
        return np.array(mean_p), np.array(frac_pos), np.array(counts)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (p_, label, color) in zip(axes, [(pt, "Raw probabilities", C_RAW), (pt_cal, "Isotonic-calibrated", C_CAL)]):
        mp, fp, counts = reliability(p_, yt)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect calibration")
        ax.scatter(mp, fp, s=np.sqrt(counts) * 3, color=color, zorder=3, edgecolor="white", linewidth=0.5)
        ax.plot(mp, fp, color=color, lw=1.5, alpha=0.7)
        ax.set_xlabel("Mean predicted probability"); ax.set_title(label)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Observed fraction positive")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Raw probabilities are badly overconfident; isotonic regression\nfit on validation fixes it almost completely (test set shown)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_calibration.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_calibration.png")
    return dict(val=val, test=test, iso=iso)


# ---------------------------------------------------------------------------
# Figure 3: ROC and PR curves, pooled test set, with cryptic-subset overlay
# ---------------------------------------------------------------------------
def fig_roc_pr(ens):
    test = ens["test"]
    labels = json.load(open(f"{ROOT}/phase276/phase276a_crypticity_labels.json"))
    label_of = {tuple(k.rsplit("_", 1)): v["label"] for k, v in labels.items()}
    cryptic_keys = [k for k in test if label_of.get(k) == "cryptic"]
    print(f"{len(cryptic_keys)} cryptic-labelled test chains found among {len(test)}")

    pt = np.concatenate([test[k]["p"] for k in test]); yt = np.concatenate([test[k]["y"] for k in test])
    pc = np.concatenate([test[k]["p"] for k in cryptic_keys]); yc = np.concatenate([test[k]["y"] for k in cryptic_keys])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    fpr, tpr, _ = roc_curve(yt, pt); fpr_c, tpr_c, _ = roc_curve(yc, pc)
    axes[0].plot(fpr, tpr, color=C_PREFORMED, lw=2, label=f"All 182 test chains (AUC {roc_auc_score(yt,pt):.3f})")
    axes[0].plot(fpr_c, tpr_c, color=C_CRYPTIC, lw=2, label=f"16 truly-cryptic chains (AUC {roc_auc_score(yc,pc):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate"); axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right", fontsize=8.5, frameon=False)

    prec, rec, _ = precision_recall_curve(yt, pt); prec_c, rec_c, _ = precision_recall_curve(yc, pc)
    axes[1].plot(rec, prec, color=C_PREFORMED, lw=2, label=f"All (AP {average_precision_score(yt,pt):.3f})")
    axes[1].plot(rec_c, prec_c, color=C_CRYPTIC, lw=2, label=f"Cryptic-only (AP {average_precision_score(yc,pc):.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-recall curve")
    axes[1].legend(loc="upper right", fontsize=8.5, frameon=False)
    fig.suptitle("Pooled headline performance is much better on the full test set\nthan on the strictly cryptic subset", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_roc_pr.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_roc_pr.png")


# ---------------------------------------------------------------------------
# Figure 4: Cryptic vs pre-formed stratified bar chart (multiple metrics)
# ---------------------------------------------------------------------------
def fig_cryptic_stratified(ens):
    test = ens["test"]
    labels = json.load(open(f"{ROOT}/phase276/phase276a_crypticity_labels.json"))
    label_of = {tuple(k.rsplit("_", 1)): v["label"] for k, v in labels.items()}
    groups = {"All 182 chains": list(test.keys()),
              "Pre-formed (111)": [k for k in test if label_of.get(k) == "preformed"],
              "Cryptic (16)": [k for k in test if label_of.get(k) == "cryptic"]}
    rows = []
    for name, keys in groups.items():
        p = np.concatenate([test[k]["p"] for k in keys]); y = np.concatenate([test[k]["y"] for k in keys])
        rows.append(dict(group=name, roc=roc_auc_score(y, p), pr=average_precision_score(y, p)))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    x = np.arange(len(df)); width = 0.35
    colors = [C_NONE, C_PREFORMED, C_CRYPTIC]
    b1 = ax.bar(x - width / 2, df.roc, width, label="ROC-AUC", color=colors, alpha=0.95)
    b2 = ax.bar(x + width / 2, df.pr, width, label="PR-AUC", color=colors, alpha=0.55, hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(df.group, fontsize=9.5)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}", ha="center", fontsize=8.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="gray", label="ROC-AUC"), Patch(facecolor="gray", alpha=0.55, hatch="//", label="PR-AUC")],
              loc="lower left", fontsize=9, frameon=False)
    ax.set_title("The pooled headline numbers describe pre-formed pockets\nfar better than genuinely cryptic ones")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_cryptic_stratified.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_cryptic_stratified.png")


# ---------------------------------------------------------------------------
# Figure 5: H0/H1/H2/burial decomposition (Phase 285)
# ---------------------------------------------------------------------------
def fig_decomposition():
    df = pd.read_csv(f"{ROOT}/phase285/results/phase285_fold_means.csv")
    arms = ["geom", "geom+H0", "geom+H1", "geom+H2", "geom+burial", "geom+topo"]
    labels = ["Geometry\nalone", "+H0", "+H1", "+H2", "+burial", "+all four\n(topo)"]
    colors = [C_NONE, "#937860", "#DA8BC3", "#8172B2", "#CCB974", C_BOTH]
    means = [df[df.arm == a].test_pr.mean() for a in arms]
    sems = [df[df.arm == a].test_pr.std(ddof=1) / np.sqrt(10) for a in arms]

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    bars = ax.bar(labels, means, yerr=sems, capsize=4, color=colors)
    ax.set_ylabel("Test PR-AUC"); ax.set_ylim(min(means) - 0.02, max(means) + 0.015)
    ax.set_title("H1 (loops/tunnels) alone tracks the full topology block;\nH0 and H2 alone each fall short (650M, 10-fold CV)")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001, f"{m:.3f}", ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_decomposition.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_decomposition.png")


# ---------------------------------------------------------------------------
# Figure 6: Per-protein accuracy distribution (headline model, test set)
# ---------------------------------------------------------------------------
def fig_per_protein(ens):
    test = ens["test"]
    rows = []
    for k, v in test.items():
        if 0 < v["y"].sum() < len(v["y"]):
            rows.append(dict(key=f"{k[0]}_{k[1]}", roc=roc_auc_score(v["y"], v["p"]), n=len(v["y"]), pos=int(v["y"].sum())))
    df = pd.DataFrame(rows).sort_values("roc")
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.hist(df.roc, bins=25, color=C_PREFORMED, edgecolor="white")
    ax.axvline(df.roc.median(), color=C_RAW, ls="--", lw=1.5, label=f"Median {df.roc.median():.3f}")
    ax.set_xlabel("Per-protein ROC-AUC"); ax.set_ylabel("Number of test proteins")
    ax.set_title(f"Per-protein discrimination across all {len(df)} scorable test proteins")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_per_protein_dist.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_per_protein_dist.png,", len(df), "proteins")
    df.to_csv(f"{OUT}/per_protein_metrics.csv", index=False)


if __name__ == "__main__":
    fig_geometry_control()
    ens = fig_calibration()
    fig_roc_pr(ens)
    fig_cryptic_stratified(ens)
    fig_decomposition()
    fig_per_protein(ens)
    print("done")
