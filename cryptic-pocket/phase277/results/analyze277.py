"""Phase 277: score the headline model (Phase 266c's exact recipe, verified node-format match: 2638 = 2587 + 34 + 17 on both sides) on two
external test sets never seen in any form -- PocketMiner's 35-protein held-out set (Meller et al. 2023, Nat Commun; MD-simulation-derived
labels) and 99 genuinely-cryptic entries from an independent, adversarially-designed benchmark (pmfran4252-afk/cryptic-site-benchmark).
Stratifies by Phase 277c's homology check (29/35 PocketMiner, 82/98 cryptic-site-benchmark share a >=30%-identity homolog with our training
pool or CryptoBench train split); results are reported for the full set AND the homology-clean subset.

usage: python3 analyze277.py <phase277d_full_preds.pt> <phase277c_homology.json> <out_prefix>

IMPORTANT design note, discovered from the data itself (not assumed): PocketMiner's label scheme leaves ZERO of its 34 usable proteins with
BOTH classes present among evaluated (non-masked) residues -- 24 proteins are "positive-only" annotated (their known cryptic-pocket residues
are labelled 1, everything else is masked as unconfirmed) and 10 are pure "negative example" proteins (fully labelled 0). This matches their
own evaluation script (src/test_performance_on_xtal_residues.py), which pools every evaluated residue from every protein into one array
before computing AUC/PR-AUC -- a per-protein score is undefined for this dataset. So PocketMiner is scored POOLED ONLY, exactly as they do.
cryptic-site-benchmark's site/non-site labelling is a standard single-chain label (both classes present per protein), so it is scored both
ways: pooled AND per-protein.

Pre-specified primary metric: pooled ROC-AUC and PR-AUC on each set's homology-CLEAN subset (full-set numbers are secondary/context, since a
training-pool homolog is a real advantage, not a fair test of generalization). Per-seed (8 seeds) mean and an 8-seed ensemble are both given."""
import json, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score

d277d, hom_path, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
P = torch.load(d277d, map_location="cpu", weights_only=False)
hom = json.load(open(hom_path))
seeds = sorted(P["seeds"])

pm_key = lambda k: k[0] + k[1]
csb_key = lambda k: f"{k[0]}_{k[1]}"
pm_clean = lambda k: not hom["pm_leak"].get(k, True)
csb_clean = lambda k: not hom["csb_leak"].get(k, True)


def pooled_by_seed(name, key_fn, clean_fn, use_mask):
    rows = []
    for s in seeds:
        ext = P["seeds"][s][name]
        for subset, keep in [("all", lambda k: True), ("clean", lambda k: clean_fn(key_fn(k)))]:
            ks = [k for k in ext if keep(k)]
            m = [ext[k].get("eval_mask", np.ones_like(ext[k]["y"], dtype=bool)) if use_mask else np.ones_like(ext[k]["y"], dtype=bool) for k in ks]
            y = np.concatenate([ext[k]["y"][mm] for k, mm in zip(ks, m)])
            p = np.concatenate([ext[k]["p"][mm] for k, mm in zip(ks, m)])
            rows.append(dict(seed=s, subset=subset, n_proteins=len(ks), n_res=len(y), n_pos=int(y.sum()),
                              pr=average_precision_score(y, p), roc=roc_auc_score(y, p)))
    return pd.DataFrame(rows)


def protein_level_by_seed(name, key_fn, clean_fn):
    rows = []
    for s in seeds:
        ext = P["seeds"][s][name]
        for k, rec in ext.items():
            y, p = rec["y"], rec["p"]
            if y.sum() == 0 or y.sum() == len(y): continue
            rows.append(dict(seed=s, key=key_fn(k), clean=clean_fn(key_fn(k)), pr=average_precision_score(y, p), roc=roc_auc_score(y, p)))
    return pd.DataFrame(rows)


print("########## PocketMiner (34 usable of 35; MD-simulation-derived labels; POOLED evaluation only -- see docstring) ##########")
pm_pooled = pooled_by_seed("pocketminer", pm_key, pm_clean, True)
pm_pooled.to_csv(OUT + "_pocketminer_pooled_per_seed.csv", index=False)
for subset in ("all", "clean"):
    s = pm_pooled[pm_pooled.subset == subset]
    print(f"  [{subset:4s}] {int(s.n_proteins.iloc[0])} proteins, {int(s.n_res.iloc[0])} evaluated residues, {int(s.n_pos.iloc[0])} positive | "
          f"per-seed mean over 8 seeds: pooled PR-AUC {s.pr.mean():.4f} +- {s.pr.std(ddof=0):.4f}  ROC-AUC {s.roc.mean():.4f} +- {s.roc.std(ddof=0):.4f}")

print("\n########## cryptic-site-benchmark cryptic subset (99 chains; standard per-chain site/non-site labels) ##########")
csb_pooled = pooled_by_seed("csb_cryptic", csb_key, csb_clean, False)
csb_pooled.to_csv(OUT + "_csb_pooled_per_seed.csv", index=False)
csb_protein = protein_level_by_seed("csb_cryptic", csb_key, csb_clean)
csb_protein.to_csv(OUT + "_csb_protein_per_seed.csv", index=False)
for subset in ("all", "clean"):
    sp = pm_pooled  # placeholder to keep structure; overwritten below
    s = csb_pooled[csb_pooled.subset == subset]
    pr_prot = csb_protein[csb_protein.clean if subset == "clean" else csb_protein.key.notna()]
    prot_by_seed = pr_prot.groupby("seed").agg(pr=("pr", "mean"), roc=("roc", "mean")) if len(pr_prot) else None
    print(f"  [{subset:4s}] {int(s.n_proteins.iloc[0])} proteins, {int(s.n_res.iloc[0])} residues, {int(s.n_pos.iloc[0])} positive | "
          f"pooled PR-AUC {s.pr.mean():.4f} +- {s.pr.std(ddof=0):.4f}  ROC-AUC {s.roc.mean():.4f} +- {s.roc.std(ddof=0):.4f}"
          + (f"  |  protein-level PR-AUC {prot_by_seed.pr.mean():.4f}  ROC-AUC {prot_by_seed.roc.mean():.4f}" if prot_by_seed is not None else ""))

print("\n8-seed ENSEMBLE (mean probability across seeds)")
for name, key_fn, clean_fn, use_mask in [("pocketminer", pm_key, pm_clean, True), ("csb_cryptic", csb_key, csb_clean, False)]:
    ext0 = P["seeds"][seeds[0]][name]
    for subset, keep in [("all", lambda k: True), ("clean", lambda k: clean_fn(key_fn(k)))]:
        ks = [k for k in ext0 if keep(k)]
        m = [ext0[k].get("eval_mask", np.ones_like(ext0[k]["y"], dtype=bool)) if use_mask else np.ones_like(ext0[k]["y"], dtype=bool) for k in ks]
        y = np.concatenate([ext0[k]["y"][mm] for k, mm in zip(ks, m)])
        p = np.concatenate([np.mean([P["seeds"][s][name][k]["p"][mm] for s in seeds], axis=0) for k, mm in zip(ks, m)])
        print(f"  {name:12s} [{subset:4s}] n={len(ks):3d} proteins  pooled PR {average_precision_score(y,p):.4f}  ROC {roc_auc_score(y,p):.4f}")
