# Phase 276: is this project's result actually about cryptic pockets?

Source: "A benchmark selected for crypticity" (pmfran4252-afk/cryptic-site-benchmark, preprint, CC BY 4.0),
https://github.com/pmfran4252-afk/cryptic-site-benchmark — an independent, adversarially-designed relabeling
of CryptoBench's own test split (strict apo/holo cavity-voxel rule) plus a separate 2,401-pair screened benchmark.
CryptoBench data/splits from https://osf.io/pz4a9/ (Skrhak et al. 2025, Bioinformatics, btae745).

No new training. Re-scores prediction files already produced by this project (Phase 262/264/266c), stratified by
their crypticity relabeling, using their exact top-20 metric so numbers are comparable to their published table.

## Files
- `build_crypticity_labels.py` / `phase276a_crypticity_labels.json`: maps CryptoBench's own (apo_pdb, apo_chain, is_main_holo_structure)
  entries + folds.json test-fold membership to a uniprot_id, joined against their crypticity relabeling. Reproduces their
  claimed "180 labelled proteins" count exactly (111 pre-formed, 52 unlabelled, 17 cryptic) before intersecting with our 182.
- `analyze276.py` / `phase276_per_seed.csv`: every arm's pooled/protein-level PR-AUC, ROC-AUC and top-20 metrics, stratified
  by cryptic / preformed / neither, for 8 seeds each.
- head-to-head script output `phase276_p2rank_cryptic_headtohead.csv`: our headline model vs our own Phase-267 P2Rank rerun,
  on the 16 (of their 17) cryptic-labelled test proteins we could match.

## Findings
1. Confirms the literature: performance on the 16-17 genuinely cryptic test proteins is much weaker than the pooled number
   the paper otherwise reports (headline model: pooled ROC-AUC 0.896 -> 0.721, protein PR-AUC 0.459 -> 0.262, precision@20
   0.383 -> 0.200 on the cryptic-only subset). Most of this project's reported numbers are, numerically, dominated by the
   61% of test proteins (111/182) whose "cryptic" pocket was already visible in the unbound structure.
2. Does NOT replicate their central negative claim ("cavity/geometry features are actively harmful for cryptic sites").
   On our architecture, geometry vs topology-only within the cryptic stratum trends positive (+0.018 pooled PR, +0.015
   pooled ROC, 5-7/8 seeds) though not confirmed by this project's two-test rule (p=0.08-0.11); it does not go the
   negative direction they report.
3. Confirmed, real result: on the 16 cryptic proteins where both P2Rank (Phase 267 rerun) and our model have residue
   scores, our homology-clean 3B-transfer headline model beats P2Rank by +0.115 precision@20 (paired over proteins,
   t p=0.022, Wilcoxon p=0.022, bootstrap 95% CI [+0.035,+0.203]) and reaches precision@20 0.209 / recall@20 0.314
   (8-seed ensemble) -- above every baseline in their preprint's table, including their own new method (0.174/0.278).

## Caveats
- n=16-17 proteins; every number above should be read with that in mind.
- This is a stricter RELABELING of CryptoBench's existing test split, not new structures -- it is an honest audit of
  what our existing numbers mean, not yet an independent-structure external validation (that is Phase 277+, using
  PocketMiner's 39-protein set and the 99 truly-cryptic pairs from this same repository's larger `cryptic_benchmark.json`).
- Their own preprint is not yet peer-reviewed.
