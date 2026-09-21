# Phase 163 — CryptoBench Feature-Group Ablation

Uses the recovered Phase159 ANM+TDA CryptoBench graph dataset. Six feature-group arms are trained with identical architecture, split, optimizer, schedule, localization loss, and eight paired seeds.

Feature layout: structure 0:27; ESM2-35M 27:507; ANM-MSF 507:508; TDA 508:542.

Arms: `esm_only`, `esm_structure`, `esm_structure_tda`, `esm_structure_anm`, `full`, `tda_anm`.

Primary metric: validation PR-AUC. Secondary: ROC-AUC, MCC, F1, precision, recall, DCC@4A, mean center error. Test results are recorded after validation-based checkpoint selection.

Expected outputs: `phase163_ablation_results.json`, `phase163_ablation_seed_results.csv`, `phase163_ablation_summary.csv`, `phase163_paired_significance.json`, and 48 checkpoints.
