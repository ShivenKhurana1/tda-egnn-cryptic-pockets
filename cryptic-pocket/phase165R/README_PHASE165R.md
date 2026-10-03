# Phase165R: H1 Persistence Feature Ablation

Targeted follow-up to Phase165. Phase165 completed 8 seeds for the full H1 persistence representation, so those runs are not repeated.

This notebook trains four compact H1 descriptor families with the exact Phase159 dataset and Phase163/164 architecture/training protocol:
- h1_total_max
- h1_mean_std
- h1_entropy
- h1_birth_death

8 seeds per arm = 32 new trainings. CPU mode is enabled in kernel metadata.

Outputs:
- phase165R_h1_feature_seed_results.csv
- phase165R_h1_feature_summary.csv
- phase165R_h1_feature_paired_statistics.csv
- phase165R_h1_feature_paired_statistics.json
- phase165R_h1_feature_results.json
- cache/*.pt
