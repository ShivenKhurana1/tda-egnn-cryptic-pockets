# Phase165 Persistence Feature Ablation

72 total trainings: 9 arms × 8 seeds.

This notebook preserves the Phase164 dataset, split, EGNN architecture, optimizer, losses, localization objective, and seed protocol. It changes only the selected TDA persistence features.

The persistence representation is 23 features:
- H0: 7
- H1: 8
- H2: 8

Arms:
h0_persistence, h1_persistence, h2_persistence, total_max, mean_std,
entropy, birth_death, counts, all_persistence.

Kaggle metadata is included. The dataset source is the same Phase159 ingestion used by Phase164.
