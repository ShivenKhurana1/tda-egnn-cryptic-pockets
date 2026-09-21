# Phase166 — full_model

Hard-generalization experiment for the cryptic-pocket EGNN.

## Design
- 942 CryptoBench graphs from Phase159.
- Deterministic label-free clustering of graph-level mean ESM embeddings using cosine distance 0.20.
- Entire clusters are assigned to train/validation/test, targeting 70/10/20% of graphs.
- No residue labels are used to construct the split.
- 8 seeds.
- Same Phase163/164/165R architecture and training settings.
- CPU-only Kaggle configuration.

This is an **embedding-cluster holdout**, not a claim of verified sequence-family membership.

## Feature configuration
`full_model`

## Outputs
- phase166_full_model_seed_results.csv
- phase166_full_model_summary.csv
- phase166_full_model_results.json
- cache/*.pt


FIXED VERSION: defines DEVICE explicitly and uses a fixed 30-cluster embedding holdout so a single giant cluster cannot consume most of the dataset. The exact split is saved as phase166_hard_split_manifest.json. This remains a label-free embedding-cluster holdout, not a verified sequence-family split.
