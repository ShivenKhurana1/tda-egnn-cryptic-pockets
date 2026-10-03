# Phase 164: CryptoBench TDA Component Ablation

Built directly from the Phase 163 experiment implementation. It preserves the Phase 159 dataset, official train/val/test split, EGNN, focal loss, localization loss, optimizer, 40-epoch maximum, patience 8, and eight-seed protocol.

## Arms
- h0
- h1
- h2
- h0_h1
- h1_h2
- h0_h2
- persistence
- betti
- burial
- all_tda

Every arm retains ESM + structure and masks all TDA features except the selected subset. ANM-MSF is excluded intentionally.

## Kaggle
1. Create a new Kaggle Notebook.
2. Add dataset `blackpearls1/phase159-cryptobench-anmmsf-graphs`.
3. Enable GPU, preferably Tesla T4.
4. Upload/open `phase164_cryptobench_tda_component_ablation.ipynb`, or copy its cells into Kaggle.
5. Run all cells.
6. Download the files named below from `/kaggle/working/`.

## Outputs
- `phase164_tda_component_results.json`
- `phase164_tda_component_seed_results.csv`
- `phase164_tda_component_summary.csv`
- `phase164_tda_component_paired_statistics.csv`
- `phase164_tda_component_paired_statistics.json`

## Important
Do not change the CryptoBench test split or use test performance for model selection. The primary selection metric remains validation PR-AUC.
