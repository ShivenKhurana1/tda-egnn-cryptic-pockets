# Phase 201: Scientific Results Freeze

## Dataset

- CryptoBench total: 942 chain graphs
- Experimental holdout: 185 proteins
- Experimental validation labels: experimentally defined ligand-contact residues
- Model arms:
  - ESM + structure
  - Full TDA
  - Full model = TDA + ANM-MSF

## Primary experimental residue validation

### Protein-level mean PR-AUC

| Model | Mean PR-AUC |
|---|---:|
| ESM + structure | 0.2952 |
| Full TDA | 0.3485 |
| Full model | 0.3476 |

### Protein-level median ROC-AUC

| Model | Median ROC-AUC |
|---|---:|
| ESM + structure | 0.8291 |
| Full TDA | 0.8595 |
| Full model | 0.8553 |

### Global residue-level PR-AUC

| Model | PR-AUC |
|---|---:|
| ESM + structure | 0.2581 |
| Full TDA | 0.3115 |
| Full model | 0.3076 |

### Statistical comparison

Full TDA vs ESM + structure:

- 17/17 evaluated metrics significant after BH correction
- 17/17 significant after Bonferroni correction
- paired sign-flip permutation test, 10,000 permutations
- permutation p < 1e-4 for all metrics

Full model vs ESM + structure:

- 17/17 evaluated metrics significant after BH correction
- 17/17 significant after Bonferroni correction
- paired sign-flip permutation test, 10,000 permutations
- permutation p < 1e-4 for all metrics

Full model vs Full TDA:

- PR-AUC difference: -0.000954
- PR-AUC permutation p = 0.539646
- ROC-AUC difference: -0.002665
- ROC-AUC permutation p = 0.022998
- ROC-AUC difference did not survive Bonferroni correction
- enrichment/precision/recall differences were not significant

Primary interpretation:

TDA provides the principal measurable improvement over ESM + structure. ANM-MSF does not provide a meaningful additional improvement over TDA.

## Localization

### Mean center error

| Model | Probability centroid | Gamma-10 |
|---|---:|---:|
| ESM + structure | 11.768 Å | 10.159 Å |
| Full TDA | 11.671 Å | 10.066 Å |
| Full model | 11.736 Å | 10.053 Å |

Gamma-10:

w_i = p_i^10 / sum_j p_j^10

Gamma-10 is a post-processing method and does not retrain the model.

### Gamma-10 protein-level median center error

| Model | Median center error |
|---|---:|
| ESM + structure | 8.2167 Å |
| Full TDA | 7.5736 Å |
| Full model | 7.5035 Å |

### Gamma-10 DCC10

| Model | DCC10 |
|---|---:|
| ESM + structure | 0.6311 |
| Full TDA | 0.6358 |
| Full model | 0.6358 |

Primary interpretation:

Probability sharpening substantially improves localization over unsharpened probability centroiding. Gamma-10 performs comparably to top-10% residue centroiding.

Do NOT claim that gamma-10 is categorically superior to top-10%.

## Experimental-positive centroid

Oracle/reference only:

- Mean center error: approximately 1.43 Å
- This uses experimentally defined positive residues
- It is NOT a deployable prediction method
- Do NOT use this value as model localization performance

## Spatial enrichment

### Full model

Top 1% predicted residues:

- approximately 6.4-fold enrichment within 4 Å
- approximately 10.7-fold enrichment within 6 Å

### Full TDA

Top 1% predicted residues:

- approximately 8.1-fold enrichment within 4 Å
- approximately 10.5-fold enrichment within 6 Å

Primary interpretation:

High-probability residues are spatially enriched around experimentally defined pocket centers.

## Pocket-size analysis

### Raw Full-model correlations

Experimental pocket size vs:

- PR-AUC: rho = +0.7629
- center error: rho = -0.5842
- DCC8: rho = +0.5796
- DCC10: rho = +0.5541

### Protein-size-adjusted Full-model partial correlations

Experimental pocket size vs:

- PR-AUC: partial rho = +0.7558, p = 1.7078e-35
- ROC-AUC: partial rho = +0.4560, p = 6.9091e-11
- center error: partial rho = -0.6029, p = 1.0889e-19
- DCC4: partial rho = +0.3705, p = 2.0837e-07
- DCC8: partial rho = +0.5608, p = 1.0229e-16
- DCC10: partial rho = +0.5295, p = 9.2078e-15

### Full TDA protein-size-adjusted

Pocket size vs PR-AUC:

- partial rho = +0.7536
- p = 3.5085e-35

### ESM + structure protein-size-adjusted

Pocket size vs PR-AUC:

- partial rho = +0.7051
- p = 4.0204e-29

Primary interpretation:

Experimental pocket size is strongly associated with prediction performance, and the association persists after controlling for overall protein size.

Do NOT claim causality.

## TDA gain analysis

TDA PR-AUC gain relative to ESM + structure:

- pocket size: rho = +0.3663, p = 2.9262e-07
- pocket fraction: rho = +0.3434, p = 1.7109e-06
- protein size: rho = -0.0421, p = 0.5694

Primary interpretation:

The incremental TDA benefit is positively associated with experimental pocket size and is not significantly associated with protein size.

## Pocket-size quintiles

Full model mean PR-AUC:

| Pocket-size percentile | Mean PR-AUC |
|---|---:|
| P0–20 | 0.0867 |
| P20–40 | 0.1780 |
| P40–60 | 0.3332 |
| P60–80 | 0.4300 |
| P80–100 | 0.5589 |

Full model mean center error:

| Pocket-size percentile | Mean center error |
|---|---:|
| P0–20 | 16.82 Å |
| P20–40 | 11.83 Å |
| P40–60 | 8.34 Å |
| P60–80 | 7.35 Å |
| P80–100 | 6.00 Å |

## Case studies

| Protein | Pocket | PR-AUC | Gamma-10 error |
|---|---|---:|---:|
| 3q6u A | success | 0.7618 | 0.855 Å |
| 1xt3 B | intermediate | 0.3419 | 6.761 Å |
| 5ytb C | difficult failure | 0.0338 | 40.835 Å |
| 7yjc A | small-pocket failure | 0.0135 | 33.112 Å |

Case studies are illustrative and are NOT an independent validation set.

## Figures

### Main synthesis

phase167_validation/phase198_final_figure/phase198_final_synthesis.png

### Pocket-size figures

phase167_validation/phase190_figures/figure1_pocket_size_vs_pr_auc.png

phase167_validation/phase190_figures/figure2_pocket_size_vs_center_error.png

phase167_validation/phase190_figures/figure3_pr_auc_by_pocket_size_bin.png

### Case-study figures

phase167_validation/phase194_figures/phase194_case_study_4panel.png

phase167_validation/phase195_figures/3q6u_A_strong_success_zoom.png

phase167_validation/phase195_figures/1xt3_B_intermediate_zoom.png

phase167_validation/phase195_figures/5ytb_C_difficult_failure_zoom.png

phase167_validation/phase195_figures/7yjc_A_small_pocket_failure_zoom.png

## Claims allowed

1. TDA improves experimental-site prediction over ESM + structure on the 185-protein holdout.
2. ANM-MSF provides little additional benefit beyond TDA in this evaluation.
3. Probability sharpening improves localization over the unsharpened probability centroid.
4. The residue-probability field contains spatial information about experimentally defined binding sites.
5. Predicted residues are enriched around experimentally defined pocket centers.
6. Experimental pocket size is strongly associated with prediction performance.
7. Pocket-size associations persist after controlling for protein size.
8. Small experimentally defined pockets are a major observed source of difficulty.
9. TDA gain is positively associated with pocket size.

## Claims NOT allowed

1. Do not claim universal generalization.
2. Do not claim pocket size causes prediction difficulty.
3. Do not claim ANM-MSF is biologically irrelevant.
4. Do not claim gamma-10 discovers better residues than top-10%.
5. Do not claim gamma-10 is universally optimal.
6. Do not claim the oracle 1.43 Å localization is model performance.
7. Do not claim the case studies constitute independent validation.
8. Do not claim the model explicitly simulates pocket opening.
9. Do not claim the model predicts every possible cryptic pocket.
