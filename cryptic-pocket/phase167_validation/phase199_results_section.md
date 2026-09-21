# Results

## Experimental validation of cryptic-pocket residue prediction

We evaluated the three model configurations on an independent holdout of 185 proteins using experimentally defined ligand-contact residues as the reference labels. The ESM + structure baseline achieved a mean protein-level PR-AUC of 0.2952 and a median ROC-AUC of 0.8291. Incorporating topological data analysis (TDA) increased the mean protein-level PR-AUC to 0.3485 and the median ROC-AUC to 0.8595. The full model, which additionally incorporated ANM-MSF features, achieved a mean protein-level PR-AUC of 0.3476 and a median ROC-AUC of 0.8553.

The improvement from TDA was consistent across multiple evaluation criteria. Relative to ESM + structure, Full TDA significantly improved protein-level PR-AUC, ROC-AUC, and top-k experimental-site enrichment, precision, and recall across the evaluated fractions. Paired sign-flip permutation tests with 10,000 permutations showed significance after both Benjamini-Hochberg and Bonferroni correction for all 17 evaluated metrics (permutation p < 10^-4). In contrast, adding ANM-MSF to the TDA model did not produce a meaningful additional improvement in experimental-site PR-AUC or top-k enrichment. Thus, the principal feature contribution in the experimental holdout was associated with the TDA representation rather than the ANM-MSF feature.

At the global residue level, the same pattern was observed. PR-AUC increased from 0.2581 for ESM + structure to 0.3115 for Full TDA and 0.3076 for the Full model. These results provide direct experimental-site validation that the predicted residue probabilities identify residues associated with experimentally observed ligand-binding regions.

## Probability-based localization

We next investigated whether the residue-level probability field could be used to localize the predicted cryptic pocket in three-dimensional space. A learned coordinate-prediction head produced substantially weaker localization than several simple probability-based aggregation methods. We therefore tested probability sharpening, in which each residue probability was raised to the tenth power before normalization:

w_i = p_i^10 / sum_j p_j^10.

This operation concentrates the localization estimate around the highest-confidence residues without retraining the model.

For ESM + structure, the mean center error decreased from 11.768 Å using the unsharpened probability centroid to 10.159 Å after probability sharpening. For the Full model, center error decreased from 11.736 Å to 10.053 Å, while Full TDA decreased from 11.671 Å to 10.066 Å. The improvement was also observed across distance-based localization metrics, including the fraction of predictions within 4, 6, 8, and 10 Å of the experimentally defined pocket center.

Paired statistical testing showed that probability sharpening substantially improved localization over the unsharpened probability centroid and highly sparse top-1% aggregation. However, sharpened probabilities were generally comparable to selecting the top 10% of residues. These results indicate that the residue classifier contains spatial information that can be extracted without introducing an additional trained localization model.

The experimental-positive residue centroid produced a substantially smaller localization error, but this quantity uses the experimentally defined residues themselves and therefore serves only as an oracle/reference measurement rather than a deployable prediction method.

## Spatial enrichment of predicted residues

The predicted probability fields were further evaluated for spatial enrichment around experimentally defined pocket centers. Across the 185-protein holdout, high-probability residues were substantially enriched near the experimentally defined centers relative to the corresponding protein-wide background.

For the Full model, the strongest enrichment occurred among the top 1% of predicted residues: the enrichment was approximately 6.4-fold within 4 Å of the experimental center and approximately 10.7-fold within 6 Å. Full TDA produced comparable enrichment, including approximately 10.5-fold enrichment within 6 Å. ESM + structure showed the same qualitative pattern but with lower enrichment.

Protein-level paired permutation tests confirmed that these enrichments were highly significant across the evaluated probability fractions and distance thresholds. Direct comparisons between Full TDA and ESM + structure showed significant improvements across many localization-enrichment conditions, whereas Full TDA and Full model were not meaningfully different in these localization measures.

Together, these analyses show that the predicted residue probabilities are not merely enriched for experimental binding-site labels; they are also spatially concentrated around the experimentally defined pocket region.

## Dependence on experimental pocket size

We analyzed performance as a function of experimentally defined pocket size to identify systematic failure modes. Experimental pocket size was strongly associated with prediction performance across all three model configurations.

For the Full model, experimental pocket size had a Spearman correlation of 0.762 with protein-level PR-AUC. Pocket size was also negatively associated with localization error (rho = -0.584), indicating that larger experimentally defined binding sites were generally easier to identify and localize. These relationships remained strong after controlling for overall protein size. The partial Spearman correlation between pocket size and Full-model PR-AUC was 0.756 (p = 1.7 × 10^-35), while the partial correlation with center error was -0.603 (p = 1.1 × 10^-19).

The same trend was observed for Full TDA and ESM + structure. Across pocket-size quintiles, Full-model mean PR-AUC increased from 0.0867 in the smallest pocket-size group to 0.5589 in the largest group, while mean center error decreased from 16.82 Å to 6.00 Å.

The incremental benefit of TDA also varied with pocket size. TDA PR-AUC gain relative to ESM + structure was positively associated with experimental pocket size (rho = 0.366, p = 2.9 × 10^-7) and experimental pocket fraction (rho = 0.343, p = 1.7 × 10^-6). The gain was not significantly associated with overall protein size. These results identify small experimentally defined binding sites as a major observed failure regime while indicating that the TDA contribution is not simply explained by protein size.

## Representative successes and failures

Representative holdout proteins illustrate the range of model behavior. In 3q6u chain A, the Full model achieved a PR-AUC of 0.7618 and a probability-sharpened localization error of approximately 0.85 Å, with the predicted residues concentrated strongly around the experimentally defined binding region. In contrast, 5ytb chain C produced a PR-AUC of 0.0338 and a localization error of approximately 40.8 Å. The 7yjc chain A example represents an especially difficult small-pocket case, with only six experimentally defined residues, a PR-AUC of 0.0135, and a localization error of approximately 33.1 Å. The intermediate 1xt3 chain B case achieved a PR-AUC of 0.3419 and a localization error of approximately 6.8 Å.

The case studies illustrate that high-confidence predictions do not necessarily correspond to experimentally defined pockets. In difficult examples, the model could assign substantial probability to a region of the protein while failing to overlap the experimental site. This behavior is consistent with the broader pocket-size analysis, which showed that smaller experimental binding sites were systematically more difficult to identify.
