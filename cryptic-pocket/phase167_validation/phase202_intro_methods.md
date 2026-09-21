# Introduction

## Background

Proteins are dynamic three-dimensional molecules whose biological functions depend on interactions with other molecules. Many therapeutically relevant interactions occur at binding pockets that are not readily apparent in a single experimentally determined protein structure. These cryptic pockets can emerge when local conformational changes expose or reshape regions capable of binding a ligand.

Identifying cryptic binding pockets from apo structures is therefore a challenging computational problem. Conventional pocket-detection approaches generally rely on geometric or physicochemical properties of the observed structure. However, cryptic pockets may not exhibit the geometric characteristics of conventional binding sites in the apo state.

Recent advances in protein language models provide another source of information. Protein language models can encode sequence-derived representations that capture evolutionary and structural information without requiring explicit evolutionary alignment during inference. Combining such representations with structural features provides a potential framework for detecting cryptic binding regions.

At the same time, protein structure contains information that is inherently geometric and topological. Topological data analysis (TDA) provides a mathematical framework for representing structural relationships across multiple spatial scales. Such representations may capture characteristics of protein regions that are difficult to express using individual residue-level structural descriptors.

## Research gap

Existing computational approaches provide useful baselines for pocket detection, but cryptic-pocket prediction remains challenging because the target site may be poorly defined in the apo structure. In addition, successful residue identification does not necessarily imply successful three-dimensional localization of the pocket.

A model may assign high probability to experimentally relevant residues while producing a spatial prediction that is poorly concentrated. Conversely, a spatially concentrated prediction may not correspond to the experimentally observed binding site. Evaluating both residue-level identification and spatial localization therefore provides a more complete assessment of cryptic-pocket prediction.

A further challenge is determining which structural representations provide useful information beyond protein-language-model embeddings. In particular, the contribution of topological structure and dynamic information can be evaluated through controlled feature comparisons.

## Research question

This study asks:

> Can topological structural representations improve cryptic-pocket residue prediction from apo protein structures, and can the resulting residue probability field be used to localize experimentally defined binding sites?

We additionally investigate whether ANM-MSF dynamic features provide incremental predictive information and which characteristics of experimentally defined pockets are associated with prediction difficulty.

## Hypotheses

We hypothesized that:

1. Adding TDA-derived structural information to ESM and conventional structural features would improve cryptic-pocket residue prediction.
2. Adding ANM-MSF features would provide additional information related to protein dynamics.
3. The residue-probability field would contain spatial information that could be used to localize the predicted pocket.
4. Smaller experimentally defined binding sites would be more difficult to identify and localize.

# Methods

## Dataset

The study used the CryptoBench benchmark, consisting of 942 protein-chain graphs. The original benchmark contains predefined training, validation, and test partitions. Model development was performed using the available training and validation data, while the final evaluation described here used a separate 185-protein experimental holdout.

Each protein-chain graph contains residue-level features, graph connectivity, three-dimensional coordinates, and binary residue labels. The feature representation contains 542 dimensions per residue, including conventional structural descriptors, ESM2-35M sequence embeddings, ANM-MSF information, and TDA/burial features.

The experimental holdout was reserved for final evaluation and was not used to retrain the prediction model.

## Experimental binding-site labels

For final validation, residue labels were derived from experimentally characterized ligand-binding structures associated with the CryptoBench proteins.

An experimental binding-site residue was defined using the protein residues whose heavy atoms were within the predefined contact cutoff of the experimentally associated ligand. Water and unrelated heteroatoms were excluded. The resulting experimentally defined ligand-contact residues were mapped onto the corresponding apo protein graph.

These labels were used only for final evaluation and localization analysis.

## Model configurations

Three feature configurations were evaluated:

### ESM + structure

The baseline model used ESM2-35M sequence embeddings together with conventional structural features.

### Full TDA

The baseline representation was augmented with TDA and burial-derived structural features.

### Full model

The Full model included the TDA representation and ANM-MSF features in addition to the baseline features.

This design allowed the contribution of TDA to be evaluated relative to the baseline and the incremental contribution of ANM-MSF to be evaluated relative to TDA.

## Graph representation

Each protein was represented as a residue-level graph.

For each residue node i, the model received a feature vector x_i and three-dimensional coordinate p_i. Edges represented structural relationships between residues and included edge attributes describing the corresponding pairwise relationship.

The model produced a probability p_i for each residue representing its predicted likelihood of belonging to the cryptic-pocket region.

## Residue-level evaluation

Performance was evaluated using precision-recall area under the curve (PR-AUC) and receiver operating characteristic area under the curve (ROC-AUC).

PR-AUC was emphasized because the positive class represents a relatively small fraction of protein residues. Additional evaluations measured enrichment, precision, and recall among the highest-probability predicted residues.

Metrics were calculated both globally across residues and at the protein level. Protein-level results were obtained by calculating metrics independently for each protein and then summarizing across the 185-protein holdout.

## Statistical comparison

For paired model comparisons, each protein contributed one observation per model configuration after collapsing across the eight evaluation seeds.

Paired sign-flip permutation tests with 10,000 permutations were used to evaluate differences between model configurations. Both Benjamini-Hochberg false-discovery-rate correction and Bonferroni correction were applied to account for multiple comparisons.

All statistical tests were performed on the predefined experimental holdout rather than on training or model-selection data.

## Three-dimensional localization

The predicted residue probabilities were converted into a three-dimensional pocket-center estimate using the residue coordinates.

The initial probability centroid was calculated as:

c = sum_i p_i r_i / sum_i p_i

where p_i is the predicted residue probability and r_i is the three-dimensional coordinate of residue i.

Because the probability field was sparse, a probability-sharpening transformation was also evaluated:

w_i = p_i^gamma / sum_j p_j^gamma

A common gamma value of 10 was selected from the evaluated set of exponents and then applied to all three model configurations.

The sharpened center was calculated as:

c_gamma = sum_i w_i r_i

No model retraining was performed for this localization procedure.

Localization was evaluated using Euclidean center error and distance-based contact metrics, including DCC4, DCC6, DCC8, and DCC10. These represent the fraction of predictions whose estimated pocket center lies within 4, 6, 8, or 10 Å of the experimental pocket center.

## Spatial enrichment

To determine whether high-probability residues were spatially concentrated around the experimental pocket, predicted residues were ranked by probability.

For several top-ranked fractions, including the top 1%, 5%, 10%, and 20%, the fraction of residues located within specified distances of the experimentally defined pocket center was compared with the corresponding protein-wide background fraction.

Enrichment was calculated as:

enrichment = observed fraction / background fraction.

This analysis evaluates whether the residue-probability field contains spatial information independently of the localization aggregation method.

## Pocket-size analysis

The number of experimentally defined ligand-contact residues was used as the experimental pocket-size measure. Pocket fraction was calculated as the number of experimental residues divided by the total number of residues in the protein.

Spearman rank correlations were used to evaluate associations between pocket size, prediction performance, and localization error.

Because larger proteins may contain more residues and potentially larger binding sites, partial Spearman correlations were additionally calculated while controlling for overall protein size.

These analyses were observational and were used to characterize performance differences rather than establish causal relationships.

## TDA gain analysis

For each protein, the improvement associated with TDA was calculated as the difference between Full TDA and ESM + structure performance.

The relationship between this gain and experimental pocket size, experimental pocket fraction, and protein size was then evaluated using Spearman correlation.

This analysis was performed after model evaluation and was not used to tune the model.

## Case-study analysis

Four proteins were selected to illustrate qualitatively different performance regimes:

- 3q6u chain A: strong success
- 1xt3 chain B: intermediate performance
- 5ytb chain C: difficult failure
- 7yjc chain A: small-pocket failure

For each case, the experimentally defined binding site, predicted high-probability residues, experimental center, and probability-sharpened predicted center were visualized in three dimensions.

These examples were selected to illustrate patterns observed in the full holdout and were not treated as an independent validation set.
