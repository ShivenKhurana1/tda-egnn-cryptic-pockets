# Phase 203: Front Matter

## Proposed Title

Topological Representation Learning for Cryptic Protein Pocket Prediction and Spatial Localization

## Alternative Title

Integrating Protein Language Models and Topological Structure for Cryptic Binding-Site Prediction

## Abstract

Cryptic protein binding pockets are ligand-binding regions that are not readily apparent in an apo protein structure, making their computational identification challenging. We developed a residue-level graph-learning framework that combines protein language model embeddings, structural descriptors, topological data analysis (TDA), and anisotropic network model mean-square fluctuations (ANM-MSF) to predict experimentally defined cryptic binding-site residues. The model was evaluated on an independent 185-protein experimental holdout using ligand-contact residues derived from experimentally characterized protein-ligand structures.

Adding TDA to ESM-derived and conventional structural features increased mean protein-level PR-AUC from 0.2952 to 0.3485 and median ROC-AUC from 0.8291 to 0.8595. Adding ANM-MSF provided little additional improvement, with the full model achieving a mean PR-AUC of 0.3476. The predicted residue-probability field also contained spatial information that could be extracted without retraining the model. Probability sharpening reduced mean center error from 11.768 Å to 10.159 Å for ESM + structure and from 11.736 Å to 10.053 Å for the full model. High-probability residues were also strongly enriched around experimentally defined pocket centers.

Performance depended strongly on experimental pocket size. For the full model, pocket size showed a partial Spearman correlation of 0.756 with PR-AUC after controlling for protein size (p = 1.7 × 10^-35), while the partial correlation with localization error was -0.603 (p = 1.1 × 10^-19). These associations identify small experimentally defined binding sites as an important observed source of prediction difficulty.

Overall, the results show that topological structural information improves cryptic-pocket residue prediction on the experimental holdout and that the learned residue-probability field contains spatial information that can be used for pocket localization. The study also provides a quantitative characterization of pocket-size-dependent performance and identifies small binding sites as an important direction for future model development.

## Research Question

Can topological structural representations improve cryptic-pocket residue prediction from apo protein structures, and can the resulting residue probability field be used to localize experimentally defined binding sites?

## Hypotheses

### H1: TDA

Adding TDA-derived structural features to ESM and conventional structural features will improve cryptic-pocket residue prediction on an independent experimental holdout.

### H2: ANM-MSF

Adding ANM-MSF features to the TDA-enhanced representation will provide additional predictive information related to cryptic binding sites.

### H3: Spatial information

The residue-level probability field will contain spatial information that can be used to localize experimentally defined binding sites.

### H4: Pocket size

Smaller experimentally defined binding sites will be more difficult for the model to identify and localize.

## Independent Variables

1. Model feature configuration:
   - ESM + structure
   - ESM + structure + TDA
   - ESM + structure + TDA + ANM-MSF

2. Probability localization method:
   - unsharpened probability centroid
   - probability sharpening
   - top-k residue centroid methods

3. Experimental pocket size and pocket fraction.

## Dependent Variables

1. Protein-level PR-AUC
2. ROC-AUC
3. Top-k enrichment
4. Top-k precision
5. Top-k recall
6. Pocket-center localization error
7. DCC4
8. DCC6
9. DCC8
10. DCC10

## Key Contributions

1. A graph-based cryptic-pocket prediction framework combining protein language model, structural, and topological representations.

2. Direct evaluation against experimentally defined ligand-contact residues on a 185-protein holdout.

3. Evidence that TDA provides a measurable improvement over ESM + structure for experimental-site prediction.

4. Demonstration that the residue-probability field contains spatial information that can be extracted through non-retrained probability sharpening.

5. Quantitative characterization of pocket-size-dependent prediction difficulty after controlling for protein size.

6. Identification of small experimentally defined pockets as an important failure regime for future model development.

## Keywords

cryptic pockets; protein binding sites; protein language models; topological data analysis; graph neural networks; computational biology; protein structure; ligand binding; machine learning; structural bioinformatics
