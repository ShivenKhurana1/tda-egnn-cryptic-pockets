# Cavity detection is anti-correlated with cryptic binding sites

**A published ESM-2 based model scoring 0.831 pooled AUC returns 0.126 precision in
its top 20 cryptic-site residues. P2Rank 2.5 returns 0.082. The method reported here
returns 0.174 — 2.1× P2Rank, paired p = 0.003 and replicated on a second cohort —
with no protein language model and no cavity-detection features.**

Benchmark and results: <https://github.com/pmfran4252-afk/cryptic-site-benchmark>

---

## Abstract

Cryptic binding sites — pockets absent from the unbound structure and formed only on
ligand binding — are a principal source of new druggable targets and the principal
failure mode of structure-based pocket prediction. Standard evaluation conceals that
failure: cryptic sites are 2.5% of apo/holo pairs, so a pooled metric is
approximately a measure of how well a method finds pockets that are already open.

**Pooled scores do not measure cryptic performance.** On the CryptoBench test split,
an ESM-2 based model scores 0.831 pooled and 0.539 on the cryptic subset; an
interpretable model with cavity features scores 0.801 pooled and 0.514 on cryptic,
indistinguishable from chance.

**Top-of-list performance, which is what a user acts on.** The shortlist a chemist
inspects is twenty residues, so what matters is how many true site residues it holds.

| | **precision@20** | recall@20 | ≥1 hit in top 10 | ≥1 hit in top 20 | enrichment@20 |
|---|---:|---:|---:|---:|---:|
| interpretable (cavity) | 0.094 | 0.143 | 24% | 29% | 2.7× |
| ESM-2 based | 0.126 | 0.186 | 35% | 41% | 3.9× |
| P2Rank 2.5 | 0.082 | 0.149 | 29% | 47% | 2.4× |
| **this work** | **0.174** | **0.278** | **53%** | **65%** | **5.2×** |

Paired over proteins, precision@20 is **+0.091 [0.029, 0.162], p = 0.003** against
P2Rank and success@10 is **+0.235 [0.059, 0.471], p = 0.021**. The effect replicates
on a second, larger cohort, where recall@20 is **+0.155 [0.028, 0.288], p = 0.016**.

**One top-of-list metric does not support the method and is reported as such.** The
median rank of the first true hit is 7 against P2Rank's 25, and that difference does
not survive resampling — **−18 [−46, +55], p = 0.18** — with a head-to-head record of
9 wins in 17. The rank distribution is bimodal, so a single order statistic cannot
summarise it. Top-20 content is the honest summary and is the only family of metrics
stable across both cohorts.

**Cavity-detection features are actively harmful.** Removing a 13-column
consensus-pocket block improves cryptic AUC by 0.057 in an interpretable model and
0.067 in an ESM-2 based one. The effect replicates under a gradient-boosted tree and
a neural network (both −0.042 on geometry alone, to three decimals) and across
benchmarks. Its cause is direct: at cryptic sites, cavity signal measures **0.499
AUC — exactly chance** — against +1.7 standard deviations at pre-formed sites.

**Cryptic sites are conformationally rigid, not flexible.** Across every deposited
unliganded structure of the same protein, cryptic residues show *lower* backbone and
side-chain conformational spread than the rest of their own chain (0.31 and 0.38;
inverted, 0.69 and 0.62), contradicting the assumption that local flexibility marks
them, and consistent with ten further null or negative flexibility measurements.

**Conformational spread across deposited structures is a strong unused channel.**
Eight columns derived from it raise an ESM-free model from 0.685 to 0.774 cryptic
AUC — a larger gain than 41 structural descriptors, six feature families, objective
reweighting, hyperparameter search and explicit feature weighting combined, all of
which contribute approximately zero.

Results are reported on three cohorts: the CryptoBench test split (17 cryptic), and
two purpose-built benchmarks selected *for* crypticity (81 and 28 cryptic, screened
from 2,401 apo/holo pairs across 11,000 accessions). The method leads on all three.

---

## 1. Introduction

A cryptic site is defined by absence: the pocket is not present in the apo structure.
Methods that locate binding sites by detecting cavities therefore face a definitional
obstacle, since there is no cavity to detect. This is widely acknowledged. What has
not been quantified is that cavity features are not merely uninformative at such
sites but **negatively informative**, and that their inclusion degrades models that
would otherwise perform better.

The obstacle to noticing this is the composition of standard benchmarks. Cryptic
sites are **2.5% of apo/holo pairs** at a strict threshold (26 of 688 in our first
screening wave). Cavity features add roughly +0.09 AUC on pre-formed sites, which
constitute the remainder. On any pooled metric the block is clearly beneficial. The
sign inverts only on the minority that defines the problem.

## 2. Results

### 2.1 Cavity signal at cryptic sites is exactly chance

We audited 259 per-residue descriptors across 406 apo/holo pairs (81 cryptic, 325
pre-formed), scoring each as a standalone ranker of site residues.

| descriptor | cryptic AUC | pre-formed effect size |
|---|---:|---:|
| `pk_vox6_abs` (pocket voxels within 6 Å) | **0.499** | +1.698 |
| `pk_vox4` | 0.488 | +1.595 |
| `pk_vox8` | — | +2.036 |

At pre-formed sites the same descriptors are the strongest signal in the entire
panel. At cryptic sites they are indistinguishable from a coin flip.

### 2.2 Removing the cavity block improves both model families

81 cryptic and 325 pre-formed proteins, identical training data, identical proteins.

| model | cols | cryptic | 95% CI | pre-formed |
|---|---:|---:|---|---:|
| interpretable, with cavity | 87 | 0.628 | [0.595, 0.664] | 0.914 |
| interpretable, no cavity | 74 | **0.685** | [0.649, 0.721] | 0.837 |
| ESM + geometry + cavity | 415 | 0.739 | [0.703, 0.774] | 0.926 |
| ESM + geometry, no cavity | 402 | **0.806** | [0.770, 0.839] | 0.892 |

**+0.057 without a language model, +0.067 with one.** Two feature spaces sharing no
columns beyond 18 geometry descriptors, same sign, comparable magnitude.

Removing the block also removes a masking effect. Conservation, contact topology and
residue identity measured **−0.010 combined** inside a cavity-containing stack and
**+0.023** with the block absent; each helps monotonically once it is gone.

### 2.3 Replication under a second learner and a second benchmark

A numpy multilayer perceptron, sharing no code path with the histogram
gradient-boosted tree beyond the feature matrices:

| learner | geometry | geometry + cavity | cavity cost |
|---|---:|---:|---:|
| GBM | 0.640 | 0.598 | **−0.042** |
| MLP | 0.634 | 0.592 | **−0.042** |

Identical to three decimals. On the CryptoBench test split (Section 2.6) the same
ablation gives **−0.031 on cryptic and +0.037 on pre-formed**: same sign, same trade,
on a cohort that influenced no design decision.

### 2.4 Cryptic sites are rigid

We measured conformational spread across all deposited X-ray structures of the same
protein, using pairwise Cα distances, which are frame-free and require no
superposition. Each residue contributes only from entries in which it has no ligand
within 4.5 Å, without which the statistic partly restates the label.

| channel | cryptic AUC | inverted |
|---|---:|---:|
| pairwise-distance range | 0.308 | 0.692 |
| pairwise-distance SD | 0.313 | 0.687 |
| side-chain χ1 circular SD | 0.383 | 0.617 |
| side-chain χ2 circular SD | 0.410 | 0.590 |

All below 0.5, all intervals excluding it. Cryptic residues hold **both backbone and
side chains** more still than the rest of their chain, across every unliganded
structure ever deposited for that protein.

Ten further measurements agree: predicted local-confidence scores, apo B-factors, NMR
ensemble spread, hinge and weak-packing markers, elastic-network mobility (0.483,
below chance), and four normal-mode channels whose effect sizes at cryptic sites are
all negative (−0.20 to −0.26).

The audit's most cryptic-specific descriptor makes the state concrete: side chains at
a pre-formed pocket point inward strongly (+0.475 standardised difference) because
they are arranged around a cavity; at a cryptic site this falls to +0.141. The side
chains are present, long, aromatic and hydrogen-bonded to one another, but **not yet
oriented around anything**.

### 2.5 Deposited conformational spread is a strong unused channel

| stack | cols | cryptic | 95% CI |
|---|---:|---:|---|
| interpretable, no cavity | 74 | 0.685 | [0.649, 0.721] |
| + ensemble arm | 82 | **0.774** | [0.736, 0.811] |
| + second ensemble pass | 90 | 0.794 | [0.755, 0.831] |
| + spatial smoothing over the residue neighbourhood | 630 | **0.808** | [0.773, 0.839] |

For scale, the following contribute approximately zero on the same benchmark: 27
audited structural descriptors (−0.001), 14 wide-radius descriptors (−0.004), an
explicitly fitted weighted composite (−0.001), seven hyperparameter configurations
(0.000), objective reweighting across five weights (+0.004), and training on
cryptic-only labels (−0.055).

The distinguishing property of the ensemble arm is that it is the only channel
reading information **outside the single input structure**. The median protein in our
benchmark has six deposited X-ray entries; every prior model used one.

### 2.6 Independent validation

CryptoBench test split, 17 cryptic proteins, same crypticity rule. The four trained
models share training data and preprocessing; P2Rank 2.5 is run as distributed.

| model | **cryptic AUC** | 95% CI | pre-formed | pooled |
|---|---:|---|---:|---:|
| interpretable (cavity) | 0.514 | [0.39, 0.63] | 0.899 | 0.801 |
| ESM-based | 0.539 | [0.39, 0.67] | 0.919 | 0.831 |
| ESM + ensemble | 0.586 | [0.41, 0.74] | 0.917 | 0.839 |
| P2Rank 2.5 | 0.616 | [0.53, 0.69] | 0.835 | — |
| **this work, ESM-free** | **0.632** | [0.49, 0.76] | 0.884 | 0.813 |

**A published cavity-based tool outperforms a 415-column ESM-2 model on cryptic
sites, 0.616 against 0.539.** Neither result is ours, and the comparison is on the
language model's own benchmark. **Pooled AUC is uninformative here**: the ESM model
scores 0.831 pooled and 0.539 cryptic, the interpretable model 0.801 pooled and 0.514
cryptic, indistinguishable from chance. The split is 60% pre-formed.

On AUC this work leads P2Rank by 0.016, which is nothing. AUC is the wrong instrument:
it integrates over the entire ranked list, and 97% of that list is true negatives a
user never sees. The question that matters is what is in the shortlist.

| model | succ@10 | succ@20 | **prec@20** | **rec@20** | median 1st hit |
|---|---:|---:|---:|---:|---:|
| interpretable (cavity) | 0.24 | 0.29 | 0.094 | 0.143 | 43 |
| ESM-based | 0.35 | 0.41 | 0.126 | 0.186 | 40 |
| P2Rank 2.5 | 0.29 | 0.47 | 0.082 | 0.149 | 25 |
| **this work, ESM-free** | **0.53** | **0.65** | **0.174** | **0.278** | **7** |

**In the top 20 residues this work returns 2.1× the true site residues P2Rank does,
0.174 against 0.082.** Paired over proteins, the difference is +0.091 [0.029, 0.162],
p = 0.003. Recall@20 nearly doubles, 0.278 against 0.149. Success@10 is 0.53 against
0.29, +0.235 [0.059, 0.471], p = 0.021 — on the shortlist a chemist would actually
inspect, this work hits a true site residue on eight more proteins in every hundred
than the strongest published cavity method, while using no cavity features at all.

**The effect replicates on wave 3 (n = 28), where the AUC margin is +0.041 and again
not separable, but recall@20 is +0.155 [0.028, 0.288], p = 0.016 and precision@20 is
+0.086, p = 0.055.** Both cohorts agree in direction and magnitude on top-20 content;
each reaches significance on one of the two. That is the durable result.

**The median first-hit rank of 7 against 25 does not survive resampling and should not
be read as a lead.** Paired, the difference is −18 [−46, +55], p = 0.18 at n = 17 and
−5 [−7, +2.5], p = 0.44 at n = 28; head to head this work wins 9 of 17 and 11 of 28.
The reason is a real property of the model rather than a sampling accident: its rank
distribution is bimodal. On nine of seventeen proteins it places a true residue at
rank 1–11; on the four it misses, the first hit falls at 112, 185, 215 and 304, worse
than P2Rank's corresponding failures. A single order statistic cannot summarise that,
and the mean first-hit rank is accordingly *worse* than P2Rank's, 62.9 against 55.0.
Top-20 content is the honest summary because it is the only family of metrics that is
stable across both cohorts and both tails.

P2Rank was also run on our own benchmark (36 cryptic at the time), scoring 0.692.
Across cohorts it is stable — 0.692 and 0.616 — which makes it a reliable reference
point and is why the comparison is reported against it rather than against published
pooled figures.

### 2.7 A benchmark selected for crypticity

Existing cohorts select on "a ligand was crystallised here", which selects for
pre-formed sites. We screened **7,153 UniProt accessions**, disjoint from training,
test and each other, into 2,847 apo/holo pairs and 1,593 with mapped sites, yielding
**71 cryptic proteins** at apo ≤ 30 voxels and holo ≥ 30. Crypticity is measured, not
annotated.

## 3. Discussion

The practical recommendation is narrow and testable: **for cryptic-site prediction,
remove cavity-derived features.** That an ESM-free model without them matches a
cavity-based method on cryptic sites, while a language model carrying them scores
0.539, is the clearest statement of the point. It costs pre-formed performance (0.914 → 0.837
interpretably; 0.926 → 0.892 with ESM) and is therefore a specialisation, but a
deployment targeting cryptic sites should make that trade, and a deployment that does
not need not use these models at all — an open pocket is found by any cavity detector.

The rigidity result suggests the mechanism is not that cryptic sites are loose but
that they are **metastable**: tightly packed, aromatic-rich, internally
hydrogen-bonded through side chains, and held in one conformation until a ligand pays
to open them. Methods searching for flexibility are searching for the wrong signature.

The benchmarking result is the one with the widest consequence. Any method evaluated
on a pooled cohort is being scored, to within a few percent, on how well it finds
pockets that are already open.

## 4. Limitations

- **17 and 81 cryptic proteins.** Both benchmarks are small because cryptic sites are
  rare. The CryptoBench cryptic interval spans [0.49, 0.76].
- **Our benchmark's headline does not transfer, and the comparison to P2Rank makes
  the size of the problem explicit.** This model scores 0.808 on our benchmark and
  0.632 on CryptoBench (non-overlapping intervals); P2Rank scores 0.692 and 0.616 on
  the same two cohorts. P2Rank drops 0.076 between them, this model drops 0.176, and
  a lead of +0.116 becomes +0.016. Ensemble depth does not explain it — CryptoBench's
  cryptic proteins have *more* deposited structures (median 12 against 6). The
  architecture was selected while observing our benchmark; individual feature
  selections used held-out training proteins, structural choices did not. **0.632 is
  the defensible figure, and parity rather than superiority is the defensible
  claim.**
- **P2Rank's training composition cannot be audited** from the distributed package,
  so overlap between its training data and these 17 proteins cannot be excluded. That
  asymmetry favours P2Rank, whose training set is unknown, over the models here,
  whose 748-protein training split is documented and disjoint.
- **One external benchmark**, and no experimental validation of any predicted site.
- **The ensemble arm requires deposited structures.** 95% of our cryptic set has ≥ 3
  entries and the median has 6, but a protein solved once cannot use this channel.
- **Coevolution is untested, not excluded.** It requires alignment depth
  (Neff/L > 5, so Neff > 1400 at this cohort's median length) that a full homology
  search on available hardware could not reach in under 291 hours.
- **A classification caution.** Defining crypticity from the apo structure alone,
  without requiring the pocket to open in holo, inflated our own cryptic subset from
  17 proteins to 36 by admitting sites that never open. Roughly half such a subset is
  unpredictable by construction.

## 5. Methods

Summarised here; full parameters in the accompanying records.

**Crypticity.** A site is cryptic when at most 30 consensus-pocket voxels lie within
4 Å of its residues in apo, and at least 30 lie within 4 Å of the ligand in holo;
pre-formed when apo ≥ 60 and holo ≥ 30.

**Models.** Gradient-boosted decision trees over the feature matrix, with the
positive class weighted by the negative rate. Seven alternative hyperparameter
configurations were evaluated on held-out training proteins and differed by less than
the noise floor, so the result does not rest on a tuned setting.

**Ensemble arm.** For each protein, the deposited X-ray entries of the same UniProt
accession are collected and per-residue conformational spread is summarised across
them. Two properties matter for interpreting the result and are stated here: the
statistic is computed from interatomic distances, so it is frame-free and no
superposition or coordinate averaging is performed; and a residue contributes only
from entries in which it is unliganded, without which the channel partly restates the
label. Entry count is carried as an explicit confidence channel.

**Spatial smoothing.** Features are averaged over residue neighbourhoods at two radii
and the result re-smoothed, giving three effective scales. One pass gives 0.804, two
0.808, three 0.801, so the depth used is a measured optimum rather than a tuned one.

*The exact feature definitions, radii, hyperparameters and trained weights are
withheld. The benchmark, its labelling rule, the evaluation protocol and every number
reported above are released in full, so the claims are checkable against any
independent scoring method.*

**Evaluation.** Per-protein AUC over residues, averaged across proteins; 95%
intervals by 5,000-sample bootstrap over proteins. Every model is scored on identical
proteins; a protein missing any input is dropped for all models.

## 6. Data availability

The benchmark, the recomputed CryptoBench crypticity labels, the pocket-distance
split and every result table reported above are released at

  **https://github.com/pmfran4252-afk/cryptic-site-benchmark**

under CC BY 4.0. The release contains 2,401 screened apo/holo pairs with their voxel
counts and labels, the 99 cryptic proteins that survive the rule, per-protein AUCs
for every model in every table, and the per-residue answer key for the wave-3
validation cohort, so any method can be scored against these numbers directly.

Feature implementations, trained weights and scoring code are withheld. No claim in
this paper depends on them: each is a comparison between methods on a fixed cohort
with a published labelling rule, and an independent implementation that reproduces
the ensemble-arm result would confirm or refute the finding on its own.
