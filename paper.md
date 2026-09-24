# Fusing Persistent Homology with Equivariant Graph Neural Networks for Protein Binding-Site and Cryptic-Pocket Localization

## Abstract

Predicting where a small molecule will bind a protein is a central problem in
structure-based drug discovery, and it becomes substantially harder for
*cryptic* pockets — binding sites that are not visible in a ligand-free (apo)
structure and only open up upon binding. This project tests whether features
from persistent homology (a tool from topological data analysis, TDA) improve
an E(n)-equivariant graph neural network's (EGNN) ability to both classify
binding-site residues and localize a pocket's 3D center, on standard and
cryptic-pocket benchmarks alike. Protein structures are represented as
residue-level graphs; each residue carries one-hot identity, backbone
dihedral, ESM2 protein-language-model embedding, and persistent-homology
(H0/H1/H2 + burial) features, which are propagated through equivariant graph
convolution layers and fused via a per-layer gated fusion module before two
task heads: residue classification and centroid-based pocket localization.
On the COACH420 and HOLO4K training pool, adding topology to sequence and
burial features gave a statistically significant classification gain
(+0.0423 mean-average-precision against a two-feature burial baseline,
10/10 cross-validation folds, p<1e-5). A later, stronger control shows
that most of this gain is ordinary local geometry: against 17 plain
per-residue geometry features (neighbor counts, solvent exposure,
half-sphere exposure, local shape, ray-cast buriedness, hull and residue
depth), geometry alone outperforms topology alone at every embedding scale
tested (35M/650M/3B), and topology adds a residual on top of geometry that
is confirmed by both a paired t-test and a Wilcoxon test at 650M
(+0.0091 PR-AUC, 8/10 folds; +0.0032 ROC-AUC, 9/10 folds) and at 3B
(+0.0115 PR-AUC, 10/10 folds; +0.0040 ROC-AUC, 9/10 folds), though only on
PR-AUC at the original 35M scale (+0.0097, 9/10 folds; ROC-AUC not
significant) — the effect grows more robust, not less, with embedding
scale. The same control on CryptoBench (35M only) finds no added benefit
from topology (+0.0008 PR-AUC, 95% CI -0.005 to +0.007). Separately,
after fixing a coordinate-frame bug and training a dedicated localization
head, a trained-localization system that reaches distance-to-center-success
(DCC@4Å) of 0.593 on COACH420 — a real win over a live, current run of the
standard tool P2Rank (+0.072) — while landing at approximate statistical
parity with P2Rank on HOLO4K (0.645 vs 0.658, not distinguishable from
noise). A geometry control for localization, run at the original 2,055-chain
scale, again finds geometry alone beats topology alone (+0.0305 pooled
DCC@4Å, 7/8 seeds); geometry plus topology beyond geometry alone is null
per seed (+0.0051, 3/8 seeds) but positive and increasingly consistent as
ensemble size grows (69/70 disjoint 4-seed half-splits positive), landing
at a two-way-bootstrapped +0.0188 (95% CI -0.0096 to +0.0456) that does not
clear this project's confirmation bar. Repeating the localization control at
650M and 3B embedding scale does not repeat classification's strengthening:
geometry still matches or beats topology alone at every scale, and the
topology residual over geometry is +0.0206 (650M; t p=0.037, Wilcoxon
p=0.078) and -0.0007 (3B), pooling to +0.0084 (95% CI -0.0027 to +0.0194)
across scales with no scale trend — not established for localization.
With 3.5x more training data (7,513 chains) the localization residual is
negative (-0.0242 per seed, geometry better in 7 of 8 seeds; two-way
bootstrap -0.0217, 95% CI -0.068 to +0.024), and geometry alone is the best
arm (ensemble COACH420 DCC 0.620 vs. 0.586 for topology). On the CryptoBench cryptic-pocket benchmark, the same architecture,
extended across three ESM2 embedding scales (35M/650M/3B parameters) and
combined with transfer learning from the main pipeline, a regularized
logistic-regression meta-combiner across scales, and topology and transfer
learning levers each confirmed by a paired significance test (topology:
paired t=6.11, p=0.00049, Cohen's d=2.16; transfer learning: paired
t=3.97, p=0.0054, Cohen's d=1.40; both n=8 seeds; the topology comparison
used a baseline with no burial or exposure features, and the geometry
control described below attributes that gain to ordinary geometry), reaches AUC 0.918 and
Matthews correlation coefficient (MCC) 0.470, matching or exceeding the
CryptoBench authors' own published state of the art (AUC 0.87, MCC 0.42)
and the most recent published baseline we found on this exact benchmark,
UniPocket (2026, AUC 0.816 — computed under a different, macro-averaged
convention, so not a strictly like-for-like comparison; see Results).
A chain-level bootstrap (2,000 resamples) establishes that the ensemble's
AUC gain over the best single model is statistically significant (95% CI
excludes zero) while its MCC gain is not (95% CI includes zero) — we
report both conclusions rather than rounding the second into a claimed
win. A dedicated localization head for cryptic pockets — a capability
with no existing published comparison — achieves a 20.9%–24.7% hit rate
at 4Å, confirming this is a genuinely harder geometric task than
open-pocket localization.

## Introduction

Roughly 20–30% of clinically relevant binding sites are believed to be
cryptic: absent or collapsed in the unbound protein structure and only
detectable once a ligand has induced a conformational change to open them.
This matters directly for drug discovery, because most structure-based
virtual screening and pocket-detection pipelines are built and validated on
*holo* (ligand-bound) structures, where the cavity is already visible; a
tool trained or tuned that way will systematically miss cryptic sites in the
apo structures that are actually available at the start of a drug-discovery
campaign. Detecting a cryptic pocket from an apo structure alone therefore
requires inferring a cavity's future existence from subtler shape and
dynamics precursors rather than from the cavity itself.

Two computational tools are combined in this project to attack that
problem. **Persistent homology** is a technique from topological data
analysis that tracks how the connected components, loops, and voids of a
point cloud (here, a protein's heavy atoms) appear and disappear as a
distance threshold is swept from zero to infinity; the resulting "birth" and
"death" values of these topological features (grouped by dimension: H0 for
connected components, H1 for loops/tunnels, H2 for enclosed voids) summarize
a neighborhood's *shape* in a way that is insensitive to the exact atomic
coordinates, which makes it a natural candidate for describing a pocket's
cavity geometry. **E(n)-equivariant graph neural networks (EGNNs)** are a
class of graph neural network built so that if the input 3D structure is
rotated or translated, the learned per-node representations transform
correspondingly (equivariantly) rather than needing to relearn the same
pattern in every orientation; this property is a natural fit for physical
structures where there is no privileged coordinate frame.

The hypothesis tested throughout this project is that fusing persistent
homology features into an EGNN improves both binding-site classification
and pocket-center localization, beyond what either sequence/burial features
or an EGNN alone can achieve, and that this benefit generalizes from
standard (open-pocket) benchmarks to the harder cryptic-pocket setting.

**Related work.** Neither ingredient here is individually new. Equivariant
GNNs for binding-site-type prediction are an active area: EquiPocket and
VN-EGNN target small-molecule pockets directly (the benchmarks and
literature numbers compared against throughout this paper); EquiPNAS,
DeepPANB, MPBind, and E(Q)AGNN-PPIS apply closely related equivariant
architectures to protein–nucleic-acid and protein–protein interaction
sites — a different binding partner, but the same underlying design
pattern. Persistent homology applied to protein structure is likewise
established: PersGNN combines topological features with a graph neural
network for structure-based protein *function* prediction (not binding
sites), and Jiang and Lugo-Martinez combine topological data analysis
with geometric deep learning to characterize the internal structure of
protein binding pockets that some other tool has already located — a
downstream annotation task, not pocket *detection*. Topology has also
been applied to cryptic pockets specifically: CrypToth ranks candidate
cryptic sites using persistent homology computed from mixed-solvent
molecular dynamics simulation trajectories, rather than a trained model
acting on a single static structure. This project's narrower, more
specific claim is the combination of these threads: fusing persistent
homology directly into an equivariant GNN's own message-passing (rather
than as a separate side channel), training a dedicated 3D localization
head evaluated by distance-to-center success rather than only residue
classification, and doing so for cryptic pockets from a single apo
structure with no simulation required. The broader binding-site-prediction
landscape is larger still — DeepPocket, PUResNet, IF-SitePred, and GrASP
are all actively benchmarked cavity-detection tools (Aggarwal et al.,
*J. Cheminformatics* 2024, compares several of them on cryptic sites) —
but that comparison evaluates pocket-level recall after re-scoring
geometric cavity detections, a different task and metric from the
residue-level classification used throughout this paper, so we cite it
as context rather than force an unsound numeric comparison. We did not
conduct an exhaustive systematic literature review; a closer match to
this specific combination may exist that this project's search did not
surface.

## Methods

**Architecture.** Protein chains are represented as residue-level graphs,
one node per residue (Cα-centered), with edges formed by a radius cutoff
(≈8–10 Å). Node features begin as a 21-dimensional one-hot residue identity,
6 backbone dihedral values, and are progressively enriched over the
project's development with ESM2 protein-language-model embeddings (480,
1280, or 2560 dimensions for the 35M/650M/3B-parameter ESM2 variants
respectively) and a persistent-homology feature block (34 dimensions after
filtering near-constant columns). These are propagated through stacked
E(n)-equivariant graph convolution layers (EGCLs), which update both
per-node scalar features and 3D coordinates while preserving equivariance
to rotation and translation. Topological features are fused into the node
representation at every EGCL layer through a dedicated gated fusion module
(`Fus`: a linear projection of the topology features gated by a sigmoid
computed jointly from the current node representation and the topology
features, added back as a residual) before the task heads.
Two heads sit on top of the shared trunk: a per-residue binary
classification head (is this residue part of a binding pocket) and a
localization head that produces per-residue scores, pooled with a
batch-aware softmax (summing to 1 per graph) into attention weights that are
used to compute a weighted centroid over the equivariant node coordinates —
i.e., the predicted pocket center is a soft, differentiable, geometrically
meaningful combination of node positions rather than a separately regressed
point. Equivariance of the full pipeline was verified numerically (rotating
input structures and checking that predicted centers rotate correspondingly)
with typical maximum errors on the order of 1e-5 Å, consistent with the
persistent-homology features' own measured SE(3)-invariance (typically
1e-6–1e-5 Å across multiple independent checks at different data scales).
Centering coordinates before any pairwise-distance computation proved
essential: crystal structures are stored 50–100+ Å from the coordinate
origin, and float32 distance computations at that offset lose enough
precision that edges near the cutoff threshold can flip in and out under
rotation, silently breaking equivariance (measured: 9/40 rigid motions broke
the edge set without centering, 0/40 with it).

**Topological features.** Alpha-complex persistent homology (via the GUDHI
library) was computed over 12 Å all-heavy-atom neighborhoods around each
residue, producing H0/H1/H2 persistence statistics (mean/summary birth and
death radii, entropy, and related descriptors) plus a burial-related
feature (local neighborhood radius of gyration). GUDHI's alpha-complex
filtration values are returned squared, and must be square-rooted before
use on a physically meaningful distance scale — an easy source of
mis-calibrated Betti-radius sampling if missed. A grouped, cluster-level
control analysis confirmed that the topology signal is not merely a proxy
for burial/density: after residualizing out a non-linear burial model, the
strongest single topological feature retained roughly 73% of its raw
discriminative signal. That control residualized only against a burial
model; a later and stronger control, retraining the models with 17 plain
geometry features in place of, and alongside, the topology block, is
reported under "Geometry control" in Results and changes the
interpretation of the topology gain.

**Datasets.** The main pipeline was trained on a leakage-clean pool of PDB
chains built from BioLiP-style ligand-proximity labels (4.5 Å heavy-atom
contact), clustered at 30% sequence identity via MMseqs2
(`easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 1`; verified to agree with
an exact Biopython all-vs-all identity calculation on 98.8% of pairs). This
training pool was scaled across the project from an original 2,056 clusters
up to 3,577, 5,339, and finally 7,513 clusters (3.65× the original size),
each round built as a strict superset of the previous one. Evaluation used
the standard COACH420 (420 structures) and HOLO4K (approximately 1,000–1,700
chains depending on the sampled benchmark round) pocket-detection
benchmarks, with training-side chains excluded whenever they were homologous
(≥30% identity) to the benchmark structures being scored — an initial
uncontrolled comparison had been invalidated when 74% of COACH420+HOLO4K
chains turned out to be homologous to the original training pool. The
cryptic-pocket extension used **CryptoBench** (Skrháková et al., 2025): 1,107
apo structures with 1,361 labeled cryptic pockets and an author-provided
10%-identity-clustered 80:20 train/test split, of which 942 chains survived
this project's own structure-fetching and labeling filters (700 train / 60
validation / 182 test).

**Training protocol.** All decision thresholds used for MCC and F1 were
tuned exclusively on the validation split and then applied, frozen, to the
test split — never tuned on test itself (an early same-run CryptoBench
ensemble check that had inadvertently tuned on test was caught and redone
properly). Because early single-seed and small-seed-count results in this
project repeatedly looked like real findings and then reversed or shrank to
noise once matched against a properly sized comparison arm — most sharply,
a 3-seed COACH420 result that appeared to show a substantial gain from a
larger protein-language-model embedding, which fully reverted to a
statistical tie once extended to 8 seeds — this project adopted a standing
discipline of comparing arms only at matched seed counts (typically 8
independently initialized training runs per arm) before treating any gap as
a real effect, and generally required an effect size of at least roughly
1.5–3× the observed seed-to-seed standard deviation before calling a result
significant.

**Transfer learning.** For CryptoBench, the main pipeline's COACH420-trained
checkpoints (same architecture, same ESM2 scale) were used as initialization
before fine-tuning fully on CryptoBench's own data and loss. Weight-tensor
shapes were directly inspected to confirm dimensional and architectural
identity across the two pipelines before transfer was attempted (e.g. a
541-dimensional input layer at the 35M ESM2 scale, and 2,621 dimensions at
the 3B scale, matched exactly on both sides).

**Cross-scale ensembling.** Independently trained models at each of the
three ESM2 scales (35M, 650M, 3B parameters) were combined by averaging
predicted probabilities, both with equal weighting and with weights chosen
by a grid search over the weighting simplex (0.1 increments across the
three scales), with the weights selected to maximize either validation AUC
or validation MCC and then applied, frozen, to the test set.

**Practical note.** Embedding a full training pool with ESM2 on a GPU was
roughly 200× faster than on CPU for the largest embedding model tested
(minutes versus roughly 16 hours for a comparable chunk of data); GPU
embedding was used wherever GPU quota allowed, and all reported metrics are
identical in validity regardless of which hardware produced the underlying
embeddings, since the numerical embeddings themselves do not depend on the
compute device.

## Results

### Main pipeline: classification (COACH420/HOLO4K training pool, held-out CV)

The core classification claim — that persistent homology features improve
binding-site prediction beyond sequence and burial features alone — was
tested with 10-fold cross-validation × 2 seeds × 3 feature arms (60 runs) on
a 2,056-cluster training pool. Results (test PR-AUC, mean ± std):

| Arm | Test PR-AUC |
|---|---|
| Sequence only | 0.2889 ± 0.0196 |
| Burial only | 0.3613 ± 0.0220 |
| Full (sequence + burial + topology) | 0.4036 ± 0.0230 |

Topology's contribution beyond burial alone was **+0.0423 PR-AUC**, positive
in 10/10 folds, paired t-test p < 1e-5, 95% CI [+0.0377, +0.0469], an effect
roughly 9× the seed-to-seed noise floor (0.0047). After ESM2-35M embeddings
and real backbone dihedrals were added to the sequence block, this
comparison was repeated under the project's full nested-CV discipline
(10-fold × 2 seeds × 3 arms, with a live P2Rank run matched per fold): full
0.4949 ± 0.0170 vs no-topology 0.4405 ± ... — topology's contribution
**survived at +0.0320 PR-AUC, 10/10 folds, p = 1.6e-5**, smaller than the
pre-ESM2 gain (expected, since the language-model embedding absorbs some of
the same signal) but still highly significant. In the same nested comparison
the full model also beat a matched, per-fold live run of P2Rank by +0.0524
PR-AUC (9/10 folds).

### Geometry control: how much of the topology gain is ordinary local geometry?

The comparisons above measure topology against sequence and burial features
in which "burial" is only two summaries (neighborhood atom count and radius
of gyration). A structural biologist would also compute solvent
accessibility, residue depth, exposure, and local shape without any
topology. We built a 17-feature per-residue geometry set (heavy-atom counts
within 4, 6, 8, 10, 12 and 15 Å; relative solvent-accessible surface area;
half-sphere exposure up and down; linearity, planarity and sphericity of the
12 Å atom cloud; offset of the Cα from that cloud's centroid; the fraction
of 100 rays from the Cα blocked within 10 Å and within 20 Å, expressed in a
canonical local frame so the feature is rotation-invariant; convex-hull
depth; and residue depth) and re-ran the Phase 11 nested cross-validation:
10 folds × 2 seeds, the same 2,055-chain homology-clean pool (one of the
2,056 chains was dropped because its re-fetched structure could not be
aligned to its graph), the same ESM2-35M plus dihedral inputs, model, loss
and early-stopping protocol. Every arm used an identical 51-wide side
channel (34 topology plus 17 geometry columns, unselected columns zeroed),
so architecture and parameter count are equal across arms. The rerun
reproduces the earlier result (no-topology 0.4404 vs. 0.4405 and full
topology 0.4947 vs. 0.4949 test PR-AUC). The primary comparison, geometry
plus topology against geometry alone, was fixed before the run.

| Arm | Test PR-AUC | Test ROC-AUC |
|---|---|---|
| No topology, no geometry | 0.4404 ± 0.0270 | 0.8478 ± 0.0162 |
| Topology (34 features) | 0.4947 ± 0.0201 | 0.8733 ± 0.0096 |
| Geometry (17 features) | 0.5076 ± 0.0225 | 0.8797 ± 0.0111 |
| Geometry + topology | 0.5173 ± 0.0196 | 0.8812 ± 0.0103 |

(mean ± std over the 10 folds; the 2 seeds are averaged within fold, and all
comparisons below are paired over the 10 independent folds.)

- **Geometry alone beats topology alone.** Geometry minus topology:
  +0.0129 PR-AUC (95% CI +0.0049 to +0.0209; geometry better in 10/10
  folds; paired t p=0.005, Wilcoxon p=0.002) and +0.0064 ROC-AUC
  (p=0.0015 / 0.0039).
- **Both add to the no-feature baseline.** Geometry minus none: +0.0672
  PR-AUC (10/10 folds, p=2.5e-8); topology minus none: +0.0543 (10/10 folds,
  p=2.2e-6).
- **Topology adds a small residual on top of geometry (primary
  comparison).** Geometry + topology minus geometry: +0.0097 PR-AUC (95% CI
  +0.0022 to +0.0171; better in 9/10 folds; paired t p=0.017, Wilcoxon
  p=0.037), but only +0.0015 ROC-AUC (CI -0.0013 to +0.0043; p=0.25 / 0.16,
  not significant). This meets this project's two-test rule for PR-AUC but
  not for ROC-AUC, rests on 10 folds, and is about a third of the +0.032
  originally attributed to topology beyond burial. Geometry adds far more to
  topology (+0.0226 PR-AUC, 10/10 folds, p=4.8e-5) than topology adds to
  geometry.

**Does the residual survive larger embeddings? (Phases 238–241.)** The
2,055-chain pool was re-embedded at ESM2-650M and ESM2-3B — holding
training-pool size fixed and varying only embedding scale, the same
design CryptoBench used to isolate its own embedding-scale lever — and
the full 10-fold × 2-seed × 4-arm control was rerun at each scale.

| Scale | Arm | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| 650M | no-topology | 0.4796 ± 0.0169 | 0.8687 ± 0.0118 |
| 650M | topology | 0.5192 ± 0.0201 | 0.8856 ± 0.0105 |
| 650M | geometry | 0.5242 ± 0.0189 | 0.8854 ± 0.0108 |
| 650M | geometry + topology | 0.5332 ± 0.0202 | 0.8886 ± 0.0120 |
| 3B | no-topology | 0.4754 ± 0.0180 | 0.8630 ± 0.0129 |
| 3B | topology | 0.5134 ± 0.0192 | 0.8817 ± 0.0101 |
| 3B | geometry | 0.5217 ± 0.0170 | 0.8844 ± 0.0107 |
| 3B | geometry + topology | 0.5332 ± 0.0194 | 0.8884 ± 0.0098 |

The primary comparison — geometry plus topology minus geometry alone — is
**confirmed by both tests at both larger scales**, and grows rather than
shrinks: 650M gives +0.0091 PR-AUC (95% CI +0.0038 to +0.0144; 8/10
folds; paired t p=0.0037, Wilcoxon p=0.0098) and +0.0032 ROC-AUC (9/10
folds; p=0.015/0.037); 3B gives +0.0115 PR-AUC (95% CI +0.0072 to
+0.0158; 10/10 folds; p=1.9e-4/0.0020) and +0.0040 ROC-AUC (9/10 folds;
p=0.0026/0.014). Both are stronger than the 35M result above, where only
PR-AUC cleared this project's two-test bar. Geometry continues to match
or beat topology alone at every scale (650M: geometry ahead by 0.0050
PR-AUC, p=0.0049/0.0059, tied on ROC-AUC; 3B: geometry ahead by 0.0082
PR-AUC, p=0.047/0.064, borderline). Absolute performance also rises with
scale (test PR-AUC for geometry+topology: 0.517 at 35M, 0.533 at both
650M and 3B), but the *residual specifically attributable to topology
beyond geometry* rises with it rather than being absorbed by the richer
embedding — the opposite of what we expected going in, since a language
model with more capacity might have been expected to make hand-crafted
topological features redundant. We do not have a mechanistic account of
why; it is reported as a robust, scale-confirmed empirical finding, not
an explained one.

The same control on CryptoBench (run only at the 35M scale) found no benefit from topology at all. On
the official split (8 seeds, benchmark labels), geometry plus persistence
features minus geometry alone was -0.011 validation PR-AUC (paired t
p=0.070, Wilcoxon p=0.109), +0.003 test PR-AUC (p=0.42 / 0.46), and geometry
alone beat topology alone on test PR-AUC (0.283 vs. 0.261, 8/8 seeds,
p=5e-4). On the 185-protein experimental-site holdout (an embedding-cluster
hard split scored against ligand-contact residues), geometry plus
persistence minus geometry was +0.0008 protein-level PR-AUC (95% bootstrap
CI -0.005 to +0.007, p=0.79). Geometry alone gave PR-AUC 0.352 against
0.345 for topology and 0.301 for neither, and modestly outperformed a
pretrained P2Rank 2.5.1 (ROC-AUC 0.796 vs. 0.756, p=1e-4). A plausible but
untested reason the main pipeline retains a small topology residual while
CryptoBench does not is the roughly three-fold larger training pool
(2,055 vs. 700 chains); the two benchmarks also differ in label definition.
Scope of this control: the classification arm and the localization arm
(below) were each run at all three main-pipeline embedding scales
(35M/650M/3B) on the same 2,055-chain pool; the CryptoBench arm was run
only at 35M, so whether its null would hold at larger scale is untested.

### Ablations: what helped and what did not

A rigorous accounting of this project's ablations, including several
explicitly negative or null findings:

- **GNN capacity/depth (positive-turned-null).** A single-seed sweep over
  width/depth/dropout at both the original (2,056-chain) and later scaled-up
  (7,181-chain) training pools initially suggested a deeper network
  (h=128, L=6) beat the default (h=128, L=4) by roughly 0.014–0.038 in
  validation PR-AUC/DCC. A matched 3-vs-3-seed comparison reversed this
  entirely: mean validation PR-AUC 0.5460 (default) vs 0.5471 (deeper), a
  gap of 0.0011, fully within noise. **Conclusion: GNN capacity does not
  help, at either data scale — a genuine null result on both attempts.**
- **Fusion-gate interpretability (negative).** Mean learned gate values sit
  at 0.4479 (binding-site residues) vs 0.4695 (non-site residues) — close to
  the sigmoid's 0.5 initialization, and in the wrong direction for a
  "topology is selectively routed to sites" story. The classification gain
  from adding topology is real, but the specific mechanistic claim that the
  gating layer learns to discriminate where topology should be trusted is
  not supported and was retracted from this project's claims.
- **Catalytic-residue transfer (negative, not harmful).** Testing whether
  the topology gain transfers to *catalytic* residues (Catalytic Site Atlas
  labels, a much sparser and more specific class than general binding-site
  residues) gave full − burial-only = **−0.0152** (1/5 folds positive,
  p=0.14) — no benefit, though also not significantly harmful. A coherent
  interpretation: persistent homology finds cavities, but a catalytic triad
  is a chemical detail *inside* a cavity, not itself a cavity-shape signal.
- **Topology-dimension decomposition (partially resolved after extensive
  effort).** Which specific homology dimension (H0/H1/H2) or the burial
  feature drives the localization gain was tested at the original data
  scale and found **fully diffuse** — no pairwise comparison among H0/H1/H2/
  burial cleared even a 1× noise-floor threshold. Repeated at the largest
  (7,513-cluster) training scale with all six arms matched at a true 8 seeds
  each, one, and only one, pairwise comparison became significant: **H0
  (connected components) is measurably and reproducibly the weakest
  contributor** (H0 vs H1: 1.82× the noise floor); H1, H2, and burial remain
  mutually indistinguishable from one another. The honest claim is "H0
  underperforms the rest of the topology block," not "one dimension
  explains the whole benefit."
- **Cross-chain arbitration heuristics for multi-chain structures
  (negative).** Six alternative rules for selecting which chain represents a
  multi-chain HOLO4K structure were tested on a held-out tune/test split;
  the apparent winner on the tuning split (+0.018 over the existing rule)
  reversed to a small loss (−0.0016) on held-out test — a clean demonstration
  of the tune/test discipline catching its own false lead. The existing
  rule (summed per-residue probability) was kept.
- **Embedding-capacity lever on the main pipeline (negative — differs from
  CryptoBench).** Moving from ESM2-35M to ESM2-3B on the main pipeline gave
  an exciting 3-seed COACH420 result (DCC 0.6053) that fully reverted at 8
  seeds (DCC 0.5908, statistically tied with the 35M 8-seed baseline of
  0.5932). HOLO4K showed the same pattern at only 3 seeds (0.6443 vs the
  35M 8-seed baseline of 0.6453). **Conclusion: larger protein-language-model
  embeddings do not improve the main COACH420/HOLO4K pipeline**, even though
  the identical lever *did* produce large, validated gains on CryptoBench
  (see below) — the benefit of embedding scale is task- and dataset-
  dependent, not universal.
- **Auxiliary "any-pocket" (cryptic + non-cryptic) multi-task head on
  CryptoBench (negative).** Adding a second BCE loss term for a denser,
  easier "is this pocket-like at all" label produced a clean, monotonic
  *decline* in validation PR-AUC as its loss weight increased (0.3592 →
  0.3557 → 0.3476). A follow-up variant that masked these residues out of
  the main loss instead of treating them as a competing target gave a
  marginal, likely noise-level improvement (+0.0023). Neither formulation is
  adopted.
- **B-factor flexibility feature on CryptoBench (ambiguous, not adopted).**
  A single per-residue crystallographic B-factor (z-scored within-chain)
  gave a small, single-seed gain of +0.0052 validation PR-AUC — smaller than
  this project's typical single-seed noise band, and not extended to
  multi-seed confirmation given the modest effect size.

### Main pipeline: pocket-center localization

An auxiliary localization loss (SmoothL1 against the true ligand center of
mass, weight 0.1, chosen by a validation-only sweep over {0.02, 0.1, 0.5})
gave the model its own trained centroid predictor rather than relying on a
post-hoc clustering heuristic over classification probabilities. On the
project's own held-out split this improved DCC (distance-to-center success
at 4 Å) by **+0.0478 (+14.5% relative)** over the best heuristic applied to
the classification-only model, at essentially no PR-AUC cost (−0.0026,
noise-level). After fixing a coordinate-frame bug (predicted centers were
being compared in a re-centered frame against ground truth in the raw
crystal frame), the corrected, externally validated result on the full
COACH420/HOLO4K benchmark was pooled DCC 0.5298 (COACH420 0.5115, n=174;
HOLO4K 0.5331, n=968), with topology contributing **+0.0289** pooled DCC
over a no-topology control (COACH420 +0.0920, HOLO4K +0.0176) — later
confirmed with strong multi-seed evidence at the original 2,056-chain
training scale (8/8 seed-paired wins, mean delta +0.0528, 3.37× the
single-seed noise floor). This was, to our knowledge, the first
demonstration that persistent homology improves *where* a trained
equivariant model believes a pocket center is, not only *whether* a
residue is classified as part of one.

**Geometry control for localization (Phases 236–237).** Retraining the
same 2,055-chain pool and scoring on the same 1,149-unit COACH420/HOLO4K
benchmark with the 17-feature geometry block used for classification
(identical 51-wide side channel across arms, 8 seeds each), geometry again
outperforms topology alone: geometry minus topology is **+0.0305 pooled
DCC@4Å** (7/8 seeds, paired t p=0.0075, Wilcoxon p=0.016), and geometry
minus no-topology is +0.0855 (8/8 seeds, p=2.2e-5/0.0078) — both larger
than topology's own +0.0549-over-no-topology gain in this rerun (8/8
seeds, p=1.6e-4/0.0078, closely replicating the +0.0528 figure above). The
primary, pre-specified comparison — geometry plus topology minus geometry
alone — is **null per seed**: +0.0051, positive in only 3/8 seeds (paired t
p=0.62, Wilcoxon p=0.92). A follow-up analysis (Phase 237) reloaded the
same 32 checkpoints, with no retraining, and asked whether this reflects a
true absence of benefit or a small effect swamped by single-seed noise: as
ensemble size grows from 1 to 8 seeds, the geometry+topology-minus-geometry
delta rises monotonically from +0.0051 to +0.0254 while its spread across
seed subsets shrinks (positive in 3/8 single-seed pairs, 55/56 three-seed
subsets, and 69/70 disjoint 4-versus-4-seed ensemble splits) — the
signature of a real small effect obscured by seed noise, not a clean null.
A stricter two-way bootstrap that resamples both seeds and benchmark
structures together puts the 8-seed-ensemble delta at +0.0188 (95% CI
-0.0096 to +0.0456, P(Δ>0)=0.90): directionally consistent with a small
residual benefit but not clearing this project's two-test confirmation bar.
**Localization at larger embedding scales (Phases 244–247).** Because
classification's residual strengthened with embedding scale (above), the
same localization control was repeated at ESM2-650M and ESM2-3B: the
COACH420/HOLO4K benchmark cache was re-embedded at each scale by decoding
each chain's sequence losslessly from the existing cached one-hot block
(no structure was re-fetched), rebuilt and checked column-for-column
(1,341 and 2,621 features per residue), and 4 arms × 8 seeds were retrained
on the corresponding 2,055-chain training blob (Phases 238/239).

| Scale | No topology | Topology | Geometry | Geometry + topology |
|---|---|---|---|---|
| 35M | 0.4441 | 0.4990 | 0.5296 | 0.5347 |
| 650M | 0.4790 | 0.5016 | 0.5385 | 0.5591 |
| 3B | 0.4709 | 0.5164 | 0.5418 | 0.5412 |

(pooled DCC@4Å, mean of 8 single-seed models; seed-to-seed standard
deviation is 0.013–0.025.) The picture does **not** repeat
classification's:

- **Geometry still matches or beats topology alone at every scale:**
  +0.0305 (35M, 7/8 seeds), +0.0369 (650M, 8/8 seeds; t p=0.0018,
  Wilcoxon p=0.0078), +0.0254 (3B, 5/8 seeds; p=0.064/0.11 — the weakest of
  the three, not significant on its own).
- **Topology and geometry each still beat having neither**, replicating the
  original +0.0528 finding in direction at every scale (topology minus
  none: +0.0549, +0.0227, +0.0455; geometry minus none: +0.0855, +0.0595,
  +0.0709), although topology's own gain is notably smaller at 650M.
- **The pre-specified comparison — geometry plus topology minus geometry
  alone — is not confirmed at any scale and does not trend with scale:**
  +0.0051 (35M; 3/8 seeds; p=0.62/0.92), +0.0206 (650M; 6/8 seeds; paired
  t p=0.037 but Wilcoxon p=0.078, so it fails this project's two-test bar),
  and -0.0007 (3B; 4/8 seeds; p=0.94/1.0). Pooling all 24 seed-level
  differences across the three scales gives +0.0084 (95% CI -0.0027 to
  +0.0194; positive in 13/24; t p=0.13, Wilcoxon p=0.18), with no
  relationship between the difference and embedding scale (Spearman
  ρ=-0.16, p=0.45). The 8-seed-ensemble differences (+0.0254, +0.0053,
  +0.0184) are equally inconsistent in size across scales. At 3B the
  benchmarks also disagree in sign (COACH420: geometry 0.530 vs.
  geometry+topology 0.489; HOLO4K: 0.544 vs. 0.551).

**Localization with 3.5x more training data (Phases 250-254).** The
original localization result (Phase 22, the paper's headline) used the
2,056-chain pool, and the topology gain was largest on the round-3 pool
(7,513 chains; 7,181 after removing every chain homologous to COACH420).
The control was therefore repeated there: the same 17 geometry features
for 7,510 of the 7,513 chains (three failed a residue-count check), the
same COACH420-clean training list as the original round-3 runs (recovered
exactly: 332 excluded, 7,181 kept, 7,178 with a recoverable ligand
center), 4 arms x 8 seeds trained on 6,604 chains (574 held out for early
stopping), and scoring on the full COACH420 benchmark (420 structures, 413
with a ligand). Pooled DCC@4Å:

| Arm | Mean of 8 single models | 8-model ensemble |
|---|---|---|
| No topology | 0.4401 | 0.5012 |
| Topology | 0.5342 | 0.5860 |
| Geometry | 0.5681 | 0.6199 |
| Geometry + topology | 0.5439 | 0.5884 |

- **Topology adds nothing over geometry here, and the sign is negative.**
  Geometry plus topology minus geometry alone was -0.0242 per seed (geometry
  better in 7 of 8 seeds; paired t p=0.086, Wilcoxon p=0.055), -0.0315 for
  the ensemble (bootstrap over structures -0.063 to 0.000), and -0.0217 in
  a two-way bootstrap over seeds and structures together (95% CI -0.068 to
  +0.024, P(difference>0)=0.17). Only 7 of 70 disjoint 4-versus-4-seed
  ensemble splits favored adding topology, and the difference was negative
  at every ensemble size, the opposite of the weakly positive ensemble
  trend seen at 35M.
- **Geometry beats topology alone:** +0.0339 per seed (6/8 seeds; t p=0.048,
  Wilcoxon p=0.047, both barely under 0.05).
- **Both still beat having neither:** topology minus none +0.0941 (8/8
  seeds; p=0.0008/0.0078) and geometry minus none +0.1280 (8/8; p=9e-6/
  0.0078). The original claim that topology helps localization therefore
  replicates in direction, but geometry helps more.
- **One discrepancy is unexplained.** The no-topology arm here reaches an
  ensemble DCC of 0.501, well above the 0.429 reported for the same arm in
  the earlier round-3 decomposition (Phase 86), while the topology arm
  reproduces (0.586 vs. 0.593). Topology's gain over no topology is thus
  about +0.085 here rather than the earlier +0.165. The earlier no-topology
  arm was assembled from salvaged partial runs and may have differed in
  setup; this could not be resolved, so the earlier larger figure should
  not be relied on.

The best localization model on this benchmark is therefore geometry alone
(ensemble DCC 0.620, against 0.521 for a same-run P2Rank in the earlier
round-3 evaluation; the two use slightly different structure counts, 413
with a ligand here versus 420, so the margin should be treated as
approximate).

**Honest reading.** Topology's most defensible, consistently reproduced
contribution across every setting tested is beating *no* structural
features at all. Beating a strong geometric baseline is a harder bar,
and the two tasks differ: it is clearly met by main-pipeline
classification at 650M and 3B embedding scale (confirmed on both PR-AUC
and ROC-AUC), met only partially at 35M classification (PR-AUC only),
**not established for localization at any scale or training-set size**
(inconsistent at 35M/650M/3B on the 2,055-chain pool, pooling to +0.008
with an interval including zero, and negative at 7,513 chains), and not
met at all on CryptoBench classification (35M only). The classification
residual firms up with embedding scale; the localization residual does
not, so the two should not be described as one consistent topology effect.

**Scaling the training pool** turned out to be the largest single lever for
localization accuracy on the standard benchmarks. Across three
progressively larger, strictly nested training pools (2,056 → 3,577 → 5,339
→ 7,513 clusters), 8-seed-validated COACH420 DCC rose from 0.5161 to 0.5642
to 0.5956, then plateaued at 0.5932 at the largest scale — a real, large,
now-saturated win over a live, current run of P2Rank (this-run P2Rank DCC
0.5209, a gap of **+0.0716**). HOLO4K told a more nuanced story: the same
scale-up moved its gap to the live P2Rank tool from a clear loss (−0.0668
DCC at the smallest scale) to a narrow, non-significant deficit at the
largest scale (−0.0127, 0.6× the noise floor) — an intermediate round had
shown an apparent win (+0.0124) that did not survive the next round of
scale-up, so **the corrected, final conclusion for HOLO4K is approximate
statistical parity with live P2Rank, not a stable win**, and we report it
that way rather than citing the more flattering intermediate number. On
DCA, the same largest-scale 8-seed ensemble reaches 0.8236 (n=998),
beating published fpocket (0.457), P2Rank (0.621), and EquiPocket
(0.662) — the DCC parity with live P2Rank does not extend to a parity
on DCA, where our model stays clearly ahead of every published baseline.

Comparison against literature baselines on the full, unfiltered
benchmark (COACH420, n=413 evaluable): our model (largest-scale, 8-seed) DCC
0.5932 vs published fpocket 0.228, published P2Rank (2018) 0.366, published
EquiPocket (2023) 0.423 — beating all three. Against the same-run live
current P2Rank tool (v2.4.2, substantially improved since its 2018
publication), the comparison is +0.0716 DCC, a genuine, contemporaneous win,
distinct from (and a stronger claim than) simply beating a seven-year-old
published number. On DCA (distance to the nearest ligand atom, the looser
companion metric), the same largest-scale 8-seed ensemble reaches 0.8087,
beating published fpocket (0.444), P2Rank (0.628), and EquiPocket (0.656)
by a wider margin than on DCC.

### CryptoBench: cryptic-pocket classification and localization

The first trained model combining persistent homology, ESM2 embeddings, and
a trained equivariant localization head for cryptic pockets, evaluated on
CryptoBench's official 182-chain test fold (54,992 residues, 4.72%
cryptic-pocket positive rate):

| Model | Seeds | AUC | AUPRC | MCC | F1 | DCC@4Å |
|---|---|---|---|---|---|---|
| PocketMiner (literature) | — | 0.76 | 0.19 | 0.22 | 0.78* | — |
| P2Rank-apo (literature) | — | 0.81 | 0.21 | — | 0.81* | — |
| CryptoBench SOTA (ESM2-3B + shallow NN, literature) | — | 0.87 | 0.40 | 0.42 | — | — |
| UniPocket (2026, literature)† | — | 0.816 | 0.410 | — | — | — |
| **Ours, 35M-scratch** | 3 | 0.8665 | 0.3245 | 0.3620 | 0.3933 | 0.2088 |
| **Ours, 650M-scratch** | 8 | 0.8873 | 0.3627 | 0.3986 | 0.4264 | 0.2088 |
| **Ours, 3B-scratch** | 3 | 0.8869 | 0.3695 | 0.4144 | 0.4389 | 0.2363 |
| **Ours, 35M-transfer** | 8 | 0.8971 | 0.4019 | 0.4272 | 0.4514 | 0.3077 |
| **Ours, 650M-transfer** | 8 | 0.9141 | 0.4270 | 0.4473 | 0.4741 | 0.3407 |
| **Ours, 3B-transfer** | 8 | 0.9111 | 0.4307 | **0.4664** | 0.4896 | 0.3077 |
| Ensemble (all-transfer, equal weight) | 8 each | 0.9173 | 0.4461 | 0.4611 | 0.4871 | — |
| Ensemble (all-transfer, grid search, AUC-optimal weights) | 8 each | 0.9177 | 0.4464 | 0.4651 | 0.4910 | — |
| Ensemble (all-transfer, grid search, MCC-optimal weights) | 8 each | **0.9182** | 0.4453 | 0.4553 | 0.4816 | — |
| **Ensemble (all-transfer, logistic-regression combiner)** | 8 each | 0.9177 | 0.4484 | **0.4699** | 0.4956 | — |

†UniPocket (Zhao, Zhai, and Shen, ACM-BCB 2026) is the most recent
published baseline we found on this exact benchmark: a multitask
ESM-2-embedding model trained jointly on ligand-binding and cryptic-pocket
labels, evaluated on 193 CryptoBench test proteins (vs. our 182 — a small
discrepancy we did not track down). Its own paper reports these metrics
**macro-averaged per protein and then averaged across proteins**, not
pooled across all test residues as every other row in this table is —
a real methodological difference, not just a presentation choice: macro-averaging
weights every protein equally regardless of size, while pooling weights every
residue equally, so the two conventions can legitimately disagree even for
identical underlying predictions. We report UniPocket's numbers as published
rather than re-deriving a pooled equivalent, and flag this rather than
implying a cleaner comparison than actually exists. CryptoBench's own
paper does not explicitly state which convention its own reported
baselines (PocketMiner, P2Rank-apo, its ESM2-3B model) use either, so the
same caveat applies there with lower confidence.

The two from-scratch baselines still reported at 3 seeds (35M and 3B)
were run early, before the 8-seed discipline was adopted for the
transfer-learning comparisons that are this project's main claims;
extending them to 8 seeds would need new training runs and was not done,
since they serve only as a reference point, not a claim this project
rests on. The 650M-scratch baseline happened to already exist at 8
seeds and is reported as such. The ensemble's per-seed prediction
pipeline does not track predicted pocket centers, so no DCC is
available for the three ensemble rows.

*A note on F1: this project's F1 numbers (0.39–0.49 depending on model) are
not directly comparable to the literature's reported F1 (0.78–0.81, marked
with an asterisk in the table) for PocketMiner and P2Rank-apo. Reading CryptoBench's own evaluation code showed
that those literature F1 values use scikit-learn's *class-weighted* F1
average, which at a ~95%-negative class balance produces a trivially high
number dominated by the majority (non-pocket) class; CryptoBench's own model
evaluation code for its SOTA baseline uses the same plain positive-class F1
convention this project uses, and never reports an F1 for its own SOTA
model at all. MCC — threshold-sensitive but class-balance-robust — is the
fair, apples-to-apples metric here, and is the one emphasized throughout.

**Individual levers**, each tested by holding architecture and training
data fixed and varying one factor, all run at 8 seeds (results ranged
from clearly confirmed to a clean null, reported honestly in each case):

- **Topology ablation:** full 0.3492 vs no-topology 0.3130 validation
  PR-AUC, paired by seed across the same 8 random initializations
  (mean difference +0.0363). A paired t-test rejects the null of no
  difference (t=6.11, **p=0.00049**), confirmed by the non-parametric
  Wilcoxon signed-rank test (**p=0.0078**, appropriate given the small
  n=8 and no assumption of normality); Cohen's d=2.16 (a very large
  effect by conventional standards) — the single strongest and most
  confidently confirmed lever tested on CryptoBench, and notably
  *larger* in relative terms than the equivalent gain on the main
  pipeline. **Caveat added after a geometry control:** this comparison
  used a baseline with no burial or exposure features. Replacing or
  supplementing the topology block with 17 plain geometry features gave
  an equal or larger gain, and topology added nothing detectable on top
  of geometry (see "Geometry control" above); the topology lever is
  therefore confirmed only relative to a weak baseline.
- **Transfer learning from the main pipeline:** scratch 0.3492 vs
  transfer-initialized 0.3710 validation PR-AUC, paired by seed (mean
  difference +0.0218). Paired t-test t=3.97, **p=0.0054**; Wilcoxon
  **p=0.0156**; Cohen's d=1.40 (large effect) — confirmed consistent
  across five distinct pretrained-checkpoint seeds, not dependent on
  which specific main-pipeline run supplied the initialization.
- **GNN depth, tested specifically on CryptoBench** (h128L6, +2 EGCL/Fus
  layers, vs. the default h128L4, trained from scratch at the 650M
  scale to keep this a clean architecture comparison): mean validation
  PR-AUC rose from 0.3674 to 0.3762 (+0.0088), and test-set metrics
  moved the same direction (AUC 0.8878 vs. 0.8873, MCC 0.4041 vs.
  0.3986). Unlike the two levers above, this result is **borderline**:
  a paired t-test narrowly crosses the conventional significance
  threshold (t=2.43, p=0.046), but the more appropriate test for n=8
  without a normality guarantee, the Wilcoxon signed-rank test, does
  not (p=0.055); Cohen's d=0.857 (medium-large, but with inconsistent
  significance across tests). We report this as suggestive rather than
  established — a real contrast worth noting is that this same
  architectural change was tested and returned a clean null result on
  the (saturated) main pipeline, so a modest positive signal appearing
  here, even a borderline one, is consistent with the hypothesis that
  CryptoBench's smaller, harder task retains some capacity headroom the
  main pipeline no longer has.
- **Loss function** (focal loss, alpha=0.9/gamma=2.0, replacing
  BCE+pos_weight, otherwise identical protocol including the same 8
  pretrained 3B main-pipeline checkpoints): a clear **negative** result.
  Mean validation PR-AUC fell from 0.4023 (BCE) to 0.3835 (focal), a
  loss on all 8 of 8 paired seeds (t=-6.51, **p=0.00033**; Wilcoxon
  **p=0.0078**; Cohen's d=-2.30, a very large effect in the unfavorable
  direction). BCE with a fixed positive-class weight already matched to
  the benchmark's known ~4.7% positive rate outperforms the additional
  gamma-modulated down-weighting focal loss adds, at least at this
  untuned alpha/gamma setting; we did not further tune these
  hyperparameters given the size and consistency of the gap.
- **Topological variability across an ANM-perturbed conformational
  ensemble** — a new feature type, not a tuning variant of the existing
  recipe. Persistent homology throughout this project is computed on a
  single static structure; cryptic pockets are by definition cavities
  absent from that static structure, so we tested whether the local
  topology's *instability* across plausible nearby conformations — not
  just its static value — carries additional signal. For every
  residue, we built an Anisotropic Network Model (ANM, a Cα-level
  elastic network whose eigenvectors give directional, not merely
  isotropic, per-residue displacements) from the apo structure, generated
  8 perturbed conformations by randomly combining the 10 lowest-frequency
  modes scaled to a 1.2 Å target Cα RMSD, rigidly translated each
  residue's full atom set by its own Cα displacement, and recomputed the
  full persistent-homology feature vector on each perturbed conformation.
  The standard deviation of that vector across the 9-conformation
  ensemble (original plus 8 perturbations) became a new 34-dimensional
  feature block, doubling the topology-derived feature width (34 static
  + 34 variability) with no change to the model architecture. Trained
  from scratch (the wider topology block is shape-incompatible with
  existing pretrained checkpoints, so transfer learning was unavailable
  here, matching how the original topology ablation above was also
  trained from scratch), both arms paired by seed: full (68 columns)
  reached mean validation PR-AUC 0.3459 (std 0.0098) versus static-only
  (the new 34 columns zeroed) at 0.3564 (std 0.0122) — **the static-only
  arm won**, a clear **negative** result (mean difference −0.0105,
  t=−3.15, **p=0.0161**; Wilcoxon **p=0.0156**; Cohen's d=−1.115, a
  large effect in the unfavorable direction), losing on 7 of 8 paired
  seeds. We report this plainly rather than omit it: a well-motivated,
  correctly implemented (the ANM computation passed the same
  rotation-invariance check, at the same tolerance, as the untouched
  static topology code) new feature type does not always help, and an
  8-conformation ensemble sampled from an approximate elastic-network
  model is a plausible source of noise rather than signal on a
  700-chain training set — 34 additional, largely uninformative input
  dimensions is a real cost for a dataset this size. We did not tune the
  ensemble size, target RMSD, or mode count further given the
  consistency of the result across seeds.
- **ANM-predicted flexibility as a single feature** — a cheaper,
  distinct hypothesis from the variability test above: rather than the
  *instability* of local topology across a perturbed ensemble, does
  simply knowing which residues an elastic-network model predicts will
  move the most help, on its own? We computed one Anisotropic Network
  Model per apo structure (same Hessian construction as above, but here
  only the predicted mean-square-fluctuation magnitude is needed, not
  displacement directions, so this required a single eigendecomposition
  per chain — no perturbed conformations, no persistent homology
  recomputation, roughly seven times cheaper to compute than the
  variability block). Added as one new z-scored input column. Trained
  from scratch (again shape-incompatible with pretrained checkpoints),
  full (feature intact) vs the column zeroed, 8 seeds each, paired by
  seed: mean validation PR-AUC was 0.3539 for *both* arms to four
  decimal places (std 0.0064 vs 0.0055) — a clean **null** result, not
  a negative one: mean difference 0.0000, **p=0.995** (t-test),
  **p=0.945** (Wilcoxon), Cohen's d=0.002. The model appears to simply
  not use this feature in either direction. The most likely explanation
  is scale, not lack of signal: the raw predicted fluctuation values
  were extremely heavy-tailed (mean 2.6, but a maximum of 129,000 driven
  by near-zero eigenvalues on some chains), and a naive z-score
  normalization on a distribution that skewed compresses the
  informative middle of the range into near-identical values for most
  residues. We did not pursue a log-transformed or rank-based
  normalization to test that explanation, since this feature was
  explicitly scoped as the cheap, quick check in this round; it is
  recorded as an open question rather than a settled negative.
- **Cross-scale ensembling** (35M+650M+3B, from-scratch models): AUC rose
  from the best single scale's 0.8878 to 0.9000 (equal weighting) and 0.9157
  (validation-AUC-optimal weighting); MCC rose from 0.4157 (best single
  scale) to a best combination value of 0.4260–0.4669 depending on
  weighting scheme.
- **Top-K local-attention localization head** — an architecture change,
  not a feature or a hyperparameter. The existing localization head
  predicts the pocket center as a softmax-weighted average of predicted
  positions over *every* residue in the chain; for a 300+-residue
  protein where the true pocket is a small, spatially local subset,
  that global softmax can be diluted by the long tail of low-scoring
  residues, pulling the predicted center away from the true cluster.
  We masked the per-residue localization logit to the top-K (K=20)
  highest-scoring residues before the softmax, forcing a locally
  committed prediction. This changes zero parameter shapes (identical
  `loc_head`), so — unlike the topological-variability and
  ANM-flexibility experiments above — the existing pretrained 3B
  main-pipeline checkpoints loaded unchanged, and we fine-tuned the
  same 8 checkpoints Phase129/the transfer-learning lever used, 1:1,
  for a direct paired-by-seed comparison. On validation, classification
  (PR-AUC) rose from 0.4023 to 0.4070 (mean diff +0.0047): borderline,
  the same inconsistent-between-tests pattern as the GNN-depth result
  above (t-test **p=0.043**, Wilcoxon **p=0.055**, Cohen's d=0.875).
  Localization (DCC) — the metric this change was specifically designed
  to improve — moved in the right direction on validation (0.2667 vs.
  0.2542, diff +0.0125) but was **not significant** (t-test p=0.476,
  Wilcoxon p=0.625, Cohen's d=0.266, a small effect). On the held-out
  TEST set, classification improved modestly and consistently with the
  validation signal (AUC 0.9124 vs. 0.9111, MCC 0.4717 vs. 0.4664,
  AUPRC 0.4285, F1 0.4965), but DCC@4Å was **0.3077, identical to the
  baseline to four decimal places** — on the same 182 chains, 54,992
  residues, and 2,597 positive residues, so this is a genuine tie, not
  a rounding coincidence. We report this as a **null result on its own
  target metric**: a well-motivated architecture change that measurably
  did not move the localization accuracy it was built to improve,
  alongside a small, only borderline-significant classification
  side-effect. A plausible explanation is that K=20 was not tuned — for
  chains far longer or shorter than the ~20-residue cryptic pockets
  typical here, a fixed K may be too permissive or too restrictive —
  but we did not sweep K given the size of the signal already observed
  at K=20, and report this as an open question rather than pursue
  further untuned variants.

**Final combined result**, applying transfer learning at every scale where
a matching main-pipeline checkpoint existed and then cross-scale
ensembling the transfer-learned models: solo scores were 35M-transfer AUC
0.897/MCC 0.427, 650M-transfer AUC 0.914/MCC 0.447, and 3B-transfer AUC
0.911/MCC 0.466 (the best single-model MCC obtained anywhere in this
project). Four ways of combining the three transfer-learned scales were
tested: equal weighting, two grid-search weightings (66-point simplex
search over 0.1 increments, tuned on the 60-chain validation fold —
separately for whichever weighting maximized validation MCC vs.
validation AUC), and a logistic-regression meta-combiner (4 parameters:
one weight per scale plus an intercept, fit by maximum likelihood on the
same validation fold). The grid search's exhaustive maximization over 66
candidate points, using only 60 validation chains, has considerably more
effective flexibility than it looks and is correspondingly more prone to
selecting validation-set noise than a smooth, 4-parameter regularized
fit; the two methods' very different learned weightings on the same data
(grid search: 35M=0.2/650M=0.5/3B=0.3 by MCC, 35M=0.2/650M=0.3/3B=0.5 by
AUC; logistic regression: 35M=0.20/650M=0.34/3B=0.59, all three positive
and monotonic with scale) illustrate exactly this fragility, and we adopt
the logistic-regression combiner as the principled default.

To settle whether any of these combinations produce a genuine
improvement over the best single model, rather than continuing to
eyeball a "noise floor," we ran a chain-level bootstrap (2,000
resamples of the 182 test chains, resampling whole chains rather than
individual residues since residues within one chain are correlated) and
computed 95% confidence intervals for each method's metrics and for its
paired difference against 3B-transfer alone. **The AUC improvement is
real for every combination method tested**: the 95% CI of the AUC
difference excludes zero in all four cases (logistic-regression
combiner: 0.9177 vs 0.911, difference 95% CI [+0.0038, +0.0095]; grid
search AUC-optimal: [+0.0036, +0.0098]; grid search MCC-optimal:
[+0.0032, +0.0113]; equal weight: [+0.0022, +0.0103]). **The MCC
improvement is not**: the 95% CI of the MCC difference includes zero
for every one of the four methods, including the logistic-regression
combiner, which nonetheless reaches the highest point estimate of any
model in this project (MCC 0.4699, 95% CI [0.424, 0.513], difference
vs. 3B-transfer alone 95% CI [−0.0104, +0.0164]). This is a stronger,
more useful finding than simply confirming or overturning the earlier
informal "tie": a well-regularized combiner that is much less prone to
overfitting the validation fold shows the *same* non-significant MCC
gain as the naive grid search, which argues that the tie reflects a
genuine limit of this 182-chain, 4.72%-positive test set's statistical
power to distinguish MCC differences of this size, rather than an
artifact of a particular weight-search procedure. We report the AUC
gain as statistically established and the MCC gain as a real but
not-yet-significant point-estimate improvement, and recommend the
logistic-regression combiner (AUC 0.918, MCC 0.470) as the final
CryptoBench result, while noting that 3B-transfer alone (MCC 0.466,
no ensembling required) remains a legitimate, simpler choice given
the two are statistically indistinguishable on MCC.

Localization on CryptoBench — a task with no existing published comparison
we found (see Related Work: prior cryptic-pocket work either ranks
candidate sites from simulation or characterizes pockets some other tool
already located, rather than training a continuous 3D center predictor) —
reached a
DCC@4Å hit rate of 0.209–0.247 depending on embedding scale, with mean
localization error around 9.6–10.4 Å. This is substantially worse than the
50–65%-range hit rates on COACH420/HOLO4K's *open* pockets, which is the
expected and coherent outcome: an apo structure has no cavity to anchor on,
so the model must infer a pocket's future location from shape precursors
alone, a fundamentally harder geometric problem than localizing a pocket
that is already visibly present.

## Discussion

Adding persistent-homology features to sequence and burial features
produces a large, reproducible classification gain against the baselines
this project started with, across two structurally distinct benchmarks
(open-pocket COACH420/HOLO4K and cryptic-pocket CryptoBench), across
multiple training scales, and across a decisive nested cross-validation
design. The geometry control reported above shows that this gain is not
specific to topology. Seventeen conventional per-residue geometry features
give an equal or larger gain on both benchmarks (+0.0672 vs. +0.0543
PR-AUC on the main pipeline), and topology adds either nothing detectable on CryptoBench (+0.0008
PR-AUC, 95% CI -0.005 to +0.007, 35M only) or a real, scale-confirmed
residual on the main pipeline that *strengthens* with embedding scale
(+0.0097 PR-AUC at 35M, significant only on PR-AUC; +0.0091 at 650M and
+0.0115 at 3B, both confirmed on PR-AUC and ROC-AUC) beyond them. The
earlier reading, that persistent homology encodes cavity *shape* in a way
that raw coordinates and language-model embeddings do not, is therefore
only partly supported: most of the classification gain is ordinary
burial, exposure and local shape, however computed, and the
persistent-homology block is one, slightly less efficient, way of
supplying it — but a smaller residual contribution survives even a 3B-
parameter embedding, and grows rather than shrinks as the embedding gets
larger, which is not what we expected of a redundant feature. We cannot
yet attribute this residual to any particular topological property (H0
versus H1 versus H2), and its absence on the smaller, differently labeled
700-chain CryptoBench training set may reflect data volume, the label
definition, or the task itself.

Transfer learning from the larger, more diverse main-pipeline training pool
to CryptoBench's comparatively small 700-chain training set produced a
clear, multi-seed-confirmed gain, plausibly because the main pipeline's
much larger pool of general protein structures teaches more broadly useful
shared representations (residue embeddings, graph-convolution weights) that
CryptoBench's smaller, more specialized dataset alone cannot learn as
reliably.

Embedding scale (35M → 650M → 3B parameter ESM2 models) helped consistently
and monotonically on CryptoBench but showed no benefit at all, once properly
seed-validated, on the main COACH420/HOLO4K pipeline. The most likely
explanation is that the main pipeline had already saturated its available
training-data scale and GNN capacity, leaving no further headroom for a
richer embedding to exploit, while CryptoBench's smaller dataset and harder
task still had headroom that a more capable language model could fill —
this project's ablations found the main pipeline's three most obvious
remaining levers (more training data, larger embeddings, deeper/wider GNN)
all null or plateaued at the same time, suggesting whatever is capping its
performance now lies elsewhere (for instance, in the pocket-localization
heuristic/architecture itself, or in the benchmark's own label quality).

The honest limitation of the final CryptoBench ensemble result is
statistical power at the test-set level, and we tested this directly
rather than leaving it as a caveat. CryptoBench's official validation
split is only 60 chains; searching a weighting simplex with roughly 66
grid points over three models on that small a validation set has
considerably more effective flexibility than it looks, and is
correspondingly prone to selecting weights that fit validation noise
rather than a real cross-scale complementarity — the two grid-search
variants' quite different learned weightings (35M=0.2/650M=0.5/3B=0.3
by MCC vs. 35M=0.2/650M=0.3/3B=0.5 by AUC) is itself evidence of this.
Replacing the grid search with a regularized logistic-regression
combiner (4 parameters, fit by maximum likelihood rather than
exhaustive maximization) did not resolve the ambiguity: a chain-level
bootstrap (2,000 resamples of the 182 test chains) found the AUC gain
over the best single model statistically significant for every
combination method tested, including the regularized one, while the
MCC gain remained not statistically significant for every method,
including the regularized one. The consistency of this result across a
method explicitly designed to be less prone to validation overfitting
is evidence that the ambiguity is a genuine limit of this benchmark's
182-chain test set — which can only carry so much statistical power to
distinguish MCC differences of the size observed here (a raw difference
of +0.003 to +0.005 depending on combiner) — rather than an artifact of
any one weight-search procedure. A substantially larger cryptic-pocket
benchmark would be needed to determine whether a genuine MCC
improvement from ensembling exists at all.

## Conclusion

Fusing persistent homology features into an E(n)-equivariant graph neural
network improves protein binding-site classification over sequence and
two-feature burial baselines (+0.0423 PR-AUC, p<1e-5), but a stronger
control shows most of this is ordinary local geometry: against 17 plain
geometry features, topology adds a residual beyond geometry that is
confirmed on both PR-AUC and ROC-AUC at 650M and 3B embedding scale
(+0.0091 to +0.0115 PR-AUC) and on PR-AUC only at the original 35M scale
(+0.0097) — the effect strengthens, not weakens, as embedding scale grows
— while adding nothing detectable on CryptoBench (35M only). With a
trained localization head, the model
also improved pocket-center localization over a no-topology control
(+0.0289 to +0.0528 DCC depending on data scale, multi-seed confirmed). A
geometry control for localization found the same pattern as classification:
geometry alone beats topology alone (+0.0305 pooled DCC, 7/8 seeds), and
geometry-plus-topology beyond geometry alone is not established at any
of the three embedding scales tested (per-seed +0.0051, +0.0206 and
-0.0007 at 35M/650M/3B; pooled +0.0084, 95% CI -0.0027 to +0.0194, with no
scale trend), and it is negative with 3.5x more training data (-0.0242
per seed at 7,513 chains, where geometry alone is the best arm) — in
contrast to classification, where the residual is confirmed at 650M and
3B. The combination
of topology-fused equivariant message passing with a trained 3D
localization objective is more specific than either ingredient's own prior
art: published work fusing
topology with a graph neural network exists for protein function
prediction and for characterizing already-located pockets (see Related
Work), and published equivariant architectures for small-molecule
pocket prediction (EquiPocket, VN-EGNN) use no topological features, but
we did not find prior work combining topology-fused equivariant message
passing with a trained 3D localization objective specifically. At sufficient training-data scale, the resulting
system reaches a genuine, contemporaneously validated win over a live,
current run of P2Rank on COACH420 (DCC +0.072) and approximate statistical
parity with it on HOLO4K. Extended to CryptoBench's cryptic-pocket task —
a structurally distinct and harder problem, since no cavity is visible in
the input structure — the same architecture, strengthened with transfer
learning and cross-scale ensembling across three ESM2 embedding sizes,
reaches AUC 0.918 and MCC 0.470 via a regularized logistic-regression
cross-scale combiner, matching or exceeding the CryptoBench authors' own
published state of the art, while providing a dedicated 3D localization
capability for cryptic pockets that has no prior published comparison at
all. Not every result improved: several ablations (GNN depth/capacity, an
auxiliary noncryptic-pocket loss, a naive cross-chain arbitration fix,
larger embeddings on the main pipeline) returned null or negative results
and are reported as such, and a chain-level bootstrap formally confirms
that the final ensemble's AUC gain over its best single component (3B-
transfer alone) is statistically significant while its MCC gain is not —
both conclusions are reported plainly rather than rounding the second
into a further claimed improvement.

## Future Work

Natural extensions of this specific line of work, grounded in gaps this
project identified but did not close:

- **Finish the geometry control.** The control has now been run at all
  three main-pipeline embedding scales for classification (35M/650M/3B, on
  the 2,055-chain pool), at 35M for CryptoBench classification, and at 35M
  for COACH420/HOLO4K localization. Geometry alone matches or beats
  topology alone in every setting. Topology's residual contribution beyond
  geometry is null on CryptoBench classification, confirmed on both
  PR-AUC and ROC-AUC at 650M and 3B main-pipeline classification, only
  partially confirmed at 35M classification (PR-AUC only), and not
  established for localization at any of the three scales (pooled
  +0.0084, 95% CI -0.0027 to +0.0194; and -0.0242 per seed at 7,513
  chains). The classification control at the 7,513-chain pool was started
  but not completed: only 4 of its 10 cross-validation folds finished
  before a weekly compute quota ran out, and those folds are not reported
  here as a result (the geometry-plus-topology advantage was positive in
  all 4, but four folds cannot meet this project's two-test rule). Still
  untested: the remaining 6 folds, CryptoBench at 650M/3B, and CryptoBench's own
  localization head. Two questions matter most. First, why classification's
  residual strengthens with embedding scale while localization's does not,
  which no experiment here explains. Second, which specific topological
  property (if any) drives the confirmed classification residual, which
  remains unidentified; a localization study with more seeds (the
  pooled interval above is wide enough to include an effect of about +0.02)
  would also be needed to rule a small localization effect in or out.
- **A larger cryptic-pocket benchmark.** CryptoBench's 60-chain validation
  and 182-chain test splits are small enough to leave the ensemble's MCC
  gain statistically underpowered: a chain-level bootstrap over the
  182-chain test set found a real, significant AUC improvement but could
  not distinguish the MCC improvement from zero at 95% confidence for
  *any* of four different combination methods tried, including a
  regularized logistic-regression combiner explicitly designed to be
  less prone to validation-set overfitting than the grid search. That
  consistency across methods argues the limitation is the test set's
  size, not the combination procedure — a benchmark an order of
  magnitude larger would let this question (and the topology-dimension
  decomposition question) be answered with the same statistical
  confidence the main pipeline's much larger COACH420/HOLO4K pool
  allowed.
- **A better-resolved topological-variability signal.** This project
  tested one specific implementation (an 8-conformation ANM-perturbed
  ensemble, rigid per-residue atom translation, 1.2 Å target RMSD) and
  found it hurt rather than helped, plausibly because that small an
  ensemble under-samples the true conformational distribution. A larger
  ensemble, a proper all-atom elastic-network extension in place of the
  rigid-translation approximation, or a learned (rather than fixed a
  priori) RMSD/mode-count setting are the natural next steps before
  concluding topological variability itself has no value here — this
  project's negative result rules out the specific implementation
  tried, not the underlying hypothesis. Multi-scale filtrations,
  alternative complexes, and topological features computed on predicted
  (rather than only true) atomic geometry remain separately unexplored.
- **Resolve whether ANM-predicted flexibility is truly uninformative or
  just badly scaled.** The single-feature version of this idea (raw
  predicted mean-square-fluctuation, no ensemble) came back a clean null
  rather than a negative, and the leading candidate explanation — a
  heavy-tailed raw distribution poorly served by plain z-score
  normalization — was never tested. A log-transform or rank-based
  normalization of the same feature, re-run with the same 8-seed
  protocol, would settle whether this specific null reflects the
  feature itself or just its scaling.
- **A dedicated cryptic-pocket localization benchmark and stronger
  localization architecture.** The 20.9–24.7% DCC@4Å hit rate on
  CryptoBench, while the first result of its kind, is far below the
  50–65% range achieved on open pockets. One alternative localization
  head design was tested — masking the center-prediction softmax to the
  top-K highest-scoring residues (see Individual Levers) — and came back
  a null result on DCC itself, on both validation and test; more
  embedding capacity was also tried and did not help localization
  specifically. Whether a fundamentally different mechanism (e.g. an
  auxiliary spatial-clustering loss, or explicitly predicting a pocket
  radius rather than a single point) can close some of the gap remains
  untested.
- **Larger, more diverse ensembles.** Only three ESM2 scales were combined
  here, using a single validation split to fit the combiner; combining
  models trained with different architectural choices (not just different
  embedding sizes), or fitting the combination rule with cross-validation
  across multiple splits rather than one fixed 60-chain validation fold,
  is a natural next step once a larger benchmark makes such a search more
  informative.
