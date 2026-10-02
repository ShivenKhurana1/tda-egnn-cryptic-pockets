# Fusing Persistent Homology with Equivariant Graph Neural Networks for Protein Binding-Site and Cryptic-Pocket Localization

## Abstract

Predicting where a small molecule will bind a protein is a central problem in
structure-based drug discovery, and it becomes substantially harder for
*cryptic* pockets — binding sites that are not visible in a ligand-free (apo)
structure and only open up upon binding. This project tests whether features
from persistent homology (a tool from topological data analysis, TDA) improve
an E(n)-equivariant graph neural network's (EGNN) ability to both classify
binding-site residues and localize a pocket's 3D center, on standard and
cryptic-pocket benchmarks alike, and — the project's central, hard-won
finding — asks how much of any such gain survives once it is checked against
a strong non-topological baseline, an independent relabeling of what "cryptic"
actually means, and structures the model never trained on at all.

Protein structures are represented as residue-level graphs; each residue
carries one-hot identity, backbone dihedral, ESM2 protein-language-model
embedding, and persistent-homology (H0/H1/H2 + burial) features, propagated
through equivariant graph convolution layers and fused via a per-layer gated
fusion module before two task heads: residue classification and
centroid-based pocket localization. On the COACH420/HOLO4K training pool,
adding topology to sequence and burial features gave a large classification
gain against a two-feature burial baseline (+0.0423 PR-AUC, 10/10
cross-validation folds, p<1e-5). A stronger control shows most of this is
ordinary local geometry, not anything specifically topological: against 17
plain per-residue geometry features (neighbor counts, solvent exposure,
half-sphere exposure, local shape, ray-cast buriedness, hull and residue
depth) built to match the same architecture exactly, geometry alone matches
or outperforms topology alone at every embedding scale tested (35M/650M/3B)
and at every training-pool size tested (2,055 up to 7,513 chains). Topology
does add a small residual beyond geometry that survives both a paired t-test
and a Wilcoxon test for classification at 650M and 3B scale (+0.0091 to
+0.0115 PR-AUC) and shrinks toward the resolution of the experimental design
at the largest training-pool size (+0.0035 PR-AUC, not confirmed; and a
same-fold hardware check found the training device alone moves the result by
about this much). The identical control finds no confirmed topology residual
for CryptoBench classification, for pocket-center localization at any scale
or pool size tested, or once the underlying 3B model is pretrained rather
than trained from scratch. Read plainly: seventeen ordinary geometric
measurements deliver most of what persistent homology was hypothesized to
provide uniquely, and what remains attributable to topology specifically is
small, inconsistent across tasks, and in at least one case the same size as
measurement noise from the training hardware.

A trained localization head (predicting a pocket's 3D center as a
differentiable weighted centroid rather than a post-hoc heuristic) beat a
live, current run of the standard tool P2Rank on COACH420 (DCC@4Å +0.072)
and reached approximate statistical parity with it on HOLO4K. On
CryptoBench's cryptic-pocket benchmark, the same architecture, extended
across three ESM2 embedding scales and strengthened by transfer learning
from the main pipeline (paired t=3.97, p=0.0054, confirmed against five
independent pretrained checkpoints) and a regularized logistic-regression
cross-scale combiner, reaches AUC 0.918 and Matthews correlation coefficient
(MCC) 0.470 — matching or exceeding the CryptoBench authors' own published
state of the art (AUC 0.87, MCC 0.42). Transfer learning, not the feature
set, is the confirmed driver of this result (+0.058 PR-AUC over training
from scratch, 8/8 seeds), and it survives a homology audit that removed
every pretraining-pool chain related to a CryptoBench test protein
(unchanged result, +0.001 PR-AUC, p=0.86). The model beats a fresh P2Rank
2.5.1 run on the official split by a wide margin on both residue ranking
(+0.19 protein-level PR-AUC) and pocket-center guessing (35.8% vs. 16.1%
within 4 Å for a single guess; blending the two models does not help,
since P2Rank contributes nothing the model does not already know).

Two further checks, run after the model was otherwise finished and not used
to tune anything, describe the actual scope of every number above. First, an
independent, adversarially designed relabeling of CryptoBench's own test
split — built by a different group specifically to separate pockets that are
genuinely invisible in the unbound structure ("cryptic," 16 of 182 test
proteins under a strict cavity-voxel rule) from those already visible before
anything binds ("pre-formed," 111 of 182) — shows that this project's pooled
numbers describe the pre-formed majority far better than the cryptic
minority: pooled ROC-AUC falls from 0.896 (all 182 proteins) to 0.721 on the
16 truly cryptic ones. The model still beats P2Rank specifically on that
16-protein subset (+0.115 precision among the top 20 ranked residues per
protein, paired t and Wilcoxon p=0.022 each), and — testing an external
claim that cavity/geometry features actively *hurt* performance on truly
cryptic sites — geometry trends the opposite direction on our architecture
(not confirmed, but positive rather than negative). Second, the unmodified
model was run on two datasets built entirely outside this project, with no
threshold-fitting or model selection performed against either: PocketMiner's
35-protein test set (labels from molecular-dynamics simulation, a
fundamentally different definition of "cryptic" than anything else in this
paper) and 99 genuinely-cryptic entries from the same adversarial benchmark
above, both fetched and featurized fresh. On PocketMiner, performance is
essentially unchanged after removing every protein with any relative in the
training data (pooled ROC-AUC 0.925 all proteins, 0.920 on the 6 with no
training-pool relative) and exceeds PocketMiner's own published result for
its own method (0.87). On the second, harder benchmark, homology matters
more (0.833 all proteins, 0.757 on the 16 homology-clean) — but that
homology-clean number lands within statistical noise of the 0.721 found by
the independent relabeling above, from a different research group using a
different definition of "cryptic." Three separate checks, two of them on
structures the model never saw in any form, converge on the same
~0.72–0.76 ROC-AUC ceiling for pockets that are genuinely, strictly
invisible until something binds. A battery of further attempts to close
that specific gap — per-residue chemistry features, elastic-network-derived
protein-movement features, explicitly up-weighting small pockets (and,
once a validated geometric crypticity proxy made it possible, up-weighting
cryptic-scoring chains the same way) in the training loss, extending
training with 1,997 real chains from a second benchmark, and (after
solving a real GPU-infrastructure problem along the way) a genuine
molecular-dynamics pilot on the hardest, independently-confirmed cryptic
test proteins — each returned a null, negative, or statistically
inconclusive result (the MD pilot's own single most standard metric,
Cα-RMSF, landed at exactly chance, AUC 0.500, though on only 8 of 17
planned proteins before a platform time limit ended the run; the stronger
crypticity-proxy up-weighting arm significantly hurt pooled performance,
PR-AUC -0.021, DCC@4 -0.039, while leaving the true-cryptic subset itself
unmoved) — consistent with this ceiling reflecting a real information
limit of a single static structure rather than a fixable modeling gap.
Separately, and unlike every
attempt above, checking whether the model's raw predicted probabilities can
be trusted at face value (a question no discrimination metric answers)
found them badly and specifically overconfident — at the model's own
highest-confidence decile, the true positive rate is less than half what
it claims (0.331 actual vs. 0.838 predicted) — and found that isotonic
regression fit on the existing 60-chain validation set fixes nearly all of
it (expected calibration error 0.166 to 0.0029, a 57-fold reduction) without
changing any ranking-based result in this paper.

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
are all actively benchmarked cavity-detection tools (Utgés and Barton,
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
depth; and residue depth) and re-ran the original nested cross-validation:
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

**Does the residual survive larger embeddings?** The
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

**Which topological dimension drives this residual?** The decomposition below (H0/H1/H2/burial, each
combined with the full 17-feature geometry block) was run at 650M scale — where the residual is confirmed
and the effect is large enough to decompose without needing 3B's heavier compute — reusing the `geom` and
`geom+topo` arms already produced above (not rerun) and adding four new arms on the identical 10-fold split.
H1 (loops/tunnels) is the only single dimension confirmed against geometry alone on *both* metrics: +0.0068
PR-AUC (9/10 folds, p=0.001/0.004) and +0.0025 ROC-AUC (8/10 folds, p=0.034/0.037). H0 and burial each
confirm on ROC-AUC alone (+0.0027, p=0.004/0.010; +0.0024, p=0.013/0.020) but not PR-AUC; H2 confirms on
neither. Pairwise among the four, H1 significantly beats both H2 (+0.0064 PR-AUC, p=0.028/0.027) and burial
(+0.0043 PR-AUC, p=0.036/0.049), and — mirroring the earlier, independently-run localization decomposition's
one confirmed finding, on a different pool, task, and metric — H0 comes in significantly below H1 here too
(−0.0048 PR-AUC, p=0.032/0.049). Most tellingly, geometry-plus-H1-alone is statistically indistinguishable
from geometry-plus-all-four-dimensions (+0.0023 PR-AUC, p=0.255/0.193, not confirmed), while geometry alone
is significantly worse than either (geom+topo beats geom+H0 by +0.0071 PR-AUC, p=0.021/0.027; beats geom+H2
by +0.0087, p=0.004/0.010) — H1 alone appears to carry the residual that the full topology block provides,
with H0, H2, and burial adding nothing detectable on top of it. Sixteen pairwise tests were run here without
a formal multiple-comparison correction, so any single p-value near 0.05 should be read cautiously, but the
pattern is not one isolated hit: H1 clears both metrics against geometry, beats two of the other three
dimensions pairwise, and geometry+topology tracks it rather than exceeding it — four mutually consistent
contrasts, not a scattered one-off. Read plainly: of the four things this project's topology block computes,
the one describing loops and tunnels in the local atomic point cloud — the feature type most naturally
analogous to a cavity's shape — is doing essentially all of the confirmed work.

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
pretrained P2Rank 2.5.1 (ROC-AUC 0.796 vs. 0.756, p=1e-4). Training-pool
size does not explain the difference between the main pipeline and CryptoBench:
when the main-pipeline pool was enlarged 3.65-fold to 7,513 chains and
evaluated by 10-fold cross-validation (all ten folds, two seeds per fold,
35M embeddings), the topology residual over geometry shrank rather than
grew: geometry plus topology minus geometry was +0.0035 PR-AUC (95% CI
-0.0008 to +0.0078; geometry+topology better in 7 of 10 folds; paired t
p=0.097, Wilcoxon p=0.084; not confirmed) and +0.0018 ROC-AUC (CI +0.0004
to +0.0032; 8 of 10 folds; t p=0.015, Wilcoxon p=0.020; confirmed but
tiny). Geometry alone is again far above no topology (PR-AUC 0.558 vs.
0.496, ROC-AUC 0.898 vs. 0.873, 10 of 10 folds) and above topology alone
(0.558 vs. 0.542 PR-AUC, 10 of 10 folds). So at this scale 17 plain
geometry features deliver more of the gain than topology does (+0.062 vs.
+0.046 PR-AUC over no topology), and what topology adds on top is at most a
few thousandths of PR-AUC. (Folds
0-3 were trained on GPU and folds 4-9 on CPU because the weekly GPU quota ran
out. A same-fold check that reran folds 0 and 1 for the geometry and
geometry-plus-topology arms on CPU (2 seeds each, 8 matched records) found CPU
scores 0.002 lower in PR-AUC and 0.001 lower in ROC-AUC on average (largest
single difference 0.008 / 0.005), and the sign of the topology residual over
geometry was kept on PR-AUC in both folds; that device difference is about the
same size as the residual itself, so the 7,513-chain topology residual should be
read as small and near the resolution of this design.)
The control was also run on the strongest CryptoBench setting (ESM2-3B, official
split, 8 seeds per arm, trained from scratch). There, geometry alone raised
test PR-AUC from 0.348 to 0.369 (+0.021; 7 of 8 seeds; t p=0.041, Wilcoxon
p=0.078, so not confirmed) and ROC-AUC from 0.872 to 0.881 (p=0.028 / 0.039),
and adding topology on top of geometry gave nothing (PR-AUC -0.005, p=0.21 /
0.20). Topology alone did not improve PR-AUC over no topology (+0.002).
Geometry did not improve pocket-center accuracy (DCC@4 Å 0.217 vs. 0.219).
These from-scratch 3B models (ensemble ROC-AUC 0.891-0.895) remain below the
transfer-learned 3B model reported earlier (0.911), so the geometry gain
here is a modest addition to a weaker starting point, not a new best result.
Scope of this control: classification and localization were each run at all
three main-pipeline embedding scales (35M/650M/3B) on the 2,055-chain pool,
classification also at 7,513 chains (35M), and CryptoBench at 35M and (from
scratch only) 3B, and with 3B transfer learning (next paragraph).
Geometry does not help once the model is pretrained. We retrained the
main-pipeline 3B checkpoints with the 17 geometry features (7,178 clean
chains, 8 seeds per arm) and repeated the CryptoBench 3B transfer with each
seed fine-tuned from its matching checkpoint. Geometry plus topology minus
topology alone was -0.003 test PR-AUC (2 of 8 seeds better; t p=0.53), +0.004
ROC-AUC (p=0.11), +0.003 MCC and +0.013 pocket-center hit rate at 4 Å (p=0.21);
geometry alone was equally indistinguishable from topology alone. The
8-seed ensembles are essentially tied (ROC-AUC 0.9120 topology, 0.9138
geometry, 0.9128 both). Pretraining, not the feature set, carries the gain:
transfer beat from-scratch training by +0.058 PR-AUC and +0.021 ROC-AUC
(8 of 8 seeds, Wilcoxon p=0.0078) for the topology arm, and by +0.041 PR-AUC
for the geometry-plus-topology arm. Leakage check: the pretraining pool was
cleaned only against COACH420, and an MMseqs2 check (30% identity, 80%
coverage) shows that 135 of the 182 official test chains (74%) have a
homolog in it (539 of 700 training chains). The transfer gain survives on the
47 test chains with no such homolog (+0.108 PR-AUC, +0.024 ROC-AUC, 8 of 8
seeds), so the homolog overlap does not explain it, although the 47-chain
subset is small. On that subset topology-only transfer was better than
geometry plus topology on PR-AUC (0.469 vs. 0.437, 0/8 seeds; p=0.008),
which we do not over-read (47 chains, one subset). We then removed the
174 pool chains (of 7,178) that have a homolog of any CryptoBench validation
or test chain and repeated the whole transfer (retrained 3B checkpoints on the
remaining 7,004 chains, 8 seeds, topology-only arm). On all 182 test chains,
now homology-clean relative to pretraining, clean-pool transfer scored PR-AUC
0.408, ROC-AUC 0.896, MCC 0.437 and DCC@4 Å 0.298 per seed (8-seed ensemble ROC-AUC
0.911, DCC@4 Å 0.357), which is statistically the same as the leaky-pool
transfer (PR-AUC +0.001, p=0.86; ROC-AUC -0.004, p=0.16) and still beats
from-scratch training by +0.059 PR-AUC, +0.017 ROC-AUC and +0.061 DCC@4 Å (8 of 8
seeds each, Wilcoxon p=0.0078). Transfer learning's gain is therefore not a
homology artifact. (The 0.911 figure in the table below is the earlier,
leaky-pool ensemble; the clean-pool ensemble gives the same 0.911.) The
clean check removes homologs of validation and test chains only; homologs of
CryptoBench training chains are left in the pool by design.

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
- **Per-residue chemistry features (inconclusive, not adopted).** 16
  features per residue derived from amino-acid identity and Cα geometry
  alone (no ligand information): hydropathy, volume, formal charge,
  aromaticity, hydrogen-bond donor/acceptor counts (self and 10 Å
  neighbourhood-averaged), local net and absolute charge, a
  charge-heterogeneity term, and the size of the nearest hydrophobic
  patch (residues with positive hydropathy connected at 6 Å). Verified
  rotation/translation-invariant to machine precision. Same matched-control
  design as the movement-feature test (homology-clean 3B pretraining pool,
  8 seeds, fine-tuned on CryptoBench). Pooled test PR-AUC +0.0112 (paired t
  p=0.046, Wilcoxon p=0.055 — one test crosses this project's 0.05 bar, the
  other does not, so this is **not confirmed** by the pre-registered
  two-test rule); ROC-AUC +0.0011 (p=0.75). On the small-pocket stratum the
  hydrophobic-patch feature was aimed at, centre-within-4Å accuracy went
  down rather than up (-0.026, t p=0.028, Wilcoxon p=0.055 — again split
  between the two tests). Read together: a small, inconsistent effect that
  does not meet this project's bar for a confirmed result in either
  direction. Not adopted.
- **Protein-movement (elastic-network) features (null, not adopted).** A
  richer test of the flexibility hypothesis than the earlier single-column
  or perturbed-topology attempts (see below): 10 features per residue from
  a Cα-level elastic network built on the apo structure alone, including
  rank-scaled predicted fluctuation, how collective a residue's motion is,
  distance to the nearest structural hinge, and — aimed directly at the
  "could this open up" question — how much each residue's local
  neighbourhood loosens or tightens when the structure is nudged along its
  5 slowest natural motions. Verified rotation/translation-invariant to
  machine precision before use. Added to the homology-clean 3B pretraining
  pool (7,004 chains) and to CryptoBench, matched-control design (topology
  alone vs. topology plus the 10 movement columns, identical 44-wide side
  channel, both arms fine-tuned from their own pretrained checkpoints, 8
  seeds each). Result: pooled test PR-AUC +0.0042 (p=0.60), ROC-AUC +0.0059
  (p=0.089) — in the same direction as hoped but not confirmed, and no
  better on the small-pocket stratum where the idea predicted the most
  benefit (ROC-AUC -0.006, p=0.19). Not adopted.
- **Small-pocket loss weighting (negative).** On the 49 official test
  chains whose pocket has 8 or fewer residues, the transfer model is far
  weaker than elsewhere (protein-level ROC-AUC 0.69 vs. 0.90-0.93; pocket
  centre within 4 Å for 8% of chains vs. 36-50%; mean centre error 16.9 Å vs.
  5.4-7.5 Å). Up-weighting proteins with small pockets in the loss (weight
  (median pocket size / size)^beta, clipped to [1, max]; beta 0.5/max 3 and
  beta 1.0/max 6, applied to the residue and centre terms; 8 seeds each,
  fine-tuned from the same homology-clean 3B checkpoints as the unweighted
  baseline) did not help: small-pocket ROC-AUC +0.002 (p=0.55) and +0.006
  (p=0.14), small-pocket DCC@4 Å -0.013 and -0.005. The stronger weighting
  slightly hurt overall performance (pooled PR-AUC -0.008, p=0.002 / Wilcoxon
  0.008; DCC@4 Å -0.024, p=0.004). The small-pocket gap therefore does not
  come from loss imbalance; with 8 or fewer positive residues there is little
  signal for the model to learn from, and a different kind of information
  (not a re-weighting) would be needed.
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
  explains the whole benefit." (This was the localization gain,
  at 35M-embedding/round-3-pool scale; the same question for the
  *classification* residual, at the 650M/3B scale where that residual is
  actually confirmed, is answered separately under "Geometry control" in
  Results, above — and resolves further there: H1 specifically, not just
  "not H0," carries the confirmed classification residual.)
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

**Geometry control for localization.** Retraining the
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
p=0.62, Wilcoxon p=0.92). A follow-up analysis reloaded the
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
**Localization at larger embedding scales.** Because
classification's residual strengthened with embedding scale (above), the
same localization control was repeated at ESM2-650M and ESM2-3B: the
COACH420/HOLO4K benchmark cache was re-embedded at each scale by decoding
each chain's sequence losslessly from the existing cached one-hot block
(no structure was re-fetched), rebuilt and checked column-for-column
(1,341 and 2,621 features per residue), and 4 arms × 8 seeds were retrained
on the corresponding 2,055-chain training blob.

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

**Localization with 3.5x more training data.** The
original localization result (this paper's headline) used the
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
  an earlier round-3 decomposition, while the topology arm
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
- **Genuinely local (spatial) attention for pocket-center prediction** —
  the other "most promising untried direction" this project's own Future
  Work had flagged, since top-K restricts the softmax to the
  highest-*scoring* residues, which can still be spatially scattered
  across a long chain if several unrelated regions all score highly.
  This instead adds a second softmax pass whose logit is penalized by
  squared distance from the *first* pass's own preliminary center
  estimate (`logit' = logit − (dist-to-preliminary-center / R)²`, R = 10
  Å, the same edge-cutoff radius already used elsewhere in this
  architecture, not tuned for this result) — a purely spatial, rather
  than score-based, locality constraint. Because the distance term is a
  norm of a difference of two already-equivariant quantities (the same
  construction EGCL itself uses for its own edge features), this
  preserves equivariance by construction; we verified it numerically
  before training (rotated-batch check, max error 2.4e-5, matching every
  other equivariance check in this project) rather than only after the
  fact. Like top-K, this changes zero parameter shapes, so the same 8
  Phase 266b homology-clean checkpoints loaded unchanged and were
  fine-tuned 1:1 against the existing Phase 266c baseline (not rerun).
  Result: **null on the target metric, again** — DCC@4Å fell slightly
  (0.2981 to 0.2898, 3/8 seeds, t p=0.54/Wilcoxon p=0.50) and DCC@8Å rose
  slightly (0.5968 to 0.6051, 3/8 seeds, p=0.53/0.83), neither confirmed;
  mean center error was essentially unchanged (9.62 to 9.60 Å). We
  initially expected classification metrics to be unaffected, since the
  classification head's logits never read from the localization
  readout — this was wrong, and the result corrects it: PR-AUC, ROC-AUC,
  and MCC all shifted by up to 0.026 per seed, because the shared EGCL
  trunk is trained jointly against *both* losses, so changing the
  localization loss's gradient changes the learned representations the
  classification head also reads from (none of these shifts were
  confirmed either, in either direction — consistent with noise from
  retraining a shared trunk, not a systematic side-effect). This is the
  same qualitative outcome top-K found by a different mechanism: a
  well-motivated, mechanistically distinct change to how the center
  prediction pools over residues, which does not move the metric it
  targets. Between the two untried directions this project's Future Work
  had flagged, multi-candidate pockets (below) is the one that
  confirmed helping; a smarter single-point attention mechanism, tried
  twice now by two different mechanisms, has not.
- **Global (long-range) attention in the trunk** — a different kind of
  architecture change from everything above: not a smarter readout for
  the center prediction, but a change to the shared representations
  *both* heads read from. Every layer in this architecture (EGCL) is
  local message passing over an 8–10 Å radius graph; GNN capacity
  (width/depth within that same local-receptive-field design) was
  already shown not to help, but receptive field is a different claim —
  a residue can currently only be informed by a distant part of the fold
  through many diluted hops. We added ONE global multi-head
  self-attention block after the full EGCL+Fus stack, letting every
  residue attend to every other residue directly regardless of distance.
  It operates entirely on the invariant scalar channel (never reads or
  writes coordinates), so it is trivially SE(3)-equivariant by
  construction — verified numerically before training anyway (rotated-
  batch check, max error 2.1e-6). This adds new parameters, so, like the
  topological-variability and ANM-flexibility experiments, it must be
  trained from scratch; both arms (baseline and baseline+global-attention)
  were retrained fresh here, 8 seeds each, at 650M scale, for a clean,
  fully paired comparison. **Result: the most promising signal of any
  architecture or feature change attempted in this project's later
  stretch, though not cleanly confirmed on its own pre-specified primary
  metric.** Validation PR-AUC — the metric this project uses first for
  every other CryptoBench lever — rose 0.3663 to 0.3751 (+0.0088, 6/8
  seeds) but narrowly missed significance on both tests (t p=0.093,
  Wilcoxon p=0.078). The held-out TEST set, three times larger than
  validation (182 vs. 60 chains) and correspondingly less noisy, shows a
  stronger and *confirmed* version of the same pattern: PR-AUC +0.0156
  (7/8 seeds, t p=0.003, Wilcoxon p=0.016) and ROC-AUC +0.0059 (7/8
  seeds, t p=0.011, Wilcoxon p=0.016). MCC, DCC@4Å, and mean center error
  all moved the same positive direction without individually clearing
  significance (MCC +0.0064; DCC@4Å +0.0192; error −0.15 Å). We did not
  chase additional seeds to force the validation metric past 0.05 — that
  would be the same p-hacking this project declined to do for the
  GNN-depth result — and report the honest, partial picture: a
  consistently positive, test-confirmed-on-two-metrics signal from a
  genuinely new architectural ingredient, on a from-scratch 650M model
  that itself sits below this project's transfer-learned headline. Taking
  this further — retraining the much larger main pipeline with global
  attention and transferring the result to CryptoBench, the path that
  turned every other confirmed CryptoBench gain in this paper into its
  final, strongest form — would be a substantially larger compute
  commitment than this pilot, and was not attempted here.

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

**More training data, from a different benchmark, hurt rather than helped.** An independent, adversarially-designed
benchmark used elsewhere in this paper for external validation ("A benchmark selected for crypticity") also screened
2,401 apo/holo pairs of its own, of which only 99 are labelled genuinely cryptic (already spent as external-validation
data); the remaining 2,300 "pre-formed" and "neither" pairs were never used anywhere in this project. After a homology
check removed the 305 that share a >=30%-identity relative with anything already evaluated on, 1,997 real, independently-
labelled apo/holo pairs were available to extend CryptoBench's own 700-chain training set (to 2,697), using the exact
same recipe (checkpoints, architecture, loss, optimiser, 8 seeds) as the headline result. This made performance worse,
not better, and the effect is confirmed by this project's two-test rule: test PR-AUC fell by -0.0272 (0 of 8 seeds
improved; paired t p=0.005, Wilcoxon p=0.008) and MCC by -0.0253 (1 of 8 seeds improved; p=0.018/0.016); ROC-AUC also
fell (-0.0124) but not significantly (p=0.13/0.15). The 8-seed ensemble shows the same pattern (PR-AUC 0.415 vs. 0.442,
MCC 0.449 vs. 0.463). We do not read this as "more data hurts" in general — the training recipe (40 epochs, patience
8, batch 16, a fixed learning-rate schedule) was tuned for 700 chains and left completely unchanged for a training set
2.75x its original size; it is entirely possible the extra chains, coming from a differently-curated benchmark, teach
the model something CryptoBench's own test distribution does not reward. Both
explanations are plausible and this project cannot distinguish them without either re-tuning the recipe for the larger
set or auditing what specifically differs between the two benchmarks' site definitions; what can be said cleanly is
that naively pooling real, correctly-labelled training data from a second cryptic-pocket benchmark, with no other
change, does not improve this model and measurably hurts it.

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

**Multiple candidate pockets and comparison with P2Rank on the official split.**
Using the homology-clean 3B transfer ensemble (8 seeds) on the
official test chains, with P2Rank 2.5.1 rerun on exactly those structures
(179 of 182 chains usable; 4 residues of 4 chains had no P2Rank score),
the ensemble's single predicted pocket centre lands within 4 Å of the
experimental pocket for 35.8% of chains (within 8 Å: 62.6%; mean error 9.1 Å),
against 16.1% (37.4% within 8 Å) for P2Rank's top-ranked pocket on the same
174 like-for-like chains (paired sign test at 4 Å: 42 chains hit only by the
model vs. 9 only by P2Rank, p<0.001). Returning up to three candidates
raises the model's rate: a hybrid list (the model's own centre first, then up
to two further residue-cluster centres more than 8 Å away) hits within 4 Å for
37.4% of chains and within 8 Å for 71.5% (10 Å: 78.8%), against
25.3% / 57.5% (10 Å: 69.5%) for P2Rank's best of three pockets (like-for-like
sign tests: 4 Å, 35 vs. 15, p=0.007; 8 Å, 35 vs. 12, p=0.001). The clustering
parameters were chosen on the 60 validation chains only (grid of 9; the
test chains were scored once). Clustering by itself, without the model's
own centre, is not better at 4 Å (25.1%), so the gain at tight thresholds
comes from the learned centre and the extra candidates mainly help at 8 Å
and beyond. The threshold matters for the claim: at 4 Å, a second and third
guess add little (35.8% to 37.4%); at 8 Å they add roughly nine points.
Blending with P2Rank does not help this model: a logistic stack fit on the
validation chains puts essentially zero weight on P2Rank's score (coefficient
-0.04 against 1.77 on the model's logit), and the blend was not better than
the model alone (protein-level PR-AUC -0.004, ROC-AUC -0.001; rank-averaging
was worse, -0.056 PR-AUC). The model beats P2Rank by +0.19 protein-level
PR-AUC and +0.08 ROC-AUC (73% and 84% of proteins, p<1e-15). An earlier
+0.014 PR-AUC blending gain on a from-scratch ensemble (hard-split test set)
therefore does not carry over to the stronger transfer model.

### External audit: is this actually about cryptic pockets?

Every number reported above pools all 182 official CryptoBench test proteins. An independent, adversarially designed
relabeling of that same test split — "A benchmark selected for crypticity" (unpublished preprint; code and labels at
github.com/pmfran4252-afk/cryptic-site-benchmark, CC BY 4.0) — applies a strict apo/holo cavity-voxel rule (cryptic =
pocket essentially absent in the unbound structure AND clearly present in the bound one) and finds that only 17 of the
182 test proteins qualify; 111 are "pre-formed" (already visible in the apo structure) and the rest are ambiguous. We
mapped their relabeling onto our own test keys via CryptoBench's own dataset metadata (180 of their 182-labelled
proteins matched; 16 of the 17 cryptic ones fell inside our exact test set) and re-scored our existing prediction files
against it — no new training.

**The pooled numbers are mostly about pre-formed pockets.** On the 16-protein cryptic-only subset, the headline
homology-clean 3B-transfer model's pooled ROC-AUC falls from 0.896 (all 182) to 0.721, protein-level PR-AUC from 0.459
to 0.262, and precision among the top 20 ranked residues per protein from 0.383 to 0.200. This matches the pattern the
external preprint reports for other published models and is the honest scope of every number in the table below: this
project's headline results describe finding pockets in general far better than it describes the harder cryptic subset.

**Their central negative claim does not replicate on our architecture.** The preprint's strongest finding is that
cavity/geometry features are *actively harmful* specifically at cryptic sites (removing them improved cryptic-subset
AUC by 0.06-0.07 in their models). Repeating their stratification on our own geometry-vs-topology CryptoBench 3B-transfer arms shows the opposite trend, not the same one: geometry minus topology-only, within the cryptic stratum, is
+0.018 pooled PR-AUC and +0.015 pooled ROC-AUC (5-7 of 8 seeds) — not confirmed by this project's two-test rule
(p=0.08-0.11), but positive, not negative. We do not claim geometry helps cryptic sites here; we do not replicate their
claim that it hurts.

**A real, confirmed result: this model beats P2Rank on the cryptic subset specifically.** Using our own P2Rank 2.5.1
rerun (run on the official split for an unrelated localization comparison, so not tuned for this question)
restricted to the 16 cryptic proteins where both methods have scores, the headline model's precision among the top 20
residues beats P2Rank's by +0.115, paired over proteins (t p=0.022, Wilcoxon p=0.022; bootstrap 95% CI on the mean
difference [+0.035, +0.203], n=16). The 8-seed ensemble reaches precision@20 0.209 and recall@20 0.314 on this subset --
above every baseline the external preprint reports on the same rule, including its own new method (0.174 / 0.278).

This is not an independent-structure validation — it re-labels proteins already in our test split, not new ones — and
n=16-17 is small; both caveats are why it is reported here rather than folded into the headline table. It is, however,
the first check of this project's result against a relabeling we had no part in designing and could not have tuned
for, and it survives that check on the one statistically confirmed claim it produces.

### External validation on genuinely new structures

Everything reported so far, including the audit above, evaluates the model on CryptoBench's own test structures --
under an outside relabeling, but the same 182 proteins. We built a separate pipeline (fetch structure, compute the
same persistent-homology and ESM2-3B features, run inference) for two datasets built entirely outside this project,
using our own homology-clean 3B-transfer headline model, unmodified, with no threshold-fitting or model selection
performed against either one:

- **PocketMiner's held-out test set** (Meller et al. 2023, *Nat. Commun.*; github.com/Mickdub/gvp): 35 proteins whose
  cryptic-pocket labels come from molecular-dynamics simulations, not from comparing a static bound and unbound
  structure — a genuinely different definition of "cryptic" than every other dataset in this project. Their own
  label scheme evaluates residues pooled across all proteins, not per protein (24 of the 35 are "positive-only"
  annotated and 10 are pure negative-example proteins; zero contain both classes after their own masking, confirmed
  directly from the label arrays), so we score it the same way they do.
- **99 genuinely-cryptic entries** from an independent, adversarially-designed benchmark (pmfran4252-afk/cryptic-site-
  benchmark, the same source as the audit above), fetched fresh from RCSB — standard per-chain site/non-site labels,
  scored both pooled and per-protein.

A homology check (mmseqs2, 30% identity, 80% coverage, against both the CryptoBench train split and the actual
7,339-chain 3B pretraining pool) found that most proteins in both sets have a relative somewhere in our training data
(29/35 PocketMiner, 82/98 cryptic-site-benchmark) — expected, since well-characterized proteins with known cryptic
pockets are exactly the kind of protein a large structural pool is likely to contain. We report the full set and the
homology-clean subset separately; the clean subset is the honest test of generalization to unrelated proteins.

**PocketMiner: the result holds essentially unchanged on the clean subset.** Pooled ROC-AUC is 0.925 (PR-AUC 0.897,
8-seed mean) on all 34 usable proteins and 0.920 (PR-AUC 0.887) on the 6 with no training-pool relative — almost no
drop. Both exceed the 0.87 ROC-AUC PocketMiner's own paper reports for its own method on its own dataset. This is the
single strongest external-validation result in this project: a different research group's structures, a different
labeling philosophy (simulated dynamics instead of a crystallized bound structure), and performance survives removing
every protein with a training-pool relative.

**cryptic-site-benchmark: weaker, and homology matters more here, but still real.** Pooled ROC-AUC is 0.833 (PR-AUC
0.363; protein-level PR-AUC 0.476) on all 98 usable proteins and drops to 0.757 (PR-AUC 0.311; protein-level PR-AUC
0.374) on the 16 with no training-pool relative — a real gap, unlike PocketMiner. Even so, 0.74–0.76 ROC-AUC on
proteins with both no training relative and a strict, independently-verified cryptic-pocket label is consistent with
-- if anything slightly better than — what the audit above found for our own CryptoBench-internal cryptic subset
(0.721 pooled ROC-AUC, in the audit above). Two different research groups, two different definitions of "genuinely cryptic,"
and the same model lands in the same place both times. That agreement is the most useful single number this section
produces: it says the ~0.72–0.76 ROC-AUC range is a property of the problem on hard, truly cryptic sites, not an
artifact of one benchmark's construction.

### Calibration: does the model's confidence mean what it says?

Every metric reported above (PR-AUC, ROC-AUC, MCC, precision@20) measures discrimination — whether the model ranks
true site residues above non-site residues — and none of them checks whether the predicted probability itself is
trustworthy: whether a residue the model scores 0.5 is actually a pocket residue roughly half the time. That is a
different, practically important question, since a real user deciding whether to act on a single prediction needs
the number itself to mean something, not just its rank. We checked this for the headline homology-clean 3B-transfer
model's 8-seed ensemble, using only predictions this project had already produced (no new training).

**The raw probabilities are badly, and specifically overconfidently, miscalibrated.** Sorting the 182 test chains'
54,992 residues into ten equal-sized bins by raw predicted probability and comparing the mean prediction in each bin
to the true positive rate: at the model's highest-confidence decile (mean predicted probability 0.838), the true
positive rate is 0.331 — less than half what the model claims. At predicted probability 0.50 (the conventional
decision threshold), the true rate is 0.065. Expected calibration error (ECE, 15 quantile bins) on the raw
probabilities is 0.166; Brier score 0.090; log-loss 0.300. This is the expected, known cost of the class-weighted
loss (`pos_weight` 20.73) used throughout this project to handle the ~4.7% positive rate: the same reweighting that
helps the model rank rare positives correctly systematically inflates the probabilities it outputs, and nothing in
this project's earlier evaluation (which only ever thresholded or ranked, never took a probability at face value)
would have caught it.

**A trivial, already-available fix resolves nearly all of it.** Isotonic regression fit on the existing 60-chain
CryptoBench validation set (used for nothing else in this calibration check) and applied to the held-out test set
drops ECE from 0.166 to 0.0029 (a 57-fold reduction), Brier score from 0.090 to 0.033, and log-loss from 0.300 to
0.124. Because isotonic regression is a monotonic transform, it changes none of this project's discrimination results
(every PR-AUC, ROC-AUC, and ranking-based comparison reported elsewhere in this paper is completely unaffected); it
only fixes what the probability number means.

**The fix generalizes to this project's own known-hardest subgroups, though not quite as completely.** Calibration
fit on the whole validation set, evaluated separately on the strictly-cryptic test residues (the external audit's
subset) and on small-pocket chains (<=8 residues, this project's other established weak point): cryptic-subset ECE
falls from 0.154 (raw) to 0.021 (calibrated); pre-formed-subset ECE from 0.153 to 0.0075; small-pocket ECE, the worst
in every setting (0.203 raw, the most severely miscalibrated group, matching its status as the hardest), falls to
0.025 — an 8-fold improvement, but still the residual worst-calibrated stratum, meaning a single pooled calibration
curve helps everywhere but does not fully equalize calibration quality across difficulty levels.

**Fitting a separate calibration curve per pocket-size stratum closes most of that residual gap.** Rather than one
isotonic fit on all 60 validation chains, we fit two — one on the 17 small-pocket validation chains (5,246 residues,
104 positive), one on the 43 larger ones (14,316 residues, 742 positive) — and applied each to the matching test
stratum. (Crypticity-stratified fitting was not attempted: the external audit's cryptic/pre-formed labels exist only
for the test split, so there is no validation-set ground truth to fit a cryptic-specific curve against.) On the
small-pocket test residues this cut ECE further, from 0.025 (pooled fit) to 0.0039 — a further 6.5-fold improvement,
now closer to the larger-pocket stratum's own result (0.0106 pooled -> 0.0048 stratified) than to where it started.
Critically, this did not cost anything on the pooled test set as a whole: overall ECE is unchanged within noise
(0.0029 pooled fit vs. 0.0030 stratified), while Brier score and log-loss both improve slightly (0.0330 -> 0.0314;
0.1242 -> 0.1209). As a bonus neither fit targeted directly, the size-stratified calibration also improves the
cryptic/pre-formed breakdown it never saw during fitting (cryptic ECE 0.021 -> 0.017; pre-formed 0.0075 -> 0.0030),
consistent with pocket size and crypticity being correlated in this dataset. Stratifying by pocket size before
calibrating is therefore a strict improvement over the pooled fit — better on the hardest subgroup, no worse
overall — and is the version we recommend over the single pooled curve.

We read this as a genuine, low-cost improvement worth adopting rather than a caveat: any deployment of this model
that reports a probability, rather than only a rank or a thresholded call, should apply size-stratified calibration
first, and the fact that it was never checked before this point is itself a reminder that a model's discrimination
metrics say nothing about whether its stated confidence can be trusted.

**Extending the crypticity audit to validation, and what a crypticity-proxy calibration adds beyond pocket size.**
The external crypticity audit above only ever labeled the 182-chain test split; the 60-chain validation split used
to fit calibration curves has no cryptic/pre-formed ground truth, which is why the size-stratified fit above could
not attempt a crypticity-specific curve. To close this gap we built our own geometric cavity-voxel detector --
ray-buriedness sampling over a 3D grid, reusing this project's Fibonacci-sphere ray-marching primitive — as an
apo-only proxy for the external tool's consensus-pocket-voxel concept (no equivalent cavity-detection software is
available to us). Validated against the external tool's own published apo-side voxel counts, on the 128 test chains
it labels cryptic or pre-formed: Spearman r=0.256 (p=0.0035), pooled ROC-AUC 0.740 for cryptic-vs-pre-formed
discrimination — real signal, clearly above the ROC-AUC-0.611 geometric proxy an earlier attempt (a local
atom-density delta, apo vs. holo) had already been rejected as too weak, but not strong enough to trust as a hard
per-chain relabeling: a threshold fit by Youden's J and evaluated by 5-fold cross-validation on the same 128 chains
reaches only 0.23 precision at 0.57 recall, a direct consequence of how rare true cryptic sites are (17/128, ~13%)
even in an already crypticity-enriched benchmark.

Used as a soft, continuous stratifier rather than a hard label — fitting one isotonic curve on the validation
chains this proxy scores as more cryptic-like, one on the rest, then applying each curve to test chains by their
*own* proxy score, never their true label — calibration improves on every metric and every truly-labeled stratum
against the pooled fit: cryptic-residue ECE 0.0212 to 0.0167, Brier 0.0315 to 0.0285; pre-formed ECE 0.0075 to
0.0048; pooled-overall ECE 0.0056 to 0.0035. This confirms crypticity carries real, usable calibration-relevant
information and closes the item flagged as open above. It is not, however, an improvement over the size-stratified
fit already adopted: the two land within noise of each other on every stratum (e.g. cryptic ECE 0.0167 for the
crypticity proxy vs. 0.0169 for size-stratification), and checking why shows the reason — pocket size and this
crypticity proxy are themselves highly correlated on the validation set (Spearman r=0.876), because the proxy counts
cavity voxels within 4 Å of the site residues, so a chain with more site residues mechanically tends to register a
larger counted cavity. The two stratifications are largely measuring the same axis of difficulty, not independent
ones. We report this as a genuine, if modest, positive result — the crypticity audit now extends to validation, and
crypticity-aware calibration does work — rather than a new practical recommendation: the simpler, already-deployed
size-stratified calibration captures essentially the same benefit without any extra geometry computation, so it
remains what we recommend.

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
larger, which is not what we expected of a redundant feature. A
per-dimension decomposition at 650M scale attributes most of it to one
specific property: H1 (loops/tunnels) is the only one of the four
things this project's topology block computes that is confirmed against
geometry alone on both PR-AUC and ROC-AUC, beats the other two homology
dimensions pairwise, and — combined with geometry — is statistically
indistinguishable from the full four-dimension block (see "Geometry
control" in Results). This is a satisfying, if post-hoc, mechanistic
story: a feature type describing loops and tunnels in the local atomic
point cloud is conceptually closer to a cavity's shape than H0 (mere
connectivity) or H2 (enclosed voids, sparser and noisier at this
neighborhood scale). The residual's absence on the smaller, differently
labeled 700-chain CryptoBench training set may reflect data volume, the
label definition, or the task itself.

Transfer learning from the larger, more diverse main-pipeline training pool
to CryptoBench's comparatively small 700-chain training set produced a
clear, multi-seed-confirmed gain, plausibly because the main pipeline's
much larger pool of general protein structures teaches more broadly useful
shared representations (residue embeddings, graph-convolution weights) that
CryptoBench's smaller, more specialized dataset alone cannot learn as
reliably.

What that gain actually means depends on which pockets are being counted,
and this is the most important interpretive correction this project made to
its own results. CryptoBench's 182-chain test set is only 9% genuinely
cryptic by a strict, independently applied definition (pocket essentially
absent in the unbound structure); the rest are already visible before
anything binds, or ambiguous. Every pooled number in this paper — the 0.918
AUC, the win over P2Rank, the transfer-learning gain — is therefore mostly a
statement about how well the model finds pockets in general, not
specifically about the harder, rarer, more clinically interesting case the
project set out to study. Restricting to the strictly cryptic subset drops
pooled ROC-AUC from 0.896 to 0.721, a pattern independently reported for
other published models by the source of that relabeling. We treat this as
the honest scope of the work rather than a caveat to bury: a model that is
very good at binding-site prediction in general, and specifically better
than a live run of the standard tool on the hardest, truly cryptic cases it
was pointed at, but not a solved instance of the cryptic-pocket problem as
originally framed. That the same ~0.72–0.76 ROC-AUC band reappears on two
datasets built by two other research groups, using two different
operational definitions of "cryptic" and structures our model never
trained on, is the strongest evidence in this paper that this number
reflects the actual difficulty of the task rather than a quirk of one
benchmark's construction — and it is a materially more useful thing to know
than the pooled headline figure alone would have suggested.

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

Once the geometry control and the crypticity audit narrowed the real
question to "can anything close the ~0.72–0.76 gap on genuinely cryptic
sites," this project tried five further, structurally different answers
before concluding the current architecture and input structure are near a
real ceiling rather than leaving obvious gains unclaimed. Per-residue
chemistry features (hydropathy, charge, aromaticity, hydrogen-bonding,
local hydrophobic-patch size — all computable from sequence and geometry
alone, with no ligand information) gave a result split exactly down the
middle by this project's own two-test rule: one test passed, one did not,
and the feature aimed most directly at hard cases (a hydrophobic-patch
measure) made the small-pocket subset slightly worse, not better.
Elastic-network-derived protein-movement features — a first-principles
attempt at the central intuition behind cryptic pockets, "does this
region's local packing loosen or tighten as the protein flexes along its
own natural motions" — returned a clean null on both the overall metric and
the small-pocket subset it was specifically designed to help. Explicitly
up-weighting small pockets in the training loss, addressing the observed
weak point directly (protein-level ROC-AUC 0.69 on pockets of 8 or fewer
residues versus 0.90–0.93 elsewhere), also failed to help and mildly hurt
overall accuracy at the stronger weighting tested. Once a validated
geometric crypticity proxy existed (the audit extension described in
Results, above), the same up-weighting scheme was retried with cryptic-scoring chains in
place of small-pocket chains as the target of the extra weight, on the
theory that they might be the more informative examples the model needed
to see more of — this reproduced the small-pocket result almost exactly
(no significant change on the true-cryptic test chains themselves, and a
stronger weighting again measurably hurt pooled performance, test PR-AUC
-0.0214, DCC@4Å -0.0391, both confirmed), consistent with the two proxies
being highly correlated (Spearman r=0.876) rather than independent signals.
Separately, extending
CryptoBench's 700-chain training set with 1,997 real, independently-labelled,
homology-clean apo/holo pairs from a second benchmark (same recipe, 8
seeds, no other change) made performance measurably *worse* (test PR-AUC
-0.0272, MCC -0.0253, both confirmed by this project's two-test rule) —
more correctly-labelled data, naively pooled across two differently-curated
benchmarks, is not a free win either. Five independent attempts, five
different mechanisms, the same outcome: with a single static apo structure
as the only input, there appears to be little further signal left to
extract with the feature types tried here.

The next most plausible remaining lever was a genuine conformational
ensemble rather than a linear elastic-network approximation of one, so we
ran a small pilot: real, GPU-accelerated, unbiased molecular dynamics
(OpenMM, 5 ns production per protein after 200 ps equilibration, apo
structure only, the single target chain, explicit solvent) on the 17
CryptoBench test proteins independently confirmed genuinely cryptic. Making
this work at all required solving an infrastructure problem first: neither
pip's OpenMM wheel (CPU-only, no CUDA platform; 0.17 ns/hour, 118 hours for
a 20 ns run) nor this compute platform's preinstalled package manager (which
turned out to share a name with, but not be, the intended tool) could
produce a working GPU build; a self-contained conda-forge install bootstrapped
from scratch found a real CUDA platform and reached 15 ns/hour, an ~88-fold
speedup and the only reason this pilot was feasible at all. Even so, the
platform's own session ceiling was the binding constraint in practice: of
the 17 proteins queued, 8 completed their full trajectory before the run
was cancelled after roughly 14 hours, an infrastructure limit beyond every
precedent this project had previously encountered. Per-residue Cα-RMSF
(rigid-body motion removed by iterative Kabsch alignment, verified on
synthetic data before use) and the change in local heavy-atom neighbour
count relative to the starting structure were checked, pooled across the
2,394 residues of the 8 completed proteins, for raw predictive power
against the true site-residue label — the same pre-commitment check used to
reject the earlier elastic-network approximation and a geometric crypticity
proxy elsewhere in this project. The result was weak to null: RMSF, the
single most standard MD flexibility metric, showed no discriminative power
at all (pooled AUC 0.500, exactly chance), and the neighbourhood-change
features were mixed and modest (pooled AUC 0.37-0.64 depending on which
summary statistic). With only 8 of the planned 17 proteins and roughly 14
GPU-hours already spent to get them, this pilot is underpowered to rule the
idea out definitively, but it gives no indication that scaling it up would
be a good use of the compute it would require, and we do not recommend
pursuing short unbiased MD further as this project's next step. Closing the
remaining gap most plausibly requires either much longer sampling or
enhanced-sampling methods this project did not have the infrastructure or
time budget to run, or information the input structure does not carry at
all regardless of how it is sampled — most concretely, information about
the binding partner rather than the protein alone.

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

The final and most important correction this project made to its own
result is a scope correction, not a performance one. An independent,
adversarially designed relabeling of what "cryptic" actually means showed
that the CryptoBench numbers above are overwhelmingly a statement about
pockets that are already visible before anything binds; on the roughly 9%
of test proteins that are genuinely, strictly cryptic, pooled ROC-AUC falls
from 0.896 to 0.721. The model still beats a live P2Rank run specifically
on that hard subset, and — tested directly against an external claim to
the contrary — geometry features do not appear to hurt performance there on
this architecture. Run unmodified on two datasets built entirely outside
this project, using two different research groups' own definitions of
"cryptic" and structures the model never trained on, performance lands at
0.925 (a set with molecular-dynamics-derived labels, matching or exceeding
that dataset's own published state of the art) and 0.757 (an independent,
adversarially constructed hard benchmark, homology-clean subset) — the
latter number statistically indistinguishable from the 0.721 the internal
relabeling found. Three checks, two of them genuinely external, agree: on
the specific case this project set out to address, the honest number is
roughly 0.72–0.76 ROC-AUC, not the 0.9-plus pooled figure that a
CryptoBench-only evaluation would otherwise suggest, and six further,
mechanistically distinct attempts to close that specific gap (chemistry
features, protein-movement features, small-pocket loss weighting, extra
training data from a second benchmark, a real molecular-dynamics pilot, and
— once a validated proxy made it possible to try — crypticity-proxy loss
up-weighting itself) each came back null, negative, or inconclusive —
evidence that this is a
real limit of what a single static structure can tell a model about a
pocket that does not yet exist, rather than an unclaimed modeling gain. One
genuine improvement did come from this final stretch of work, and it is
not an accuracy gain at all: the model's raw predicted probabilities turned
out to be badly overconfident (at its own highest-confidence decile, the
true hit rate is less than half what it claims), a problem no
discrimination metric in this paper would have caught, and a simple,
already-available calibration step fixes nearly all of it (expected
calibration error 0.166 to 0.0029) without touching any ranking-based
result reported elsewhere. Whether this model's numbers can be trusted and
whether it ranks correctly are different questions, and this project had
only ever checked the second one until this point. A follow-up built a
validated geometric proxy to extend the crypticity audit to the validation
split for the first time and confirmed that crypticity-aware calibration
also works, closing that specific open question — but it turned out to be
largely redundant with the pocket-size stratification already adopted, so
the practical recommendation did not change.

## Future Work

The geometry control, the crypticity audit, and the two external-validation
datasets close out this project's original open questions about whether the
topology gain is real (mostly not, beyond ordinary geometry) and whether the
result generalizes beyond CryptoBench (yes, to a consistent ~0.72–0.76
ROC-AUC on genuinely cryptic sites specifically, on two datasets this
project never trained on). What remains is a narrower, harder set of
questions this project could not close with the data and information
already on hand:

- **Close the ~0.72–0.76 ROC-AUC gap on genuinely cryptic sites with a
  different kind of information, not a different feature on the same
  input.** Six mechanistically distinct attempts — hand-crafted chemistry
  features, an elastic-network movement approximation, explicitly
  up-weighting small pockets in the loss, extending training with a second
  benchmark's data, a real (if underpowered, 8-of-17-protein) short
  unbiased molecular-dynamics pilot, and crypticity-proxy loss up-weighting
  — each returned a null, negative, or inconclusive result (see
  Discussion), which is real evidence against more features computed from
  a single static apo structure, or more of the same kind of training
  data, closing this gap. What remains untried is *properly
  powered and properly long* MD (this project's pilot ran 5 ns per protein,
  likely too short to sample genuine pocket-opening events, and stopped at
  8 of 17 proteins when the compute platform's own session limit was hit)
  or enhanced-sampling methods (metadynamics, mixed-solvent MD) that reach
  the relevant timescale far more cheaply than brute-force unbiased
  sampling — both would need dedicated infrastructure and compute budget
  this project did not have. (Crypticity-aware training up-weighting, the
  one item that used to be listed here as blocked on missing validation-
  split labels, has since been tried and closed — see Discussion — so it is
  no longer an open direction.)
- **Scale global (long-range) attention to the headline pipeline.** The
  one genuinely promising signal from this project's later architecture
  experiments (Individual levers, above) was a global self-attention
  block in the trunk, tested from scratch on CryptoBench at 650M scale:
  confirmed on test PR-AUC and ROC-AUC, consistent in direction on every
  other metric, though not confirmed on its own pre-specified primary
  (validation PR-AUC). Every other confirmed CryptoBench gain in this
  paper reached its strongest, most trustworthy form only after the same
  two steps this result has not yet had: transfer learning from a
  main-pipeline checkpoint trained with the same architecture change, and
  cross-scale ensembling. Both require retraining the much larger main
  pipeline from scratch with global attention added — a substantially
  larger compute commitment than the pilot reported here — before this
  lead could be pursued to the same standard as the rest of this paper's
  results.
- **(Resolved.)** Which specific topological property drives the confirmed
  classification residual beyond geometry was, until this point, answered
  only for the *localization* gain (H0 measurably weakest, everything else
  tied) and left open for classification. A per-dimension decomposition at
  650M scale — the scale where the classification residual is actually
  confirmed — found H1 (loops/tunnels) is the only dimension confirmed
  against geometry alone on both PR-AUC and ROC-AUC, beats H2 and burial
  pairwise, and geometry-plus-H1-alone is statistically indistinguishable
  from geometry plus the full four-dimension block (see "Geometry control"
  in Results). This is now a settled, if secondary, finding rather than an
  open question.
- **A larger cryptic-pocket benchmark.** CryptoBench's 60-chain validation
  and 182-chain test splits are small enough to leave the cross-scale
  ensemble's MCC gain statistically underpowered (a chain-level bootstrap
  found the AUC gain significant but the MCC gain not, for every combiner
  tested, including one designed to resist exactly this kind of
  validation-set overfitting), and small enough that only 16–17 test
  proteins are genuinely cryptic by a strict definition — the single
  biggest constraint on how precisely this project could characterize
  cryptic-specific performance internally. The two external datasets used
  for validation (35 and 99 proteins respectively) partially substitute for
  this but were built for a different original purpose and are themselves
  not large.
- **A dedicated cryptic-pocket localization benchmark.** The 20.9–24.7%
  DCC@4Å hit rate on CryptoBench's single-guess localization is far below
  the 50–65% range on open pockets. Two alternative localization-head
  designs have now been tried and both returned a null result on the
  metric they were built to improve — top-K score-based masking, and a
  genuinely local *spatial* attention mechanism penalizing distance from
  a preliminary center estimate (see "Individual levers," above) — so this
  is no longer an untried direction so much as a harder wall than either
  mechanism could move. Returning multiple candidate pockets rather than
  one, which this project tested and confirmed helps at looser distance
  thresholds (roughly nine points at 8 Å, little at 4 Å), remains the one
  approach to single-guess-adjacent localization that has actually worked;
  a dedicated benchmark large enough to distinguish finer localization
  architecture differences would be needed to tell whether a third
  mechanism could do better, or whether the gap is closer to this
  project's single-static-structure information ceiling (Discussion)
  reappearing in a different metric.
- **External validation on a larger scale.** The two outside datasets used
  here (35 and 99 proteins) were chosen because their labels were tractable
  to obtain and align to this project's exact input format; a systematic
  search for every other public cryptic-pocket dataset, and a combined
  external test set an order of magnitude larger, would tighten the
  confidence interval around the ~0.72–0.76 estimate considerably.
- **Package this project for reuse.** The full pipeline currently exists as
  a long, chronological sequence of individual experiments rather than one
  reproducible codebase; consolidating the final architecture, the
  homology-clean pretraining recipe, and the CryptoBench fine-tune into a
  single runnable pipeline — separate from the experimental record this
  paper reports on — would be needed before anyone else could use this
  model rather than only read about it.
