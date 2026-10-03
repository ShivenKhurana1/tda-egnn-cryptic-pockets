[DRAFT for author review — not yet the official submission. Word count noted at the end; trim names/affiliations as your fair's form requires.
Per Society for Science rules: no citations, no acknowledgments, no institution/mentor names, no self-promotion. Fill in your name(s) and category below.]

---

**Title:** Predicting Cryptic Ligand-Binding Sites from Apo Protein Structures with Equivariant Graph Neural Networks: A Controlled Evaluation of Persistent-Homology Features

**Student(s):** [your name]
**Category:** [your category, e.g. Biomedical Engineering / Computational Biology & Bioinformatics]

---

Roughly a fifth of drug-binding pockets are "cryptic": invisible in a protein's resting shape, only appearing once a drug-like molecule binds, and hard for existing tools to find. This project tested whether adding a shape-description technique from topological data analysis (persistent homology) to a 3D-aware neural network improves prediction of which parts of a protein form a pocket and where its center is, and — more importantly — checked how much of any improvement holds up under strict scrutiny.

A graph neural network was trained on protein structures, one node per amino acid, combining language-model embeddings, geometric shape features, and topological features. Adding topological features improved pocket-finding accuracy on standard and cryptic-pocket benchmarks. A designed control test then showed that simpler geometric measurements (surface exposure, local shape, burial depth) explained most of that improvement; the topological method's own contribution was small and inconsistent once a larger language model was used. The model's real gain instead came from reusing patterns learned on a larger, general protein dataset (transfer learning), and it beat a widely used existing tool on both pocket-finding and pocket-center localization.

A further check showed most of the project's headline accuracy came from easier pockets already visible before binding; on strictly cryptic pockets, accuracy was lower. That lower number, roughly three correct out of four, was confirmed on two independent datasets the model had never seen, built by other groups using different definitions of "cryptic" — evidence this reflects a real difficulty of the problem, not one benchmark's quirk.

---
[Word count of the body text above (excluding title/student/category lines): ~250 words — trim as needed to fit your fair's exact form.]
