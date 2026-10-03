"""Assembles benchmark_data.json: our model's numbers computed here (cryptic_head2head.json, ours_macro_pooled.json) plus competitor numbers
transcribed from the cited sources. Every external number carries its source; nothing is estimated."""
import json
h = json.load(open("cryptic_head2head.json")); m = json.load(open("ours_macro_pooled.json"))
rows = {r["model"]: r for r in h["rows"]}
def A(name, key, color, role):
    r = rows[name]; return dict(model=key, color=color, role=role, auc=r["auc"], ci=r["auc_ci"], p20=r["precision@20"], r20=r["recall@20"], hit10=r["success@10"], hit20=r["success@20"])
cryptic = [
  A("this work (TDA-EGNN, 3B transfer)", "TDA-EGNN (this work)", "c1", "ours"),
  A("this work (ESM-free)", "Benchmark authors’ ESM-free model", "c2", "preprint; also uses every deposited X-ray structure of the protein"),
  A("P2Rank 2.5", "P2Rank 2.5", "c3", "published tool"),
  A("ESM-based", "ESM-based model (CryptoBench-style)", "c4", "benchmark authors’ reimplementation"),
  A("interpretable (cavity)", "Cavity-feature model", "c5", "benchmark authors’ model"),
]
D = dict(
 cryptic=dict(n=h["common"].__len__(), proteins=h["common"], models=cryptic, preformed_ours=0.938,
   paired_vs_p2rank=dict(p20=dict(d=0.125, ci=[0.038, 0.222], t=0.020, w=0.022), r20=dict(d=0.187, ci=[0.066, 0.315], t=0.013, w=0.017), auc=dict(d=0.019, ci=[-0.103, 0.140], t=0.774, w=0.860)),
   paired_vs_authors=dict(p20=dict(d=0.025, ci=[-0.059, 0.122], t=0.611, w=0.789), auc=dict(d=-0.016, ci=[-0.143, 0.108], t=0.807, w=1.000))),
 pooled=[
  dict(m="TDA-EGNN, cross-scale combiner (this work)", set="182 chains · 54,992 residues · 4.72% positive", auc=.918, pr=.448, mcc=.470, ours=True, note="8 seeds at each of 3 ESM2 scales; threshold tuned on validation"),
  dict(m="TDA-EGNN, 3B transfer (this work)", set="182 chains · 54,992 residues · 4.72% positive", auc=.911, pr=.431, mcc=.466, ours=True, note="8-seed mean of single models"),
  dict(m="Seq2Pocket (bioRxiv preprint, May 2026)", set="184 chains · 55,572 residues · 5.68% positive", auc=.894, pr=.475, mcc=.460, note="MCC 0.473 without its smoothing step. Higher positive rate lifts AUPRC, so its AUPRC is not directly comparable", src="seq2pocket"),
  dict(m="CryptoBench model, re-evaluated by Seq2Pocket", set="same 184 chains", auc=.871, pr=.382, mcc=.395, note="the CryptoBench baseline as scored in Seq2Pocket’s own evaluation", src="seq2pocket"),
  dict(m="CryptoBench model, as published (CB-full)", set="CryptoBench full test split", auc=.86, pr=.36, mcc=.39, note="ESM2-3B + neural network", src="cryptobench"),
  dict(m="P2Rank on apo structures (CB-P2RANK-apo)", set="single-chain apo subset", auc=.81, pr=.21, mcc=.27, note="general pocket tool, not cryptic-specific", src="cryptobench"),
  dict(m="PocketMiner (CB-PM)", set="subset; 60 structures could not be scored", auc=.76, pr=.19, mcc=.22, note="structure-based cryptic-site model", src="cryptobench"),
  dict(m="P2Rank on holo structures (CB-P2RANK-holo)", set="ligand-bound structures", auc=.89, pr=.34, mcc=.38, note="uses the bound structure, so not an apo prediction; shown for reference", src="cryptobench"),
 ],
 macro=[
  dict(m="TDA-EGNN (this work)", set="182 proteins", auc=m["macro"]["roc"], pr=m["macro"]["pr"], mcc=m["macro"]["mcc"], f1=m["macro"]["f1"], ours=True, note="threshold tuned on validation"),
  dict(m="UniPocket (ACM-BCB 2026)", set="193 proteins", auc=.816, pr=.410, mcc=None, f1=None, note="AUC re-confirmed as macro-averaged; AUPRC as cited in the paper draft", src="unipocket"),
  dict(m="Algebraic Counting Field (arXiv, Oct 2026)", set="192 units", auc=.799, pr=.376, mcc=.304, f1=.333, note="per-unit average; accepted to PSB 2027", src="acf"),
  dict(m="P2Rank 2.5, same protocol as above", set="192 units", auc=.793, pr=.358, mcc=.285, f1=.302, note="reported in the same paper", src="acf"),
 ],
 pocketminer=[
  dict(m="PocketMiner", set="its own test set, 35 proteins", auc="0.87", pr="—", note="value as reported in its paper"),
  dict(m="TDA-EGNN (this work)", set="same set, 34 usable proteins, pooled", auc="0.925", pr="0.897", ours=True, note="0.920 on the 6 proteins with no relative in training data"),
  dict(m="ProMoSite", set="19-pair filtered subset only", auc="0.91", pr="0.84", note="different subset, so not comparable", src="promosite"),
  dict(m="AE-PocketMiner (bioRxiv, May 2026)", set="PocketMiner test set", auc="not retrieved", pr="not retrieved", note="reported to beat PocketMiner; the numbers could not be read from the preprint", src="aepm"),
 ],
 sources=dict(
  cryptobench=("CryptoBench, Bioinformatics 2025 (Table 5; GitHub README)", "https://github.com/skrhakv/CryptoBench/"),
  seq2pocket=("Seq2Pocket, bioRxiv 2026 (Tables 1–3)", "https://www.biorxiv.org/content/10.64898/2026.01.28.702257v2"),
  crypticity=("A benchmark selected for crypticity (preprint + results/ tables)", "https://github.com/pmfran4252-afk/cryptic-site-benchmark"),
  unipocket=("UniPocket, ACM-BCB 2026", "https://doi.org/10.1145/3807503.3819482"),
  acf=("Algebraic Counting Field, arXiv:2610.00988", "https://arxiv.org/abs/2610.00988"),
  promosite=("ProMoSite (PMC)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC13371367/"),
  aepm=("AE-PocketMiner, bioRxiv 2026", "https://www.biorxiv.org/content/10.64898/2026.05.21.726899v1"),
  pocketminer=("PocketMiner, Nat. Commun. 2023", "https://doi.org/10.1038/s41467-023-36699-3")))
json.dump(D, open("benchmark_data.json", "w"), indent=1)
import csv
with open("benchmark_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["tier", "method", "test_set", "auc", "auprc_or_pr", "mcc", "source"])
    for r in D["pooled"]: w.writerow(["pooled residue-level, CryptoBench", r["m"], r["set"], r["auc"], r["pr"], r["mcc"], r.get("src", "this work")])
    for r in D["macro"]: w.writerow(["per-protein macro, CryptoBench", r["m"], r["set"], round(r["auc"], 3), round(r["pr"], 3), r["mcc"], r.get("src", "this work")])
    for r in cryptic: w.writerow(["strictly cryptic (16 matched proteins), per-protein AUC", r["model"], "16 proteins", r["auc"], "P@20=%s R@20=%s" % (r["p20"], r["r20"]), "", "crypticity" if r["color"] != "c1" else "this work"])
print("ok", D["cryptic"]["n"])
