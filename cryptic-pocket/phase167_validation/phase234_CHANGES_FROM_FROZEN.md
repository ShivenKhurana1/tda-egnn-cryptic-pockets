# What changed from phase214_FINAL_MANUSCRIPT.md (frozen) to phase234_REVISED_MANUSCRIPT.md

The frozen manuscript and `phase216_submission_package_FINAL.zip` are untouched. Reasons for the revision, in order of importance:

1. **Central claim reversed.** The frozen text says TDA improves cryptic-pocket prediction. A 17-feature plain-geometry control
   (Phases 227/228/232/233) matches or beats TDA, and topology adds nothing detectable on top of geometry
   (geometry+persistence vs geometry: PR-AUC +0.0008, 95% CI -0.005..+0.007, p=0.79; same null on the official split, 8 seeds).
2. **Evaluation set mis-described.** The "predefined 185-protein experimental holdout" is the Phase 166 label-free embedding-cluster
   hard split (668/89/185), not the official CryptoBench split. The revision says so and states it is not a verified family split.
3. **Pocket-size result inflated.** Partial Spearman 0.756 (PR-AUC vs pocket size | protein size) is largely PR-AUC's dependence on
   prevalence; with chance-level control or ROC-AUC it is about 0.30-0.44 depending on model/control (Phase 225, 233).
4. **P2Rank baseline added** (Phase 226/233). Geometry models beat it modestly (ROC 0.796 vs 0.756); ESM+structure alone does not.
   P2Rank is comparable or better at localization when given 3 guesses.
5. **Dropped from the claims:** "TDA gain grows with pocket size" (not re-tested against geometry) and "TDA captures information not in
   ESM + structure" (true only relative to a baseline lacking geometry).
6. **Main pipeline (Phase 230, done).** Phase 11 nested CV re-run with the geometry control: geometry beats topology alone
   (+0.0129 PR-AUC, 10/10 folds); topology adds a small residual over geometry (+0.0097, 9/10 folds, t p=0.017, W p=0.037; ROC n.s.).
   So "topology adds nothing" holds for CryptoBench only. `paper.md` was updated accordingly (abstract, Methods, new Results
   subsection, Discussion, Conclusion, Future Work).
