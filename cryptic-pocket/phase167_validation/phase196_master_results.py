import os
import pandas as pd

BASE = "phase167_validation"

OUT = os.path.join(
    BASE,
    "phase196_master_results.csv"
)

rows = [

    # ---------------------------------------------------------------
    # Experimental residue validation
    # ---------------------------------------------------------------

    {
        "section": "experimental_residue_validation",
        "arm": "esm_structure",
        "metric": "protein_mean_PR_AUC",
        "value": 0.295203,
        "comparison": ""
    },

    {
        "section": "experimental_residue_validation",
        "arm": "full_tda",
        "metric": "protein_mean_PR_AUC",
        "value": 0.348529,
        "comparison": "vs ESM"
    },

    {
        "section": "experimental_residue_validation",
        "arm": "full_model",
        "metric": "protein_mean_PR_AUC",
        "value": 0.347575,
        "comparison": "vs ESM"
    },

    {
        "section": "experimental_residue_validation",
        "arm": "esm_structure",
        "metric": "protein_median_ROC_AUC",
        "value": 0.829078,
        "comparison": ""
    },

    {
        "section": "experimental_residue_validation",
        "arm": "full_tda",
        "metric": "protein_median_ROC_AUC",
        "value": 0.859539,
        "comparison": "vs ESM"
    },

    {
        "section": "experimental_residue_validation",
        "arm": "full_model",
        "metric": "protein_median_ROC_AUC",
        "value": 0.855311,
        "comparison": "vs ESM"
    },

    # ---------------------------------------------------------------
    # Experimental localization
    # ---------------------------------------------------------------

    {
        "section": "localization",
        "arm": "esm_structure",
        "metric": "gamma10_mean_center_error_A",
        "value": 10.158503,
        "comparison": ""
    },

    {
        "section": "localization",
        "arm": "full_tda",
        "metric": "gamma10_mean_center_error_A",
        "value": 10.066239,
        "comparison": ""
    },

    {
        "section": "localization",
        "arm": "full_model",
        "metric": "gamma10_mean_center_error_A",
        "value": 10.052842,
        "comparison": ""
    },

    {
        "section": "localization",
        "arm": "esm_structure",
        "metric": "gamma10_DCC10",
        "value": 0.631081,
        "comparison": ""
    },

    {
        "section": "localization",
        "arm": "full_tda",
        "metric": "gamma10_DCC10",
        "value": 0.635811,
        "comparison": ""
    },

    {
        "section": "localization",
        "arm": "full_model",
        "metric": "gamma10_DCC10",
        "value": 0.635811,
        "comparison": ""
    },

    # ---------------------------------------------------------------
    # Pocket-size failure mode
    # ---------------------------------------------------------------

    {
        "section": "pocket_size",
        "arm": "full_model",
        "metric": "partial_spearman_pocket_size_vs_PR_AUC",
        "value": 0.7558,
        "comparison": "controlled for protein size"
    },

    {
        "section": "pocket_size",
        "arm": "full_model",
        "metric": "partial_spearman_pocket_size_vs_center_error",
        "value": -0.6029,
        "comparison": "controlled for protein size"
    },

    {
        "section": "pocket_size",
        "arm": "full_tda",
        "metric": "partial_spearman_pocket_size_vs_PR_AUC",
        "value": 0.7536,
        "comparison": "controlled for protein size"
    },

    {
        "section": "pocket_size",
        "arm": "esm_structure",
        "metric": "partial_spearman_pocket_size_vs_PR_AUC",
        "value": 0.7051,
        "comparison": "controlled for protein size"
    },

    # ---------------------------------------------------------------
    # TDA gain
    # ---------------------------------------------------------------

    {
        "section": "tda_gain",
        "arm": "full_tda",
        "metric": "pocket_size_vs_PR_AUC_gain",
        "value": 0.366323,
        "comparison": "TDA vs ESM"
    },

    {
        "section": "tda_gain",
        "arm": "full_tda",
        "metric": "pocket_fraction_vs_PR_AUC_gain",
        "value": 0.343364,
        "comparison": "TDA vs ESM"
    },

    # ---------------------------------------------------------------
    # Case studies
    # ---------------------------------------------------------------

    {
        "section": "case_study",
        "arm": "full_model",
        "metric": "3q6u_PR_AUC",
        "value": 0.7618,
        "comparison": ""
    },

    {
        "section": "case_study",
        "arm": "full_model",
        "metric": "3q6u_gamma10_error_A",
        "value": 0.854976,
        "comparison": ""
    },

    {
        "section": "case_study",
        "arm": "full_model",
        "metric": "5ytb_PR_AUC",
        "value": 0.0338,
        "comparison": ""
    },

    {
        "section": "case_study",
        "arm": "full_model",
        "metric": "5ytb_gamma10_error_A",
        "value": 40.835433,
        "comparison": ""
    },

    {
        "section": "case_study",
        "arm": "full_model",
        "metric": "7yjc_PR_AUC",
        "value": 0.0135,
        "comparison": ""
    },

    {
        "section": "case_study",
        "arm": "full_model",
        "metric": "7yjc_gamma10_error_A",
        "value": 33.111639,
        "comparison": ""
    }
]

df = pd.DataFrame(rows)

df.to_csv(
    OUT,
    index=False
)

print("=" * 100)
print("PHASE 196 — MASTER RESULTS")
print("=" * 100)

print(
    f"\nRows: {len(df)}"
)

print(
    f"Saved: {OUT}"
)

print("\nMASTER RESULTS:\n")

print(
    df.to_string(index=False)
)

print("\n" + "=" * 100)
print("PHASE 196 COMPLETE")
print("=" * 100)
