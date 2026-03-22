# Caregiver Distress Modeling Spec

## Product goal
This repo now implements a 2-stage caregiver distress analytics system for organizational decision support:
- Stage 1: weighted XGBoost prediction of analyst-defined `distress_flag`
- Stage 2: dashboard-ready scored outputs, explainability, subgroup risk concentration views, and resource-allocation tables

## Verified data reality
- Raw file present: `src/c32pumfm.sas7bdat`
- Installed stack verified locally during implementation: `xgboost==3.0.5`, `scikit-learn==1.6.1`, `streamlit==1.45.1`
- Local PDF codebook/user-guide files were not present in the workspace at implementation time, so critical assumptions were re-verified directly against raw columns and distributions
- `WGHT_PER` is used as the person-level analysis weight
- `WTBS_*` are excluded from prediction

## Target definition
- Source items: `CRH_20`, `CRH_30`, `CRH_60`
- Recoding: `1 -> 1`, `2 -> 0`, reserve/nonresponse codes -> `NaN`
- `distress_score = count_yes(CRH_20, CRH_30, CRH_60)` with null preserved if all items are missing
- `distress_flag = 1 if distress_score >= 1 else 0`
- The target is an analyst-defined distress risk label, not a clinical diagnosis

## Universe logic
Official codebook logic references raw `HAP_10`, but this PUMF contains only grouped `HAP_10C`. The implemented approximation is:
- `DV_PROXY == 2`
- `PAR_10 in 1..99`
- `HAP_10C in 1..6`
- non-missing analytic responses on the target items

Rows with `HAP_10C == 1` are intentionally kept because valid CRH responses exist there.

## Final predictor philosophy
Deployed predictors are limited to non-leaky circumstance/context variables covering:
- demographics and geography
- household structure and caregiving relationship context
- caregiving intensity
- care receiver proximity and dwelling context
- employment and flexibility context
- caregiving task pattern
- financial context
- immigration, language, and transportation access context
- caregiver health context

Borderline or extremely sparse narrow-universe variables were not deployed if they would make the model brittle or look co-determined with the outcome. The final context expansion was pruned to retain broadly populated objective fields and drop the sparsest off-path items.

## Leakage exclusions
The pipeline blocks:
- `CRH_*`
- `ICS_*`
- `FIS_*`
- `ICL_*`
- `ICB_*`
- `ICP_*`
- `ITL_*`
- `ITO_*`
- `WLB_*`
- `EMO_*`

## Cleaning and encoding
- Reserve/special codes are explicitly mapped to `NaN`
- Binary yes/no features are explicitly recoded to `1/0`
- Numeric and ordinal fields keep `NaN` so XGBoost can route missing values
- Nominal categoricals use deterministic impute-plus-one-hot preprocessing with a visible missing category
- No target imputation is performed

## Anti-overfitting controls
- Train/validation/test split is created before fitting transforms
- Model selection uses bounded 3-fold CV on the train/validation pool
- Early stopping is used on a proper validation split
- Sparse category growth is controlled with `OneHotEncoder(min_frequency=20)`
- Train-vs-validation gaps are reported
- A top-feature dominance sanity check is saved with the validation summary

## Outputs
The pipeline writes:
- model artifact and metrics JSON
- validation summary JSON
- scored caregiver dataset
- held-out test predictions
- global SHAP importance table
- SHAP summary / feature importance / calibration / threshold / subgroup / heatmap figures
- subgroup risk summary
- highest-risk segments and individuals tables
- subgroup-specific top-driver table

## Dashboard
The app is a Streamlit analytics dashboard reading saved artifacts from disk. It is designed for governments, health systems, and nonprofits reviewing concentrated distress risk rather than for caregiver self-assessment.
