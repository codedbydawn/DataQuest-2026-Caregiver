# Caregiver Distress Model Report

## Final system
- Stage 1: weighted XGBoost model predicting the analyst-defined `distress_flag`
- Stage 2: dashboard-ready scored dataset, explainability outputs, subgroup summaries, and resource-allocation artifacts

## Verified assumptions used
- Raw file: `src/c32pumfm.sas7bdat`
- Target: `CRH_20`, `CRH_30`, `CRH_60`
- Universe approximation: `DV_PROXY == 2`, `PAR_10 in 1..99`, `HAP_10C in 1..6`
- Local PDF codebook/user-guide files were not present in the workspace during implementation, so final checks were anchored to repo contents plus raw-column inspection

## Analytic sample
- Raw rows: 20258
- Final analytic rows: 5587
- Weighted prevalence: 0.6908
- Unweighted prevalence: 0.6975

## Final performance
- Validation ROC AUC: 0.7524
- Validation PR AUC: 0.8593
- Test ROC AUC: 0.7148
- Test PR AUC: 0.8429
- Test weighted accuracy: 0.6965
- Test weighted Brier score: 0.1790
- Selected threshold: 0.625

## Outputs
- Artifact manifest: `/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/artifact_manifest.json`
- Scored dataset: `/Users/anuda/Desktop/DataQuest-2026-DAMB/data/processed/scored_caregiver_distress.csv`
- SHAP importance table: `/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/tables/global_shap_importance.csv`
- Highest-risk segments: `/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/tables/highest_risk_segments.csv`
- Dashboard app: `streamlit run app/app.py`

## Limitations
- The PUMF exposes grouped `HAP_10C` rather than raw `HAP_10`, so the official target universe is approximated
- Survey design bootstrap weights are not used in the predictive model itself
- Several narrow-universe health/detail variables were excluded from deployment to avoid brittle modeling with extreme missingness
- This output estimates analyst-defined distress risk, not a clinical diagnosis
