# Caregiver Distress Model Report

## Final system
- Stage 1: weighted XGBoost model predicting the analyst-defined `distress_flag`
- Stage 2: dashboard-ready scored dataset, explainability outputs, subgroup summaries, and resource-allocation artifacts

## Verified assumptions used
- Raw file: `src/c32pumfm.sas7bdat`
- Target: `CRH_20`, `CRH_30`, `CRH_60`
- Universe approximation: `DV_PROXY == 2`, `PAR_10 in 1..99`, `HAP_10C in 1..6`
- Valid skip values are treated as off-path / structural missingness rather than substantive "No"

## Analytic sample
- Raw rows: 20258
- Final analytic rows: 5587
- Weighted prevalence: 0.6908
- Unweighted prevalence: 0.6975

## Final performance
- Validation ROC AUC: 0.7527
- Validation PR AUC: 0.8597
- Test ROC AUC: 0.7311
- Test PR AUC: 0.8556
- Test weighted accuracy: 0.6967
- Test weighted Brier score: 0.1811
- Selected threshold (balanced mode): 0.650
- Threshold modes: {'balanced': 0.6499999999999999, 'high_precision': 0.8, 'high_recall': 0.575}
- Selected missing-data strategy: native
- Selected feature count: 40

## Outputs
- Artifact manifest: `/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/artifact_manifest.json`
- Scored dataset: `/Users/anuda/Desktop/DataQuest-2026-DAMB/data/processed/scored_caregiver_distress.csv`
- SHAP importance table: `/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/tables/global_shap_importance.csv`
- Highest-risk segments: `/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/tables/highest_risk_segments.csv`
- Calibration report: `/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/calibration_report.json`
- Feature audit: `/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/tables/feature_audit.csv`
- Fold metrics: `/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/tables/fold_metrics.csv`
- Subgroup diagnostics: `/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/tables/subgroup_diagnostics.csv`
- Dashboard app: `streamlit run app/app.py`

## Limitations
- The PUMF exposes grouped `HAP_10C` rather than raw `HAP_10`, so the official target universe is approximated
- Calibration and threshold choice are optimized for stakeholder-facing ranking and triage, not for causal interpretation
- Several work and receiver-detail variables carry structural missingness because of survey routing
- This output estimates analyst-defined distress risk, not a clinical diagnosis
