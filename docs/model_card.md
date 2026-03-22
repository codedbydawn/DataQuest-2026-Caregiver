# Model Card

## Intended use
- Stakeholder-facing caregiver distress risk ranking and subgroup concentration review.
- Appropriate for organizational triage, planning, and dashboarding.
- Not appropriate for clinical diagnosis, causal claims, or automated adverse decisions.

## Model summary
- Model family: XGBoost binary classifier with post-hoc sigmoid calibration.
- Operating modes: balanced, high_precision, high_recall.
- Default dashboard threshold: 0.500 (balanced mode).
- Selected missing-data strategy: unknown.

## Data summary
- Raw rows: 30
- Analytic rows: 24
- Weighted prevalence: 0.3500

## Test performance
- ROC AUC: 0.7000
- PR AUC: 0.6000
- Weighted accuracy: 0.6500
- Precision / Recall / Specificity: nan / nan / nan
- Weighted Brier score: 0.2100
- Log loss: nan

## Key limitations
- The official CRH universe uses raw HAP_10, but the PUMF only exposes grouped HAP_10C.
- Valid skips are structural and can induce heavy routed missingness.
- The dataset is an all-respondent survey with routed caregiving and care-receiving sections.
- Model explanations reflect model behavior, not causality.
