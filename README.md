# DAMB — Caregiver Distress Analytics System

This repo now ships a 2-stage caregiver distress analytics product for organizations such as governments, health systems, and nonprofits:

1. a weighted XGBoost model that predicts individual caregiver distress risk
2. a dashboard-oriented analytics layer that turns those predictions into SHAP explainability outputs, subgroup risk views, geography summaries, and resource-allocation tables

The project is no longer centered on a caregiver survey front end, caregiver profile classes, or clustering-based personas.

## Data and target
- Raw file: [`/Users/anuda/Desktop/DataQuest-2026-DAMB/src/c32pumfm.sas7bdat`](/Users/anuda/Desktop/DataQuest-2026-DAMB/src/c32pumfm.sas7bdat)
- Main target items:
  - `CRH_20`
  - `CRH_30`
  - `CRH_60`
- Derived fields:
  - `distress_score = count_yes(CRH_20, CRH_30, CRH_60)`
  - `distress_flag = 1 if distress_score >= 1 else 0`

This is an analyst-defined distress-risk label, not a clinical diagnosis.

## Universe approximation
The official CRH-item universe references raw `HAP_10`, but the PUMF in this repo exposes only grouped `HAP_10C`. The implemented modeling universe is:
- `DV_PROXY == 2`
- `PAR_10 in 1..99`
- `HAP_10C in 1..6`
- non-missing target-item responses

Rows with `HAP_10C == 1` are intentionally kept because valid target responses are present there.

## Modeling design
- Survey reserve / nonresponse codes are explicitly recoded to `NaN`
- Yes/no predictors are explicitly recoded to `1/0`
- Numeric and ordinal features keep missing values so XGBoost can route them
- Nominal features use deterministic impute-plus-one-hot preprocessing
- Leakage-prone prefixes such as `CRH_*`, `ICS_*`, `FIS_*`, `ICL_*`, `ICB_*`, `ICP_*`, `ITL_*`, `ITO_*`, `WLB_*`, and `EMO_*` are blocked from the feature set
- `WGHT_PER` is used as the sample weight

## Stage 2 outputs
Training writes organization-facing artifacts including:
- scored caregiver dataset with predicted probability, label, and risk band
- held-out test predictions
- global SHAP importance table
- SHAP summary, feature importance, risk distribution, calibration, threshold tradeoff, subgroup comparison, and heatmap figures
- subgroup risk summaries
- highest-risk segment and highest-risk individual tables
- subgroup-specific top-driver tables
- artifact manifest and model report markdown

## Repo layout

```text
DataQuest-2026-DAMB/
├── app/                     # Streamlit analytics dashboard
├── damb/                    # Data prep, modeling, scoring, and reporting modules
├── data/processed/          # Scored outputs
├── docs/                    # Modeling spec and generated report
├── model/                   # Serialized model and metrics
├── reports/                 # Figures, tables, validation summaries
├── scripts/                 # Training entrypoint
├── src/
│   ├── c32pumfm.sas7bdat    # Raw Statistics Canada PUMF
│   └── eda.ipynb            # Historical notebook, not the deployed system
└── tests/                   # Focused regression tests
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Train

```bash
python scripts/train_binary_model.py
```

Key outputs land in:
- [`/Users/anuda/Desktop/DataQuest-2026-DAMB/model`](/Users/anuda/Desktop/DataQuest-2026-DAMB/model)
- [`/Users/anuda/Desktop/DataQuest-2026-DAMB/reports`](/Users/anuda/Desktop/DataQuest-2026-DAMB/reports)
- [`/Users/anuda/Desktop/DataQuest-2026-DAMB/data/processed`](/Users/anuda/Desktop/DataQuest-2026-DAMB/data/processed)

## Latest verified run
- Final analytic sample: `5,587`
- Weighted prevalence: `0.6908`
- Validation ROC AUC / PR AUC: `0.7524 / 0.8593`
- Held-out test ROC AUC / PR AUC: `0.7148 / 0.8429`
- Held-out test weighted accuracy: `0.6965`
- Selected operating threshold: `0.625`

## Dashboard

```bash
streamlit run app/app.py
```

The dashboard reads saved artifacts and focuses on organization-facing decision support rather than caregiver self-assessment.

## Tests

```bash
pytest
```

## Caveats
- No local PDF codebook/user-guide files were present in the workspace during implementation, so the final system was verified directly against repo contents and raw-column inspection
- The PUMF contains grouped `HAP_10C`, not raw `HAP_10`
- Bootstrap survey weights are not used as predictors
- Historical notebook logic remains for reference only and does not define the deployed target
