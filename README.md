# DAMB — Caregiver Distress Risk Pipeline

DAMB is a healthcare/wellness hackathon repo for estimating caregiver distress risk from the 2018 Statistics Canada General Social Survey (Cycle 32: Caregiving and Care Receiving) public-use microdata file. The repo now ships one coherent modeling path: a weighted binary XGBoost model for an analyst-defined `distress_flag`, plus a Flask demo that scores a single caregiver row and explains the prediction with top Tree SHAP contributors.

## What This Repo Now Does
- Loads the real raw PUMF in [`/Users/anuda/Desktop/DataQuest-2026-DAMB/src/c32pumfm.sas7bdat`](/Users/anuda/Desktop/DataQuest-2026-DAMB/src/c32pumfm.sas7bdat)
- Applies reserve-code recoding without collapsing off-path skips into false negatives
- Uses the codebook-faithful target based on:
  - `CRH_20` worried/anxious
  - `CRH_30` overwhelmed
  - `CRH_60` depressed
- Fits a weighted binary XGBoost classifier using `WGHT_PER`
- Saves model, metrics, validation summary, and global feature importance artifacts
- Exposes a Flask demo for single-row scoring

## What It Explicitly Does Not Do
- It does not use the notebook’s old burnout composite
- It does not use clustering
- It does not claim to predict a true 4-tier distress class
- It does not use bootstrap weights as ordinary predictors

## Repo Layout

```text
DataQuest-2026-DAMB/
├── app/                     # Flask demo for single-row scoring
├── damb/                    # Audited data, modeling, and scoring modules
├── docs/                    # Modeling spec and limitations
├── model/                   # Trained model artifact and metrics
├── reports/                 # Validation and global feature-importance outputs
├── scripts/                 # Training entrypoints
├── src/
│   ├── c32pumfm.sas7bdat    # Raw Statistics Canada PUMF
│   └── eda.ipynb            # Historical notebook, not the production pipeline
└── tests/                   # Focused regression tests
```

## Modeling Summary

### Universe approximation
The official codebook universe for the target items uses raw `HAP_10`, but this PUMF exposes only grouped `HAP_10C`. The shipped approximation is:
- `DV_PROXY == 2`
- `PAR_10 in 1..99`
- `HAP_10C in 1..6`

Rows with `HAP_10C == 1` remain in scope because valid target responses exist there. Filtering to `HAP_10C >= 2` would be wrong.

### Target
- `distress_score = count_yes(CRH_20, CRH_30, CRH_60)`
- `distress_flag = 1 if distress_score >= 1 else 0`

### Predictors
The predictor set is intentionally limited to non-leaky demographics, employment, caregiving intensity, activity flags, support network, financial context, and respondent health variables. Leakage-prone variables such as `ICS_*`, `FIS_*`, `CRH_*`, and downstream caregiving-consequence variables are excluded.

### Weights and explanations
- `WGHT_PER` is used as the person-level analysis weight
- `WTBS_*` are preserved for survey-design sensitivity work, but not used as predictors
- Feature explanations use XGBoost Tree SHAP contributions via `pred_contribs=True`

## Install

```bash
python -m pip install -r requirements.txt
```

## Train The Model

```bash
python scripts/train_binary_model.py
```

This writes:
- [`/Users/anuda/Desktop/DataQuest-2026-DAMB/model/caregiver_distress_model.joblib`](/Users/anuda/Desktop/DataQuest-2026-DAMB/model/caregiver_distress_model.joblib)
- [`/Users/anuda/Desktop/DataQuest-2026-DAMB/model/model_metrics.json`](/Users/anuda/Desktop/DataQuest-2026-DAMB/model/model_metrics.json)
- [`/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/validation_summary.json`](/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/validation_summary.json)
- [`/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/global_shap_importance.csv`](/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/global_shap_importance.csv)
- [`/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/feature_missingness.csv`](/Users/anuda/Desktop/DataQuest-2026-DAMB/reports/feature_missingness.csv)

## Run Tests

```bash
pytest
```

## Run The Demo App

```bash
flask --app app.app run
```

The demo form captures a subset of high-signal inputs and leaves all other expected model features as missing. The scorer still runs because the trained preprocessing pipeline handles explicit missing categories and numeric `NaN`s consistently.

## Notes
- The authoritative implementation details live in [`/Users/anuda/Desktop/DataQuest-2026-DAMB/docs/modeling_spec.md`](/Users/anuda/Desktop/DataQuest-2026-DAMB/docs/modeling_spec.md)
- The Flask app is a hackathon demo, not a clinical device
- The model output is best interpreted as caregiver-distress risk probability under this analyst-defined labeling scheme
