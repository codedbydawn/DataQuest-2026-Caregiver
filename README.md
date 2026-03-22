# DataQuest 2026

Our Group built a Caregiver Burnout Risk Screener. It is a hackathon-built machine learning web application for profiling burnout risk among unpaid caregivers in Canada. Caregiver data is inputted into the app and is routed through a locally saved clustering model, and the results page returns a caregiver profile, the main factors shaping that profile, and tailored Canadian support resources. The project uses Statistics Canada's General Social Survey on Caregiving and Care Receiving and is designed to support earlier visibility into caregiver strain.

This repository currently contains an architecture scaffold only. The frontend flow, form fields, and caregiver profile framing may change as the team completes data cleaning, feature engineering, exploratory analysis, and final clustering validation.

## Caregiver Profiles

### Profile 1 — Solo Caregiver, High Risk
No support network, high hours, employment affected, financially strained. Highest burnout risk. Requires immediate intervention.

### Profile 2 — Overburdened Balancer
Moderate hours with inconsistent support. Managing work and caregiving simultaneously with no reliable relief. Elevated and escalating burnout risk.

### Profile 3 — Supported but Fatigued
Consistent support network present, higher hours, but fatigue is accumulating over time. Moderate burnout risk with a clear trajectory if support is not maintained.

### Profile 4 — High Emotional Load, Low Visibility
Lower caregiving hours but severe psychological impact. Often invisible in standard burnout assessments because hours appear manageable. Moderate to high risk.

### Profile 5 — Stable Long-Term Caregiver
Long caregiving duration, adapted routines, reliable support. Lowest acute burnout risk but should be monitored for cumulative fatigue.

## Tech Stack

- Python
- Flask / Streamlit
- pandas, NumPy, SciPy
- scikit-learn, XGBoost
- SHAP
- pyreadstat
- joblib
- matplotlib, seaborn, Plotly
- Jupyter notebooks
- Statistics Canada GSS Cycle 32 PUMF documentation

## Data Notes

- The source data comes from the 2018 General Social Survey (GSS) Cycle 32: Caregiving and Care Receiving Public Use Microdata File.
- The survey target population is non-institutionalized people aged 15 and older living in the 10 provinces of Canada.
- The repository includes a SAS file in `src/` that should remain untouched:
  - `c32pumfm.sas7bdat`: person-level respondent data
- According to the user guide, GSS variables are limited to 8 characters or fewer and reserve codes include `6` for valid skip and `9` for not stated.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Analysis

### Report (text/markdown output)

Generates a full markdown report at `results/analysis_report.md` and prints it to the terminal. This is the easiest way to review the pipeline results or feed them to another LLM.

```bash
python src/report.py
```

### Dashboard (interactive Streamlit app)

Launches an interactive browser dashboard with charts, SHAP plots, cluster profiles, and diagnostics.

```bash
streamlit run src/dashboard.py
```

> **Note:** Do not run `python src/dashboard.py` directly — Streamlit apps must be launched with the `streamlit run` command.


## Notes

- `pyreadstat` is required to read the SAS `.sas7bdat` files in `src/`.
- All inference is intended to run locally from the saved model artifact. No external ML APIs are used.
