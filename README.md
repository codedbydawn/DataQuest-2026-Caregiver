# DAMB — DataQuest 2026

DAMB is a hackathon-built machine learning web application for profiling burnout risk among unpaid caregivers in Canada. A caregiver submits their situation through a web form, the app routes the inputs through a locally saved clustering model, and the results page returns a caregiver profile, the main factors shaping that profile, and tailored Canadian support resources. The project uses Statistics Canada's General Social Survey on Caregiving and Care Receiving and is designed to support earlier visibility into caregiver strain.

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

## Folder Structure

```text
DataQuest-2026-DAMB/
├── src/                  # Source SAS files and existing notebooks; do not modify
├── data/
│   └── processed/        # Cleaned and joined CSV outputs from preprocessing
├── model/                # Saved model artifacts such as model.pkl
├── app/
│   ├── app.py            # Flask entry point
│   ├── templates/
│   │   ├── index.html    # Caregiver input form
│   │   └── results.html  # Profile results page
│   └── static/
│       └── style.css     # Custom styling overrides
├── requirements.txt      # Python dependencies across data, ML, and app phases
└── README.md
```

## Data Notes

- The source data comes from the 2018 General Social Survey (GSS) Cycle 32: Caregiving and Care Receiving Public Use Microdata File.
- The survey target population is non-institutionalized people aged 15 and older living in the 10 provinces of Canada.
- The repository includes two SAS files in `src/` that should remain untouched:
  - `maindata.sas7bdat`: person-level respondent data
  - `episode.sas7bdat`: episode-level activity data
- The two files share `PUMFID`, which is the common identifier available in both files.
- The person-level file includes `WGHT_PER`, while the episode-level file includes `WGHT_EPI`. The team should use the appropriate survey weight during analysis and reporting.
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

### Flask App (caregiver form)

From the `app/` directory:

```bash
flask run
```

## Notes

- The model must be trained in the notebooks and saved to `model/model.pkl` before the Flask app can run end-to-end.
- `pyreadstat` is required to read the SAS `.sas7bdat` files in `src/`.
- All inference is intended to run locally from the saved model artifact. No external ML APIs are used.
- The preprocessing notebooks will need to decide how to aggregate episode-level records into person-level features before training the clustering model.
- The Statistics Canada user guide recommends weighted analysis for estimates; unweighted survey summaries should be treated with caution.
- The current frontend is a placeholder scaffold. Input fields, wording, and submission flow may change once the final feature set is confirmed.
- The five caregiver profiles listed above are the current intended product framing, but naming, descriptions, and cluster mapping may still change after cleaning the data and validating the model outputs.
