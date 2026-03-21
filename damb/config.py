from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "src" / "c32pumfm.sas7bdat"
MODEL_DIR = ROOT / "model"
REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_ARTIFACT_PATH = MODEL_DIR / "caregiver_distress_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
VALIDATION_PATH = REPORTS_DIR / "validation_summary.json"
GLOBAL_SHAP_PATH = REPORTS_DIR / "global_shap_importance.csv"
MISSINGNESS_PATH = REPORTS_DIR / "feature_missingness.csv"

TARGET_COLUMNS = ("CRH_20", "CRH_30", "CRH_60")
TARGET_ITEM_LABELS = {
    "CRH_20": "Worried or anxious",
    "CRH_30": "Overwhelmed",
    "CRH_60": "Depressed",
}
WEIGHT_COLUMN = "WGHT_PER"

FEATURE_GROUPS = {
    "demographics": [
        "AGEGR10",
        "SEX",
        "MARSTAT",
        "PRV",
        "LIVARR08",
    ],
    "employment": [
        "COW_10",
        "WTI_110",
        "NWE_110",
        "UHW_16GR",
        "UCA_10",
        "FWA_134",
        "FWA_137",
    ],
    "caregiving_intensity": [
        "HAP_10C",
        "PAR_10",
    ],
    "activity_flags": [
        "APR_10",
        "APR_20",
        "APR_30",
        "APR_40",
        "APR_50",
        "APR_60",
        "APR_70",
        "APR_80",
        "ARV_10",
        "ARX_10",
    ],
    "support_network": [
        "RES_10",
        "ARV_40",
        "ARX_40",
        "OAC_20",
        "ACD_80",
        "ACD_90",
    ],
    "financial": [
        "TTLINCG1",
        "FAMINCG1",
    ],
    "respondent_health": [
        "CHC_100",
        "CHC_110K",
        "CHC_110S",
    ],
}

FEATURE_COLUMNS = tuple(
    column for columns in FEATURE_GROUPS.values() for column in columns
)

NUMERIC_FEATURES = (
    "PAR_10",
    "NWE_110",
)

CATEGORICAL_FEATURES = tuple(
    column for column in FEATURE_COLUMNS if column not in NUMERIC_FEATURES
)

LEAKAGE_COLUMNS = {
    *TARGET_COLUMNS,
    "CRH_10",
    "CRH_35",
    "CRH_40",
    "CRH_50",
    "CRH_70",
    "CRH_80",
    "ICS_40",
    "FIS_10A",
    "FIS_10B",
    "FIS_10C",
    "FIS_10D",
    "FIS_10E",
    "FIS_10F",
    "FIS_10G",
    "FIS_10H",
    "FIS_10I",
    "ICP_10",
    "ICP_15",
    "ITL_10",
    "ITL_20",
    "ITL_30",
    "ITO_10",
    "ITO_20",
    "ITO_30",
}

RESERVE_CODE_MAP = {
    "DV_PROXY": {6, 7, 8, 9},
    "PAR_10": {996, 997, 998, 999},
    "HAP_10C": {96, 97, 98, 99},
    "AGEGR10": {96, 97, 98, 99},
    "SEX": {6, 7, 8, 9},
    "MARSTAT": {96, 97, 98, 99},
    "PRV": {96, 97, 98, 99},
    "LIVARR08": {96, 97, 98, 99},
    "COW_10": {6, 7, 8, 9},
    "WTI_110": {6, 7, 8, 9},
    "NWE_110": {96, 97, 98, 99},
    "UHW_16GR": {6, 7, 8, 9},
    "UCA_10": {6, 7, 8, 9},
    "FWA_134": {6, 7, 8, 9},
    "FWA_137": {6, 7, 8, 9},
    "APR_10": {6, 7, 8, 9},
    "APR_20": {6, 7, 8, 9},
    "APR_30": {6, 7, 8, 9},
    "APR_40": {6, 7, 8, 9},
    "APR_50": {6, 7, 8, 9},
    "APR_60": {6, 7, 8, 9},
    "APR_70": {6, 7, 8, 9},
    "APR_80": {6, 7, 8, 9},
    "ARV_10": {6, 7, 8, 9},
    "ARX_10": {6, 7, 8, 9},
    "RES_10": {6, 7, 8, 9},
    "ARV_40": {6, 7, 8, 9},
    "ARX_40": {6, 7, 8, 9},
    "OAC_20": {6, 7, 8, 9},
    "ACD_80": {6, 7, 8, 9},
    "ACD_90": {6, 7, 8, 9},
    "TTLINCG1": {96, 97, 98, 99},
    "FAMINCG1": {96, 97, 98, 99},
    "CHC_100": {6, 7, 8, 9},
    "CHC_110K": {6, 7, 8, 9},
    "CHC_110S": {6, 7, 8, 9},
    "CRH_20": {6, 7, 8, 9},
    "CRH_30": {6, 7, 8, 9},
    "CRH_60": {6, 7, 8, 9},
}

REQUIRED_COLUMNS = (
    WEIGHT_COLUMN,
    "DV_PROXY",
    "PAR_10",
    "HAP_10C",
    *TARGET_COLUMNS,
    *FEATURE_COLUMNS,
)

FORM_FIELDS = [
    {
        "name": "AGEGR10",
        "label": "Age Group",
        "options": [
            ("", "Leave blank"),
            ("1", "15 to 24"),
            ("2", "25 to 34"),
            ("3", "35 to 44"),
            ("4", "45 to 54"),
            ("5", "55 to 64"),
            ("6", "65 to 74"),
            ("7", "75 and older"),
        ],
    },
    {
        "name": "SEX",
        "label": "Sex",
        "options": [
            ("", "Leave blank"),
            ("1", "Male"),
            ("2", "Female"),
        ],
    },
    {
        "name": "MARSTAT",
        "label": "Marital Status",
        "options": [
            ("", "Leave blank"),
            ("1", "Married / common-law"),
            ("2", "Widowed / separated / divorced"),
            ("3", "Single"),
        ],
    },
    {
        "name": "HAP_10C",
        "label": "Average Weekly Caregiving Hours",
        "options": [
            ("", "Leave blank"),
            ("1", "Less than 10 hours"),
            ("2", "10 to 19 hours"),
            ("3", "20 to 29 hours"),
            ("4", "30 to 39 hours"),
            ("5", "40 to 49 hours"),
            ("6", "50 hours or more"),
        ],
    },
    {
        "name": "PAR_10",
        "label": "Number of People Helped",
        "input_type": "number",
    },
    {
        "name": "APR_40",
        "label": "Provides personal care",
        "options": [
            ("", "Leave blank"),
            ("1", "Yes"),
            ("2", "No"),
        ],
    },
    {
        "name": "APR_50",
        "label": "Helps with medical treatment",
        "options": [
            ("", "Leave blank"),
            ("1", "Yes"),
            ("2", "No"),
        ],
    },
    {
        "name": "ARV_10",
        "label": "Visits or calls primary care receiver",
        "options": [
            ("", "Leave blank"),
            ("1", "Yes"),
            ("2", "No"),
        ],
    },
    {
        "name": "ARX_10",
        "label": "Provides emotional support",
        "options": [
            ("", "Leave blank"),
            ("1", "Yes"),
            ("2", "No"),
        ],
    },
    {
        "name": "RES_10",
        "label": "Received emotional support",
        "options": [
            ("", "Leave blank"),
            ("1", "Yes"),
            ("2", "No"),
        ],
    },
    {
        "name": "OAC_20",
        "label": "Wants additional caregiving support",
        "options": [
            ("", "Leave blank"),
            ("1", "Yes"),
            ("2", "No"),
        ],
    },
    {
        "name": "TTLINCG1",
        "label": "Personal Income Group",
        "options": [
            ("", "Leave blank"),
            ("1", "Less than $20,000"),
            ("2", "$20,000 to $39,999"),
            ("3", "$40,000 to $59,999"),
            ("4", "$60,000 to $79,999"),
            ("5", "$80,000 to $99,999"),
            ("6", "$100,000 to $119,999"),
            ("7", "$120,000 or more"),
        ],
    },
    {
        "name": "CHC_100",
        "label": "Has own long-term health condition",
        "options": [
            ("", "Leave blank"),
            ("1", "Yes"),
            ("2", "No"),
        ],
    },
    {
        "name": "CHC_110K",
        "label": "Own mental illness reported",
        "options": [
            ("", "Leave blank"),
            ("1", "Yes"),
            ("2", "No"),
        ],
    },
    {
        "name": "FWA_134",
        "label": "Work offers family-care leave",
        "options": [
            ("", "Leave blank"),
            ("1", "Yes"),
            ("2", "No"),
        ],
    },
    {
        "name": "FWA_137",
        "label": "Work offers telework option",
        "options": [
            ("", "Leave blank"),
            ("1", "Yes"),
            ("2", "No"),
        ],
    },
]

APP_RESOURCES = {
    1: [
        "Family Caregiver Alliance Canada resources",
        "211 Ontario or local 211 for respite and community navigation",
        "Speak with a clinician or caregiver support program if distress is escalating",
    ],
    0: [
        "Maintain existing support routines and respite planning",
        "Review employer flexibility and local caregiver supports before strain escalates",
        "Re-screen if caregiving hours or emotional burden increase",
    ],
}
