from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "src" / "c32pumfm.sas7bdat"

MODEL_DIR = ROOT / "model"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"

MODEL_ARTIFACT_PATH = MODEL_DIR / "caregiver_distress_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
VALIDATION_PATH = REPORTS_DIR / "validation_summary.json"
SCORED_DATA_PATH = PROCESSED_DIR / "scored_caregiver_distress.csv"
TEST_PREDICTIONS_PATH = PROCESSED_DIR / "test_predictions.csv"
FEATURE_MISSINGNESS_PATH = TABLES_DIR / "feature_missingness.csv"
GLOBAL_SHAP_PATH = TABLES_DIR / "global_shap_importance.csv"
SUBGROUP_RISK_PATH = TABLES_DIR / "subgroup_risk_summary.csv"
HIGH_RISK_SEGMENTS_PATH = TABLES_DIR / "highest_risk_segments.csv"
HIGH_RISK_INDIVIDUALS_PATH = TABLES_DIR / "highest_risk_individuals.csv"
SUBGROUP_DRIVERS_PATH = TABLES_DIR / "subgroup_top_drivers.csv"
THRESHOLD_CURVE_PATH = TABLES_DIR / "threshold_curve.csv"
CALIBRATION_TABLE_PATH = TABLES_DIR / "calibration_table.csv"
ARTIFACT_MANIFEST_PATH = REPORTS_DIR / "artifact_manifest.json"
MODEL_REPORT_PATH = DOCS_DIR / "model_report.md"

FEATURE_IMPORTANCE_FIG_PATH = FIGURES_DIR / "feature_importance_bar.png"
SHAP_SUMMARY_FIG_PATH = FIGURES_DIR / "shap_summary.png"
RISK_DISTRIBUTION_FIG_PATH = FIGURES_DIR / "risk_distribution_histogram.png"
SUBGROUP_COMPARISON_FIG_PATH = FIGURES_DIR / "subgroup_risk_comparison.png"
CALIBRATION_FIG_PATH = FIGURES_DIR / "calibration_curve.png"
THRESHOLD_FIG_PATH = FIGURES_DIR / "threshold_tradeoff.png"

HEATMAP_PATHS = {
    "province_risk_band": {
        "csv": TABLES_DIR / "province_by_risk_band.csv",
        "png": FIGURES_DIR / "province_by_risk_band.png",
    },
    "care_hours_risk_band": {
        "csv": TABLES_DIR / "care_hours_by_risk_band.csv",
        "png": FIGURES_DIR / "care_hours_by_risk_band.png",
    },
    "income_risk_band": {
        "csv": TABLES_DIR / "income_by_risk_band.csv",
        "png": FIGURES_DIR / "income_by_risk_band.png",
    },
    "relationship_risk_band": {
        "csv": TABLES_DIR / "relationship_by_risk_band.csv",
        "png": FIGURES_DIR / "relationship_by_risk_band.png",
    },
}

ID_COLUMN = "PUMFID"
WEIGHT_COLUMN = "WGHT_PER"
TARGET_COLUMNS = ("CRH_20", "CRH_30", "CRH_60")
TARGET_ITEM_LABELS = {
    "CRH_20": "Worried or anxious",
    "CRH_30": "Overwhelmed",
    "CRH_60": "Depressed",
}

MODEL_FEATURES = (
    "AGEGR10",
    "SEX",
    "MARSTAT",
    "LIVARR08",
    "PRV",
    "PAR_10",
    "HAP_10C",
    "COW_10",
    "WTI_110",
    "NWE_110",
    "UHW_16GR",
    "UCA_10",
    "FWA_134",
    "FWA_137",
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
    "TTLINCG1",
    "FAMINCG1",
    "CHC_100",
)

ANALYTICS_ONLY_COLUMNS = (
    "PRA_10GR",
)

NUMERIC_FEATURES = (
    "PAR_10",
    "NWE_110",
)

ORDINAL_FEATURES = (
    "AGEGR10",
    "HAP_10C",
    "UHW_16GR",
    "TTLINCG1",
    "FAMINCG1",
)

NOMINAL_FEATURES = (
    "SEX",
    "MARSTAT",
    "LIVARR08",
    "PRV",
    "COW_10",
    "WTI_110",
    "UCA_10",
)

BINARY_FEATURES = tuple(
    column
    for column in MODEL_FEATURES
    if column not in NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
)

SUBGROUP_COLUMNS = (
    "PRV",
    "HAP_10C",
    "TTLINCG1",
    "PRA_10GR",
    "AGEGR10",
    "SEX",
)

PAIRWISE_SEGMENTS = (
    ("PRV", "HAP_10C"),
    ("HAP_10C", "TTLINCG1"),
    ("PRV", "TTLINCG1"),
    ("PRA_10GR", "HAP_10C"),
)

RISK_BAND_BINS = (0.0, 0.30, 0.50, 0.70, 1.01)
RISK_BAND_LABELS = ("Low", "Moderate", "High", "Very high")

LEAKAGE_PREFIXES = (
    "CRH_",
    "FIS_",
    "ICL_",
    "ICB_",
    "ICP_",
    "ITL_",
    "ITO_",
    "WLB_",
    "EMO_",
    "ICS_",
)

LEAKAGE_EXACT = set(TARGET_COLUMNS)

RESERVE_CODE_MAP = {
    "DV_PROXY": {6, 7, 8, 9},
    "PAR_10": {996, 997, 998, 999},
    "HAP_10C": {96, 97, 98, 99},
    "AGEGR10": {96, 97, 98, 99},
    "SEX": {6, 7, 8, 9},
    "MARSTAT": {96, 97, 98, 99},
    "LIVARR08": {96, 97, 98, 99},
    "PRV": {96, 97, 98, 99},
    "PRA_10GR": {96, 97, 98, 99},
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
    "TTLINCG1": {96, 97, 98, 99},
    "FAMINCG1": {96, 97, 98, 99},
    "CHC_100": {6, 7, 8, 9},
    "CRH_20": {6, 7, 8, 9},
    "CRH_30": {6, 7, 8, 9},
    "CRH_60": {6, 7, 8, 9},
}

REQUIRED_COLUMNS = tuple(
    dict.fromkeys(
        (
            ID_COLUMN,
            WEIGHT_COLUMN,
            "DV_PROXY",
            "PAR_10",
            "HAP_10C",
            *TARGET_COLUMNS,
            *MODEL_FEATURES,
            *ANALYTICS_ONLY_COLUMNS,
        )
    )
)

PROVINCE_LABELS = {
    10: "Newfoundland and Labrador",
    11: "Prince Edward Island",
    12: "Nova Scotia",
    13: "New Brunswick",
    24: "Quebec",
    35: "Ontario",
    46: "Manitoba",
    47: "Saskatchewan",
    48: "Alberta",
    59: "British Columbia",
}

AGE_GROUP_LABELS = {
    1: "15-24",
    2: "25-34",
    3: "35-44",
    4: "45-54",
    5: "55-64",
    6: "65-74",
    7: "75+",
}

SEX_LABELS = {
    1: "Male",
    2: "Female",
}

HOUR_GROUP_LABELS = {
    1: "<10 hours",
    2: "10-19 hours",
    3: "20-29 hours",
    4: "30-39 hours",
    5: "40-49 hours",
    6: "50+ hours",
}

INCOME_GROUP_LABELS = {
    1: "<$20k",
    2: "$20k-$39.9k",
    3: "$40k-$59.9k",
    4: "$60k-$79.9k",
    5: "$80k-$99.9k",
    6: "$100k-$119.9k",
    7: "$120k+",
}

FRIENDLY_NAMES = {
    "AGEGR10": "Age group",
    "SEX": "Sex",
    "MARSTAT": "Marital status",
    "LIVARR08": "Living arrangement",
    "PRV": "Province",
    "PAR_10": "People helped",
    "HAP_10C": "Weekly care hours",
    "COW_10": "Class of worker",
    "WTI_110": "Work tenure indicator",
    "NWE_110": "Weeks worked",
    "UHW_16GR": "Usual hours worked",
    "UCA_10": "Work flexibility indicator",
    "FWA_134": "Employer family-care leave",
    "FWA_137": "Employer telework option",
    "ARV_10": "Visits or calls care receiver",
    "ARX_10": "Provides emotional support",
    "TTLINCG1": "Personal income group",
    "FAMINCG1": "Household income group",
    "CHC_100": "Own long-term health condition",
}

