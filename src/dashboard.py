"""
dashboard.py — Caregiver Burnout Risk Screener
===============================================
Streamlit dashboard built on the GSS 2018 Caregiving & Care Receiving data
(Statistics Canada, Cycle 32).

Purpose:
    Help government agencies and nonprofits identify which caregiver groups
    are most at risk for burnout and understand the key drivers — so they can
    direct resources where they matter most.

How to run:
    streamlit run src/dashboard.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # non-interactive backend required for Streamlit
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyreadstat
import shap
import streamlit as st
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# =============================================================================
# PATHS
# =============================================================================
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
SAS_FILE    = os.path.join(SRC_DIR, "c32pumfm.sas7bdat")
CB_DIR      = os.path.normpath(os.path.join(SRC_DIR, "..", "gss32_simplified_formats"))
ANSWER_CATS = os.path.join(CB_DIR, "codebook_answer_categories.csv")
VAR_CSV     = os.path.join(CB_DIR, "codebook_variables.csv")


# =============================================================================
# CONSTANTS
# =============================================================================
# These are the special "reserve" codes Statistics Canada uses in the survey.
# They mean the question was not applicable (Valid skip) or the person didn't
# answer (Not stated). We treat them all as missing values (NaN).
RESERVE_CODES = [
    6, 9, 96, 99, 996, 999,
    9996, 9999, 99996, 99999, 999996, 999999,
]

# The 12 survey questions that together measure caregiver burnout:
#   ICS_40       — How stressful is your caregiving overall?
#   FIS_10A–H    — Has caregiving affected your family life in these ways?
#   CRH_10/20/30 — Has caregiving affected your personal relationships?
BURNOUT_ITEMS = [
    "ICS_40",
    "FIS_10A", "FIS_10B", "FIS_10C", "FIS_10D",
    "FIS_10E", "FIS_10F", "FIS_10G", "FIS_10H",
    "CRH_10", "CRH_20", "CRH_30",
]

# Workplace flexibility items used to build the composite flexibility score
FWA_POS = ["FWA_132", "FWA_133", "FWA_134", "FWA_136", "FWA_137"]
FWA_NEG = ["FWA_150"]
# Work-enabling circumstances items used to build the ITE composite score
ITE_POS = ["ITE_30A", "ITE_30B", "ITE_30C", "ITE_30D", "ITE_30E"]
ITE_NEG = ["ITE_10"]

# All predictor columns included in the model (from Include Variables spec)
INCLUDE_VARS = [
    "WGHT_PER",
    "SEX", "MARSTAT", "PHSDFLG", "AGEPRGR0", "SENFLAG",
    "LIVARR08", "LIVARRSN", "LUC_RST", "PRV", "NLC_100C",
    "CARUNPAI", "CARPAID", "CRGVAGGR", "DPA_10",
    "NFA_10", "NFA_30", "PGN_25", "PGW_10", "PGW_20",
    "APR_10", "APR_20", "APR_30", "APR_40", "APR_50",
    "APR_60", "APR_70", "APR_80",
    "HAP_10C", "PAR_10",
    "RRA_10C", "RRA_20C", "RRA_30C", "DPR_10C", "DPR_40C",
    "APX_10GR", "APX_20GR", "APX_30C", "APX_50GR",
    "RPE_10", "CRRCPAGR", "PRN_25", "PRG10GR", "PRP10GR", "PRP_15",
    "PRW_10", "PRW_20", "PRD_10", "PRU_10", "PRH_20",
    "ART_30", "ART_40",
    "ARI_20", "ARI_30", "ARI_40",
    "ARO_10", "ARO_20", "ARO_30", "ARO_40",
    "ARP_10", "ARP_20", "ARP_30", "ARP_40",
    "ARM_10", "ARM_20", "ARM_30C", "ARM_40",
    "ARS_10", "ARS_20", "ARS_30C", "ARS_40",
    "ARB_10", "ARB_20", "ARB_30C", "ARB_40",
    "ARV_10", "ARV_40", "ARX_10", "ARX_40",
    "CCP_20", "DVCG120C",
    "RNA_10C", "RNA_20C", "RNA_30C", "RNA_40C",
    "HRA_10",
    "ACD_10", "ACD_20", "ACD_30", "ACD_40", "ACD_50",
    "ACD_60", "ACD_70", "ACD_80", "ACD_90",
    "OAC_20", "AGEBEG1C", "CGE_150", "CCL_20",
    "ICL_110", "ICL_120", "ICL_130", "ICL_135", "ICL_140",
    "ICL_150", "ICL_154", "ICL_180", "ICL_210",
    "ICB_15", "ICB_20", "ICB_25", "ICP_15", "ICP_30",
    "ICS_20", "ICS_30",
    "HOME_EXP", "HLTH_EXP", "HELP_EXP", "TRNS_EXP", "AID_EXP", "MED_EXP",
    "ICF2_290", "ICF2_300", "ICF2_310", "ICF2_320", "ICF2_330", "ICF2_340",
    "EDM_02", "ICE_50", "COW_10", "IPL_10",
    "UWS230GR", "TOE_240", "ITO_10", "INE_10",
    "PTN_10", "FAMINCG1", "BPR_16", "VISMIN", "LAN_01",
    "flexibility_score", "ite_score",
]

# Answer labels from the codebook that represent missing / not-applicable —
# we exclude these from the display label map since they become NaN in the data
_RESERVE_LABELS = {"Valid skip", "Don't know", "Refusal", "Not stated"}


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Caregiver Burnout Risk Screener",
    page_icon="🧡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CACHED DATA LOADING
# =============================================================================

@st.cache_data(show_spinner="Loading answer labels from codebook…")
def load_label_map():
    """
    Reads the codebook answer-categories file and builds a lookup table:
        { variable_code: { numeric_answer: human_readable_label } }

    For example:
        label_map["SEX"][1]  →  "Male"
        label_map["PRV"][35] →  "Ontario"

    Reserve-code rows (Valid skip, Don't know, Refusal, Not stated) are
    excluded because those values become NaN in the cleaned data.

    "Y es" is a known PDF-extraction artefact and is corrected to "Yes".
    """
    cats = pd.read_csv(ANSWER_CATS)
    cats["label"] = cats["label"].str.strip().str.replace(r"^Y\s+es$", "Yes", regex=True)
    cats = cats[~cats["label"].isin(_RESERVE_LABELS)]

    label_map: dict[str, dict[int, str]] = {}
    for _, row in cats.iterrows():
        var = row["variable_name"]
        try:
            code = int(float(row["code"]))
        except (ValueError, TypeError):
            continue   # skip range-style codes like "00 - 84"
        label_map.setdefault(var, {})[code] = str(row["label"])
    return label_map


@st.cache_data(show_spinner="Loading variable descriptions…")
def load_var_info():
    """
    Returns a dict mapping variable codes to plain-English descriptions:
        { "SEX": "Sex of respondent",
          "PRV": "Province / territory of residence", ... }

    Used to replace raw column names with readable titles in all charts.
    """
    df = pd.read_csv(VAR_CSV, usecols=["variable_name", "concept", "question_text"])
    result = {}
    for _, row in df.iterrows():
        desc = row["concept"]
        if pd.isna(desc) or str(desc).strip() == "":
            desc = str(row["question_text"])[:80] if pd.notna(row["question_text"]) else row["variable_name"]
        result[row["variable_name"]] = str(desc).strip()
    # Add readable names for the two composite scores (not in codebook)
    result["flexibility_score"] = "Workplace Flexibility Score"
    result["ite_score"]         = "Work-Enabling Circumstances Score"
    return result


@st.cache_data(show_spinner="Loading and processing survey data (first run only — ~30 s)…")
def load_and_process():
    """
    Runs the full data pipeline:
      1. Load the SAS public-use microdata file
      2. Drop the 500+ bootstrap weight columns (not needed for modelling)
      3. Build two composite scores from the FWA and ITE survey questions
      4. Assemble the final predictor table; replace all reserve codes with NaN
      5. Compute the burnout target from 12 validated survey items
      6. Separate predictors (X), burnout label (y), and survey weight (W)
    """
    main, _ = pyreadstat.read_sas7bdat(SAS_FILE)

    # Drop bootstrap weight columns
    main_clean = main.drop(columns=[c for c in main.columns if c.startswith("WTBS_")])

    # --- Composite score 1: Workplace Flexibility (-1 to 5) ------------------
    # Positive items: respondent CAN do this (Yes=1, No=0)
    #   FWA_132 — Can work from home
    #   FWA_133 — Can vary start/end time
    #   FWA_134 — Can take time off during the day (and make it up later)
    #   FWA_136 — Can temporarily reduce hours
    #   FWA_137 — Has other flexible work arrangements
    # Negative item (subtracts 1 if true):
    #   FWA_150 — Using flexibility hurts career advancement
    scores_base = main_clean.copy()
    for col in FWA_POS + FWA_NEG + ITE_POS + ITE_NEG:
        if col in scores_base.columns:
            scores_base[col] = scores_base[col].replace(RESERVE_CODES, np.nan)
            scores_base[col] = scores_base[col].map({1: 1, 2: 0})

    scores_base["flexibility_score"] = (
        scores_base[FWA_POS].sum(axis=1, min_count=1)
        - scores_base[FWA_NEG[0]].fillna(0)
    )

    # --- Composite score 2: Work-Enabling Circumstances (-1 to 5) ------------
    # Positive items: this circumstance WOULD help the respondent stay employed
    #   ITE_30A — Flexible scheduling
    #   ITE_30B — Option to work from home
    #   ITE_30C — Reduced hours with no pay penalty
    #   ITE_30D — Access to eldercare near workplace
    #   ITE_30E — Other support at work
    # Negative item (subtracts 1 if true):
    #   ITE_10  — Already had to quit or reduce hours because of caregiving
    scores_base["ite_score"] = (
        scores_base[ITE_POS].sum(axis=1, min_count=1)
        - scores_base[ITE_NEG[0]].fillna(0)
    )

    # --- Build predictor table ------------------------------------------------
    available  = [c for c in INCLUDE_VARS if c in scores_base.columns]
    include_df = scores_base[available].copy()
    for col in available:
        if col not in ("flexibility_score", "ite_score", "WGHT_PER"):
            include_df[col] = include_df[col].replace(RESERVE_CODES, np.nan)

    # --- Burnout target -------------------------------------------------------
    # Recode all 12 burnout items to a 0-1 scale where 1.0 = maximum strain:
    #   Binary items (FIS_10*, CRH_*): 1(Yes/strain) -> 1.0,  2(No) -> 0.0
    #   ICS_40 (4-point):  1(Very stressful) -> 1.0,  2 -> 0.667,  3 -> 0.333,  4(Not at all) -> 0.0
    # Then average across answered items; higher = more burnout.
    burnout_raw = main_clean[BURNOUT_ITEMS].replace(RESERVE_CODES, np.nan)
    for col in BURNOUT_ITEMS:
        if col not in burnout_raw.columns:
            continue
        if col == "ICS_40":
            burnout_raw[col] = burnout_raw[col].map({1: 1.0, 2: 2/3, 3: 1/3, 4: 0.0})
        else:
            burnout_raw[col] = burnout_raw[col].map({1: 1.0, 2: 0.0})
    burnout_score = burnout_raw.mean(axis=1)
    median_cutoff = burnout_score.median()
    burnout_high  = (burnout_score >= median_cutoff).astype(int)

    valid_idx       = burnout_score.dropna().index
    include_aligned = include_df.loc[valid_idx].copy()
    y_score         = burnout_score.loc[valid_idx]
    y_clf           = burnout_high.loc[valid_idx]

    exclude_x = ["WGHT_PER"] + [c for c in BURNOUT_ITEMS if c in include_aligned.columns]
    X = include_aligned.drop(columns=[c for c in exclude_x if c in include_aligned.columns])
    W = include_aligned.get("WGHT_PER", pd.Series(np.ones(len(include_aligned)),
                                                    index=include_aligned.index))

    return dict(X=X, y_clf=y_clf, y_reg=y_score, W=W,
                include_df=include_aligned, burnout_score=y_score,
                median_cutoff=median_cutoff)


@st.cache_resource(show_spinner="Training burnout prediction model (first run only — ~1 min)…")
def train_model():
    """
    Trains an XGBoost gradient-boosted tree classifier to predict high vs
    low burnout risk.  Survey weights are passed as sample_weight so the
    model accounts for the complex survey design.

    XGBoost handles missing values natively — no imputation needed.

    Returns the trained model plus evaluation results (AUC, ROC curve,
    precision/recall report).
    """
    data = load_and_process()
    X, y_clf, W = data["X"], data["y_clf"], data["W"]

    XGB_CONFIG = dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        scale_pos_weight=1, eval_metric="logloss",
        early_stopping_rounds=30, random_state=42, n_jobs=-1,
    )

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y_clf, W.values, test_size=0.20, random_state=42, stratify=y_clf,
    )

    model = XGBClassifier(**XGB_CONFIG)
    model.fit(X_train, y_train, sample_weight=w_train,
              eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    report = classification_report(
        y_test, y_pred,
        target_names=["Low Burnout", "High Burnout"],
        output_dict=True,
    )
    return model, auc, fpr, tpr, report, X_test, y_test, y_prob


@st.cache_data(show_spinner="Computing SHAP values for all respondents (1–2 min)…")
def get_shap_values():
    """
    Calculates a SHAP (SHapley Additive exPlanation) value for every
    respondent and every feature.

    A positive SHAP value means the feature pushed that person's predicted
    burnout risk HIGHER.  A negative value means it pushed it LOWER.

    Returns a DataFrame (one row per respondent, one column per feature).
    """
    data  = load_and_process()
    X     = data["X"]
    model, *_ = train_model()
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return pd.DataFrame(shap_values, index=X.index, columns=X.columns)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def decode_col(series: pd.Series, label_map: dict, var_name: str) -> pd.Series:
    """Translate numeric survey codes to their human-readable answer labels."""
    mapping = label_map.get(var_name, {})
    if not mapping:
        return series
    def _f(x):
        if pd.isna(x):
            return np.nan
        try:
            return mapping.get(int(x), x)
        except (ValueError, TypeError):
            return x
    return series.map(_f)


def burnout_rate_chart(df: pd.DataFrame, var: str, label_map: dict,
                       var_info: dict, title: str = None) -> go.Figure:
    """
    Horizontal bar chart showing the percentage of high-burnout respondents
    within each category of the chosen variable.
    Bars are colour-coded from green (low) to red (high).
    """
    tmp = df[[var, "burnout_high"]].dropna().copy()
    tmp["_label"] = decode_col(tmp[var], label_map, var)
    tmp = tmp.dropna(subset=["_label"])

    stats = (
        tmp.groupby("_label")["burnout_high"]
        .agg(rate="mean", count="count")
        .reset_index()
    )
    stats["pct"] = (stats["rate"] * 100).round(1)
    stats = stats.sort_values("pct")

    readable_title = title or f"High Burnout Rate by {var_info.get(var, var)}"
    fig = px.bar(
        stats, x="pct", y="_label", orientation="h",
        text="pct",
        labels={"pct": "% with High Burnout Risk", "_label": ""},
        title=readable_title,
        color="pct",
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 100],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        height=max(280, len(stats) * 36 + 80),
        margin=dict(l=10, r=50, t=50, b=30),
    )
    return fig


# =============================================================================
# LOAD DATA (runs once; cached for the rest of the session)
# =============================================================================
label_map = load_label_map()
var_info  = load_var_info()
data      = load_and_process()

X             = data["X"]
y_clf         = data["y_clf"]
y_reg         = data["y_reg"]
W             = data["W"]
burnout_score = data["burnout_score"]
median_cutoff = data["median_cutoff"]

# Main analysis table used throughout the dashboard
analysis_df = X.copy()
analysis_df["burnout_high"]  = y_clf
analysis_df["burnout_score"] = y_reg
analysis_df["WGHT_PER"]      = W

# Human-readable names for every feature column (used in SHAP plots)
feat_display_names = [
    var_info.get(c, c)[:45] + ("…" if len(var_info.get(c, c)) > 45 else "")
    for c in X.columns
]


# =============================================================================
# SIDEBAR — DEMOGRAPHIC FILTERS
# =============================================================================
st.sidebar.title("🧡 Caregiver Burnout Risk")
st.sidebar.markdown("**GSS 2018 — Statistics Canada**")
st.sidebar.divider()
st.sidebar.markdown("### Filter the Population")
st.sidebar.markdown(
    "Narrow down to a specific group to see how burnout risk looks for them."
)

FILTER_VARS = {
    "SEX":      "Sex",
    "PRV":      "Province / Territory",
    "FAMINCG1": "Household Income Group",
    "MARSTAT":  "Marital Status",
    "VISMIN":   "Visible Minority Status",
    "SENFLAG":  "Senior (65+) Flag",
}

active_filters: dict[str, int] = {}
for col, display_name in FILTER_VARS.items():
    if col not in analysis_df.columns:
        continue
    raw_codes = sorted(analysis_df[col].dropna().unique().astype(int).tolist())
    labels    = [label_map.get(col, {}).get(c, str(c)) for c in raw_codes]
    choice    = st.sidebar.selectbox(display_name, ["All"] + labels)
    if choice != "All":
        active_filters[col] = raw_codes[labels.index(choice)]

filter_mask = pd.Series(True, index=analysis_df.index)
for col, code in active_filters.items():
    filter_mask &= (analysis_df[col] == code)
filtered_df = analysis_df[filter_mask]

if active_filters:
    st.sidebar.success(f"{filter_mask.sum():,} respondents match your filters.")
else:
    st.sidebar.info(f"Showing all {len(analysis_df):,} respondents.")

st.sidebar.divider()
st.sidebar.caption("Data: Statistics Canada, GSS Cycle 32, 2018")


# =============================================================================
# PAGE HEADER
# =============================================================================
st.title("🧡 Caregiver Burnout Risk Screener")
st.markdown(
    "*Helping government agencies and nonprofits identify the caregivers most at "
    "risk for burnout — and understand what's driving that risk — so resources "
    "can be directed where they're needed most.*"
)


# =============================================================================
# TABS
# =============================================================================
tab_overview, tab_diag, tab_drivers, tab_explore = st.tabs([
    "📊 Overview",
    "🔍 Data Diagnostics",
    "📈 What Drives Burnout Risk",
    "🔬 Explore Any Variable",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    n_total  = len(filtered_df)
    n_high   = int(filtered_df["burnout_high"].sum())
    pct_high = n_high / n_total * 100 if n_total > 0 else 0.0
    avg_score = filtered_df["burnout_score"].mean()

    # --- Top-line numbers ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents in View", f"{n_total:,}")
    c2.metric(
        "Flagged High Burnout Risk",
        f"{pct_high:.1f}%",
        f"{n_high:,} people",
        help="Respondents whose average score across 12 burnout questions "
             "is at or above the overall median.",
    )
    c3.metric(
        "Average Burnout Score",
        f"{avg_score:.3f}",
        f"Split threshold: {median_cutoff:.3f}",
        help="Mean of 12 burnout survey items (0–4 scale). "
             "Respondents at or above the overall median are labelled High Risk.",
    )
    if "flexibility_score" in filtered_df.columns:
        avg_flex = filtered_df["flexibility_score"].mean()
        c4.metric(
            "Average Workplace Flexibility",
            f"{avg_flex:.2f}",
            "Scale: −1 (no flexibility, career penalty) to +5 (maximum flexibility)",
        )

    st.divider()

    # --- Row 1: Province and Sex ---
    col_a, col_b = st.columns(2)
    with col_a:
        if "PRV" in filtered_df.columns:
            st.plotly_chart(
                burnout_rate_chart(filtered_df, "PRV", label_map, var_info,
                                   "High Burnout Rate by Province or Territory"),
                width="stretch",
            )
    with col_b:
        if "SEX" in filtered_df.columns:
            st.plotly_chart(
                burnout_rate_chart(filtered_df, "SEX", label_map, var_info,
                                   "High Burnout Rate by Sex"),
                width="stretch",
            )

    # --- Row 2: Score distribution and Marital status ---
    col_c, col_d = st.columns(2)
    with col_c:
        fig_hist = px.histogram(
            filtered_df, x="burnout_score", nbins=30,
            title="Distribution of Burnout Scores",
            labels={
                "burnout_score": "Burnout Score (average of 12 survey items)",
                "count": "Number of Respondents",
            },
            color_discrete_sequence=["#e07b54"],
        )
        fig_hist.add_vline(
            x=median_cutoff, line_dash="dash", line_color="red",
            annotation_text=f"High / Low split point: {median_cutoff:.2f}",
            annotation_position="top right",
        )
        fig_hist.update_layout(margin=dict(t=50))
        st.plotly_chart(fig_hist, width="stretch")

    with col_d:
        if "MARSTAT" in filtered_df.columns:
            st.plotly_chart(
                burnout_rate_chart(filtered_df, "MARSTAT", label_map, var_info,
                                   "High Burnout Rate by Marital Status"),
                width="stretch",
            )

    # --- Row 3: Income and Visible minority ---
    col_e, col_f = st.columns(2)
    with col_e:
        if "FAMINCG1" in filtered_df.columns:
            st.plotly_chart(
                burnout_rate_chart(filtered_df, "FAMINCG1", label_map, var_info,
                                   "High Burnout Rate by Household Income"),
                width="stretch",
            )
    with col_f:
        if "VISMIN" in filtered_df.columns:
            st.plotly_chart(
                burnout_rate_chart(filtered_df, "VISMIN", label_map, var_info,
                                   "High Burnout Rate by Visible Minority Status"),
                width="stretch",
            )

    # --- Composite scores ---
    st.divider()
    st.subheader("Do Workplace Conditions Affect Burnout Risk?")
    st.markdown(
        "These two composite scores summarise workplace flexibility and access "
        "to enabling work conditions. Lower scores mean fewer supports in place."
    )

    col_g, col_h = st.columns(2)

    def _score_bar(df, col, xlabel, title):
        tmp = df[[col, "burnout_high"]].dropna()
        stats = (
            tmp.groupby(col)["burnout_high"]
            .agg(pct="mean", count="count").reset_index()
        )
        stats["pct"] = (stats["pct"] * 100).round(1)
        fig = px.bar(
            stats, x=col, y="pct", title=title,
            labels={col: xlabel, "pct": "% with High Burnout Risk"},
            color="pct", color_continuous_scale="RdYlGn_r",
            range_color=[0, 100], text="pct",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 105])
        return fig

    with col_g:
        if "flexibility_score" in filtered_df.columns:
            st.plotly_chart(
                _score_bar(filtered_df, "flexibility_score",
                           "Flexibility Score (−1 = none, +5 = maximum)",
                           "Burnout Risk by Workplace Flexibility Score"),
                width="stretch",
            )
    with col_h:
        if "ite_score" in filtered_df.columns:
            st.plotly_chart(
                _score_bar(filtered_df, "ite_score",
                           "Work-Enabling Score (−1 = none, +5 = maximum)",
                           "Burnout Risk by Work-Enabling Circumstances Score"),
                width="stretch",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    st.subheader("Data Diagnostics — A Look at the Data Before Modelling")
    st.markdown(
        "This section explores the raw survey data to give a sense of who the "
        "caregivers are, how much they are doing, and which variables are most "
        "connected to burnout — before the machine-learning model is involved."
    )

    # ── 1. Data completeness ────────────────────────────────────────────────
    st.markdown("### 1. How Complete Is the Data?")
    st.markdown(
        "Many survey questions are only asked to specific groups "
        "(e.g. only employed caregivers, only those providing personal care). "
        "Variables with high missingness are typically module-specific — "
        "most respondents skipped that section entirely."
    )

    missing_pct = (
        analysis_df
        .drop(columns=["burnout_high", "burnout_score", "WGHT_PER"], errors="ignore")
        .isnull()
        .mean() * 100
    )
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False).head(30)

    if len(missing_pct) > 0:
        miss_labels = [
            var_info.get(c, c)[:40] + ("…" if len(var_info.get(c, c)) > 40 else "")
            for c in missing_pct.index
        ]
        col_miss_a, col_miss_b = st.columns([3, 1])
        with col_miss_a:
            fig_miss = px.bar(
                x=missing_pct.values[::-1],
                y=miss_labels[::-1],
                orientation="h",
                title="Top 30 Variables by % Missing Values",
                labels={"x": "% of Respondents with No Answer", "y": ""},
                color=missing_pct.values[::-1],
                color_continuous_scale="Reds",
            )
            fig_miss.update_layout(
                coloraxis_showscale=False,
                height=700,
                margin=dict(l=10, r=30, t=50, b=20),
            )
            st.plotly_chart(fig_miss, width="stretch")
        with col_miss_b:
            st.markdown("**Summary**")
            n_complete = (missing_pct == 0).sum()
            n_partial  = (missing_pct.between(1, 50, inclusive="both")).sum()
            n_sparse   = (missing_pct > 50).sum()
            st.metric("Fully answered variables", f"{n_complete}")
            st.metric("Partially answered (1–50% missing)", f"{n_partial}")
            st.metric("Mostly skipped (>50% missing)", f"{n_sparse}")
            st.caption(
                "High missingness usually means the question was only shown to "
                "a subset of respondents (module routing), not that people refused "
                "to answer."
            )

    st.divider()

    # ── 2. Correlation with burnout score ───────────────────────────────────
    st.markdown("### 2. Which Variables Are Most Strongly Linked to Burnout?")
    st.markdown(
        "Pearson correlation between each predictor and the continuous burnout "
        "score. A bar pointing **right (positive)** means higher values of that "
        "variable go with higher burnout. A bar pointing **left (negative)** "
        "means higher values are associated with *lower* burnout. "
        "Note: correlation captures linear relationships only; the model picks up "
        "non-linear patterns too."
    )

    numeric_x = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_x = [c for c in numeric_x if c not in ("burnout_high", "WGHT_PER")]

    corr_series = (
        analysis_df[numeric_x]
        .corrwith(analysis_df["burnout_score"])
        .dropna()
    )
    top_corr = corr_series.abs().sort_values(ascending=False).head(25)
    top_corr_signed = corr_series.loc[top_corr.index]

    top_corr_labels = [
        var_info.get(c, c)[:40] + ("…" if len(var_info.get(c, c)) > 40 else "")
        for c in top_corr_signed.index
    ]
    bar_colors = ["#d73027" if v > 0 else "#1a9850" for v in top_corr_signed.values]

    fig_corr = go.Figure(go.Bar(
        x=top_corr_signed.values[::-1],
        y=top_corr_labels[::-1],
        orientation="h",
        marker_color=bar_colors[::-1],
        text=[f"{v:+.3f}" for v in top_corr_signed.values[::-1]],
        textposition="outside",
    ))
    fig_corr.update_layout(
        title="Top 25 Variables by Absolute Correlation with Burnout Score",
        xaxis_title="Pearson Correlation with Burnout Score  "
                    "(red = higher → more burnout, green = higher → less burnout)",
        xaxis=dict(range=[-0.55, 0.55]),
        height=600,
        margin=dict(l=10, r=60, t=50, b=20),
    )
    st.plotly_chart(fig_corr, width="stretch")

    st.divider()

    # ── 3. Correlation heatmap (top features with each other) ───────────────
    st.markdown("### 3. How Are the Top Risk Factors Related to Each Other?")
    st.markdown(
        "This heatmap shows how the 15 variables most correlated with burnout "
        "relate to *each other*. Clusters of highly correlated variables "
        "(dark red/blue squares) suggest they are measuring a similar underlying "
        "concept and may reinforce each other."
    )

    heat_features = top_corr.head(15).index.tolist()
    corr_matrix   = analysis_df[heat_features].corr()

    # Build unique labels: truncate description, then append [CODE] if duplicate
    raw_labels = [var_info.get(c, c)[:30] for c in heat_features]
    heat_labels = []
    for c, lbl in zip(heat_features, raw_labels):
        if raw_labels.count(lbl) > 1:
            heat_labels.append(f"{lbl[:24]}… [{c}]")
        else:
            heat_labels.append(lbl + ("…" if len(var_info.get(c, c)) > 30 else ""))

    corr_matrix.index   = heat_labels
    corr_matrix.columns = heat_labels

    fig_heat = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Matrix — Top 15 Burnout-Linked Variables",
        text_auto=".2f",
        aspect="auto",
    )
    fig_heat.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_colorbar_title="Pearson r",
    )
    st.plotly_chart(fig_heat, width="stretch")

    st.divider()

    # ── 4. Caregiver workload — how much are caregivers doing? ──────────────
    st.markdown("### 4. How Overworked Are Caregivers?")
    st.markdown(
        "Each respondent can be helping their care receiver with multiple types "
        "of activities simultaneously. The chart below shows how many *categories* "
        "of activity each caregiver is responsible for, and what the burnout rate "
        "looks like at each level."
    )

    # Activity gateway variables (Yes=1 means caregiver helps with this category)
    ACTIVITY_GATEWAY = {
        "ARO_10":  "House maintenance",
        "ARP_10":  "Personal care",
        "ARM_10":  "Medical treatments",
        "ARS_10":  "Social / recreational activities",
        "ARB_10":  "Behavioural support",
        "ARV_10":  "Vision / hearing assistance",
        "ARX_10":  "Other assistance",
    }
    avail_act = {k: v for k, v in ACTIVITY_GATEWAY.items() if k in analysis_df.columns}

    # Count how many activity types each caregiver does (= 1 means Yes)
    act_df = analysis_df[list(avail_act.keys())].copy()
    activity_count = (act_df == 1).sum(axis=1)
    analysis_df["activity_count"] = activity_count

    col_act_a, col_act_b = st.columns(2)

    with col_act_a:
        act_dist = activity_count.value_counts().sort_index().reset_index()
        act_dist.columns = ["Number of Activity Types", "Respondents"]
        act_dist["label"] = act_dist["Number of Activity Types"].map(
            lambda x: "0 — Not actively caregiving (in this module)"
            if x == 0 else str(x)
        )
        fig_act = px.bar(
            act_dist,
            x="Number of Activity Types",
            y="Respondents",
            title="How Many Types of Caregiving Activities Are Respondents Doing?",
            labels={
                "Number of Activity Types": "Number of Activity Categories",
                "Respondents": "Number of Respondents",
            },
            color="Number of Activity Types",
            color_continuous_scale="OrRd",
        )
        fig_act.update_layout(coloraxis_showscale=False,
                               margin=dict(t=55))
        st.plotly_chart(fig_act, width="stretch")

        # Key stats
        active = activity_count[activity_count > 0]
        st.markdown(
            f"- **{len(active):,}** respondents are actively helping with at least one activity type  \n"
            f"- On average, active caregivers help with **{active.mean():.1f} types** of activities  \n"
            f"- **{(activity_count >= 3).sum():,}** are juggling **3 or more** activity types at once"
        )

    with col_act_b:
        # Burnout rate by number of activities
        act_burnout = (
            analysis_df[["activity_count", "burnout_high"]]
            .groupby("activity_count")["burnout_high"]
            .agg(rate="mean", count="count")
            .reset_index()
        )
        act_burnout["pct"] = (act_burnout["rate"] * 100).round(1)
        fig_act_br = px.bar(
            act_burnout,
            x="activity_count",
            y="pct",
            title="Does Doing More Activities Raise Burnout Risk?",
            labels={
                "activity_count": "Number of Caregiving Activity Types",
                "pct": "% with High Burnout Risk",
            },
            color="pct",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
            text="pct",
        )
        fig_act_br.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_act_br.update_layout(
            coloraxis_showscale=False,
            yaxis_range=[0, 105],
            margin=dict(t=55),
        )
        st.plotly_chart(fig_act_br, width="stretch")

    st.divider()

    # ── 5. Which activities are most common? ────────────────────────────────
    st.markdown("### 5. What Types of Help Are Caregivers Providing Most?")
    st.markdown(
        "Among respondents who help with at least one activity, the chart shows "
        "how many are doing each type — and what the burnout rate is for each group."
    )

    act_summary_rows = []
    for col, label in avail_act.items():
        n_yes    = int((analysis_df[col] == 1).sum())
        n_total  = int(analysis_df[col].notna().sum())
        br       = analysis_df.loc[analysis_df[col] == 1, "burnout_high"].mean() * 100
        act_summary_rows.append({
            "Activity Type":        label,
            "Caregivers Involved":  n_yes,
            "% of Respondents":     f"{n_yes / len(analysis_df) * 100:.1f}%",
            "High Burnout Rate":    f"{br:.1f}%",
        })

    act_summary_df = pd.DataFrame(act_summary_rows).sort_values(
        "Caregivers Involved", ascending=False
    )

    col_act_c, col_act_d = st.columns([1, 2])
    with col_act_c:
        st.dataframe(act_summary_df, width="stretch", hide_index=True)

    with col_act_d:
        fig_act_type = px.bar(
            act_summary_df.sort_values("Caregivers Involved"),
            x="Caregivers Involved",
            y="Activity Type",
            orientation="h",
            title="Number of Caregivers Per Activity Type",
            labels={"Caregivers Involved": "Respondents Providing This Type of Help",
                    "Activity Type": ""},
            color="Caregivers Involved",
            color_continuous_scale="Blues",
        )
        fig_act_type.update_layout(coloraxis_showscale=False,
                                    height=360, margin=dict(l=10, r=20, t=50, b=20))
        st.plotly_chart(fig_act_type, width="stretch")

    st.divider()

    # ── 6. Financial burden ─────────────────────────────────────────────────
    st.markdown("### 6. What Is the Financial Burden on Caregivers?")
    st.markdown(
        "Caregivers often pay out-of-pocket for their care receiver's needs. "
        "Expense variables are coded as **ordinal brackets** (1 = Less than $200, "
        "2 = $200-$499, 3 = $500-$999, 4 = $1,000-$1,999, 5 = $2,000-$4,999, "
        "6 = $5,000+), **not actual dollar amounts**."
    )

    EXPENSE_COLS = {
        "HOME_EXP":  "Home modifications",
        "HLTH_EXP":  "Health care costs",
        "HELP_EXP":  "Paid help / services",
        "TRNS_EXP":  "Transportation",
        "AID_EXP":   "Assistive devices / aids",
        "MED_EXP":   "Medications",
    }
    _EXPENSE_RANGES = {
        1: "<$200", 2: "$200-$499", 3: "$500-$999",
        4: "$1k-$2k", 5: "$2k-$5k", 6: "$5k+",
    }
    avail_exp = {k: v for k, v in EXPENSE_COLS.items() if k in analysis_df.columns}

    if avail_exp:
        exp_df = analysis_df[list(avail_exp.keys())].copy()

        col_exp_a, col_exp_b = st.columns(2)
        with col_exp_a:
            # Count of respondents per expense bracket (across all categories combined)
            all_codes = []
            for k in avail_exp:
                vals = exp_df[k].dropna()
                all_codes.extend(vals.tolist())
            if all_codes:
                code_counts = pd.Series(all_codes).value_counts().sort_index()
                bracket_labels = [_EXPENSE_RANGES.get(int(c), str(c)) for c in code_counts.index]
                fig_exp = px.bar(
                    x=bracket_labels, y=code_counts.values,
                    title="All Expense Responses by Bracket (Across All 6 Categories)",
                    labels={"x": "Expense Bracket", "y": "Number of Responses"},
                    color_discrete_sequence=["#e07b54"],
                )
                fig_exp.update_layout(margin=dict(t=55))
                st.plotly_chart(fig_exp, width="stretch")

            n_answered = exp_df.notna().any(axis=1).sum()
            st.markdown(
                f"- **{n_answered:,}** respondents answered at least one expense question  \n"
                f"- The rest had 'Valid skip' (not asked due to survey routing)"
            )

        with col_exp_b:
            # Per-category: number of respondents who answered
            cat_rows = []
            for k, v in avail_exp.items():
                n_ans = int(exp_df[k].notna().sum())
                if n_ans == 0:
                    continue
                # % reporting $1,000+ (codes 4, 5, 6)
                n_high = int((exp_df[k] >= 4).sum())
                cat_rows.append({"Category": v, "n answered": n_ans,
                                 "% >= $1,000": round(n_high / n_ans * 100, 1)})
            cat_df = pd.DataFrame(cat_rows).sort_values("% >= $1,000", ascending=True)

            fig_cat = px.bar(
                cat_df, x="% >= $1,000", y="Category", orientation="h",
                title="% of Respondents Spending $1,000+ Per Category",
                labels={"% >= $1,000": "% Spending $1,000 or More", "Category": ""},
                color="% >= $1,000", color_continuous_scale="Oranges",
            )
            fig_cat.update_layout(coloraxis_showscale=False,
                                   height=360, margin=dict(l=10, r=20, t=50, b=20))
            st.plotly_chart(fig_cat, width="stretch")

    st.divider()

    # ── 7. Burnout item breakdown ───────────────────────────────────────────
    st.markdown("### 7. Which Burnout Indicators Are Most Prevalent?")
    st.markdown(
        "The 12 survey items that make up the burnout score each capture a "
        "different dimension of caregiver strain. This chart shows what fraction "
        "of respondents reported the highest-strain response for each item "
        "(among those who answered it)."
    )

    BURNOUT_LABELS = {
        "ICS_40":  "Overall caregiving is Very/Somewhat stressful",
        "FIS_10A": "Caregiving reduced time for other family members",
        "FIS_10B": "Caregiving caused family conflict or tension",
        "FIS_10C": "Caregiving required other family members to take on extra duties",
        "FIS_10D": "Caregiving limited ability to do things as a family",
        "FIS_10E": "Caregiving caused financial difficulties for the family",
        "FIS_10F": "Caregiving affected family members' school or work",
        "FIS_10G": "Caregiving affected the caregiver's own health",
        "FIS_10H": "Caregiving reduced caregiver's social activities",
        "CRH_10":  "Relationship with care receiver became more difficult",
        "CRH_20":  "Caregiver felt resentment or frustration toward care receiver",
        "CRH_30":  "Caregiver considered stopping or reducing caregiving",
    }

    # Load raw SAS data to access burnout items (they were excluded from X)
    _raw_main, _ = pyreadstat.read_sas7bdat(SAS_FILE)
    _raw_clean = _raw_main.drop(columns=[c for c in _raw_main.columns if c.startswith("WTBS_")])

    item_rows = []
    for col, label in BURNOUT_LABELS.items():
        if col not in _raw_clean.columns:
            continue
        # Raw codes: Binary 1=Yes(strain), 2=No; ICS_40 1=Very stressful .. 4=Not at all
        col_vals = _raw_clean[col].replace(RESERVE_CODES, np.nan).dropna()
        if len(col_vals) == 0:
            continue
        if col == "ICS_40":
            pct_strain = ((col_vals <= 2).sum() / len(col_vals)) * 100
            note = "Very or Stressful (top 2 of 4)"
        else:
            pct_strain = ((col_vals == 1).sum() / len(col_vals)) * 100
            note = "Yes (strain present)"
        item_rows.append({
            "Burnout Indicator": label,
            "pct": round(pct_strain, 1),
            "note": note,
        })

    if item_rows:
        item_df = pd.DataFrame(item_rows).sort_values("pct", ascending=True)
        fig_items = px.bar(
            item_df,
            x="pct",
            y="Burnout Indicator",
            orientation="h",
            title="Prevalence of Each Burnout Indicator (% of Respondents Who Answered)",
            labels={"pct": "% Reporting This Strain", "Burnout Indicator": ""},
            color="pct",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
            text="pct",
        )
        fig_items.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_items.update_layout(
            coloraxis_showscale=False,
            height=500,
            margin=dict(l=10, r=60, t=50, b=20),
        )
        st.plotly_chart(fig_items, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK DRIVERS (MODEL + SHAP)
# ══════════════════════════════════════════════════════════════════════════════
with tab_drivers:
    st.subheader("What Factors Drive Burnout Risk?")
    st.markdown(
        "An XGBoost machine-learning model was trained to predict which caregivers "
        "are at high risk of burnout. **SHAP values** then measure how much each "
        "survey variable pushed each person's predicted risk up or down — giving "
        "a transparent, factor-by-factor explanation for every respondent."
    )

    model, auc, fpr, tpr, report, X_test, y_test, y_prob = train_model()
    shap_df = get_shap_values()

    # --- Model accuracy banner ---
    st.markdown("#### Model Accuracy")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "ROC-AUC Score", f"{auc:.4f}",
        help="Ranges from 0.5 (random guessing) to 1.0 (perfect). "
             "Above 0.70 is considered useful for population screening.",
    )
    c2.metric(
        "Precision — High Burnout",
        f"{report['High Burnout']['precision']:.3f}",
        help="Of everyone the model flagged as high-risk, this fraction truly were.",
    )
    c3.metric(
        "Recall — High Burnout",
        f"{report['High Burnout']['recall']:.3f}",
        help="Of all truly high-risk people, this fraction were correctly identified.",
    )
    c4.metric("F1 Score — High Burnout", f"{report['High Burnout']['f1-score']:.3f}")

    st.divider()

    col_roc, col_shap = st.columns([1, 2])

    with col_roc:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"This model  (AUC = {auc:.3f})",
            line=dict(color="steelblue", width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name="Random guessing",
            line=dict(dash="dash", color="gray"),
        ))
        fig_roc.update_layout(
            title="ROC Curve — How Well the Model Separates Risk Groups",
            xaxis_title="False Positive Rate\n(High-risk caregivers incorrectly flagged as Low)",
            yaxis_title="True Positive Rate\n(High-risk caregivers correctly identified)",
            height=400,
            legend=dict(x=0.45, y=0.1),
        )
        st.plotly_chart(fig_roc, width="stretch")

    with col_shap:
        # Global feature importance (top 20 by mean |SHAP|)
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False).head(20)
        readable_names = [
            var_info.get(c, c)[:40] + ("…" if len(var_info.get(c, c)) > 40 else "")
            for c in mean_abs_shap.index
        ]
        fig_imp = px.bar(
            x=mean_abs_shap.values[::-1],
            y=readable_names[::-1],
            orientation="h",
            title="Top 20 Factors That Most Influence Burnout Risk Predictions",
            labels={"x": "Average influence on burnout prediction (SHAP value)", "y": ""},
            color=mean_abs_shap.values[::-1],
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(coloraxis_showscale=False, height=520,
                               margin=dict(l=10, r=20, t=50, b=20))
        st.plotly_chart(fig_imp, width="stretch")

    # --- SHAP beeswarm ---
    st.divider()
    st.subheader("How Does Each Factor Push Risk Up or Down?")
    st.markdown(
        "The chart below shows every respondent as a dot.  \n"
        "- **Position on the horizontal axis**: how much this factor increased "
        "(right of centre) or decreased (left of centre) that person's predicted "
        "burnout risk.  \n"
        "- **Dot colour**: red = respondent had a high value for that factor; "
        "blue = low value.  \n"
        "This reveals not just *which* factors matter, but *in which direction* "
        "they affect risk."
    )

    fig_bee = plt.figure(figsize=(11, 7))
    shap.summary_plot(
        shap_df.values, X,
        feature_names=feat_display_names,   # show plain English, not column codes
        max_display=15,
        show=False,
    )
    plt.xlabel("Impact on burnout risk prediction\n"
               "(← lowers risk   |   raises risk →)", fontsize=10)
    st.pyplot(fig_bee, width="stretch")
    plt.close(fig_bee)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VARIABLE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.subheader("Explore Any Survey Variable")
    st.markdown(
        "Choose any survey variable to see how its answer categories relate to "
        "high burnout risk in the **currently filtered** population. "
        "The table on the right shows the exact percentages."
    )

    # Only offer variables with a manageable number of distinct categories
    explorable = [
        c for c in filtered_df.columns
        if c not in ("burnout_high", "burnout_score", "WGHT_PER",
                     "flexibility_score", "ite_score")
        and filtered_df[c].nunique(dropna=True) <= 20
        and filtered_df[c].notna().sum() > 50
    ]
    var_options = {
        f"[{c}]  {var_info.get(c, c)[:50]}": c
        for c in explorable
    }

    sel_label = st.selectbox("Choose a survey variable", list(var_options.keys()))
    sel_var   = var_options[sel_label]

    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        st.plotly_chart(
            burnout_rate_chart(
                filtered_df, sel_var, label_map, var_info,
                f"High Burnout Rate by: {var_info.get(sel_var, sel_var)}",
            ),
            width="stretch",
        )

    with col_table:
        tmp = filtered_df[[sel_var, "burnout_high"]].dropna().copy()
        tmp["Answer"] = decode_col(tmp[sel_var], label_map, sel_var)
        freq = (
            tmp.dropna(subset=["Answer"])
            .groupby("Answer")
            .agg(
                Respondents=(sel_var, "count"),
                high_rate=("burnout_high", "mean"),
            )
            .reset_index()
        )
        freq["High Burnout %"] = (freq["high_rate"] * 100).round(1)
        freq = freq[["Answer", "Respondents", "High Burnout %"]]
        freq = freq.sort_values("High Burnout %", ascending=False)
        st.dataframe(freq, width="stretch", hide_index=True)

    # --- SHAP distribution for the selected variable ---
    st.divider()
    st.subheader(f"How Strongly Does This Factor Influence Burnout Predictions?")
    st.markdown(
        f"The histogram below shows the distribution of SHAP values for "
        f"**{var_info.get(sel_var, sel_var)}** across all respondents.  \n"
        f"Values to the **right of zero** increase predicted burnout risk; "
        f"values to the **left** decrease it."
    )

    if sel_var in shap_df.columns:
        fig_dist = px.histogram(
            shap_df[sel_var], nbins=40,
            title=f"Influence Distribution: {var_info.get(sel_var, sel_var)[:65]}",
            labels={
                "value": "SHAP value  (positive = raises burnout risk, "
                          "negative = lowers it)",
                "count": "Number of Respondents",
            },
            color_discrete_sequence=["#5b9bd5"],
        )
        fig_dist.add_vline(x=0, line_dash="dash", line_color="gray",
                           annotation_text="No effect on prediction")
        st.plotly_chart(fig_dist, width="stretch")
    else:
        st.info(
            f"SHAP values are not available for **{var_info.get(sel_var, sel_var)}** "
            f"because it is a composite score or weight column, not a direct model input."
        )
