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

Requirements (install if missing):
    pip install streamlit plotly shap xgboost scikit-learn pyreadstat
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
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# Reserve codes treated as NaN throughout the pipeline.
# Valid skip  (not applicable): 6, 96, 996, 9996, 99996, 999996
# Not stated  (missing/refused): 9, 99, 999, 9999, 99999, 999999
RESERVE_CODES = [
    6, 9, 96, 99, 996, 999,
    9996, 9999, 99996, 99999, 999996, 999999,
]

# 12 validated burnout items (ICS_40 = overall strain; FIS_10A-H = family impact;
# CRH_10/20/30 = caregiver relationship health). All are 1-digit Yes/No.
BURNOUT_ITEMS = [
    "ICS_40",
    "FIS_10A", "FIS_10B", "FIS_10C", "FIS_10D",
    "FIS_10E", "FIS_10F", "FIS_10G", "FIS_10H",
    "CRH_10", "CRH_20", "CRH_30",
]

# FWA / ITE variable lists for composite scores
FWA_POS = ["FWA_132", "FWA_133", "FWA_134", "FWA_136", "FWA_137"]
FWA_NEG = ["FWA_150"]
ITE_POS = ["ITE_30A", "ITE_30B", "ITE_30C", "ITE_30D", "ITE_30E"]
ITE_NEG = ["ITE_10"]

# All predictor variables (matches pipeline.py INCLUDE_VARS)
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

# Reserve-code labels to drop from the display label map
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

@st.cache_data(show_spinner="Loading codebook answer labels…")
def load_label_map():
    """
    Build {variable_name: {numeric_code: human_label}} from codebook_answer_categories.csv.

    The codebook contains one row per answer option for each variable.
    Reserve-code rows (Valid skip, Don't know, Refusal, Not stated) are
    excluded because those values are treated as NaN in the analysis.

    PDF-extraction artifact: "Y es" is normalised to "Yes".
    """
    cats = pd.read_csv(ANSWER_CATS)
    # Normalise PDF extraction artifacts (e.g. "Y es" → "Yes")
    cats["label"] = cats["label"].str.strip().str.replace(r"^Y\s+es$", "Yes", regex=True)
    # Drop reserve-code rows — they won't appear in the cleaned data
    cats = cats[~cats["label"].isin(_RESERVE_LABELS)]

    label_map: dict[str, dict[int, str]] = {}
    for _, row in cats.iterrows():
        var = row["variable_name"]
        try:
            code = int(float(row["code"]))
        except (ValueError, TypeError):
            continue  # skip range codes like "00 - 84"
        label_map.setdefault(var, {})[code] = str(row["label"])
    return label_map


@st.cache_data(show_spinner="Loading variable descriptions…")
def load_var_info():
    """
    Build {variable_name: short_description} from the codebook variables CSV.
    Prefers the 'concept' field; falls back to the first 80 chars of question_text.
    """
    df = pd.read_csv(VAR_CSV, usecols=["variable_name", "concept", "question_text"])
    result = {}
    for _, row in df.iterrows():
        label = row["concept"]
        if pd.isna(label) or str(label).strip() == "":
            label = str(row["question_text"])[:80] if pd.notna(row["question_text"]) else row["variable_name"]
        result[row["variable_name"]] = str(label).strip()
    return result


@st.cache_data(show_spinner="Loading and processing survey data (first run only — takes ~30 s)…")
def load_and_process():
    """
    Full pipeline:
      1. Load SAS PUMF file
      2. Drop WTBS_ bootstrap weight columns
      3. Build flexibility_score and ite_score composite columns
      4. Build Include Variables dataframe with reserve codes → NaN
      5. Compute burnout target (12-item mean, median split → binary)
      6. Return X, y_clf, y_reg, W (survey weight), and the full analysis dataframe

    Returns a dict with keys:
        X             – feature matrix (DataFrame)
        y_clf         – binary burnout label (Series, 0/1)
        y_reg         – continuous burnout score (Series, 0–1)
        W             – WGHT_PER survey weights (Series)
        include_df    – aligned include-vars DataFrame (includes WGHT_PER)
        burnout_score – same as y_reg
        median_cutoff – threshold used for binary split
    """
    main, _ = pyreadstat.read_sas7bdat(SAS_FILE)

    # ── Step 1: Drop replicate bootstrap weights ──────────────────────────
    main_clean = main.drop(columns=[c for c in main.columns if c.startswith("WTBS_")])

    # ── Step 2: Composite scores ──────────────────────────────────────────
    # All FWA/ITE items are 1-digit binary (1=Yes, 2=No).
    # Recode Yes→1, No→0, reserve codes→NaN before summing.
    scores_base = main_clean.copy()
    for col in FWA_POS + FWA_NEG + ITE_POS + ITE_NEG:
        if col in scores_base.columns:
            scores_base[col] = scores_base[col].replace(RESERVE_CODES, np.nan)
            scores_base[col] = scores_base[col].map({1: 1, 2: 0})

    # flexibility_score (-1 to 5): work-flexibility options minus career-penalty flag
    scores_base["flexibility_score"] = (
        scores_base[FWA_POS].sum(axis=1, min_count=1)
        - scores_base[FWA_NEG[0]].fillna(0)
    )
    # ite_score (-1 to 5): enabling circumstances minus already-had-to-quit flag
    scores_base["ite_score"] = (
        scores_base[ITE_POS].sum(axis=1, min_count=1)
        - scores_base[ITE_NEG[0]].fillna(0)
    )

    # ── Step 3: Include Variables dataframe ───────────────────────────────
    available = [c for c in INCLUDE_VARS if c in scores_base.columns]
    include_df = scores_base[available].copy()
    non_composite = [
        c for c in available
        if c not in ("flexibility_score", "ite_score", "WGHT_PER")
    ]
    for col in non_composite:
        include_df[col] = include_df[col].replace(RESERVE_CODES, np.nan)

    # ── Step 4: Burnout target ─────────────────────────────────────────────
    # Mean of 12 burnout items (after replacing reserve codes with NaN).
    # Higher score = more burnout (scale 1–4 for ICS_40; binary for FIS/CRH).
    # Median-split creates balanced 0/1 classes.
    burnout_base = main_clean[BURNOUT_ITEMS].replace(RESERVE_CODES, np.nan)
    burnout_score_full = burnout_base.mean(axis=1)
    median_cutoff = burnout_score_full.median()
    burnout_high_full = (burnout_score_full >= median_cutoff).astype(int)

    valid_idx = burnout_score_full.dropna().index
    include_aligned = include_df.loc[valid_idx].copy()
    y_score = burnout_score_full.loc[valid_idx]
    y_clf = burnout_high_full.loc[valid_idx]

    # ── Step 5: X / W separation ──────────────────────────────────────────
    EXCLUDE_FROM_X = ["WGHT_PER"] + [c for c in BURNOUT_ITEMS if c in include_aligned.columns]
    X = include_aligned.drop(columns=[c for c in EXCLUDE_FROM_X if c in include_aligned.columns])
    W = include_aligned["WGHT_PER"] if "WGHT_PER" in include_aligned.columns else pd.Series(
        np.ones(len(include_aligned)), index=include_aligned.index
    )

    return {
        "X": X,
        "y_clf": y_clf,
        "y_reg": y_score,
        "W": W,
        "include_df": include_aligned,
        "burnout_score": y_score,
        "median_cutoff": median_cutoff,
    }


@st.cache_resource(show_spinner="Training XGBoost model (first run only — ~1 min)…")
def train_model():
    """
    Train the XGBoost burnout classifier and return model + evaluation metrics.

    Uses survey weights (WGHT_PER) as sample_weight so the model respects
    the complex survey design.

    XGBoost handles NaN natively — no imputation required.

    Returns: (model, auc, fpr, tpr, report_dict, X_test, y_test, y_prob)
    """
    data = load_and_process()
    X, y_clf, W = data["X"], data["y_clf"], data["W"]

    # ── XGBoost configuration ─────────────────────────────────────────────
    XGB_CONFIG = dict(
        n_estimators          = 500,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 10,
        scale_pos_weight      = 1,
        eval_metric           = "logloss",
        early_stopping_rounds = 30,
        random_state          = 42,
        n_jobs                = -1,
    )

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y_clf, W.values,
        test_size=0.20, random_state=42, stratify=y_clf,
    )

    model = XGBClassifier(**XGB_CONFIG)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    report = classification_report(
        y_test, y_pred,
        target_names=["Low Burnout", "High Burnout"],
        output_dict=True,
    )

    return model, auc, fpr, tpr, report, X_test, y_test, y_prob


@st.cache_data(show_spinner="Computing SHAP values for all respondents (this may take 1–2 min)…")
def get_shap_values():
    """
    Compute SHAP values for every respondent in X (not just the test set).
    This gives each person a risk-driver profile used for clustering.

    Returns shap_df: DataFrame (n_respondents × n_features).
    """
    data = load_and_process()
    X = data["X"]
    model, *_ = train_model()
    explainer = shap.TreeExplainer(model)
    shap_arr = explainer.shap_values(X)
    return pd.DataFrame(shap_arr, index=X.index, columns=X.columns)


@st.cache_data(show_spinner="Clustering respondents by risk-driver pattern…")
def get_clusters():
    """
    K-Means clustering on the SHAP value matrix (StandardScaler applied first).
    Auto-selects k (2–8) using silhouette score.

    Returns: (cluster_labels array, best_k, k_list, silhouette_scores)
    """
    shap_df = get_shap_values()
    K_RANGE = range(2, 9)

    scaler = StandardScaler()
    shap_scaled = scaler.fit_transform(shap_df)

    silhouettes = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(shap_scaled)
        silhouettes.append(
            silhouette_score(shap_scaled, labels, sample_size=2000, random_state=42)
        )

    best_k = K_RANGE.start + int(np.argmax(silhouettes))
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    cluster_labels = km_final.fit_predict(shap_scaled)

    return cluster_labels, best_k, list(K_RANGE), silhouettes


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def decode_col(series: pd.Series, label_map: dict, var_name: str) -> pd.Series:
    """
    Translate a numeric-coded Series to its human-readable labels.
    Values not in the label map are left as-is. NaN stays NaN.
    """
    mapping = label_map.get(var_name, {})
    if not mapping:
        return series
    def _decode(x):
        if pd.isna(x):
            return np.nan
        try:
            return mapping.get(int(x), x)
        except (ValueError, TypeError):
            return x
    return series.map(_decode)


def burnout_rate_chart(df: pd.DataFrame, var: str, label_map: dict,
                       var_info: dict, title: str = None) -> go.Figure:
    """
    Horizontal bar chart: high-burnout rate (%) for each category of `var`.
    Bars are coloured on a red-green scale (higher = more red).
    """
    tmp = df[[var, "burnout_high"]].dropna().copy()
    tmp["_label"] = decode_col(tmp[var], label_map, var)
    tmp = tmp.dropna(subset=["_label"])

    stats = (
        tmp.groupby("_label")["burnout_high"]
        .agg(burnout_rate="mean", count="count")
        .reset_index()
    )
    stats["pct"] = (stats["burnout_rate"] * 100).round(1)
    stats = stats.sort_values("pct")

    var_label = var_info.get(var, var)
    fig = px.bar(
        stats, x="pct", y="_label", orientation="h",
        text="pct",
        labels={"pct": "High Burnout Rate (%)", "_label": ""},
        title=title or f"High Burnout Rate by {var_label}",
        color="pct",
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 100],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        height=max(280, len(stats) * 36 + 80),
        margin=dict(l=10, r=40, t=50, b=30),
    )
    return fig


# =============================================================================
# LOAD DATA + CODEBOOK (runs once; cached thereafter)
# =============================================================================
label_map = load_label_map()
var_info  = load_var_info()
data      = load_and_process()

X             = data["X"]
y_clf         = data["y_clf"]
y_reg         = data["y_reg"]
W             = data["W"]
include_df    = data["include_df"]
burnout_score = data["burnout_score"]
median_cutoff = data["median_cutoff"]

# Build the main analysis dataframe used throughout the dashboard
analysis_df = X.copy()
analysis_df["burnout_high"]  = y_clf
analysis_df["burnout_score"] = y_reg
analysis_df["WGHT_PER"]      = W


# =============================================================================
# SIDEBAR — DEMOGRAPHIC FILTERS
# =============================================================================
st.sidebar.title("🧡 Caregiver Burnout Risk")
st.sidebar.markdown("**GSS 2018 — Statistics Canada**")
st.sidebar.divider()
st.sidebar.markdown("### Filter Population")
st.sidebar.markdown("Use these filters to zoom into a specific group.")

# Variables available as filters (must be categorical & present in analysis_df)
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

# Apply filters
filter_mask = pd.Series(True, index=analysis_df.index)
for col, code in active_filters.items():
    filter_mask &= (analysis_df[col] == code)
filtered_df = analysis_df[filter_mask]

if active_filters:
    st.sidebar.success(f"{filter_mask.sum():,} respondents match your filters.")
else:
    st.sidebar.info(f"{len(analysis_df):,} respondents (no filters applied).")

st.sidebar.divider()
st.sidebar.caption(
    "Data: [GSS 2018 Caregiving PUMF](https://www150.statcan.gc.ca/)  \n"
    "Statistics Canada, Cycle 32"
)


# =============================================================================
# MAIN TITLE
# =============================================================================
st.title("🧡 Caregiver Burnout Risk Screener")
st.markdown(
    "*Identifying at-risk caregivers to guide government and nonprofit resource allocation*"
)

# =============================================================================
# TABS
# =============================================================================
tab_overview, tab_drivers, tab_segments, tab_explore = st.tabs([
    "📊 Overview",
    "📈 Risk Drivers",
    "🧩 Caregiver Segments",
    "🔬 Variable Explorer",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    n_total   = len(filtered_df)
    n_high    = int(filtered_df["burnout_high"].sum())
    pct_high  = n_high / n_total * 100 if n_total > 0 else 0.0
    avg_score = filtered_df["burnout_score"].mean()

    # KPI metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents (filtered)", f"{n_total:,}")
    c2.metric(
        "High Burnout Risk",
        f"{pct_high:.1f}%",
        f"{n_high:,} people",
    )
    c3.metric(
        "Avg Burnout Score",
        f"{avg_score:.3f}",
        f"median threshold: {median_cutoff:.3f}",
    )
    if "flexibility_score" in filtered_df.columns:
        avg_flex = filtered_df["flexibility_score"].mean()
        c4.metric("Avg Flexibility Score", f"{avg_flex:.2f}", "(range −1 to 5)")

    st.divider()

    # Row 1: Province + Sex
    col_a, col_b = st.columns(2)
    with col_a:
        if "PRV" in filtered_df.columns:
            st.plotly_chart(
                burnout_rate_chart(filtered_df, "PRV", label_map, var_info,
                                   "High Burnout Rate by Province"),
                use_container_width=True,
            )
    with col_b:
        if "SEX" in filtered_df.columns:
            st.plotly_chart(
                burnout_rate_chart(filtered_df, "SEX", label_map, var_info,
                                   "High Burnout Rate by Sex"),
                use_container_width=True,
            )

    # Row 2: Score distribution + Marital status
    col_c, col_d = st.columns(2)
    with col_c:
        fig_hist = px.histogram(
            filtered_df, x="burnout_score", nbins=30,
            title="Burnout Score Distribution",
            labels={"burnout_score": "Mean burnout score (12 items)", "count": "Respondents"},
            color_discrete_sequence=["#e07b54"],
        )
        fig_hist.add_vline(
            x=median_cutoff, line_dash="dash", line_color="red",
            annotation_text=f"Median split: {median_cutoff:.2f}",
            annotation_position="top right",
        )
        fig_hist.update_layout(margin=dict(t=50))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_d:
        if "MARSTAT" in filtered_df.columns:
            st.plotly_chart(
                burnout_rate_chart(filtered_df, "MARSTAT", label_map, var_info,
                                   "High Burnout Rate by Marital Status"),
                use_container_width=True,
            )

    # Row 3: Income + Visible minority
    col_e, col_f = st.columns(2)
    with col_e:
        if "FAMINCG1" in filtered_df.columns:
            st.plotly_chart(
                burnout_rate_chart(filtered_df, "FAMINCG1", label_map, var_info,
                                   "High Burnout Rate by Household Income"),
                use_container_width=True,
            )
    with col_f:
        if "VISMIN" in filtered_df.columns:
            st.plotly_chart(
                burnout_rate_chart(filtered_df, "VISMIN", label_map, var_info,
                                   "High Burnout Rate by Visible Minority Status"),
                use_container_width=True,
            )

    # Composite scores vs burnout
    st.divider()
    st.subheader("Composite Scores vs Burnout Risk")
    col_g, col_h = st.columns(2)

    def _score_bar(df, score_col, xlabel, title):
        tmp = df[[score_col, "burnout_high"]].dropna()
        stats = (
            tmp.groupby(score_col)["burnout_high"]
            .agg(pct="mean", count="count")
            .reset_index()
        )
        stats["pct"] = (stats["pct"] * 100).round(1)
        fig = px.bar(
            stats, x=score_col, y="pct",
            title=title,
            labels={score_col: xlabel, "pct": "High Burnout Rate (%)"},
            color="pct", color_continuous_scale="RdYlGn_r", range_color=[0, 100],
            text="pct",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 105])
        return fig

    with col_g:
        if "flexibility_score" in filtered_df.columns:
            st.plotly_chart(
                _score_bar(filtered_df, "flexibility_score",
                           "Flexibility Score (−1 to 5)",
                           "High Burnout Rate by Workplace Flexibility Score"),
                use_container_width=True,
            )
    with col_h:
        if "ite_score" in filtered_df.columns:
            st.plotly_chart(
                _score_bar(filtered_df, "ite_score",
                           "ITE Score (−1 to 5)",
                           "High Burnout Rate by Enabling-to-Work Score"),
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK DRIVERS (MODEL + SHAP)
# ══════════════════════════════════════════════════════════════════════════════
with tab_drivers:
    st.subheader("Model Performance")
    st.markdown(
        "An XGBoost classifier predicts high-burnout risk from ~140 survey variables. "
        "SHAP values then show **how much** and **in which direction** each variable "
        "drives the burnout prediction for every respondent."
    )

    model, auc, fpr, tpr, report, X_test, y_test, y_prob = train_model()
    shap_df = get_shap_values()

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC",
              f"{auc:.4f}",
              help="1.0 = perfect; 0.5 = random. >0.70 is useful for screening.")
    c2.metric("Precision (High Burnout)",
              f"{report['High Burnout']['precision']:.3f}",
              help="Of those flagged high-risk, what fraction truly are?")
    c3.metric("Recall (High Burnout)",
              f"{report['High Burnout']['recall']:.3f}",
              help="Of all truly high-risk people, what fraction did we catch?")
    c4.metric("F1 (High Burnout)",
              f"{report['High Burnout']['f1-score']:.3f}")

    st.divider()

    col_roc, col_shap = st.columns([1, 2])

    with col_roc:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"Model (AUC={auc:.3f})",
            line=dict(color="steelblue", width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random chance",
            line=dict(dash="dash", color="gray"),
        ))
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=380,
            legend=dict(x=0.55, y=0.1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_shap:
        # Global SHAP importance bar
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False).head(20)
        feat_labels = [
            var_info.get(c, c)[:55] + ("…" if len(var_info.get(c, c)) > 55 else "")
            for c in mean_abs_shap.index
        ]
        fig_shap_bar = px.bar(
            x=mean_abs_shap.values[::-1],
            y=feat_labels[::-1],
            orientation="h",
            title="Top 20 Global Risk Drivers (Mean |SHAP Value|)",
            labels={"x": "Mean |SHAP value|", "y": ""},
            color=mean_abs_shap.values[::-1],
            color_continuous_scale="Blues",
        )
        fig_shap_bar.update_layout(
            coloraxis_showscale=False,
            height=500,
            margin=dict(l=10, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_shap_bar, use_container_width=True)

    st.divider()
    st.subheader("Feature Impact Direction (SHAP Beeswarm)")
    st.markdown(
        "Each dot represents one respondent.  "
        "**Red** = high feature value, **blue** = low feature value.  "
        "X-axis position shows whether the feature **increases** (right) or "
        "**decreases** (left) burnout risk."
    )

    fig_bee = plt.figure(figsize=(11, 7))
    shap.summary_plot(
        shap_df.values, X,
        max_display=15,
        show=False,
    )
    st.pyplot(fig_bee, use_container_width=True)
    plt.close(fig_bee)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CAREGIVER SEGMENTS (CLUSTERS)
# ══════════════════════════════════════════════════════════════════════════════
with tab_segments:
    st.subheader("Caregiver Segments")
    st.markdown(
        "Caregivers are grouped by **shared risk-driver patterns** using K-Means "
        "clustering on SHAP values — not by burnout level alone. Each segment "
        "represents a group with a distinct combination of factors driving their risk. "
        "This helps you target the **right intervention** for each group."
    )

    cluster_labels, best_k, k_list, silhouettes = get_clusters()
    shap_df_c = get_shap_values()
    shap_feat_cols = list(X.columns)

    # Silhouette plot
    with st.expander("Cluster selection — silhouette scores"):
        fig_sil = px.line(
            x=k_list, y=silhouettes, markers=True,
            title="Silhouette Score by Number of Clusters",
            labels={"x": "k (number of clusters)", "y": "Silhouette Score"},
        )
        fig_sil.add_vline(
            x=best_k, line_dash="dash", line_color="red",
            annotation_text=f"Selected k={best_k}",
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    st.divider()

    # Summary table: one row per cluster
    summary_rows = []
    for c_id in range(best_k):
        pos_idx = np.where(cluster_labels == c_id)[0]
        n_c  = len(pos_idx)
        br_c = float(y_clf.iloc[pos_idx].mean()) * 100
        mean_shap = shap_df_c.iloc[pos_idx][shap_feat_cols].abs().mean()
        top_feat  = mean_shap.idxmax()
        top_label = var_info.get(top_feat, top_feat)
        summary_rows.append({
            "Segment":          f"Segment {c_id + 1}",
            "Size":             n_c,
            "% of Total":       f"{n_c / len(cluster_labels) * 100:.1f}%",
            "High Burnout Rate": f"{br_c:.1f}%",
            "Top Risk Driver":  top_label[:70] + ("…" if len(top_label) > 70 else ""),
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.divider()

    # Per-segment detail
    seg_choice = st.selectbox(
        "Select a segment to explore in detail",
        [f"Segment {i + 1}" for i in range(best_k)],
    )
    c_sel = int(seg_choice.split()[-1]) - 1
    pos_idx_sel = np.where(cluster_labels == c_sel)[0]

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Signed SHAP mean → direction of effect
        mean_shap_signed = shap_df_c.iloc[pos_idx_sel][shap_feat_cols].mean()
        top10 = mean_shap_signed.abs().sort_values(ascending=False).head(10)
        top10_labels = [
            var_info.get(f, f)[:50] + ("…" if len(var_info.get(f, f)) > 50 else "")
            for f in top10.index
        ]
        directions = [
            "▲ Raises Risk" if mean_shap_signed[f] > 0 else "▼ Lowers Risk"
            for f in top10.index
        ]
        bar_colors = [
            "#d73027" if mean_shap_signed[f] > 0 else "#1a9850"
            for f in top10.index
        ]

        fig_seg = go.Figure(go.Bar(
            x=top10.values[::-1],
            y=[f"{l} ({d})" for l, d in zip(top10_labels[::-1], directions[::-1])],
            orientation="h",
            marker_color=bar_colors[::-1],
        ))
        fig_seg.update_layout(
            title=f"{seg_choice} — Top 10 Risk Drivers",
            xaxis_title="Mean |SHAP value|",
            height=420,
            margin=dict(l=10, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_right:
        # Burnout rate comparison
        br_overall = float(y_clf.mean()) * 100
        br_segment = float(y_clf.iloc[pos_idx_sel].mean()) * 100

        fig_br = go.Figure(go.Bar(
            x=[br_overall, br_segment],
            y=["Overall", seg_choice],
            orientation="h",
            marker_color=["#7fbfff", "#e07b54"],
            text=[f"{br_overall:.1f}%", f"{br_segment:.1f}%"],
            textposition="outside",
        ))
        fig_br.update_layout(
            title="High Burnout Rate vs Overall",
            xaxis_title="High Burnout Rate (%)",
            xaxis_range=[0, 105],
            height=200,
            margin=dict(l=10, r=40, t=50, b=20),
        )
        st.plotly_chart(fig_br, use_container_width=True)

        # Demographic mode snapshot
        DEMO_COLS = ["SEX", "PRV", "FAMINCG1", "MARSTAT", "VISMIN",
                     "SENFLAG", "COW_10", "UWS230GR"]
        demo_avail = [c for c in DEMO_COLS if c in analysis_df.columns]
        demo_rows = []
        for col in demo_avail:
            vals = analysis_df.iloc[pos_idx_sel][col].dropna()
            if len(vals) == 0:
                continue
            mode_code  = vals.mode().iloc[0]
            mode_label = label_map.get(col, {}).get(int(mode_code), str(mode_code))
            pct        = (vals == mode_code).mean() * 100
            demo_rows.append({
                "Variable":       var_info.get(col, col)[:38],
                "Most Common":    str(mode_label)[:30],
                "% in Segment":   f"{pct:.1f}%",
            })
        if demo_rows:
            st.markdown(f"**{seg_choice} — Demographic Snapshot**")
            st.dataframe(pd.DataFrame(demo_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VARIABLE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.subheader("Variable Explorer")
    st.markdown(
        "Select any survey variable to see how its categories relate to high "
        "burnout rate in the **currently filtered** population."
    )

    # Only show variables with a reasonable number of categories (≤ 20 unique)
    explorable = [
        c for c in filtered_df.columns
        if c not in ("burnout_high", "burnout_score", "WGHT_PER",
                     "flexibility_score", "ite_score")
        and filtered_df[c].nunique(dropna=True) <= 20
        and filtered_df[c].notna().sum() > 50
    ]
    var_display = {
        f"{c}  —  {var_info.get(c, c)[:65]}": c
        for c in explorable
    }

    sel_label = st.selectbox("Choose a variable", list(var_display.keys()))
    sel_var   = var_display[sel_label]

    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        st.plotly_chart(
            burnout_rate_chart(
                filtered_df, sel_var, label_map, var_info,
                f"High Burnout Rate by {var_info.get(sel_var, sel_var)}",
            ),
            use_container_width=True,
        )

    with col_table:
        tmp = filtered_df[[sel_var, "burnout_high"]].dropna().copy()
        tmp["_label"] = decode_col(tmp[sel_var], label_map, sel_var)
        freq = (
            tmp.dropna(subset=["_label"])
            .groupby("_label")
            .agg(Count=(sel_var, "count"), HighBurnout=("burnout_high", "mean"))
            .reset_index()
        )
        freq["High Burnout %"] = (freq["HighBurnout"] * 100).round(1)
        freq = freq.rename(columns={"_label": "Category"})[["Category", "Count", "High Burnout %"]]
        freq = freq.sort_values("High Burnout %", ascending=False)
        st.dataframe(freq, use_container_width=True, hide_index=True)

    # SHAP individual-feature contribution
    st.divider()
    st.subheader("SHAP Value Distribution for Selected Variable")
    st.markdown(
        "How much does this variable shift the burnout prediction for each respondent?"
    )

    if sel_var in shap_df.columns:
        shap_feature = shap_df[sel_var]
        fig_shap_dist = px.histogram(
            shap_feature, nbins=40,
            title=f"SHAP Distribution: {var_info.get(sel_var, sel_var)[:60]}",
            labels={"value": "SHAP value (+ = raises burnout risk)", "count": "Respondents"},
            color_discrete_sequence=["#5b9bd5"],
        )
        fig_shap_dist.add_vline(x=0, line_dash="dash", line_color="gray",
                                annotation_text="No effect")
        st.plotly_chart(fig_shap_dist, use_container_width=True)
    else:
        st.info(f"{sel_var} is not in the SHAP feature set (composite or weight column).")
