from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damb.config import (
    ARTIFACT_MANIFEST_PATH,
    AGE_GROUP_LABELS,
    BINARY_FEATURES,
    GLOBAL_SHAP_PATH,
    HEATMAP_PATHS,
    HEALTH_CONDITION_LABELS,
    HIGH_RISK_INDIVIDUALS_PATH,
    HIGH_RISK_SEGMENTS_PATH,
    HOUR_GROUP_LABELS,
    ID_COLUMN,
    INCOME_GROUP_LABELS,
    METRICS_PATH,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    PROVINCE_LABELS,
    RAW_DATA_PATH,
    SCORED_DATA_PATH,
    SEX_LABELS,
    SUBGROUP_DRIVERS_PATH,
    SUBGROUP_RISK_PATH,
    VALIDATION_PATH,
)
from damb.pipeline import prepare_training_frame


st.set_page_config(page_title="Caregiver Distress Analytics", layout="wide")


def _exists(path: Path) -> bool:
    return Path(path).exists()


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def _load_analytic_frame() -> pd.DataFrame:
    prepared = prepare_training_frame(str(RAW_DATA_PATH))
    return prepared.frame.copy()


def _correlation_columns(frame: pd.DataFrame) -> list[str]:
    ordered = [*NUMERIC_FEATURES, *ORDINAL_FEATURES, *BINARY_FEATURES]
    columns = [column for column in ordered if column in frame.columns and column in MODEL_FEATURES]
    usable = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().sum() >= 25 and series.nunique(dropna=True) > 1:
            usable.append(column)
    return usable


def _plot_correlation_heatmap(correlation: pd.DataFrame) -> plt.Figure:
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        correlation,
        mask=mask,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        linewidths=0.4,
        square=False,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Spearman Correlation Matrix")
    fig.tight_layout()
    return fig


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(34, 94, 168, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(11, 132, 96, 0.10), transparent 24%),
                linear-gradient(180deg, #f6f8fb 0%, #eef3f7 100%);
        }
        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1f2e 0%, #152b3f 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        [data-testid="stSidebar"] * {
            color: #edf3f8;
        }
        .hero-panel {
            background: linear-gradient(135deg, #13304a 0%, #1f5c7a 54%, #2d8f83 100%);
            color: white;
            border-radius: 24px;
            padding: 1.35rem 1.5rem;
            box-shadow: 0 18px 40px rgba(17, 43, 67, 0.18);
            margin-bottom: 1rem;
        }
        .hero-eyebrow {
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.78rem;
            opacity: 0.82;
            margin-bottom: 0.35rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            color: #f7fbff;
        }
        .hero-copy {
            font-size: 1rem;
            opacity: 0.96;
            max-width: 68rem;
            margin-bottom: 0;
            color: rgba(247, 251, 255, 0.92);
        }
        .section-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(20, 44, 63, 0.08);
            border-radius: 22px;
            padding: 1rem 1rem 0.4rem 1rem;
            box-shadow: 0 14px 30px rgba(20, 44, 63, 0.08);
            backdrop-filter: blur(6px);
            margin-bottom: 1rem;
            color: #102331;
        }
        .section-card h1, .section-card h2, .section-card h3, .section-card p,
        .section-card div, .section-card label, .section-card span {
            color: #102331;
        }
        .stat-chip {
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(19,48,74,0.08);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.65);
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(20,44,63,0.08);
            padding: 1rem 1rem 0.8rem 1rem;
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(20, 44, 63, 0.07);
        }
        div[data-testid="stMetricLabel"] {
            font-weight: 600;
        }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"] {
            color: #17384d;
        }
        .section-card [data-testid="stMarkdownContainer"] p,
        .section-card [data-testid="stMarkdownContainer"] li,
        .section-card [data-testid="stCaptionContainer"] {
            color: #17384d;
        }
        .hero-panel [data-testid="stMarkdownContainer"] p,
        .hero-panel p,
        .hero-panel span,
        .hero-panel div {
            color: rgba(247, 251, 255, 0.94);
        }
        [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        [data-baseweb="tab"] {
            background: rgba(255,255,255,0.72);
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
            color: #17384d;
        }
        [aria-selected="true"][data-baseweb="tab"] {
            background: #163a56;
            color: #f7fbff;
        }
        .main [data-testid="stHeading"] h2,
        .main [data-testid="stHeading"] h3 {
            color: #163247;
        }
        .sidebar-note {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            font-size: 0.92rem;
            line-height: 1.45;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _label_mapping(column: str) -> dict[int, str] | None:
    return {
        "PRV": PROVINCE_LABELS,
        "AGEGR10": AGE_GROUP_LABELS,
        "SEX": SEX_LABELS,
        "HAP_10C": HOUR_GROUP_LABELS,
        "TTLINCG1": INCOME_GROUP_LABELS,
        "PRA_10GR": HEALTH_CONDITION_LABELS,
    }.get(column)


def _display_series(series: pd.Series, column: str) -> pd.Series:
    mapping = _label_mapping(column)
    if mapping is None:
        return series.fillna("Missing").astype(str)
    def convert(value: object) -> str:
        if pd.isna(value):
            return "Missing"
        try:
            return mapping.get(int(float(value)), str(value))
        except (TypeError, ValueError):
            return str(value)
    return series.map(convert)


def _friendly_label(column: str) -> str:
    return {
        "PUMFID": "Respondent ID",
        "WGHT_PER": "Survey weight",
        "PRV": "Province",
        "HAP_10C": "Weekly care hours",
        "TTLINCG1": "Personal income group",
        "PRA_10GR": "Main health condition",
        "AGEGR10": "Age group",
        "predicted_probability_raw": "Predicted probability (raw)",
        "predicted_probability": "Predicted probability",
        "predicted_label": "High-priority flag",
        "risk_band": "Risk band",
        "dimension": "Dimension",
        "group_value": "Group",
        "avg_predicted_risk": "Average predicted risk",
        "high_priority_rate": "High-priority rate",
        "high_priority_weighted_count": "High-priority weighted count",
        "weighted_count": "Weighted count",
        "unweighted_count": "Respondents",
        "segment_type": "Segment type",
        "segment_value_1": "Segment value 1",
        "segment_value_2": "Segment value 2",
        "feature": "Feature",
        "mean_abs_shap": "Mean absolute SHAP",
    }.get(column, column.replace("_", " ").title())


def _format_display_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ["PRV", "HAP_10C", "TTLINCG1", "PRA_10GR", "AGEGR10", "SEX"]:
        if column in display.columns:
            display[column] = _display_series(display[column], column)
    rename_map = {column: _friendly_label(column) for column in display.columns}
    return display.rename(columns=rename_map)


def _risk_filter_options(scored: pd.DataFrame, column: str) -> list[str]:
    if column not in scored.columns:
        return []
    return sorted(_display_series(scored[column], column).dropna().astype(str).unique().tolist())


def main() -> None:
    _inject_styles()

    required = [
        METRICS_PATH,
        VALIDATION_PATH,
        SCORED_DATA_PATH,
        SUBGROUP_RISK_PATH,
        HIGH_RISK_SEGMENTS_PATH,
        HIGH_RISK_INDIVIDUALS_PATH,
        GLOBAL_SHAP_PATH,
    ]
    missing = [str(path) for path in required if not _exists(path)]
    if missing:
        st.error("Training artifacts are missing. Run `python scripts/train_binary_model.py` first.")
        st.code("\n".join(missing))
        return

    metrics = _load_json(METRICS_PATH)
    validation = _load_json(VALIDATION_PATH)
    scored = pd.read_csv(SCORED_DATA_PATH)
    subgroup = pd.read_csv(SUBGROUP_RISK_PATH)
    segments = pd.read_csv(HIGH_RISK_SEGMENTS_PATH)
    individuals = pd.read_csv(HIGH_RISK_INDIVIDUALS_PATH)
    shap_table = pd.read_csv(GLOBAL_SHAP_PATH)
    subgroup_drivers = pd.read_csv(SUBGROUP_DRIVERS_PATH) if _exists(SUBGROUP_DRIVERS_PATH) else pd.DataFrame()
    manifest = _load_json(ARTIFACT_MANIFEST_PATH) if _exists(ARTIFACT_MANIFEST_PATH) else {}
    analytic = _load_analytic_frame()

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-eyebrow">Caregiver Distress Risk Dashboard</div>
            <div class="hero-title">Stakeholder-facing screening and explainability for caregiver distress</div>
            <p class="hero-copy">
                This dashboard summarizes modeled distress risk across the Statistics Canada 2018 caregiving PUMF,
                with emphasis on honest risk ranking, interpretable patterns, and operational triage rather than diagnosis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("## Control Panel")
    st.sidebar.markdown(
        """
        <div class="sidebar-note">
            Narrow the cohort to inspect how risk, segments, and correlations change across provinces, care intensity,
            income, age, and received-help condition.
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_band = st.sidebar.multiselect(
        "Risk band",
        options=_risk_filter_options(scored, "risk_band"),
        default=_risk_filter_options(scored, "risk_band"),
    )
    selected_province = st.sidebar.multiselect(
        "Province",
        options=_risk_filter_options(scored, "PRV"),
        default=_risk_filter_options(scored, "PRV"),
    )
    selected_hours = st.sidebar.multiselect(
        "Care hours",
        options=_risk_filter_options(scored, "HAP_10C"),
        default=_risk_filter_options(scored, "HAP_10C"),
    )
    selected_income = st.sidebar.multiselect(
        "Personal income group",
        options=_risk_filter_options(scored, "TTLINCG1"),
        default=_risk_filter_options(scored, "TTLINCG1"),
    )
    selected_age = st.sidebar.multiselect(
        "Age group",
        options=_risk_filter_options(scored, "AGEGR10"),
        default=_risk_filter_options(scored, "AGEGR10"),
    )
    selected_condition = st.sidebar.multiselect(
        "Main health condition received help for",
        options=_risk_filter_options(scored, "PRA_10GR"),
        default=_risk_filter_options(scored, "PRA_10GR"),
    )
    min_probability = st.sidebar.slider("Minimum predicted risk", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    row_limit = st.sidebar.slider("Rows shown in tables", min_value=10, max_value=200, value=25, step=5)

    filtered = scored.copy()
    if selected_band:
        filtered = filtered.loc[filtered["risk_band"].isin(selected_band)]
    if selected_province and "PRV" in filtered.columns:
        filtered = filtered.loc[_display_series(filtered["PRV"], "PRV").isin(selected_province)]
    if selected_hours and "HAP_10C" in filtered.columns:
        filtered = filtered.loc[_display_series(filtered["HAP_10C"], "HAP_10C").isin(selected_hours)]
    if selected_income and "TTLINCG1" in filtered.columns:
        filtered = filtered.loc[_display_series(filtered["TTLINCG1"], "TTLINCG1").isin(selected_income)]
    if selected_age and "AGEGR10" in filtered.columns:
        filtered = filtered.loc[_display_series(filtered["AGEGR10"], "AGEGR10").isin(selected_age)]
    if selected_condition and "PRA_10GR" in filtered.columns:
        filtered = filtered.loc[_display_series(filtered["PRA_10GR"], "PRA_10GR").isin(selected_condition)]
    filtered = filtered.loc[filtered["predicted_probability"] >= float(min_probability)]

    correlation_ids = filtered[ID_COLUMN] if ID_COLUMN in filtered.columns else pd.Series(dtype=float)
    if ID_COLUMN in analytic.columns and not correlation_ids.empty:
        correlation_frame = analytic.loc[analytic[ID_COLUMN].isin(correlation_ids)].copy()
    else:
        correlation_frame = analytic.copy()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Analytic sample", f"{validation['target_summary']['analytic_rows']:,}")
    col2.metric("Test ROC AUC", f"{metrics['test_metrics']['roc_auc']:.3f}")
    col3.metric("Test PR AUC", f"{metrics['test_metrics']['pr_auc']:.3f}")
    col4.metric("Selected threshold", f"{metrics['selected_threshold']:.2f}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Filtered caregivers", f"{len(filtered):,}")
    col6.metric(
        "High-priority cases",
        f"{int(filtered['predicted_label'].sum()):,}",
    )
    col7.metric(
        "Average predicted risk",
        f"{filtered['predicted_probability'].mean():.3f}",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "This system estimates analyst-defined caregiver distress risk from survey responses. "
        "It is not a clinical diagnosis, and the CRH-item universe is approximated with grouped `HAP_10C` because raw `HAP_10` is not present in the PUMF."
    )

    overview_tab, explain_tab, ops_tab, data_tab = st.tabs(["Overview", "Explainability", "Operations", "Data"])

    with overview_tab:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Core visuals")
        figure_pairs = [
            ("Feature importance", manifest.get("feature_importance_fig")),
            ("SHAP summary", manifest.get("shap_summary_fig")),
            ("Risk distribution", manifest.get("risk_distribution_fig")),
            ("Subgroup comparison", manifest.get("subgroup_comparison_fig")),
        ]
        for title, figure_path in figure_pairs:
            if figure_path and Path(figure_path).exists():
                st.markdown(f"**{title}**")
                st.image(str(figure_path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Risk concentration heatmaps")
        heatmap_cols = st.columns(2)
        heatmap_specs = [
            ("Province × risk band", HEATMAP_PATHS["province_risk_band"]["png"]),
            ("Care hours × risk band", HEATMAP_PATHS["care_hours_risk_band"]["png"]),
            ("Income × risk band", HEATMAP_PATHS["income_risk_band"]["png"]),
            ("Main health condition × risk band", HEATMAP_PATHS["relationship_risk_band"]["png"]),
        ]
        for idx, (title, path) in enumerate(heatmap_specs):
            if Path(path).exists():
                with heatmap_cols[idx % 2]:
                    st.markdown(f"**{title}**")
                    st.image(str(path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with explain_tab:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Model calibration and thresholding")
        for title, figure_path in [
            ("Threshold tradeoff", manifest.get("threshold_fig")),
            ("Calibration", manifest.get("calibration_fig")),
        ]:
            if figure_path and Path(figure_path).exists():
                st.markdown(f"**{title}**")
                st.image(str(figure_path), use_container_width=True)
        st.markdown("**Top SHAP drivers overall**")
        st.dataframe(_format_display_table(shap_table.head(15)), use_container_width=True)
        if not subgroup_drivers.empty:
            st.markdown("**Subgroup-specific drivers**")
            st.dataframe(_format_display_table(subgroup_drivers), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Correlation matrix")
        st.caption(
            "Spearman correlations across quantitative, ordinal, and binary model features in the filtered analytic caregiver sample. "
            "Nominal coded fields like province are excluded to avoid misleading numeric correlations."
        )
        corr_columns = _correlation_columns(correlation_frame)
        if len(corr_columns) >= 2:
            corr_source = correlation_frame.loc[:, corr_columns].apply(pd.to_numeric, errors="coerce")
            corr_matrix = corr_source.corr(method="spearman").round(2)
            st.pyplot(_plot_correlation_heatmap(corr_matrix), use_container_width=True)
            with st.expander("View correlation table"):
                st.dataframe(corr_matrix.rename(index=_friendly_label, columns=_friendly_label), use_container_width=True)
        else:
            st.warning("Not enough filtered numeric/ordered features are available to compute a stable correlation matrix.")
        st.markdown("</div>", unsafe_allow_html=True)

    with ops_tab:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Resource allocation tables")
        left, right = st.columns(2)
        with left:
            st.markdown("**Highest-risk segments**")
            st.dataframe(_format_display_table(segments.head(row_limit)), use_container_width=True)
            st.markdown("**Subgroup risk summary**")
            st.dataframe(_format_display_table(subgroup.head(row_limit)), use_container_width=True)
        with right:
            filtered_individuals = _format_display_table(
                filtered.sort_values("predicted_probability", ascending=False).head(row_limit)
            )
            st.markdown("**Highest-risk individuals in current filter**")
            st.dataframe(filtered_individuals, use_container_width=True)
            st.markdown("**Export top-risk examples**")
            st.dataframe(_format_display_table(individuals.head(row_limit)), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with data_tab:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Filtered scored dataset")
        st.dataframe(_format_display_table(filtered.head(max(row_limit * 4, 40))), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
