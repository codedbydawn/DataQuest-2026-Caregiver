from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damb.config import (
    ARTIFACT_MANIFEST_PATH,
    GLOBAL_SHAP_PATH,
    HEATMAP_PATHS,
    HIGH_RISK_INDIVIDUALS_PATH,
    HIGH_RISK_SEGMENTS_PATH,
    METRICS_PATH,
    SCORED_DATA_PATH,
    SUBGROUP_DRIVERS_PATH,
    SUBGROUP_RISK_PATH,
    VALIDATION_PATH,
)


st.set_page_config(page_title="Caregiver Distress Analytics", layout="wide")


def _exists(path: Path) -> bool:
    return Path(path).exists()


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _risk_filter_options(scored: pd.DataFrame, column: str) -> list[str]:
    if column not in scored.columns:
        return []
    return sorted(scored[column].dropna().astype(str).unique().tolist())


def main() -> None:
    st.title("Caregiver Distress Analytics Dashboard")
    st.caption(
        "Organization-facing risk scoring and explainability for the Statistics Canada GSS 2018 caregiving PUMF."
    )

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

    st.sidebar.header("Filters")
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
    filtered = scored.copy()
    if selected_band:
        filtered = filtered.loc[filtered["risk_band"].isin(selected_band)]
    if selected_province and "PRV" in filtered.columns:
        filtered = filtered.loc[filtered["PRV"].astype(str).isin(selected_province)]
    if selected_hours and "HAP_10C" in filtered.columns:
        filtered = filtered.loc[filtered["HAP_10C"].astype(str).isin(selected_hours)]

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

    st.info(
        "This system estimates analyst-defined caregiver distress risk from survey responses. "
        "It is not a clinical diagnosis, and the CRH-item universe is approximated with grouped `HAP_10C` because raw `HAP_10` is not present in the PUMF."
    )

    st.subheader("Core visuals")
    figure_pairs = [
        ("Feature importance", manifest.get("feature_importance_fig")),
        ("SHAP summary", manifest.get("shap_summary_fig")),
        ("Risk distribution", manifest.get("risk_distribution_fig")),
        ("Threshold tradeoff", manifest.get("threshold_fig")),
        ("Calibration", manifest.get("calibration_fig")),
        ("Subgroup comparison", manifest.get("subgroup_comparison_fig")),
    ]
    for title, figure_path in figure_pairs:
        if figure_path and Path(figure_path).exists():
            st.markdown(f"**{title}**")
            st.image(str(figure_path), use_container_width=True)

    st.subheader("Risk concentration heatmaps")
    heatmap_cols = st.columns(2)
    heatmap_specs = [
        ("Province × risk band", HEATMAP_PATHS["province_risk_band"]["png"]),
        ("Care hours × risk band", HEATMAP_PATHS["care_hours_risk_band"]["png"]),
        ("Income × risk band", HEATMAP_PATHS["income_risk_band"]["png"]),
        ("Relationship × risk band", HEATMAP_PATHS["relationship_risk_band"]["png"]),
    ]
    for idx, (title, path) in enumerate(heatmap_specs):
        if Path(path).exists():
            with heatmap_cols[idx % 2]:
                st.markdown(f"**{title}**")
                st.image(str(path), use_container_width=True)

    st.subheader("Resource allocation tables")
    left, right = st.columns(2)
    with left:
        st.markdown("**Highest-risk segments**")
        st.dataframe(segments.head(25), use_container_width=True)
        st.markdown("**Top SHAP drivers overall**")
        st.dataframe(shap_table.head(15), use_container_width=True)
    with right:
        st.markdown("**Highest-risk individuals**")
        st.dataframe(individuals.head(25), use_container_width=True)
        st.markdown("**Subgroup risk summary**")
        st.dataframe(subgroup.head(25), use_container_width=True)

    if not subgroup_drivers.empty:
        st.subheader("Subgroup-specific drivers")
        st.dataframe(subgroup_drivers, use_container_width=True)

    st.subheader("Filtered scored dataset")
    st.dataframe(filtered.head(200), use_container_width=True)


if __name__ == "__main__":
    main()
