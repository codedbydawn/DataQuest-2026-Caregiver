from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    AGE_GROUP_LABELS,
    ARTIFACT_MANIFEST_PATH,
    CALIBRATION_FIG_PATH,
    CALIBRATION_REPORT_PATH,
    CALIBRATION_TABLE_PATH,
    FEATURE_IMPORTANCE_FIG_PATH,
    FEATURE_AUDIT_PATH,
    FEATURE_MISSINGNESS_PATH,
    FOLD_METRICS_PATH,
    GLOBAL_SHAP_PATH,
    HEALTH_CONDITION_LABELS,
    HEATMAP_PATHS,
    HIGH_RISK_INDIVIDUALS_PATH,
    HIGH_RISK_SEGMENTS_PATH,
    HOUR_GROUP_LABELS,
    ID_COLUMN,
    INCOME_GROUP_LABELS,
    METRICS_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_CARD_PATH,
    MODEL_FEATURES,
    MODEL_REPORT_PATH,
    MISSINGNESS_REPORT_PATH,
    PAIRWISE_SEGMENTS,
    PROCESSED_DIR,
    PROVINCE_LABELS,
    REPORTS_DIR,
    RISK_BAND_BINS,
    RISK_BAND_LABELS,
    RISK_DISTRIBUTION_FIG_PATH,
    SCORED_DATA_PATH,
    SEX_LABELS,
    SHAP_SUMMARY_FIG_PATH,
    SUBGROUP_COLUMNS,
    SUBGROUP_DIAGNOSTICS_PATH,
    SUBGROUP_COMPARISON_FIG_PATH,
    SUBGROUP_DRIVERS_PATH,
    SUBGROUP_RISK_PATH,
    TABLES_DIR,
    TEST_PREDICTIONS_PATH,
    THRESHOLD_ANALYSIS_PATH,
    THRESHOLD_CURVE_PATH,
    THRESHOLD_FIG_PATH,
    VALIDATION_PATH,
    WEIGHT_COLUMN,
)
from .data import PreparedDataset
from .modeling import TrainingResult, leakage_audit
from .scoring import compute_aggregated_contributions, score_frame


matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def _default_output_paths() -> dict[str, Any]:
    return {
        "model_artifact": MODEL_ARTIFACT_PATH,
        "metrics": METRICS_PATH,
        "validation": VALIDATION_PATH,
        "scored_data": SCORED_DATA_PATH,
        "test_predictions": TEST_PREDICTIONS_PATH,
        "feature_missingness": FEATURE_MISSINGNESS_PATH,
        "feature_audit": FEATURE_AUDIT_PATH,
        "missingness_report": MISSINGNESS_REPORT_PATH,
        "fold_metrics": FOLD_METRICS_PATH,
        "global_shap": GLOBAL_SHAP_PATH,
        "subgroup_risk": SUBGROUP_RISK_PATH,
        "subgroup_diagnostics": SUBGROUP_DIAGNOSTICS_PATH,
        "high_risk_segments": HIGH_RISK_SEGMENTS_PATH,
        "high_risk_individuals": HIGH_RISK_INDIVIDUALS_PATH,
        "subgroup_drivers": SUBGROUP_DRIVERS_PATH,
        "threshold_curve": THRESHOLD_CURVE_PATH,
        "threshold_analysis": THRESHOLD_ANALYSIS_PATH,
        "calibration_table": CALIBRATION_TABLE_PATH,
        "calibration_report": CALIBRATION_REPORT_PATH,
        "artifact_manifest": ARTIFACT_MANIFEST_PATH,
        "model_report": MODEL_REPORT_PATH,
        "model_card": MODEL_CARD_PATH,
        "feature_importance_fig": FEATURE_IMPORTANCE_FIG_PATH,
        "shap_summary_fig": SHAP_SUMMARY_FIG_PATH,
        "risk_distribution_fig": RISK_DISTRIBUTION_FIG_PATH,
        "subgroup_comparison_fig": SUBGROUP_COMPARISON_FIG_PATH,
        "calibration_fig": CALIBRATION_FIG_PATH,
        "threshold_fig": THRESHOLD_FIG_PATH,
        "heatmaps": HEATMAP_PATHS,
    }


def _merge_paths(overrides: dict[str, Any] | None) -> dict[str, Any]:
    base = _default_output_paths()
    if not overrides:
        return base
    merged = base.copy()
    for key, value in overrides.items():
        if key == "heatmaps":
            merged["heatmaps"] = {**base["heatmaps"], **value}
        else:
            merged[key] = value
    return merged


def _ensure_parent_dirs(paths: dict[str, Any]) -> None:
    def ensure(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                ensure(nested)
        else:
            Path(value).parent.mkdir(parents=True, exist_ok=True)

    for path_value in paths.values():
        ensure(path_value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _map_code(value: Any, mapping: dict[int, str]) -> str:
    if pd.isna(value):
        return "Missing"
    try:
        return mapping.get(int(float(value)), str(value))
    except (TypeError, ValueError):
        return str(value)


def _display_series(series: pd.Series, column: str) -> pd.Series:
    mapping = {
        "PRV": PROVINCE_LABELS,
        "AGEGR10": AGE_GROUP_LABELS,
        "SEX": SEX_LABELS,
        "HAP_10C": HOUR_GROUP_LABELS,
        "TTLINCG1": INCOME_GROUP_LABELS,
        "PRA_10GR": HEALTH_CONDITION_LABELS,
    }.get(column)
    if mapping is None:
        return series.fillna("Missing").astype(str)
    return series.map(lambda value: _map_code(value, mapping))


def _weighted_group_summary(scored: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = scored.copy()
    summary["group_value"] = _display_series(summary[column], column)
    grouped = summary.groupby("group_value", observed=False)
    rows = []
    for group_value, group in grouped:
        weights = group[WEIGHT_COLUMN].astype(float)
        rows.append(
            {
                "dimension": column,
                "group_value": group_value,
                "unweighted_count": int(len(group)),
                "weighted_count": float(weights.sum()),
                "avg_predicted_risk": float(np.average(group["predicted_probability"], weights=weights)),
                "high_priority_weighted_count": float(weights[group["predicted_label"] == 1].sum()),
                "high_priority_rate": float(np.average(group["predicted_label"], weights=weights)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dimension", "avg_predicted_risk", "weighted_count"],
        ascending=[True, False, False],
        ignore_index=True,
    )


def _build_subgroup_risk_summary(scored: pd.DataFrame) -> pd.DataFrame:
    parts = [_weighted_group_summary(scored, column) for column in SUBGROUP_COLUMNS if column in scored.columns]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _build_heatmap_table(scored: pd.DataFrame, group_col: str) -> pd.DataFrame:
    frame = scored.copy()
    frame[group_col] = _display_series(frame[group_col], group_col)
    pivot = pd.pivot_table(
        frame,
        values=WEIGHT_COLUMN,
        index=group_col,
        columns="risk_band",
        aggfunc="sum",
        fill_value=0.0,
        observed=False,
    )
    for label in RISK_BAND_LABELS:
        if label not in pivot.columns:
            pivot[label] = 0.0
    return pivot.loc[:, list(RISK_BAND_LABELS)].sort_index()


def _build_high_risk_segments(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left, right in PAIRWISE_SEGMENTS:
        if left not in scored.columns or right not in scored.columns:
            continue
        frame = scored.copy()
        frame[left] = _display_series(frame[left], left)
        frame[right] = _display_series(frame[right], right)
        grouped = frame.groupby([left, right], observed=False)
        for (left_value, right_value), group in grouped:
            weights = group[WEIGHT_COLUMN].astype(float)
            rows.append(
                {
                    "segment_type": f"{left} x {right}",
                    "segment_value_1": left_value,
                    "segment_value_2": right_value,
                    "unweighted_count": int(len(group)),
                    "weighted_count": float(weights.sum()),
                    "avg_predicted_risk": float(np.average(group["predicted_probability"], weights=weights)),
                    "high_priority_weighted_count": float(weights[group["predicted_label"] == 1].sum()),
                    "high_priority_rate": float(np.average(group["predicted_label"], weights=weights)),
                }
            )
    segments = pd.DataFrame(rows)
    if segments.empty:
        return segments
    return (
        segments.loc[segments["weighted_count"] >= 25]
        .sort_values(["avg_predicted_risk", "weighted_count"], ascending=[False, False], ignore_index=True)
        .head(100)
    )


def _build_high_risk_individuals(scored: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in [ID_COLUMN, WEIGHT_COLUMN, "PRV", "HAP_10C", "TTLINCG1", "PRA_10GR"] if column in scored.columns]
    ordered = (
        scored.loc[:, columns + ["predicted_probability", "predicted_label", "risk_band"]]
        .sort_values("predicted_probability", ascending=False, ignore_index=True)
        .head(100)
    )
    for column in ["PRV", "HAP_10C", "TTLINCG1"]:
        if column in ordered.columns:
            ordered[column] = _display_series(ordered[column], column)
    if "PRA_10GR" in ordered.columns:
        ordered["PRA_10GR"] = _display_series(ordered["PRA_10GR"], "PRA_10GR")
    return ordered


def _build_subgroup_top_drivers(scored: pd.DataFrame, shap_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in ["PRV", "HAP_10C", "SEX"]:
        if dimension not in scored.columns:
            continue
        dimension_values = _display_series(scored[dimension], dimension)
        weighted_counts = (
            pd.DataFrame({"group_value": dimension_values, "weight": scored[WEIGHT_COLUMN]})
            .groupby("group_value", observed=False)["weight"]
            .sum()
            .sort_values(ascending=False)
            .head(3)
        )
        for group_value in weighted_counts.index:
            mask = dimension_values == group_value
            if int(mask.sum()) == 0:
                continue
            feature_scores = shap_frame.loc[mask].abs().mean().sort_values(ascending=False).head(5)
            for feature, value in feature_scores.items():
                rows.append(
                    {
                        "dimension": dimension,
                        "group_value": group_value,
                        "feature": feature,
                        "mean_abs_shap": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _plot_feature_importance(global_shap: pd.DataFrame, output_path: Path) -> None:
    top = global_shap.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#1d6f8b")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    ax.set_title("Global feature importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_shap_summary(shap_frame: pd.DataFrame, output_path: Path) -> None:
    top_features = shap_frame.abs().mean().sort_values(ascending=False).head(12).index.tolist()
    sample = shap_frame.loc[:, top_features]
    if len(sample) > 2500:
        sample = sample.sample(2500, random_state=42)
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(11, 7))
    for idx, feature in enumerate(reversed(top_features)):
        values = sample[feature].to_numpy()
        jitter = rng.normal(loc=idx, scale=0.08, size=len(values))
        ax.scatter(values, jitter, s=10, alpha=0.22, color="#10454f")
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(list(reversed(top_features)))
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("")
    ax.set_title("SHAP summary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_risk_distribution(scored: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scored["predicted_probability"], bins=20, color="#297373", edgecolor="white")
    ax.set_xlabel("Predicted distress risk")
    ax.set_ylabel("Caregivers")
    ax.set_title("Distribution of predicted distress risk")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_subgroup_comparison(subgroup_summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specs = [("PRV", 8), ("HAP_10C", 6), ("TTLINCG1", 7), ("SEX", 2)]
    for ax, (dimension, limit) in zip(axes.flat, specs):
        subset = subgroup_summary.loc[subgroup_summary["dimension"] == dimension].head(limit)
        sns.barplot(data=subset, x="avg_predicted_risk", y="group_value", ax=ax, color="#5c9ead")
        ax.set_xlabel("Average predicted risk")
        ax.set_ylabel("")
        ax.set_title(dimension)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_heatmap(table: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, len(table) * 0.4)))
    sns.heatmap(table, cmap="YlOrRd", linewidths=0.5, fmt=".0f", annot=True, cbar=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Risk band")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_calibration(calibration_table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.plot(
        calibration_table["mean_predicted_probability"],
        calibration_table["observed_rate"],
        marker="o",
        color="#28536b",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed rate")
    ax.set_title("Calibration on held-out test set")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_threshold_curve(threshold_curve: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for column, color in [("precision", "#4d9078"), ("recall", "#f4a259"), ("f1", "#bc4b51")]:
        ax.plot(threshold_curve["threshold"], threshold_curve[column], label=column, color=color)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Validation threshold tradeoff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_model_report(
    prepared: PreparedDataset,
    training_result: TrainingResult,
    manifest: dict[str, Any],
    output_path: Path,
) -> None:
    test_metrics = training_result.metrics["test_metrics"]
    validation_metrics = training_result.metrics["validation_metrics"]
    threshold_modes = training_result.metrics.get("threshold_modes", {"balanced": training_result.metrics["selected_threshold"]})
    missing_strategy = training_result.metrics.get("selected_missing_strategy", "unknown")
    feature_count = training_result.metrics.get("selected_feature_count", len(training_result.artifact.get("feature_columns", [])))
    content = f"""# Caregiver Distress Model Report

## Final system
- Stage 1: weighted XGBoost model predicting the analyst-defined `distress_flag`
- Stage 2: dashboard-ready scored dataset, explainability outputs, subgroup summaries, and resource-allocation artifacts

## Verified assumptions used
- Raw file: `src/c32pumfm.sas7bdat`
- Target: `CRH_20`, `CRH_30`, `CRH_60`
- Universe approximation: `DV_PROXY == 2`, `PAR_10 in 1..99`, `HAP_10C in 1..6`
- Valid skip values are treated as off-path / structural missingness rather than substantive "No"

## Analytic sample
- Raw rows: {prepared.universe_counts["raw_rows"]}
- Final analytic rows: {prepared.universe_counts["after_target_nonmissing"]}
- Weighted prevalence: {prepared.target_summary["weighted_prevalence"]:.4f}
- Unweighted prevalence: {prepared.target_summary["unweighted_prevalence"]:.4f}

## Final performance
- Validation ROC AUC: {validation_metrics["roc_auc"]:.4f}
- Validation PR AUC: {validation_metrics["pr_auc"]:.4f}
- Test ROC AUC: {test_metrics["roc_auc"]:.4f}
- Test PR AUC: {test_metrics["pr_auc"]:.4f}
- Test weighted accuracy: {test_metrics["weighted_accuracy"]:.4f}
- Test weighted Brier score: {test_metrics["weighted_brier"]:.4f}
- Selected threshold (balanced mode): {training_result.metrics["selected_threshold"]:.3f}
- Threshold modes: {threshold_modes}
- Selected missing-data strategy: {missing_strategy}
- Selected feature count: {feature_count}

## Outputs
- Artifact manifest: `{manifest["artifact_manifest"]}`
- Scored dataset: `{manifest["scored_data"]}`
- SHAP importance table: `{manifest["global_shap"]}`
- Highest-risk segments: `{manifest["high_risk_segments"]}`
- Calibration report: `{manifest["calibration_report"]}`
- Feature audit: `{manifest["feature_audit"]}`
- Fold metrics: `{manifest["fold_metrics"]}`
- Subgroup diagnostics: `{manifest["subgroup_diagnostics"]}`
- Dashboard app: `streamlit run app/app.py`

## Limitations
- The PUMF exposes grouped `HAP_10C` rather than raw `HAP_10`, so the official target universe is approximated
- Calibration and threshold choice are optimized for stakeholder-facing ranking and triage, not for causal interpretation
- Several work and receiver-detail variables carry structural missingness because of survey routing
- This output estimates analyst-defined distress risk, not a clinical diagnosis
"""
    output_path.write_text(content)


def _write_model_card(prepared: PreparedDataset, training_result: TrainingResult, output_path: Path) -> None:
    test_metrics = training_result.metrics["test_metrics"]
    missing_strategy = training_result.metrics.get("selected_missing_strategy", "unknown")
    precision = test_metrics.get("precision", float("nan"))
    recall = test_metrics.get("recall", float("nan"))
    specificity = test_metrics.get("specificity", float("nan"))
    card = f"""# Model Card

## Intended use
- Stakeholder-facing caregiver distress risk ranking and subgroup concentration review.
- Appropriate for organizational triage, planning, and dashboarding.
- Not appropriate for clinical diagnosis, causal claims, or automated adverse decisions.

## Model summary
- Model family: XGBoost binary classifier with post-hoc sigmoid calibration.
- Operating modes: balanced, high_precision, high_recall.
- Default dashboard threshold: {training_result.metrics["selected_threshold"]:.3f} (balanced mode).
- Selected missing-data strategy: {missing_strategy}.

## Data summary
- Raw rows: {prepared.universe_counts["raw_rows"]}
- Analytic rows: {prepared.universe_counts["after_target_nonmissing"]}
- Weighted prevalence: {prepared.target_summary["weighted_prevalence"]:.4f}

## Test performance
- ROC AUC: {test_metrics["roc_auc"]:.4f}
- PR AUC: {test_metrics["pr_auc"]:.4f}
- Weighted accuracy: {test_metrics["weighted_accuracy"]:.4f}
- Precision / Recall / Specificity: {precision:.4f} / {recall:.4f} / {specificity:.4f}
- Weighted Brier score: {test_metrics["weighted_brier"]:.4f}
- Log loss: {test_metrics.get("log_loss", float("nan")):.4f}

## Key limitations
- The official CRH universe uses raw HAP_10, but the PUMF only exposes grouped HAP_10C.
- Valid skips are structural and can induce heavy routed missingness.
- The dataset is an all-respondent survey with routed caregiving and care-receiving sections.
- Model explanations reflect model behavior, not causality.
"""
    output_path.write_text(card)


def save_training_outputs(
    prepared: PreparedDataset,
    training_result: TrainingResult,
    output_paths: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = _merge_paths(output_paths)
    _ensure_parent_dirs(paths)

    joblib.dump(training_result.artifact, paths["model_artifact"])
    prepared.missingness.to_csv(paths["feature_missingness"], index=False)
    training_result.feature_audit.to_csv(paths["feature_audit"], index=False)
    training_result.missingness_report.to_csv(paths["missingness_report"], index=False)
    training_result.fold_metrics.to_csv(paths["fold_metrics"], index=False)
    training_result.threshold_curve.to_csv(paths["threshold_curve"], index=False)
    training_result.threshold_analysis.to_csv(paths["threshold_analysis"], index=False)
    training_result.calibration_table.to_csv(paths["calibration_table"], index=False)
    training_result.test_predictions.to_csv(paths["test_predictions"], index=False)
    training_result.subgroup_diagnostics.to_csv(paths["subgroup_diagnostics"], index=False)

    score_columns = [column for column in dict.fromkeys([ID_COLUMN, WEIGHT_COLUMN, *MODEL_FEATURES, *SUBGROUP_COLUMNS]) if column in prepared.frame.columns]
    passthrough_columns = [column for column in [ID_COLUMN, WEIGHT_COLUMN, *SUBGROUP_COLUMNS] if column in prepared.frame.columns]
    scored = score_frame(
        artifact=training_result.artifact,
        frame=prepared.frame.loc[:, score_columns].copy(),
        passthrough_columns=passthrough_columns,
    )
    scored.to_csv(paths["scored_data"], index=False)

    shap_frame = compute_aggregated_contributions(training_result.artifact, prepared.frame.loc[:, MODEL_FEATURES])
    global_shap = (
        shap_frame.abs()
        .mean()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    global_shap.to_csv(paths["global_shap"], index=False)

    subgroup_summary = _build_subgroup_risk_summary(scored)
    subgroup_summary.to_csv(paths["subgroup_risk"], index=False)
    high_risk_segments = _build_high_risk_segments(scored)
    high_risk_segments.to_csv(paths["high_risk_segments"], index=False)
    high_risk_individuals = _build_high_risk_individuals(scored)
    high_risk_individuals.to_csv(paths["high_risk_individuals"], index=False)
    subgroup_drivers = _build_subgroup_top_drivers(scored, shap_frame)
    subgroup_drivers.to_csv(paths["subgroup_drivers"], index=False)

    province_heatmap = _build_heatmap_table(scored, "PRV")
    hours_heatmap = _build_heatmap_table(scored, "HAP_10C")
    income_heatmap = _build_heatmap_table(scored, "TTLINCG1")
    relationship_heatmap = _build_heatmap_table(scored, "PRA_10GR")
    province_heatmap.to_csv(paths["heatmaps"]["province_risk_band"]["csv"])
    hours_heatmap.to_csv(paths["heatmaps"]["care_hours_risk_band"]["csv"])
    income_heatmap.to_csv(paths["heatmaps"]["income_risk_band"]["csv"])
    relationship_heatmap.to_csv(paths["heatmaps"]["relationship_risk_band"]["csv"])

    _plot_feature_importance(global_shap, Path(paths["feature_importance_fig"]))
    _plot_shap_summary(shap_frame, Path(paths["shap_summary_fig"]))
    _plot_risk_distribution(scored, Path(paths["risk_distribution_fig"]))
    _plot_subgroup_comparison(subgroup_summary, Path(paths["subgroup_comparison_fig"]))
    _plot_heatmap(province_heatmap, "Province by risk band", Path(paths["heatmaps"]["province_risk_band"]["png"]))
    _plot_heatmap(hours_heatmap, "Care hours by risk band", Path(paths["heatmaps"]["care_hours_risk_band"]["png"]))
    _plot_heatmap(income_heatmap, "Income by risk band", Path(paths["heatmaps"]["income_risk_band"]["png"]))
    _plot_heatmap(
        relationship_heatmap,
        "Main health condition by risk band",
        Path(paths["heatmaps"]["relationship_risk_band"]["png"]),
    )
    _plot_calibration(training_result.calibration_table, Path(paths["calibration_fig"]))
    _plot_threshold_curve(training_result.threshold_curve, Path(paths["threshold_fig"]))

    importance_share = float(global_shap.iloc[0]["mean_abs_shap"] / global_shap["mean_abs_shap"].sum())
    importance_ratio = float(
        global_shap.iloc[0]["mean_abs_shap"] / global_shap.iloc[1]["mean_abs_shap"]
    ) if len(global_shap) > 1 else np.nan

    validation_summary = {
        "raw_shape": {"rows": prepared.raw_shape[0], "columns": prepared.raw_shape[1]},
        "universe_counts": prepared.universe_counts,
        "target_summary": prepared.target_summary,
        "weight_checks": {
            "non_null": bool(prepared.frame[WEIGHT_COLUMN].notna().all()),
            "strictly_positive": bool((prepared.frame[WEIGHT_COLUMN] > 0).all()),
        },
        "feature_missingness_top10": prepared.missingness.head(10).to_dict(orient="records"),
        "feature_strategy_comparison_top": training_result.tuning_results.head(10).to_dict(orient="records"),
        "final_feature_list": list(training_result.artifact["feature_columns"]),
        "leakage_audit": leakage_audit(training_result.artifact["feature_columns"]),
        "hap_10c_approximation_note": (
            "The official CRH universe uses raw HAP_10, but this PUMF exposes only grouped HAP_10C. "
            "The production system therefore uses HAP_10C in {1..6} and keeps HAP_10C == 1 rows when valid CRH responses exist."
        ),
        "importance_sanity_check": {
            "top_feature": str(global_shap.iloc[0]["feature"]),
            "top_feature_share_of_total_mean_abs_shap": importance_share,
            "top_to_second_feature_ratio": importance_ratio,
            "flagged_as_suspicious": bool(importance_share > 0.22 or importance_ratio > 2.0),
        },
        "metrics": training_result.metrics,
    }

    with open(paths["metrics"], "w", encoding="utf-8") as handle:
        json.dump(_json_ready(training_result.metrics), handle, indent=2)
    with open(paths["validation"], "w", encoding="utf-8") as handle:
        json.dump(_json_ready(validation_summary), handle, indent=2)
    with open(paths["calibration_report"], "w", encoding="utf-8") as handle:
        json.dump(_json_ready(training_result.calibration_report), handle, indent=2)

    manifest = {
        "model_artifact": str(paths["model_artifact"]),
        "metrics": str(paths["metrics"]),
        "validation": str(paths["validation"]),
        "scored_data": str(paths["scored_data"]),
        "test_predictions": str(paths["test_predictions"]),
        "feature_missingness": str(paths["feature_missingness"]),
        "feature_audit": str(paths["feature_audit"]),
        "missingness_report": str(paths["missingness_report"]),
        "fold_metrics": str(paths["fold_metrics"]),
        "global_shap": str(paths["global_shap"]),
        "subgroup_risk": str(paths["subgroup_risk"]),
        "subgroup_diagnostics": str(paths["subgroup_diagnostics"]),
        "high_risk_segments": str(paths["high_risk_segments"]),
        "high_risk_individuals": str(paths["high_risk_individuals"]),
        "subgroup_drivers": str(paths["subgroup_drivers"]),
        "threshold_curve": str(paths["threshold_curve"]),
        "threshold_analysis": str(paths["threshold_analysis"]),
        "calibration_table": str(paths["calibration_table"]),
        "calibration_report": str(paths["calibration_report"]),
        "feature_importance_fig": str(paths["feature_importance_fig"]),
        "shap_summary_fig": str(paths["shap_summary_fig"]),
        "risk_distribution_fig": str(paths["risk_distribution_fig"]),
        "subgroup_comparison_fig": str(paths["subgroup_comparison_fig"]),
        "calibration_fig": str(paths["calibration_fig"]),
        "threshold_fig": str(paths["threshold_fig"]),
        "province_heatmap": str(paths["heatmaps"]["province_risk_band"]["png"]),
        "care_hours_heatmap": str(paths["heatmaps"]["care_hours_risk_band"]["png"]),
        "income_heatmap": str(paths["heatmaps"]["income_risk_band"]["png"]),
        "relationship_heatmap": str(paths["heatmaps"]["relationship_risk_band"]["png"]),
        "artifact_manifest": str(paths["artifact_manifest"]),
        "model_report": str(paths["model_report"]),
        "model_card": str(paths["model_card"]),
    }
    with open(paths["artifact_manifest"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    _write_model_report(prepared, training_result, manifest, Path(paths["model_report"]))
    _write_model_card(prepared, training_result, Path(paths["model_card"]))
    return {"manifest": manifest, "validation_summary": validation_summary, "global_shap": global_shap}
