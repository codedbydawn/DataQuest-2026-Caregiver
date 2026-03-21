from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from damb.analytics import save_training_outputs
from damb.config import MODEL_FEATURES, WEIGHT_COLUMN
from damb.data import (
    PreparedDataset,
    apply_modeling_universe,
    build_target,
    coerce_feature_types,
    recode_binary_yes_no,
    recode_special_codes,
)
from damb.modeling import TrainingResult, build_preprocessing_pipeline, leakage_audit
from damb.scoring import CaregiverDistressScorer, score_frame


def make_feature_frame(rows: int = 20) -> pd.DataFrame:
    base = pd.DataFrame(index=range(rows))
    for idx, column in enumerate(MODEL_FEATURES):
        if column in {"PAR_10", "NWE_110"}:
            base[column] = np.linspace(1, rows, rows)
        elif column in {"AGEGR10", "HAP_10C", "UHW_16GR", "TTLINCG1", "FAMINCG1"}:
            base[column] = (np.arange(rows) % 4) + 1
        elif column in {"SEX", "UCA_10", "FWA_134", "FWA_137", "APR_10", "APR_20", "APR_30", "APR_40", "APR_50", "APR_60", "APR_70", "APR_80", "ARV_10", "ARX_10", "CHC_100"}:
            base[column] = np.where(np.arange(rows) % 2 == 0, 1.0, 2.0)
        elif column == "PRV":
            base[column] = np.where(np.arange(rows) % 2 == 0, 35.0, 24.0)
        else:
            base[column] = (np.arange(rows) % 3) + 1
    base["PUMFID"] = np.arange(1000, 1000 + rows)
    base["PRA_10GR"] = np.where(np.arange(rows) % 4 == 0, 1.0, np.nan)
    base[WEIGHT_COLUMN] = np.linspace(1.0, 2.0, rows)
    base["distress_flag"] = np.where(np.arange(rows) % 3 == 0, 1.0, 0.0)
    base["distress_score"] = base["distress_flag"]
    return coerce_feature_types(base)


def make_trained_artifact(frame: pd.DataFrame) -> dict:
    x = frame.loc[:, MODEL_FEATURES]
    y = frame["distress_flag"].astype(int)
    preprocessor = build_preprocessing_pipeline()
    x_tx = preprocessor.fit_transform(x)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=20,
        max_depth=2,
        learning_rate=0.2,
        tree_method="hist",
        random_state=42,
        n_jobs=1,
    )
    model.fit(x_tx, y)
    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": list(MODEL_FEATURES),
        "threshold": 0.5,
    }


def test_recode_special_codes_is_explicit() -> None:
    frame = pd.DataFrame(
        {
            "HAP_10C": [1.0, 96.0, 99.0],
            "CRH_20": [1.0, 6.0, 9.0],
            "PAR_10": [1.0, 996.0, 999.0],
        }
    )
    recoded = recode_special_codes(frame)
    assert recoded.loc[0, "HAP_10C"] == 1.0
    assert pd.isna(recoded.loc[1, "HAP_10C"])
    assert pd.isna(recoded.loc[2, "HAP_10C"])
    assert pd.isna(recoded.loc[1, "CRH_20"])
    assert pd.isna(recoded.loc[2, "CRH_20"])
    assert pd.isna(recoded.loc[1, "PAR_10"])


def test_build_target_uses_only_three_crh_items() -> None:
    frame = pd.DataFrame(
        {
            "CRH_20": [1.0, 2.0, np.nan, np.nan],
            "CRH_30": [2.0, 2.0, np.nan, 1.0],
            "CRH_60": [2.0, 1.0, np.nan, np.nan],
        }
    )
    built = build_target(frame)
    assert built["distress_score"].tolist()[:2] == [1.0, 1.0]
    assert pd.isna(built.loc[2, "distress_score"])
    assert built.loc[3, "distress_flag"] == 1.0
    assert pd.isna(built.loc[2, "distress_flag"])


def test_apply_modeling_universe_preserves_hap_10c_group_one_when_targets_exist() -> None:
    frame = pd.DataFrame(
        {
            "DV_PROXY": [2.0, 2.0, 1.0],
            "PAR_10": [1.0, 1.0, 1.0],
            "HAP_10C": [1.0, 2.0, 1.0],
            "distress_flag": [1.0, 0.0, 1.0],
        }
    )
    filtered, counts = apply_modeling_universe(frame)
    assert len(filtered) == 2
    assert counts["hap_10c_eq_1_kept"] == 1


def test_binary_feature_recode_is_consistent() -> None:
    frame = pd.DataFrame({"ARX_10": [1.0, 2.0, np.nan]})
    recoded = recode_binary_yes_no(frame, ["ARX_10"])
    assert recoded["ARX_10"].tolist()[:2] == [1.0, 0.0]
    assert pd.isna(recoded.loc[2, "ARX_10"])


def test_leakage_audit_flags_prefixes() -> None:
    audit = leakage_audit(["AGEGR10", "CRH_20", "ICS_40"])
    assert "CRH_20" in audit["exact_overlap"]
    assert "CRH_20" in audit["prefix_overlap"]
    assert "ICS_40" in audit["prefix_overlap"]


def test_preprocessing_is_deterministic() -> None:
    frame = make_feature_frame()
    preprocessor_one = build_preprocessing_pipeline()
    preprocessor_two = build_preprocessing_pipeline()
    preprocessor_one.fit(frame.loc[:, MODEL_FEATURES])
    preprocessor_two.fit(frame.loc[:, MODEL_FEATURES])
    assert preprocessor_one.get_feature_names_out().tolist() == preprocessor_two.get_feature_names_out().tolist()


def test_single_row_and_batch_scoring_work() -> None:
    frame = make_feature_frame()
    artifact = make_trained_artifact(frame)
    scorer = CaregiverDistressScorer(artifact=artifact)
    row_result = scorer.score_row(frame.iloc[0])
    batch_result = scorer.score_batch(frame.iloc[:5])
    assert 0.0 <= row_result["probability"] <= 1.0
    assert row_result["label"] in {0, 1}
    assert len(row_result["top_contributors"]) == 5
    assert len(batch_result) == 5
    assert {"predicted_probability", "predicted_label", "risk_band"} <= set(batch_result.columns)


def test_missing_feature_columns_are_backfilled_for_scoring() -> None:
    frame = make_feature_frame()
    artifact = make_trained_artifact(frame)
    minimal = pd.DataFrame([{"AGEGR10": 3.0, "PAR_10": 2.0, "HAP_10C": 1.0}])
    scored = score_frame(artifact, minimal)
    assert len(scored) == 1
    assert 0.0 <= scored.loc[0, "predicted_probability"] <= 1.0


def test_dashboard_artifact_generation_smoke(tmp_path: Path) -> None:
    frame = make_feature_frame(rows=24)
    artifact = make_trained_artifact(frame)
    prepared = PreparedDataset(
        frame=frame,
        universe_counts={
            "raw_rows": 30,
            "after_dv_proxy_eq_2": 28,
            "after_par_10_1_99": 26,
            "after_hap_10c_1_6": 24,
            "after_target_nonmissing": 24,
            "hap_10c_eq_1_kept": 6,
        },
        target_summary={
            "analytic_rows": 24,
            "unweighted_prevalence": 0.33,
            "weighted_prevalence": 0.35,
            "distress_score_distribution": {"0": 16, "1": 8},
            "target_items": {},
        },
        missingness=pd.DataFrame(
            {
                "feature": list(MODEL_FEATURES),
                "missing_pct": np.zeros(len(MODEL_FEATURES)),
                "missing_count": np.zeros(len(MODEL_FEATURES), dtype=int),
                "non_missing_count": np.full(len(MODEL_FEATURES), 24),
            }
        ),
        raw_shape=(30, 40),
    )
    training_result = TrainingResult(
        artifact=artifact,
        metrics={
            "selected_threshold": 0.5,
            "test_metrics": {"roc_auc": 0.7, "pr_auc": 0.6, "weighted_accuracy": 0.65, "weighted_brier": 0.21},
            "validation_metrics": {"roc_auc": 0.72, "pr_auc": 0.61},
            "train_metrics": {"roc_auc": 0.8, "pr_auc": 0.74},
        },
        tuning_results=pd.DataFrame([{"candidate_id": 0, "selection_score": 0.65}]),
        threshold_curve=pd.DataFrame({"threshold": [0.4, 0.5], "precision": [0.5, 0.6], "recall": [0.7, 0.6], "f1": [0.58, 0.60], "weighted_accuracy": [0.6, 0.65]}),
        calibration_table=pd.DataFrame({"bin": ["a"], "weighted_count": [10.0], "mean_predicted_probability": [0.5], "observed_rate": [0.4]}),
        test_predictions=pd.DataFrame({"predicted_probability": [0.2, 0.8], "predicted_label": [0, 1], "actual_distress_flag": [0, 1], "weight": [1.0, 1.0]}),
        validation_predictions=pd.DataFrame({"predicted_probability": [0.3], "actual_distress_flag": [0], "weight": [1.0]}),
    )
    outputs = save_training_outputs(
        prepared=prepared,
        training_result=training_result,
        output_paths={
            "model_artifact": tmp_path / "model.joblib",
            "metrics": tmp_path / "metrics.json",
            "validation": tmp_path / "validation.json",
            "scored_data": tmp_path / "scored.csv",
            "test_predictions": tmp_path / "test_predictions.csv",
            "feature_missingness": tmp_path / "missingness.csv",
            "global_shap": tmp_path / "global_shap.csv",
            "subgroup_risk": tmp_path / "subgroup_risk.csv",
            "high_risk_segments": tmp_path / "segments.csv",
            "high_risk_individuals": tmp_path / "individuals.csv",
            "subgroup_drivers": tmp_path / "drivers.csv",
            "threshold_curve": tmp_path / "threshold_curve.csv",
            "calibration_table": tmp_path / "calibration.csv",
            "artifact_manifest": tmp_path / "manifest.json",
            "model_report": tmp_path / "model_report.md",
            "feature_importance_fig": tmp_path / "feature_importance.png",
            "shap_summary_fig": tmp_path / "shap_summary.png",
            "risk_distribution_fig": tmp_path / "risk_distribution.png",
            "subgroup_comparison_fig": tmp_path / "subgroup.png",
            "calibration_fig": tmp_path / "calibration.png",
            "threshold_fig": tmp_path / "threshold.png",
            "heatmaps": {
                "province_risk_band": {"csv": tmp_path / "province.csv", "png": tmp_path / "province.png"},
                "care_hours_risk_band": {"csv": tmp_path / "hours.csv", "png": tmp_path / "hours.png"},
                "income_risk_band": {"csv": tmp_path / "income.csv", "png": tmp_path / "income.png"},
                "relationship_risk_band": {"csv": tmp_path / "relationship.csv", "png": tmp_path / "relationship.png"},
            },
        },
    )
    assert Path(outputs["manifest"]["artifact_manifest"]).exists()
    assert Path(outputs["manifest"]["scored_data"]).exists()
    assert Path(outputs["manifest"]["feature_importance_fig"]).exists()


def test_fail_fast_when_required_columns_are_missing() -> None:
    from damb.modeling import fit_binary_model  # noqa: PLC0415

    broken = pd.DataFrame({"distress_flag": [1.0], WEIGHT_COLUMN: [1.0]})
    with pytest.raises(ValueError, match="Missing columns required for training"):
        fit_binary_model(broken)
