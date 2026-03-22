from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .config import BINARY_FEATURES, MODEL_ARTIFACT_PATH, MODEL_FEATURES
from .data import coerce_feature_types


def _ensure_frame(row_or_frame: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(row_or_frame, pd.DataFrame):
        return row_or_frame.copy()
    if isinstance(row_or_frame, pd.Series):
        return row_or_frame.to_frame().T
    return pd.DataFrame([row_or_frame])


def _prepare_features(frame: pd.DataFrame, feature_columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    prepared = frame.copy()
    for column in feature_columns:
        if column not in prepared.columns:
            prepared[column] = np.nan
    typed = coerce_feature_types(prepared)
    for column in BINARY_FEATURES:
        if column in typed.columns:
            typed[column] = pd.to_numeric(typed[column], errors="coerce").replace({2.0: 0.0})
    return typed.loc[:, feature_columns]


def _resolve_base_feature_name(transformed_name: str, feature_columns: list[str] | tuple[str, ...]) -> str:
    if transformed_name.startswith("numeric__"):
        return transformed_name.split("__", 1)[1]
    remainder = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    for feature in sorted(feature_columns, key=len, reverse=True):
        if remainder == feature or remainder.startswith(f"{feature}_"):
            return feature
    return remainder


def compute_aggregated_contributions(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    features = _prepare_features(frame, artifact["feature_columns"])
    preprocessor = artifact["preprocessor"]
    model: xgb.XGBClassifier = artifact["model"]
    transformed = preprocessor.transform(features)
    feature_names = preprocessor.get_feature_names_out().tolist()
    dmatrix = xgb.DMatrix(transformed, feature_names=feature_names)
    contributions = model.get_booster().predict(dmatrix, pred_contribs=True)[:, :-1]
    grouped: dict[str, list[int]] = {}
    for idx, transformed_name in enumerate(feature_names):
        base_name = _resolve_base_feature_name(transformed_name, artifact["feature_columns"])
        grouped.setdefault(base_name, []).append(idx)
    aggregated = {
        feature: contributions[:, indices].sum(axis=1) for feature, indices in grouped.items()
    }
    return pd.DataFrame(aggregated, index=features.index)


def score_frame(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
    passthrough_columns: list[str] | None = None,
) -> pd.DataFrame:
    passthrough = list(passthrough_columns or [])
    prepared = _prepare_features(frame, artifact["feature_columns"])
    raw_probabilities = artifact["model"].predict_proba(artifact["preprocessor"].transform(prepared))[:, 1]
    calibrator_name = artifact.get("calibrator_name", "uncalibrated")
    if calibrator_name == "sigmoid" and artifact.get("sigmoid_calibrator") is not None:
        probabilities = artifact["sigmoid_calibrator"].predict_proba(np.clip(raw_probabilities, 1e-6, 1.0 - 1e-6).reshape(-1, 1))[:, 1]
    elif calibrator_name == "isotonic" and artifact.get("isotonic_calibrator") is not None:
        probabilities = artifact["isotonic_calibrator"].predict(np.clip(raw_probabilities, 1e-6, 1.0 - 1e-6))
    else:
        probabilities = raw_probabilities
    scored = frame.loc[:, [column for column in passthrough if column in frame.columns]].copy()
    scored["predicted_probability_raw"] = raw_probabilities
    scored["predicted_probability"] = probabilities
    scored["predicted_label"] = (probabilities >= float(artifact["threshold"])).astype(int)
    scored["risk_band"] = pd.cut(
        scored["predicted_probability"],
        bins=[0.0, 0.30, 0.50, 0.70, 1.01],
        labels=["Low", "Moderate", "High", "Very high"],
        include_lowest=True,
        right=False,
    ).astype(str)
    return scored


@dataclass
class CaregiverDistressScorer:
    artifact: dict[str, Any]

    @property
    def feature_columns(self) -> list[str]:
        return list(self.artifact["feature_columns"])

    def score_row(self, row: dict[str, Any] | pd.Series | pd.DataFrame) -> dict[str, Any]:
        frame = _ensure_frame(row)
        scored = score_frame(self.artifact, frame)
        contributions = compute_aggregated_contributions(self.artifact, frame).iloc[0]
        top = contributions.abs().sort_values(ascending=False).head(5).index
        top_rows = []
        for feature in top:
            top_rows.append(
                {
                    "feature": feature,
                    "feature_value": None if pd.isna(frame.iloc[0].get(feature)) else frame.iloc[0].get(feature),
                    "shap_value": float(contributions[feature]),
                    "direction": "higher risk" if float(contributions[feature]) >= 0 else "lower risk",
                }
            )
        return {
            "probability": float(scored.iloc[0]["predicted_probability"]),
            "label": int(scored.iloc[0]["predicted_label"]),
            "risk_band": str(scored.iloc[0]["risk_band"]),
            "top_contributors": top_rows,
        }

    def score_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        return score_frame(self.artifact, frame)


def load_trained_scorer(path: str | None = None) -> CaregiverDistressScorer:
    artifact = joblib.load(path or str(MODEL_ARTIFACT_PATH))
    missing = sorted(set(MODEL_FEATURES) - set(artifact["feature_columns"]))
    if missing:
        raise ValueError(f"Serialized artifact is missing expected feature columns: {missing}")
    return CaregiverDistressScorer(artifact=artifact)
