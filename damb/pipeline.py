from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyreadstat
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    GLOBAL_SHAP_PATH,
    LEAKAGE_COLUMNS,
    METRICS_PATH,
    MISSINGNESS_PATH,
    MODEL_ARTIFACT_PATH,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    RESERVE_CODE_MAP,
    TARGET_COLUMNS,
    TARGET_ITEM_LABELS,
    VALIDATION_PATH,
    WEIGHT_COLUMN,
)


@dataclass
class PreparedData:
    frame: pd.DataFrame
    universe_counts: dict[str, int]
    target_summary: dict[str, Any]


_SORTED_FEATURE_COLUMNS = sorted(FEATURE_COLUMNS, key=len, reverse=True)


def load_raw_data(path: str | None = None, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    dataset_path = path or str(RAW_DATA_PATH)
    usecols = list(columns) if columns is not None else list(REQUIRED_COLUMNS)
    frame, _ = pyreadstat.read_sas7bdat(dataset_path, usecols=usecols)
    missing = sorted(set(usecols) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame


def recode_reserve_codes(
    frame: pd.DataFrame,
    reserve_code_map: dict[str, set[int]] | None = None,
) -> pd.DataFrame:
    code_map = reserve_code_map or RESERVE_CODE_MAP
    recoded = frame.copy()
    for column, reserve_codes in code_map.items():
        if column in recoded.columns:
            recoded[column] = recoded[column].replace(sorted(reserve_codes), np.nan)
    return recoded


def build_target(frame: pd.DataFrame) -> pd.DataFrame:
    target_frame = frame.copy()
    binary_items = pd.DataFrame(index=target_frame.index)
    for column in TARGET_COLUMNS:
        binary_items[column] = target_frame[column].map({1.0: 1.0, 2.0: 0.0})
        target_frame[f"{column}_binary"] = binary_items[column]

    target_frame["distress_score"] = binary_items.sum(axis=1, min_count=1)
    any_observed = binary_items.notna().any(axis=1)
    target_frame["distress_flag"] = np.where(
        any_observed,
        (target_frame["distress_score"] >= 1).astype(float),
        np.nan,
    )
    return target_frame


def apply_modeling_universe(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = {
        "raw_rows": int(len(frame)),
        "dv_proxy_eq_2": int((frame["DV_PROXY"] == 2).sum()),
        "par_10_between_1_99": int(frame["PAR_10"].between(1, 99, inclusive="both").sum()),
        "hap_10c_between_1_6": int(frame["HAP_10C"].isin([1, 2, 3, 4, 5, 6]).sum()),
    }
    mask = (
        (frame["DV_PROXY"] == 2)
        & frame["PAR_10"].between(1, 99, inclusive="both")
        & frame["HAP_10C"].isin([1, 2, 3, 4, 5, 6])
    )
    filtered = frame.loc[mask].copy()
    counts["modeled_universe_rows"] = int(len(filtered))
    counts["hap_10c_eq_1_rows"] = int((filtered["HAP_10C"] == 1).sum())
    return filtered, counts


def summarize_target(frame: pd.DataFrame) -> dict[str, Any]:
    analytic_mask = frame["distress_flag"].notna()
    weights = frame.loc[analytic_mask, WEIGHT_COLUMN].astype(float)
    labels = frame.loc[analytic_mask, "distress_flag"].astype(int)
    weighted_prevalence = float(np.average(labels, weights=weights)) if len(labels) else np.nan
    item_counts = {}
    for column in TARGET_COLUMNS:
        item_series = frame[column].map({1.0: 1.0, 2.0: 0.0})
        item_counts[column] = {
            "label": TARGET_ITEM_LABELS[column],
            "available_rows": int(item_series.notna().sum()),
            "yes_count": int((item_series == 1).sum()),
        }
    return {
        "analytic_rows": int(analytic_mask.sum()),
        "unweighted_prevalence": float(labels.mean()) if len(labels) else np.nan,
        "weighted_prevalence": weighted_prevalence,
        "distress_score_distribution": {
            str(int(key)): int(value)
            for key, value in frame.loc[analytic_mask, "distress_score"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
        },
        "target_items": item_counts,
    }


def assert_no_leakage(feature_columns: tuple[str, ...] | list[str]) -> None:
    overlap = sorted(set(feature_columns) & LEAKAGE_COLUMNS)
    if overlap:
        raise ValueError(f"Leakage columns found in feature set: {overlap}")


def select_and_prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    assert_no_leakage(FEATURE_COLUMNS)
    missing_columns = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")
    if WEIGHT_COLUMN not in frame.columns:
        raise ValueError(f"Missing weight column: {WEIGHT_COLUMN}")
    if "distress_flag" not in frame.columns:
        raise ValueError("distress_flag must be built before feature selection")

    analytic = frame.loc[frame["distress_flag"].notna()].copy()
    if analytic.empty:
        raise ValueError("No analytic rows remain after target construction")
    if analytic[WEIGHT_COLUMN].isna().any() or (analytic[WEIGHT_COLUMN] <= 0).any():
        raise ValueError("Modeling weights must be positive and non-null")

    features = analytic.loc[:, FEATURE_COLUMNS].copy()
    labels = analytic["distress_flag"].astype(int)
    weights = analytic[WEIGHT_COLUMN].astype(float)
    return features, labels, weights


def build_preprocessor() -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ]
    )


def _resolve_base_feature_name(transformed_name: str) -> str:
    if transformed_name.startswith("numeric__"):
        return transformed_name.split("__", 1)[1]
    remainder = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    for feature in _SORTED_FEATURE_COLUMNS:
        if remainder == feature or remainder.startswith(f"{feature}_"):
            return feature
    return remainder


def _aggregate_contributions(
    contributions: np.ndarray,
    feature_names: list[str] | np.ndarray,
) -> dict[str, np.ndarray]:
    feature_names = [str(name) for name in feature_names]
    grouped: dict[str, list[int]] = {}
    for index, transformed_name in enumerate(feature_names):
        base_name = _resolve_base_feature_name(transformed_name)
        grouped.setdefault(base_name, []).append(index)
    return {
        feature: contributions[:, indices].sum(axis=1)
        for feature, indices in grouped.items()
    }


def evaluate_model(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    weight_test: pd.Series,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    weight_train: pd.Series,
) -> dict[str, float]:
    train_proba = pipeline.predict_proba(x_train)[:, 1]
    test_proba = pipeline.predict_proba(x_test)[:, 1]
    train_pred = (train_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    def weighted_accuracy(pred: np.ndarray, truth: pd.Series, weights: pd.Series) -> float:
        return float(np.average((pred == truth.to_numpy()).astype(float), weights=weights))

    return {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_prevalence_unweighted": float(y_train.mean()),
        "test_prevalence_unweighted": float(y_test.mean()),
        "train_prevalence_weighted": float(np.average(y_train, weights=weight_train)),
        "test_prevalence_weighted": float(np.average(y_test, weights=weight_test)),
        "roc_auc_unweighted": float(roc_auc_score(y_test, test_proba)),
        "roc_auc_weighted": float(roc_auc_score(y_test, test_proba, sample_weight=weight_test)),
        "pr_auc_unweighted": float(average_precision_score(y_test, test_proba)),
        "pr_auc_weighted": float(
            average_precision_score(y_test, test_proba, sample_weight=weight_test)
        ),
        "weighted_accuracy": weighted_accuracy(test_pred, y_test, weight_test),
        "unweighted_accuracy": float((test_pred == y_test.to_numpy()).mean()),
        "train_weighted_accuracy": weighted_accuracy(train_pred, y_train, weight_train),
    }


def compute_shap_outputs(
    pipeline: Pipeline,
    features: pd.DataFrame,
) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    model: xgb.XGBClassifier = pipeline.named_steps["model"]
    transformed = preprocessor.transform(features)
    feature_names = preprocessor.get_feature_names_out()
    dmatrix = xgb.DMatrix(transformed, feature_names=feature_names.tolist())
    contributions = model.get_booster().predict(dmatrix, pred_contribs=True)
    aggregated = _aggregate_contributions(contributions[:, :-1], feature_names)
    global_importance = pd.DataFrame(
        {
            "feature": list(aggregated.keys()),
            "mean_abs_shap": [float(np.mean(np.abs(values))) for values in aggregated.values()],
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    return global_importance


def compute_row_contributors(
    pipeline: Pipeline,
    row: pd.DataFrame,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    preprocessor = pipeline.named_steps["preprocess"]
    model: xgb.XGBClassifier = pipeline.named_steps["model"]
    transformed = preprocessor.transform(row)
    feature_names = preprocessor.get_feature_names_out()
    dmatrix = xgb.DMatrix(transformed, feature_names=feature_names.tolist())
    contributions = model.get_booster().predict(dmatrix, pred_contribs=True)[:, :-1]
    aggregated = _aggregate_contributions(contributions, feature_names)
    ranked_features = sorted(
        aggregated.items(),
        key=lambda item: abs(float(item[1][0])),
        reverse=True,
    )[:top_n]
    output = []
    row_values = row.iloc[0].to_dict()
    for feature, values in ranked_features:
        value = float(values[0])
        output.append(
            {
                "feature": feature,
                "feature_value": None if pd.isna(row_values.get(feature)) else float(row_values.get(feature)),
                "shap_value": value,
                "direction": "higher risk" if value >= 0 else "lower risk",
            }
        )
    return output


def fit_binary_model(
    frame: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, Any]:
    x, y, weights = select_and_prepare_features(frame)
    normalized_weights = weights / weights.mean()

    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
        x,
        y,
        normalized_weights,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=2.0,
                    reg_lambda=1.0,
                    random_state=random_state,
                    tree_method="hist",
                    n_jobs=2,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train, model__sample_weight=w_train.to_numpy())

    metrics = evaluate_model(
        pipeline=pipeline,
        x_test=x_test,
        y_test=y_test,
        weight_test=w_test,
        x_train=x_train,
        y_train=y_train,
        weight_train=w_train,
    )
    shap_importance = compute_shap_outputs(pipeline, x)
    missingness = x.isna().mean().sort_values(ascending=False).rename("missing_rate")
    artifact = {
        "pipeline": pipeline,
        "feature_columns": list(FEATURE_COLUMNS),
        "threshold": 0.5,
        "metrics": metrics,
        "top_features": shap_importance.head(15).to_dict(orient="records"),
        "feature_missingness": missingness.to_dict(),
    }
    return {
        "artifact": artifact,
        "metrics": metrics,
        "global_shap": shap_importance,
        "missingness": missingness.reset_index().rename(columns={"index": "feature"}),
    }


def prepare_training_frame(path: str | None = None) -> PreparedData:
    raw = load_raw_data(path=path)
    recoded = recode_reserve_codes(raw)
    targeted = build_target(recoded)
    universe, counts = apply_modeling_universe(targeted)
    target_summary = summarize_target(universe)
    return PreparedData(frame=universe, universe_counts=counts, target_summary=target_summary)


def save_training_outputs(
    artifact: dict[str, Any],
    metrics: dict[str, Any],
    global_shap: pd.DataFrame,
    missingness: pd.DataFrame,
    validation_summary: dict[str, Any],
) -> None:
    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_SHAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MISSINGNESS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, MODEL_ARTIFACT_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    global_shap.to_csv(GLOBAL_SHAP_PATH, index=False)
    missingness.to_csv(MISSINGNESS_PATH, index=False)
    with VALIDATION_PATH.open("w", encoding="utf-8") as handle:
        json.dump(validation_summary, handle, indent=2)
