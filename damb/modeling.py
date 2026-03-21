from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    BINARY_FEATURES,
    LEAKAGE_EXACT,
    LEAKAGE_PREFIXES,
    MODEL_FEATURES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    WEIGHT_COLUMN,
)


@dataclass
class TrainingResult:
    artifact: dict[str, Any]
    metrics: dict[str, Any]
    tuning_results: pd.DataFrame
    threshold_curve: pd.DataFrame
    calibration_table: pd.DataFrame
    test_predictions: pd.DataFrame
    validation_predictions: pd.DataFrame


def leakage_audit(feature_columns: tuple[str, ...] | list[str]) -> dict[str, list[str]]:
    exact = sorted(set(feature_columns) & LEAKAGE_EXACT)
    prefixed = sorted(
        column
        for column in feature_columns
        if any(column.startswith(prefix) for prefix in LEAKAGE_PREFIXES)
    )
    return {"exact_overlap": exact, "prefix_overlap": prefixed}


def assert_no_leakage(feature_columns: tuple[str, ...] | list[str]) -> None:
    audit = leakage_audit(feature_columns)
    offenders = sorted(set(audit["exact_overlap"] + audit["prefix_overlap"]))
    if offenders:
        raise ValueError(f"Leakage columns found in feature set: {offenders}")


def build_preprocessing_pipeline() -> ColumnTransformer:
    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", list(NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES)),
            ("nominal", nominal_pipeline, list(NOMINAL_FEATURES)),
        ]
    )


def _normalize_weights(weights: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    return values / values.mean()


def _build_model(params: dict[str, Any], random_state: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
        early_stopping_rounds=40,
        **params,
    )


def _candidate_param_grid() -> list[dict[str, Any]]:
    baseline = {
        "n_estimators": 600,
        "learning_rate": 0.04,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
        "max_delta_step": 0,
    }
    return [
        baseline,
        {**baseline, "max_depth": 4, "min_child_weight": 4},
        {**baseline, "learning_rate": 0.06, "n_estimators": 450},
        {**baseline, "subsample": 0.75, "colsample_bytree": 0.75},
        {**baseline, "gamma": 0.2, "reg_alpha": 0.1},
        {**baseline, "reg_lambda": 4.0, "min_child_weight": 8},
        {**baseline, "max_depth": 5, "min_child_weight": 10, "max_delta_step": 1},
        {**baseline, "learning_rate": 0.03, "n_estimators": 800, "subsample": 0.9},
    ]


def _fit_with_validation(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    w_val: pd.Series,
    params: dict[str, Any],
    random_state: int,
) -> tuple[ColumnTransformer, xgb.XGBClassifier]:
    preprocessor = build_preprocessing_pipeline()
    x_train_tx = preprocessor.fit_transform(x_train)
    x_val_tx = preprocessor.transform(x_val)
    train_weights = _normalize_weights(w_train)
    val_weights = _normalize_weights(w_val)
    model = _build_model(params, random_state=random_state)
    model.fit(
        x_train_tx,
        y_train,
        sample_weight=train_weights,
        eval_set=[(x_val_tx, y_val)],
        sample_weight_eval_set=[val_weights],
        verbose=False,
    )
    return preprocessor, model


def _score_predictions(
    truth: pd.Series,
    probabilities: np.ndarray,
    weights: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    binary = (probabilities >= threshold).astype(int)
    y_true = truth.to_numpy()
    weight_values = np.asarray(weights, dtype=float)
    tp = float(weight_values[(binary == 1) & (y_true == 1)].sum())
    fp = float(weight_values[(binary == 1) & (y_true == 0)].sum())
    tn = float(weight_values[(binary == 0) & (y_true == 0)].sum())
    fn = float(weight_values[(binary == 0) & (y_true == 1)].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    weighted_accuracy = float(np.average(binary == y_true, weights=weight_values))
    weighted_brier = float(np.average((probabilities - y_true) ** 2, weights=weight_values))
    return {
        "roc_auc": float(roc_auc_score(truth, probabilities, sample_weight=weights)),
        "pr_auc": float(average_precision_score(truth, probabilities, sample_weight=weights)),
        "prevalence": float(np.average(truth, weights=weights)),
        "weighted_accuracy": weighted_accuracy,
        "weighted_brier": weighted_brier,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _select_threshold(
    truth: pd.Series,
    probabilities: np.ndarray,
    weights: pd.Series,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.20, 0.80, 25):
        metrics = _score_predictions(truth, probabilities, weights, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "weighted_accuracy": metrics["weighted_accuracy"],
            }
        )
    curve = pd.DataFrame(rows)
    best_row = curve.sort_values(["f1", "weighted_accuracy", "specificity"], ascending=False).iloc[0]
    return float(best_row["threshold"]), curve


def _make_calibration_table(
    truth: pd.Series,
    probabilities: np.ndarray,
    weights: pd.Series,
    bins: int = 10,
) -> pd.DataFrame:
    calibration = pd.DataFrame(
        {
            "truth": truth.to_numpy(),
            "probability": probabilities,
            "weight": np.asarray(weights, dtype=float),
        }
    )
    calibration["bin"] = pd.qcut(calibration["probability"], q=bins, duplicates="drop")
    grouped = calibration.groupby("bin", observed=False)
    rows = []
    for interval, group in grouped:
        rows.append(
            {
                "bin": str(interval),
                "weighted_count": float(group["weight"].sum()),
                "mean_predicted_probability": float(np.average(group["probability"], weights=group["weight"])),
                "observed_rate": float(np.average(group["truth"], weights=group["weight"])),
            }
        )
    return pd.DataFrame(rows)


def _tune_parameters(
    x_train_val: pd.DataFrame,
    y_train_val: pd.Series,
    w_train_val: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    search_results: list[dict[str, Any]] = []
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    for idx, params in enumerate(_candidate_param_grid()):
        fold_scores: list[dict[str, float]] = []
        for fold_id, (train_idx, val_idx) in enumerate(folds.split(x_train_val, y_train_val), start=1):
            x_train = x_train_val.iloc[train_idx]
            y_train = y_train_val.iloc[train_idx]
            w_train = w_train_val.iloc[train_idx]
            x_val = x_train_val.iloc[val_idx]
            y_val = y_train_val.iloc[val_idx]
            w_val = w_train_val.iloc[val_idx]
            preprocessor, model = _fit_with_validation(
                x_train=x_train,
                y_train=y_train,
                w_train=w_train,
                x_val=x_val,
                y_val=y_val,
                w_val=w_val,
                params=params,
                random_state=random_state + idx + fold_id,
            )
            probabilities = model.predict_proba(preprocessor.transform(x_val))[:, 1]
            fold_scores.append(
                {
                    "fold_roc_auc": float(roc_auc_score(y_val, probabilities, sample_weight=w_val)),
                    "fold_pr_auc": float(average_precision_score(y_val, probabilities, sample_weight=w_val)),
                    "best_iteration": float(model.best_iteration if model.best_iteration is not None else params["n_estimators"]),
                }
            )
        fold_frame = pd.DataFrame(fold_scores)
        search_results.append(
            {
                "candidate_id": idx,
                **params,
                "mean_cv_roc_auc": float(fold_frame["fold_roc_auc"].mean()),
                "mean_cv_pr_auc": float(fold_frame["fold_pr_auc"].mean()),
                "selection_score": float(
                    0.6 * fold_frame["fold_pr_auc"].mean() + 0.4 * fold_frame["fold_roc_auc"].mean()
                ),
                "mean_best_iteration": int(round(fold_frame["best_iteration"].mean())),
            }
        )
    return pd.DataFrame(search_results).sort_values(
        ["selection_score", "mean_cv_pr_auc", "mean_cv_roc_auc"],
        ascending=False,
        ignore_index=True,
    )


def fit_binary_model(frame: pd.DataFrame, random_state: int = 42) -> TrainingResult:
    assert_no_leakage(MODEL_FEATURES)
    missing = sorted(set(MODEL_FEATURES + (WEIGHT_COLUMN, "distress_flag")) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns required for training: {missing}")

    analytic = frame.loc[frame["distress_flag"].notna()].copy()
    x = analytic.loc[:, MODEL_FEATURES]
    y = analytic["distress_flag"].astype(int)
    weights = analytic[WEIGHT_COLUMN].astype(float)

    x_train_val, x_test, y_train_val, y_test, w_train_val, w_test = train_test_split(
        x,
        y,
        weights,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )

    tuning_results = _tune_parameters(
        x_train_val=x_train_val,
        y_train_val=y_train_val,
        w_train_val=w_train_val,
        random_state=random_state,
    )
    best_row = tuning_results.iloc[0].to_dict()
    selected_params = {
        key: best_row[key]
        for key in [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "max_delta_step",
        ]
    }
    for int_key in ["n_estimators", "max_depth", "min_child_weight", "max_delta_step"]:
        selected_params[int_key] = int(selected_params[int_key])
    for float_key in ["learning_rate", "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda"]:
        selected_params[float_key] = float(selected_params[float_key])

    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(
        x_train_val,
        y_train_val,
        w_train_val,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_val,
    )
    selection_preprocessor, selection_model = _fit_with_validation(
        x_train=x_train,
        y_train=y_train,
        w_train=w_train,
        x_val=x_val,
        y_val=y_val,
        w_val=w_val,
        params=selected_params,
        random_state=random_state,
    )
    train_selection_probs = selection_model.predict_proba(selection_preprocessor.transform(x_train))[:, 1]
    val_selection_probs = selection_model.predict_proba(selection_preprocessor.transform(x_val))[:, 1]
    threshold, threshold_curve = _select_threshold(y_val, val_selection_probs, w_val)
    train_metrics = _score_predictions(y_train, train_selection_probs, w_train, threshold)
    validation_metrics = _score_predictions(y_val, val_selection_probs, w_val, threshold)
    final_n_estimators = int(selection_model.best_iteration + 1) if selection_model.best_iteration is not None else int(best_row["mean_best_iteration"])
    final_params = {**selected_params, "n_estimators": max(final_n_estimators, 50)}

    final_preprocessor = build_preprocessing_pipeline()
    x_train_val_tx = final_preprocessor.fit_transform(x_train_val)
    x_test_tx = final_preprocessor.transform(x_test)
    final_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
        **final_params,
    )
    final_model.fit(
        x_train_val_tx,
        y_train_val,
        sample_weight=_normalize_weights(w_train_val),
        verbose=False,
    )
    train_val_probs = final_model.predict_proba(x_train_val_tx)[:, 1]
    test_probs = final_model.predict_proba(x_test_tx)[:, 1]
    test_metrics = _score_predictions(y_test, test_probs, w_test, threshold)
    train_val_metrics = _score_predictions(y_train_val, train_val_probs, w_train_val, threshold)
    calibration_table = _make_calibration_table(y_test, test_probs, w_test)

    validation_predictions = x_val.copy()
    validation_predictions["actual_distress_flag"] = y_val.to_numpy()
    validation_predictions["predicted_probability"] = val_selection_probs
    validation_predictions["weight"] = w_val.to_numpy()

    test_predictions = x_test.copy()
    test_predictions["actual_distress_flag"] = y_test.to_numpy()
    test_predictions["predicted_probability"] = test_probs
    test_predictions["predicted_label"] = (test_probs >= threshold).astype(int)
    test_predictions["weight"] = w_test.to_numpy()

    metrics = {
        "selected_threshold": threshold,
        "selected_params": final_params,
        "tuning": tuning_results.to_dict(orient="records"),
        "split_rows": {
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_val)),
            "train_val_rows": int(len(x_train_val)),
            "test_rows": int(len(x_test)),
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "train_val_metrics": train_val_metrics,
        "test_metrics": test_metrics,
        "train_validation_gap": {
            "roc_auc_gap": float(train_metrics["roc_auc"] - validation_metrics["roc_auc"]),
            "pr_auc_gap": float(train_metrics["pr_auc"] - validation_metrics["pr_auc"]),
        },
    }

    artifact = {
        "model": final_model,
        "preprocessor": final_preprocessor,
        "feature_columns": list(MODEL_FEATURES),
        "threshold": threshold,
        "selected_params": final_params,
        "xgboost_version": xgb.__version__,
        "feature_names_out": final_preprocessor.get_feature_names_out().tolist(),
    }

    return TrainingResult(
        artifact=artifact,
        metrics=metrics,
        tuning_results=tuning_results,
        threshold_curve=threshold_curve,
        calibration_table=calibration_table,
        test_predictions=test_predictions,
        validation_predictions=validation_predictions,
    )
