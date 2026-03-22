from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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
    RESERVE_CODE_MAP,
    WEIGHT_COLUMN,
)


MISSING_STRATEGIES = ("native", "indicators", "drop_sparse")
SPARSE_MISSINGNESS_THRESHOLD = 0.45
SUBGROUP_MIN_COUNT = 50
SUBGROUP_MIN_WEIGHT = 5_000.0
BOOTSTRAP_ITERATIONS = 250


@dataclass
class TrainingResult:
    artifact: dict[str, Any]
    metrics: dict[str, Any]
    tuning_results: pd.DataFrame
    threshold_curve: pd.DataFrame
    calibration_table: pd.DataFrame
    test_predictions: pd.DataFrame
    validation_predictions: pd.DataFrame
    threshold_analysis: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration_report: dict[str, Any] = field(default_factory=dict)
    feature_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    missingness_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    subgroup_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    global_shap: pd.DataFrame = field(default_factory=pd.DataFrame)


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


def _family(column: str) -> str:
    if column in NOMINAL_FEATURES:
        return "nominal"
    if column in NUMERIC_FEATURES:
        return "numeric"
    if column in ORDINAL_FEATURES:
        return "ordinal"
    if column in BINARY_FEATURES:
        return "binary"
    return "other"


def _skip_artifact_risk(column: str, missing_pct: float) -> str:
    if missing_pct < 0.10:
        return "low"
    if column.startswith(("FWA_", "WTI_", "UHW_", "NWE_", "UCA_", "COW_")) and missing_pct >= 0.25:
        return "high"
    if column.startswith(("PR", "ARV_", "AGEPR", "PHSD")) and missing_pct >= 0.25:
        return "high"
    if missing_pct >= 0.35:
        return "medium"
    return "low"


def _select_feature_columns(
    frame: pd.DataFrame,
    strategy: str,
    sparse_threshold: float = SPARSE_MISSINGNESS_THRESHOLD,
) -> list[str]:
    columns = list(MODEL_FEATURES)
    if strategy != "drop_sparse":
        return columns
    missing_pct = frame.loc[:, columns].isna().mean()
    kept = [column for column in columns if float(missing_pct[column]) <= sparse_threshold]
    return kept or columns


def build_preprocessing_pipeline(
    missing_strategy: str = "native",
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> ColumnTransformer:
    active_features = list(feature_columns or MODEL_FEATURES)
    numeric_columns = [column for column in NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES if column in active_features]
    nominal_columns = [column for column in NOMINAL_FEATURES if column in active_features]

    if missing_strategy == "indicators":
        numeric_transformer: Any = SimpleImputer(strategy="constant", fill_value=-999.0, add_indicator=True)
    else:
        numeric_transformer = "passthrough"

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
            ("numeric", numeric_transformer, numeric_columns),
            ("nominal", nominal_pipeline, nominal_columns),
        ]
    )


def _normalize_weights(weights: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    return values / values.mean()


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)


def _weighted_log_loss(truth: pd.Series, probabilities: np.ndarray, weights: pd.Series) -> float:
    probs = _clip_probabilities(probabilities)
    y_true = np.asarray(truth, dtype=int)
    weight_values = np.asarray(weights, dtype=float)
    losses = -(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
    return float(np.average(losses, weights=weight_values))


def _weighted_brier(truth: pd.Series, probabilities: np.ndarray, weights: pd.Series) -> float:
    y_true = np.asarray(truth, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    weight_values = np.asarray(weights, dtype=float)
    return float(np.average((probs - y_true) ** 2, weights=weight_values))


def _score_predictions(
    truth: pd.Series,
    probabilities: np.ndarray,
    weights: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    probs = _clip_probabilities(probabilities)
    binary = (probs >= threshold).astype(int)
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
    positive_rate = float(np.average(binary, weights=weight_values))
    return {
        "roc_auc": float(roc_auc_score(truth, probs, sample_weight=weights)),
        "pr_auc": float(average_precision_score(truth, probs, sample_weight=weights)),
        "prevalence": float(np.average(truth, weights=weights)),
        "weighted_accuracy": weighted_accuracy,
        "weighted_brier": _weighted_brier(truth, probs, weights),
        "log_loss": _weighted_log_loss(truth, probs, weights),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "positive_rate": positive_rate,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _candidate_param_grid() -> list[dict[str, Any]]:
    return [
        {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 2.0, "max_delta_step": 0},
        {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 4, "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 2.0, "max_delta_step": 0},
        {"n_estimators": 800, "learning_rate": 0.02, "max_depth": 3, "min_child_weight": 8, "subsample": 0.90, "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 4.0, "max_delta_step": 0},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 5, "subsample": 0.75, "colsample_bytree": 0.75, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 2.0, "max_delta_step": 0},
        {"n_estimators": 500, "learning_rate": 0.04, "max_depth": 5, "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 2.0, "max_delta_step": 1},
        {"n_estimators": 500, "learning_rate": 0.04, "max_depth": 4, "min_child_weight": 8, "subsample": 0.80, "colsample_bytree": 0.80, "gamma": 0.2, "reg_alpha": 0.1, "reg_lambda": 4.0, "max_delta_step": 0},
        {"n_estimators": 450, "learning_rate": 0.06, "max_depth": 3, "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.2, "reg_lambda": 2.0, "max_delta_step": 0},
        {"n_estimators": 700, "learning_rate": 0.025, "max_depth": 5, "min_child_weight": 12, "subsample": 0.85, "colsample_bytree": 0.75, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 6.0, "max_delta_step": 1},
        {"n_estimators": 450, "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 10, "subsample": 0.90, "colsample_bytree": 0.90, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 2.0, "max_delta_step": 0},
        {"n_estimators": 500, "learning_rate": 0.04, "max_depth": 4, "min_child_weight": 6, "subsample": 0.70, "colsample_bytree": 0.70, "gamma": 0.3, "reg_alpha": 0.2, "reg_lambda": 4.0, "max_delta_step": 0},
        {"n_estimators": 650, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 12, "subsample": 0.80, "colsample_bytree": 0.80, "gamma": 0.5, "reg_alpha": 0.5, "reg_lambda": 6.0, "max_delta_step": 1},
        {"n_estimators": 350, "learning_rate": 0.08, "max_depth": 3, "min_child_weight": 4, "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0, "max_delta_step": 0},
    ]


def _build_xgb_model(params: dict[str, Any], random_state: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
        early_stopping_rounds=40,
        **params,
    )


def _fit_xgb_fold(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    w_eval: pd.Series,
    params: dict[str, Any],
    random_state: int,
    missing_strategy: str,
    feature_columns: list[str],
) -> tuple[ColumnTransformer, xgb.XGBClassifier]:
    preprocessor = build_preprocessing_pipeline(missing_strategy=missing_strategy, feature_columns=feature_columns)
    x_train_tx = preprocessor.fit_transform(x_train.loc[:, feature_columns])
    x_eval_tx = preprocessor.transform(x_eval.loc[:, feature_columns])
    model = _build_xgb_model(params, random_state=random_state)
    model.fit(
        x_train_tx,
        y_train,
        sample_weight=_normalize_weights(w_train),
        eval_set=[(x_eval_tx, y_eval)],
        sample_weight_eval_set=[_normalize_weights(w_eval)],
        verbose=False,
    )
    return preprocessor, model


def _prepare_training_frame(
    x: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    return train_test_split(
        x,
        y,
        w,
        test_size=0.15,
        random_state=random_state,
        stratify=y,
    )


def _tune_parameters(
    x_development: pd.DataFrame,
    y_development: pd.Series,
    w_development: pd.Series,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    candidate_id = 0
    for missing_strategy in MISSING_STRATEGIES:
        for params in _candidate_param_grid():
            per_fold: list[dict[str, float]] = []
            for fold_id, (train_idx, val_idx) in enumerate(folds.split(x_development, y_development), start=1):
                x_train_full = x_development.iloc[train_idx]
                y_train_full = y_development.iloc[train_idx]
                w_train_full = w_development.iloc[train_idx]
                x_val = x_development.iloc[val_idx]
                y_val = y_development.iloc[val_idx]
                w_val = w_development.iloc[val_idx]
                feature_columns = _select_feature_columns(x_train_full, missing_strategy)
                x_fit, x_early, y_fit, y_early, w_fit, w_early = _prepare_training_frame(
                    x_train_full.loc[:, feature_columns],
                    y_train_full,
                    w_train_full,
                    random_state=random_state + candidate_id + fold_id,
                )
                preprocessor, model = _fit_xgb_fold(
                    x_train=x_fit,
                    y_train=y_fit,
                    w_train=w_fit,
                    x_eval=x_early,
                    y_eval=y_early,
                    w_eval=w_early,
                    params=params,
                    random_state=random_state + candidate_id + fold_id,
                    missing_strategy=missing_strategy,
                    feature_columns=feature_columns,
                )
                val_probs = model.predict_proba(preprocessor.transform(x_val.loc[:, feature_columns]))[:, 1]
                fold_metrics = _score_predictions(y_val, val_probs, w_val, threshold=0.5)
                row = {
                    "candidate_id": candidate_id,
                    "missing_strategy": missing_strategy,
                    "fold_id": fold_id,
                    "feature_count": len(feature_columns),
                    "best_iteration": int(model.best_iteration if model.best_iteration is not None else params["n_estimators"]),
                    "roc_auc": fold_metrics["roc_auc"],
                    "pr_auc": fold_metrics["pr_auc"],
                    "weighted_brier": fold_metrics["weighted_brier"],
                    "log_loss": fold_metrics["log_loss"],
                    **params,
                }
                fold_rows.append(row)
                per_fold.append(row)
            fold_frame = pd.DataFrame(per_fold)
            summary_rows.append(
                {
                    "candidate_id": candidate_id,
                    "missing_strategy": missing_strategy,
                    "feature_count": float(fold_frame["feature_count"].mean()),
                    "mean_best_iteration": int(round(fold_frame["best_iteration"].mean())),
                    "mean_cv_roc_auc": float(fold_frame["roc_auc"].mean()),
                    "mean_cv_pr_auc": float(fold_frame["pr_auc"].mean()),
                    "mean_cv_brier": float(fold_frame["weighted_brier"].mean()),
                    "mean_cv_log_loss": float(fold_frame["log_loss"].mean()),
                    **params,
                }
            )
            candidate_id += 1
    tuning_results = pd.DataFrame(summary_rows).sort_values(
        ["mean_cv_pr_auc", "mean_cv_roc_auc", "mean_cv_brier", "mean_cv_log_loss"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    return tuning_results, pd.DataFrame(fold_rows)


def _threshold_table(
    truth: pd.Series,
    probabilities: np.ndarray,
    weights: pd.Series,
) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.20, 0.90, 29):
        metrics = _score_predictions(truth, probabilities, weights, float(threshold))
        rows.append({"threshold": float(threshold), **{key: metrics[key] for key in ["precision", "recall", "specificity", "f1", "weighted_accuracy", "positive_rate", "weighted_brier", "log_loss"]}})
    return pd.DataFrame(rows)


def _pick_threshold_row(curve: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "high_precision":
        eligible = curve.loc[(curve["precision"] >= 0.85) & (curve["specificity"] >= 0.20)]
        candidate = eligible if not eligible.empty else curve
        return candidate.sort_values(["recall", "f1", "specificity", "threshold"], ascending=[False, False, False, False]).iloc[0]
    if mode == "high_recall":
        eligible = curve.loc[(curve["recall"] >= 0.85) & (curve["specificity"] >= 0.15)]
        candidate = eligible if not eligible.empty else curve
        return candidate.sort_values(["precision", "f1", "specificity", "threshold"], ascending=[False, False, False, False]).iloc[0]
    guarded = curve.loc[(curve["specificity"] >= 0.10) & (curve["positive_rate"] <= 0.90)]
    candidate = guarded if not guarded.empty else curve
    candidate = candidate.copy()
    candidate["balanced_score"] = (candidate["precision"] + candidate["recall"] + candidate["specificity"]) / 3.0
    return candidate.sort_values(["balanced_score", "f1", "weighted_accuracy", "threshold"], ascending=[False, False, False, False]).iloc[0]


def _select_thresholds(
    truth: pd.Series,
    probabilities: np.ndarray,
    weights: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame]:
    curve = _threshold_table(truth, probabilities, weights)
    selected_rows = []
    thresholds: dict[str, float] = {}
    for mode in ["balanced", "high_precision", "high_recall"]:
        chosen = _pick_threshold_row(curve, mode)
        thresholds[mode] = float(chosen["threshold"])
        selected_rows.append({"mode": mode, **chosen.to_dict()})
    selected = pd.DataFrame(selected_rows)
    curve["selected_mode"] = ""
    for _, row in selected.iterrows():
        mask = np.isclose(curve["threshold"], float(row["threshold"]))
        curve.loc[mask, "selected_mode"] = row["mode"]
    return thresholds, curve


def _select_threshold(
    truth: pd.Series,
    probabilities: np.ndarray,
    weights: pd.Series,
) -> tuple[float, pd.DataFrame]:
    thresholds, curve = _select_thresholds(truth, probabilities, weights)
    return float(thresholds["balanced"]), curve


def _make_calibration_table(
    truth: pd.Series,
    probabilities: np.ndarray,
    weights: pd.Series,
    bins: int = 10,
) -> pd.DataFrame:
    calibration = pd.DataFrame(
        {
            "truth": truth.to_numpy(),
            "probability": _clip_probabilities(probabilities),
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


def _fit_sigmoid_calibrator(probabilities: np.ndarray, truth: pd.Series, weights: pd.Series) -> LogisticRegression:
    calibrator = LogisticRegression(solver="lbfgs", max_iter=2000)
    calibrator.fit(_clip_probabilities(probabilities).reshape(-1, 1), truth, sample_weight=np.asarray(weights, dtype=float))
    return calibrator


def _apply_sigmoid_calibrator(calibrator: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(_clip_probabilities(probabilities).reshape(-1, 1))[:, 1]


def _fit_isotonic_calibrator(probabilities: np.ndarray, truth: pd.Series, weights: pd.Series) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(_clip_probabilities(probabilities), np.asarray(truth, dtype=float), sample_weight=np.asarray(weights, dtype=float))
    return calibrator


def _evaluate_probability_set(
    truth: pd.Series,
    probabilities: np.ndarray,
    weights: pd.Series,
) -> dict[str, Any]:
    return {
        "roc_auc": float(roc_auc_score(truth, probabilities, sample_weight=weights)),
        "pr_auc": float(average_precision_score(truth, probabilities, sample_weight=weights)),
        "weighted_brier": _weighted_brier(truth, probabilities, weights),
        "log_loss": _weighted_log_loss(truth, probabilities, weights),
    }


def _select_calibration_method(
    calibration_report: dict[str, dict[str, dict[str, Any]]],
) -> str:
    baseline = calibration_report["uncalibrated"]["calibration_set"]
    sigmoid = calibration_report["sigmoid"]["calibration_set"]
    sigmoid_brier_gain = baseline["weighted_brier"] - sigmoid["weighted_brier"]
    sigmoid_log_loss_gain = baseline["log_loss"] - sigmoid["log_loss"]
    if sigmoid_brier_gain >= 0.005 and sigmoid_log_loss_gain >= 0.005:
        isotonic = calibration_report["isotonic"]["calibration_set"]
        isotonic_brier_gain = baseline["weighted_brier"] - isotonic["weighted_brier"]
        isotonic_log_loss_gain = baseline["log_loss"] - isotonic["log_loss"]
        if isotonic_brier_gain >= 0.015 and isotonic_log_loss_gain >= 0.015:
            return "isotonic"
        return "sigmoid"
    return "uncalibrated"


def _bootstrap_metric_intervals(
    truth: pd.Series,
    probabilities: np.ndarray,
    weights: pd.Series,
    threshold: float,
    random_state: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(random_state)
    rows = []
    truth_arr = np.asarray(truth)
    prob_arr = np.asarray(probabilities)
    weight_arr = np.asarray(weights, dtype=float)
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample_idx = rng.integers(0, len(truth_arr), size=len(truth_arr))
        sample_truth = pd.Series(truth_arr[sample_idx])
        sample_prob = prob_arr[sample_idx]
        sample_weight = pd.Series(weight_arr[sample_idx])
        if sample_truth.nunique() < 2:
            continue
        rows.append(_score_predictions(sample_truth, sample_prob, sample_weight, threshold))
    frame = pd.DataFrame(rows)
    intervals: dict[str, dict[str, float]] = {}
    for metric in ["roc_auc", "pr_auc", "weighted_accuracy", "precision", "recall", "specificity", "weighted_brier", "log_loss"]:
        intervals[metric] = {
            "lower": float(frame[metric].quantile(0.025)),
            "upper": float(frame[metric].quantile(0.975)),
        }
    return intervals


def _fit_logistic_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_test: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    preprocessor = build_preprocessing_pipeline(missing_strategy="indicators", feature_columns=feature_columns)
    x_train_tx = preprocessor.fit_transform(x_train.loc[:, feature_columns])
    x_test_tx = preprocessor.transform(x_test.loc[:, feature_columns])
    model = LogisticRegression(max_iter=10000, C=1.0)
    model.fit(x_train_tx, y_train, sample_weight=np.asarray(w_train, dtype=float))
    return model.predict_proba(x_test_tx)[:, 1]


def _fit_dummy_baseline(
    y_train: pd.Series,
    w_train: pd.Series,
    rows: int,
) -> np.ndarray:
    model = DummyClassifier(strategy="prior")
    model.fit(np.zeros((len(y_train), 1)), y_train, sample_weight=np.asarray(w_train, dtype=float))
    return model.predict_proba(np.zeros((rows, 1)))[:, 1]


def _feature_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    missingness_rows = []
    numeric_candidates = [column for column in MODEL_FEATURES if column in frame.columns and _family(column) in {"numeric", "ordinal", "binary"}]
    numeric_frame = frame.loc[:, numeric_candidates].apply(pd.to_numeric, errors="coerce")
    correlation = numeric_frame.corr(numeric_only=True).abs() if len(numeric_candidates) > 1 else pd.DataFrame()
    for column in MODEL_FEATURES:
        series = frame[column]
        missing_pct = float(series.isna().mean())
        non_missing = series.dropna()
        dominant_share = float(non_missing.value_counts(normalize=True).iloc[0]) if not non_missing.empty else np.nan
        max_corr = np.nan
        if not correlation.empty and column in correlation.columns:
            corr_series = correlation[column].drop(index=column, errors="ignore")
            if not corr_series.empty:
                max_corr = float(corr_series.max())
        rows.append(
            {
                "feature": column,
                "family": _family(column),
                "missing_pct": missing_pct,
                "non_missing_count": int(series.notna().sum()),
                "unique_non_missing": int(non_missing.nunique()),
                "dominant_value_share": dominant_share,
                "cardinality_class": "high" if non_missing.nunique() > 15 else "medium" if non_missing.nunique() > 6 else "low",
                "leakage_risk": "blocked" if column in LEAKAGE_EXACT or any(column.startswith(prefix) for prefix in LEAKAGE_PREFIXES) else "none",
                "reserve_codes": sorted(RESERVE_CODE_MAP.get(column, set())),
                "skip_artifact_risk": _skip_artifact_risk(column, missing_pct),
                "redundancy_max_abs_corr": max_corr,
                "drop_sparse_candidate": bool(missing_pct > SPARSE_MISSINGNESS_THRESHOLD),
            }
        )
        missingness_rows.append(
            {
                "feature": column,
                "missing_pct": missing_pct,
                "missing_count": int(series.isna().sum()),
                "non_missing_count": int(series.notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["drop_sparse_candidate", "missing_pct"], ascending=[False, False], ignore_index=True), pd.DataFrame(missingness_rows).sort_values("missing_pct", ascending=False, ignore_index=True)


def _subgroup_diagnostics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    truth: pd.Series,
    weights: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    diagnostics = frame.copy()
    diagnostics["probability"] = probabilities
    diagnostics["truth"] = truth.to_numpy()
    diagnostics["weight"] = np.asarray(weights, dtype=float)
    diagnostics["predicted_label"] = (probabilities >= threshold).astype(int)
    for dimension in ["PRV", "HAP_10C", "TTLINCG1", "PRA_10GR", "AGEGR10", "SEX"]:
        if dimension not in diagnostics.columns:
            continue
        grouped = diagnostics.groupby(dimension, dropna=False, observed=False)
        for group_value, group in grouped:
            weighted_count = float(group["weight"].sum())
            unweighted_count = int(len(group))
            suppressed = unweighted_count < SUBGROUP_MIN_COUNT or weighted_count < SUBGROUP_MIN_WEIGHT or group["truth"].nunique() < 2
            row = {
                "dimension": dimension,
                "group_value": "Missing" if pd.isna(group_value) else str(group_value),
                "unweighted_count": unweighted_count,
                "weighted_count": weighted_count,
                "avg_predicted_risk": float(np.average(group["probability"], weights=group["weight"])),
                "observed_rate": float(np.average(group["truth"], weights=group["weight"])),
                "suppressed": bool(suppressed),
                "suppression_reason": "low_support_or_single_class" if suppressed else "",
            }
            if not suppressed:
                group_metrics = _score_predictions(group["truth"], group["probability"], group["weight"], threshold)
                row.update(
                    {
                        "roc_auc": group_metrics["roc_auc"],
                        "pr_auc": group_metrics["pr_auc"],
                        "weighted_brier": group_metrics["weighted_brier"],
                        "precision": group_metrics["precision"],
                        "recall": group_metrics["recall"],
                        "specificity": group_metrics["specificity"],
                    }
                )
            else:
                row.update(
                    {
                        "roc_auc": np.nan,
                        "pr_auc": np.nan,
                        "weighted_brier": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "specificity": np.nan,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["dimension", "suppressed", "weighted_count"], ascending=[True, True, False], ignore_index=True)


def fit_binary_model(frame: pd.DataFrame, random_state: int = 42) -> TrainingResult:
    assert_no_leakage(MODEL_FEATURES)
    missing = sorted(set(MODEL_FEATURES + (WEIGHT_COLUMN, "distress_flag")) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns required for training: {missing}")

    analytic = frame.loc[frame["distress_flag"].notna()].copy()
    feature_audit, missingness_report = _feature_audit(analytic)

    x = analytic.loc[:, MODEL_FEATURES]
    y = analytic["distress_flag"].astype(int)
    weights = analytic[WEIGHT_COLUMN].astype(float)

    x_development, x_test, y_development, y_test, w_development, w_test = train_test_split(
        x,
        y,
        weights,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )

    tuning_results, fold_metrics = _tune_parameters(
        x_development=x_development,
        y_development=y_development,
        w_development=w_development,
        random_state=random_state,
    )
    best_row = tuning_results.iloc[0].to_dict()
    best_strategy = str(best_row["missing_strategy"])
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

    x_train_full, x_calibration, y_train_full, y_calibration, w_train_full, w_calibration = train_test_split(
        x_development,
        y_development,
        w_development,
        test_size=0.25,
        random_state=random_state,
        stratify=y_development,
    )
    feature_columns = _select_feature_columns(x_train_full, best_strategy)
    x_fit, x_early, y_fit, y_early, w_fit, w_early = _prepare_training_frame(
        x_train_full.loc[:, feature_columns],
        y_train_full,
        w_train_full,
        random_state=random_state,
    )
    preprocessor, model = _fit_xgb_fold(
        x_train=x_fit,
        y_train=y_fit,
        w_train=w_fit,
        x_eval=x_early,
        y_eval=y_early,
        w_eval=w_early,
        params=selected_params,
        random_state=random_state,
        missing_strategy=best_strategy,
        feature_columns=feature_columns,
    )

    calibration_raw_probs = model.predict_proba(preprocessor.transform(x_calibration.loc[:, feature_columns]))[:, 1]
    test_raw_probs = model.predict_proba(preprocessor.transform(x_test.loc[:, feature_columns]))[:, 1]
    train_full_raw_probs = model.predict_proba(preprocessor.transform(x_train_full.loc[:, feature_columns]))[:, 1]

    sigmoid_calibrator = _fit_sigmoid_calibrator(calibration_raw_probs, y_calibration, w_calibration)
    isotonic_calibrator = _fit_isotonic_calibrator(calibration_raw_probs, y_calibration, w_calibration)

    calibration_prob_sets = {
        "uncalibrated": calibration_raw_probs,
        "sigmoid": _apply_sigmoid_calibrator(sigmoid_calibrator, calibration_raw_probs),
        "isotonic": isotonic_calibrator.predict(_clip_probabilities(calibration_raw_probs)),
    }
    test_prob_sets = {
        "uncalibrated": test_raw_probs,
        "sigmoid": _apply_sigmoid_calibrator(sigmoid_calibrator, test_raw_probs),
        "isotonic": isotonic_calibrator.predict(_clip_probabilities(test_raw_probs)),
    }
    train_prob_sets = {
        "uncalibrated": train_full_raw_probs,
        "sigmoid": _apply_sigmoid_calibrator(sigmoid_calibrator, train_full_raw_probs),
        "isotonic": isotonic_calibrator.predict(_clip_probabilities(train_full_raw_probs)),
    }

    calibration_report = {
        name: {
            "calibration_set": _evaluate_probability_set(y_calibration, cal_probs, w_calibration),
            "test_set": _evaluate_probability_set(y_test, test_probs, w_test),
        }
        for name, cal_probs, test_probs in zip(
            calibration_prob_sets.keys(),
            calibration_prob_sets.values(),
            test_prob_sets.values(),
        )
    }

    selected_calibration = _select_calibration_method(calibration_report)
    selected_calibration_probs = calibration_prob_sets[selected_calibration]
    selected_test_probs = test_prob_sets[selected_calibration]
    selected_train_probs = train_prob_sets[selected_calibration]
    thresholds, threshold_curve = _select_thresholds(y_calibration, selected_calibration_probs, w_calibration)
    selected_threshold = thresholds["balanced"]
    validation_metrics = _score_predictions(y_calibration, selected_calibration_probs, w_calibration, selected_threshold)
    train_metrics = _score_predictions(y_train_full, selected_train_probs, w_train_full, selected_threshold)
    test_metrics = _score_predictions(y_test, selected_test_probs, w_test, selected_threshold)

    calibration_table = _make_calibration_table(y_test, selected_test_probs, w_test)
    bootstrap_cis = _bootstrap_metric_intervals(y_test, selected_test_probs, w_test, selected_threshold, random_state=random_state)

    logistic_test_probs = _fit_logistic_baseline(x_train_full, y_train_full, w_train_full, x_test, feature_columns)
    dummy_test_probs = _fit_dummy_baseline(y_train_full, w_train_full, len(x_test))

    baseline_comparison = {
        "xgboost_selected": _evaluate_probability_set(y_test, selected_test_probs, w_test),
        "xgboost_uncalibrated": _evaluate_probability_set(y_test, test_raw_probs, w_test),
        "xgboost_sigmoid": _evaluate_probability_set(y_test, test_prob_sets["sigmoid"], w_test),
        "xgboost_isotonic": _evaluate_probability_set(y_test, test_prob_sets["isotonic"], w_test),
        "logistic_regression": _evaluate_probability_set(y_test, logistic_test_probs, w_test),
        "dummy_prior": _evaluate_probability_set(y_test, dummy_test_probs, w_test),
    }

    validation_predictions = x_calibration.loc[:, feature_columns].copy()
    validation_predictions["actual_distress_flag"] = y_calibration.to_numpy()
    validation_predictions["predicted_probability_raw"] = calibration_raw_probs
    validation_predictions["predicted_probability"] = selected_calibration_probs
    validation_predictions["weight"] = w_calibration.to_numpy()

    test_predictions = x_test.loc[:, feature_columns].copy()
    test_predictions["actual_distress_flag"] = y_test.to_numpy()
    test_predictions["predicted_probability_raw"] = test_raw_probs
    test_predictions["predicted_probability"] = selected_test_probs
    test_predictions["predicted_label"] = (selected_test_probs >= selected_threshold).astype(int)
    for mode, threshold in thresholds.items():
        test_predictions[f"predicted_label_{mode}"] = (selected_test_probs >= threshold).astype(int)
    test_predictions["weight"] = w_test.to_numpy()

    subgroup_diagnostics = _subgroup_diagnostics(
        frame=x_test.copy(),
        probabilities=selected_test_probs,
        truth=y_test,
        weights=w_test,
        threshold=selected_threshold,
    )

    metrics = {
        "selected_threshold": selected_threshold,
        "threshold_modes": thresholds,
        "selected_missing_strategy": best_strategy,
        "selected_feature_count": len(feature_columns),
        "selected_params": selected_params,
        "tuning": tuning_results.to_dict(orient="records"),
        "split_rows": {
            "development_rows": int(len(x_development)),
            "train_rows": int(len(x_train_full)),
            "calibration_rows": int(len(x_calibration)),
            "test_rows": int(len(x_test)),
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "bootstrap_confidence_intervals": bootstrap_cis,
        "calibration_report": calibration_report,
        "baseline_comparison": baseline_comparison,
        "train_validation_gap": {
            "roc_auc_gap": float(train_metrics["roc_auc"] - validation_metrics["roc_auc"]),
            "pr_auc_gap": float(train_metrics["pr_auc"] - validation_metrics["pr_auc"]),
            "brier_gap": float(train_metrics["weighted_brier"] - validation_metrics["weighted_brier"]),
        },
        "limitations": [
            "The official CRH universe uses raw HAP_10, but this PUMF exposes only grouped HAP_10C.",
            "Valid skip means off-path/not asked and should not be interpreted as a substantive no.",
            "Several work and receiver-detail variables carry structural missingness because of survey routing.",
            "This is a predictive risk-ranking model, not a causal model and not a clinical diagnosis.",
        ],
    }

    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": feature_columns,
        "threshold": selected_threshold,
        "threshold_modes": thresholds,
        "selected_params": selected_params,
        "missing_strategy": best_strategy,
        "sparse_threshold": SPARSE_MISSINGNESS_THRESHOLD,
        "calibrator_name": selected_calibration,
        "sigmoid_calibrator": sigmoid_calibrator,
        "isotonic_calibrator": isotonic_calibrator,
        "feature_names_out": preprocessor.get_feature_names_out().tolist(),
        "xgboost_version": xgb.__version__,
    }

    return TrainingResult(
        artifact=artifact,
        metrics=metrics,
        tuning_results=tuning_results,
        threshold_curve=threshold_curve,
        calibration_table=calibration_table,
        test_predictions=test_predictions,
        validation_predictions=validation_predictions,
        threshold_analysis=threshold_curve,
        calibration_report=calibration_report,
        feature_audit=feature_audit,
        missingness_report=missingness_report,
        fold_metrics=fold_metrics,
        subgroup_diagnostics=subgroup_diagnostics,
    )
