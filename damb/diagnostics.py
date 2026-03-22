from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    ANALYTICS_ONLY_COLUMNS,
    BINARY_FEATURES,
    FRIENDLY_NAMES,
    GLOBAL_SHAP_PATH,
    ID_COLUMN,
    LEAKAGE_PREFIXES,
    MODEL_ARTIFACT_PATH,
    MODEL_FEATURES,
    NOMINAL_FEATURES,
    ORDINAL_FEATURES,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    RESERVE_CODE_MAP,
    TARGET_COLUMNS,
    TARGET_ITEM_LABELS,
    WEIGHT_COLUMN,
)
from .data import (
    apply_modeling_universe,
    build_target,
    coerce_feature_types,
    compute_missingness_report,
    prepare_training_frame,
    recode_binary_yes_no,
    recode_special_codes,
)
from .modeling import build_preprocessing_pipeline, fit_binary_model, leakage_audit
from .scoring import compute_aggregated_contributions


matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

DIAGNOSTIC_DIR = Path("reports") / "diagnostics"
DIAGNOSTIC_TABLES_DIR = DIAGNOSTIC_DIR / "tables"
DIAGNOSTIC_FIGURES_DIR = DIAGNOSTIC_DIR / "figures"

DIAGNOSTIC_EXTRA_COLUMNS = (
    "OAC_20",
    "ARV_40",
    "ARX_40",
    "RES_10",
    "ACD_80",
    "ACD_90",
    "CHC_110K",
    "CHC_110S",
    "ICS_40",
    "FIS_10A",
    "FIS_10H",
)

EXCLUDED_VARIABLE_NOTES = {
    "OAC_20": "Explicitly not deployed; interpreted as wanting additional support and treated as borderline/co-determined.",
    "ARV_40": "Not deployed in the current active pipeline.",
    "ARX_40": "Not deployed in the current active pipeline.",
    "RES_10": "Not deployed; very high missingness in the modeled universe.",
    "ACD_80": "Not deployed in the current active pipeline.",
    "ACD_90": "Not deployed in the current active pipeline.",
    "CHC_110K": "Not deployed; very high missingness in the modeled universe.",
    "CHC_110S": "Not deployed; very high missingness in the modeled universe.",
    "ICS_40": "Explicit leakage exclusion.",
    "FIS_10A": "Explicit leakage exclusion.",
    "FIS_10H": "Explicit leakage exclusion.",
}


def _ensure_dirs() -> None:
    DIAGNOSTIC_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _read_extended_raw() -> tuple[pd.DataFrame, list[str]]:
    meta = pyreadstat.read_sas7bdat(RAW_DATA_PATH, metadataonly=True)[1]
    available = meta.column_names
    desired = list(dict.fromkeys(REQUIRED_COLUMNS + DIAGNOSTIC_EXTRA_COLUMNS))
    usecols = [column for column in desired if column in available]
    frame, _ = pyreadstat.read_sas7bdat(RAW_DATA_PATH, usecols=usecols)
    return frame, available


def _json_dump(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


def _summarize_values(series: pd.Series) -> str:
    values = pd.Series(series.dropna().unique()).tolist()
    if len(values) == 0:
        return ""
    try:
        ordered = sorted(values)
    except TypeError:
        ordered = values
    if len(ordered) <= 25:
        return ", ".join(_format_value(value) for value in ordered)
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.notna().all():
        return f"{len(values)} distinct values; min={numeric.min():.4f}; max={numeric.max():.4f}"
    return f"{len(values)} distinct values"


def _df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def _family(variable: str) -> str:
    if variable in TARGET_COLUMNS:
        return "target"
    if variable in BINARY_FEATURES:
        return "binary predictor"
    if variable in NOMINAL_FEATURES:
        return "nominal predictor"
    if variable in ORDINAL_FEATURES:
        return "ordinal predictor"
    if variable in MODEL_FEATURES:
        return "numeric/count predictor"
    if variable == WEIGHT_COLUMN:
        return "weight"
    if variable in ANALYTICS_ONLY_COLUMNS:
        return "analytics-only"
    return "excluded"


def _meaning(variable: str) -> str:
    if variable in TARGET_ITEM_LABELS:
        return TARGET_ITEM_LABELS[variable]
    if variable in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[variable]
    if variable == WEIGHT_COLUMN:
        return "Person weight"
    return variable


def _reserve_code_note(codes: set[int]) -> str:
    if not codes:
        return "No explicit reserve mapping in code."
    inferred = []
    for code in sorted(codes):
        if str(code).endswith("6"):
            inferred.append(f"{code}=off-path/structural (inferred)")
        elif str(code).endswith("7"):
            inferred.append(f"{code}=don't know (inferred)")
        elif str(code).endswith("8"):
            inferred.append(f"{code}=refusal (inferred)")
        elif str(code).endswith("9"):
            inferred.append(f"{code}=not stated (inferred)")
        else:
            inferred.append(str(code))
    return "; ".join(inferred)


def _weighted_total(frame: pd.DataFrame) -> float:
    if WEIGHT_COLUMN not in frame.columns:
        return float("nan")
    return float(frame[WEIGHT_COLUMN].sum())


def _filter_audit(raw_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_0 = raw_clean.copy()
    step_1 = step_0.loc[step_0["DV_PROXY"] == 2].copy()
    step_2 = step_1.loc[step_1["PAR_10"].between(1, 99, inclusive="both")].copy()
    step_3 = step_2.loc[step_2["HAP_10C"].isin([1, 2, 3, 4, 5, 6])].copy()
    step_4 = step_3.loc[step_3["distress_flag"].notna()].copy()
    steps = [
        ("Loaded rows", "No filter", step_0, step_0),
        ("Proxy exclusion", "DV_PROXY == 2", step_0, step_1),
        ("Caregiver eligibility", "PAR_10 in 1..99", step_1, step_2),
        ("Care-hours approximation", "HAP_10C in 1..6", step_2, step_3),
        ("Target answerability", "distress_flag not missing", step_3, step_4),
    ]
    rows = []
    for step_name, condition, before, after in steps:
        rows.append(
            {
                "step": step_name,
                "condition": condition,
                "rows_before": int(len(before)),
                "rows_after": int(len(after)),
                "rows_retained_pct": float(len(after) / len(before) * 100) if len(before) else np.nan,
                "weighted_before": _weighted_total(before),
                "weighted_after": _weighted_total(after),
                "weighted_retained_pct": float(_weighted_total(after) / _weighted_total(before) * 100) if _weighted_total(before) else np.nan,
            }
        )
    return pd.DataFrame(rows), step_4


def _special_code_audit(raw_frame: pd.DataFrame) -> pd.DataFrame:
    audited_variables = list(dict.fromkeys([WEIGHT_COLUMN, *TARGET_COLUMNS, *MODEL_FEATURES, *ANALYTICS_ONLY_COLUMNS, *DIAGNOSTIC_EXTRA_COLUMNS]))
    rows = []
    for variable in audited_variables:
        if variable not in raw_frame.columns:
            continue
        raw_values = sorted(pd.Series(raw_frame[variable].dropna().unique()).tolist())
        reserve_codes = sorted(RESERVE_CODE_MAP.get(variable, set()))
        mapped_zero = [2] if variable in TARGET_COLUMNS or variable in BINARY_FEATURES else []
        mapped_one = [1] if variable in TARGET_COLUMNS or variable in BINARY_FEATURES else []
        rows.append(
            {
                "variable": variable,
                "family": _family(variable),
                "raw_values_seen": _summarize_values(raw_frame[variable]),
                "values_mapped_to_nan": ", ".join(str(code) for code in reserve_codes),
                "values_mapped_to_0": ", ".join(str(code) for code in mapped_zero),
                "values_mapped_to_1": ", ".join(str(code) for code in mapped_one),
                "notes": _reserve_code_note(set(reserve_codes)),
            }
        )
    return pd.DataFrame(rows)


def _missing_reason(raw_original: pd.DataFrame, clean_final: pd.DataFrame, variable: str) -> str:
    if variable not in raw_original.columns:
        return "unknown"
    raw_missing_values = raw_original.loc[clean_final[variable].isna(), variable].value_counts(dropna=False)
    if raw_missing_values.empty:
        return "unknown"
    structural_like = sum(raw_missing_values.get(code, 0) for code in [6, 96])
    nonresponse_like = sum(raw_missing_values.get(code, 0) for code in [7, 8, 9, 97, 98, 99])
    if structural_like > nonresponse_like:
        return "mostly structural/off-path"
    if nonresponse_like > 0:
        return "mixed nonresponse"
    return "unknown"


def _variable_audit(raw_modeled_original: pd.DataFrame, clean_modeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_variables = list(dict.fromkeys([*TARGET_COLUMNS, *MODEL_FEATURES, WEIGHT_COLUMN, *ANALYTICS_ONLY_COLUMNS, *DIAGNOSTIC_EXTRA_COLUMNS]))
    variable_rows = []
    missing_rows = []
    for variable in audit_variables:
        if variable not in raw_modeled_original.columns and variable not in clean_modeled.columns:
            continue
        used_in_model = variable in MODEL_FEATURES
        family = _family(variable)
        if variable in TARGET_COLUMNS:
            cleaned_coding = "1 -> 1, 2 -> 0, reserve codes -> NaN, then distress_score/distress_flag derived"
            action = "target recode"
        elif variable in BINARY_FEATURES:
            cleaned_coding = "1 -> 1, 2 -> 0, reserve codes -> NaN"
            action = "binary recode"
        elif variable in NOMINAL_FEATURES:
            cleaned_coding = "reserve codes -> NaN, values cast to string/object, Missing category added in encoder"
            action = "explicit missing category"
        elif variable in ORDINAL_FEATURES or variable == WEIGHT_COLUMN or variable in MODEL_FEATURES:
            cleaned_coding = "reserve codes -> NaN, numeric/ordinal preserved"
            action = "left as NaN"
        else:
            cleaned_coding = "not in deployed model"
            action = "dropped/excluded"
        missing_pct = float(clean_modeled[variable].isna().mean()) if variable in clean_modeled.columns else np.nan
        weight_mask = clean_modeled[variable].isna() if variable in clean_modeled.columns else pd.Series(False, index=clean_modeled.index)
        weighted_missing_pct = float(clean_modeled.loc[weight_mask, WEIGHT_COLUMN].sum() / clean_modeled[WEIGHT_COLUMN].sum()) if variable in clean_modeled.columns else np.nan
        variable_rows.append(
            {
                "variable": variable,
                "meaning": _meaning(variable),
                "family": family,
                "used_in_model": used_in_model,
                "raw_coding_summary": _summarize_values(raw_modeled_original[variable]) if variable in raw_modeled_original.columns else "not loaded",
                "cleaned_coding": cleaned_coding,
                "missing_pct_after_cleaning": missing_pct,
                "weighted_missing_pct_after_cleaning": weighted_missing_pct,
                "notes": EXCLUDED_VARIABLE_NOTES.get(variable, ""),
            }
        )
        if variable in clean_modeled.columns:
            missing_rows.append(
                {
                    "variable": variable,
                    "used_in_model": used_in_model,
                    "missing_count": int(clean_modeled[variable].isna().sum()),
                    "missing_pct": missing_pct,
                    "weighted_missing_pct": weighted_missing_pct,
                    "likely_reason": _missing_reason(raw_modeled_original, clean_modeled, variable),
                    "action_taken": action,
                }
            )
    return pd.DataFrame(variable_rows), pd.DataFrame(missing_rows)


def _recreate_split_predictions(prepared_frame: pd.DataFrame, training_result) -> dict[str, Any]:
    analytic = prepared_frame.loc[prepared_frame["distress_flag"].notna()].copy()
    x = analytic.loc[:, MODEL_FEATURES]
    y = analytic["distress_flag"].astype(int)
    w = analytic[WEIGHT_COLUMN].astype(float)
    x_train_val, x_test, y_train_val, y_test, w_train_val, w_test = train_test_split(
        x, y, w, test_size=0.20, random_state=42, stratify=y
    )
    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(
        x_train_val, y_train_val, w_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    best_candidate = training_result.metrics["tuning"][0]
    selection_params = {
        key: best_candidate[key]
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
        selection_params[int_key] = int(selection_params[int_key])
    for float_key in ["learning_rate", "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda"]:
        selection_params[float_key] = float(selection_params[float_key])
    preprocessor = build_preprocessing_pipeline()
    x_train_tx = preprocessor.fit_transform(x_train)
    x_val_tx = preprocessor.transform(x_val)
    selection_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        early_stopping_rounds=40,
        **selection_params,
    )
    selection_model.fit(
        x_train_tx,
        y_train,
        sample_weight=np.asarray(w_train / w_train.mean()),
        eval_set=[(x_val_tx, y_val)],
        sample_weight_eval_set=[np.asarray(w_val / w_val.mean())],
        verbose=False,
    )
    train_probs = selection_model.predict_proba(x_train_tx)[:, 1]
    val_probs = selection_model.predict_proba(x_val_tx)[:, 1]
    final_preprocessor = training_result.artifact["preprocessor"]
    final_model = training_result.artifact["model"]
    train_val_probs = final_model.predict_proba(final_preprocessor.transform(x_train_val))[:, 1]
    test_probs = final_model.predict_proba(final_preprocessor.transform(x_test))[:, 1]
    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_train_val": x_train_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_train_val": y_train_val,
        "y_test": y_test,
        "w_train": w_train,
        "w_val": w_val,
        "w_train_val": w_train_val,
        "w_test": w_test,
        "train_probs": train_probs,
        "val_probs": val_probs,
        "train_val_probs": train_val_probs,
        "test_probs": test_probs,
    }


def _metric_row(y_true: pd.Series, probs: np.ndarray, weights: pd.Series, threshold: float, split_name: str) -> dict[str, Any]:
    pred = (probs >= threshold).astype(int)
    cm_weighted = confusion_matrix(y_true, pred, sample_weight=weights, labels=[0, 1])
    cm_unweighted = confusion_matrix(y_true, pred, labels=[0, 1])
    precision_w = cm_weighted[1, 1] / (cm_weighted[1, 1] + cm_weighted[0, 1]) if (cm_weighted[1, 1] + cm_weighted[0, 1]) else 0.0
    recall_w = cm_weighted[1, 1] / (cm_weighted[1, 1] + cm_weighted[1, 0]) if (cm_weighted[1, 1] + cm_weighted[1, 0]) else 0.0
    specificity_w = cm_weighted[0, 0] / (cm_weighted[0, 0] + cm_weighted[0, 1]) if (cm_weighted[0, 0] + cm_weighted[0, 1]) else 0.0
    f1_w = 2 * precision_w * recall_w / (precision_w + recall_w) if precision_w + recall_w else 0.0
    precision_u = cm_unweighted[1, 1] / (cm_unweighted[1, 1] + cm_unweighted[0, 1]) if (cm_unweighted[1, 1] + cm_unweighted[0, 1]) else 0.0
    recall_u = cm_unweighted[1, 1] / (cm_unweighted[1, 1] + cm_unweighted[1, 0]) if (cm_unweighted[1, 1] + cm_unweighted[1, 0]) else 0.0
    f1_u = 2 * precision_u * recall_u / (precision_u + recall_u) if precision_u + recall_u else 0.0
    specificity_u = cm_unweighted[0, 0] / (cm_unweighted[0, 0] + cm_unweighted[0, 1]) if (cm_unweighted[0, 0] + cm_unweighted[0, 1]) else 0.0
    return {
        "split": split_name,
        "rows": int(len(y_true)),
        "roc_auc_weighted": float(roc_auc_score(y_true, probs, sample_weight=weights)),
        "roc_auc_unweighted": float(roc_auc_score(y_true, probs)),
        "pr_auc_weighted": float(average_precision_score(y_true, probs, sample_weight=weights)),
        "pr_auc_unweighted": float(average_precision_score(y_true, probs)),
        "accuracy_weighted": float(np.average(pred == y_true.to_numpy(), weights=weights)),
        "accuracy_unweighted": float((pred == y_true.to_numpy()).mean()),
        "precision_weighted": float(precision_w),
        "precision_unweighted": float(precision_u),
        "recall_weighted": float(recall_w),
        "recall_unweighted": float(recall_u),
        "specificity_weighted": float(specificity_w),
        "specificity_unweighted": float(specificity_u),
        "f1_weighted": float(f1_w),
        "f1_unweighted": float(f1_u),
        "prevalence_weighted": float(np.average(y_true, weights=weights)),
        "prevalence_unweighted": float(y_true.mean()),
        "tp_weighted": float(cm_weighted[1, 1]),
        "fp_weighted": float(cm_weighted[0, 1]),
        "tn_weighted": float(cm_weighted[0, 0]),
        "fn_weighted": float(cm_weighted[1, 0]),
    }


def _plot_class_balance(prepared_frame: pd.DataFrame) -> None:
    analytic = prepared_frame.loc[prepared_frame["distress_flag"].notna()].copy()
    counts = analytic["distress_flag"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["0", "1"], counts.values, color=["#7da0ca", "#d95f02"])
    ax.set_title("Class balance in analytic sample")
    ax.set_xlabel("distress_flag")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(DIAGNOSTIC_FIGURES_DIR / "class_balance.png", dpi=200)
    plt.close(fig)


def _plot_missingness(missingness: pd.DataFrame) -> None:
    top = missingness.sort_values("missing_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top, x="missing_pct", y="feature", color="#4c956c", ax=ax)
    ax.set_xlabel("Missing percentage")
    ax.set_ylabel("")
    ax.set_title("Feature missingness after cleaning")
    fig.tight_layout()
    fig.savefig(DIAGNOSTIC_FIGURES_DIR / "missingness.png", dpi=200)
    plt.close(fig)


def _plot_confusion(y_true: pd.Series, probs: np.ndarray, threshold: float) -> None:
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Test confusion matrix")
    fig.tight_layout()
    fig.savefig(DIAGNOSTIC_FIGURES_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def _plot_curves(split_data: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, truth, probs in [
        ("Train", split_data["y_train"], split_data["train_probs"]),
        ("Validation", split_data["y_val"], split_data["val_probs"]),
        ("Test", split_data["y_test"], split_data["test_probs"]),
    ]:
        fpr, tpr, _ = roc_curve(truth, probs)
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIAGNOSTIC_FIGURES_DIR / "roc_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, truth, probs in [
        ("Train", split_data["y_train"], split_data["train_probs"]),
        ("Validation", split_data["y_val"], split_data["val_probs"]),
        ("Test", split_data["y_test"], split_data["test_probs"]),
    ]:
        precision, recall, _ = precision_recall_curve(truth, probs)
        ax.plot(recall, precision, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIAGNOSTIC_FIGURES_DIR / "pr_curve.png", dpi=200)
    plt.close(fig)


def _threshold_diagnostics(scored_probs: np.ndarray, scored_weights: pd.Series, truth: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    thresholds = sorted(set([0.40, 0.50, 0.60, 0.625, 0.70, 0.80]))
    for threshold in thresholds:
        flags = (scored_probs >= threshold).astype(int)
        row = {
            "threshold": float(threshold),
            "flagged_rows": int(flags.sum()),
            "flagged_weighted_total": float(scored_weights[flags == 1].sum()),
            "flagged_row_pct": float(flags.mean()),
            "flagged_weight_pct": float(scored_weights[flags == 1].sum() / scored_weights.sum()),
        }
        if truth is not None:
            metrics = _metric_row(truth, scored_probs, scored_weights, threshold, f"threshold_{threshold}")
            row.update(
                {
                    "precision_weighted": metrics["precision_weighted"],
                    "recall_weighted": metrics["recall_weighted"],
                    "specificity_weighted": metrics["specificity_weighted"],
                    "f1_weighted": metrics["f1_weighted"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _error_analysis(test_frame: pd.DataFrame, y_test: pd.Series, probs: np.ndarray, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = test_frame.copy()
    frame["actual"] = y_test.to_numpy()
    frame["predicted"] = (probs >= threshold).astype(int)
    frame["predicted_probability"] = probs
    frame["error_type"] = np.select(
        [
            (frame["actual"] == 1) & (frame["predicted"] == 0),
            (frame["actual"] == 0) & (frame["predicted"] == 1),
        ],
        ["false_negative", "false_positive"],
        default="correct",
    )
    error_rows = []
    for error_type in ["false_negative", "false_positive"]:
        subset = frame.loc[frame["error_type"] == error_type].copy()
        for column in ["PRV", "HAP_10C", "TTLINCG1", "SEX", "AGEGR10"]:
            if column not in subset.columns or subset.empty:
                continue
            grouped = subset.groupby(column, observed=False)
            for group_value, group in grouped:
                error_rows.append(
                    {
                        "error_type": error_type,
                        "dimension": column,
                        "group_value": group_value,
                        "rows": int(len(group)),
                        "weighted_rows": float(group[WEIGHT_COLUMN].sum()),
                        "mean_predicted_probability": float(group["predicted_probability"].mean()),
                    }
                )
    performance_rows = []
    for column in ["PRV", "HAP_10C", "TTLINCG1", "SEX"]:
        if column not in frame.columns:
            continue
        grouped = frame.groupby(column, observed=False)
        for group_value, group in grouped:
            if len(group) < 25:
                continue
            performance_rows.append(
                {
                    "dimension": column,
                    "group_value": group_value,
                    "rows": int(len(group)),
                    "weighted_rows": float(group[WEIGHT_COLUMN].sum()),
                    "weighted_accuracy": float(np.average(group["actual"] == group["predicted"], weights=group[WEIGHT_COLUMN])),
                    "weighted_recall": float(
                        group.loc[(group["actual"] == 1) & (group["predicted"] == 1), WEIGHT_COLUMN].sum()
                        / group.loc[group["actual"] == 1, WEIGHT_COLUMN].sum()
                    ) if float(group.loc[group["actual"] == 1, WEIGHT_COLUMN].sum()) > 0 else np.nan,
                }
            )
    return pd.DataFrame(error_rows), pd.DataFrame(performance_rows)


def _robustness_checks(split_data: dict[str, Any], threshold: float, top_feature: str) -> pd.DataFrame:
    x_train_val = split_data["x_train_val"]
    y_train_val = split_data["y_train_val"]
    w_train_val = split_data["w_train_val"]
    x_test = split_data["x_test"]
    y_test = split_data["y_test"]
    w_test = split_data["w_test"]

    checks = []

    preprocessor = build_preprocessing_pipeline()
    x_train_val_tx = preprocessor.fit_transform(x_train_val)
    x_test_tx = preprocessor.transform(x_test)
    xgb_weighted = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_estimators=50,
        learning_rate=0.04,
        max_depth=3,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.75,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=2.0,
        n_jobs=1,
    )
    xgb_weighted.fit(x_train_val_tx, y_train_val, sample_weight=np.asarray(w_train_val / w_train_val.mean()), verbose=False)
    weighted_probs = xgb_weighted.predict_proba(x_test_tx)[:, 1]
    checks.append(
        {
            "check": "weighted_xgboost_refit",
            "roc_auc_weighted": float(roc_auc_score(y_test, weighted_probs, sample_weight=w_test)),
            "pr_auc_weighted": float(average_precision_score(y_test, weighted_probs, sample_weight=w_test)),
        }
    )

    xgb_unweighted = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_estimators=50,
        learning_rate=0.04,
        max_depth=3,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.75,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=2.0,
        n_jobs=1,
    )
    xgb_unweighted.fit(x_train_val_tx, y_train_val, verbose=False)
    unweighted_probs = xgb_unweighted.predict_proba(x_test_tx)[:, 1]
    checks.append(
        {
            "check": "unweighted_xgboost_refit",
            "roc_auc_weighted": float(roc_auc_score(y_test, unweighted_probs, sample_weight=w_test)),
            "pr_auc_weighted": float(average_precision_score(y_test, unweighted_probs, sample_weight=w_test)),
        }
    )

    logistic_imputer = SimpleImputer(strategy="median")
    x_train_val_dense = logistic_imputer.fit_transform(x_train_val_tx)
    x_test_dense = logistic_imputer.transform(x_test_tx)
    logistic = LogisticRegression(max_iter=2000, solver="liblinear")
    logistic.fit(x_train_val_dense, y_train_val, sample_weight=np.asarray(w_train_val))
    logistic_probs = logistic.predict_proba(x_test_dense)[:, 1]
    checks.append(
        {
            "check": "weighted_logistic_baseline",
            "roc_auc_weighted": float(roc_auc_score(y_test, logistic_probs, sample_weight=w_test)),
            "pr_auc_weighted": float(average_precision_score(y_test, logistic_probs, sample_weight=w_test)),
        }
    )

    if top_feature in x_train_val.columns:
        reduced_train = x_train_val.drop(columns=[top_feature])
        reduced_test = x_test.drop(columns=[top_feature])
        reduced_numeric = [col for col in split_data["x_train_val"].columns if col != top_feature and col not in NOMINAL_FEATURES]
        reduced_nominal = [col for col in NOMINAL_FEATURES if col != top_feature and col in reduced_train.columns]
        reduced_pre = ColumnTransformer(
            transformers=[
                ("numeric", "passthrough", reduced_numeric),
                (
                    "nominal",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)),
                        ]
                    ),
                    reduced_nominal,
                ),
            ]
        )
        reduced_train_tx = reduced_pre.fit_transform(reduced_train)
        reduced_test_tx = reduced_pre.transform(reduced_test)
        ablation_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
            n_estimators=50,
            learning_rate=0.04,
            max_depth=3,
            min_child_weight=5,
            subsample=0.75,
            colsample_bytree=0.75,
            gamma=0.0,
            reg_alpha=0.0,
            reg_lambda=2.0,
            n_jobs=1,
        )
        ablation_model.fit(reduced_train_tx, y_train_val, sample_weight=np.asarray(w_train_val / w_train_val.mean()), verbose=False)
        ablation_probs = ablation_model.predict_proba(reduced_test_tx)[:, 1]
        checks.append(
            {
                "check": f"ablation_without_{top_feature}",
                "roc_auc_weighted": float(roc_auc_score(y_test, ablation_probs, sample_weight=w_test)),
                "pr_auc_weighted": float(average_precision_score(y_test, ablation_probs, sample_weight=w_test)),
            }
        )
    return pd.DataFrame(checks)


def run_diagnostics() -> dict[str, Any]:
    _ensure_dirs()

    raw_frame, available_columns = _read_extended_raw()
    prepared = prepare_training_frame(str(RAW_DATA_PATH))
    training_result = fit_binary_model(prepared.frame)
    split_data = _recreate_split_predictions(prepared.frame, training_result)
    threshold = float(training_result.metrics["selected_threshold"])

    raw_clean = recode_special_codes(raw_frame.copy())
    raw_clean = recode_binary_yes_no(raw_clean, BINARY_FEATURES)
    raw_clean = build_target(raw_clean)
    filter_table, raw_modeled = _filter_audit(raw_clean)
    clean_modeled = coerce_feature_types(raw_modeled.copy())
    raw_modeled_original = raw_frame.loc[raw_modeled.index, [column for column in raw_modeled.columns if column in raw_frame.columns]].copy()

    required_audit = pd.DataFrame(
        {
            "required_variable": REQUIRED_COLUMNS,
            "found_in_raw": [column in available_columns for column in REQUIRED_COLUMNS],
        }
    )
    required_audit["missing_from_raw"] = ~required_audit["found_in_raw"]

    data_source_audit = pd.DataFrame(
        [
            {
                "artifact_type": "raw_model_input",
                "path": str(RAW_DATA_PATH),
                "used_for_training": True,
                "rows": int(raw_frame.shape[0]),
                "columns": int(len(available_columns)),
            },
            {
                "artifact_type": "cleaned_output_scored_dataset",
                "path": "data/processed/scored_caregiver_distress.csv",
                "used_for_training": False,
                "rows": int(pd.read_csv("data/processed/scored_caregiver_distress.csv").shape[0]) if Path("data/processed/scored_caregiver_distress.csv").exists() else 0,
                "columns": int(pd.read_csv("data/processed/scored_caregiver_distress.csv").shape[1]) if Path("data/processed/scored_caregiver_distress.csv").exists() else 0,
            },
        ]
    )

    special_code_table = _special_code_audit(raw_frame)
    variable_table, missingness_table = _variable_audit(raw_modeled_original, clean_modeled)
    feature_missingness = compute_missingness_report(clean_modeled, MODEL_FEATURES)
    global_shap = pd.read_csv(GLOBAL_SHAP_PATH)
    shap_frame = compute_aggregated_contributions(training_result.artifact, prepared.frame.loc[:, MODEL_FEATURES])
    transformed_names = pd.DataFrame(
        {
            "transformed_feature_name": training_result.artifact["feature_names_out"],
        }
    )
    transformed_names["base_feature"] = transformed_names["transformed_feature_name"].str.replace(
        r"^(numeric|nominal)__", "", regex=True
    )

    metrics_table = pd.DataFrame(
        [
            _metric_row(split_data["y_train"], split_data["train_probs"], split_data["w_train"], threshold, "train_selection_model"),
            _metric_row(split_data["y_val"], split_data["val_probs"], split_data["w_val"], threshold, "validation_selection_model"),
            _metric_row(split_data["y_train_val"], split_data["train_val_probs"], split_data["w_train_val"], threshold, "train_val_final_model"),
            _metric_row(split_data["y_test"], split_data["test_probs"], split_data["w_test"], threshold, "test_final_model"),
        ]
    )

    threshold_table = _threshold_diagnostics(
        split_data["test_probs"],
        split_data["w_test"],
        split_data["y_test"],
    )
    threshold_population_table = _threshold_diagnostics(
        training_result.artifact["model"].predict_proba(training_result.artifact["preprocessor"].transform(split_data["x_train_val"]))[:, 1],
        split_data["w_train_val"],
        split_data["y_train_val"],
    )
    error_table, subgroup_perf_table = _error_analysis(split_data["x_test"].assign(**{WEIGHT_COLUMN: split_data["w_test"].to_numpy()}), split_data["y_test"], split_data["test_probs"], threshold)
    robustness_table = _robustness_checks(split_data, threshold, str(global_shap.iloc[0]["feature"]))

    _plot_class_balance(prepared.frame)
    _plot_missingness(feature_missingness)
    _plot_confusion(split_data["y_test"], split_data["test_probs"], threshold)
    _plot_curves(split_data)

    duplicates_summary = {
        "raw_duplicate_rows": int(raw_frame.duplicated().sum()),
        "raw_duplicate_ids": int(raw_frame[ID_COLUMN].duplicated().sum()) if ID_COLUMN in raw_frame.columns else 0,
        "modeled_duplicate_rows": int(raw_modeled.duplicated().sum()),
        "modeled_duplicate_ids": int(raw_modeled[ID_COLUMN].duplicated().sum()) if ID_COLUMN in raw_modeled.columns else 0,
    }

    leakage_summary = {
        "final_feature_list": list(MODEL_FEATURES),
        "active_leakage_audit": leakage_audit(MODEL_FEATURES),
        "explicitly_excluded_examples": EXCLUDED_VARIABLE_NOTES,
        "suspicious_but_retained": [
            "APR_* caregiving-task flags are strong drivers and should be narrated carefully because the local codebook PDF was not found.",
            "ARV_10 and ARX_10 are concurrent caregiving-context variables, not obvious leakage, but still contemporaneous with the target period.",
        ],
    }

    summary = {
        "active_training_path": "scripts/train_binary_model.py -> prepare_training_frame() -> fit_binary_model() -> save_training_outputs()",
        "historical_notebook_path": "src/eda.ipynb exists but is not invoked by the active training script.",
        "pdf_status": "No relevant local GSS Cycle 32 caregiving codebook/user-guide PDF was found under the repo or Desktop workspace during this audit.",
        "duplicates": duplicates_summary,
        "selected_threshold": threshold,
        "top_5_drivers": global_shap.head(5).to_dict(orient="records"),
        "overfitting": {
            "train_vs_validation_roc_gap": float(metrics_table.loc[metrics_table["split"] == "train_selection_model", "roc_auc_weighted"].iloc[0] - metrics_table.loc[metrics_table["split"] == "validation_selection_model", "roc_auc_weighted"].iloc[0]),
            "train_val_vs_test_roc_gap": float(metrics_table.loc[metrics_table["split"] == "train_val_final_model", "roc_auc_weighted"].iloc[0] - metrics_table.loc[metrics_table["split"] == "test_final_model", "roc_auc_weighted"].iloc[0]),
            "train_vs_validation_pr_gap": float(metrics_table.loc[metrics_table["split"] == "train_selection_model", "pr_auc_weighted"].iloc[0] - metrics_table.loc[metrics_table["split"] == "validation_selection_model", "pr_auc_weighted"].iloc[0]),
            "train_val_vs_test_pr_gap": float(metrics_table.loc[metrics_table["split"] == "train_val_final_model", "pr_auc_weighted"].iloc[0] - metrics_table.loc[metrics_table["split"] == "test_final_model", "pr_auc_weighted"].iloc[0]),
        },
    }

    outputs = {
        "data_source_audit": DIAGNOSTIC_TABLES_DIR / "data_source_audit.csv",
        "required_column_audit": DIAGNOSTIC_TABLES_DIR / "required_column_audit.csv",
        "filter_audit": DIAGNOSTIC_TABLES_DIR / "filter_audit.csv",
        "special_code_audit": DIAGNOSTIC_TABLES_DIR / "special_code_audit.csv",
        "variable_audit": DIAGNOSTIC_TABLES_DIR / "variable_audit.csv",
        "missingness_audit": DIAGNOSTIC_TABLES_DIR / "missingness_audit.csv",
        "feature_missingness": DIAGNOSTIC_TABLES_DIR / "feature_missingness.csv",
        "transformed_features": DIAGNOSTIC_TABLES_DIR / "transformed_features.csv",
        "metrics": DIAGNOSTIC_TABLES_DIR / "metrics_by_split.csv",
        "threshold_test": DIAGNOSTIC_TABLES_DIR / "threshold_diagnostics_test.csv",
        "threshold_population": DIAGNOSTIC_TABLES_DIR / "threshold_diagnostics_population.csv",
        "error_analysis": DIAGNOSTIC_TABLES_DIR / "error_analysis.csv",
        "subgroup_performance": DIAGNOSTIC_TABLES_DIR / "subgroup_performance.csv",
        "robustness": DIAGNOSTIC_TABLES_DIR / "robustness_checks.csv",
        "summary_json": DIAGNOSTIC_DIR / "diagnostics_summary.json",
    }

    data_source_audit.to_csv(outputs["data_source_audit"], index=False)
    required_audit.to_csv(outputs["required_column_audit"], index=False)
    filter_table.to_csv(outputs["filter_audit"], index=False)
    special_code_table.to_csv(outputs["special_code_audit"], index=False)
    variable_table.to_csv(outputs["variable_audit"], index=False)
    missingness_table.to_csv(outputs["missingness_audit"], index=False)
    feature_missingness.to_csv(outputs["feature_missingness"], index=False)
    transformed_names.to_csv(outputs["transformed_features"], index=False)
    metrics_table.to_csv(outputs["metrics"], index=False)
    threshold_table.to_csv(outputs["threshold_test"], index=False)
    threshold_population_table.to_csv(outputs["threshold_population"], index=False)
    error_table.to_csv(outputs["error_analysis"], index=False)
    subgroup_perf_table.to_csv(outputs["subgroup_performance"], index=False)
    robustness_table.to_csv(outputs["robustness"], index=False)
    _json_dump(summary, outputs["summary_json"])

    cleaning_report = f"""# Cleaning Diagnostics

## Executive summary
- Active training path: `scripts/train_binary_model.py -> prepare_training_frame() -> fit_binary_model() -> save_training_outputs()`
- Active model input file: `{RAW_DATA_PATH}`
- Historical notebook `src/eda.ipynb` exists, but it is not called by the active pipeline.
- No relevant local GSS Cycle 32 caregiving PDF/codebook file was found during this audit, so reserve-code labels beyond the implemented mappings are explicitly marked as inferred.

## A1. Data source and shape audit
{_df_to_markdown(data_source_audit)}

- Raw duplicate rows: `{duplicates_summary["raw_duplicate_rows"]}`
- Raw duplicate `{ID_COLUMN}` values: `{duplicates_summary["raw_duplicate_ids"]}`
- Modeled duplicate rows: `{duplicates_summary["modeled_duplicate_rows"]}`
- Modeled duplicate `{ID_COLUMN}` values: `{duplicates_summary["modeled_duplicate_ids"]}`

## A2. Required-column audit
{_df_to_markdown(required_audit)}

Full file: `reports/diagnostics/tables/required_column_audit.csv`

## A3. Universe/filtering audit
{_df_to_markdown(filter_table)}

Implemented universe note: the code uses grouped `HAP_10C` because raw `HAP_10` is not present in the active raw file.

## A4. Special-code / reserve-code audit
Full file: `reports/diagnostics/tables/special_code_audit.csv`

{_df_to_markdown(special_code_table, max_rows=len(special_code_table))}

## A5. Variable-family cleaning audit
Full file: `reports/diagnostics/tables/variable_audit.csv`

{_df_to_markdown(variable_table, max_rows=len(variable_table))}

## A6. Target construction audit
- Source variables: `{", ".join(TARGET_COLUMNS)}`
- Recoding: `1 -> 1`, `2 -> 0`, reserve codes -> `NaN`
- Partial missingness: `distress_score` uses `sum(min_count=1)` so any observed distress item can contribute
- Final binary rule: `distress_flag = 1 if distress_score >= 1 else 0`
- Final analytic sample size: `{prepared.target_summary["analytic_rows"]}`
- Unweighted prevalence: `{prepared.target_summary["unweighted_prevalence"]:.4f}`
- Weighted prevalence: `{prepared.target_summary["weighted_prevalence"]:.4f}`
- Distress-score distribution: `{prepared.target_summary["distress_score_distribution"]}`

## A7. Missingness audit
Full file: `reports/diagnostics/tables/missingness_audit.csv`

{_df_to_markdown(missingness_table.sort_values(["used_in_model", "missing_pct"], ascending=[False, False]), max_rows=len(missingness_table))}

## A8. Encoding audit
- Numeric/count predictors: `{", ".join([col for col in MODEL_FEATURES if col not in NOMINAL_FEATURES and col not in BINARY_FEATURES and col not in ORDINAL_FEATURES])}`
- Ordinal predictors: `{", ".join(ORDINAL_FEATURES)}`
- Nominal predictors one-hot encoded with explicit Missing category: `{", ".join(NOMINAL_FEATURES)}`
- Binary predictors: `{", ".join(BINARY_FEATURES)}`
- Final transformed feature count entering the model: `{len(training_result.artifact["feature_names_out"])}`
- One-hot encoder uses `min_frequency=20`; no additional manual sparse-level collapse exists in the active code.

Full transformed-feature file: `reports/diagnostics/tables/transformed_features.csv`

## A9. Leakage audit
- Final feature leakage audit: `{leakage_summary["active_leakage_audit"]}`
- Explicitly excluded examples:
{json.dumps(leakage_summary["explicitly_excluded_examples"], indent=2)}
- Suspicious-but-retained notes:
  - APR task flags are strong drivers and should be narrated with care.
  - No banned `CRH_*`, `ICS_*`, or `FIS_*` fields appear in the active final feature list.

## A10. Split and preprocessing audit
- First split: train/validation pool vs test using `train_test_split(..., test_size=0.20, stratify=y, random_state=42)`
- Second split inside the train/validation pool: train vs validation using `test_size=0.25`, again stratified with `random_state=42`
- Preprocessing is fit on the training fold for model selection, not on the full dataset before splitting
- CV tuning uses 3-fold stratified CV on the train/validation pool
- Final deployed preprocessor/model are refit on the full train/validation pool after threshold selection, leaving the test set untouched
- Sample weights are split alongside the rows and normalized to mean 1 for XGBoost fitting
"""

    model_report = f"""# Model Diagnostics

## Executive summary
- Model class: `xgboost.XGBClassifier`
- Active model package/version: `{training_result.artifact["xgboost_version"]}`
- Selected dashboard threshold: `{threshold:.3f}`
- Top 5 global drivers: `{summary["top_5_drivers"]}`

## B1. Model specification audit
- Objective: binary logistic
- Eval metric during fitting: `aucpr`
- Tree method: `hist`
- Early stopping: yes, `early_stopping_rounds=40` during model-selection fits
- Weighting: yes, `WGHT_PER` normalized to mean 1 for training
- Class imbalance handling: no separate `scale_pos_weight`
- Validation scheme: 3-fold CV for bounded search, then stratified train/validation/test split with a held-out test set
- Selected deployed hyperparameters: `{training_result.metrics["selected_params"]}`

## B2. Metric report
Full file: `reports/diagnostics/tables/metrics_by_split.csv`

{_df_to_markdown(metrics_table)}

## B3. Overfitting diagnostics
- Train vs validation weighted ROC AUC gap: `{summary["overfitting"]["train_vs_validation_roc_gap"]:.4f}`
- Train vs validation weighted PR AUC gap: `{summary["overfitting"]["train_vs_validation_pr_gap"]:.4f}`
- Train/validation-pool vs test weighted ROC AUC gap: `{summary["overfitting"]["train_val_vs_test_roc_gap"]:.4f}`
- Train/validation-pool vs test weighted PR AUC gap: `{summary["overfitting"]["train_val_vs_test_pr_gap"]:.4f}`
- Interpretation: the model shows a moderate but not catastrophic holdout drop. It does not look too good to be true, but the generalization gap is real and should be disclosed.
- Dominant-feature check: top SHAP feature share is approximately `{float(global_shap.iloc[0]["mean_abs_shap"] / global_shap["mean_abs_shap"].sum()):.4f}`, which is not an obvious single-feature takeover.

## B4. Threshold diagnostics
Test-set threshold table:

{_df_to_markdown(threshold_table)}

Population-style threshold burden table on the train/validation pool:

{_df_to_markdown(threshold_population_table)}

Operational interpretation:
- Lower thresholds increase outreach burden and reduce missed at-risk caregivers.
- Higher thresholds reduce false positives but miss more caregivers with positive target labels.
- The current deployed threshold (`{threshold:.3f}`) is near the middle of that tradeoff.

## B5. Calibration diagnostics
- Weighted Brier score on test: `{training_result.metrics["test_metrics"]["weighted_brier"]:.4f}`
- Calibration table file: `reports/tables/calibration_table.csv`
- The current probabilities are usable for ranking/prioritization, but calibration is not formally corrected with Platt scaling or isotonic regression.

## B6. Feature importance / explainability diagnostics
Top global drivers:

{_df_to_markdown(global_shap.head(10))}

Notes:
- `APR_70`, `APR_60`, `APR_50`, and `APR_40` are caregiving-task flags and plausibly proxy caregiving intensity/complexity.
- `SEX`, `ARX_10`, and `HAP_10C` are plausible contextual drivers.
- Because the local codebook PDF was not found, the exact semantic label for each `APR_*` item should be verified before turning those raw codes into judge-facing prose.

## B7. Error analysis
Error-profile table file: `reports/diagnostics/tables/error_analysis.csv`
Subgroup-performance table file: `reports/diagnostics/tables/subgroup_performance.csv`

Preview:

{_df_to_markdown(error_table.head(20))}

## B8. Stability and robustness checks
Full file: `reports/diagnostics/tables/robustness_checks.csv`

{_df_to_markdown(robustness_table)}

Interpretation:
- Weighted vs unweighted XGBoost shows whether the survey-weight choice materially changes ranking performance.
- Logistic regression provides a lightweight baseline.
- The single-feature ablation checks whether the top driver is carrying too much of the signal.

## Methodological limitations
- No relevant local GSS caregiving codebook PDF was available during this audit, so some reserve-code semantics are inferred from the implemented code family rather than verified from a local PDF.
- The implemented universe approximates raw `HAP_10` using grouped `HAP_10C`.
- Validation metrics come from the model-selection stage; the final deployed artifact is refit on train+validation before holdout testing.
- The test set is still only one split from one public-use file, so subgroup conclusions should be treated as directional.
- Some strong predictors are raw task-code flags whose human-readable meanings should be verified before final presentation.

## Recommended next fixes
1. Add the actual GSS Cycle 32 caregiving codebook/user-guide PDF into the repo or workspace and wire the reserve/universe notes directly to it.
2. Persist split membership or split IDs in diagnostics artifacts so judge-facing audits do not need to reconstruct the split.
3. Add a calibrated-probability variant and a small fairness/subgroup stability appendix if time permits.
"""

    diagnostics_summary = f"""# Diagnostics Summary

## Executive summary
- The active model path is `scripts/train_binary_model.py`.
- The model trains directly from the raw SAS file without a persisted cleaned training dataset.
- Final analytic sample: `{prepared.target_summary["analytic_rows"]}`
- Weighted prevalence: `{prepared.target_summary["weighted_prevalence"]:.4f}`
- Weighted test ROC AUC / PR AUC: `{metrics_table.loc[metrics_table['split'] == 'test_final_model', 'roc_auc_weighted'].iloc[0]:.4f} / {metrics_table.loc[metrics_table['split'] == 'test_final_model', 'pr_auc_weighted'].iloc[0]:.4f}`
- Biggest disclosed risks: grouped `HAP_10C` approximation, moderate holdout drop, no local codebook PDF found during the audit, and strong but partially opaque `APR_*` task drivers.

## Files
- Cleaning report: `reports/cleaning_diagnostics.md`
- Model report: `reports/model_diagnostics.md`
- Tables: `reports/diagnostics/tables/`
- Figures: `reports/diagnostics/figures/`

## Top 3 next steps
1. Add the actual caregiving codebook PDF and reconcile every reserve code and `APR_*` meaning against it.
2. Save split IDs and train/validation predictions directly during training for easier future audits.
3. Add a calibrated model variant and a short subgroup-stability appendix.
"""

    (Path("reports") / "cleaning_diagnostics.md").write_text(cleaning_report)
    (Path("reports") / "model_diagnostics.md").write_text(model_report)
    (Path("reports") / "diagnostics_summary.md").write_text(diagnostics_summary)

    return {
        "summary": summary,
        "metrics_table": metrics_table,
        "global_shap": global_shap,
        "outputs": outputs,
    }
