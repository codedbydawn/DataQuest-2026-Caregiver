from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyreadstat

from .config import (
    ANALYTICS_ONLY_COLUMNS,
    BINARY_FEATURES,
    MODEL_FEATURES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    RESERVE_CODE_MAP,
    TARGET_COLUMNS,
    TARGET_ITEM_LABELS,
    WEIGHT_COLUMN,
)


@dataclass
class PreparedDataset:
    frame: pd.DataFrame
    universe_counts: dict[str, int]
    target_summary: dict[str, Any]
    missingness: pd.DataFrame
    raw_shape: tuple[int, int]


def load_raw_data(path: str | None = None, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    dataset_path = path or str(RAW_DATA_PATH)
    usecols = list(columns) if columns is not None else list(REQUIRED_COLUMNS)
    frame, _ = pyreadstat.read_sas7bdat(dataset_path, usecols=usecols)
    missing = sorted(set(usecols) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame


def recode_special_codes(
    frame: pd.DataFrame,
    reserve_code_map: dict[str, set[int]] | None = None,
) -> pd.DataFrame:
    recoded = frame.copy()
    code_map = reserve_code_map or RESERVE_CODE_MAP
    for column, codes in code_map.items():
        if column in recoded.columns:
            recoded[column] = recoded[column].replace(sorted(codes), np.nan)
    return recoded


def recode_binary_yes_no(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    recoded = frame.copy()
    for column in columns:
        if column in recoded.columns:
            recoded[column] = recoded[column].map({1.0: 1.0, 2.0: 0.0})
    return recoded


def build_target(frame: pd.DataFrame) -> pd.DataFrame:
    target_frame = frame.copy()
    binary_items = pd.DataFrame(index=target_frame.index)
    for column in TARGET_COLUMNS:
        binary_items[column] = target_frame[column].map({1.0: 1.0, 2.0: 0.0})
        target_frame[f"{column}_binary"] = binary_items[column]

    target_frame["distress_score"] = binary_items.sum(axis=1, min_count=1)
    observed = binary_items.notna().any(axis=1)
    target_frame["distress_flag"] = np.where(
        observed,
        (target_frame["distress_score"] >= 1).astype(float),
        np.nan,
    )
    return target_frame


def apply_modeling_universe(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    counts: dict[str, int] = {"raw_rows": int(len(frame))}
    step_1 = frame.loc[frame["DV_PROXY"] == 2].copy()
    counts["after_dv_proxy_eq_2"] = int(len(step_1))
    step_2 = step_1.loc[step_1["PAR_10"].between(1, 99, inclusive="both")].copy()
    counts["after_par_10_1_99"] = int(len(step_2))
    step_3 = step_2.loc[step_2["HAP_10C"].isin([1, 2, 3, 4, 5, 6])].copy()
    counts["after_hap_10c_1_6"] = int(len(step_3))
    final_frame = step_3.loc[step_3["distress_flag"].notna()].copy()
    counts["after_target_nonmissing"] = int(len(final_frame))
    counts["hap_10c_eq_1_kept"] = int((final_frame["HAP_10C"] == 1).sum())
    return final_frame, counts


def summarize_target(frame: pd.DataFrame) -> dict[str, Any]:
    analytic_mask = frame["distress_flag"].notna()
    weights = frame.loc[analytic_mask, WEIGHT_COLUMN].astype(float)
    labels = frame.loc[analytic_mask, "distress_flag"].astype(int)
    item_counts: dict[str, Any] = {}
    for column in TARGET_COLUMNS:
        item_series = frame[column].map({1.0: 1.0, 2.0: 0.0})
        item_counts[column] = {
            "label": TARGET_ITEM_LABELS[column],
            "available_rows": int(item_series.notna().sum()),
            "yes_count": int((item_series == 1).sum()),
        }
    weighted_prevalence = float(np.average(labels, weights=weights)) if len(labels) else np.nan
    distress_distribution = (
        frame.loc[analytic_mask, "distress_score"].value_counts(dropna=False).sort_index()
    )
    return {
        "analytic_rows": int(analytic_mask.sum()),
        "unweighted_prevalence": float(labels.mean()) if len(labels) else np.nan,
        "weighted_prevalence": weighted_prevalence,
        "distress_score_distribution": {
            str(int(score)): int(count) for score, count in distress_distribution.items()
        },
        "target_items": item_counts,
    }


def coerce_feature_types(frame: pd.DataFrame) -> pd.DataFrame:
    typed = frame.copy()
    for column in NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES:
        if column in typed.columns:
            typed[column] = pd.to_numeric(typed[column], errors="coerce")
    for column in NOMINAL_FEATURES + ANALYTICS_ONLY_COLUMNS:
        if column in typed.columns:
            numeric = pd.to_numeric(typed[column], errors="coerce").round()
            typed[column] = numeric.map(lambda value: str(int(value)) if pd.notna(value) else np.nan).astype(object)
    return typed


def compute_missingness_report(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    missing_pct = frame.loc[:, columns].isna().mean().sort_values(ascending=False)
    return pd.DataFrame(
        {
            "feature": missing_pct.index,
            "missing_pct": missing_pct.values,
            "missing_count": frame.loc[:, missing_pct.index].isna().sum().values,
            "non_missing_count": frame.loc[:, missing_pct.index].notna().sum().values,
        }
    )


def prepare_training_frame(path: str | None = None) -> PreparedDataset:
    raw_frame = load_raw_data(path=path)
    recoded = recode_special_codes(raw_frame)
    recoded = recode_binary_yes_no(recoded, BINARY_FEATURES)
    targeted = build_target(recoded)
    modeled, counts = apply_modeling_universe(targeted)
    if modeled.empty:
        raise ValueError("No rows remain after applying the modeling universe")
    if modeled[WEIGHT_COLUMN].isna().any() or (modeled[WEIGHT_COLUMN] <= 0).any():
        raise ValueError("WGHT_PER must be positive and non-null in the analytic sample")
    typed = coerce_feature_types(modeled)
    missingness = compute_missingness_report(typed, MODEL_FEATURES)
    return PreparedDataset(
        frame=typed,
        universe_counts=counts,
        target_summary=summarize_target(typed),
        missingness=missingness,
        raw_shape=raw_frame.shape,
    )
