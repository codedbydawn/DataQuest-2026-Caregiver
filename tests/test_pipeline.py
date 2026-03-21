from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

from damb.config import FEATURE_COLUMNS
from damb.pipeline import (
    apply_modeling_universe,
    build_target,
    compute_row_contributors,
    recode_reserve_codes,
    select_and_prepare_features,
)
from damb.scoring import CaregiverDistressScorer


def make_feature_frame() -> pd.DataFrame:
    data = {}
    for column in FEATURE_COLUMNS:
        data[column] = [1.0, 2.0, np.nan, 1.0]
    data["PAR_10"] = [1.0, 2.0, 3.0, 4.0]
    data["NWE_110"] = [20.0, 35.0, np.nan, 10.0]
    return pd.DataFrame(data)


def test_recode_reserve_codes_is_width_aware() -> None:
    frame = pd.DataFrame(
        {
            "HAP_10C": [1.0, 6.0, 96.0, 99.0],
            "CRH_20": [1.0, 2.0, 6.0, 9.0],
            "PAR_10": [1.0, 996.0, 999.0, 10.0],
        }
    )
    recoded = recode_reserve_codes(frame)
    assert recoded.loc[0, "HAP_10C"] == 1.0
    assert recoded.loc[1, "HAP_10C"] == 6.0
    assert pd.isna(recoded.loc[2, "HAP_10C"])
    assert pd.isna(recoded.loc[3, "HAP_10C"])
    assert recoded.loc[0, "CRH_20"] == 1.0
    assert recoded.loc[1, "CRH_20"] == 2.0
    assert pd.isna(recoded.loc[2, "CRH_20"])
    assert pd.isna(recoded.loc[3, "CRH_20"])
    assert np.isnan(recoded.loc[1, "PAR_10"])
    assert np.isnan(recoded.loc[2, "PAR_10"])


def test_build_target_uses_only_crh_items() -> None:
    frame = pd.DataFrame(
        {
            "CRH_20": [1.0, 2.0, np.nan, np.nan],
            "CRH_30": [2.0, 2.0, np.nan, 1.0],
            "CRH_60": [2.0, 1.0, np.nan, np.nan],
        }
    )
    built = build_target(frame)
    assert built.loc[0, "distress_score"] == 1.0
    assert built.loc[1, "distress_score"] == 1.0
    assert pd.isna(built.loc[2, "distress_score"])
    assert built.loc[3, "distress_score"] == 1.0
    assert built.loc[0, "distress_flag"] == 1.0
    assert built.loc[1, "distress_flag"] == 1.0
    assert pd.isna(built.loc[2, "distress_flag"])
    assert built.loc[3, "distress_flag"] == 1.0


def test_apply_modeling_universe_keeps_hap_10c_one() -> None:
    frame = pd.DataFrame(
        {
            "DV_PROXY": [2.0, 2.0, 1.0],
            "PAR_10": [1.0, 1.0, 1.0],
            "HAP_10C": [1.0, 2.0, 1.0],
        }
    )
    filtered, counts = apply_modeling_universe(frame)
    assert len(filtered) == 2
    assert counts["hap_10c_eq_1_rows"] == 1


def test_select_and_prepare_features_fails_on_missing_required_column() -> None:
    frame = make_feature_frame()
    frame["distress_flag"] = [0.0, 1.0, 0.0, 1.0]
    frame["WGHT_PER"] = [1.0, 1.0, 1.0, 1.0]
    broken = frame.drop(columns=["ARX_10"])
    with pytest.raises(ValueError, match="Missing feature columns"):
        select_and_prepare_features(broken)


def test_corrected_feature_columns_are_present() -> None:
    assert "ARX_10" in FEATURE_COLUMNS
    assert "ARV_10" in FEATURE_COLUMNS
    assert "OAC_20" in FEATURE_COLUMNS
    assert "CHC_110K" in FEATURE_COLUMNS
    assert "CHC_110S" in FEATURE_COLUMNS


def test_single_row_scorer_returns_probability_label_and_top_contributors() -> None:
    features = make_feature_frame()
    labels = pd.Series([0, 1, 0, 1])
    numeric_features = ["PAR_10", "NWE_110"]
    categorical_features = [column for column in FEATURE_COLUMNS if column not in numeric_features]
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=5,
        max_depth=2,
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        tree_method="hist",
        n_jobs=1,
    )
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    pipeline.fit(features, labels)
    scorer = CaregiverDistressScorer(
        artifact={
            "pipeline": pipeline,
            "feature_columns": list(FEATURE_COLUMNS),
            "threshold": 0.5,
        }
    )
    result = scorer.score_row({"HAP_10C": 1.0, "PAR_10": 2.0, "ARX_10": 1.0, "ARV_10": 1.0})
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] in {0, 1}
    assert len(result["top_contributors"]) == 5
    assert all("direction" in item for item in result["top_contributors"])


def test_compute_row_contributors_preserves_feature_direction_keys() -> None:
    features = make_feature_frame()
    labels = pd.Series([0, 1, 0, 1])
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", ["PAR_10", "NWE_110"]),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                [column for column in FEATURE_COLUMNS if column not in {"PAR_10", "NWE_110"}],
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=5,
                    max_depth=2,
                    learning_rate=0.3,
                    random_state=42,
                    tree_method="hist",
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(features, labels)
    contributors = compute_row_contributors(pipeline, features.iloc[[0]], top_n=3)
    assert len(contributors) == 3
    assert set(contributors[0]) == {"feature", "feature_value", "shap_value", "direction"}
