from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import APP_RESOURCES, FEATURE_COLUMNS, MODEL_ARTIFACT_PATH
from .pipeline import compute_row_contributors


@dataclass
class CaregiverDistressScorer:
    artifact: dict[str, Any]

    @property
    def pipeline(self):
        return self.artifact["pipeline"]

    @property
    def feature_columns(self) -> list[str]:
        return self.artifact["feature_columns"]

    @property
    def threshold(self) -> float:
        return float(self.artifact.get("threshold", 0.5))

    def prepare_row(self, row: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
        if isinstance(row, pd.DataFrame):
            frame = row.copy()
        elif isinstance(row, pd.Series):
            frame = row.to_frame().T
        else:
            frame = pd.DataFrame([row])
        for column in self.feature_columns:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame.loc[:, self.feature_columns]

    def score_row(self, row: dict[str, Any] | pd.Series | pd.DataFrame) -> dict[str, Any]:
        prepared = self.prepare_row(row)
        probability = float(self.pipeline.predict_proba(prepared)[:, 1][0])
        label = int(probability >= self.threshold)
        contributors = compute_row_contributors(self.pipeline, prepared, top_n=5)
        return {
            "probability": probability,
            "label": label,
            "top_contributors": contributors,
            "resources": APP_RESOURCES[label],
        }


def load_trained_scorer(path: str | None = None) -> CaregiverDistressScorer:
    artifact_path = path or str(MODEL_ARTIFACT_PATH)
    artifact = joblib.load(artifact_path)
    missing = sorted(set(FEATURE_COLUMNS) - set(artifact["feature_columns"]))
    if missing:
        raise ValueError(f"Serialized artifact is missing expected feature columns: {missing}")
    return CaregiverDistressScorer(artifact=artifact)
