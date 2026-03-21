from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damb.config import MODEL_FEATURES, RAW_DATA_PATH, WEIGHT_COLUMN
from damb.pipeline import fit_binary_model, leakage_audit, prepare_training_frame, save_training_outputs


def main() -> None:
    prepared = prepare_training_frame(str(RAW_DATA_PATH))
    training_result = fit_binary_model(prepared.frame)
    outputs = save_training_outputs(prepared=prepared, training_result=training_result)
    summary = outputs["validation_summary"]
    summary["saved_artifacts"] = outputs["manifest"]
    summary["row_count_trace"] = prepared.universe_counts
    summary["weight_checks"] = {
        "non_null": bool(prepared.frame[WEIGHT_COLUMN].notna().all()),
        "strictly_positive": bool((prepared.frame[WEIGHT_COLUMN] > 0).all()),
    }
    summary["leakage_audit"] = leakage_audit(MODEL_FEATURES)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
