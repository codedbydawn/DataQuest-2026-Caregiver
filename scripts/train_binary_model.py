from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damb.config import FEATURE_COLUMNS, LEAKAGE_COLUMNS, RAW_DATA_PATH, WEIGHT_COLUMN
from damb.pipeline import fit_binary_model, prepare_training_frame, save_training_outputs


def main() -> None:
    prepared = prepare_training_frame(str(RAW_DATA_PATH))
    fit_output = fit_binary_model(prepared.frame)
    analytic = prepared.frame.loc[prepared.frame["distress_flag"].notna()].copy()
    validation_summary = {
        "raw_shape": {"rows": prepared.universe_counts["raw_rows"], "columns": len(prepared.frame.columns)},
        "universe_counts": prepared.universe_counts,
        "target_summary": prepared.target_summary,
        "weights_check": {
            "non_null": bool(analytic[WEIGHT_COLUMN].notna().all()),
            "strictly_positive": bool((analytic[WEIGHT_COLUMN] > 0).all()),
        },
        "missingness_top10": fit_output["missingness"].head(10).to_dict(orient="records"),
        "leakage_check": {
            "overlap": sorted(set(FEATURE_COLUMNS) & LEAKAGE_COLUMNS),
        },
        "hap_10c_approximation_note": (
            "The official codebook universe uses raw HAP_10 >= 2. "
            "This PUMF exposes only grouped HAP_10C, so the model uses HAP_10C in {1..6}. "
            "Rows with HAP_10C == 1 remain in-scope because valid CRH responses exist there."
        ),
        "metrics": fit_output["metrics"],
    }
    save_training_outputs(
        artifact=fit_output["artifact"],
        metrics=fit_output["metrics"],
        global_shap=fit_output["global_shap"],
        missingness=fit_output["missingness"],
        validation_summary=validation_summary,
    )
    print(json.dumps(validation_summary, indent=2))


if __name__ == "__main__":
    main()
