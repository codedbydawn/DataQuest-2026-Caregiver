from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request

from damb.config import FORM_FIELDS, MODEL_ARTIFACT_PATH
from damb.scoring import CaregiverDistressScorer, load_trained_scorer


app = Flask(__name__)

_SCORER: CaregiverDistressScorer | None = None


def get_scorer() -> CaregiverDistressScorer:
    global _SCORER
    if _SCORER is None:
        if not Path(MODEL_ARTIFACT_PATH).exists():
            raise FileNotFoundError(
                "Model artifact not found. Run `python scripts/train_binary_model.py` first."
            )
        _SCORER = load_trained_scorer(str(MODEL_ARTIFACT_PATH))
    return _SCORER


def parse_form_payload(form_payload: dict[str, str]) -> dict[str, float | None]:
    parsed: dict[str, float | None] = {}
    for field in FORM_FIELDS:
        raw_value = form_payload.get(field["name"], "").strip()
        if raw_value == "":
            parsed[field["name"]] = None
            continue
        parsed[field["name"]] = float(raw_value)
    return parsed


@app.get("/")
def index():
    return render_template("index.html", fields=FORM_FIELDS)


@app.post("/predict")
def predict():
    form_data = request.form.to_dict()
    parsed = parse_form_payload(form_data)
    try:
        scorer = get_scorer()
        score = scorer.score_row(parsed)
        probability = score["probability"]
        label = score["label"]
        results = {
            "risk_label": "Elevated caregiver-distress risk" if label == 1 else "Lower caregiver-distress risk",
            "risk_summary": (
                "This row scores above the binary distress threshold derived from the CRH caregiver-distress items."
                if label == 1
                else "This row scores below the binary distress threshold derived from the CRH caregiver-distress items."
            ),
            "probability_pct": round(probability * 100, 1),
            "binary_label": label,
            "top_factors": score["top_contributors"],
            "resources": score["resources"],
            "submitted_data": form_data,
        }
        return render_template("results.html", results=results, artifact_ready=True)
    except FileNotFoundError as exc:
        return render_template(
            "results.html",
            artifact_ready=False,
            error_message=str(exc),
            results=None,
        )


if __name__ == "__main__":
    app.run(debug=True)
