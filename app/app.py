from flask import Flask, render_template, request


app = Flask(__name__)


# TODO: Load the trained clustering model from ../model/model.pkl once training is complete.
# TODO: Load any preprocessing artifacts needed to reproduce notebook feature engineering at runtime.
# TODO: Keep inference local only; no external ML APIs should be introduced here.
MODEL = None


@app.get("/")
def index():
    """Render the caregiver input form."""
    return render_template("index.html")


@app.post("/predict")
def predict():
    """Receive caregiver input, run clustering, and render profile results."""
    form_data = request.form.to_dict()

    # TODO: Validate and transform form inputs into the feature format expected by the model.
    # TODO: Align form fields with the final person-level feature set derived from maindata and any
    # TODO: episode-level aggregations built from episode via PUMFID.
    # TODO: Load the trained model artifact if it is not already available in memory.
    # TODO: Run local inference against the saved clustering model.
    # TODO: Map the predicted cluster to the final caregiver profile copy and resource list.
    # TODO: Pass real prediction results into the template context.
    results = {
        "profile_name": "TODO: Profile Name",
        "profile_description": "TODO: Replace with generated caregiver profile summary.",
        "top_factors": [
            "TODO: Factor 1",
            "TODO: Factor 2",
            "TODO: Factor 3",
        ],
        "next_steps": [
            "TODO: Add recommended next step.",
        ],
        "resources": [
            "TODO: Add Canadian caregiver support resource.",
        ],
        "submitted_data": form_data,
    }

    return render_template("results.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
