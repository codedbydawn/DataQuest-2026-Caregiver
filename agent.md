
```md
# AGENTS.md

## Project goal
Build a technically defensible caregiver-distress risk system for Statistics Canada GSS Cycle 32 (2018) that helps organizations identify higher-risk caregivers and understand the main drivers of risk. This should help them allocate resources in terms of extra taxes or help needed.

The final product is:
1. an individual caregiver distress-risk model
2. dashboard-ready analytics and visuals for organizations

This repo is **not** centered on caregiver profile classes or a front-end self-assessment survey unless the user explicitly asks for that later.

## Instruction priority
Use this order of authority:
1. actual repo contents
2. workspace PDFs / codebook / user guide
3. audit summary already produced in the active Codex thread
4. outdated repo docs only if they do not conflict with the above

If README or notebook logic conflicts with the PDFs/audit, update it.

## Required workflow
1. Inspect repo structure, dependency files, raw data, and existing modeling code before coding.
2. Verify critical data logic against the PDFs and raw columns.
3. Keep changes modular, testable, and production-usable.
4. Run validation checks and tests before finishing.
5. Summarize:
   - assumptions used
   - files changed
   - metrics
   - artifacts produced
   - limitations

## Modeling direction
Use a 2-stage system:
- **Stage 1:** weighted XGBoost caregiver distress-risk prediction
- **Stage 2:** dashboard-ready analytics, explainability, and resource-allocation outputs

Do not build clustering-based caregiver classes unless explicitly requested later.

## Target rules
Build the main binary target `distress_flag` from:
- `CRH_20`
- `CRH_30`
- `CRH_60`

Recode:
- `1 -> 1`
- `2 -> 0`
- reserve / off-path / nonresponse codes -> `NaN`

Create:
- `distress_score = count of Yes across the three CRH items`, preserving null if all three are missing
- `distress_flag = 1 if distress_score >= 1 else 0`

Do not:
- use ad hoc burnout targets from notebooks
- blend in `ICS_40`
- blend in `FIS_*`
- present the target as a clinical diagnosis

## Universe and cleaning rules
- Verify the exact available care-hours field in the raw data before applying the final universe filter.
- Use the best feasible PUMF-supported approximation and document it.
- Convert special codes to `NaN` consistently.
- Preserve missingness when off-path is not equivalent to “No”.
- Do not impute the target.
- Do not silently mode/median-impute everything.
- Keep raw-to-clean mappings explicit and testable.

## Predictor rules
Use circumstance-based predictors only.

Exclude leakage or downstream consequence variables, including:
- target CRH items
- other `CRH_*`
- `FIS_*`
- `ICL_*`
- `ICB_*`
- `ICP_*`
- `ITL_*`
- `ITO_*`
- `WLB_*`
- `EMO_*`
- other obviously post-outcome variables

Be cautious with `ICS_*`.
If any are retained, justify them and run an ablation check.

## Confirmed variable corrections to respect
Unless direct inspection disproves them:
- `ARX_10` = emotional support
- `ARV_10` = visiting/calling, not emotional support
- `OAC_20` = wants additional/other support
- `CHC_110K` = respondent mental illness
- `CHC_110S` = aging/frailty
- `FWA_137` has stricter universe than `FWA_134`
- `WGHT_PER` is a weight, not a predictor
- `WTBS_*` are not predictors

## Anti-overfitting rules
- Split before fitting data-dependent transforms.
- Keep feature engineering simple and interpretable.
- Limit sparse category explosion.
- Use regularization, subsampling, and early stopping.
- Compare train vs validation performance and flag overfitting.
- Prefer a smaller stable feature set over a wide brittle one.
- remove variables with no use(make sure to verify with pdf documentation provided in prompt)

## XGBoost rules
- Inspect installed xgboost version before finalizing implementation details.
- Use early stopping correctly with a proper validation set.
- Use `WGHT_PER` as sample/analysis weight where appropriate.
- Do not blindly stack `scale_pos_weight` on top of survey weights without justification.
- Tune a compact, disciplined parameter set rather than a huge sweep.
- Save model artifacts predictably.

## Dashboard/output rules
Produce org-facing outputs such as:
- scored dataset
- SHAP importance outputs
- subgroup summaries
- risk heatmaps
- highest-risk segment tables
- hackathon-presentable visuals

Keep the dashboard simple, reliable, and easy to demo.

## Preferred implementation shape
Prefer modules/functions similar to:
- `load_raw_data`
- `recode_special_codes`
- `build_target`
- `apply_modeling_universe`
- `select_features`
- `build_preprocessing_pipeline`
- `train_xgb_model`
- `tune_xgb_model`
- `evaluate_model`
- `compute_shap_outputs`
- `build_dashboard_artifacts`

## Testing requirements
Add tests for:
- recoding
- target construction
- universe filtering
- corrected variable handling
- leakage exclusions
- preprocessing determinism
- scoring behavior
- dashboard artifact smoke tests
- fail-fast behavior on missing columns

## Completion standard
Do not stop at code edits alone. A complete task includes:
- implemented code
- updated docs
- tests
- validation output
- artifact generation
- concise final summary
