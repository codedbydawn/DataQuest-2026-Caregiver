# AGENTS.md

## Project Goal
Maintain one coherent binary caregiver-distress modeling pipeline for the Statistics Canada GSS Cycle 32 PUMF in this repo.

## Ground Rules
1. Inspect the real repo and the raw SAS file before changing modeling logic.
2. Treat the English user guide and codebook PDFs as the source of truth for variable meaning, reserve codes, universes, and survey-weight handling.
3. Keep the production path aligned with [`docs/modeling_spec.md`](/Users/anuda/Desktop/DataQuest-2026-DAMB/docs/modeling_spec.md).
4. Do not reintroduce clustering or the deprecated burnout composite unless a new audited spec explicitly replaces the current one.

## Current Modeling Contract
- Main shipped target: `distress_flag`
- Target components: `CRH_20`, `CRH_30`, `CRH_60`
- Main shipped model: weighted binary XGBoost classifier
- Explanation method: XGBoost Tree SHAP contributions
- Single-row scoring output:
  - probability
  - binary label
  - top contributors

## Cautions
- The production universe is a PUMF approximation because raw `HAP_10` is unavailable; use `HAP_10C in 1..6`, not `>= 2`.
- Preserve off-path reserve codes as missing.
- Keep `WGHT_PER` out of predictors.
- Keep `WTBS_*` out of the base model.
- Exclude leakage and downstream consequence variables.
