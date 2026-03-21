# Caregiver Distress Modeling Spec

## Goal
Ship one coherent, production-usable binary caregiver-distress risk model for the 2018 Statistics Canada GSS Cycle 32 PUMF in this repo.

## Source of Truth
1. Raw repo contents
2. `C32PUMF_Guide_E.pdf`
3. `c32pumf_cgcr_codebook_NEW_E.pdf`

## Data Reality
- The repo contains one raw person-level SAS file: `src/c32pumfm.sas7bdat`
- The PUMF contains the required target variables and all selected non-leaky predictors
- `WGHT_PER` is the person-level analysis weight
- `WTBS_001` to `WTBS_500` are bootstrap weights reserved for survey variance and sensitivity analysis, not ordinary predictors

## Modeling Universe
Official codebook universe for `CRH_20`, `CRH_30`, and `CRH_60`:
- `DV_PROXY = 2`
- `1 <= PAR_10 <= 99`
- `2 <= HAP_10 <= 168`

The public file in this repo does not expose raw `HAP_10`, only grouped `HAP_10C`. The production approximation implemented here is:
- `DV_PROXY == 2`
- `PAR_10 in 1..99`
- `HAP_10C in 1..6`
- plus non-missing analytic responses on the target items

This intentionally keeps `HAP_10C == 1` rows because valid target responses exist there. Filtering to `HAP_10C >= 2` would be wrong.

## Target Construction
Primary target items:
- `CRH_20`: worried or anxious because of caregiving
- `CRH_30`: overwhelmed because of caregiving
- `CRH_60`: depressed because of caregiving

Recoding:
- `1 -> 1`
- `2 -> 0`
- reserve codes -> `NaN`

Derived targets:
- `distress_score = count_yes(CRH_20, CRH_30, CRH_60)`
- `distress_flag = 1 if distress_score >= 1 else 0`

Important exclusions:
- `ICS_*` is not part of the target
- `FIS_*` is not part of the target
- this is an analyst-defined distress risk label, not a clinical diagnosis

## Predictor Set
Approved feature families:
- demographics
- employment context
- caregiving intensity
- caregiving activity flags
- support network
- financial context
- respondent health

Critical coding corrections enforced in code:
- `ARX_10` is emotional support
- `ARV_10` is visiting or calling
- `OAC_20` means wanting additional support
- `CHC_110K` is respondent mental illness
- `CHC_110S` is aging or frailty
- `FWA_137` has a stricter universe than `FWA_134`
- `TTLINCG1` is used as derived grouped income without inventing extra missingness

## Leakage Policy
The following are excluded from the predictor set:
- `CRH_*`
- `ICS_*`
- `FIS_*`
- `ICL_*`
- `ICB_*`
- `ICP_*`
- `ITL_*`
- `ITO_*`
- other obvious downstream consequences of caregiving strain

## Missingness and Encoding
- Reserve codes are converted to `NaN` using column-specific reserve-code handling
- Numeric features keep `NaN`
- Categorical features use explicit `"Missing"` as a category through the preprocessing pipeline
- No median or mode imputation is used

## Model
- Weighted binary XGBoost classifier
- `WGHT_PER` is normalized to mean 1 for fitting
- `scale_pos_weight` is not layered on top of survey weights

## Evaluation
The training script reports:
- analytic sample size
- weighted and unweighted prevalence
- ROC AUC
- PR AUC
- weighted and unweighted accuracy
- feature missingness summary

## Explanations
- Global and row-level feature contributions are produced with XGBoost Tree SHAP via `pred_contribs=True`
- The shipped app returns probability, binary label, and top contributors
- No fabricated 4-tier class is produced
