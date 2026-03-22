# Cleaning Diagnostics

## Executive summary
- Active training path: `scripts/train_binary_model.py -> prepare_training_frame() -> fit_binary_model() -> save_training_outputs()`
- Active model input file: `/Users/anuda/Desktop/DataQuest-2026-DAMB/src/c32pumfm.sas7bdat`
- Historical notebook `src/eda.ipynb` exists, but it is not called by the active pipeline.
- No relevant local GSS Cycle 32 caregiving PDF/codebook file was found during this audit, so reserve-code labels beyond the implemented mappings are explicitly marked as inferred.

## A1. Data source and shape audit
| artifact_type | path | used_for_training | rows | columns |
| --- | --- | --- | --- | --- |
| raw_model_input | /Users/anuda/Desktop/DataQuest-2026-DAMB/src/c32pumfm.sas7bdat | True | 20258 | 1071 |
| cleaned_output_scored_dataset | data/processed/scored_caregiver_distress.csv | False | 5587 | 11 |

- Raw duplicate rows: `0`
- Raw duplicate `PUMFID` values: `0`
- Modeled duplicate rows: `0`
- Modeled duplicate `PUMFID` values: `0`

## A2. Required-column audit
| required_variable | found_in_raw | missing_from_raw |
| --- | --- | --- |
| PUMFID | True | False |
| WGHT_PER | True | False |
| DV_PROXY | True | False |
| PAR_10 | True | False |
| HAP_10C | True | False |
| CRH_20 | True | False |
| CRH_30 | True | False |
| CRH_60 | True | False |
| AGEGR10 | True | False |
| SEX | True | False |
| MARSTAT | True | False |
| LIVARR08 | True | False |
| PRV | True | False |
| COW_10 | True | False |
| WTI_110 | True | False |
| NWE_110 | True | False |
| UHW_16GR | True | False |
| UCA_10 | True | False |
| FWA_134 | True | False |
| FWA_137 | True | False |
| APR_10 | True | False |
| APR_20 | True | False |
| APR_30 | True | False |
| APR_40 | True | False |
| APR_50 | True | False |
| APR_60 | True | False |
| APR_70 | True | False |
| APR_80 | True | False |
| ARV_10 | True | False |
| ARX_10 | True | False |
| TTLINCG1 | True | False |
| FAMINCG1 | True | False |
| CHC_100 | True | False |
| PRA_10GR | True | False |

Full file: `reports/diagnostics/tables/required_column_audit.csv`

## A3. Universe/filtering audit
| step | condition | rows_before | rows_after | rows_retained_pct | weighted_before | weighted_after | weighted_retained_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Loaded rows | No filter | 20258 | 20258 | 100 | 30755184.5265 | 30755184.5265 | 100 |
| Proxy exclusion | DV_PROXY == 2 | 20258 | 19882 | 98.1439 | 30755184.5265 | 30159499.6800 | 98.0631 |
| Caregiver eligibility | PAR_10 in 1..99 | 19882 | 7395 | 37.1944 | 30159499.6800 | 7401806.7764 | 24.5422 |
| Care-hours approximation | HAP_10C in 1..6 | 7395 | 6972 | 94.2799 | 7401806.7764 | 7021963.7264 | 94.8682 |
| Target answerability | distress_flag not missing | 6972 | 5587 | 80.1348 | 7021963.7264 | 5651841.4577 | 80.4880 |

Implemented universe note: the code uses grouped `HAP_10C` because raw `HAP_10` is not present in the active raw file.

## A4. Special-code / reserve-code audit
Full file: `reports/diagnostics/tables/special_code_audit.csv`

| variable | family | raw_values_seen | values_mapped_to_nan | values_mapped_to_0 | values_mapped_to_1 | notes |
| --- | --- | --- | --- | --- | --- | --- |
| WGHT_PER | weight | 20092 distinct values; min=10.3541; max=30702.1052 |  |  |  | No explicit reserve mapping in code. |
| CRH_20 | target | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| CRH_30 | target | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| CRH_60 | target | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| AGEGR10 | ordinal predictor | 1, 2, 3, 4, 5, 6, 7 | 96, 97, 98, 99 |  |  | 96=off-path/structural (inferred); 97=don't know (inferred); 98=refusal (inferred); 99=not stated (inferred) |
| SEX | nominal predictor | 1, 2 | 6, 7, 8, 9 |  |  | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| MARSTAT | nominal predictor | 1, 2, 3, 4, 5, 6, 99 | 96, 97, 98, 99 |  |  | 96=off-path/structural (inferred); 97=don't know (inferred); 98=refusal (inferred); 99=not stated (inferred) |
| LIVARR08 | nominal predictor | 1, 2, 3, 4, 5, 6, 7, 8 | 96, 97, 98, 99 |  |  | 96=off-path/structural (inferred); 97=don't know (inferred); 98=refusal (inferred); 99=not stated (inferred) |
| PRV | nominal predictor | 10, 11, 12, 13, 24, 35, 46, 47, 48, 59 | 96, 97, 98, 99 |  |  | 96=off-path/structural (inferred); 97=don't know (inferred); 98=refusal (inferred); 99=not stated (inferred) |
| PAR_10 | numeric/count predictor | 28 distinct values; min=0.0000; max=999.0000 | 996, 997, 998, 999 |  |  | 996=off-path/structural (inferred); 997=don't know (inferred); 998=refusal (inferred); 999=not stated (inferred) |
| HAP_10C | ordinal predictor | 0, 1, 2, 3, 4, 5, 6, 96, 99 | 96, 97, 98, 99 |  |  | 96=off-path/structural (inferred); 97=don't know (inferred); 98=refusal (inferred); 99=not stated (inferred) |
| COW_10 | nominal predictor | 1, 2, 3, 6, 9 | 6, 7, 8, 9 |  |  | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| WTI_110 | nominal predictor | 1, 2, 3, 6, 9 | 6, 7, 8, 9 |  |  | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| NWE_110 | numeric/count predictor | 54 distinct values; min=1.0000; max=99.0000 | 96, 97, 98, 99 |  |  | 96=off-path/structural (inferred); 97=don't know (inferred); 98=refusal (inferred); 99=not stated (inferred) |
| UHW_16GR | ordinal predictor | 1, 2, 3, 4, 6, 9 | 6, 7, 8, 9 |  |  | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| UCA_10 | nominal predictor | 1, 2, 6, 9 | 6, 7, 8, 9 |  |  | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| FWA_134 | binary predictor | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| FWA_137 | binary predictor | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| APR_10 | binary predictor | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| APR_20 | binary predictor | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| APR_30 | binary predictor | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| APR_40 | binary predictor | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| APR_50 | binary predictor | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| APR_60 | binary predictor | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| APR_70 | binary predictor | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| APR_80 | binary predictor | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| ARV_10 | binary predictor | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| ARX_10 | binary predictor | 1, 2, 6, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| TTLINCG1 | ordinal predictor | 1, 2, 3, 4, 5, 6, 7 | 96, 97, 98, 99 |  |  | 96=off-path/structural (inferred); 97=don't know (inferred); 98=refusal (inferred); 99=not stated (inferred) |
| FAMINCG1 | ordinal predictor | 1, 2, 3, 4, 5, 6, 7, 8 | 96, 97, 98, 99 |  |  | 96=off-path/structural (inferred); 97=don't know (inferred); 98=refusal (inferred); 99=not stated (inferred) |
| CHC_100 | binary predictor | 1, 2, 9 | 6, 7, 8, 9 | 2 | 1 | 6=off-path/structural (inferred); 7=don't know (inferred); 8=refusal (inferred); 9=not stated (inferred) |
| PRA_10GR | analytics-only | 1, 2, 3, 4, 5, 6, 7, 8, 9, 96, 99 | 96, 97, 98, 99 |  |  | 96=off-path/structural (inferred); 97=don't know (inferred); 98=refusal (inferred); 99=not stated (inferred) |
| OAC_20 | excluded | 1, 2, 6, 9 |  |  |  | No explicit reserve mapping in code. |
| ARV_40 | excluded | 1, 2, 6, 9 |  |  |  | No explicit reserve mapping in code. |
| ARX_40 | excluded | 1, 2, 6, 9 |  |  |  | No explicit reserve mapping in code. |
| RES_10 | excluded | 1, 2, 6, 9 |  |  |  | No explicit reserve mapping in code. |
| ACD_80 | excluded | 1, 2, 6, 9 |  |  |  | No explicit reserve mapping in code. |
| ACD_90 | excluded | 1, 2, 6, 9 |  |  |  | No explicit reserve mapping in code. |
| CHC_110K | excluded | 1, 2, 6 |  |  |  | No explicit reserve mapping in code. |
| CHC_110S | excluded | 1, 2, 6 |  |  |  | No explicit reserve mapping in code. |
| ICS_40 | excluded | 1, 2, 3, 4, 6, 9 |  |  |  | No explicit reserve mapping in code. |
| FIS_10A | excluded | 1, 2, 6 |  |  |  | No explicit reserve mapping in code. |
| FIS_10H | excluded | 1, 2, 6 |  |  |  | No explicit reserve mapping in code. |

## A5. Variable-family cleaning audit
Full file: `reports/diagnostics/tables/variable_audit.csv`

| variable | meaning | family | used_in_model | raw_coding_summary | cleaned_coding | missing_pct_after_cleaning | weighted_missing_pct_after_cleaning | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRH_20 | Worried or anxious | target | False | 1, 2, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN, then distress_score/distress_flag derived | 0.0041 | 0.0046 |  |
| CRH_30 | Overwhelmed | target | False | 1, 2, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN, then distress_score/distress_flag derived | 0.0174 | 0.0181 |  |
| CRH_60 | Depressed | target | False | 1, 2, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN, then distress_score/distress_flag derived | 0.0260 | 0.0284 |  |
| AGEGR10 | Age group | ordinal predictor | True | 1, 2, 3, 4, 5, 6, 7 | reserve codes -> NaN, numeric/ordinal preserved | 0 | 0 |  |
| SEX | Sex | nominal predictor | True | 1, 2 | reserve codes -> NaN, values cast to string/object, Missing category added in encoder | 0 | 0 |  |
| MARSTAT | Marital status | nominal predictor | True | 1, 2, 3, 4, 5, 6, 99 | reserve codes -> NaN, values cast to string/object, Missing category added in encoder | 0.0014 | 0.0018 |  |
| LIVARR08 | Living arrangement | nominal predictor | True | 1, 2, 3, 4, 5, 6, 7, 8 | reserve codes -> NaN, values cast to string/object, Missing category added in encoder | 0 | 0 |  |
| PRV | Province | nominal predictor | True | 10, 11, 12, 13, 24, 35, 46, 47, 48, 59 | reserve codes -> NaN, values cast to string/object, Missing category added in encoder | 0 | 0 |  |
| PAR_10 | People helped | numeric/count predictor | True | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 22, 24, 25, 30, 32, 40, 60 | reserve codes -> NaN, numeric/ordinal preserved | 0 | 0 |  |
| HAP_10C | Weekly care hours | ordinal predictor | True | 1, 2, 3, 4, 5, 6 | reserve codes -> NaN, numeric/ordinal preserved | 0 | 0 |  |
| COW_10 | Class of worker | nominal predictor | True | 1, 2, 3, 6, 9 | reserve codes -> NaN, values cast to string/object, Missing category added in encoder | 0.3610 | 0.3113 |  |
| WTI_110 | Work tenure indicator | nominal predictor | True | 1, 2, 3, 6, 9 | reserve codes -> NaN, values cast to string/object, Missing category added in encoder | 0.4750 | 0.4322 |  |
| NWE_110 | Weeks worked | numeric/count predictor | True | 54 distinct values; min=1.0000; max=99.0000 | reserve codes -> NaN, numeric/ordinal preserved | 0.3623 | 0.3156 |  |
| UHW_16GR | Usual hours worked | ordinal predictor | True | 1, 2, 3, 4, 6, 9 | reserve codes -> NaN, numeric/ordinal preserved | 0.3601 | 0.3135 |  |
| UCA_10 | Work flexibility indicator | nominal predictor | True | 1, 2, 6, 9 | reserve codes -> NaN, values cast to string/object, Missing category added in encoder | 0.3522 | 0.3089 |  |
| FWA_134 | Employer family-care leave | binary predictor | True | 1, 2, 6, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN | 0.4899 | 0.4468 |  |
| FWA_137 | Employer telework option | binary predictor | True | 1, 2, 6, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN | 0.5645 | 0.5320 |  |
| APR_10 | APR_10 | binary predictor | True | 1, 2, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN | 0.0009 | 0.0010 |  |
| APR_20 | APR_20 | binary predictor | True | 1, 2, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN | 0.0030 | 0.0035 |  |
| APR_30 | APR_30 | binary predictor | True | 1, 2, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN | 0.0041 | 0.0045 |  |
| APR_40 | APR_40 | binary predictor | True | 1, 2, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN | 0.0052 | 0.0058 |  |
| APR_50 | APR_50 | binary predictor | True | 1, 2, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN | 0.0055 | 0.0066 |  |
| APR_60 | APR_60 | binary predictor | True | 1, 2, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN | 0.0041 | 0.0049 |  |
| APR_70 | APR_70 | binary predictor | True | 1, 2, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN | 0.0043 | 0.0036 |  |
| APR_80 | APR_80 | binary predictor | True | 1, 2, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN | 0.0043 | 0.0040 |  |
| ARV_10 | Visits or calls care receiver | binary predictor | True | 1, 2, 6, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN | 0.3304 | 0.4228 |  |
| ARX_10 | Provides emotional support | binary predictor | True | 1, 2, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN | 0.0063 | 0.0102 |  |
| TTLINCG1 | Personal income group | ordinal predictor | True | 1, 2, 3, 4, 5, 6, 7 | reserve codes -> NaN, numeric/ordinal preserved | 0 | 0 |  |
| FAMINCG1 | Household income group | ordinal predictor | True | 1, 2, 3, 4, 5, 6, 7, 8 | reserve codes -> NaN, numeric/ordinal preserved | 0 | 0 |  |
| CHC_100 | Own long-term health condition | binary predictor | True | 1, 2, 9 | 1 -> 1, 2 -> 0, reserve codes -> NaN | 0.0192 | 0.0175 |  |
| WGHT_PER | Person weight | weight | False | 5568 distinct values; min=17.3681; max=24649.8574 | reserve codes -> NaN, numeric/ordinal preserved | 0 | 0 |  |
| PRA_10GR | PRA_10GR | analytics-only | False | 1, 2, 3, 4, 5, 6, 7, 8, 9, 96, 99 | not in deployed model | 0.8616 | 0.8686 |  |
| OAC_20 | OAC_20 | excluded | False | 1, 2, 9 | not in deployed model | 0 | 0 | Explicitly not deployed; interpreted as wanting additional support and treated as borderline/co-determined. |
| ARV_40 | ARV_40 | excluded | False | 1, 2, 6, 9 | not in deployed model | 0 | 0 | Not deployed in the current active pipeline. |
| ARX_40 | ARX_40 | excluded | False | 1, 2, 6, 9 | not in deployed model | 0 | 0 | Not deployed in the current active pipeline. |
| RES_10 | RES_10 | excluded | False | 1, 2, 6, 9 | not in deployed model | 0 | 0 | Not deployed; very high missingness in the modeled universe. |
| ACD_80 | ACD_80 | excluded | False | 1, 2, 9 | not in deployed model | 0 | 0 | Not deployed in the current active pipeline. |
| ACD_90 | ACD_90 | excluded | False | 1, 2, 9 | not in deployed model | 0 | 0 | Not deployed in the current active pipeline. |
| CHC_110K | CHC_110K | excluded | False | 1, 2, 6 | not in deployed model | 0 | 0 | Not deployed; very high missingness in the modeled universe. |
| CHC_110S | CHC_110S | excluded | False | 1, 2, 6 | not in deployed model | 0 | 0 | Not deployed; very high missingness in the modeled universe. |
| ICS_40 | ICS_40 | excluded | False | 1, 2, 3, 4, 9 | not in deployed model | 0 | 0 | Explicit leakage exclusion. |
| FIS_10A | FIS_10A | excluded | False | 1, 2, 6 | not in deployed model | 0 | 0 | Explicit leakage exclusion. |
| FIS_10H | FIS_10H | excluded | False | 1, 2, 6 | not in deployed model | 0 | 0 | Explicit leakage exclusion. |

## A6. Target construction audit
- Source variables: `CRH_20, CRH_30, CRH_60`
- Recoding: `1 -> 1`, `2 -> 0`, reserve codes -> `NaN`
- Partial missingness: `distress_score` uses `sum(min_count=1)` so any observed distress item can contribute
- Final binary rule: `distress_flag = 1 if distress_score >= 1 else 0`
- Final analytic sample size: `5587`
- Unweighted prevalence: `0.6975`
- Weighted prevalence: `0.6908`
- Distress-score distribution: `{'0': 1690, '1': 1565, '2': 1208, '3': 1124}`

## A7. Missingness audit
Full file: `reports/diagnostics/tables/missingness_audit.csv`

| variable | used_in_model | missing_count | missing_pct | weighted_missing_pct | likely_reason | action_taken |
| --- | --- | --- | --- | --- | --- | --- |
| FWA_137 | True | 3154 | 0.5645 | 0.5320 | mostly structural/off-path | binary recode |
| FWA_134 | True | 2737 | 0.4899 | 0.4468 | mostly structural/off-path | binary recode |
| WTI_110 | True | 2654 | 0.4750 | 0.4322 | mostly structural/off-path | explicit missing category |
| NWE_110 | True | 2024 | 0.3623 | 0.3156 | mostly structural/off-path | left as NaN |
| COW_10 | True | 2017 | 0.3610 | 0.3113 | mostly structural/off-path | explicit missing category |
| UHW_16GR | True | 2012 | 0.3601 | 0.3135 | mostly structural/off-path | left as NaN |
| UCA_10 | True | 1968 | 0.3522 | 0.3089 | mostly structural/off-path | explicit missing category |
| ARV_10 | True | 1846 | 0.3304 | 0.4228 | mostly structural/off-path | binary recode |
| CHC_100 | True | 107 | 0.0192 | 0.0175 | mixed nonresponse | binary recode |
| ARX_10 | True | 35 | 0.0063 | 0.0102 | mixed nonresponse | binary recode |
| APR_50 | True | 31 | 0.0055 | 0.0066 | mixed nonresponse | binary recode |
| APR_40 | True | 29 | 0.0052 | 0.0058 | mixed nonresponse | binary recode |
| APR_70 | True | 24 | 0.0043 | 0.0036 | mixed nonresponse | binary recode |
| APR_80 | True | 24 | 0.0043 | 0.0040 | mixed nonresponse | binary recode |
| APR_30 | True | 23 | 0.0041 | 0.0045 | mixed nonresponse | binary recode |
| APR_60 | True | 23 | 0.0041 | 0.0049 | mixed nonresponse | binary recode |
| APR_20 | True | 17 | 0.0030 | 0.0035 | mixed nonresponse | binary recode |
| MARSTAT | True | 8 | 0.0014 | 0.0018 | mixed nonresponse | explicit missing category |
| APR_10 | True | 5 | 0.0009 | 0.0010 | mixed nonresponse | binary recode |
| AGEGR10 | True | 0 | 0 | 0 | unknown | left as NaN |
| SEX | True | 0 | 0 | 0 | unknown | explicit missing category |
| LIVARR08 | True | 0 | 0 | 0 | unknown | explicit missing category |
| PRV | True | 0 | 0 | 0 | unknown | explicit missing category |
| PAR_10 | True | 0 | 0 | 0 | unknown | left as NaN |
| HAP_10C | True | 0 | 0 | 0 | unknown | left as NaN |
| TTLINCG1 | True | 0 | 0 | 0 | unknown | left as NaN |
| FAMINCG1 | True | 0 | 0 | 0 | unknown | left as NaN |
| PRA_10GR | False | 4814 | 0.8616 | 0.8686 | mostly structural/off-path | dropped/excluded |
| CRH_60 | False | 145 | 0.0260 | 0.0284 | mixed nonresponse | target recode |
| CRH_30 | False | 97 | 0.0174 | 0.0181 | mixed nonresponse | target recode |
| CRH_20 | False | 23 | 0.0041 | 0.0046 | mixed nonresponse | target recode |
| WGHT_PER | False | 0 | 0 | 0 | unknown | left as NaN |
| OAC_20 | False | 0 | 0 | 0 | unknown | dropped/excluded |
| ARV_40 | False | 0 | 0 | 0 | unknown | dropped/excluded |
| ARX_40 | False | 0 | 0 | 0 | unknown | dropped/excluded |
| RES_10 | False | 0 | 0 | 0 | unknown | dropped/excluded |
| ACD_80 | False | 0 | 0 | 0 | unknown | dropped/excluded |
| ACD_90 | False | 0 | 0 | 0 | unknown | dropped/excluded |
| CHC_110K | False | 0 | 0 | 0 | unknown | dropped/excluded |
| CHC_110S | False | 0 | 0 | 0 | unknown | dropped/excluded |
| ICS_40 | False | 0 | 0 | 0 | unknown | dropped/excluded |
| FIS_10A | False | 0 | 0 | 0 | unknown | dropped/excluded |
| FIS_10H | False | 0 | 0 | 0 | unknown | dropped/excluded |

## A8. Encoding audit
- Numeric/count predictors: `PAR_10, NWE_110`
- Ordinal predictors: `AGEGR10, HAP_10C, UHW_16GR, TTLINCG1, FAMINCG1`
- Nominal predictors one-hot encoded with explicit Missing category: `SEX, MARSTAT, LIVARR08, PRV, COW_10, WTI_110, UCA_10`
- Binary predictors: `FWA_134, FWA_137, APR_10, APR_20, APR_30, APR_40, APR_50, APR_60, APR_70, APR_80, ARV_10, ARX_10, CHC_100`
- Final transformed feature count entering the model: `58`
- One-hot encoder uses `min_frequency=20`; no additional manual sparse-level collapse exists in the active code.

Full transformed-feature file: `reports/diagnostics/tables/transformed_features.csv`

## A9. Leakage audit
- Final feature leakage audit: `{'exact_overlap': [], 'prefix_overlap': []}`
- Explicitly excluded examples:
{
  "OAC_20": "Explicitly not deployed; interpreted as wanting additional support and treated as borderline/co-determined.",
  "ARV_40": "Not deployed in the current active pipeline.",
  "ARX_40": "Not deployed in the current active pipeline.",
  "RES_10": "Not deployed; very high missingness in the modeled universe.",
  "ACD_80": "Not deployed in the current active pipeline.",
  "ACD_90": "Not deployed in the current active pipeline.",
  "CHC_110K": "Not deployed; very high missingness in the modeled universe.",
  "CHC_110S": "Not deployed; very high missingness in the modeled universe.",
  "ICS_40": "Explicit leakage exclusion.",
  "FIS_10A": "Explicit leakage exclusion.",
  "FIS_10H": "Explicit leakage exclusion."
}
- Suspicious-but-retained notes:
  - APR task flags are strong drivers and should be narrated with care.
  - No banned `CRH_*`, `ICS_*`, or `FIS_*` fields appear in the active final feature list.

## A10. Split and preprocessing audit
- First split: train/validation pool vs test using `train_test_split(..., test_size=0.20, stratify=y, random_state=42)`
- Second split inside the train/validation pool: train vs validation using `test_size=0.25`, again stratified with `random_state=42`
- Preprocessing is fit on the training fold for model selection, not on the full dataset before splitting
- CV tuning uses 3-fold stratified CV on the train/validation pool
- Final deployed preprocessor/model are refit on the full train/validation pool after threshold selection, leaving the test set untouched
- Sample weights are split alongside the rows and normalized to mean 1 for XGBoost fitting
