# Model Diagnostics

## Executive summary
- Model class: `xgboost.XGBClassifier`
- Active model package/version: `3.0.5`
- Selected dashboard threshold: `0.625`
- Top 5 global drivers: `[{'feature': 'APR_70', 'mean_abs_shap': 0.22542666}, {'feature': 'APR_60', 'mean_abs_shap': 0.1733961}, {'feature': 'SEX', 'mean_abs_shap': 0.1470236}, {'feature': 'ARX_10', 'mean_abs_shap': 0.113660604}, {'feature': 'HAP_10C', 'mean_abs_shap': 0.11316969}]`

## B1. Model specification audit
- Objective: binary logistic
- Eval metric during fitting: `aucpr`
- Tree method: `hist`
- Early stopping: yes, `early_stopping_rounds=40` during model-selection fits
- Weighting: yes, `WGHT_PER` normalized to mean 1 for training
- Class imbalance handling: no separate `scale_pos_weight`
- Validation scheme: 3-fold CV for bounded search, then stratified train/validation/test split with a held-out test set
- Selected deployed hyperparameters: `{'n_estimators': 50, 'learning_rate': 0.04, 'max_depth': 3, 'min_child_weight': 5, 'subsample': 0.75, 'colsample_bytree': 0.75, 'gamma': 0.0, 'reg_alpha': 0.0, 'reg_lambda': 2.0, 'max_delta_step': 0}`

## B2. Metric report
Full file: `reports/diagnostics/tables/metrics_by_split.csv`

| split | rows | roc_auc_weighted | roc_auc_unweighted | pr_auc_weighted | pr_auc_unweighted | accuracy_weighted | accuracy_unweighted | precision_weighted | precision_unweighted | recall_weighted | recall_unweighted | specificity_weighted | specificity_unweighted | f1_weighted | f1_unweighted | prevalence_weighted | prevalence_unweighted | tp_weighted | fp_weighted | tn_weighted | fn_weighted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_selection_model | 3351 | 0.7806 | 0.7420 | 0.8714 | 0.8537 | 0.7550 | 0.7374 | 0.7640 | 0.7511 | 0.9306 | 0.9324 | 0.3701 | 0.2880 | 0.8391 | 0.8320 | 0.6867 | 0.6974 | 2180943.9966 | 673578.5604 | 395730.0944 | 162682.7556 |
| validation_selection_model | 1118 | 0.7524 | 0.7380 | 0.8593 | 0.8569 | 0.7571 | 0.7469 | 0.7654 | 0.7559 | 0.9329 | 0.9410 | 0.3694 | 0.2988 | 0.8409 | 0.8384 | 0.6880 | 0.6977 | 740185.5441 | 226852.8452 | 132879.3252 | 53242.2528 |
| train_val_final_model | 4469 | 0.7887 | 0.7505 | 0.8781 | 0.8617 | 0.7473 | 0.7357 | 0.8018 | 0.7857 | 0.8399 | 0.8540 | 0.5442 | 0.4630 | 0.8204 | 0.8184 | 0.6870 | 0.6975 | 2634684.8631 | 651397.2827 | 777643.5425 | 502369.6860 |
| test_final_model | 1118 | 0.7148 | 0.7251 | 0.8429 | 0.8436 | 0.6965 | 0.7227 | 0.7863 | 0.7825 | 0.7836 | 0.8346 | 0.4866 | 0.4645 | 0.7850 | 0.8077 | 0.7068 | 0.6977 | 601349.0578 | 163426.6643 | 154900.4180 | 166069.9433 |

## B3. Overfitting diagnostics
- Train vs validation weighted ROC AUC gap: `0.0282`
- Train vs validation weighted PR AUC gap: `0.0121`
- Train/validation-pool vs test weighted ROC AUC gap: `0.0739`
- Train/validation-pool vs test weighted PR AUC gap: `0.0352`
- Interpretation: the model shows a moderate but not catastrophic holdout drop. It does not look too good to be true, but the generalization gap is real and should be disclosed.
- Dominant-feature check: top SHAP feature share is approximately `0.1805`, which is not an obvious single-feature takeover.

## B4. Threshold diagnostics
Test-set threshold table:

| threshold | flagged_rows | flagged_weighted_total | flagged_row_pct | flagged_weight_pct | precision_weighted | recall_weighted | specificity_weighted | f1_weighted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4000 | 1090 | 1044865.9749 | 0.9750 | 0.9623 | 0.7304 | 0.9944 | 0.1150 | 0.8422 |
| 0.5000 | 1064 | 1015698.4960 | 0.9517 | 0.9355 | 0.7395 | 0.9788 | 0.1689 | 0.8425 |
| 0.6000 | 884 | 834997.4980 | 0.7907 | 0.7691 | 0.7805 | 0.8493 | 0.4243 | 0.8134 |
| 0.6250 | 832 | 764775.7221 | 0.7442 | 0.7044 | 0.7863 | 0.7836 | 0.4866 | 0.7850 |
| 0.7000 | 629 | 576408.3109 | 0.5626 | 0.5309 | 0.8266 | 0.6209 | 0.6860 | 0.7091 |
| 0.8000 | 313 | 262607.1978 | 0.2800 | 0.2419 | 0.8730 | 0.2987 | 0.8952 | 0.4451 |

Population-style threshold burden table on the train/validation pool:

| threshold | flagged_rows | flagged_weighted_total | flagged_row_pct | flagged_weight_pct | precision_weighted | recall_weighted | specificity_weighted | f1_weighted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4000 | 4329 | 4331263.5642 | 0.9687 | 0.9486 | 0.7208 | 0.9951 | 0.1536 | 0.8360 |
| 0.5000 | 4242 | 4196423.3548 | 0.9492 | 0.9190 | 0.7357 | 0.9841 | 0.2238 | 0.8419 |
| 0.6000 | 3576 | 3503772.0952 | 0.8002 | 0.7673 | 0.7938 | 0.8866 | 0.4945 | 0.8377 |
| 0.6250 | 3388 | 3286082.1458 | 0.7581 | 0.7197 | 0.8018 | 0.8399 | 0.5442 | 0.8204 |
| 0.7000 | 2508 | 2447070.8905 | 0.5612 | 0.5359 | 0.8606 | 0.6713 | 0.7613 | 0.7542 |
| 0.8000 | 1280 | 1168564.5379 | 0.2864 | 0.2559 | 0.9135 | 0.3403 | 0.9293 | 0.4959 |

Operational interpretation:
- Lower thresholds increase outreach burden and reduce missed at-risk caregivers.
- Higher thresholds reduce false positives but miss more caregivers with positive target labels.
- The current deployed threshold (`0.625`) is near the middle of that tradeoff.

## B5. Calibration diagnostics
- Weighted Brier score on test: `0.1790`
- Calibration table file: `reports/tables/calibration_table.csv`
- The current probabilities are usable for ranking/prioritization, but calibration is not formally corrected with Platt scaling or isotonic regression.

## B6. Feature importance / explainability diagnostics
Top global drivers:

| feature | mean_abs_shap |
| --- | --- |
| APR_70 | 0.2254 |
| APR_60 | 0.1734 |
| SEX | 0.1470 |
| ARX_10 | 0.1137 |
| HAP_10C | 0.1132 |
| APR_50 | 0.0972 |
| APR_40 | 0.0827 |
| APR_20 | 0.0757 |
| LIVARR08 | 0.0594 |
| AGEGR10 | 0.0308 |

Notes:
- `APR_70`, `APR_60`, `APR_50`, and `APR_40` are caregiving-task flags and plausibly proxy caregiving intensity/complexity.
- `SEX`, `ARX_10`, and `HAP_10C` are plausible contextual drivers.
- Because the local codebook PDF was not found, the exact semantic label for each `APR_*` item should be verified before turning those raw codes into judge-facing prose.

## B7. Error analysis
Error-profile table file: `reports/diagnostics/tables/error_analysis.csv`
Subgroup-performance table file: `reports/diagnostics/tables/subgroup_performance.csv`

Preview:

| error_type | dimension | group_value | rows | weighted_rows | mean_predicted_probability |
| --- | --- | --- | --- | --- | --- |
| false_negative | PRV | 10 | 12 | 2269.4078 | 0.5407 |
| false_negative | PRV | 11 | 6 | 683.5381 | 0.5731 |
| false_negative | PRV | 12 | 12 | 3773.5054 | 0.5554 |
| false_negative | PRV | 13 | 6 | 920.0781 | 0.5250 |
| false_negative | PRV | 24 | 16 | 37686.1162 | 0.5569 |
| false_negative | PRV | 35 | 27 | 59803.5727 | 0.5767 |
| false_negative | PRV | 46 | 10 | 12817.3725 | 0.5456 |
| false_negative | PRV | 47 | 12 | 7439.8106 | 0.5583 |
| false_negative | PRV | 48 | 13 | 23725.3103 | 0.5857 |
| false_negative | PRV | 59 | 15 | 16951.2316 | 0.5704 |
| false_negative | HAP_10C | 1 | 102 | 131717.0395 | 0.5628 |
| false_negative | HAP_10C | 2 | 22 | 27468.5258 | 0.5681 |
| false_negative | HAP_10C | 3 | 1 | 80.2326 | 0.3285 |
| false_negative | HAP_10C | 4 | 1 | 1324.0715 | 0.6083 |
| false_negative | HAP_10C | 5 | 1 | 3218.9686 | 0.5999 |
| false_negative | HAP_10C | 6 | 2 | 2261.1053 | 0.5506 |
| false_negative | TTLINCG1 | 1 | 39 | 76426.6164 | 0.5654 |
| false_negative | TTLINCG1 | 2 | 36 | 42377.6607 | 0.5524 |
| false_negative | TTLINCG1 | 3 | 25 | 20226.8161 | 0.5814 |
| false_negative | TTLINCG1 | 4 | 9 | 5870.2359 | 0.5481 |

## B8. Stability and robustness checks
Full file: `reports/diagnostics/tables/robustness_checks.csv`

| check | roc_auc_weighted | pr_auc_weighted |
| --- | --- | --- |
| weighted_xgboost_refit | 0.7148 | 0.8429 |
| unweighted_xgboost_refit | 0.7241 | 0.8424 |
| weighted_logistic_baseline | 0.7017 | 0.8300 |
| ablation_without_APR_70 | 0.7201 | 0.8397 |

Interpretation:
- Weighted vs unweighted XGBoost shows whether the survey-weight choice materially changes ranking performance.
- Logistic regression provides a lightweight baseline.
- The single-feature ablation checks whether the top driver is carrying too much of the signal.

## Methodological limitations
- No relevant local GSS caregiving codebook PDF was available during this audit, so some reserve-code semantics are inferred from the implemented code family rather than verified from a local PDF.
- The implemented universe approximates raw `HAP_10` using grouped `HAP_10C`.
- Validation metrics come from the model-selection stage; the final deployed artifact is refit on train+validation before holdout testing.
- The test set is still only one split from one public-use file, so subgroup conclusions should be treated as directional.
- Some strong predictors are raw task-code flags whose human-readable meanings should be verified before final presentation.

## Recommended next fixes
1. Add the actual GSS Cycle 32 caregiving codebook/user-guide PDF into the repo or workspace and wire the reserve/universe notes directly to it.
2. Persist split membership or split IDs in diagnostics artifacts so judge-facing audits do not need to reconstruct the split.
3. Add a calibrated-probability variant and a small fairness/subgroup stability appendix if time permits.
