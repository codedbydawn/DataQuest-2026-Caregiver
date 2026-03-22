# Diagnostics Summary

## Executive summary
- The active model path is `scripts/train_binary_model.py`.
- The model trains directly from the raw SAS file without a persisted cleaned training dataset.
- Final analytic sample: `5587`
- Weighted prevalence: `0.6908`
- Weighted test ROC AUC / PR AUC: `0.7148 / 0.8429`
- Biggest disclosed risks: grouped `HAP_10C` approximation, moderate holdout drop, no local codebook PDF found during the audit, and strong but partially opaque `APR_*` task drivers.

## Files
- Cleaning report: `reports/cleaning_diagnostics.md`
- Model report: `reports/model_diagnostics.md`
- Tables: `reports/diagnostics/tables/`
- Figures: `reports/diagnostics/figures/`

## Top 3 next steps
1. Add the actual caregiving codebook PDF and reconcile every reserve code and `APR_*` meaning against it.
2. Save split IDs and train/validation predictions directly during training for easier future audits.
3. Add a calibrated model variant and a short subgroup-stability appendix.
