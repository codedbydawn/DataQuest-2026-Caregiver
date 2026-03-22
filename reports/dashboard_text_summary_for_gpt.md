# Dashboard Narrative For GPT

Use the following as a factual narrative summary of the current Streamlit dashboard outputs for the caregiver distress model. Treat this as a predictive-risk dashboard, not a causal analysis.

## 1. Analytic Universe and Target
- Raw file rows: 20,258.
- Final analytic sample: 5,587 caregivers after filtering to DV_PROXY=2, PAR_10 in 1..99, HAP_10C in 1..6, and non-missing target items.
- Row-count trace: raw 20,258 -> proxy-eligible 19,882 -> caregiver-universe 7,395 -> grouped-care-hours universe 6,972 -> analytic 5,587.
- Weighted distress prevalence in the analytic sample: 0.691.
- Distress score distribution (count of yes across CRH_20, CRH_30, CRH_60): {'0': 1690, '1': 1565, '2': 1208, '3': 1124}.
- Important caveat: the official CRH universe uses raw HAP_10, but the PUMF only exposes grouped HAP_10C, so the production universe is an approximation.

## 2. Model Performance
- Selected threshold: 0.675.
- Held-out test ROC AUC: 0.736.
- Held-out test PR AUC: 0.860.
- Held-out test weighted accuracy: 0.669.
- Held-out test precision/recall/specificity: 0.816 / 0.686 / 0.626.
- Held-out test weighted Brier score: 0.175.
- Train-vs-validation gap: ROC AUC gap 0.041, PR AUC gap 0.017. This suggests some overfitting but not a total collapse.

## 3. Risk Distribution
- Unweighted risk-band counts: {'Very high': 3222, 'High': 1805, 'Moderate': 487, 'Low': 73}.
- Weighted risk-band totals: {'Very high': 3143616.5, 'High': 1792036.1, 'Moderate': 565783.0, 'Low': 150405.8}.
- In the scored file, the dashboard highlights high-priority cases as predicted_label = 1 using threshold 0.675.

## 4. Top Global Drivers (Feature Importance / SHAP Summary)
- The strongest global drivers are:
  - APR_70: mean absolute SHAP 0.202
  - APR_60: mean absolute SHAP 0.199
  - SEX: mean absolute SHAP 0.152
  - ARX_10: mean absolute SHAP 0.117
  - APR_50: mean absolute SHAP 0.103
  - CRRCPAGR: mean absolute SHAP 0.101
  - HAP_10C: mean absolute SHAP 0.089
  - APR_40: mean absolute SHAP 0.083
  - APR_20: mean absolute SHAP 0.083
  - LIVARR08: mean absolute SHAP 0.057
  - PRG10GR: mean absolute SHAP 0.048
  - AGEGR10: mean absolute SHAP 0.034
- Interpretation: the model is driven most strongly by caregiving task pattern variables (especially banking, scheduling, medical treatment, emotional support), then care intensity, then caregiver/receiver context variables such as receiver age, relationship, living arrangement, sex, and age.

## 5. Subgroup Comparison Highlights
- Highest-risk groups within PRV:
  - British Columbia: avg predicted risk 0.716, weighted count 720351.0, high-priority rate 0.696
  - Nova Scotia: avg predicted risk 0.701, weighted count 171488.7, high-priority rate 0.611
  - Quebec: avg predicted risk 0.697, weighted count 1060922.5, high-priority rate 0.618
- Highest-risk groups within HAP_10C:
  - 50+ hours: avg predicted risk 0.824, weighted count 463926.0, high-priority rate 0.907
  - 30-39 hours: avg predicted risk 0.800, weighted count 232905.5, high-priority rate 0.922
  - 40-49 hours: avg predicted risk 0.791, weighted count 203520.5, high-priority rate 0.888
- Highest-risk groups within TTLINCG1:
  - $100k-$119.9k: avg predicted risk 0.711, weighted count 223976.3, high-priority rate 0.676
  - $40k-$59.9k: avg predicted risk 0.709, weighted count 994253.4, high-priority rate 0.642
  - $80k-$99.9k: avg predicted risk 0.700, weighted count 377982.1, high-priority rate 0.626
- Highest-risk groups within PRA_10GR:
  - 1: avg predicted risk 0.786, weighted count 49175.9, high-priority rate 0.843
  - 6: avg predicted risk 0.727, weighted count 6723.7, high-priority rate 0.774
  - 4: avg predicted risk 0.727, weighted count 31714.6, high-priority rate 0.666
- Highest-risk groups within AGEGR10:
  - 45-54: avg predicted risk 0.730, weighted count 1264201.2, high-priority rate 0.683
  - 35-44: avg predicted risk 0.729, weighted count 712187.4, high-priority rate 0.698
  - 55-64: avg predicted risk 0.719, weighted count 1341128.4, high-priority rate 0.675
- Highest-risk groups within SEX:
  - Female: avg predicted risk 0.738, weighted count 3156799.7, high-priority rate 0.708
  - Male: avg predicted risk 0.632, weighted count 2495041.7, high-priority rate 0.486

## 6. Heatmap Patterns
- Province x risk band: Ontario and Quebec have the largest weighted concentration of high and very-high risk cases in absolute terms, while British Columbia also has a large very-high-risk weighted count.
- Province weighted very-high-risk totals (top 5): [{'PRV': 'Ontario', 'Very high': 1358689.3019}, {'PRV': 'Quebec', 'Very high': 570704.4984}, {'PRV': 'British Columbia', 'Very high': 462851.4383}, {'PRV': 'Alberta', 'Very high': 342101.5411}, {'PRV': 'Manitoba', 'Very high': 110234.4389}].
- Care hours x risk band: higher weekly care-hour categories are much more concentrated in the very-high-risk band. The 30+ hour groups are overwhelmingly in high/very-high risk, and 50+ hours is especially concentrated.
- Care-hour table snapshot: [{'HAP_10C': '10-19 hours', 'Low': 53544.7743, 'Moderate': 92865.3112, 'High': 268365.0457, 'Very high': 626556.4222}, {'HAP_10C': '20-29 hours', 'Low': 0.0, 'Moderate': 6061.548, 'High': 88881.7469, 'Very high': 424109.9812}, {'HAP_10C': '30-39 hours', 'Low': 0.0, 'Moderate': 1754.7175, 'High': 25916.8937, 'Very high': 205233.8741}, {'HAP_10C': '40-49 hours', 'Low': 0.0, 'Moderate': 8884.1593, 'High': 24304.9142, 'Very high': 170331.4342}, {'HAP_10C': '50+ hours', 'Low': 0.0, 'Moderate': 2140.3028, 'High': 44494.5501, 'Very high': 417291.1176}, {'HAP_10C': '<10 hours', 'Low': 96861.0359, 'Moderate': 454076.9794, 'High': 1340072.9852, 'Very high': 1300093.6642}].
- Income x risk band: the largest weighted high and very-high risk totals are in the lower income groups, especially <20k and 20k-39.9k.
- Income weighted very-high-risk totals sorted: [{'TTLINCG1': '<$20k', 'Very high': 908481.0436}, {'TTLINCG1': '$20k-$39.9k', 'Very high': 746557.3033}, {'TTLINCG1': '$40k-$59.9k', 'Very high': 582514.2235}, {'TTLINCG1': '$60k-$79.9k', 'Very high': 374468.9613}, {'TTLINCG1': '$80k-$99.9k', 'Very high': 216683.5689}, {'TTLINCG1': '$120k+', 'Very high': 176280.1784}, {'TTLINCG1': '$100k-$119.9k', 'Very high': 138631.2145}].
- Relationship x risk band: the largest weighted totals are currently under Missing for PRA_10GR in the heatmap output, so relationship-based heatmap interpretation should be treated cautiously. Among coded categories, group 9 and group 5 stand out more than the others in absolute weighted very-high-risk total.

## 7. Highest-Risk Segments
- The highest-risk segment table is dominated by combinations involving high care hours, especially 50+ hours, across provinces and relationship categories.
  - PRA_10GR x HAP_10C | 1 x 50+ hours | avg risk 0.891 | weighted count 8683.7 | high-priority rate 1.000
  - PRA_10GR x HAP_10C | 2 x 20-29 hours | avg risk 0.888 | weighted count 3503.2 | high-priority rate 1.000
  - PRA_10GR x HAP_10C | 6 x 50+ hours | avg risk 0.878 | weighted count 1693.0 | high-priority rate 1.000
  - PRA_10GR x HAP_10C | 8 x 50+ hours | avg risk 0.875 | weighted count 684.1 | high-priority rate 1.000
  - PRA_10GR x HAP_10C | 4 x 40-49 hours | avg risk 0.866 | weighted count 1031.3 | high-priority rate 1.000
  - PRA_10GR x HAP_10C | 3 x 50+ hours | avg risk 0.865 | weighted count 7766.4 | high-priority rate 1.000
  - PRA_10GR x HAP_10C | 5 x 50+ hours | avg risk 0.861 | weighted count 18838.6 | high-priority rate 1.000
  - PRV x HAP_10C | Prince Edward Island x 30-39 hours | avg risk 0.857 | weighted count 706.7 | high-priority rate 1.000
  - PRV x HAP_10C | Alberta x 50+ hours | avg risk 0.848 | weighted count 45738.9 | high-priority rate 0.995
  - PRV x HAP_10C | British Columbia x 50+ hours | avg risk 0.847 | weighted count 56988.4 | high-priority rate 0.964

## 8. Highest-Risk Individuals Table
- The highest-risk individual rows are mostly people in the 20-29, 40-49, or 50+ weekly care-hour groups, often in Ontario, Nova Scotia, or British Columbia, and often in lower income groups.
  - PUMFID 5485: province=Ontario, care_hours=40-49 hours, income=$100k-$119.9k, predicted_probability=0.920, risk_band=Very high
  - PUMFID 3256: province=British Columbia, care_hours=50+ hours, income=<$20k, predicted_probability=0.917, risk_band=Very high
  - PUMFID 15702: province=Nova Scotia, care_hours=50+ hours, income=$20k-$39.9k, predicted_probability=0.914, risk_band=Very high
  - PUMFID 9812: province=Ontario, care_hours=50+ hours, income=$20k-$39.9k, predicted_probability=0.914, risk_band=Very high
  - PUMFID 15452: province=Newfoundland and Labrador, care_hours=50+ hours, income=$20k-$39.9k, predicted_probability=0.913, risk_band=Very high
  - PUMFID 18925: province=Ontario, care_hours=20-29 hours, income=$20k-$39.9k, predicted_probability=0.911, risk_band=Very high
  - PUMFID 8382: province=Saskatchewan, care_hours=50+ hours, income=$60k-$79.9k, predicted_probability=0.911, risk_band=Very high
  - PUMFID 4824: province=Manitoba, care_hours=50+ hours, income=$100k-$119.9k, predicted_probability=0.910, risk_band=Very high

## 9. Subgroup-Specific Drivers
- In province-specific subgroup driver tables, APR_70, APR_60, SEX, ARX_10, and APR_50 recur as the dominant features. This suggests the model is using a similar task-pattern signal across major provinces rather than completely different logic by province.
  - Top drivers for Ontario: [{'feature': 'APR_70', 'mean_abs_shap': 0.211797147989273}, {'feature': 'APR_60', 'mean_abs_shap': 0.1983343213796615}, {'feature': 'SEX', 'mean_abs_shap': 0.154122918844223}, {'feature': 'ARX_10', 'mean_abs_shap': 0.1087733432650566}, {'feature': 'APR_50', 'mean_abs_shap': 0.1017629504203796}]
  - Top drivers for Quebec: [{'feature': 'APR_60', 'mean_abs_shap': 0.1950211971998214}, {'feature': 'APR_70', 'mean_abs_shap': 0.1879608035087585}, {'feature': 'SEX', 'mean_abs_shap': 0.1489766240119934}, {'feature': 'ARX_10', 'mean_abs_shap': 0.1331323534250259}, {'feature': 'APR_50', 'mean_abs_shap': 0.1041027456521987}]
  - Top drivers for British Columbia: [{'feature': 'APR_60', 'mean_abs_shap': 0.1983178257942199}, {'feature': 'APR_70', 'mean_abs_shap': 0.1879350841045379}, {'feature': 'SEX', 'mean_abs_shap': 0.1482267081737518}, {'feature': 'ARX_10', 'mean_abs_shap': 0.1167049035429954}, {'feature': 'APR_50', 'mean_abs_shap': 0.0996764153242111}]

## 10. Calibration and Threshold Tradeoff
- Calibration is directionally reasonable but imperfect. In higher probability bins, observed distress rates are generally high and often exceed predicted risk, suggesting some underprediction in the upper range.
- Top calibration bins: [{'bin': '(0.731, 0.768]', 'weighted_count': 119294.0558, 'mean_predicted_probability': 0.751562648065932, 'observed_rate': 0.7805692251432365}, {'bin': '(0.768, 0.801]', 'weighted_count': 95492.1421, 'mean_predicted_probability': 0.7864870763519198, 'observed_rate': 0.8660050626301742}, {'bin': '(0.801, 0.827]', 'weighted_count': 110221.4108, 'mean_predicted_probability': 0.8148516727153887, 'observed_rate': 0.8072949643282917}, {'bin': '(0.827, 0.852]', 'weighted_count': 86610.4405, 'mean_predicted_probability': 0.8378949689823764, 'observed_rate': 0.9149425547604738}, {'bin': '(0.852, 0.917]', 'weighted_count': 95797.7872, 'mean_predicted_probability': 0.8752109820179182, 'observed_rate': 0.9421686621170724}].
- The chosen threshold 0.675 corresponds to precision 0.757, recall 0.828, specificity 0.413, weighted_accuracy 0.699, and positive_rate 0.753 on the selection curve.
- Lower thresholds from 0.20 to around 0.625 collapsed into an all-positive regime, so the threshold guard was necessary to avoid a degenerate operating point.

## 11. Data Quality and Missingness Caveats
- The most missing active features are FWA_137, FWA_134, WTI_110, AGEPRGR0, NWE_110, COW_10, UHW_16GR, UCA_10, and ARV_10. Many of these are structurally missing because they are only asked in certain work or caregiving universes.
- The file is an all-respondent survey with routed caregiving and care-receiving sections. Valid skip means off-path/not asked, not a substantive no.
- Some derived care-receiving variables such as CARUNPAI and CARPAID collapse off-path cases into their No categories by construction, so those variables should be interpreted differently from direct survey yes/no items.
- This dashboard is best interpreted as predictive decision support for distress risk concentration, not as a causal explanation of why distress happens.