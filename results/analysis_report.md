# Caregiver Burnout Risk Analysis Report

**Generated:** 2026-03-21 20:53  
**Data:** GSS Caregiving and Care Receiving 2018 (Statistics Canada, Cycle 32)  
**Method:** XGBoost binary classifier + SHAP explanations  
**Burnout split:** `median` (threshold = 0.361)  
**Purpose:** Identify at-risk caregiver groups for government/nonprofit resource allocation  

---

## 1. Dataset Overview

- **Total survey respondents (GSS 2018):** 20,258
- **Respondents routed to caregiving modules (with burnout data):** 5,633
  - The remaining 14,625 respondents were not asked caregiving stress questions (all 12 burnout items = 'Valid skip'). They are likely non-caregivers or were routed to different survey modules.
- **Predictor features in X:** 141
- **Burnout score (0-1 scale, higher = more strain):** mean=0.406, median=0.361, std=0.288, range=[0.000, 1.000]
- **Target split strategy:** `median` at threshold=0.361
- **High-burnout:** 2,863 (50.8%)
- **Low-burnout:** 2,770 (49.2%)
- **Survey weight range:** [17, 24650] (passed as sample_weight during training)

**Note on variable coding:** Most variables in this survey are coded categories (e.g., income brackets, expense ranges, Yes/No flags) not continuous values. XGBoost handles ordinal codes correctly via threshold splits. Descriptive statistics below decode categories where possible.


## 2. Burnout Rate by Key Demographics

% of respondents with high burnout within each group.

### Sex (`SEX`)

| Category | n | High Burnout % |
| --- | --- | --- |
| Female | 3,457 | 56.0% |
| Male | 2,176 | 42.6% |

### Province / Territory (`PRV`)

| Category | n | High Burnout % |
| --- | --- | --- |
| British Columbia | 667 | 56.5% |
| Manitoba | 436 | 53.2% |
| Ontario | 1,273 | 52.9% |
| Alberta | 559 | 52.2% |
| Nova Scotia | 526 | 51.0% |
| Quebec | 636 | 49.4% |
| Newfoundland and Labrador | 378 | 47.1% |
| Saskatchewan | 458 | 46.9% |
| New Brunswick | 465 | 46.0% |
| Prince Edward Island | 235 | 42.1% |

### Household Income Group (`FAMINCG1`)

| Category | n | High Burnout % |
| --- | --- | --- |
| $120,000 to $139,999 | 384 | 54.2% |
| $140,000 or more | 1,232 | 53.3% |
| Less than $20,000 | 304 | 53.0% |
| $60,000 to $79,999 | 800 | 51.1% |
| $80,000 to $99,999 | 653 | 49.6% |
| $40,000 to $59,999 | 878 | 48.1% |
| $20,000 to $39,999 | 851 | 47.6% |

### Marital Status (`MARSTAT`)

| Category | n | High Burnout % |
| --- | --- | --- |
| Separated | 172 | 54.7% |
| Widowed | 377 | 52.5% |
| Married | 3,095 | 51.8% |
| Living common law | 524 | 51.1% |
| Divorced | 513 | 49.9% |

### Visible Minority Status (`VISMIN`)

| Category | n | High Burnout % |
| --- | --- | --- |
| Visible minority | 496 | 53.2% |
| Not a visible minority | 5,043 | 50.6% |

### Senior (65+) Flag (`SENFLAG`)

| Category | n | High Burnout % |
| --- | --- | --- |
| No | 3,461 | 52.8% |
| Yes | 2,172 | 47.7% |


## 3. Top 30 Variables by Pearson Correlation with Burnout Score

Pearson correlation with the 0-1 burnout score (higher = more strain).

**Important caveats:**
- Most predictor variables are ordinal codes, not continuous measurements. Pearson r captures monotonic trends in the codes but assumes equal intervals.
- For binary Yes(1)/No(2) variables with reserve codes replaced by NaN, a negative r means Yes is associated with higher burnout (because code 1 < 2, while burnout score is recoded so higher = more strain).
- Correlations are computed on the raw survey codes in X (not recoded). The SHAP analysis (Section 5) is more reliable for identifying risk factors.

| Rank | Variable | Description | Pearson r | Direction |
| --- | --- | --- | --- | --- |
| 1 | burnout_score | burnout_score | +1.0000 | More -> more burnout |
| 2 | ICL_150 | Care. and family life - Less time take care of self - Past 12 mon | -0.5479 | More -> less burnout |
| 3 | ICE_50 | Caregiving responsibilities affected studies - Past 12 months | -0.5420 | More -> less burnout |
| 4 | ICL_110 | Caregiving and family life - Less time with partner - Past 12 mon | -0.5235 | More -> less burnout |
| 5 | ICL_135 | Caregiving and family life - Less time with friends - Past 12 mon | -0.5165 | More -> less burnout |
| 6 | ICL_140 | Caregiving and family life - Less time on social act. - Past 12 m | -0.5160 | More -> less burnout |
| 7 | ICL_210 | Caregiving and family life - Strain in relationships - Past 12 mo | -0.5073 | More -> less burnout |
| 8 | ICP_15 | Physical strenuous of caregiving responsibilities - Past 12 month | -0.4476 | More -> less burnout |
| 9 | ICL_130 | Care. and family life - Less time with other family - Past 12 mon | -0.4428 | More -> less burnout |
| 10 | ICL_120 | Caregiving and family life - Less time with children - Past 12 mo | -0.4261 | More -> less burnout |
| 11 | ICB_20 | Caregiving affected eating habits - Past 12 months | -0.4248 | More -> less burnout |
| 12 | ICL_154 | Caregiving and family life - Less time social groups - Past 12 mo | -0.3959 | More -> less burnout |
| 13 | OAC_20 | Other caregiving supports - Indicator | -0.3749 | More -> less burnout |
| 14 | APR_60 | Respondent provided help with scheduling - Past 12 months | -0.3661 | More -> less burnout |
| 15 | APR_70 | Respondent provided help with banking - Past 12 months | -0.2983 | More -> less burnout |
| 16 | ite_score | Work-Enabling Circumstances Score | +0.2820 | More -> more burnout |
| 17 | APR_40 | Respondent provided help with personal care - Past 12 months | -0.2801 | More -> less burnout |
| 18 | CCL_20 | Took compassionate care leave for end-of-life care - Past 12 mont | -0.2753 | More -> less burnout |
| 19 | PRG10GR | Relationship with primary care receiver | -0.2714 | More -> less burnout |
| 20 | APR_50 | Respondent provided help with medical treatment - Past 12 months | -0.2688 | More -> less burnout |
| 21 | ACD_10 | Caregiving accommodations - Partner modified their life | -0.2674 | More -> less burnout |
| 22 | EDM_02 | Education - Full or part-time | +0.2530 | More -> more burnout |
| 23 | ITO_10 | Did not advance career due caregiving resp. - Indicator - Past 12 | -0.2502 | More -> less burnout |
| 24 | ICB_25 | Eating habits healthier or less healthy due of caregiving resp. | +0.2444 | More -> more burnout |
| 25 | INE_10 | Caregiving responsibilities prevented respondent from working | -0.2341 | More -> less burnout |
| 26 | TRNS_EXP | Estimate of expenses - Transportation and accommodation | +0.2338 | More -> more burnout |
| 27 | HAP_10C | Number of hours care or help provided by respondent - Per average | +0.2272 | More -> more burnout |
| 28 | RPE_10 | Respondent provided emotional support - Past 12 months | -0.2268 | More -> less burnout |
| 29 | ARX_10 | Emotionally support primary care receiver - Indicator - Past 12 m | -0.2077 | More -> less burnout |
| 30 | APR_20 | Respondent provided help with household chores - Past 12 months | -0.2066 | More -> less burnout |


## 4. XGBoost Model Performance

Train / test split: 80% / 20%  
Trees used (early stopping): 50  
Target split: `median`

- **ROC-AUC:** 0.8488  (0.5 = random, 1.0 = perfect; >0.70 useful for screening)

**Low Burnout:**
  - Precision: 0.764  (of those flagged, this fraction truly belong here)
  - Recall:    0.762  (of all true members, this fraction were caught)
  - F1 Score:  0.763
  - Support:   554 respondents

**High Burnout:**
  - Precision: 0.770  (of those flagged, this fraction truly belong here)
  - Recall:    0.773  (of all true members, this fraction were caught)
  - F1 Score:  0.772
  - Support:   573 respondents

**Overall accuracy:** 0.768


## 5. Top 30 Risk Factors by SHAP Importance

Mean |SHAP| = average influence on the burnout prediction across all respondents.

| Rank | Variable | Description | Mean |SHAP| |
| --- | --- | --- | --- |
| 1 | ICL_210 | Caregiving and family life - Strain in relationships - Past 12 mo | 0.45833 |
| 2 | ICL_150 | Care. and family life - Less time take care of self - Past 12 mon | 0.32137 |
| 3 | ICP_15 | Physical strenuous of caregiving responsibilities - Past 12 month | 0.31298 |
| 4 | ICL_140 | Caregiving and family life - Less time on social act. - Past 12 m | 0.24912 |
| 5 | ICL_135 | Caregiving and family life - Less time with friends - Past 12 mon | 0.19062 |
| 6 | OAC_20 | Other caregiving supports - Indicator | 0.13811 |
| 7 | ICB_25 | Eating habits healthier or less healthy due of caregiving resp. | 0.09757 |
| 8 | ICB_20 | Caregiving affected eating habits - Past 12 months | 0.09304 |
| 9 | ICL_130 | Care. and family life - Less time with other family - Past 12 mon | 0.05825 |
| 10 | ICS_30 | Rewarding aspect of caregiving experiences - Past 12 months | 0.05810 |
| 11 | RPE_10 | Respondent provided emotional support - Past 12 months | 0.04401 |
| 12 | SEX | Sex of respondent | 0.04227 |
| 13 | ARS_20 | Helped primary care receiver, scheduling - Frequency | 0.04091 |
| 14 | CRRCPAGR | Age of primary care receiver | 0.03805 |
| 15 | PRG10GR | Relationship with primary care receiver | 0.03220 |
| 16 | ARS_10 | Helped primary care receiver, scheduling - Indicator - Past 12 mo | 0.03219 |
| 17 | APR_40 | Respondent provided help with personal care - Past 12 months | 0.03189 |
| 18 | HRA_10 | Primary care rece. helped by professionals - Avg. hours per week | 0.03037 |
| 19 | PRP_15 | Severity of primary care receiver's health condition | 0.02971 |
| 20 | ARM_20 | Helped primary care receiver, medical treatment - Frequency | 0.02927 |
| 21 | ARS_30C | Helped primary care receiver, scheduling - Average number of hour | 0.02903 |
| 22 | TRNS_EXP | Estimate of expenses - Transportation and accommodation | 0.02771 |
| 23 | ARB_30C | Helped primary care receiver, banking - Average number of hours | 0.02654 |
| 24 | ICB_15 | Change in amount of exercise due of caregiving responsibilities | 0.02569 |
| 25 | ACD_20 | Caregiving accommodations - Children helped | 0.02541 |
| 26 | ARB_20 | Helped primary care receiver, banking - Frequency | 0.02464 |
| 27 | HOME_EXP | Estimate of expenses - Home modifications | 0.02274 |
| 28 | ARS_40 | Someone else could help with scheduling | 0.02251 |
| 29 | HAP_10C | Number of hours care or help provided by respondent - Per average | 0.02207 |
| 30 | ACD_10 | Caregiving accommodations - Partner modified their life | 0.02069 |

### Average SHAP Direction (Top 20)

Positive = on average raises predicted burnout. Negative = on average lowers it.

| Variable | Description | Mean SHAP | Net effect |
| --- | --- | --- | --- |
| ICL_210 | Caregiving and family life - Strain in relationships - Past 12 mo | -0.01012 | Lowers risk |
| ICL_150 | Care. and family life - Less time take care of self - Past 12 mon | -0.01640 | Lowers risk |
| ICP_15 | Physical strenuous of caregiving responsibilities - Past 12 month | +0.01255 | Raises risk |
| ICL_140 | Caregiving and family life - Less time on social act. - Past 12 m | -0.01512 | Lowers risk |
| ICL_135 | Caregiving and family life - Less time with friends - Past 12 mon | -0.00774 | Lowers risk |
| OAC_20 | Other caregiving supports - Indicator | -0.01586 | Lowers risk |
| ICB_25 | Eating habits healthier or less healthy due of caregiving resp. | -0.00563 | Lowers risk |
| ICB_20 | Caregiving affected eating habits - Past 12 months | -0.00197 | Lowers risk |
| ICL_130 | Care. and family life - Less time with other family - Past 12 mon | +0.02023 | Raises risk |
| ICS_30 | Rewarding aspect of caregiving experiences - Past 12 months | -0.00477 | Lowers risk |
| RPE_10 | Respondent provided emotional support - Past 12 months | +0.00070 | Raises risk |
| SEX | Sex of respondent | +0.00406 | Raises risk |
| ARS_20 | Helped primary care receiver, scheduling - Frequency | -0.00422 | Lowers risk |
| CRRCPAGR | Age of primary care receiver | -0.00537 | Lowers risk |
| PRG10GR | Relationship with primary care receiver | +0.00100 | Raises risk |
| ARS_10 | Helped primary care receiver, scheduling - Indicator - Past 12 mo | +0.00127 | Raises risk |
| APR_40 | Respondent provided help with personal care - Past 12 months | -0.00251 | Lowers risk |
| HRA_10 | Primary care rece. helped by professionals - Avg. hours per week | +0.00163 | Raises risk |
| PRP_15 | Severity of primary care receiver's health condition | -0.00350 | Lowers risk |
| ARM_20 | Helped primary care receiver, medical treatment - Frequency | +0.00052 | Raises risk |


## 6. Burnout Indicator Prevalence

% of respondents reporting strain for each of the 12 burnout items.  
Note: burnout_raw has been recoded to 0-1 (0=no strain, 1=max strain). For binary items, 1.0 = Yes/strain present. For ICS_40, values >= 0.667 correspond to 'Very stressful' or 'Stressful' on the original 4-point scale.

| Variable | Burnout Indicator | % Reporting Strain | Scale | n answered |
| --- | --- | --- | --- | --- |
| CRH_20 | Felt resentment or frustration toward care receiver | 64.1% | Yes (strain present) | 5,564 |
| CRH_10 | Relationship with care receiver became harder | 59.7% | Yes (strain present) | 5,570 |
| FIS_10D | Limited ability to do things as a family | 56.5% | Yes (strain present) | 4,225 |
| FIS_10A | Reduced time for other family members | 54.5% | Yes (strain present) | 4,225 |
| FIS_10H | Reduced caregiver's social activities | 51.1% | Yes (strain present) | 4,225 |
| FIS_10B | Caused family conflict or tension | 48.1% | Yes (strain present) | 4,225 |
| CRH_30 | Considered stopping or reducing caregiving | 42.3% | Yes (strain present) | 5,490 |
| FIS_10G | Affected caregiver's own health | 40.5% | Yes (strain present) | 4,225 |
| FIS_10C | Other family members took on extra duties | 34.7% | Yes (strain present) | 4,225 |
| FIS_10E | Caused financial difficulties for family | 30.7% | Yes (strain present) | 4,225 |
| FIS_10F | Affected family members' school or work | 28.7% | Yes (strain present) | 4,225 |
| ICS_40 | Overall caregiving is Very/Somewhat stressful | 14.0% | Very or Stressful (top 2 of 4) | 5,619 |


## 7. Caregiver Workload — Activity Breadth

Number of activity types each caregiver is responsible for (max 7).

- Respondents with at least one activity type: **5,516**
- Average types per active caregiver: **2.4**
- Juggling 3+ types: **1,673** (29.7%)

| Activity Types | Respondents | High Burnout % |
| --- | --- | --- |
| 0 | 117 | 18.8% |
| 1 | 1,451 | 50.3% |
| 2 | 2,392 | 44.6% |
| 3 | 589 | 46.9% |
| 4 | 464 | 65.1% |
| 5 | 358 | 69.6% |
| 6 | 198 | 83.8% |
| 7 | 64 | 81.2% |

### Activity Type Breakdown

| Activity Type | Caregivers Involved | % of Respondents | High Burnout Rate |
| --- | --- | --- | --- |
| Other assistance | 5,225 | 92.8% | 53.2% |
| Vision / hearing assistance | 3,657 | 64.9% | 47.6% |
| Social / recreational activities | 1,112 | 19.7% | 71.8% |
| House maintenance | 1,025 | 18.2% | 55.7% |
| Behavioural support | 879 | 15.6% | 70.3% |
| Personal care | 693 | 12.3% | 73.2% |
| Medical treatments | 693 | 12.3% | 70.6% |


## 8. Financial Burden on Caregivers

Out-of-pocket expenses paid on behalf of the care receiver.

**Important:** Expense variables are coded as ordinal brackets (1='Less than $200', 2='$200-$499', ... 6='$5,000+'), NOT actual dollar amounts. We report the distribution across brackets.

- Respondents who answered at least one expense question: **3,459** (61.4% of caregivers with burnout data)
- The remaining 2,174 had 'Valid skip' on all expense items (likely not asked due to module routing).

### Home modifications (`HOME_EXP`) -- n=550 respondents answered

| Expense Range | Respondents | % of Answers | High Burnout % |
| --- | --- | --- | --- |
| Less than $200 | 129 | 23.5% | 72.1% |
| $200 to $499 | 135 | 24.5% | 69.6% |
| $500 to $999 | 102 | 18.5% | 74.5% |
| $1,000 to $1,999 | 89 | 16.2% | 76.4% |
| $2,000 to $4,999 | 95 | 17.3% | 84.2% |

### Health care costs (`HLTH_EXP`) -- n=578 respondents answered

| Expense Range | Respondents | % of Answers | High Burnout % |
| --- | --- | --- | --- |
| Less than $200 | 143 | 24.7% | 71.3% |
| $200 to $499 | 147 | 25.4% | 68.0% |
| $500 to $999 | 100 | 17.3% | 77.0% |
| $1,000 to $1,999 | 98 | 17.0% | 81.6% |
| $2,000 to $4,999 | 90 | 15.6% | 80.0% |

### Paid help / services (`HELP_EXP`) -- n=342 respondents answered

| Expense Range | Respondents | % of Answers | High Burnout % |
| --- | --- | --- | --- |
| Less than $200 | 93 | 27.2% | 74.2% |
| $200 to $499 | 63 | 18.4% | 74.6% |
| $500 to $999 | 64 | 18.7% | 71.9% |
| $1,000 to $1,999 | 57 | 16.7% | 87.7% |
| $2,000 to $4,999 | 65 | 19.0% | 78.5% |

### Transportation (`TRNS_EXP`) -- n=2,785 respondents answered

| Expense Range | Respondents | % of Answers | High Burnout % |
| --- | --- | --- | --- |
| Less than $200 | 753 | 27.0% | 49.0% |
| $200 to $499 | 794 | 28.5% | 59.2% |
| $500 to $999 | 536 | 19.2% | 65.7% |
| $1,000 to $1,999 | 423 | 15.2% | 72.3% |
| $2,000 to $4,999 | 279 | 10.0% | 77.1% |

### Assistive devices / aids (`AID_EXP`) -- n=787 respondents answered

| Expense Range | Respondents | % of Answers | High Burnout % |
| --- | --- | --- | --- |
| Less than $200 | 211 | 26.8% | 70.6% |
| $200 to $499 | 250 | 31.8% | 73.6% |
| $500 to $999 | 153 | 19.4% | 75.8% |
| $1,000 to $1,999 | 97 | 12.3% | 70.1% |
| $2,000 to $4,999 | 76 | 9.7% | 68.4% |

### Medications (`MED_EXP`) -- n=1,267 respondents answered

| Expense Range | Respondents | % of Answers | High Burnout % |
| --- | --- | --- | --- |
| Less than $200 | 413 | 32.6% | 68.8% |
| $200 to $499 | 342 | 27.0% | 74.0% |
| $500 to $999 | 204 | 16.1% | 71.1% |
| $1,000 to $1,999 | 193 | 15.2% | 73.1% |
| $2,000 to $4,999 | 115 | 9.1% | 73.9% |


## 9. Composite Score Analysis

Workplace flexibility (FWA module) and work-enabling circumstances (ITE module) by score value and associated burnout rate.

### Workplace Flexibility Score (FWA, -1 to +5)

Sum of 5 positive flexibility items minus 1 if career penalty for using flexibility.

- n=2,926 | mean=1.84 | median=2 | std=1.23

| Score | Respondents | High Burnout % |
| --- | --- | --- |
| -1 | 20 | 65.0% |
| 0 | 525 | 52.8% |
| 1 | 545 | 55.2% |
| 2 | 892 | 51.6% |
| 3 | 720 | 53.2% |
| 4 | 212 | 58.5% |
| 5 | 12 | 58.3% |

### Work-Enabling Circumstances Score (ITE, -1 to +5)

Sum of 5 enabling-support items minus 1 if already had to quit/reduce hours.

- n=68 | mean=0.88 | median=0 | std=1.64

| Score | Respondents | High Burnout % |
| --- | --- | --- |
| -1 | 16 | 87.5% |
| 0 | 19 | 78.9% |
| 1 | 10 | 90.0% |
| 2 | 11 | 100.0% |
| 3 | 4 | 100.0% |
| 4 | 8 | 100.0% |


## 10. Data Completeness — Top 20 Variables with Most Missing Values

High missingness is usually module-specific routing (valid skip), not refusal to answer.

| Variable | Description | % Missing |
| --- | --- | --- |
| ite_score | Work-Enabling Circumstances Score | 98.8% |
| NFA_30 | Ask for help | 97.2% |
| EDM_02 | Education - Full or part-time | 96.8% |
| CCL_20 | Took compassionate care leave for end-of-life care - Past 12 mont | 96.3% |
| ICE_50 | Caregiving responsibilities affected studies - Past 12 months | 96.0% |
| HELP_EXP | Estimate of expenses - Hiring people | 93.9% |
| PGW_20 | Primary caregiver worked 30 hours or more - Per week | 93.9% |
| CRGVAGGR | Age of respondent's primary caregiver (groups of 5). | 92.2% |
| DPA_10 | Hours of professional help provided to respondent - Per average w | 92.2% |
| HOME_EXP | Estimate of expenses - Home modifications | 90.2% |
| HLTH_EXP | Estimate of expenses - Professional services | 89.7% |
| PGN_25 | Sex of primary caregiver | 89.6% |
| PGW_10 | Employment status of primary caregiver | 89.6% |
| ICF2_340 | Financial hardship - File for bankruptcy | 88.0% |
| ICF2_330 | Financial hardship - Sell off assets | 87.9% |
| ICF2_290 | Financial hardship - Borrowed money from family or friends | 87.7% |
| PRW_20 | Primary care receiver worked 30 hours or more | 87.7% |
| ICF2_300 | Financial hardship - Loans from a bank or financial institution | 87.6% |
| ICF2_310 | Financial hardship - Use or defer savings | 87.3% |
| ICF2_320 | Financial hardship - Modify spending | 87.2% |

- Variables 1–50% missing: **0**
- Variables >50% missing (module-specific): **20**

---
_End of report._
