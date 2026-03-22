# =============================================================================
# pipeline.py — Caregiver Burnout Risk Screening Pipeline
# =============================================================================
# Purpose:
#   Identify high-risk caregiver groups and the factors that drive their burnout,
#   so government/nonprofits can allocate resources where they are most needed.
#
# Data source:
#   GSS Caregiving and Care Receiving 2018 (Statistics Canada)
#   SAS file: c32pumfm.sas7bdat
#
# Pipeline steps:
#   1.  Load raw data
#   2.  Explore: shape, column labels, dtypes, null counts, skip codes
#   3.  Clean: drop bootstrap weight columns (WTBS_)
#   4.  Reduce: drop irrelevant module columns (kept for reference; pipeline
#               uses main_clean directly for the Include Variables dataframe)
#   5.  Composite scores: flexibility_score and ite_score
#   6.  Include Variables dataframe (reserve codes → NaN)
#   7.  Define X (features) and Y (burnout target)
#   8.  XGBoost binary classifier (easy config block at top of section)
#   9.  SHAP values — global importance + beeswarm plot
#   10. K-Means clustering on SHAP profiles (auto-selects k)
#   11. Per-cluster reporting for stakeholder presentation
# =============================================================================

import pandas as pd
import numpy as np
import pyreadstat
import matplotlib.pyplot as plt


# =============================================================================
# SECTION 1 — Load Raw Data
# =============================================================================
# Loads the SAS PUMF (Public Use Microdata File) into a pandas DataFrame.
# main_meta contains column labels from the SAS file (useful for codebook lookup).

import os
SAS_FILE = os.path.join(os.path.dirname(__file__), 'c32pumfm.sas7bdat')
main, main_meta = pyreadstat.read_sas7bdat(SAS_FILE)


# =============================================================================
# SECTION 2 — Exploratory Data Analysis
# =============================================================================
# Quick overview of the dataset: dimensions, column labels, data types, and
# missing value counts.  In the original notebook these were run interactively;
# they are kept here for documentation purposes.

print('=== main.sas7bdat ===')
print(f'Shape: {main.shape}')
print('\nColumn labels (first 20):')
for col, label in list(main_meta.column_names_to_labels.items())[:20]:
    print(f'  {col}: {label}')
print(f'\ndtypes:\n{main.dtypes}')
print(f'\nNull counts:\n{main.isnull().sum()}')

# Summary statistics for numeric columns
print('\n=== main describe ===')
print(main.describe())


# =============================================================================
# SECTION 3 — Clean: Drop Bootstrap Weight Columns
# =============================================================================
# The WTBS_ columns are replicate bootstrap weights used for variance
# estimation.  They are not needed for modelling and add ~500 columns.
# All other columns are kept in main_clean.

main_clean = main.drop(columns=[c for c in main.columns if c.startswith('WTBS_')])

print(f'\nmain_clean shape: {main_clean.shape}')
# Verify no WTBS_ columns remain
assert not any(c.startswith('WTBS_') for c in main_clean.columns), 'WTBS_ columns still present!'


# =============================================================================
# SECTION 4 — Reduce: Drop Irrelevant Module Columns (reference only)
# =============================================================================
# NOTE: main_reduced is kept here for historical reference.
# The modelling pipeline (Sections 5–11) uses main_clean directly so that
# variables like AGEPRGR0, CARUNPAI, CARPAID, NFA_10/30, PGN_25, PGW_10/20,
# AGEBEG1C, and WGHT_PER — which are dropped here — remain available.

drop_exact = [
    'PUMFID', 'WGHT_PER', 'DV_PROXY', 'EQFLAG',
    'SURVMNTH', 'CXRFLAG', 'DDEV_FL', 'DDEX_FL',
]
drop_prefixes        = ['WTBS_']
drop_module_prefixes = ['CAR', 'PRA', 'ARE', 'HAR', 'NPA', 'RPA', 'NFA', 'PG', 'AG']

cols_to_drop = [
    col for col in main_clean.columns
    if col in drop_exact
    or any(col.startswith(p) for p in drop_prefixes)
    or any(col.startswith(p) for p in drop_module_prefixes)
]

main_reduced = main_clean.drop(columns=cols_to_drop)
print(f'\nmain_reduced shape (not used in modelling): {main_reduced.shape}')


# =============================================================================
# SECTION 5 — Composite Scores: flexibility_score and ite_score
# =============================================================================
# Reserve codes (valid skip + not stated) are treated as NaN throughout.
# Valid skip  (not applicable)  → 6, 96, 996, 9996, 99996, 999996
# Not stated  (refused/missing) → 9, 99, 999, 9999, 99999, 999999
#
# All FWA_*/ITE_* items are 1-digit binary: 1 = Yes, 2 = No.
# We recode: 1→1, 2→0, reserve→NaN  before computing composite scores.
#
# ── flexibility_score  (range: −1 to 5) ────────────────────────────────────
#   Measures how many flexible work options a caregiver has access to,
#   penalised if using flexibility hurts their career.
#
#   Positive items (have this option → +1 each):
#     FWA_132 – Can work from home
#     FWA_133 – Can vary start/end time
#     FWA_134 – Can take time off during day (make up later)
#     FWA_136 – Can reduce hours temporarily
#     FWA_137 – Has other flexibility arrangements
#
#   Negative item (career penalty → −1):
#     FWA_150 – Using flexibility hurts career advancement
#
# ── ite_score  (range: −1 to 5) ────────────────────────────────────────────
#   Measures how many circumstances would enable the respondent to stay
#   employed while caregiving, penalised if they already had to quit.
#
#   Positive items (this would help → +1 each):
#     ITE_30A – Flexible scheduling
#     ITE_30B – Work from home option
#     ITE_30C – Reduced hours with no penalty
#     ITE_30D – Access to eldercare near workplace
#     ITE_30E – Other support at work
#
#   Negative item (already quit/reduced work → −1):
#     ITE_10  – Had to quit or reduce hours because of caregiving

reserve_codes = [6, 9, 96, 99, 996, 999, 9996, 9999, 99996, 99999, 999996, 999999]

FWA_POS = ['FWA_132', 'FWA_133', 'FWA_134', 'FWA_136', 'FWA_137']
FWA_NEG = ['FWA_150']
ITE_POS = ['ITE_30A', 'ITE_30B', 'ITE_30C', 'ITE_30D', 'ITE_30E']
ITE_NEG = ['ITE_10']

# Start from main_clean so all variables are available
scores_base = main_clean.copy()

# Replace reserve codes with NaN, then recode Yes→1 / No→0
for col in FWA_POS + FWA_NEG + ITE_POS + ITE_NEG:
    scores_base[col] = scores_base[col].replace(reserve_codes, np.nan)
    scores_base[col] = scores_base[col].map({1: 1, 2: 0})  # anything else → NaN

# flexibility_score: positive options minus career-penalty flag
# min_count=1 means: returns NaN only if ALL positive items are missing
scores_base['flexibility_score'] = (
    scores_base[FWA_POS].sum(axis=1, min_count=1)
    - scores_base[FWA_NEG[0]].fillna(0)
)

# ite_score: enabling circumstances minus already-quit flag
scores_base['ite_score'] = (
    scores_base[ITE_POS].sum(axis=1, min_count=1)
    - scores_base[ITE_NEG[0]].fillna(0)
)

print('\n=== flexibility_score (range −1 to 5) ===')
print(scores_base['flexibility_score'].describe().round(2))
print('\n', scores_base['flexibility_score'].value_counts(dropna=False).sort_index())

print('\n=== ite_score (range −1 to 5) ===')
print(scores_base['ite_score'].describe().round(2))
print('\n', scores_base['ite_score'].value_counts(dropna=False).sort_index())


# =============================================================================
# SECTION 6 — Build Include Variables Dataframe
# =============================================================================
# Selects exactly the columns listed in the Include Variables specification,
# plus the two composite scores computed above.
#
# Why pull from main_clean (not main_reduced)?
#   Several Include Variables were dropped in Section 4 via prefix rules:
#   AGEPRGR0, CARUNPAI, CARPAID, NFA_10, NFA_30, PGN_25, PGW_10, PGW_20,
#   AGEBEG1C, WGHT_PER.  main_clean still has them all.
#
# Raw FWA_*/ITE_* items are NOT included — they are replaced by the composites.
#
# Reserve codes → NaN applied to all columns except the two composites
# (which were already cleaned in Section 5) and WGHT_PER (a continuous weight).

INCLUDE_VARS = [
    # ── Survey weight (NOT a predictor; used as sample_weight in XGBoost) ────
    'WGHT_PER',

    # ── Socio-demographic ────────────────────────────────────────────────────
    'SEX', 'MARSTAT', 'PHSDFLG', 'AGEPRGR0', 'SENFLAG',
    'LIVARR08', 'LIVARRSN', 'LUC_RST', 'PRV', 'NLC_100C',

    # ── Caregiving context ───────────────────────────────────────────────────
    'CARUNPAI', 'CARPAID', 'CRGVAGGR', 'DPA_10',
    'NFA_10', 'NFA_30', 'PGN_25', 'PGW_10', 'PGW_20',

    # ── Care recipient profile ───────────────────────────────────────────────
    'APR_10', 'APR_20', 'APR_30', 'APR_40', 'APR_50',
    'APR_60', 'APR_70', 'APR_80',
    'HAP_10C', 'PAR_10',

    # ── Caregiver stress / health ────────────────────────────────────────────
    'RRA_10C', 'RRA_20C', 'RRA_30C', 'DPR_10C', 'DPR_40C',
    'APX_10GR', 'APX_20GR', 'APX_30C', 'APX_50GR',

    # ── Employment / work situation ──────────────────────────────────────────
    'RPE_10', 'CRRCPAGR', 'PRN_25', 'PRG10GR', 'PRP10GR', 'PRP_15',
    'PRW_10', 'PRW_20', 'PRD_10', 'PRU_10', 'PRH_20',

    # ── Activities of daily living (ADL/IADL) ────────────────────────────────
    'ART_30', 'ART_40',
    'ARI_20', 'ARI_30', 'ARI_40',
    'ARO_10', 'ARO_20', 'ARO_30', 'ARO_40',
    'ARP_10', 'ARP_20', 'ARP_30', 'ARP_40',
    'ARM_10', 'ARM_20', 'ARM_30C', 'ARM_40',
    'ARS_10', 'ARS_20', 'ARS_30C', 'ARS_40',
    'ARB_10', 'ARB_20', 'ARB_30C', 'ARB_40',
    'ARV_10', 'ARV_40', 'ARX_10', 'ARX_40',

    # ── Coordination and discharge ───────────────────────────────────────────
    'CCP_20', 'DVCG120C',
    'RNA_10C', 'RNA_20C', 'RNA_30C', 'RNA_40C',

    # ── Health resource access ───────────────────────────────────────────────
    'HRA_10',
    'ACD_10', 'ACD_20', 'ACD_30', 'ACD_40', 'ACD_50',
    'ACD_60', 'ACD_70', 'ACD_80', 'ACD_90',

    # ── Other contextual ────────────────────────────────────────────────────
    'OAC_20', 'AGEBEG1C', 'CGE_150', 'CCL_20',

    # ── Informal care / financial ────────────────────────────────────────────
    'ICL_110', 'ICL_120', 'ICL_130', 'ICL_135', 'ICL_140',
    'ICL_150', 'ICL_154', 'ICL_180', 'ICL_210',
    'ICB_15', 'ICB_20', 'ICB_25', 'ICP_15', 'ICP_30',
    'ICS_20', 'ICS_30',                                  # NOT burnout items (stay in X)
    'HOME_EXP', 'HLTH_EXP', 'HELP_EXP', 'TRNS_EXP', 'AID_EXP', 'MED_EXP',
    'ICF2_290', 'ICF2_300', 'ICF2_310', 'ICF2_320', 'ICF2_330', 'ICF2_340',

    # ── Education / employment status ────────────────────────────────────────
    'EDM_02', 'ICE_50', 'COW_10', 'IPL_10',
    'UWS230GR', 'TOE_240', 'ITO_10', 'INE_10',

    # ── Income / identity ───────────────────────────────────────────────────
    'PTN_10', 'FAMINCG1', 'BPR_16', 'VISMIN', 'LAN_01',

    # ── Composite scores (computed in Section 5) ─────────────────────────────
    'flexibility_score', 'ite_score',
]

# Keep only columns that exist in scores_base (main_clean + composites)
available   = [c for c in INCLUDE_VARS if c in scores_base.columns]
missing_src = [c for c in INCLUDE_VARS if c not in scores_base.columns]
if missing_src:
    print(f'\nWARNING — not found in main_clean: {missing_src}')

include_df = scores_base[available].copy()

# Replace reserve codes with NaN for all non-composite, non-weight columns
non_composite = [c for c in available if c not in ('flexibility_score', 'ite_score', 'WGHT_PER')]
for col in non_composite:
    include_df[col] = include_df[col].replace(reserve_codes, np.nan)

print(f'\nInclude Variables dataframe shape: {include_df.shape}')
print(f'Columns present: {len(available)} | Missing from source: {len(missing_src)}')
print(include_df.head(3))


# =============================================================================
# SECTION 7 — Define X (Features) and Y (Burnout Target)
# =============================================================================
# Y — burnout_high (binary: 0 = low burnout, 1 = high burnout)
#   Derived from the mean of 12 validated burnout items, median-split.
#   ALL 12 items are cleaned with reserve_codes → NaN before averaging.
#
# Burnout items:
#   ICS_40        – Caregiver strain (overall)
#   FIS_10A–H     – Financial impact of caregiving (8 items)
#   CRH_10/20/30  – Caregiver relationship health (3 items)
#
# X — all Include Variables columns EXCEPT:
#   • WGHT_PER (survey weight, kept separately as W)
#   • Any burnout items that ended up in include_df (prevent data leakage)
#
# NOTE: ICS_20, ICS_30 are NOT burnout items — they are unmet need indicators
#       and remain in X.
#
# W — WGHT_PER, passed to XGBoost as sample_weight so the model respects the
#     complex survey design.  Set USE_SURVEY_WEIGHT=False in Section 8 to skip.

BURNOUT_ITEMS = [
    'ICS_40',
    'FIS_10A', 'FIS_10B', 'FIS_10C', 'FIS_10D',
    'FIS_10E', 'FIS_10F', 'FIS_10G', 'FIS_10H',
    'CRH_10', 'CRH_20', 'CRH_30',
]

# Compute burnout target from main_clean with all items fully cleaned
burnout_base       = main_clean[BURNOUT_ITEMS].copy()
burnout_base       = burnout_base.replace(reserve_codes, np.nan)
burnout_score_full = burnout_base.mean(axis=1)        # NaN if all items missing
median_cutoff      = burnout_score_full.median()
burnout_high_full  = (burnout_score_full >= median_cutoff).astype(int)

# Align include_df with burnout target; drop rows with no valid burnout score
valid_idx       = burnout_score_full.dropna().index
include_aligned = include_df.loc[valid_idx].copy()
y_score         = burnout_score_full.loc[valid_idx]
y_clf           = burnout_high_full.loc[valid_idx]

# Separate out survey weight; build final X
WEIGHT_COL     = 'WGHT_PER'
EXCLUDE_FROM_X = [WEIGHT_COL] + [c for c in BURNOUT_ITEMS if c in include_aligned.columns]

X     = include_aligned.drop(columns=[c for c in EXCLUDE_FROM_X if c in include_aligned.columns])
W     = include_aligned[WEIGHT_COL] if WEIGHT_COL in include_aligned.columns else None
y_reg = y_score    # continuous burnout score (for potential regression use)
# y_clf is already defined above (binary classification target)

print('=' * 55)
print(f'  X  (features)  : {X.shape[0]} rows × {X.shape[1]} columns')
print(f'  y_reg           : burnout_score  (continuous mean of 12 items)')
print(f'  y_clf           : burnout_high   (0/1, median split at {median_cutoff:.3f})')
print(f'  W               : WGHT_PER survey weight')
print('=' * 55)
print(f'\nClass balance (y_clf):\n{y_clf.value_counts().rename({0: "low burnout", 1: "high burnout"})}')
print(f'\nNo burnout items leaked into X: {[c for c in BURNOUT_ITEMS if c in X.columns]}')
print(f'\nX columns ({X.shape[1]} total):\n{list(X.columns)}')


# =============================================================================
# SECTION 8 — XGBoost Classifier
# =============================================================================
# XGBoost handles NaN natively — no imputation is needed.
# Adjust the CONFIG block below to tune the model.
#
# Key parameters:
#   n_estimators          – maximum trees (early stopping may use fewer)
#   max_depth             – controls overfitting; 4–8 is typical
#   learning_rate         – lower = more trees needed but better generalisation
#   subsample             – fraction of rows used per tree (prevents overfitting)
#   colsample_bytree      – fraction of features used per tree
#   min_child_weight      – minimum samples in a leaf (higher = more regularisation)
#   scale_pos_weight      – raise above 1 if classes are imbalanced
#   early_stopping_rounds – stops if no logloss improvement after N rounds
#   USE_SURVEY_WEIGHT     – pass WGHT_PER as sample_weight to XGBoost

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ── CONFIGURATION — edit this block to tune the model ────────────────────────
XGB_CONFIG = dict(
    n_estimators          = 500,
    max_depth             = 6,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 10,
    scale_pos_weight      = 1,      # raise (e.g. 2) if class imbalance is an issue
    eval_metric           = 'logloss',
    early_stopping_rounds = 30,
    random_state          = 42,
    n_jobs                = -1,
)
TEST_SIZE         = 0.20    # 80/20 train-test split
USE_SURVEY_WEIGHT = True    # set False to ignore WGHT_PER
# ─────────────────────────────────────────────────────────────────────────────

sample_weights = W.values if (USE_SURVEY_WEIGHT and W is not None) else None

# Stratified split keeps burnout class proportions equal in train and test
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y_clf,
    (sample_weights if sample_weights is not None else np.ones(len(y_clf))),
    test_size=TEST_SIZE,
    random_state=XGB_CONFIG['random_state'],
    stratify=y_clf,
)
w_train_fit = w_train if (USE_SURVEY_WEIGHT and W is not None) else None

model = XGBClassifier(**XGB_CONFIG)
model.fit(
    X_train, y_train,
    sample_weight = w_train_fit,
    eval_set      = [(X_test, y_test)],
    verbose       = False,
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc    = roc_auc_score(y_test, y_prob)

print('=' * 55)
print(f'  Best iteration : {model.best_iteration}')
print(f'  ROC-AUC        : {auc:.4f}')
print('=' * 55)
print(classification_report(y_test, y_pred, target_names=['low burnout', 'high burnout']))


# =============================================================================
# SECTION 9 — SHAP Values (Model Interpretability)
# =============================================================================
# SHAP (SHapley Additive exPlanations) assigns each feature a contribution
# score for each individual prediction.
#
# Why compute on the FULL X (not just the test set)?
#   We want a SHAP profile for every respondent so we can cluster ALL caregivers
#   by risk-driver pattern in Section 10.
#
# Outputs:
#   shap_df   – DataFrame (n_respondents × n_features) of SHAP values
#   Bar chart – top 20 features by mean |SHAP| (global importance)
#   Beeswarm  – shows feature direction and spread across respondents

import shap

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)          # shape: (n_samples, n_features)
shap_df     = pd.DataFrame(shap_values, index=X.index, columns=X.columns)

print(f'\nSHAP matrix shape: {shap_df.shape}  (rows=respondents, cols=features)')

# Global importance bar chart (top 20 features)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
mean_abs_shap.head(20).sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('Top 20 Global Risk Drivers for Caregiver Burnout')
plt.tight_layout()
plt.show()

# Beeswarm summary plot: each dot = one respondent, colour = feature value
shap.summary_plot(shap_values, X, max_display=20, show=True)


