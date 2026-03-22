"""
report.py -- Caregiver Burnout Risk Analysis Report
====================================================
Runs the full pipeline (no Streamlit) and writes a structured Markdown report
to both the terminal AND a file automatically.

Usage:
    python src/report.py

Output file is set by REPORT_OUTPUT_FILE below (default: results/analysis_report.md).
Progress messages go to the terminal only (stderr); the report goes to both.

Runtime: ~2-4 minutes (data loading + model training + SHAP computation).
"""

import os
import sys
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyreadstat
import shap
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# =============================================================================
# ── CONFIGURATION BLOCK ──────────────────────────────────────────────────────
# All user-tunable settings live here. Change these and re-run report.py.
# =============================================================================

# --- Output file ---
# The report is written to this path (relative to the repo root).
# The results/ folder will be created automatically if it doesn't exist.
# To disable file output and print to terminal only, set this to None.
REPORT_OUTPUT_FILE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "analysis_report.md")
)

# --- Train / test split ---
# TEST_SIZE: fraction of respondents held out for evaluation.
#   0.20 = 20% test, 80% train — standard for this dataset size (~10 k respondents).
#   Increase (e.g. 0.30) if you want a larger, more stable test set.
#   Decrease (e.g. 0.10) if you want more training data and are willing to accept
#   a noisier evaluation on the held-out set.
TEST_SIZE = 0.20

# RANDOM_STATE: seed for reproducibility.
#   Change this to verify that results are stable across different random splits.
#   If AUC changes substantially (>0.02) between seeds, the model is overfit or
#   the dataset is too small for the chosen complexity.
RANDOM_STATE = 42

# --- Burnout target construction ---
# BURNOUT_SPLIT: determines what fraction of respondents are labelled "High Burnout".
#   "median" (default) = 50/50 split at the median burnout score.
#     Pro: perfectly balanced classes, good for exploration.
#     Con: the label is relative — someone just above median may not be truly at risk.
#   "percentile_75" = top 25% are labelled High Burnout.
#     Pro: more focused on the genuinely high-burden tail.
#     Con: class imbalance (25/75); set scale_pos_weight=3 in XGB_CONFIG to compensate.
#   "fixed_2.0" = everyone with mean burnout score >= 2.0 is labelled High.
#     Pro: threshold is interpretable (score is on a 1–4 scale; 2.0 = "somewhat" burden).
#     Con: class balance depends on the data; check reported class split before trusting AUC.
BURNOUT_SPLIT = "median"

# --- XGBoost classifier ---
# Each parameter is explained below. These defaults are conservative and well-suited
# to survey microdata with ~10 k respondents, many missing values, and ~130 features.
XGB_CONFIG = dict(
    # n_estimators: maximum number of boosting rounds (trees to add sequentially).
    #   The model stops early if validation loss stops improving (see early_stopping_rounds).
    #   500 is a safe upper bound — actual trees used will be shown in model.best_iteration.
    #   Increase to 1000 if the model is still improving at round 500.
    n_estimators=500,

    # max_depth: maximum depth of each individual tree.
    #   Deeper trees = more complex, can capture interactions but risk overfitting.
    #   6 is the XGBoost default and works well for tabular survey data.
    #   Try 4 for a simpler, more interpretable model.
    #   Try 8 if AUC is below 0.65 and you suspect underfitting.
    max_depth=6,

    # learning_rate (eta): how much each new tree corrects the ensemble.
    #   Smaller = slower learning = less overfitting, but needs more n_estimators.
    #   0.05 with 500 rounds is a standard conservative combination.
    #   If training is too slow, raise to 0.1 and reduce n_estimators to 300.
    learning_rate=0.05,

    # subsample: fraction of the training rows sampled for each boosting round.
    #   0.8 = 80% of rows used per tree, which acts as stochastic regularisation.
    #   Reduces variance (overfitting). Keep between 0.6 and 1.0.
    subsample=0.8,

    # colsample_bytree: fraction of features sampled for each tree.
    #   0.8 = 80% of columns used per tree — reduces feature correlation between trees.
    #   Particularly useful here because many INCLUDE_VARS are correlated (e.g. all
    #   the ARP_* sub-items within Personal Care are likely highly correlated).
    #   Try 0.5 if you want to force the model to rely on diverse features.
    colsample_bytree=0.8,

    # min_child_weight: minimum sum of instance weights in a child node.
    #   Acts as a regulariser: the model won't split a node unless it contains
    #   at least this much total survey weight.
    #   10 prevents overfitting to small demographic subgroups.
    #   Lower (e.g. 5) allows finer splits; higher (e.g. 20) produces simpler trees.
    min_child_weight=10,

    # scale_pos_weight: ratio of negative to positive class weight.
    #   Use 1 when classes are balanced (i.e. BURNOUT_SPLIT = "median").
    #   If using BURNOUT_SPLIT = "percentile_75" (25% positive), set to 3
    #   to tell the model that missing a high-burnout case is 3x more costly
    #   than a false alarm.
    #   Formula: scale_pos_weight = n_negative / n_positive
    scale_pos_weight=1,

    # eval_metric: loss function monitored on the validation set for early stopping.
    #   "logloss" = log-loss / binary cross-entropy — standard for binary classification.
    #   "auc" is also valid but slightly less stable as an early-stopping signal.
    eval_metric="logloss",

    # early_stopping_rounds: stop training if validation metric doesn't improve
    #   for this many consecutive rounds. Prevents overfitting and speeds up training.
    #   30 rounds is conservative (roughly 6% of 500 max rounds).
    #   Lower to 15 for faster runs; raise to 50 if the model stops too early.
    early_stopping_rounds=30,

    # random_state: seed for reproducibility (same as RANDOM_STATE above).
    random_state=RANDOM_STATE,

    # n_jobs: CPU threads for parallelism. -1 = use all available cores.
    n_jobs=-1,
)

# --- SHAP computation ---
# SHAP_SAMPLE_SIZE: number of respondents to compute SHAP values for.
#   None = compute for ALL respondents (most accurate, but slow: ~2 min for 10 k rows).
#   Set to e.g. 3000 for a faster approximation.
#   SHAP values are used for the importance ranking in the report, not for prediction,
#   so a sample of 3000+ is usually sufficient for stable rankings.
SHAP_SAMPLE_SIZE = None


# =============================================================================
# PATHS
# =============================================================================
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
SAS_FILE    = os.path.join(SRC_DIR, "c32pumfm.sas7bdat")
CB_DIR      = os.path.normpath(os.path.join(SRC_DIR, "..", "gss32_simplified_formats"))
ANSWER_CATS = os.path.join(CB_DIR, "codebook_answer_categories.csv")
VAR_CSV     = os.path.join(CB_DIR, "codebook_variables.csv")


# =============================================================================
# BURNOUT TARGET ITEMS
# =============================================================================
# These 12 survey questions together form the burnout score.
# They were chosen because they cover three validated dimensions of caregiver strain:
#
#   1. Perceived overall stress of caregiving (ICS module)
#      ICS_40 — "In general, how stressful is your caregiving situation?"
#               (1=Very stressful … 4=Not at all stressful; reverse-coded in scoring)
#
#   2. Family interference (FIS module — Family Impact Scale)
#      FIS_10A–H — Did caregiving reduce time for family / cause conflict / affect health /
#                  limit family activities / cause financial strain / affect school or work /
#                  affect the caregiver's social life?
#                  (1=Yes, 2=No — all scored as 1=strain present)
#
#   3. Relationship strain with the care receiver (CRH module)
#      CRH_10 — Did your relationship with the care receiver become more difficult?
#      CRH_20 — Did you feel resentment or frustration toward the care receiver?
#      CRH_30 — Did you consider stopping or reducing caregiving?
#               (1=Yes, 2=No)
#
# Scoring: reserve codes (6, 9) → NaN; row mean across available items.
# A higher score = more strain. Respondents missing all 12 items are excluded.
BURNOUT_ITEMS = [
    "ICS_40",
    "FIS_10A", "FIS_10B", "FIS_10C", "FIS_10D",
    "FIS_10E", "FIS_10F", "FIS_10G", "FIS_10H",
    "CRH_10", "CRH_20", "CRH_30",
]

BURNOUT_LABELS = {
    "ICS_40":  "Overall caregiving is Very/Somewhat stressful",
    "FIS_10A": "Reduced time for other family members",
    "FIS_10B": "Caused family conflict or tension",
    "FIS_10C": "Other family members took on extra duties",
    "FIS_10D": "Limited ability to do things as a family",
    "FIS_10E": "Caused financial difficulties for family",
    "FIS_10F": "Affected family members' school or work",
    "FIS_10G": "Affected caregiver's own health",
    "FIS_10H": "Reduced caregiver's social activities",
    "CRH_10":  "Relationship with care receiver became harder",
    "CRH_20":  "Felt resentment or frustration toward care receiver",
    "CRH_30":  "Considered stopping or reducing caregiving",
}


# =============================================================================
# COMPOSITE SCORE ITEMS
# =============================================================================
# These items are aggregated into two composite scores BEFORE entering the model,
# instead of being used as raw binary items. This reduces dimensionality and
# produces a more interpretable, theoretically grounded predictor.

# Workplace Flexibility Score (flexibility_score, range -1 to +5)
# Rationale: workplace accommodation is one of the most actionable policy levers
# for caregiver retention. Lumping these 6 items into a single score lets the
# model treat "has access to flexible work" as a gradient rather than separate flags.
FWA_POS = [
    "FWA_132",   # Can work from home / telework
    "FWA_133",   # Can vary start and end times
    "FWA_134",   # Can take time off during the day and make it up later
    "FWA_136",   # Can temporarily reduce working hours
    "FWA_137",   # Has other flexible work arrangements
]
FWA_NEG = [
    "FWA_150",   # Using flexibility hurts career advancement (SUBTRACTS 1 from score)
                 # Captures whether flexibility exists in name only — if using it
                 # carries a career penalty, it effectively isn't available.
]

# Work-Enabling Circumstances Score (ite_score, range -1 to +5)
# Rationale: identifies what workplace supports would help caregivers stay employed.
# A low score means the caregiver has already lost employment capacity (ITE_10 = 1)
# and lacks the supports that would help. This is a strong proxy for role overload.
ITE_POS = [
    "ITE_30A",   # Flexible scheduling would help
    "ITE_30B",   # Option to work from home would help
    "ITE_30C",   # Reduced hours without pay penalty would help
    "ITE_30D",   # Access to eldercare near the workplace would help
    "ITE_30E",   # Other workplace support would help
]
ITE_NEG = [
    "ITE_10",    # Already had to quit or reduce hours because of caregiving (SUBTRACTS 1)
                 # This is the most direct measure of labour-force impact. If True,
                 # the caregiver has already crossed a threshold of role conflict.
]


# =============================================================================
# PREDICTOR VARIABLES (INCLUDE_VARS)
# =============================================================================
# This list defines every column that enters the model as a potential predictor (X).
# Variables are grouped by theme below, with rationale for inclusion.
#
# Design decisions:
#   - WGHT_PER is included here to keep the alignment logic clean, but is
#     extracted before fitting and passed as sample_weight instead.
#   - Raw FWA_* and ITE_* items are NOT included — they are replaced by
#     flexibility_score and ite_score to reduce collinearity and improve
#     interpretability in SHAP output.
#   - Burnout items (ICS_40, FIS_10*, CRH_*) are explicitly excluded from X
#     in the pipeline to prevent label leakage.
#   - XGBoost handles NaN natively, so no imputation is needed.
#     High missingness (module-specific routing) is expected and acceptable.
#
INCLUDE_VARS = [

    # ── Survey weight (not a predictor) ────────────────────────────────────
    # WGHT_PER: person-level survey weight.
    # Statistics Canada recommends using this for all person-level estimates.
    # It corrects for the complex survey design and non-response.
    "WGHT_PER",

    # ── Core demographics ───────────────────────────────────────────────────
    # Basic sociodemographic controls that also function as risk stratifiers.
    # Government resource allocation typically needs to be explainable by
    # demographic group, so these must be in the model.
    "SEX",       # Sex of respondent (1=Male, 2=Female)
    "MARSTAT",   # Marital status — a key social-support proxy: married/partnered
                 # caregivers have a second adult in the household; single caregivers bear all strain alone.
    "PHSDFLG",   # Physical health flag (respondent's own disability/limitation) —
                 # a caregiver with their own health issues is at much higher risk.
    "AGEPRGR0",  # Age group (grouped) — age interacts with everything: older caregivers
                 # are more likely to be caring for a spouse; younger ones may be sandwiched.
    "SENFLAG",   # Senior (65+) indicator — useful for targeting senior-specific programs.

    # ── Geography and community context ────────────────────────────────────
    # Geography shapes access to formal support services. Rural caregivers have
    # fewer alternatives and less respite availability.
    "LIVARR08",  # Living arrangement — whether caregiver lives with the care receiver.
                 # Co-residence dramatically increases daily caregiving burden.
    "LIVARRSN",  # Reason for living arrangement — captures whether co-residence is
                 # a choice or forced by circumstances (e.g. care receiver cannot live alone).
    "LUC_RST",   # Urban / rural classification — rural caregivers have fewer
                 # service options and longer travel to supports.
    "PRV",       # Province / territory — captures regional policy differences
                 # (program availability, funding levels) and cultural context.
    "NLC_100C",  # Neighbourhood characteristics — disadvantaged neighbourhoods
                 # have fewer community supports and higher caregiver isolation.

    # ── Caregiving intensity and context ───────────────────────────────────
    # These variables describe how much care is being provided and under what terms.
    "CARUNPAI",  # Whether the respondent provides unpaid care — core eligibility
                 # variable; distinguishes informal from paid caregivers.
    "CARPAID",   # Whether any paid care is also in place — if paid care exists,
                 # the burden on the informal caregiver may be partly offset.
    "CRGVAGGR",  # Age group of the care receiver — caring for a very elderly person
                 # (e.g. 85+) typically means higher dependency and medical complexity.
    "DPA_10",    # Daily hours spent on caregiving — the single most direct measure
                 # of caregiving intensity; higher hours = more role overload.

    # ── Formal care services and unmet needs ───────────────────────────────
    # Whether formal services are available and accessible. When formal services
    # are absent or unaffordable, all care falls on the informal caregiver.
    "NFA_10",    # Does the care receiver receive any formal home care services?
    "NFA_30",    # Are there unmet formal care needs?
    "PGN_25",    # Does the caregiver know about government programs available to them?
    "PGW_10",    # Is the caregiver on a wait-list for formal services?
    "PGW_20",    # How long has the caregiver been waiting? (wait-list duration)

    # ── Care receiver activity limitations (ADLs / IADLs) ──────────────────
    # The APR module asks whether the care receiver needs help with specific activities.
    # Severity and breadth of limitations drive caregiving demand — more limitations
    # mean more tasks and higher emotional burden.
    "APR_10",    # Needs help with meal preparation / cooking
    "APR_20",    # Needs help with getting around inside the home
    "APR_30",    # Needs help with personal hygiene / bathing / dressing
    "APR_40",    # Needs help with taking medications
    "APR_50",    # Needs help with banking / finances
    "APR_60",    # Needs help with transportation / getting around outside
    "APR_70",    # Needs help with housework
    "APR_80",    # Needs help with other activities

    # ── Caregiver's own health and participation ────────────────────────────
    # A caregiver in poor health is both more susceptible to burnout and less
    # able to cope. Social participation acts as a protective buffer.
    "HAP_10C",   # Caregiver's general health (self-rated) — one of the strongest
                 # predictors of overall wellbeing and burnout resilience.
    "PAR_10",    # Participation in social / recreational activities — a proxy for
                 # whether the caregiver has preserved any time for themselves.

    # ── Relationship with care receiver ────────────────────────────────────
    # The emotional quality of the caregiving relationship is a major burnout pathway.
    # Even if caregiving is manageable in hours, a difficult relationship amplifies strain.
    "RRA_10C",   # How close is the relationship between caregiver and care receiver?
    "RRA_20C",   # Has the relationship changed since caregiving began?
    "RRA_30C",   # Direction of relationship change (closer / more distant)
    "DPR_10C",   # Does the care receiver appreciate the care provided?
    "DPR_40C",   # Does the care receiver interfere with or resist care?

    # ── Hours of care and changes over time ────────────────────────────────
    # Captures not just how much the caregiver does, but whether the burden is growing.
    "APX_10GR",  # Average hours per week providing care (grouped)
    "APX_20GR",  # Has the amount of care provided changed over the past year?
    "APX_30C",   # Reason for change in hours — voluntary vs. forced increase
    "APX_50GR",  # Months providing this level of care (duration)

    # ── Employment status and work-caregiving conflict ──────────────────────
    # The intersection of employment and caregiving is the most common burnout pathway.
    # Caregivers who are also employed face dual demands with no natural release valve.
    "RPE_10",    # Is the respondent currently employed?
    "CRRCPAGR",  # Number of caregiving-related changes to paid work
    "PRN_25",    # Has caregiving affected ability to do paid work?
    "PRG10GR",   # Number of days of paid work missed due to caregiving
    "PRP10GR",   # Number of days working from home due to caregiving
    "PRP_15",    # Had to reduce hours of paid work due to caregiving
    "PRW_10",    # Had to pass up a promotion or advancement opportunity
    "PRW_20",    # Had to turn down or quit a job due to caregiving
    "PRD_10",    # Had to change career plans or goals due to caregiving
    "PRU_10",    # Had to take unpaid leave due to caregiving
    "PRH_20",    # Had to use vacation or personal time for caregiving

    # ── Transportation assistance provided ─────────────────────────────────
    # Driving / escorting the care receiver to appointments is time-consuming
    # and often falls entirely on one family member.
    "ART_30",    # Provides transportation to medical appointments
    "ART_40",    # Provides transportation to other destinations

    # ── Instrumental domestic activities ───────────────────────────────────
    # Housekeeping and home management on behalf of care receiver.
    "ARI_20",    # Does grocery shopping / errands for care receiver
    "ARI_30",    # Does housework / laundry for care receiver
    "ARI_40",    # Manages finances / paperwork for care receiver

    # ── Activity gateway flags + sub-items (7 domains) ─────────────────────
    # For each domain, the gateway (e.g. ARO_10) asks "do you help with this type?"
    # Sub-items (ARO_20, ARO_30, ARO_40) capture specific tasks within that domain.
    # Including both gateway and sub-items lets the model detect whether it's the
    # sheer breadth vs. specific task types that drive burnout.

    # Outdoor / home maintenance
    "ARO_10",    # Gateway: helps with outdoor activities / home maintenance
    "ARO_20",    # Specific: yard work / gardening
    "ARO_30",    # Specific: home repairs / maintenance
    "ARO_40",    # Specific: other outdoor activities

    # Personal care (ADLs — Activities of Daily Living)
    # Personal care is the most physically and emotionally demanding domain.
    # Respondents who provide intimate personal care have substantially higher burnout.
    "ARP_10",    # Gateway: helps with personal care
    "ARP_20",    # Specific: bathing / showering
    "ARP_30",    # Specific: dressing / grooming
    "ARP_40",    # Specific: toileting / incontinence care

    # Medical / nursing treatments
    # Quasi-medical tasks (wound care, medication management) typically require
    # training that informal caregivers rarely receive, creating high anxiety.
    "ARM_10",    # Gateway: helps with medical treatments
    "ARM_20",    # Specific: wound care / dressings
    "ARM_30C",   # Specific: medication management / injections
    "ARM_40",    # Specific: other medical treatments

    # Social and recreational
    # Maintaining the care receiver's social life is often overlooked but is a
    # substantial time commitment, especially for cognitively impaired receivers.
    "ARS_10",    # Gateway: helps with social / recreational activities
    "ARS_20",    # Specific: accompanying to social events
    "ARS_30C",   # Specific: providing companionship
    "ARS_40",    # Specific: other social support

    # Behavioural support (high burnout risk domain)
    # Managing challenging behaviour (wandering, aggression, agitation) is one of
    # the most exhausting caregiving tasks and strongly predicts caregiver breakdown.
    "ARB_10",    # Gateway: helps with behavioural issues
    "ARB_20",    # Specific: managing challenging behaviour
    "ARB_30C",   # Specific: supervision to prevent harm
    "ARB_40",    # Specific: other behavioural support

    # Vision / hearing assistance
    "ARV_10",    # Gateway: helps with sensory impairment-related needs
    "ARV_40",    # Specific: other vision/hearing assistance

    # Other assistance
    "ARX_10",    # Gateway: provides other types of assistance
    "ARX_40",    # Specific: details of other assistance

    # ── Care situation changes and history ──────────────────────────────────
    "CCP_20",    # Has care receiver's condition changed recently?
                 # Sudden deterioration is a major burnout trigger.
    "DVCG120C",  # Duration of caregiving (months/years) — longer-term caregivers
                 # have often depleted their initial reserves of resilience.

    # ── Respite and time off from caregiving ───────────────────────────────
    # Respite — regular breaks from caregiving — is one of the most evidence-based
    # interventions for preventing caregiver burnout. Absence of respite is a
    # strong modifiable risk factor that programs can directly address.
    "RNA_10C",   # Does the caregiver get any time off from caregiving?
    "RNA_20C",   # How many hours of respite per week?
    "RNA_30C",   # Source of respite (family, formal services, etc.)
    "RNA_40C",   # Is the amount of respite adequate?

    # ── Support received by the caregiver ──────────────────────────────────
    "HRA_10",    # Does the caregiver themselves receive any help or support?
                 # Caregivers who are also care receivers face compounded burden.

    # ── Care receiver's conditions and diagnoses ───────────────────────────
    # The nature of the care receiver's health conditions affects caregiving complexity.
    # Dementia in particular is associated with the highest caregiver burnout rates
    # because of the unpredictability, behavioural symptoms, and lack of reciprocity.
    "ACD_10",    # Care receiver has Alzheimer's disease or dementia
    "ACD_20",    # Care receiver has a physical disability
    "ACD_30",    # Care receiver has a chronic health condition
    "ACD_40",    # Care receiver has mental health condition
    "ACD_50",    # Care receiver has an addiction
    "ACD_60",    # Care receiver has developmental disability
    "ACD_70",    # Care receiver has cancer
    "ACD_80",    # Care receiver had a stroke
    "ACD_90",    # Care receiver has another condition

    # ── Other contextual variables ──────────────────────────────────────────
    "OAC_20",    # Are there other informal caregivers helping? (shared vs. sole caregiver)
                 # Being the sole caregiver dramatically increases burden.
    "AGEBEG1C",  # Age when caregiving began — starting caregiving at a younger age
                 # accumulates more lifetime burden.
    "CGE_150",   # Does the caregiver live with the care receiver full-time?
    "CCL_20",    # Has caregiving led to a change in the caregiver's living arrangement?

    # ── Economic impact on the caregiver (income loss) ─────────────────────
    # The ICL module measures whether caregiving has reduced the caregiver's income.
    # Financial stress independently predicts burnout and also limits ability to
    # purchase relief (paid help, respite care, etc.).
    "ICL_110",   # Has caregiving reduced employment income?
    "ICL_120",   # Estimated annual income loss from caregiving
    "ICL_130",   # Has caregiving affected pension / retirement savings?
    "ICL_135",   # Has caregiving affected investments?
    "ICL_140",   # Has caregiving affected career advancement?
    "ICL_150",   # Has caregiving affected ability to save for the future?
    "ICL_154",   # Has caregiving led to drawing down savings?
    "ICL_180",   # Has caregiving created personal debt?
    "ICL_210",   # Total estimated lifetime income loss due to caregiving

    # ── Affordability of services and out-of-pocket burden ─────────────────
    "ICB_15",    # Cannot afford needed home care services
    "ICB_20",    # Cannot afford needed medical equipment / aids
    "ICB_25",    # Has had to forgo necessary care due to cost
    "ICP_15",    # Out-of-pocket caregiving expenses are a financial strain
    "ICP_30",    # Overall financial impact of caregiving on household

    # ── Caregiver emotional support and stress (non-burnout items) ─────────
    # These two items are NOT part of the burnout score but measure related constructs.
    # ICS_20 measures perceived social support; ICS_30 measures felt stress.
    # Including them lets the model detect whether having support networks is protective.
    "ICS_20",    # Does the caregiver feel supported by family / friends / community?
    "ICS_30",    # Does the caregiver feel emotionally overwhelmed or stressed?

    # ── Out-of-pocket expense amounts ($) ──────────────────────────────────
    # Dollar amounts the caregiver has spent on behalf of the care receiver.
    # These are continuous variables — higher values indicate heavier financial burden.
    "HOME_EXP",  # Home modifications (ramps, grab bars, widened doors)
    "HLTH_EXP",  # Health care costs (private clinics, physiotherapy, etc.)
    "HELP_EXP",  # Paid help / home care services (personal support workers)
    "TRNS_EXP",  # Transportation (taxi, medical transport, adapted transit)
    "AID_EXP",   # Assistive devices and aids (wheelchairs, walkers, etc.)
    "MED_EXP",   # Medications (not covered by provincial drug plans)

    # ── Formal support received by caregiver ───────────────────────────────
    # Whether the caregiver accesses formal support programs. Receiving formal support
    # may be protective (burden sharing) or a marker of high burden (sought help because
    # situation was dire). SHAP will reveal which direction dominates.
    "ICF2_290",  # Receives respite care or short-term relief services
    "ICF2_300",  # Receives training or education for caregiving tasks
    "ICF2_310",  # Receives counselling or mental health support
    "ICF2_320",  # Receives support from a caregiver support group
    "ICF2_330",  # Receives financial assistance or benefits
    "ICF2_340",  # Receives other formal supports

    # ── Employment and socioeconomic status ────────────────────────────────
    # SES variables that shape both caregiving context and access to resources.
    "EDM_02",    # Highest educational attainment — affects knowledge of services
                 # and ability to navigate the health system.
    "ICE_50",    # Employment certainty / job security — precarious workers cannot
                 # take unpaid leave or request flexibility without risk.
    "COW_10",    # Class of worker (employee vs. self-employed) — self-employed
                 # caregivers have no employer-provided benefits but more schedule control.
    "IPL_10",    # Industry / sector of employment — some sectors (health, education)
                 # are more caregiving-friendly than others.

    # ── Work schedule characteristics ──────────────────────────────────────
    # Schedule rigidity is a key moderator of work-caregiving conflict.
    "UWS230GR",  # Usual hours worked per week (grouped) — part-time workers may
                 # have more caregiving capacity but also less income security.
    "TOE_240",   # Type of work schedule (regular / shift / irregular)
    "ITO_10",    # Works irregular or on-call hours — unpredictable schedules
                 # make it impossible to arrange consistent caregiving coverage.
    "INE_10",    # Does not work evenings / nights — evening availability is often
                 # critical for care receiver supervision.

    # ── Social resources and systemic factors ──────────────────────────────
    # Captures social capital, equity dimensions, and language barriers.
    "PTN_10",    # Has a partner or spouse — partnered caregivers have a built-in
                 # source of emotional support and potential task sharing.
    "FAMINCG1",  # Household income group — income determines ability to purchase
                 # formal care, take unpaid leave, and absorb expenses.
    "BPR_16",    # Place of birth (Canada vs. abroad) — immigrant caregivers often
                 # face language barriers and exclusion from formal support networks.
    "VISMIN",    # Visible minority status — systemic barriers to health system access
                 # and cultural norms around family caregiving may amplify burden.
    "LAN_01",    # First official language — language is a key access barrier for
                 # services, especially in predominantly English or French regions.

    # ── Composite scores (constructed above; replace raw FWA_*/ITE_* items) ──
    "flexibility_score",  # Workplace Flexibility Score (FWA module, -1 to +5)
    "ite_score",          # Work-Enabling Circumstances Score (ITE module, -1 to +5)
]


# =============================================================================
# CODED VARIABLE LOOKUP TABLES
# =============================================================================
# CRITICAL: most variables in this survey are CODED CATEGORIES, not continuous
# numbers. The pipeline replaces reserve codes with NaN, but the remaining
# codes (1, 2, 3 ...) represent discrete categories or ordinal brackets.
# XGBoost handles this fine (tree splits on thresholds), but descriptive
# statistics must decode the categories rather than computing means on codes.

# Expense variables: codes 1-6 represent dollar RANGES, not actual dollars.
EXPENSE_RANGES = {
    1: "Less than $200",
    2: "$200 to $499",
    3: "$500 to $999",
    4: "$1,000 to $1,999",
    5: "$2,000 to $4,999",
    6: "$5,000 or more",
}

# Hours-of-care bands (HAP_10C, DPA_10, HRA_10):
HOURS_BANDS = {
    0: "No hours",
    1: "Less than 10 hours",
    2: "10-19 hours",
    3: "20-29 hours",
    4: "30-39 hours",
    5: "40-49 hours",
    6: "50+ hours",
}

# Income groups (FAMINCG1): codes 1-8 are brackets from tax-linked data.
INCOME_BRACKETS = {
    1: "Less than $20,000",
    2: "$20,000 to $39,999",
    3: "$40,000 to $59,999",
    4: "$60,000 to $79,999",
    5: "$80,000 to $99,999",
    6: "$100,000 to $119,999",
    7: "$120,000 to $139,999",
    8: "$140,000 or more",
}


# =============================================================================
# ACTIVITY AND EXPENSE DISPLAY CONFIG
# =============================================================================
ACTIVITY_GATEWAY = {
    "ARO_10": "House maintenance",
    "ARP_10": "Personal care",
    "ARM_10": "Medical treatments",
    "ARS_10": "Social / recreational activities",
    "ARB_10": "Behavioural support",
    "ARV_10": "Vision / hearing assistance",
    "ARX_10": "Other assistance",
}

EXPENSE_COLS = {
    "HOME_EXP": "Home modifications",
    "HLTH_EXP": "Health care costs",
    "HELP_EXP": "Paid help / services",
    "TRNS_EXP": "Transportation",
    "AID_EXP":  "Assistive devices / aids",
    "MED_EXP":  "Medications",
}

DEMO_VARS = {
    "SEX":      "Sex",
    "PRV":      "Province / Territory",
    "FAMINCG1": "Household Income Group",
    "MARSTAT":  "Marital Status",
    "VISMIN":   "Visible Minority Status",
    "SENFLAG":  "Senior (65+) Flag",
}

# Reserve-code answer labels — these map to NaN in the cleaned data
# so we exclude them from label lookups to avoid showing "Valid skip" in charts.
_RESERVE_LABELS = {"Valid skip", "Don't know", "Refusal", "Not stated"}

# Statistics Canada reserve codes (Valid skip, Not stated) by number of digits.
# 1-digit variables use 6/9; 2-digit use 96/99; 3-digit use 996/999; etc.
RESERVE_CODES = [
    6, 9, 96, 99, 996, 999,
    9996, 9999, 99996, 99999, 999996, 999999,
]


# =============================================================================
# TEE: write to file AND terminal simultaneously
# =============================================================================
class _Tee:
    """Wraps two streams so that print() goes to both at once."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# =============================================================================
# PROGRESS AND FORMATTING HELPERS
# =============================================================================
def _progress(msg):
    """Print a progress message to stderr (terminal only, not in the report file)."""
    print(f"[INFO] {msg}", file=sys.stderr, flush=True)


def _heading(text, level=2):
    prefix = "#" * level
    print(f"\n{prefix} {text}\n")


def _table(rows: list[dict], col_order: list[str] | None = None):
    """Print a Markdown table from a list of dicts."""
    if not rows:
        print("_No data._\n")
        return
    cols = col_order or list(rows[0].keys())
    print(f"| {' | '.join(cols)} |")
    print(f"| {' | '.join(['---'] * len(cols))} |")
    for row in rows:
        cells = " | ".join(str(row.get(c, "")) for c in cols)
        print(f"| {cells} |")
    print()


# =============================================================================
# LOAD CODEBOOK HELPERS
# =============================================================================
def load_label_map():
    """Build {variable: {code: label}} from the codebook answer-categories CSV."""
    _progress("Loading answer labels from codebook...")
    cats = pd.read_csv(ANSWER_CATS)
    cats["label"] = cats["label"].str.strip().str.replace(r"^Y\s+es$", "Yes", regex=True)
    cats = cats[~cats["label"].isin(_RESERVE_LABELS)]
    label_map: dict[str, dict[int, str]] = {}
    for _, row in cats.iterrows():
        var = row["variable_name"]
        try:
            code = int(float(row["code"]))
        except (ValueError, TypeError):
            continue
        label_map.setdefault(var, {})[code] = str(row["label"])
    return label_map


def load_var_info():
    """Build {variable: description} from the codebook variables CSV."""
    _progress("Loading variable descriptions...")
    df = pd.read_csv(VAR_CSV, usecols=["variable_name", "concept", "question_text"])
    result = {}
    for _, row in df.iterrows():
        desc = row["concept"]
        if pd.isna(desc) or str(desc).strip() == "":
            desc = str(row["question_text"])[:80] if pd.notna(row["question_text"]) else row["variable_name"]
        result[row["variable_name"]] = str(desc).strip()
    result["flexibility_score"] = "Workplace Flexibility Score"
    result["ite_score"]         = "Work-Enabling Circumstances Score"
    return result


def decode_col(series, label_map, var_name):
    mapping = label_map.get(var_name, {})
    if not mapping:
        return series
    def _f(x):
        if pd.isna(x):
            return np.nan
        try:
            return mapping.get(int(x), x)
        except (ValueError, TypeError):
            return x
    return series.map(_f)


# =============================================================================
# PIPELINE
# =============================================================================
def run_pipeline(label_map):
    _progress("Loading SAS microdata file (~30 s)...")
    main, _ = pyreadstat.read_sas7bdat(SAS_FILE)

    main_clean = main.drop(columns=[c for c in main.columns if c.startswith("WTBS_")])
    _progress(f"Loaded: {main_clean.shape[0]:,} respondents, {main_clean.shape[1]:,} columns.")

    # --- Build composite scores ---
    scores_base = main_clean.copy()
    for col in FWA_POS + FWA_NEG + ITE_POS + ITE_NEG:
        if col in scores_base.columns:
            scores_base[col] = scores_base[col].replace(RESERVE_CODES, np.nan)
            scores_base[col] = scores_base[col].map({1: 1, 2: 0})

    scores_base["flexibility_score"] = (
        scores_base[FWA_POS].sum(axis=1, min_count=1)
        - scores_base[FWA_NEG[0]].fillna(0)
    )
    scores_base["ite_score"] = (
        scores_base[ITE_POS].sum(axis=1, min_count=1)
        - scores_base[ITE_NEG[0]].fillna(0)
    )

    # --- Assemble predictor table; replace reserve codes with NaN ---
    available  = [c for c in INCLUDE_VARS if c in scores_base.columns]
    include_df = scores_base[available].copy()
    for col in available:
        if col not in ("flexibility_score", "ite_score", "WGHT_PER"):
            include_df[col] = include_df[col].replace(RESERVE_CODES, np.nan)

    # --- Compute burnout target ---
    # CRITICAL: all items are coded so that LOWER raw codes = MORE strain.
    #   Binary items (FIS_10*, CRH_*):  1 = Yes (strain),  2 = No
    #   ICS_40 (4-point Likert):        1 = Very stressful, 4 = Not at all
    #
    # We recode everything to a 0-1 scale where 1.0 = maximum strain:
    #   Binary:  1 -> 1.0,  2 -> 0.0
    #   ICS_40:  1 -> 1.0,  2 -> 0.667,  3 -> 0.333,  4 -> 0.0
    #
    # This way burnout_score ranges from 0 (no strain) to 1 (maximum strain)
    # and burnout_high = (score >= median) correctly identifies the high-risk group.

    burnout_raw = main_clean[BURNOUT_ITEMS].replace(RESERVE_CODES, np.nan)

    for col in BURNOUT_ITEMS:
        if col not in burnout_raw.columns:
            continue
        if col == "ICS_40":
            # 4-point scale normalised to 0-1: Very stressful=1, Not at all=0
            burnout_raw[col] = burnout_raw[col].map({1: 1.0, 2: 2/3, 3: 1/3, 4: 0.0})
        else:
            # Binary: Yes(strain)=1, No=0
            burnout_raw[col] = burnout_raw[col].map({1: 1.0, 2: 0.0})

    burnout_score = burnout_raw.mean(axis=1)

    # Apply the chosen split strategy
    if BURNOUT_SPLIT == "median":
        threshold = burnout_score.median()
    elif BURNOUT_SPLIT == "percentile_75":
        threshold = burnout_score.quantile(0.75)
    elif BURNOUT_SPLIT.startswith("fixed_"):
        threshold = float(BURNOUT_SPLIT.split("_")[1])
    else:
        raise ValueError(f"Unknown BURNOUT_SPLIT value: '{BURNOUT_SPLIT}'")

    burnout_high = (burnout_score >= threshold).astype(int)

    valid_idx       = burnout_score.dropna().index
    include_aligned = include_df.loc[valid_idx].copy()
    y_score         = burnout_score.loc[valid_idx]
    y_clf           = burnout_high.loc[valid_idx]

    exclude_x = ["WGHT_PER"] + [c for c in BURNOUT_ITEMS if c in include_aligned.columns]
    X = include_aligned.drop(columns=[c for c in exclude_x if c in include_aligned.columns])
    W = include_aligned.get("WGHT_PER", pd.Series(np.ones(len(include_aligned)),
                                                    index=include_aligned.index))

    analysis_df = X.copy()
    analysis_df["burnout_high"]  = y_clf
    analysis_df["burnout_score"] = y_score
    analysis_df["WGHT_PER"]      = W

    return dict(
        main_clean=main_clean,
        X=X, y_clf=y_clf, y_score=y_score, W=W,
        analysis_df=analysis_df,
        burnout_raw=burnout_raw,
        median_cutoff=threshold,
    )


def train_xgb(X, y_clf, W):
    _progress("Training XGBoost classifier (~30-60 s)...")
    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y_clf, W.values, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_clf,
    )
    model = XGBClassifier(**XGB_CONFIG)
    model.fit(X_train, y_train, sample_weight=w_train,
              eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    report = classification_report(
        y_test, y_pred,
        target_names=["Low Burnout", "High Burnout"],
        output_dict=True,
    )
    _progress(f"Model trained. Best iteration: {model.best_iteration}. "
              f"Test AUC: {auc:.4f}.")
    return model, auc, report, X_test, y_test, y_prob


def compute_shap(model, X):
    _progress("Computing SHAP values (~1-2 min)...")
    sample = X
    if SHAP_SAMPLE_SIZE is not None and len(X) > SHAP_SAMPLE_SIZE:
        sample = X.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    return pd.DataFrame(shap_values, index=sample.index, columns=sample.columns)


# =============================================================================
# REPORT SECTIONS
# =============================================================================
def section_dataset(data):
    _heading("1. Dataset Overview", 2)
    adf   = data["analysis_df"]
    n     = len(adf)
    n_total = len(data["main_clean"])
    n_high = int(adf["burnout_high"].sum())
    n_low  = n - n_high
    bs    = adf["burnout_score"]

    print(f"- **Total survey respondents (GSS 2018):** {n_total:,}")
    print(f"- **Respondents routed to caregiving modules (with burnout data):** {n:,}")
    print(f"  - The remaining {n_total - n:,} respondents were not asked caregiving stress "
          f"questions (all 12 burnout items = 'Valid skip'). They are likely non-caregivers "
          f"or were routed to different survey modules.")
    print(f"- **Predictor features in X:** {data['X'].shape[1]}")
    print(f"- **Burnout score (0-1 scale, higher = more strain):** mean={bs.mean():.3f}, "
          f"median={bs.median():.3f}, std={bs.std():.3f}, range=[{bs.min():.3f}, {bs.max():.3f}]")
    print(f"- **Target split strategy:** `{BURNOUT_SPLIT}` at threshold={data['median_cutoff']:.3f}")
    print(f"- **High-burnout:** {n_high:,} ({n_high/n*100:.1f}%)")
    print(f"- **Low-burnout:** {n_low:,} ({n_low/n*100:.1f}%)")
    print(f"- **Survey weight range:** [{data['W'].min():.0f}, {data['W'].max():.0f}] "
          f"(passed as sample_weight during training)")
    print()
    print("**Note on variable coding:** Most variables in this survey are coded categories "
          "(e.g., income brackets, expense ranges, Yes/No flags) not continuous values. "
          "XGBoost handles ordinal codes correctly via threshold splits. Descriptive "
          "statistics below decode categories where possible.")
    print()


def section_demographics(data, label_map):
    _heading("2. Burnout Rate by Key Demographics", 2)
    print("% of respondents with high burnout within each group.\n")
    adf = data["analysis_df"]

    for var, display in DEMO_VARS.items():
        if var not in adf.columns:
            continue
        tmp = adf[[var, "burnout_high"]].dropna().copy()
        tmp["_label"] = decode_col(tmp[var], label_map, var)
        stats = (
            tmp.groupby("_label")["burnout_high"]
            .agg(rate="mean", count="count").reset_index()
        )
        stats["High Burnout %"] = (stats["rate"] * 100).round(1)
        stats = stats.sort_values("High Burnout %", ascending=False)
        print(f"### {display} (`{var}`)\n")
        _table(
            [{"Category": r["_label"], "n": f"{int(r['count']):,}",
              "High Burnout %": f"{r['High Burnout %']:.1f}%"}
             for _, r in stats.iterrows()],
            ["Category", "n", "High Burnout %"]
        )


def section_correlations(data, var_info):
    _heading("3. Top 30 Variables by Pearson Correlation with Burnout Score", 2)
    print("Pearson correlation with the 0-1 burnout score (higher = more strain).\n")
    print("**Important caveats:**")
    print("- Most predictor variables are ordinal codes, not continuous measurements. "
          "Pearson r captures monotonic trends in the codes but assumes equal intervals.")
    print("- For binary Yes(1)/No(2) variables with reserve codes replaced by NaN, "
          "a negative r means Yes is associated with higher burnout (because code 1 < 2, "
          "while burnout score is recoded so higher = more strain).")
    print("- Correlations are computed on the raw survey codes in X (not recoded). "
          "The SHAP analysis (Section 5) is more reliable for identifying risk factors.\n")
    adf = data["analysis_df"]
    num = [c for c in adf.select_dtypes(include=[np.number]).columns
           if c not in ("burnout_high", "WGHT_PER")]
    corr = adf[num].corrwith(adf["burnout_score"]).dropna()
    top  = corr.abs().sort_values(ascending=False).head(30)
    rows = [
        {"Rank": i+1, "Variable": c, "Description": var_info.get(c, c)[:65],
         "Pearson r": f"{corr[c]:+.4f}",
         "Direction": "More -> more burnout" if corr[c] > 0 else "More -> less burnout"}
        for i, c in enumerate(top.index)
    ]
    _table(rows, ["Rank", "Variable", "Description", "Pearson r", "Direction"])


def section_model(auc, report, model):
    _heading("4. XGBoost Model Performance", 2)
    print(f"Train / test split: {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%  ")
    print(f"Trees used (early stopping): {model.best_iteration}  ")
    print(f"Target split: `{BURNOUT_SPLIT}`\n")
    print(f"- **ROC-AUC:** {auc:.4f}  (0.5 = random, 1.0 = perfect; >0.70 useful for screening)")
    for cls_name in ["Low Burnout", "High Burnout"]:
        r = report[cls_name]
        print(f"\n**{cls_name}:**")
        print(f"  - Precision: {r['precision']:.3f}  (of those flagged, this fraction truly belong here)")
        print(f"  - Recall:    {r['recall']:.3f}  (of all true members, this fraction were caught)")
        print(f"  - F1 Score:  {r['f1-score']:.3f}")
        print(f"  - Support:   {int(r['support']):,} respondents")
    print(f"\n**Overall accuracy:** {report['accuracy']:.3f}")
    print()


def section_shap(shap_df, var_info):
    _heading("5. Top 30 Risk Factors by SHAP Importance", 2)
    print("Mean |SHAP| = average influence on the burnout prediction across all respondents.\n")
    mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(30)
    _table(
        [{"Rank": i+1, "Variable": c, "Description": var_info.get(c, c)[:65],
          "Mean |SHAP|": f"{v:.5f}"}
         for i, (c, v) in enumerate(mean_abs.items())],
        ["Rank", "Variable", "Description", "Mean |SHAP|"]
    )
    print("### Average SHAP Direction (Top 20)\n")
    print("Positive = on average raises predicted burnout. Negative = on average lowers it.\n")
    mean_signed = shap_df.mean().loc[mean_abs.head(20).index]
    _table(
        [{"Variable": c, "Description": var_info.get(c, c)[:65],
          "Mean SHAP": f"{v:+.5f}",
          "Net effect": "Raises risk" if v > 0 else "Lowers risk"}
         for c, v in mean_signed.items()],
        ["Variable", "Description", "Mean SHAP", "Net effect"]
    )


def section_burnout_indicators(data):
    _heading("6. Burnout Indicator Prevalence", 2)
    print("% of respondents reporting strain for each of the 12 burnout items.  \n"
          "Note: burnout_raw has been recoded to 0-1 (0=no strain, 1=max strain). "
          "For binary items, 1.0 = Yes/strain present. For ICS_40, values >= 0.667 "
          "correspond to 'Very stressful' or 'Stressful' on the original 4-point scale.\n")
    burnout_raw = data["burnout_raw"]
    rows = []
    for col, label in BURNOUT_LABELS.items():
        if col not in burnout_raw.columns:
            continue
        col_vals = burnout_raw[col].dropna()
        if len(col_vals) == 0:
            continue
        if col == "ICS_40":
            # Recoded: 1.0=Very, 0.667=Stressful, 0.333=Somewhat, 0.0=Not at all
            pct  = ((col_vals >= 0.667).sum() / len(col_vals)) * 100
            note = "Very or Stressful (top 2 of 4)"
        else:
            # Recoded: 1.0=Yes (strain), 0.0=No
            pct  = ((col_vals == 1.0).sum() / len(col_vals)) * 100
            note = "Yes (strain present)"
        rows.append({
            "Variable": col, "Burnout Indicator": label,
            "% Reporting Strain": f"{pct:.1f}%", "Scale": note,
            "n answered": f"{len(col_vals):,}",
        })
    rows.sort(key=lambda r: float(r["% Reporting Strain"].rstrip("%")), reverse=True)
    _table(rows, ["Variable", "Burnout Indicator", "% Reporting Strain", "Scale", "n answered"])


def section_workload(data):
    _heading("7. Caregiver Workload — Activity Breadth", 2)
    print("Number of activity types each caregiver is responsible for (max 7).\n")
    adf = data["analysis_df"]
    avail_act = {k: v for k, v in ACTIVITY_GATEWAY.items() if k in adf.columns}

    activity_count = (adf[list(avail_act.keys())] == 1).sum(axis=1)
    adf = adf.copy()
    adf["activity_count"] = activity_count
    active = activity_count[activity_count > 0]

    print(f"- Respondents with at least one activity type: **{len(active):,}**")
    print(f"- Average types per active caregiver: **{active.mean():.1f}**")
    print(f"- Juggling 3+ types: **{(activity_count >= 3).sum():,}** "
          f"({(activity_count >= 3).mean()*100:.1f}%)")
    print()

    act_burnout = (
        adf[["activity_count", "burnout_high"]]
        .groupby("activity_count")["burnout_high"]
        .agg(rate="mean", count="count").reset_index()
    )
    act_burnout["High Burnout %"] = (act_burnout["rate"] * 100).round(1)
    _table(
        [{"Activity Types": int(r["activity_count"]), "Respondents": f"{int(r['count']):,}",
          "High Burnout %": f"{r['High Burnout %']:.1f}%"}
         for _, r in act_burnout.iterrows()],
        ["Activity Types", "Respondents", "High Burnout %"]
    )

    print("### Activity Type Breakdown\n")
    act_rows = []
    for col, label in avail_act.items():
        n_yes = int((adf[col] == 1).sum())
        br    = adf.loc[adf[col] == 1, "burnout_high"].mean() * 100
        act_rows.append({
            "Activity Type": label,
            "Caregivers Involved": f"{n_yes:,}",
            "% of Respondents": f"{n_yes/len(adf)*100:.1f}%",
            "High Burnout Rate": f"{br:.1f}%",
        })
    act_rows.sort(key=lambda r: int(r["Caregivers Involved"].replace(",", "")), reverse=True)
    _table(act_rows, ["Activity Type", "Caregivers Involved", "% of Respondents", "High Burnout Rate"])


def section_finances(data):
    _heading("8. Financial Burden on Caregivers", 2)
    print("Out-of-pocket expenses paid on behalf of the care receiver.\n")
    print("**Important:** Expense variables are coded as ordinal brackets "
          "(1='Less than $200', 2='$200-$499', ... 6='$5,000+'), "
          "NOT actual dollar amounts. We report the distribution across brackets.\n")
    adf      = data["analysis_df"]
    avail_ex = {k: v for k, v in EXPENSE_COLS.items() if k in adf.columns}
    if not avail_ex:
        print("_Expense columns not available._\n")
        return

    # Count respondents who answered at least one expense question (non-NaN)
    exp_df     = adf[list(avail_ex.keys())].copy()
    has_answer = exp_df.notna().any(axis=1)
    n_answered = has_answer.sum()
    print(f"- Respondents who answered at least one expense question: **{n_answered:,}** "
          f"({n_answered/len(adf)*100:.1f}% of caregivers with burnout data)")
    print(f"- The remaining {len(adf)-n_answered:,} had 'Valid skip' on all expense items "
          f"(likely not asked due to module routing).\n")

    # Per-category: distribution across expense brackets
    for col, label in avail_ex.items():
        col_vals = exp_df[col].dropna()
        if len(col_vals) == 0:
            continue
        print(f"### {label} (`{col}`) -- n={len(col_vals):,} respondents answered\n")
        bracket_rows = []
        for code in sorted(EXPENSE_RANGES.keys()):
            n_in  = int((col_vals == code).sum())
            if n_in == 0:
                continue
            pct   = n_in / len(col_vals) * 100
            br_mask = (adf[col] == code) & adf["burnout_high"].notna()
            br_rate = adf.loc[br_mask, "burnout_high"].mean() * 100 if br_mask.sum() > 0 else 0
            bracket_rows.append({
                "Expense Range": EXPENSE_RANGES[code],
                "Respondents": f"{n_in:,}",
                "% of Answers": f"{pct:.1f}%",
                "High Burnout %": f"{br_rate:.1f}%",
            })
        _table(bracket_rows, ["Expense Range", "Respondents", "% of Answers", "High Burnout %"])


def section_composite_scores(data):
    _heading("9. Composite Score Analysis", 2)
    print("Workplace flexibility (FWA module) and work-enabling circumstances (ITE module) "
          "by score value and associated burnout rate.\n")
    adf = data["analysis_df"]
    for col, title, desc in [
        ("flexibility_score",
         "Workplace Flexibility Score (FWA, -1 to +5)",
         "Sum of 5 positive flexibility items minus 1 if career penalty for using flexibility."),
        ("ite_score",
         "Work-Enabling Circumstances Score (ITE, -1 to +5)",
         "Sum of 5 enabling-support items minus 1 if already had to quit/reduce hours."),
    ]:
        if col not in adf.columns:
            continue
        s = adf[col].dropna()
        print(f"### {title}\n")
        print(f"{desc}\n")
        print(f"- n={len(s):,} | mean={s.mean():.2f} | median={s.median():.0f} | std={s.std():.2f}\n")
        sb = adf[[col, "burnout_high"]].dropna().groupby(col)["burnout_high"]
        sb = sb.agg(rate="mean", count="count").reset_index()
        sb["High Burnout %"] = (sb["rate"] * 100).round(1)
        _table(
            [{"Score": int(r[col]), "Respondents": f"{int(r['count']):,}",
              "High Burnout %": f"{r['High Burnout %']:.1f}%"}
             for _, r in sb.iterrows()],
            ["Score", "Respondents", "High Burnout %"]
        )


def section_missing_data(data, var_info):
    _heading("10. Data Completeness — Top 20 Variables with Most Missing Values", 2)
    print("High missingness is usually module-specific routing (valid skip), "
          "not refusal to answer.\n")
    adf = data["analysis_df"]
    miss = (
        adf.drop(columns=["burnout_high", "burnout_score", "WGHT_PER"], errors="ignore")
        .isnull().mean() * 100
    )
    miss = miss[miss > 0].sort_values(ascending=False).head(20)
    _table(
        [{"Variable": c, "Description": var_info.get(c, c)[:65], "% Missing": f"{v:.1f}%"}
         for c, v in miss.items()],
        ["Variable", "Description", "% Missing"]
    )
    print(f"- Variables 1–50% missing: **{miss.between(1, 50, inclusive='both').sum()}**")
    print(f"- Variables >50% missing (module-specific): **{(miss > 50).sum()}**")
    print()


# =============================================================================
# MAIN
# =============================================================================
def main():
    label_map = load_label_map()
    var_info  = load_var_info()
    data      = run_pipeline(label_map)
    model, auc, report, X_test, y_test, y_prob = train_xgb(
        data["X"], data["y_clf"], data["W"]
    )
    shap_df = compute_shap(model, data["X"])

    _progress("Generating report...")

    # --- Open output file and tee stdout to it ---
    original_stdout = sys.stdout
    if REPORT_OUTPUT_FILE is not None:
        os.makedirs(os.path.dirname(REPORT_OUTPUT_FILE), exist_ok=True)
        out_file = open(REPORT_OUTPUT_FILE, "w", encoding="utf-8")
        sys.stdout = _Tee(original_stdout, out_file)
    else:
        out_file = None

    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        print("# Caregiver Burnout Risk Analysis Report")
        print()
        print(f"**Generated:** {ts}  ")
        print("**Data:** GSS Caregiving and Care Receiving 2018 (Statistics Canada, Cycle 32)  ")
        print("**Method:** XGBoost binary classifier + SHAP explanations  ")
        print(f"**Burnout split:** `{BURNOUT_SPLIT}` (threshold = {data['median_cutoff']:.3f})  ")
        print("**Purpose:** Identify at-risk caregiver groups for government/nonprofit resource allocation  ")
        print()
        print("---")

        section_dataset(data)
        section_demographics(data, label_map)
        section_correlations(data, var_info)
        section_model(auc, report, model)
        section_shap(shap_df, var_info)
        section_burnout_indicators(data)
        section_workload(data)
        section_finances(data)
        section_composite_scores(data)
        section_missing_data(data, var_info)

        print("---")
        print("_End of report._")

    finally:
        sys.stdout = original_stdout
        if out_file is not None:
            out_file.close()
            _progress(f"Report saved to: {REPORT_OUTPUT_FILE}")

    _progress("Done.")


if __name__ == "__main__":
    main()
