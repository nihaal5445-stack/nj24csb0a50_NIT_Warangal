"""Global configuration for the CardioTwin synthetic cohort and model."""
from dataclasses import dataclass

SEED = 20261020
N_PATIENTS = 300
DAYS = 21                 # days of wearable history per patient
SAMPLE_MIN = 5            # wearable sampling interval, minutes
PRED_HORIZON_H = 24       # predict an event in the next 24 h
HISTORY_DAYS = 7          # dynamic window the model sees

# --- clinical thresholds (mmHg) -------------------------------------------
# Primary modelled adverse event: a sustained Stage-2 hypertensive excursion.
# This is the level at which Indian and international guidance triggers review
# and treatment escalation, so it is the clinically actionable target.
EXCURSION_SBP = 160.0
# Secondary, rarer: hypertensive urgency.
URGENCY_SBP = 180.0
EVENT_MIN_SUSTAIN = 30    # minutes the threshold must be held
URGENCY_MIN_SUSTAIN = 30
DIPPER_CUT = 0.10         # <10% nocturnal fall = non-dipper

# Indian epidemiology anchors (ICMR-INDIAB, Lancet Diabetes Endocrinol 2023)
EPI = {
    "hypertensive_adults_india": 315e6,
    "diabetic_adults_india": 101e6,
    "prediabetic_adults_india": 136e6,
}

# --- antihypertensive agents ----------------------------------------------
# half_life_h and peak_h are rough pharmacokinetic shapes, not dosing advice.
DRUGS = {
    "amlodipine":  {"class": "CCB",     "peak_h": 8.0,  "half_life_h": 40.0, "max_drop": 12.0},
    "telmisartan": {"class": "ARB",     "peak_h": 3.0,  "half_life_h": 24.0, "max_drop": 10.0},
    "hydrochlorothiazide": {"class": "Thiazide", "peak_h": 4.0, "half_life_h": 10.0, "max_drop": 8.0},
    "metoprolol":  {"class": "BB",      "peak_h": 2.0,  "half_life_h": 7.0,  "max_drop": 9.0},
}

PHENOTYPES = ["dipper", "non_dipper", "reverse_dipper"]
PHENOTYPE_P = [0.55, 0.33, 0.12]   # ~45% non-dipping overall, within the 30-50%
                                   # range reported for treated hypertensive cohorts
