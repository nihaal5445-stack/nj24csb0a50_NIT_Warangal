# Methods

Detail that does not fit in the README. Everything reproduces with `./run_all.sh`.

## Simulator

For each patient, 21 days at 5-minute resolution (6,048 samples), integrated forward from EHR-derived and latent parameters.

**Sleep.** Patient-specific bedtime ~ N(23:00, 0.7 h). Duration ~ N(sleep_need − 0.8·OSA, 0.85). Stages follow ~90-minute cycles with N3 weighted early and REM late. OSA raises micro-arousal rate roughly fivefold.

**Heart rate.** `HR = HR_base + circadian + activity − sleep_depth_effect + noise`, with `HR_base = 72 − 2.6·autonomic_z + 0.08·(BMI − 25)`.

**HRV.** `log RMSSD` is modelled with a nocturnal boost scaled by phenotype (dipper 1.0, non-dipper 0.58, reverse dipper 0.34), by OSA (×0.55) and by autonomic tone. This is the coupling that makes the dipping phenotype visible without a cuff, and it is the project's single most load-bearing assumption.

**Blood pressure.**
```
SBP(t) = setpoint · circ(t; night_f) + day_offset + surge(t) + 0.030·steps − med(t) + spikes + noise
```
with night factors 0.845 / 0.945 / 1.035 by phenotype, smoothed over a 45-minute transition. `day_offset` carries salt load, cumulative sleep debt (AR(1), decay 0.72) and habitual activity drift. Sympathetic spikes arrive as a Poisson process, suppressed 75% during sleep.

**Pharmacodynamics.** Per agent, rise to peak with time constant `peak_h/3` then first-order decay at the agent's half-life, normalised by the daily-dosing accumulation factor `1/(1 − exp(−24·ln2/t½))` so that `max_drop` is the **steady-state** effect rather than the single-dose peak. Missing this normalisation in an early version produced a 35 mmHg drop from amlodipine alone.

## Labels

Primary: sustained SBP ≥ 160 mmHg for ≥ 30 minutes (≥ 6 consecutive samples), detected by a moving-window convolution. Secondary: the same construction at ≥ 180 mmHg.

## Features

17 daily constructs → 3 summaries each (7-day mean, last value, OLS trend) = 51, plus sleep debt, excursion burden and days since last excursion. 22 static EHR features. The headline model excludes all 29 cuff-derived dynamic features.

## Models

XGBoost: 380 estimators, depth 4, lr 0.045, subsample 0.85, colsample 0.75, λ 2.2, min_child_weight 6. Wrapped in `CalibratedClassifierCV(method="isotonic", cv=3)`.

Split: `GroupShuffleSplit(test_size=0.30)` on `patient_id`. Bootstrap CIs: 2,000 resamples **of patients**, not rows, so the intervals account for within-patient correlation.

## Twin personalisation

From the calibration week (days 0–6):
- `setpoint` = mean awake SBP
- `night_f` = mean asleep SBP / mean awake SBP
- `surge` = mean awake SBP − mean of the lowest 1/12 of sleeping readings

Behaviour coefficients are estimated once across the cohort by within-patient OLS of daily mean SBP on sleep debt, lagged 3-day activity and awake hours, all patient-demeaned. The awake-hours covariate is necessary: without it, step count absorbs the wakefulness confound and the activity coefficient flips sign.

## Counterfactual validation

`patient_seed(pid)` gives each patient a deterministic seed. The factual and counterfactual runs are drawn from freshly seeded generators with the same value, so the two trajectories share every random draw and differ only in the intervention. Reported as MAE in mmHg and sign agreement across 120 patients × 5 interventions.
