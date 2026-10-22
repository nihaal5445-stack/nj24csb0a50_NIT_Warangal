"""
CardioTwin: the per-patient digital twin.

A predictive model answers "what will happen?". A digital twin has to answer
"what happens if I change something?". That is the distinction this module
implements.

The twin is a reduced mechanistic model of the patient's circadian blood
pressure, personalised from one calibration week of observed data:

    SBP(t) = S . circ(t; nightf) + surge(t; A) + behaviour(t) - med(t) + e

Personalisation estimates S (setpoint), nightf (nocturnal factor) and A
(morning-surge amplitude) from that patient's own trace. Behaviour
coefficients are estimated once across the cohort by within-patient
regression on observed data. The twin never reads the generator's latent
traits.

Counterfactual validation is the honest part: for every intervention we also
re-run the ground-truth simulator with the same per-patient random seed and
the same override, so the twin's predicted change can be compared against
what the simulated patient actually did. Paired, same noise, no cherry-picks.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from config import SAMPLE_MIN, DAYS, EXCURSION_SBP, DRUGS
from generate_wearables import simulate_patient, patient_seed, PHENO_NIGHT

ROOT = pathlib.Path(__file__).resolve().parents[1]

INTERVENTIONS = {
    "chronotherapy":   {"label": "Move one agent to 21:00 dosing", "dose_hours": "EVENING"},
    "sleep_plus_1h":   {"label": "Extend sleep by 1 hour",         "sleep_need_delta": 1.0},
    "salt_minus_3g":   {"label": "Reduce salt by 3 g/day",         "salt_delta": -3.0},
    "activity_plus30": {"label": "Increase activity by 30%",       "activity_delta": 0.30},
    "full_adherence":  {"label": "Perfect medication adherence",   "adherence": 1.0},
}


# ----------------------------------------------------------------------------
# population behaviour coefficients, estimated from OBSERVED data only
# ----------------------------------------------------------------------------
def estimate_behaviour_coefficients(daily: pd.DataFrame) -> dict:
    """Within-patient regression of daily mean SBP on observable behaviour.

    Patient means are removed first, so these are within-person effects and
    are not confounded by who the patient is.
    """
    d = daily.sort_values(["patient_id", "day"]).copy()
    d["sleep_debt"] = d.groupby("patient_id")["sleep_hours"].transform("mean") - d["sleep_hours"]
    # Same-day steps are confounded by the ACUTE exercise pressor response:
    # blood pressure rises while you are moving. The chronic training effect
    # is the opposite sign. We therefore regress on a lagged 3-day activity
    # average, which carries the conditioning effect without the acute rise.
    d["steps_k_lag"] = (d.groupby("patient_id")["steps_total"]
                          .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()) / 1000.0)
    d = d.dropna(subset=["steps_k_lag"])
    # Awake time drives step count mechanically, and also tracks sleep debt.
    # Without it in the model, steps absorb that confound.
    d["awake_h"] = 24.0 - d["sleep_hours"]
    for c in ["sbp_mean", "sleep_debt", "steps_k_lag", "awake_h"]:
        d[c] = d[c] - d.groupby("patient_id")[c].transform("mean")
    X = np.column_stack([d["sleep_debt"], d["steps_k_lag"], d["awake_h"], np.ones(len(d))])
    beta, *_ = np.linalg.lstsq(X, d["sbp_mean"].to_numpy(), rcond=None)
    return {
        "mmhg_per_hour_sleep_debt": float(beta[0]),
        "mmhg_per_1000_steps": float(beta[1]),
        # NOT estimated and NOT observable from a wearable. Taken from the
        # hypertension literature (roughly 1 mmHg SBP per gram of daily
        # dietary salt). Because our simulator uses a coefficient in the same
        # range, the salt arm of the counterfactual validation below is
        # partly circular and should not be read as evidence that the twin
        # recovers an unknown parameter. It is reported for completeness.
        "mmhg_per_gram_salt": 0.95,
    }


class CardioTwin:
    """A personalised twin for one patient."""

    def __init__(self, ehr_row: pd.Series, coeffs: dict):
        self.ehr = ehr_row
        self.c = coeffs
        self.params: dict = {}

    # -- personalisation ----------------------------------------------------
    def fit(self, cal: pd.DataFrame) -> "CardioTwin":
        """cal = the patient's calibration-week 5-minute trace (observed)."""
        night = cal[cal["asleep"]]
        day = cal[~cal["asleep"]]
        s_night = float(night["sbp_mmhg"].mean())
        s_day = float(day["sbp_mmhg"].mean())

        self.params["nightf"] = s_night / s_day if s_day else 1.0
        self.params["setpoint"] = s_day
        trough = float(np.sort(night["sbp_mmhg"].to_numpy())[: max(len(night) // 12, 1)].mean())
        self.params["surge"] = max(s_day - trough, 0.0)
        self.params["sleep_h"] = float(len(night) * SAMPLE_MIN / 60.0 / max(cal["day"].nunique(), 1))
        self.params["steps_day"] = float(cal["steps_total"].sum() if "steps_total" in cal
                                         else cal["steps_5min"].sum() / max(cal["day"].nunique(), 1))
        self.params["dip_pct"] = 100.0 * (1.0 - self.params["nightf"])
        self.params["observed_mean"] = float(cal["sbp_mmhg"].mean())
        return self

    # -- regimen pharmacodynamics ------------------------------------------
    def _regimen(self, dose_hours: list[int] | None = None):
        def _split(v):
            s = str(v)
            return [] if s in ("", "nan", "None") else [x for x in s.split("|") if x not in ("", "nan")]
        drugs = [d for d in _split(self.ehr["rx_agents"]) if d in DRUGS]
        hours = dose_hours if dose_hours is not None else [int(float(h)) for h in _split(self.ehr["rx_dose_hours"])]
        n = min(len(drugs), len(hours))
        return drugs[:n], hours[:n]

    def _med_profile(self, dose_hours=None, adherence=1.0, grid_h=None):
        """Mean mmHg lowering across the 24-h clock, and the night-time mean."""
        if grid_h is None:
            grid_h = np.arange(0, 24, SAMPLE_MIN / 60.0)
        drugs, hours = self._regimen(dose_hours)
        eff = np.zeros_like(grid_h)
        for drug, dh in zip(drugs, hours):
            spec = DRUGS[drug]
            acc = 1.0 / (1.0 - np.exp(-24.0 * np.log(2) / spec["half_life_h"]))
            # steady state: superpose the last 10 daily doses
            for k in range(10):
                dt = (grid_h - dh) + 24 * k
                m = dt >= 0
                shape = np.zeros_like(dt)
                shape[m] = ((1 - np.exp(-dt[m] / max(spec["peak_h"] / 3.0, 0.3)))
                            * np.exp(-dt[m] * np.log(2) / spec["half_life_h"]))
                eff += (spec["max_drop"] / acc) * shape * adherence
        return eff, grid_h

    # -- counterfactual simulation -----------------------------------------
    def predict_change(self, intervention: str) -> dict:
        """Twin's predicted change in mean and night SBP for an intervention."""
        spec = INTERVENTIONS[intervention]
        grid = np.arange(0, 24, SAMPLE_MIN / 60.0)
        night_mask = (grid >= 23) | (grid < 6.5)

        base_eff, _ = self._med_profile(adherence=1.0, grid_h=grid)
        d_mean = d_night = 0.0

        if "dose_hours" in spec:
            drugs, hours = self._regimen()
            if hours:
                new = list(hours)
                new[0] = 21                                  # shift one agent to bedtime
                new_eff, _ = self._med_profile(dose_hours=new, adherence=1.0, grid_h=grid)
                d_mean = -(new_eff.mean() - base_eff.mean())
                d_night = -(new_eff[night_mask].mean() - base_eff[night_mask].mean())
        elif "adherence" in spec:
            cur = self._estimated_adherence()
            cur_eff, _ = self._med_profile(adherence=cur, grid_h=grid)
            d_mean = -(base_eff.mean() - cur_eff.mean())
            d_night = -(base_eff[night_mask].mean() - cur_eff[night_mask].mean())
        elif "sleep_need_delta" in spec:
            # one extra hour of sleep removes one hour of accumulated debt,
            # amplified by the AR(1) carry-over in debt accumulation
            carry = 1.0 / (1.0 - 0.72)
            d_mean = d_night = self.c["mmhg_per_hour_sleep_debt"] * spec["sleep_need_delta"] * -1.0 * carry
        elif "salt_delta" in spec:
            d_mean = d_night = self.c["mmhg_per_gram_salt"] * spec["salt_delta"]
        elif "activity_delta" in spec:
            extra_k = self.params["steps_day"] * spec["activity_delta"] / 1000.0
            d_mean = d_night = self.c["mmhg_per_1000_steps"] * extra_k

        return {"d_mean_sbp": float(d_mean), "d_night_sbp": float(d_night)}

    def _estimated_adherence(self) -> float:
        """Crude adherence proxy from how much of the expected drug effect is
        visible in the calibration trace. Bounded to keep it sane."""
        drugs, _ = self._regimen()
        if not drugs:
            return 1.0
        expected = sum(DRUGS[d]["max_drop"] for d in drugs)
        shortfall = max(self.params["observed_mean"] - 118.0, 0.0)
        return float(np.clip(1.0 - shortfall / max(expected * 4.0, 1.0), 0.45, 1.0))


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def ground_truth_change(ehr_row, lat_row, intervention: str) -> dict:
    """Re-run the ground-truth simulator with the same seed and the override."""
    spec = dict(INTERVENTIONS[intervention])
    spec.pop("label", None)
    o = {}
    if spec.get("dose_hours") == "EVENING":
        def _split(v):
            s = str(v)
            return [] if s in ("", "nan", "None") else [x for x in s.split("|") if x not in ("", "nan")]
        hrs = [int(float(h)) for h in _split(ehr_row["rx_dose_hours"])]
        if not hrs:
            return {}
        hrs[0] = 21
        o["dose_hours"] = hrs
    else:
        o = spec

    base = simulate_patient(ehr_row, lat_row, DAYS,
                            rng=np.random.default_rng(patient_seed(ehr_row["patient_id"])))
    cf = simulate_patient(ehr_row, lat_row, DAYS,
                          rng=np.random.default_rng(patient_seed(ehr_row["patient_id"])),
                          overrides=o)
    nb, nc = base[base.asleep]["sbp_mmhg"], cf[cf.asleep]["sbp_mmhg"]
    return {
        "d_mean_sbp": float(cf.sbp_mmhg.mean() - base.sbp_mmhg.mean()),
        "d_night_sbp": float(nc.mean() - nb.mean()),
        "d_excursion_frac": float((cf.sbp_mmhg >= EXCURSION_SBP).mean()
                                  - (base.sbp_mmhg >= EXCURSION_SBP).mean()),
    }


def validate(n_patients: int = 120) -> pd.DataFrame:
    ehr = pd.read_csv(ROOT / "data" / "ehr.csv")
    lat = pd.read_csv(ROOT / "data" / "latent_truth.csv")
    daily = pd.read_csv(ROOT / "data" / "daily_summary.csv")
    ts = pd.read_parquet(ROOT / "data" / "wearables.parquet")
    coeffs = estimate_behaviour_coefficients(daily)
    print("Behaviour coefficients estimated from observed data:")
    for k, v in coeffs.items():
        print(f"   {k:28s} {v:+.3f}")

    rows = []
    sub = ehr.head(n_patients)
    for i in range(len(sub)):
        e, l = sub.iloc[i], lat.iloc[i]
        cal = ts[(ts.patient_id == e["patient_id"]) & (ts.day < 7)]
        if cal.empty:
            continue
        tw = CardioTwin(e, coeffs).fit(cal)
        for name in INTERVENTIONS:
            pred = tw.predict_change(name)
            truth = ground_truth_change(e, l, name)
            if not truth:
                continue
            rows.append({"patient_id": e["patient_id"], "intervention": name,
                         "pred_d_mean": pred["d_mean_sbp"], "true_d_mean": truth["d_mean_sbp"],
                         "pred_d_night": pred["d_night_sbp"], "true_d_night": truth["d_night_sbp"],
                         "true_d_excursion": truth["d_excursion_frac"]})
    return pd.DataFrame(rows), coeffs


if __name__ == "__main__":
    df, coeffs = validate()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "counterfactual_validation.csv", index=False)

    summary = {"behaviour_coefficients": coeffs, "by_intervention": {}}
    print(f"\nCounterfactual validation, {df.patient_id.nunique()} patients, paired re-simulation")
    print(f"{'intervention':<18}{'true dSBP':>10}{'twin dSBP':>10}{'MAE':>7}"
          f"{'sign':>6}{'true dNight':>13}{'twin dNight':>13}")
    for name, g in df.groupby("intervention"):
        mae = float(np.mean(np.abs(g.pred_d_mean - g.true_d_mean)))
        sign = float(np.mean(np.sign(g.pred_d_mean) == np.sign(g.true_d_mean)))
        r = float(np.corrcoef(g.pred_d_mean, g.true_d_mean)[0, 1]) if g.pred_d_mean.std() > 1e-9 else float("nan")
        mae_n = float(np.mean(np.abs(g.pred_d_night - g.true_d_night)))
        summary["by_intervention"][name] = {
            "label": INTERVENTIONS[name]["label"],
            "true_mean_d_sbp": round(float(g.true_d_mean.mean()), 2),
            "twin_mean_d_sbp": round(float(g.pred_d_mean.mean()), 2),
            "mae_mmhg": round(mae, 2), "sign_agreement": round(sign, 3),
            "pearson_r": None if np.isnan(r) else round(r, 3),
            "true_night_d_sbp": round(float(g.true_d_night.mean()), 2),
            "twin_night_d_sbp": round(float(g.pred_d_night.mean()), 2),
            "mae_night_mmhg": round(mae_n, 2),
            "true_d_excursion_pct": round(float(g.true_d_excursion.mean()) * 100, 2),
        }
        print(f"{name:<18}{g.true_d_mean.mean():>10.2f}{g.pred_d_mean.mean():>10.2f}"
              f"{mae:>7.2f}{sign:>6.0%}{g.true_d_night.mean():>13.2f}{g.pred_d_night.mean():>13.2f}")
    (out / "counterfactual_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out/'counterfactual_validation.csv'} and counterfactual_summary.json")
