"""Tests for CardioTwin. Run with: python3 -m pytest tests/ -q"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import generate_ehr as ge          # noqa: E402
import generate_wearables as gw    # noqa: E402
from config import EXCURSION_SBP, SAMPLE_MIN, URGENCY_SBP  # noqa: E402
from features import is_bp_derived, per_day_summary, split_dynamic  # noqa: E402
from twin import CardioTwin, INTERVENTIONS, estimate_behaviour_coefficients  # noqa: E402


@pytest.fixture(scope="module")
def cohort():
    ehr, lat = ge.generate_cohort(n=12)
    frames = [gw.simulate_patient(ehr.iloc[i], lat.iloc[i], 10) for i in range(len(ehr))]
    ts = gw.label_events(pd.concat(frames, ignore_index=True))
    return ehr, lat, ts


# ---------------------------------------------------------------- generator
def test_cohort_shape(cohort):
    ehr, lat, ts = cohort
    assert len(ehr) == 12 and len(lat) == 12
    assert ts.patient_id.nunique() == 12
    assert len(ts) == 12 * 10 * 24 * 60 // SAMPLE_MIN


def test_vitals_physiologically_plausible(cohort):
    _, _, ts = cohort
    assert 70 < ts.sbp_mmhg.min() and ts.sbp_mmhg.max() < 250
    assert ts.dbp_mmhg.median() < ts.sbp_mmhg.median()
    assert 35 < ts.hr_bpm.min() and ts.hr_bpm.max() < 200
    assert (ts.rmssd_ms > 0).all()
    assert (ts.steps_5min >= 0).all()


def test_no_nans(cohort):
    _, _, ts = cohort
    assert not ts.select_dtypes("number").isna().any().any()


def test_patients_sleep_a_plausible_amount(cohort):
    _, _, ts = cohort
    hours = ts.groupby(["patient_id", "day"])["asleep"].sum() * SAMPLE_MIN / 60
    assert 3 < hours.mean() < 11


def test_heart_rate_falls_during_sleep(cohort):
    _, _, ts = cohort
    assert ts.loc[ts.asleep, "hr_bpm"].mean() < ts.loc[~ts.asleep, "hr_bpm"].mean()


# ------------------------------------------------- the load-bearing physiology
def test_dipping_phenotypes_match_clinical_definitions(cohort):
    """Dippers fall >=10% overnight, non-dippers <10%, reverse dippers rise."""
    ehr, lat, ts = cohort
    daily = per_day_summary(ts).merge(lat[["patient_id", "phenotype"]], on="patient_id")
    dip = daily.groupby("phenotype")["dip_ratio"].mean()
    if "dipper" in dip:
        assert dip["dipper"] > 0.10
    if "non_dipper" in dip:
        assert 0 < dip["non_dipper"] < 0.10
    if "reverse_dipper" in dip:
        assert dip["reverse_dipper"] < 0.02
    if "dipper" in dip and "non_dipper" in dip:
        assert dip["dipper"] > dip["non_dipper"]


def test_nocturnal_hrv_rise_is_blunted_in_non_dippers(cohort):
    """The coupling the cuff-free screen depends on must actually be present."""
    _, lat, ts = cohort
    daily = per_day_summary(ts).merge(lat[["patient_id", "phenotype"]], on="patient_id")
    r = daily.groupby("phenotype")["hrv_night_ratio"].mean()
    if "dipper" in r and "non_dipper" in r:
        assert r["dipper"] > r["non_dipper"]


# -------------------------------------------------------------------- labels
def test_event_labels_are_consistent(cohort):
    _, _, ts = cohort
    assert ts.loc[ts.excursion_event == 1, "sbp_mmhg"].max() >= EXCURSION_SBP
    # urgency is a strict subset of excursion
    assert not ((ts.urgency_event == 1) & (ts.excursion_event == 0)).any()
    assert URGENCY_SBP > EXCURSION_SBP


# ------------------------------------------------------------ feature hygiene
def test_cuff_derived_features_are_excluded():
    assert is_bp_derived("sbp_mean_mean7")
    assert is_bp_derived("dip_ratio_last")
    assert is_bp_derived("morning_surge_trend")
    assert not is_bp_derived("rmssd_mean_mean7")
    assert not is_bp_derived("sleep_hours_last")
    cuffless, cuffed = split_dynamic(
        ["sbp_mean_mean7", "dip_ratio_last", "rmssd_mean_mean7", "steps_total_trend"])
    assert set(cuffless) == {"rmssd_mean_mean7", "steps_total_trend"}
    assert set(cuffed) == {"sbp_mean_mean7", "dip_ratio_last"}


# ----------------------------------------------------------- reproducibility
def test_same_seed_gives_identical_trace(cohort):
    ehr, lat, _ = cohort
    a = gw.simulate_patient(ehr.iloc[0], lat.iloc[0], 5)
    b = gw.simulate_patient(ehr.iloc[0], lat.iloc[0], 5)
    pd.testing.assert_frame_equal(a, b)


def test_different_patients_differ(cohort):
    ehr, lat, _ = cohort
    a = gw.simulate_patient(ehr.iloc[0], lat.iloc[0], 5)
    b = gw.simulate_patient(ehr.iloc[1], lat.iloc[1], 5)
    assert not np.allclose(a.sbp_mmhg.to_numpy(), b.sbp_mmhg.to_numpy())


# -------------------------------------------------------------------- twin
def test_twin_personalisation_recovers_the_patient(cohort):
    ehr, lat, ts = cohort
    daily = per_day_summary(ts)
    co = estimate_behaviour_coefficients(daily)
    pid = ehr.iloc[0]["patient_id"]
    cal = ts[(ts.patient_id == pid) & (ts.day < 7)]
    tw = CardioTwin(ehr.iloc[0], co).fit(cal)
    assert 80 < tw.params["setpoint"] < 220
    assert 0.6 < tw.params["nightf"] < 1.3
    assert tw.params["surge"] >= 0


def test_counterfactuals_point_the_right_way(cohort):
    """Every modelled intervention should lower pressure or leave it alone."""
    ehr, lat, ts = cohort
    co = estimate_behaviour_coefficients(per_day_summary(ts))
    for i in range(len(ehr)):
        pid = ehr.iloc[i]["patient_id"]
        cal = ts[(ts.patient_id == pid) & (ts.day < 7)]
        if cal.empty:
            continue
        tw = CardioTwin(ehr.iloc[i], co).fit(cal)
        for name in INTERVENTIONS:
            d = tw.predict_change(name)
            assert d["d_mean_sbp"] <= 0.5, f"{name} raised mean SBP for {pid}"
            assert np.isfinite(d["d_night_sbp"])


def test_chronotherapy_helps_at_night_not_on_average(cohort):
    """Bedtime dosing should move night pressure more than the 24h mean.

    This is the clinical point of chronotherapy and the behaviour that
    distinguishes the twin from a scalar risk score.
    """
    ehr, lat, ts = cohort
    co = estimate_behaviour_coefficients(per_day_summary(ts))
    checked = 0
    for i in range(len(ehr)):
        row = ehr.iloc[i]
        hrs = [h for h in str(row["rx_dose_hours"]).split("|") if h not in ("", "nan")]
        if not hrs or int(float(hrs[0])) == 21:
            continue
        cal = ts[(ts.patient_id == row["patient_id"]) & (ts.day < 7)]
        if cal.empty:
            continue
        d = CardioTwin(row, co).fit(cal).predict_change("chronotherapy")
        assert d["d_night_sbp"] <= d["d_mean_sbp"] + 1e-9
        checked += 1
    assert checked > 0, "no patient on a non-bedtime regimen to test"


def test_overrides_change_the_ground_truth(cohort):
    ehr, lat, _ = cohort
    base = gw.simulate_patient(ehr.iloc[0], lat.iloc[0], 8)
    less_salt = gw.simulate_patient(ehr.iloc[0], lat.iloc[0], 8, overrides={"salt_delta": -4.0})
    assert less_salt.sbp_mmhg.mean() < base.sbp_mmhg.mean()
