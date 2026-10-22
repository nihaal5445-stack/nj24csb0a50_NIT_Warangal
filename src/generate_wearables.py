"""
Synthetic wearable / IoT time-series generation for CardioTwin.

For every virtual patient we integrate a mechanistic circadian cardiovascular
model forward at 5-minute resolution for 21 days and emit the streams a
consumer wearable plus a connected BP cuff would produce:

    heart rate, HRV (RMSSD), sleep stage, step count, systolic and diastolic BP

The model encodes four pieces of established cardiovascular physiology:

  1. Circadian BP rhythm with a dipping phenotype. Healthy "dippers" fall
     10-20% overnight; non-dippers fall <10%; reverse dippers rise at night.
     Non-dipping is an independent cardiovascular risk marker.
  2. Morning surge. BP rises sharply on waking; a large surge is a risk marker
     and is exaggerated in non-dippers.
  3. Autonomic coupling. RMSSD rises at night with parasympathetic tone. That
     nocturnal rise is blunted in OSA, diabetic autonomic neuropathy and in
     non-dippers, so HRV carries information the BP level alone does not.
  4. Pharmacodynamics. Each antihypertensive contributes a dose-timed effect
     with a rise to peak and exponential offset, so a missed dose or a badly
     timed dose is visible in the trace.

Everything here is synthetic. No real patient data is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (SEED, DAYS, SAMPLE_MIN, EXCURSION_SBP, URGENCY_SBP,
                    EVENT_MIN_SUSTAIN, DRUGS)

STAGE_CODE = {"wake": 0, "light": 1, "deep": 2, "rem": 3}
PHENO_NIGHT = {"dipper": 0.845, "non_dipper": 0.945, "reverse_dipper": 1.035}
PHENO_SURGE = {"dipper": 1.0, "non_dipper": 1.45, "reverse_dipper": 1.75}
# Non-dippers show reduced nocturnal parasympathetic activity, so the
# overnight rise in RMSSD is blunted. This autonomic coupling is what makes
# the dipping phenotype visible to a cuff-free wearable at all.
PHENO_HRV = {"dipper": 1.0, "non_dipper": 0.58, "reverse_dipper": 0.34}


def _drug_effect(t_h: np.ndarray, dose_hours: list[int], drugs: list[str], taken: np.ndarray) -> np.ndarray:
    """mmHg of SBP lowering at each time point from the whole regimen."""
    eff = np.zeros_like(t_h)
    for drug, dh in zip(drugs, dose_hours):
        spec = DRUGS[drug]
        # Daily dosing accumulates. Normalise so that `max_drop` is the effect
        # at steady state rather than after a single dose.
        acc = 1.0 / (1.0 - np.exp(-24.0 * np.log(2) / spec["half_life_h"]))
        for day in range(int(t_h[-1] // 24) + 1):
            if day >= len(taken) or not taken[day]:
                continue                      # dose missed that day
            t0 = day * 24 + dh
            dt = t_h - t0
            m = dt >= 0
            # rise to peak then first-order decay
            shape = np.zeros_like(dt)
            shape[m] = (1 - np.exp(-dt[m] / max(spec["peak_h"] / 3.0, 0.3))) * \
                       np.exp(-dt[m] * np.log(2) / spec["half_life_h"])
            eff += (spec["max_drop"] / acc) * shape
    return eff


def _sleep_mask(t_h: np.ndarray, bed_h: float, dur_h: float, frag: float, rng) -> np.ndarray:
    """Boolean asleep mask plus stage codes."""
    tod = t_h % 24
    start, end = bed_h, bed_h + dur_h
    if end <= 24:
        asleep = (tod >= start) & (tod < end)
    else:
        asleep = (tod >= start) | (tod < end - 24)
    # fragmentation: random micro-arousals, more with OSA
    if frag > 0:
        arous = rng.random(t_h.size) < (frag * 0.035)
        asleep = asleep & ~arous
    return asleep


def _stages(asleep: np.ndarray, t_h: np.ndarray, rng) -> np.ndarray:
    """~90 min cycles: N3 dominant early, REM dominant late."""
    stage = np.zeros(t_h.size, dtype=np.int8)
    idx = np.where(asleep)[0]
    if idx.size == 0:
        return stage
    # position within the night, 0..1
    grp = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    for g in grp:
        if g.size < 3:
            stage[g] = STAGE_CODE["light"]
            continue
        pos = np.linspace(0, 1, g.size)
        cyc = (np.sin(2 * np.pi * pos * (g.size * SAMPLE_MIN / 90.0)) + 1) / 2
        p_deep = np.clip(0.55 * (1 - pos) * cyc, 0, 1)
        p_rem = np.clip(0.50 * pos * (1 - cyc), 0, 1)
        u = rng.random(g.size)
        s = np.full(g.size, STAGE_CODE["light"], dtype=np.int8)
        s[u < p_deep] = STAGE_CODE["deep"]
        s[(u >= p_deep) & (u < p_deep + p_rem)] = STAGE_CODE["rem"]
        stage[g] = s
    return stage


def patient_seed(pid: str) -> int:
    """Deterministic per-patient seed so a counterfactual re-run shares the
    same stochastic draws as the factual run, making the two paired."""
    return SEED + (int(str(pid).lstrip("PT") or 0) * 7919) % 2_000_003


def simulate_patient(row_ehr: pd.Series, row_lat: pd.Series, days: int,
                     rng=None, overrides: dict | None = None) -> pd.DataFrame:
    """overrides may set: dose_hours (list[int]), sleep_need_delta (h),
    salt_delta (g), activity_delta (fraction), adherence (0-1)."""
    o = overrides or {}
    if rng is None:
        rng = np.random.default_rng(patient_seed(row_ehr["patient_id"]))
    n = int(days * 24 * 60 / SAMPLE_MIN)
    t_h = np.arange(n) * SAMPLE_MIN / 60.0
    tod = t_h % 24
    day_idx = (t_h // 24).astype(int)

    pheno = row_lat["phenotype"]
    autonomic = float(row_lat["autonomic_z"])
    osa = bool(row_lat["osa"])
    adherence = float(o.get("adherence", row_lat["adherence"]))
    salt = float(row_lat["salt_load_g"]) + float(o.get("salt_delta", 0.0))

    # ---- daily behaviour -----------------------------------------------------
    bed_h = float(rng.normal(23.0, 0.7))
    need = float(row_lat["sleep_need_h"]) + float(o.get("sleep_need_delta", 0.0))
    dur = np.clip(rng.normal(need - (0.8 if osa else 0.0), 0.85, days), 3.5, 10.5)
    taken = rng.random(days) < adherence          # medication adherence per day
    base_act = np.clip(rng.normal(1.0, 0.25), 0.35, 1.9)
    # slow week-to-week drift in habitual activity, so a chronic training
    # effect is identifiable from within-patient variation
    drift = np.cumsum(rng.normal(0, 0.09, days)); drift -= drift.mean()
    act_day = np.clip((base_act + drift) * (1.0 + float(o.get("activity_delta", 0.0))), 0.25, 2.6)
    activity_level = act_day[day_idx]

    # sleep debt accumulates and decays
    debt = np.zeros(days)
    for d in range(1, days):
        debt[d] = np.clip(0.72 * debt[d - 1] + (need - dur[d - 1]), 0, 14)

    # ---- sleep / activity ----------------------------------------------------
    asleep = np.zeros(n, dtype=bool)
    for d in range(days):
        m = day_idx == d
        asleep[m] = _sleep_mask(t_h[m], bed_h, dur[d], 1.6 if osa else 0.35, rng)
    stage = _stages(asleep, t_h, rng)

    act_shape = np.clip(np.sin((tod - 6.5) / 24 * 2 * np.pi) + 0.55, 0, None)
    steps = np.where(asleep, 0.0,
                     rng.gamma(2.0, 28.0, n) * activity_level * act_shape).astype(np.float32)

    # ---- heart rate ----------------------------------------------------------
    hr_base = 72.0 - 2.6 * autonomic + 0.08 * (row_ehr["bmi"] - 25)
    hr = (hr_base - 2.0 * (act_day[day_idx] - 1.0)
          + 4.5 * np.sin((tod - 10) / 24 * 2 * np.pi)
          - np.where(asleep, 9.0 + 2.5 * (stage == STAGE_CODE["deep"]), 0.0)
          + 0.055 * steps
          + rng.normal(0, 2.2, n))
    hr = np.clip(hr, 38, 185)

    # ---- HRV (RMSSD, ms) -----------------------------------------------------
    noct_boost = (PHENO_HRV[pheno] * (1.0 if not osa else 0.55)
                  * (1.0 + 0.22 * autonomic))
    log_rmssd = (np.log(32.0)
                 + 0.30 * autonomic
                 - 0.011 * (row_ehr["age"] - 55)
                 + np.where(asleep, 0.42 * noct_boost, 0.0)
                 + 0.18 * (stage == STAGE_CODE["deep"])
                 - 0.004 * (hr - hr_base)
                 + rng.normal(0, 0.16, n))
    rmssd = np.exp(log_rmssd).astype(np.float32)

    # ---- blood pressure ------------------------------------------------------
    def _split(v):
        s = str(v)
        return [] if s in ("", "nan", "None") else [x for x in s.split("|") if x not in ("", "nan")]

    drugs = [d for d in _split(row_ehr["rx_agents"]) if d in DRUGS]
    dose_h = ([int(h) for h in o["dose_hours"]] if "dose_hours" in o
              else [int(float(x)) for x in _split(row_ehr["rx_dose_hours"])])
    drugs, dose_h = drugs[:len(dose_h)], dose_h[:len(drugs)]
    med = _drug_effect(t_h, dose_h, drugs, taken) if drugs else np.zeros(n)

    # slow day-level drift from behaviour
    day_offset = (0.95 * (salt - 9.0)
                  + 1.55 * debt[day_idx]
                  - 3.1 * (act_day[day_idx] - 1.0)
                  + rng.normal(0, 2.0, days)[day_idx])

    night_f = PHENO_NIGHT[pheno]
    circ = np.where(asleep, night_f, 1.0)
    # smooth the day/night transition
    k = max(int(45 / SAMPLE_MIN), 1)
    circ = np.convolve(circ, np.ones(k) / k, mode="same")
    circ[:k] = circ[k]
    circ[-k:] = circ[-k - 1]

    # morning surge: 2 h window after wake
    wake_h = (bed_h + dur[day_idx]) % 24
    since_wake = (tod - wake_h) % 24
    surge = PHENO_SURGE[pheno] * 7.5 * np.exp(-((since_wake - 0.8) ** 2) / (2 * 0.65 ** 2))

    # acute sympathetic spikes (stress, exertion), rarer when asleep
    spike = np.zeros(n)
    n_spk = rng.poisson(0.85 * days)
    for _ in range(n_spk):
        c = rng.integers(0, n)
        if asleep[c] and rng.random() < 0.75:
            continue
        w = rng.integers(3, 14)
        amp = rng.gamma(3.0, 7.0)
        lo, hi = max(0, c - w), min(n, c + w)
        spike[lo:hi] += amp * np.exp(-((np.arange(lo, hi) - c) ** 2) / (2 * (w / 2.2) ** 2))

    sbp = (row_lat["sbp_setpoint"] * circ
           + day_offset
           + surge
           + 0.030 * steps
           - med
           + spike
           + rng.normal(0, 3.4, n))
    sbp = np.clip(sbp, 78, 245).astype(np.float32)
    dbp = (sbp * 0.615 + 6.0 - 0.09 * (row_ehr["age"] - 55) + rng.normal(0, 2.4, n)).astype(np.float32)

    return pd.DataFrame({
        "patient_id": row_ehr["patient_id"],
        "t_hours": t_h.astype(np.float32),
        "day": day_idx.astype(np.int16),
        "hour_of_day": tod.astype(np.float32),
        "asleep": asleep,
        "sleep_stage": stage,
        "steps_5min": steps,
        "hr_bpm": hr.astype(np.float32),
        "rmssd_ms": rmssd,
        "sbp_mmhg": sbp,
        "dbp_mmhg": dbp,
        "dose_taken_today": taken[day_idx],
    })


def _sustained(x: np.ndarray, thresh: float, need: int) -> np.ndarray:
    over = (x >= thresh).astype(np.int8)
    run = np.convolve(over, np.ones(need, dtype=int), mode="same")
    return (run >= need).astype(np.int8)


def label_events(ts: pd.DataFrame) -> pd.DataFrame:
    """Primary: SBP >= 160 sustained >= 30 min. Secondary: >= 180 (urgency)."""
    need = max(int(EVENT_MIN_SUSTAIN / SAMPLE_MIN), 1)
    out = []
    for _, g in ts.groupby("patient_id", sort=False):
        s = g["sbp_mmhg"].to_numpy()
        gg = g.copy()
        gg["excursion_event"] = _sustained(s, EXCURSION_SBP, need)
        gg["urgency_event"] = _sustained(s, URGENCY_SBP, need)
        out.append(gg)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    import pathlib, time

    root = pathlib.Path(__file__).resolve().parents[1]
    data = root / "data"
    ehr = pd.read_csv(data / "ehr.csv")
    lat = pd.read_csv(data / "latent_truth.csv")
    t0 = time.time()
    frames = [simulate_patient(ehr.iloc[i], lat.iloc[i], DAYS) for i in range(len(ehr))]
    ts = pd.concat(frames, ignore_index=True)
    ts = label_events(ts)
    ts.to_parquet(data / "wearables.parquet", index=False)

    pd_day = ts.groupby(["patient_id", "day"])["excursion_event"].max()
    ur_day = ts.groupby(["patient_id", "day"])["urgency_event"].max()
    print(f"Wearables: {len(ts):,} samples ({SAMPLE_MIN}-min) x {ts.shape[1]} channels "
          f"-> data/wearables.parquet  [{time.time()-t0:.1f}s]")
    print(f"Patient-days with a Stage-2 excursion (>=160): {pd_day.mean():.1%} "
          f"({int(pd_day.sum())} of {len(pd_day)})")
    print(f"Patient-days reaching urgency (>=180):          {ur_day.mean():.1%} "
          f"({int(ur_day.sum())})")
    print(f"Patients with >=1 event: {pd_day.groupby('patient_id').max().mean():.1%}")
    print(f"Mean SBP {ts.sbp_mmhg.mean():.1f} | mean HR {ts.hr_bpm.mean():.1f} "
          f"| mean RMSSD {ts.rmssd_ms.mean():.1f} ms")
