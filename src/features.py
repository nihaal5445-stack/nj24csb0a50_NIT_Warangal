"""
Two-stream feature fusion for CardioTwin.

One row per patient-day. Each row joins:

  STATIC  : the EHR record (demographics, ICD-10 problem list, labs, regimen,
            polygenic risk proxy)
  DYNAMIC : summaries of the preceding 7 days of wearable signal

and is labelled with whether a sustained Stage-2 hypertensive excursion
occurs in the FOLLOWING 24 hours.

The dynamic block deliberately encodes clinically meaningful constructs
rather than raw statistics:

  dip_ratio        nocturnal SBP fall; <10% is the non-dipper phenotype
  morning_surge    sleep-trough to post-wake rise, a known risk marker
  hrv_night_ratio  nocturnal parasympathetic rise; blunted in OSA/neuropathy
  arv_sbp          average real variability, a BP-variability risk metric
  sleep_debt       cumulative shortfall against the patient's own baseline

Latent generator traits are never used as features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HISTORY_DAYS, SAMPLE_MIN

STATIC_NUM = ["age", "bmi", "waist_cm", "lab_creatinine_mg_dl", "lab_egfr", "lab_hba1c_pct",
              "lab_ldl_mg_dl", "lab_sodium_mmol_l", "lab_potassium_mmol_l", "lab_uacr_mg_g",
              "genetic_prs_z", "rx_n_agents"]
DX_COLS = ["dx_t2dm", "dx_ckd3", "dx_osa", "dx_dyslipidaemia", "dx_prior_mi",
           "dx_prior_stroke", "dx_obesity"]


def _arv(x: np.ndarray) -> float:
    """Average real variability: mean absolute successive difference."""
    return float(np.mean(np.abs(np.diff(x)))) if x.size > 1 else 0.0


def _night_day(g: pd.DataFrame):
    night = g[g["asleep"]]
    day = g[~g["asleep"]]
    return night, day


def per_day_summary(ts: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 5-minute stream to one row per patient-day."""
    rows = []
    for (pid, d), g in ts.groupby(["patient_id", "day"], sort=False):
        night, day = _night_day(g)
        sbp = g["sbp_mmhg"].to_numpy()
        n_sbp = night["sbp_mmhg"].to_numpy()
        d_sbp = day["sbp_mmhg"].to_numpy()

        dip = 1.0 - (n_sbp.mean() / d_sbp.mean()) if n_sbp.size and d_sbp.size else np.nan

        # morning surge: first 2 h awake minus the lowest sleeping hour
        surge = np.nan
        if n_sbp.size > 12 and d_sbp.size > 12:
            trough = np.sort(n_sbp)[: max(len(n_sbp) // 12, 1)].mean()
            wake_win = day.sort_values("t_hours")["sbp_mmhg"].to_numpy()[: int(120 / SAMPLE_MIN)]
            if wake_win.size:
                surge = float(wake_win.mean() - trough)

        hrv_n = night["rmssd_ms"].mean() if len(night) else np.nan
        hrv_d = day["rmssd_ms"].mean() if len(day) else np.nan

        rows.append({
            "patient_id": pid, "day": int(d),
            "sbp_mean": sbp.mean(), "sbp_max": sbp.max(), "sbp_sd": sbp.std(),
            "sbp_night": n_sbp.mean() if n_sbp.size else np.nan,
            "sbp_dayt": d_sbp.mean() if d_sbp.size else np.nan,
            "dbp_mean": g["dbp_mmhg"].mean(),
            "dip_ratio": dip,
            "morning_surge": surge,
            "arv_sbp": _arv(sbp),
            "hr_mean": g["hr_bpm"].mean(),
            "hr_sleep_min": night["hr_bpm"].min() if len(night) else np.nan,
            "rmssd_mean": g["rmssd_ms"].mean(),
            "hrv_night_ratio": (hrv_n / hrv_d) if (hrv_d and hrv_d == hrv_d) else np.nan,
            "sleep_hours": len(night) * SAMPLE_MIN / 60.0,
            "sleep_frag": float(np.mean(np.abs(np.diff(g["asleep"].to_numpy().astype(int))))),
            "deep_frac": float((night["sleep_stage"] == 2).mean()) if len(night) else np.nan,
            "steps_total": g["steps_5min"].sum(),
            "excursion": int(g["excursion_event"].max()),
            "urgency": int(g["urgency_event"].max()),
        })
    return pd.DataFrame(rows).sort_values(["patient_id", "day"]).reset_index(drop=True)


def build_dataset(daily: pd.DataFrame, ehr: pd.DataFrame,
                  history: int = HISTORY_DAYS) -> pd.DataFrame:
    """One row per (patient, prediction day). Window = days [d-history, d]."""
    dyn_cols = ["sbp_mean", "sbp_max", "sbp_sd", "sbp_night", "sbp_dayt", "dbp_mean",
                "dip_ratio", "morning_surge", "arv_sbp", "hr_mean", "hr_sleep_min",
                "rmssd_mean", "hrv_night_ratio", "sleep_hours", "sleep_frag",
                "deep_frac", "steps_total"]
    out = []
    for pid, g in daily.groupby("patient_id", sort=False):
        g = g.sort_values("day").reset_index(drop=True)
        for i in range(history - 1, len(g) - 1):
            w = g.iloc[i - history + 1: i + 1]
            rec = {"patient_id": pid, "pred_day": int(g.loc[i, "day"])}
            for c in dyn_cols:
                v = w[c].to_numpy(dtype=float)
                v = v[~np.isnan(v)]
                if v.size == 0:
                    rec[f"{c}_mean7"] = rec[f"{c}_last"] = rec[f"{c}_trend"] = np.nan
                    continue
                rec[f"{c}_mean7"] = v.mean()
                rec[f"{c}_last"] = v[-1]
                rec[f"{c}_trend"] = (np.polyfit(np.arange(v.size), v, 1)[0] if v.size > 2 else 0.0)
            # recent event burden and sleep debt
            rec["excursions_in_window"] = int(w["excursion"].sum())
            rec["days_since_excursion"] = int(
                history - 1 - np.max(np.where(w["excursion"].to_numpy() == 1)[0])
            ) if w["excursion"].sum() else history
            base_sleep = g["sleep_hours"].iloc[: max(i - history + 1, 1)].mean()
            if not np.isfinite(base_sleep):
                base_sleep = w["sleep_hours"].mean()
            rec["sleep_debt_7d"] = float(np.clip(base_sleep * history - w["sleep_hours"].sum(), 0, None))
            # label: excursion on the NEXT day
            rec["y"] = int(g.loc[i + 1, "excursion"])
            rec["y_urgency"] = int(g.loc[i + 1, "urgency"])
            out.append(rec)

    X = pd.DataFrame(out)
    e = ehr.copy()
    for c in DX_COLS:
        e[c.replace("dx_", "has_")] = (e[c].fillna("").astype(str) != "").astype(int)
    e["sex_m"] = (e["sex"] == "M").astype(int)
    e["urban_i"] = e["urban"].astype(int)
    e["evening_dose"] = e["rx_dose_hours"].fillna("").astype(str).str.contains("21").astype(int)
    keep = ["patient_id"] + STATIC_NUM + [c.replace("dx_", "has_") for c in DX_COLS] + \
           ["sex_m", "urban_i", "evening_dose"]
    return X.merge(e[keep], on="patient_id", how="left")


# Channels that require a blood-pressure cuff. Most Indian hypertensive
# patients never get 24-h ambulatory monitoring and do not own a home cuff,
# so our headline model is not allowed to use any of these.
BP_DERIVED = ("sbp_", "dbp_", "dip_ratio", "morning_surge", "arv_sbp",
              "excursions_in_window", "days_since_excursion")


def is_bp_derived(col: str) -> bool:
    return any(col.startswith(b) or col == b for b in BP_DERIVED)


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Returns (static EHR features, dynamic wearable features)."""
    drop = {"patient_id", "pred_day", "y", "y_urgency"}
    static = set(STATIC_NUM) | {c.replace("dx_", "has_") for c in DX_COLS} | \
             {"sex_m", "urban_i", "evening_dose"}
    cols = [c for c in df.columns if c not in drop]
    return [c for c in cols if c in static], [c for c in cols if c not in static]


def split_dynamic(dynamic: list[str]) -> tuple[list[str], list[str]]:
    """Split wearable features into cuff-free and cuff-requiring."""
    cuffless = [c for c in dynamic if not is_bp_derived(c)]
    cuffed = [c for c in dynamic if is_bp_derived(c)]
    return cuffless, cuffed


if __name__ == "__main__":
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    data = root / "data"
    ts = pd.read_parquet(data / "wearables.parquet")
    ehr = pd.read_csv(data / "ehr.csv")

    daily = per_day_summary(ts)
    daily.to_csv(data / "daily_summary.csv", index=False)
    ds = build_dataset(daily, ehr)
    ds.to_csv(data / "model_table.csv", index=False)

    s, d = feature_columns(ds)
    print(f"Daily summary : {daily.shape}  -> data/daily_summary.csv")
    print(f"Model table   : {ds.shape}  -> data/model_table.csv")
    print(f"  static features : {len(s)}")
    print(f"  dynamic features: {len(d)}")
    print(f"  positives       : {ds.y.mean():.1%} next-day Stage-2 excursion "
          f"({int(ds.y.sum())} of {len(ds)})")
    print(f"  urgency         : {ds.y_urgency.mean():.1%}")
    print(f"\nMean nocturnal dip by phenotype (sanity check):")
    lat = pd.read_csv(data / "latent_truth.csv")
    chk = daily.merge(lat[["patient_id", "phenotype"]], on="patient_id")
    print(chk.groupby("phenotype")["dip_ratio"].mean().mul(100).round(1).to_string())
