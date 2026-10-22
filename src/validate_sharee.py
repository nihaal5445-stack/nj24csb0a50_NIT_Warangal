"""
External validation of CardioTwin's core premise on REAL patients.

CardioTwin's headline screen rests on one assumption: that nocturnal autonomic
modulation, measurable from a wearable, carries cardiovascular information that
a clinic blood-pressure reading does not. In the simulator we encoded that
assumption, so the 0.963 figure cannot by itself be evidence for it.

This script tests the assumption against data we did not generate.

DATASET
    SHAREE - Smart Health for Assessing the Risk of Events via ECG
    139 hypertensive patients, nominal 24-hour Holter, 3 leads at 128 Hz,
    12 months of follow-up. 17 patients had a vascular event
    (11 myocardial infarction, 3 stroke, 3 syncope).
    Melillo P, et al. PLoS ONE 2015. Open access on PhysioNet.

WHAT THIS DOES AND DOES NOT VALIDATE
    Validates   : that the night/day RMSSD ratio, the exact feature our
                  cuff-free screen depends on, carries real prognostic
                  information in real hypertensive patients, over and above
                  age, sex and clinic blood pressure.
    Does NOT    : validate non-dipper detection itself. SHAREE has no
                  ambulatory blood pressure, so the dipping phenotype is
                  unobservable here. That still needs paired ABPM + wearable
                  data, which we say plainly in docs/LIMITATIONS.md.

GETTING THE DATA (about 42 MB, the signal files are NOT needed)
    See docs/SHAREE_DOWNLOAD.md, or:

      mkdir -p data/sharee && cd data/sharee
      wget -r -np -nH --cut-dirs=2 -A '*.hea,*.qrs,info.txt' \
           https://archive.physionet.org/pn6/shareedb/

    Then:  python3 src/validate_sharee.py
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHAREE = ROOT / "data" / "sharee"
RES = ROOT / "results"

# Night and day windows, local clock time. Chosen before looking at any
# outcome, and matched to the windows our simulator uses.
NIGHT = (0, 5)      # 00:00-05:00, core sleep
DAY = (10, 20)      # 10:00-20:00, settled waking hours

# Published event list, used only to verify that info.txt parsed correctly.
EXPECTED_EVENTS = {
    2033, 2059, 2108, 2119, 2121, 2148, 2184, 2185, 2218,
    2289, 2294, 2304, 2339, 2348, 2373, 2403, 2412,
}


# --------------------------------------------------------------------------
# RR extraction
# --------------------------------------------------------------------------
def clean_rr(rr_ms: np.ndarray) -> np.ndarray:
    """Remove ectopic beats and detector artefact.

    Two standard filters: a physiological range, then a relative jump filter
    against the running median. The .qrs annotations ship unaudited, so this
    matters.
    """
    ok = (rr_ms > 300) & (rr_ms < 2000)
    rr = rr_ms[ok]
    if rr.size < 20:
        return rr
    med = pd.Series(rr).rolling(5, center=True, min_periods=1).median().to_numpy()
    return rr[np.abs(rr - med) < 0.20 * med]


def hrv(rr_ms: np.ndarray) -> dict:
    rr = clean_rr(rr_ms)
    if rr.size < 50:
        return {k: np.nan for k in ("rmssd", "sdnn", "pnn50", "mean_hr", "n_beats")}
    d = np.diff(rr)
    return {
        "rmssd": float(np.sqrt(np.mean(d ** 2))),
        "sdnn": float(np.std(rr, ddof=1)),
        "pnn50": float(np.mean(np.abs(d) > 50) * 100),
        "mean_hr": float(60000.0 / np.mean(rr)),
        "n_beats": int(rr.size),
    }


def record_features(rec_path: pathlib.Path) -> dict | None:
    import wfdb
    stem = str(rec_path.with_suffix(""))
    try:
        hdr = wfdb.rdheader(stem)
        ann = wfdb.rdann(stem, "qrs")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  ! {rec_path.name}: {exc}")
        return None

    fs = float(hdr.fs)
    beats_s = np.asarray(ann.sample, dtype=float) / fs
    if beats_s.size < 1000:
        print(f"  ! {rec_path.stem}: only {beats_s.size} beats, skipping")
        return None

    base = hdr.base_time or dt.time(0, 0)
    start_h = base.hour + base.minute / 60 + base.second / 3600
    clock_h = (start_h + beats_s / 3600.0) % 24.0

    rr_all = np.diff(beats_s) * 1000.0
    clock_mid = clock_h[:-1]

    def window(lo, hi):
        m = (clock_mid >= lo) & (clock_mid < hi) if lo < hi else \
            ((clock_mid >= lo) | (clock_mid < hi))
        return hrv(rr_all[m])

    night, day = window(*NIGHT), window(*DAY)
    out = {"record": int(rec_path.stem), "duration_h": round(beats_s[-1] / 3600, 1)}
    out |= {f"night_{k}": v for k, v in night.items()}
    out |= {f"day_{k}": v for k, v in day.items()}

    # The CardioTwin feature. >1 means the normal overnight vagal rise is
    # present; at or below 1 it is blunted.
    out["night_day_rmssd_ratio"] = (night["rmssd"] / day["rmssd"]
                                    if day["rmssd"] and day["rmssd"] == day["rmssd"] else np.nan)
    out["night_day_hr_ratio"] = (night["mean_hr"] / day["mean_hr"]
                                 if day["mean_hr"] and day["mean_hr"] == day["mean_hr"] else np.nan)
    return out


# --------------------------------------------------------------------------
# clinical table
# --------------------------------------------------------------------------
def load_info() -> pd.DataFrame:
    p = SHAREE / "info.txt"
    if not p.exists():
        raise SystemExit(f"Missing {p}. See docs/SHAREE_DOWNLOAD.md")
    rows, header = [], None
    for line in p.read_text(errors="ignore").splitlines():
        parts = [c.strip() for c in line.replace(",", "\t").split("\t") if c.strip() != ""]
        if len(parts) < 5:
            continue
        if header is None and parts[0].lower().startswith("record"):
            header = parts
            continue
        if header and parts[0].lstrip("0").isdigit():
            rows.append(parts[: len(header)])
    if header is None:
        raise SystemExit("Could not parse info.txt header")

    cols = ["record", "sex", "age", "weight", "height", "bsa", "bmi", "smoker",
            "sbp", "dbp", "imt_max", "lvmi", "ef", "event"][: len(header)]
    df = pd.DataFrame(rows, columns=cols)
    df["record"] = df["record"].astype(int)
    for c in ("age", "weight", "height", "bsa", "bmi", "sbp", "dbp", "imt_max", "lvmi", "ef"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["event"] = (~df["event"].astype(str).str.strip().str.lower().isin(["none", "no", ""])).astype(int)
    df["male"] = df["sex"].astype(str).str.upper().str.startswith("M").astype(int)
    return df


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def boot_auc(y, x, n=5000, seed=7):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], x[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if not SHAREE.exists():
        raise SystemExit(f"Missing {SHAREE}. See docs/SHAREE_DOWNLOAD.md for the download command.")

    qrs = sorted(SHAREE.rglob("*.qrs"))
    if not qrs:
        raise SystemExit(f"No .qrs files under {SHAREE}. See docs/SHAREE_DOWNLOAD.md")
    print(f"Found {len(qrs)} SHAREE records. Extracting HRV, this takes a minute or two.")

    feats = [f for f in (record_features(p) for p in qrs) if f]
    F = pd.DataFrame(feats)
    info = load_info()

    got = set(info.loc[info.event == 1, "record"])
    print(f"info.txt parsed: {len(info)} patients, {info.event.sum()} with a vascular event")
    if got != EXPECTED_EVENTS:
        print(f"  ! event list differs from published: missing {EXPECTED_EVENTS - got}, "
              f"extra {got - EXPECTED_EVENTS}")

    df = F.merge(info, on="record", how="inner").dropna(subset=["night_day_rmssd_ratio"])
    print(f"Analysable: {len(df)} patients, {int(df.event.sum())} events\n")

    y = df["event"].to_numpy()
    r = df["night_day_rmssd_ratio"].to_numpy()

    ev, no = r[y == 1], r[y == 0]
    u, p_mw = stats.mannwhitneyu(ev, no, alternative="two-sided")
    # higher ratio should be protective, so score the negated value
    auc = roc_auc_score(y, -r)
    lo, hi = boot_auc(y, -r)
    d = (no.mean() - ev.mean()) / np.sqrt(((no.std(ddof=1) ** 2) + (ev.std(ddof=1) ** 2)) / 2)

    print("NIGHT/DAY RMSSD RATIO - the CardioTwin feature, on real patients")
    print(f"  no event : {no.mean():.3f} +/- {no.std(ddof=1):.3f}   (n={len(no)})")
    print(f"  event    : {ev.mean():.3f} +/- {ev.std(ddof=1):.3f}   (n={len(ev)})")
    print(f"  Mann-Whitney p = {p_mw:.4f}   Cohen's d = {d:.2f}")
    print(f"  AUROC for predicting a 12-month vascular event = {auc:.3f} [{lo:.3f}-{hi:.3f}]\n")

    # does it add anything over the clinic reading?
    comp = {}
    base_cols = ["age", "male", "sbp"]
    sub = df.dropna(subset=base_cols)
    yb = sub["event"].to_numpy()
    for name, cols in [("clinic only (age, sex, SBP)", base_cols),
                       ("clinic + night/day RMSSD", base_cols + ["night_day_rmssd_ratio"])]:
        X = StandardScaler().fit_transform(sub[cols].to_numpy(float))
        m = LogisticRegression(max_iter=2000, C=1.0).fit(X, yb)
        a = roc_auc_score(yb, m.predict_proba(X)[:, 1])
        comp[name] = round(float(a), 4)
        print(f"  {name:32s} AUROC {a:.3f}")
    delta = comp["clinic + night/day RMSSD"] - comp["clinic only (age, sex, SBP)"]
    print(f"  {'added by the autonomic feature':32s}       {delta:+.3f}\n")

    out = {
        "dataset": "SHAREE (Melillo et al., PLoS ONE 2015), 24-h Holter, real hypertensive patients",
        "n_analysed": int(len(df)), "n_events": int(y.sum()),
        "night_windows": list(NIGHT), "day_window": list(DAY),
        "night_day_rmssd_ratio": {
            "mean_no_event": round(float(no.mean()), 4),
            "mean_event": round(float(ev.mean()), 4),
            "mann_whitney_p": round(float(p_mw), 5),
            "cohens_d": round(float(d), 3),
            "auroc_event": round(float(auc), 4),
            "auroc_ci": [round(lo, 4), round(hi, 4)],
        },
        "incremental_over_clinic": comp,
        "delta_auroc": round(float(delta), 4),
        "caveat": ("SHAREE has no ambulatory BP, so this validates the prognostic value of the "
                   "nocturnal autonomic feature, not non-dipper detection itself."),
    }
    RES.mkdir(exist_ok=True)
    (RES / "sharee_validation.json").write_text(json.dumps(out, indent=2))
    df.to_csv(RES / "sharee_features.csv", index=False)
    _figure(df)
    print(f"Wrote {RES/'sharee_validation.json'}, sharee_features.csv and sharee_validation.png")

    if p_mw > 0.05:
        print("\nNOTE: not significant at 0.05. With 17 events this study is underpowered, and a")
        print("null here is a real result. Report it as such rather than dropping the analysis.")


def _figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), dpi=160)
    groups = [df.loc[df.event == 0, "night_day_rmssd_ratio"],
              df.loc[df.event == 1, "night_day_rmssd_ratio"]]
    bp = ax[0].boxplot(groups, labels=["No event", "Vascular event"], widths=.55,
                       patch_artist=True, medianprops=dict(color="#16202b"))
    for patch, c in zip(bp["boxes"], ["#cfe0ee", "#f2c3bd"]):
        patch.set_facecolor(c)
    for i, g in enumerate(groups, start=1):
        ax[0].scatter(np.random.normal(i, 0.055, len(g)), g, s=13, alpha=.6,
                      color="#1b6ca8" if i == 1 else "#b3141f", zorder=3)
    ax[0].axhline(1.0, ls="--", lw=1, color="#8a949e")
    ax[0].set_ylabel("Night / day RMSSD ratio")
    ax[0].set_title("Nocturnal autonomic rise by 12-month outcome\nSHAREE, real hypertensive patients", fontsize=10)

    from sklearn.metrics import roc_curve, roc_auc_score
    y = df["event"].to_numpy()
    for lab, x, c in [("Night/day RMSSD", -df["night_day_rmssd_ratio"].to_numpy(), "#1b6ca8"),
                      ("Clinic SBP", df["sbp"].to_numpy(), "#9aa3ab")]:
        m = ~np.isnan(x)
        fpr, tpr, _ = roc_curve(y[m], x[m])
        ax[1].plot(fpr, tpr, color=c, lw=2, label=f"{lab} (AUROC {roc_auc_score(y[m], x[m]):.3f})")
    ax[1].plot([0, 1], [0, 1], ls="--", lw=1, color="#b8bfc6")
    ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
    ax[1].set_title("Predicting a 12-month vascular event", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(True, color="#e4e8ec", lw=.8); a.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(RES / "sharee_validation.png"); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
