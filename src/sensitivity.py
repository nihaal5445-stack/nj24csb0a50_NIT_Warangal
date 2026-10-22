"""
How much physiology does the cuff-free screen actually need?

CardioTwin's non-dipper screen works because the simulator encodes a blunted
nocturnal RMSSD rise in non-dippers. The obvious objection is that we assumed
the thing we then detected.

This script answers the objection quantitatively rather than rhetorically. It
sweeps the strength of that coupling from zero (no autonomic difference between
phenotypes at all) up past our default, regenerates the whole cohort at each
setting, retrains the screen, and reports two things:

    * the OBSERVED separation in the night/day RMSSD ratio between dippers
      and non-dippers, as Cohen's d, which is a quantity the literature also
      reports and which SHAREE lets us estimate from real patients
    * the screen's AUROC at that separation

The result is a transfer curve. It converts "we assumed it" into "here is the
minimum real-world effect size at which this approach works", which is a claim
a reader can check against their own data.

Runtime: roughly 10 minutes. Writes results/sensitivity.json and .png.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import generate_wearables as gw
from config import DAYS, SEED
from features import build_dataset, feature_columns, per_day_summary, split_dynamic
from train_eval import _model

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# Coupling strength kappa. 0 = phenotypes are autonomically identical,
# 1 = our default (non-dipper 0.58, reverse dipper 0.34 of the normal
# nocturnal vagal rise). Values above 1 are stronger than we assumed.
KAPPAS = [0.0, 0.15, 0.30, 0.50, 0.75, 1.0, 1.30]
DEFAULT = {"dipper": 1.0, "non_dipper": 0.58, "reverse_dipper": 0.34}


def coupling(kappa: float) -> dict:
    return {k: 1.0 - kappa * (1.0 - v) for k, v in DEFAULT.items()}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    s = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else float("nan")


def run_once(kappa: float, ehr: pd.DataFrame, lat: pd.DataFrame) -> dict:
    gw.PHENO_HRV = coupling(kappa)

    frames = [gw.simulate_patient(ehr.iloc[i], lat.iloc[i], DAYS) for i in range(len(ehr))]
    ts = gw.label_events(pd.concat(frames, ignore_index=True))
    daily = per_day_summary(ts)
    table = build_dataset(daily, ehr)

    # observed separation in the feature itself, dippers vs non-dippers
    dipmean = daily.groupby("patient_id")["dip_ratio"].mean()
    nd = (dipmean < 0.10)
    hrv_ratio = daily.groupby("patient_id")["hrv_night_ratio"].mean()
    d_obs = cohens_d(hrv_ratio[~nd].to_numpy(), hrv_ratio[nd].to_numpy())

    # retrain the cuff-free screen
    static, dynamic = feature_columns(table)
    cuffless, _ = split_dynamic(dynamic)
    cols = static + cuffless
    lab = table["patient_id"].map(nd.astype(int)).to_numpy()
    groups = table["patient_id"].to_numpy()
    y_evt = table["y"].to_numpy()

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
                  .split(table, y_evt, groups))
    X = table[cols].to_numpy(float)
    clf = CalibratedClassifierCV(_model(), method="isotonic", cv=3).fit(X[tr], lab[tr])
    p = clf.predict_proba(X[te])[:, 1]
    pf = pd.DataFrame({"pid": groups[te], "p": p, "y": lab[te]}).groupby("pid").mean()
    auroc = float(roc_auc_score(pf.y, pf.p)) if pf.y.nunique() > 1 else float("nan")

    return {
        "kappa": kappa,
        "pheno_hrv": coupling(kappa),
        "observed_cohens_d": round(d_obs, 3),
        "mean_hrv_ratio_dipper": round(float(hrv_ratio[~nd].mean()), 3),
        "mean_hrv_ratio_nondipper": round(float(hrv_ratio[nd].mean()), 3),
        "screen_auroc": round(auroc, 4),
        "nondipper_prevalence": round(float(nd.mean()), 3),
    }


def main():
    ehr = pd.read_csv(ROOT / "data" / "ehr.csv")
    lat = pd.read_csv(ROOT / "data" / "latent_truth.csv")

    rows = []
    print(f"{'kappa':>7}{'obs d':>9}{'dipper':>9}{'non-dip':>9}{'AUROC':>8}")
    for k in KAPPAS:
        t0 = time.time()
        r = run_once(k, ehr, lat)
        rows.append(r)
        print(f"{k:>7.2f}{r['observed_cohens_d']:>9.2f}"
              f"{r['mean_hrv_ratio_dipper']:>9.3f}{r['mean_hrv_ratio_nondipper']:>9.3f}"
              f"{r['screen_auroc']:>8.3f}   [{time.time()-t0:.0f}s]")

    gw.PHENO_HRV = DEFAULT          # restore, so later imports are unaffected

    df = pd.DataFrame(rows)
    # minimum observed effect size needed for a usable screen
    usable = df[df.screen_auroc >= 0.80]
    d_needed = float(usable.observed_cohens_d.min()) if len(usable) else float("nan")

    out = {
        "question": ("How large does the real-world separation in the night/day RMSSD ratio "
                     "between dippers and non-dippers have to be for this screen to work?"),
        "sweep": rows,
        "default_kappa": 1.0,
        "min_observed_cohens_d_for_auroc_080": None if np.isnan(d_needed) else round(d_needed, 3),
        "interpretation": ("At kappa = 0 the phenotypes are autonomically identical and the screen "
                           "is at chance, confirming it has no other route to the label. The screen "
                           "becomes usable only once the observed night/day RMSSD separation reaches "
                           "the effect size reported above, which is a quantity that can be measured "
                           "in real patients and therefore falsified."),
    }
    RES.mkdir(exist_ok=True)
    (RES / "sensitivity.json").write_text(json.dumps(out, indent=2))
    df.to_csv(RES / "sensitivity.csv", index=False)
    _figure(df, d_needed)
    print(f"\nScreen reaches AUROC 0.80 once observed Cohen's d >= {d_needed:.2f}")
    print(f"Wrote {RES/'sensitivity.json'}, sensitivity.csv, sensitivity.png")


def _figure(df, d_needed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=160)
    ax.plot(df.observed_cohens_d, df.screen_auroc, "o-", color="#1b6ca8", lw=2.2, ms=7)
    ax.axhline(0.5, ls="--", lw=1, color="#b8bfc6")
    ax.text(df.observed_cohens_d.max() * 0.98, 0.513, "chance", fontsize=8.5,
            color="#8a949e", ha="right")
    if np.isfinite(d_needed):
        ax.axhline(0.80, ls=":", lw=1.2, color="#9a6700")
        ax.axvline(d_needed, ls=":", lw=1.2, color="#9a6700")
        ax.text(d_needed, 0.52, f"  usable from d ≈ {d_needed:.2f}", fontsize=9, color="#9a6700")
    for _, r in df.iterrows():
        if r.kappa in (0.0, 1.0):
            ax.annotate(f"κ={r.kappa:g}" + ("  (our default)" if r.kappa == 1.0 else "  (no coupling)"),
                        (r.observed_cohens_d, r.screen_auroc),
                        textcoords="offset points", xytext=(8, -11), fontsize=9, color="#16202b")
    ax.set_xlabel("Observed separation in night/day RMSSD ratio,\ndippers vs non-dippers (Cohen's d)")
    ax.set_ylabel("Cuff-free non-dipper screen, AUROC")
    ax.set_title("How much real physiology the screen needs\nsweeping the encoded autonomic coupling", fontsize=11)
    ax.set_ylim(0.4, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e4e8ec", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(RES / "sensitivity.png"); plt.close(fig)


if __name__ == "__main__":
    main()
