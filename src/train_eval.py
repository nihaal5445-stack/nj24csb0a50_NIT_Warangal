"""
Train and evaluate the CardioTwin next-24h risk model.

Design choices that matter for honesty of the result:

  * Patient-grouped split. No patient appears in both train and test, so the
    model cannot memorise an individual's baseline.
  * Ablations. EHR-only, wearable-only and fused models are trained
    identically, so the value of fusing the two streams is measured, not
    asserted.
  * Baselines. A persistence rule ("yesterday had an excursion") and a
    single-threshold rule on mean SBP, which is what current practice
    effectively does.
  * Bootstrap confidence intervals, patient-resampled.
  * Isotonic calibration on a held-out inner split, so the probability shown
    on the clinician dashboard means something.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from config import SEED
from features import feature_columns, split_dynamic

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

PALETTE = {"fused": "#1b6ca8", "ehr": "#9aa3ab", "wear": "#4aa181",
           "oracle_bp": "#7a5aa3", "base": "#c0603a", "grid": "#e4e8ec"}


def _model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=380, max_depth=4, learning_rate=0.045,
        subsample=0.85, colsample_bytree=0.75,
        reg_lambda=2.2, min_child_weight=6,
        eval_metric="logloss", random_state=SEED, n_jobs=4,
    )


def boot_ci(y, p, fn, n=2000, groups=None, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    vals = []
    for _ in range(n):
        take = rng.choice(uniq, uniq.size, replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in take])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run():
    df = pd.read_csv(ROOT / "data" / "model_table.csv")
    static, dynamic = feature_columns(df)
    y = df["y"].to_numpy()
    groups = df["patient_id"].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr, te = next(gss.split(df, y, groups))
    print(f"Train {len(tr)} rows / {df.patient_id.iloc[tr].nunique()} patients  |  "
          f"Test {len(te)} rows / {df.patient_id.iloc[te].nunique()} patients")

    cuffless, cuffed = split_dynamic(dynamic)
    # HEADLINE ARM is cuff-free: EHR + consumer-wearable channels only
    # (heart rate, HRV, sleep architecture, steps). No blood-pressure input.
    arms = {
        "ehr":   static,                       # record alone
        "wear":  cuffless,                     # wearable alone
        "fused": static + cuffless,            # the CardioTwin model
        "oracle_bp": static + cuffless + cuffed,   # reference upper bound only
    }
    results, preds = {}, {}

    for name, cols in arms.items():
        X = df[cols].to_numpy(dtype=float)
        clf = CalibratedClassifierCV(_model(), method="isotonic", cv=3)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        preds[name] = p
        auroc, auprc = roc_auc_score(y[te], p), average_precision_score(y[te], p)
        lo, hi = boot_ci(y[te], p, roc_auc_score, groups=groups[te])
        plo, phi = boot_ci(y[te], p, average_precision_score, groups=groups[te])
        results[name] = {
            "n_features": len(cols),
            "auroc": round(auroc, 4), "auroc_ci": [round(lo, 4), round(hi, 4)],
            "auprc": round(auprc, 4), "auprc_ci": [round(plo, 4), round(phi, 4)],
            "brier": round(brier_score_loss(y[te], p), 4),
        }
        print(f"  {name:6s} AUROC {auroc:.3f} [{lo:.3f}-{hi:.3f}]   "
              f"AUPRC {auprc:.3f} [{plo:.3f}-{phi:.3f}]   Brier {brier_score_loss(y[te], p):.3f}")

    # ---- baselines ----------------------------------------------------------
    persistence = df["excursions_in_window"].to_numpy()[te] / 7.0
    sbp_rule = df["sbp_mean_last"].to_numpy()[te]
    for bname, bp in [("persistence", persistence), ("sbp_threshold", sbp_rule)]:
        a = roc_auc_score(y[te], bp)
        lo, hi = boot_ci(y[te], bp, roc_auc_score, groups=groups[te])
        results[bname] = {"auroc": round(a, 4), "auroc_ci": [round(lo, 4), round(hi, 4)],
                          "auprc": round(average_precision_score(y[te], bp), 4)}
        print(f"  {bname:14s} AUROC {a:.3f} [{lo:.3f}-{hi:.3f}]")
    preds["base"] = persistence

    results["prevalence"] = round(float(y[te].mean()), 4)
    results["n_test_rows"] = int(len(te))
    results["n_test_patients"] = int(df.patient_id.iloc[te].nunique())

    # ---- operating point for the dashboard ----------------------------------
    pr, rc, th = precision_recall_curve(y[te], preds["fused"])
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    best = int(np.nanargmax(f1))
    results["operating_point"] = {
        "threshold": round(float(th[best]), 4),
        "precision": round(float(pr[best]), 4),
        "recall": round(float(rc[best]), 4),
        "f1": round(float(f1[best]), 4),
    }
    print(f"  operating point: thr {th[best]:.3f} -> precision {pr[best]:.2f} / recall {rc[best]:.2f}")

    # ---- figures ------------------------------------------------------------
    _fig_roc(y[te], preds, results)
    _fig_calibration(y[te], preds["fused"])
    _fig_importance(df, arms["fused"], tr, y)
    results["dipping_screen"] = dipping_task(df, static, cuffless, tr, te, groups)

    (RES / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RES/'metrics.json'} and 3 figures")
    return results


def dipping_task(df, static, cuffless, tr, te, groups):
    """Secondary task: detect the non-dipper phenotype WITHOUT a BP cuff.

    Nocturnal non-dipping is an independent cardiovascular risk marker, but
    confirming it normally requires 24-h ambulatory BP monitoring, which the
    large majority of Indian hypertensive patients never receive. If the
    autonomic and sleep signature alone can flag it, a consumer wearable can
    triage who actually needs the ABPM.
    """
    daily = pd.read_csv(ROOT / "data" / "daily_summary.csv")
    dip = daily.groupby("patient_id")["dip_ratio"].mean()
    nd = (dip < 0.10).astype(int)                      # <10% fall = non-dipper
    lab = df["patient_id"].map(nd).to_numpy()

    cols = static + cuffless
    X = df[cols].to_numpy(dtype=float)
    clf = CalibratedClassifierCV(_model(), method="isotonic", cv=3)
    clf.fit(X[tr], lab[tr])
    p = clf.predict_proba(X[te])[:, 1]

    # score once per patient, not per patient-day
    pf = pd.DataFrame({"pid": groups[te], "p": p, "y": lab[te]}).groupby("pid").mean()
    auroc = roc_auc_score(pf.y, pf.p)
    lo, hi = boot_ci(pf.y.to_numpy(), pf.p.to_numpy(), roc_auc_score,
                     groups=pf.index.to_numpy())
    print(f"  dipping screen (patient-level, cuff-free) AUROC {auroc:.3f} [{lo:.3f}-{hi:.3f}]  "
          f"n={len(pf)} patients, {pf.y.mean():.0%} non-dippers")
    return {"auroc": round(float(auroc), 4), "auroc_ci": [round(lo, 4), round(hi, 4)],
            "n_patients": int(len(pf)), "prevalence": round(float(pf.y.mean()), 4)}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=PALETTE["grid"], lw=0.8)
    ax.set_axisbelow(True)


def _fig_roc(y, preds, res):
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=160)
    order = [("fused", "CardioTwin: EHR + cuff-free wearable"),
             ("wear", "Cuff-free wearable only"),
             ("ehr", "EHR only"),
             ("oracle_bp", "Reference: adds BP cuff data"),
             ("base", "Persistence baseline (needs cuff)")]
    for k, lab in order:
        fpr, tpr, _ = roc_curve(y, preds[k])
        a = roc_auc_score(y, preds[k])
        ax.plot(fpr, tpr, color=PALETTE[k], lw=2.4 if k == "fused" else 1.6,
                label=f"{lab}  (AUROC {a:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#b8bfc6")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Next-24h Stage-2 hypertensive excursion\nheld-out patients, no cuff data in the headline model", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax); fig.tight_layout(); fig.savefig(RES / "roc.png"); plt.close(fig)


def _fig_calibration(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    xs, ys = [], []
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() > 5:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=160)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#b8bfc6", label="Perfect calibration")
    ax.plot(xs, ys, "o-", color=PALETTE["fused"], lw=2, ms=6, label="CardioTwin (isotonic)")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration, held-out patients", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style(ax); fig.tight_layout(); fig.savefig(RES / "calibration.png"); plt.close(fig)


def _fig_importance(df, cols, tr, y, top=16):
    m = _model().fit(df[cols].to_numpy(dtype=float)[tr], y[tr])
    imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)[:top][::-1]
    pretty = [c.replace("_", " ") for c in imp.index]
    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=160)
    ax.barh(pretty, imp.to_numpy(), color=PALETTE["fused"], height=0.68)
    ax.set_title("What the model actually uses (top 16)", fontsize=11)
    ax.set_xlabel("Gain importance")
    _style(ax); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(RES / "importance.png"); plt.close(fig)
    imp[::-1].to_csv(RES / "feature_importance.csv", header=["gain"])


if __name__ == "__main__":
    run()
