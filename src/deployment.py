"""
What the screen is worth operationally.

An AUROC is not a health-system decision. The decision is: given a fixed
number of ambulatory blood-pressure monitors, who gets one?

This script turns the cuff-free screen into that decision and costs it out.
Three strategies are compared at every referral budget:

    universal   every patient gets an ABPM          (perfect yield, impossible cost)
    random      refer k% at random                  (what happens with no triage)
    CardioTwin  refer the top k% by screen score

and reports the quantity a clinic actually cares about: **ABPMs spent per
non-dipper found**, plus the number needed to screen.

All inputs are our synthetic cohort, so these are illustrative mechanics, not
a costing of a real programme. Every assumption is printed with the result.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from config import SEED
from features import feature_columns, split_dynamic
from train_eval import _model

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# Stated assumptions, all adjustable, none of them load-bearing for the shape
# of the curve.
COHORT = 10_000          # hypertensive patients in a district programme
ABPM_COST_INR = 1_500    # indicative out-of-pocket cost of one ABPM study
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def patient_scores() -> pd.DataFrame:
    """Held-out per-patient screen score and true non-dipper status."""
    table = pd.read_csv(ROOT / "data" / "model_table.csv")
    daily = pd.read_csv(ROOT / "data" / "daily_summary.csv")
    nd = (daily.groupby("patient_id")["dip_ratio"].mean() < 0.10).astype(int)

    static, dynamic = feature_columns(table)
    cuffless, _ = split_dynamic(dynamic)
    cols = static + cuffless
    lab = table["patient_id"].map(nd).to_numpy()
    groups = table["patient_id"].to_numpy()
    y = table["y"].to_numpy()

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
                  .split(table, y, groups))
    X = table[cols].to_numpy(float)
    clf = CalibratedClassifierCV(_model(), method="isotonic", cv=3).fit(X[tr], lab[tr])
    p = clf.predict_proba(X[te])[:, 1]
    return (pd.DataFrame({"pid": groups[te], "score": p, "nondipper": lab[te]})
            .groupby("pid").mean().reset_index())


def main():
    df = patient_scores()
    n = len(df)
    prev = float(df.nondipper.mean())
    auc = float(roc_auc_score(df.nondipper, df.score))
    print(f"Held-out patients: {n}   non-dipper prevalence: {prev:.1%}   screen AUROC: {auc:.3f}\n")

    order = df.sort_values("score", ascending=False).reset_index(drop=True)
    total_nd = int(order.nondipper.sum())

    rows = []
    print(f"{'budget':>8}{'referred':>10}{'found':>8}{'yield':>8}"
          f"{'ABPM/find':>11}{'random find':>13}{'lift':>7}")
    for b in BUDGETS:
        k = max(int(round(b * n)), 1)
        found = int(order.nondipper.iloc[:k].sum())
        rand = b * total_nd
        rows.append({
            "budget_frac": b,
            "referred": k,
            "found": found,
            "sensitivity": round(found / total_nd, 4) if total_nd else None,
            "yield_precision": round(found / k, 4),
            "abpm_per_find": round(k / found, 2) if found else None,
            "random_expected_found": round(rand, 1),
            "lift_vs_random": round(found / rand, 2) if rand else None,
        })
        print(f"{b:>8.0%}{k:>10}{found:>8}{found/k:>8.1%}"
              f"{(k/found if found else float('nan')):>11.2f}{rand:>13.1f}"
              f"{(found/rand if rand else float('nan')):>7.2f}")

    # universal screening reference
    uni = {"referred": n, "found": total_nd, "abpm_per_find": round(n / total_nd, 2)}
    print(f"\n{'universal':>8}{n:>10}{total_nd:>8}{total_nd/n:>8.1%}{n/total_nd:>11.2f}")

    # scale to a district programme, with the assumption stated
    scaled = []
    for r in rows:
        ref = int(r["budget_frac"] * COHORT)
        found = ref * r["yield_precision"]
        scaled.append({
            "budget_frac": r["budget_frac"],
            "abpm_studies": ref,
            "nondippers_found": int(round(found)),
            "cost_inr": ref * ABPM_COST_INR,
            "cost_per_find_inr": int(round(ref * ABPM_COST_INR / found)) if found else None,
        })

    # Operating point: the smallest budget that still finds 80% of non-dippers.
    # Picking purely by efficiency rewards absurdly small budgets that miss
    # almost everyone, which is not a clinically useful recommendation.
    # At this prevalence the useful claim is precision, not sensitivity: how
    # much of your ABPM spend is wasted on patients who turn out to be normal
    # dippers. We headline the largest budget that still wastes nothing.
    clean = [r for r in rows if r["yield_precision"] >= 0.999]
    best = max(clean, key=lambda r: r["budget_frac"]) if clean else \
        max((r for r in rows if r["found"]), key=lambda r: r["yield_precision"])
    print(f"\nOperating point: refer the top {best['budget_frac']:.0%}. "
          f"{best['yield_precision']:.0%} of those referrals are true non-dippers "
          f"(random referral would be {prev:.0%}), catching {best['sensitivity']:.0%} "
          f"of all non-dippers for {best['referred']} studies instead of {n}.")
    print(f"Honest ceiling: at {prev:.0%} prevalence, triage can lift yield by at most "
          f"{1/prev:.2f}x. Most of the value here is avoiding wasted studies, not rarity.")
    print(f"At a {COHORT:,}-patient district programme, that is "
          f"{int(best['budget_frac']*COHORT):,} studies instead of {COHORT:,}, "
          f"catching {best['sensitivity']:.0%} of non-dippers.")

    out = {
        "assumptions": {
            "cohort_size_modelled": COHORT,
            "abpm_unit_cost_inr": ABPM_COST_INR,
            "source_of_prevalence": "our synthetic cohort, enriched for non-dipping",
            "caveat": ("Illustrative mechanics on synthetic data, not a costing of a real "
                       "programme. The shape of the curve is the result; the rupee figures "
                       "depend entirely on the stated unit cost."),
        },
        "held_out": {"n_patients": n, "nondipper_prevalence": round(prev, 4),
                     "screen_auroc": round(auc, 4)},
        "universal_screening": uni,
        "by_budget": rows,
        "district_programme": scaled,
        "headline": {
            "best_budget_frac": best["budget_frac"],
            "precision_at_best": best["yield_precision"],
            "sensitivity_at_best": best["sensitivity"],
            "abpm_per_find": best["abpm_per_find"],
            "random_precision": round(prev, 4),
            "max_possible_lift": round(1 / prev, 2),
            "note": ("At this prevalence the gain is in avoiding wasted studies rather than "
                     "in finding a rare condition. Reported as such."),
        },
    }
    RES.mkdir(exist_ok=True)
    (RES / "deployment.json").write_text(json.dumps(out, indent=2))
    _figure(order, total_nd, rows)
    print(f"\nWrote {RES/'deployment.json'} and deployment.png")


def _figure(order, total_nd, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(order)
    frac = np.arange(1, n + 1) / n
    found = np.cumsum(order.nondipper.to_numpy()) / total_nd

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), dpi=160)

    ax[0].plot(frac, found, color="#1b6ca8", lw=2.4, label="CardioTwin screen")
    ax[0].plot([0, 1], [0, 1], ls="--", lw=1.3, color="#9aa3ab", label="Random referral")
    ax[0].fill_between(frac, frac, found, where=found >= frac, color="#1b6ca8", alpha=.10)
    ax[0].set_xlabel("Fraction of patients referred for ABPM")
    ax[0].set_ylabel("Fraction of non-dippers found")
    ax[0].set_title("Referral yield against a fixed ABPM budget", fontsize=10.5)
    ax[0].legend(frameon=False, fontsize=9, loc="lower right")

    b = [r["budget_frac"] for r in rows]
    a = [r["abpm_per_find"] for r in rows]
    rnd = [1 / order.nondipper.mean()] * len(b)
    ax[1].plot(b, a, "o-", color="#1b6ca8", lw=2.2, ms=6, label="CardioTwin screen")
    ax[1].plot(b, rnd, ls="--", lw=1.3, color="#9aa3ab", label="Random / universal")
    ax[1].set_xlabel("Fraction of patients referred")
    ax[1].set_ylabel("ABPM studies per non-dipper found")
    ax[1].set_title("Cost efficiency of triage", fontsize=10.5)
    ax[1].legend(frameon=False, fontsize=9)

    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
        x.grid(True, color="#e4e8ec", lw=.8); x.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(RES / "deployment.png"); plt.close(fig)


if __name__ == "__main__":
    main()
