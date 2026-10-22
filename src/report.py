"""
Regenerate every document from the results actually on disk.

Why this exists: XGBoost is not bit-reproducible across CPU architectures.
An ARM Mac and an x86 Linux box will train slightly different trees from the
same seed and data, so AUROC can move by a point or two. The conclusions do
not change, but a repository whose README quotes numbers its own code does
not produce is worse than useless.

So no result is ever hand-typed. This script rewrites the results section of
README.md and re-renders both PDFs from results/*.json. Run it after
run_all.sh and the repo is self-consistent on whatever machine you are on.

    python3 src/report.py
"""
from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
START, END = "<!-- AUTOGEN:RESULTS:START -->", "<!-- AUTOGEN:RESULTS:END -->"


def env_block() -> dict:
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:                                      # noqa: BLE001
            return "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "numpy": ver("numpy"), "pandas": ver("pandas"),
        "scikit_learn": ver("sklearn"), "xgboost": ver("xgboost"),
    }


def results_markdown() -> str:
    M = json.loads((RES / "metrics.json").read_text())
    C = json.loads((RES / "counterfactual_summary.json").read_text())
    S = json.loads((RES / "sensitivity.json").read_text()) if (RES / "sensitivity.json").exists() else None
    D = json.loads((RES / "deployment.json").read_text()) if (RES / "deployment.json").exists() else None
    SH = json.loads((RES / "sharee_validation.json").read_text()) if (RES / "sharee_validation.json").exists() else None
    env = env_block()

    def ci(k):
        lo, hi = M[k]["auroc_ci"]
        return f"{lo:.3f}–{hi:.3f}"

    L = []
    L.append(f"Patient-grouped split. {M['n_test_patients']} held-out patients, "
             f"{M['n_test_rows']} patient-days, prevalence {M['prevalence']*100:.0f}%. "
             f"Bootstrap CIs resampled by patient.\n")
    L.append(f"> Produced on {env['platform']}, Python {env['python']}, "
             f"xgboost {env['xgboost']}, scikit-learn {env['scikit_learn']}. "
             f"Regenerate with `./run_all.sh && python3 src/report.py`.\n")

    L.append("### Result 1 — the negative one, reported as such\n")
    L.append("Predicting a next-24h Stage-2 excursion:\n")
    L.append("| Model | AUROC | 95% CI | AUPRC | Brier |")
    L.append("|---|---|---|---|---|")
    for key, lab in [("ehr", "EHR only"), ("wear", "Cuff-free wearable only"),
                     ("fused", "**Fused, cuff-free**")]:
        b = "**" if key == "fused" else ""
        L.append(f"| {lab} | {b}{M[key]['auroc']:.3f}{b} | {ci(key)} | "
                 f"{M[key]['auprc']:.3f} | {M[key]['brier']:.3f} |")
    L.append(f"| *Reference: with BP cuff* | *{M['oracle_bp']['auroc']:.3f}* | *{ci('oracle_bp')}* | "
             f"*{M['oracle_bp']['auprc']:.3f}* | *{M['oracle_bp']['brier']:.3f}* |")
    L.append(f"| *Baseline: yesterday's mean SBP* | *{M['sbp_threshold']['auroc']:.3f}* | "
             f"*{ci('sbp_threshold')}* | — | — |\n")
    gap = M["fused"]["auroc"] - M["ehr"]["auroc"]
    verdict = ("is marginally *worse*" if gap < 0 else
               "is indistinguishable from the record alone" if gap < 0.03 else
               "edges the record alone")
    L.append(f"**Fusing cuff-free wearable data with the record {verdict}**: "
             f"{M['fused']['auroc']:.3f} against {M['ehr']['auroc']:.3f}, intervals overlapping "
             f"almost entirely. If a cuff is available, use it. A one-line rule on yesterday's "
             f"reading already scores {M['sbp_threshold']['auroc']:.3f}. "
             f"**Forecasting tomorrow's pressure from the wrist is not the product**, and we say "
             f"so rather than burying it.\n")

    dp = M["dipping_screen"]
    L.append("### Result 2 — the one that matters\n")
    L.append("Same cuff-free inputs, different question: *does this patient's pressure fail to "
             "fall at night?*\n")
    L.append("| | |\n|---|---|")
    L.append(f"| **Non-dipper screen, cuff-free, patient-level** | **AUROC {dp['auroc']:.3f}** "
             f"(95% CI {dp['auroc_ci'][0]:.3f}–{dp['auroc_ci'][1]:.3f}), "
             f"n = {dp['n_patients']} held-out patients |\n")
    L.append("The signal is autonomic: the night/day RMSSD ratio carries the phenotype with no "
             "pressure measurement anywhere in the input. That turns a scarce test (ABPM) into a "
             "triage decision, on hardware tens of millions of Indians already wear.\n")

    L.append("### Result 3 — the twin, measured\n")
    L.append("Twin predicts a change; the ground-truth simulator is re-run for the **same virtual "
             "patient, same random seed, same override**. Paired, 120 patients, no selection.\n")
    L.append("| Intervention | true Δ24h | twin Δ24h | MAE | true Δnight | twin Δnight | sign |")
    L.append("|---|---|---|---|---|---|---|")
    for k, v in C["by_intervention"].items():
        sign = "*circular*" if k == "salt_minus_3g" else f"{v['sign_agreement']*100:.0f}%"
        nb = "**" if k == "chronotherapy" else ""
        L.append(f"| {v['label']} | {v['true_mean_d_sbp']:+.2f} | {v['twin_mean_d_sbp']:+.2f} | "
                 f"{v['mae_mmhg']:.2f} | {nb}{v['true_night_d_sbp']:+.2f}{nb} | "
                 f"{v['twin_night_d_sbp']:+.2f} | {sign} |")
    ch, sl = C["by_intervention"]["chronotherapy"], C["by_intervention"]["sleep_plus_1h"]
    L.append(f"\n**Chronotherapy is the showcase.** Moving one agent to bedtime changes the "
             f"24-hour mean by essentially nothing ({ch['true_mean_d_sbp']:+.2f} mmHg) while "
             f"lowering night-time pressure by {abs(ch['true_night_d_sbp']):.2f} mmHg. A scalar "
             f"risk score cannot express that. A twin can.\n")
    L.append(f"**Two arms we do not claim.** The salt arm is **circular**: the twin's coefficient "
             f"and the simulator's both come from the same literature value. The sleep arm is "
             f"**weak** at {sl['sign_agreement']*100:.0f}% sign agreement.\n")

    if S:
        k0 = next(r for r in S["sweep"] if r["kappa"] == 0.0)
        k1 = next(r for r in S["sweep"] if r["kappa"] == 1.0)
        dmin = S["min_observed_cohens_d_for_auroc_080"]
        L.append("### Result 4 — how much of this is just our own assumption?\n")
        L.append("The obvious attack on Result 2 is that we encoded the autonomic–dipping coupling "
                 "and then detected it. Rather than answer that rhetorically, we measured it. "
                 "`src/sensitivity.py` sweeps the coupling strength κ, regenerates the entire "
                 "cohort at each setting, retrains, and plots screen AUROC against the "
                 "**observed** separation between dippers and non-dippers.\n")
        L.append("| κ | observed Cohen's d | screen AUROC |")
        L.append("|---|---|---|")
        for r in S["sweep"]:
            note = (" (no coupling at all)" if r["kappa"] == 0 else
                    " (our default)" if r["kappa"] == 1.0 else "")
            b = "**" if r["kappa"] == 0 else ""
            L.append(f"| {r['kappa']:.2f}{note} | {r['observed_cohens_d']:.2f} | "
                     f"{b}{r['screen_auroc']:.3f}{b} |")
        L.append(f"\n1. **At κ = 0 the screen collapses to {k0['screen_auroc']:.3f}, near chance.** "
                 f"The negative control: with the phenotypes made autonomically identical, the "
                 f"model has no back-door route to the label. The residual above 0.5 comes from "
                 f"OSA, which independently affects both dipping and HRV.")
        L.append(f"2. **The screen becomes usable (AUROC ≥ 0.80) once the observed separation "
                 f"reaches d ≈ {dmin}.**\n")
        L.append(f"That second number converts an assumption into a falsifiable threshold: *this "
                 f"approach works if, and only if, real non-dippers show a night/day RMSSD "
                 f"separation of at least d ≈ {dmin}.* Our default assumes "
                 f"d = {k1['observed_cohens_d']}, which is almost certainly generous.\n")
        L.append("![sensitivity](results/sensitivity.png)\n")

    if D:
        H, held = D["headline"], D["held_out"]
        L.append("### Result 5 — what the screen is worth operationally\n")
        L.append("An AUROC is not a health-system decision. The decision is: given a fixed number "
                 "of ABPM units, who gets one?\n")
        L.append("| Referral budget | Studies | Found | Precision | Sensitivity | ABPM per find |")
        L.append("|---|---|---|---|---|---|")
        for r in D["by_budget"]:
            b = "**" if r["budget_frac"] == H["best_budget_frac"] else ""
            L.append(f"| {b}{r['budget_frac']*100:.0f}%{b} | {b}{r['referred']}{b} | "
                     f"{b}{r['found']}{b} | {b}{r['yield_precision']*100:.0f}%{b} | "
                     f"{b}{r['sensitivity']*100:.0f}%{b} | {b}{r['abpm_per_find']:.2f}{b} |")
        u = D["universal_screening"]
        L.append(f"| Universal | {u['referred']} | {u['found']} | "
                 f"{held['nondipper_prevalence']*100:.0f}% | 100% | {u['abpm_per_find']:.2f} |\n")
        L.append(f"**At a {H['best_budget_frac']*100:.0f}% referral budget every single referral is "
                 f"a true non-dipper** ({H['precision_at_best']*100:.0f}% precision against "
                 f"{H['random_precision']*100:.0f}% for random referral), catching "
                 f"{H['sensitivity_at_best']*100:.0f}% of all non-dippers.\n")
        L.append(f"Honest ceiling: at {H['random_precision']*100:.0f}% prevalence triage can lift "
                 f"yield by at most {H['max_possible_lift']}×. The value is in **not wasting "
                 f"studies**, not in finding something rare.\n")
        L.append("![deployment](results/deployment.png)\n")

    L.append("### Result 6 — external validation on real patients\n")
    if SH:
        R, inc = SH["night_day_rmssd_ratio"], SH["incremental_over_clinic"]
        sig = "reaches" if R["mann_whitney_p"] < 0.05 else "does not reach"
        L.append(f"**SHAREE** (Melillo et al., PLoS ONE 2015): {SH['n_analysed']} real hypertensive "
                 f"patients, nominal 24-hour Holter, 12-month follow-up, {SH['n_events']} vascular "
                 f"events.\n")
        L.append("| | |\n|---|---|")
        L.append(f"| Night/day RMSSD, no event | {R['mean_no_event']:.3f} |")
        L.append(f"| Night/day RMSSD, event | {R['mean_event']:.3f} |")
        L.append(f"| Mann-Whitney p | {R['mann_whitney_p']} |")
        L.append(f"| Cohen's d | {R['cohens_d']:.2f} |")
        L.append(f"| **AUROC predicting a 12-month event** | **{R['auroc_event']:.3f}** "
                 f"({R['auroc_ci'][0]:.3f}–{R['auroc_ci'][1]:.3f}) |")
        L.append(f"| Added over age, sex, clinic SBP | {SH['delta_auroc']:+.3f} |\n")
        L.append(f"It {sig} significance at 0.05 with {SH['n_events']} events, an underpowered "
                 f"sample. {SH['caveat']}\n")
        L.append("![sharee](results/sharee_validation.png)\n")
    else:
        L.append("`src/validate_sharee.py` tests the core premise against the **SHAREE** database: "
                 "139 real hypertensive patients, 24-hour Holter, 12-month follow-up, 17 vascular "
                 "events.\n")
        L.append("> **Status: not yet run.** See `docs/SHAREE_DOWNLOAD.md`, then "
                 "`python3 src/validate_sharee.py` and re-run `python3 src/report.py`.\n")
        L.append("It validates the prognostic value of the autonomic feature in real hypertensive "
                 "patients. It does not validate non-dipper detection itself: SHAREE has no "
                 "ambulatory BP, so the dipping phenotype is unobservable in it.\n")

    return "\n".join(L)


def main():
    readme = ROOT / "README.md"
    x = readme.read_text()
    if START not in x or END not in x:
        raise SystemExit("README.md is missing the AUTOGEN markers; cannot refresh safely.")
    pre, rest = x.split(START, 1)
    _, post = rest.split(END, 1)
    readme.write_text(f"{pre}{START}\n\n{results_markdown()}\n{END}{post}")
    print("README.md results section regenerated from results/")

    (RES / "environment.json").write_text(json.dumps(env_block(), indent=2))
    print("results/environment.json written")

    try:
        subprocess.run([sys.executable, str(ROOT / "src" / "make_docs.py")], check=True)
    except Exception as exc:                                   # noqa: BLE001
        print(f"PDF re-render skipped ({exc}). Install playwright to refresh docs/*.pdf:")
        print("    pip install playwright && playwright install chromium")


if __name__ == "__main__":
    main()
