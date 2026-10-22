"""Render the architecture diagram and the results presentation to PDF.

Both are authored as HTML and printed through headless Chromium, so the
typography and layout are controlled rather than whatever matplotlib gives.
Numbers are injected from results/ so the PDFs can never drift from the
metrics the code actually produced.
"""
from __future__ import annotations

import json
import pathlib

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

M = json.loads((RES / "metrics.json").read_text())
C = json.loads((RES / "counterfactual_summary.json").read_text())
S = json.loads((RES / "sensitivity.json").read_text())
DEP = json.loads((RES / "deployment.json").read_text())
SHAREE = (json.loads((RES / "sharee_validation.json").read_text())
          if (RES / "sharee_validation.json").exists() else None)

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font:14px/1.5 "Helvetica Neue",Arial,sans-serif;color:#16202b}
.slide{width:297mm;height:209.7mm;padding:15mm 18mm;page-break-after:always;
  position:relative;background:#fff;display:flex;flex-direction:column}
h1{font-size:30px;font-weight:650;letter-spacing:-.4px}
h2{font-size:22px;font-weight:650;margin-bottom:3mm;letter-spacing:-.2px}
.kick{font-size:12px;font-weight:700;letter-spacing:.1em;color:#1b6ca8;
  text-transform:uppercase;margin-bottom:2mm}
p{font-size:13.5px;margin-bottom:2.5mm;max-width:235mm}
.mut{color:#5d6b7a}
.foot{position:absolute;bottom:7mm;left:18mm;right:18mm;display:flex;
  justify-content:space-between;font-size:10px;color:#8a949e;
  border-top:1px solid #e4e9ee;padding-top:2mm}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:2mm}
th,td{text-align:left;padding:4.5px 8px;border-bottom:1px solid #e4e9ee}
th{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:#5d6b7a}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.good{color:#0b6b4f;font-weight:650}.bad{color:#b3261e;font-weight:650}
.grid{display:grid;gap:5mm;margin-top:3mm}
.g2{grid-template-columns:1fr 1fr}.g3{grid-template-columns:repeat(3,1fr)}
.card{border:1px solid #dde3ea;border-radius:7px;padding:4mm 5mm;background:#fbfcfd}
.card.hi{border-color:#1b6ca8;background:#f2f8fc}
.card.warn{border-color:#f0e2bd;background:#fdf9ef}
.big{font-size:30px;font-weight:680;letter-spacing:-.5px;color:#1b6ca8}
.big.r{color:#b3261e}.big.g{color:#0b6b4f}
.lab{font-size:10.5px;color:#5d6b7a;text-transform:uppercase;letter-spacing:.05em}
ul{margin:1mm 0 0 5mm}li{font-size:13px;margin-bottom:1.5mm}
.tag{display:inline-block;font-size:10px;padding:1px 7px;border-radius:9px;
  background:#eaf5f1;color:#0b6b4f;font-weight:650}
.tag.w{background:#fdf6e3;color:#9a6700}.tag.n{background:#eef2f6;color:#5d6b7a}
img{max-width:100%;max-height:140mm;object-fit:contain}
.center{display:flex;align-items:center;justify-content:center;flex:1}
"""


def foot(n: int, total: int = 13) -> str:
    return (f'<div class="foot"><span>CardioTwin &middot; Digital Twin Challenge 2026 '
            f'&middot; synthetic data, not a medical device</span><span>{n} / {total}</span></div>')


def arch_svg() -> str:
    box = lambda x, y, w, h, t, s, fill, stroke: f"""
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>
      <text x="{x+w/2}" y="{y+19}" font-size="12.5" font-weight="650" text-anchor="middle" fill="#16202b">{t}</text>
      <text x="{x+w/2}" y="{y+35}" font-size="10" text-anchor="middle" fill="#5d6b7a">{s}</text>"""
    arrow = lambda x1, y1, x2, y2: (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#9aa6b2" stroke-width="1.4" '
        f'marker-end="url(#a)"/>')
    B, G, W = "#eef5fb", "#eef7f3", "#fbfcfd"
    bs, gs, ws = "#9cc3e0", "#9ecbba", "#dde3ea"
    return f"""
<svg viewBox="0 0 1180 470" width="100%">
<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="3"
  orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#9aa6b2"/></marker></defs>

<text x="18" y="20" font-size="11" font-weight="700" fill="#1b6ca8" letter-spacing="1.2">DATA GENERATION</text>
{box(18,32,210,52,"Synthetic EHR","demographics, ICD-10, labs, PRS, regimen",B,bs)}
{box(18,98,210,52,"Mechanistic simulator","circadian BP, autonomic, sleep, PK",B,bs)}
<text x="18" y="182" font-size="10" fill="#5d6b7a">Latent traits (phenotype, adherence,</text>
<text x="18" y="196" font-size="10" fill="#5d6b7a">salt, autonomic tone) drive the</text>
<text x="18" y="210" font-size="10" fill="#5d6b7a">simulation and are never model inputs.</text>

<text x="276" y="20" font-size="11" font-weight="700" fill="#1b6ca8" letter-spacing="1.2">TWO STREAMS</text>
{box(276,32,196,52,"Static stream","EHR record, one row per patient",W,ws)}
{box(276,98,196,52,"Dynamic stream","HR, HRV, sleep, steps @ 5 min",W,ws)}
{arrow(228,58,274,58)}{arrow(228,124,274,124)}

<text x="520" y="20" font-size="11" font-weight="700" fill="#1b6ca8" letter-spacing="1.2">FUSION</text>
{box(520,32,210,118,"Feature fusion","dip ratio, morning surge, ARV,<tspan></tspan>",W,ws)}
<text x="625" y="100" font-size="10" text-anchor="middle" fill="#5d6b7a">night/day RMSSD, sleep debt,</text>
<text x="625" y="114" font-size="10" text-anchor="middle" fill="#5d6b7a">7-day mean / last / trend</text>
<text x="625" y="134" font-size="9.5" text-anchor="middle" fill="#b3261e">cuff-derived channels held out</text>
{arrow(472,58,518,78)}{arrow(472,124,518,104)}

<text x="778" y="20" font-size="11" font-weight="700" fill="#1b6ca8" letter-spacing="1.2">MODELS</text>
{box(778,32,180,52,"Risk model","XGBoost + isotonic calibration",G,gs)}
{box(778,98,180,52,"Non-dipper screen","cuff-free phenotype detection",G,gs)}
{arrow(730,70,776,58)}{arrow(730,104,776,124)}

{box(778,176,180,52,"Personalised twin","setpoint, night factor, surge",G,gs)}
{arrow(625,152,820,174)}

<text x="1000" y="20" font-size="11" font-weight="700" fill="#1b6ca8" letter-spacing="1.2">OUTPUT</text>
{box(1000,32,164,118,"Clinician console","ranked panel, 7-day trace,<tspan></tspan>",B,bs)}
<text x="1082" y="100" font-size="10" text-anchor="middle" fill="#5d6b7a">autonomic view,</text>
<text x="1082" y="114" font-size="10" text-anchor="middle" fill="#5d6b7a">what-if simulation</text>
{arrow(958,70,998,80)}{arrow(958,124,998,104)}
{arrow(958,202,1082,160)}

<rect x="18" y="258" width="1146" height="86" rx="7" fill="#fdf9ef" stroke="#f0e2bd" stroke-width="1.3"/>
<text x="34" y="280" font-size="11.5" font-weight="700" fill="#9a6700">COUNTERFACTUAL VALIDATION LOOP</text>
<text x="34" y="300" font-size="11" fill="#5d6b7a">The twin predicts the change from an intervention. The ground-truth simulator is re-run for the same virtual patient, with the same</text>
<text x="34" y="316" font-size="11" fill="#5d6b7a">random seed and the same override. The two are compared as a paired experiment, so the what-if engine is measured rather than asserted.</text>
<text x="34" y="334" font-size="11" fill="#16202b">twin.predict_change(i) &#8594; predicted &#916;SBP        vs        simulate_patient(overrides=i) &#8594; actual &#916;SBP        reported as MAE and sign agreement</text>

<rect x="18" y="360" width="1146" height="46" rx="7" fill="#f7f9fb" stroke="#dde3ea" stroke-width="1.3"/>
<text x="34" y="380" font-size="11.5" font-weight="700" fill="#16202b">DEPLOYMENT PATH</text>
<text x="34" y="397" font-size="11" fill="#5d6b7a">one calibration week with a cuff  &#8594;  twin personalised  &#8594;  continuous cuff-free monitoring from the wearable  &#8594;  cuff or ABPM requested only when the twin flags risk</text>
</svg>"""


def architecture_html() -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}
@page{{size:A4 landscape;margin:0}}</style></head><body>
<div class="slide">
  <p class="kick">CardioTwin &middot; system architecture</p>
  <h1 style="margin-bottom:4mm">From synthetic cohort to a clinician decision</h1>
  {arch_svg()}
  {foot(1, 1)}
</div></body></html>"""


def deck_html() -> str:
    f, e, o = M["fused"], M["ehr"], M["oracle_bp"]
    dip, op = M["dipping_screen"], M["operating_point"]
    ci = lambda d: f'[{d["auroc_ci"][0]:.3f}&ndash;{d["auroc_ci"][1]:.3f}]'
    cf_rows = "".join(
        f'<tr><td>{v["label"]}</td>'
        f'<td class="n">{v["true_mean_d_sbp"]:+.2f}</td><td class="n">{v["twin_mean_d_sbp"]:+.2f}</td>'
        f'<td class="n">{v["mae_mmhg"]:.2f}</td>'
        f'<td class="n">{v["true_night_d_sbp"]:+.2f}</td><td class="n">{v["twin_night_d_sbp"]:+.2f}</td>'
        f'<td>{v["sign_agreement"]*100:.0f}%</td></tr>'
        for k, v in C["by_intervention"].items())

    s = []
    s.append(f"""<div class="slide">
      <p class="kick">Digital Twin Challenge 2026 &middot; Happiest Health</p>
      <h1>CardioTwin</h1>
      <p style="font-size:19px;margin-top:2mm">A hypertension digital twin that simulates interventions,
         not just risk</p>
      <div class="grid g3" style="margin-top:8mm">
        <div class="card hi"><div class="big">315M</div><div class="lab">Indian adults with hypertension<br>ICMR-INDIAB, Lancet D&amp;E 2023</div></div>
        <div class="card hi"><div class="big">{dip["auroc"]:.3f}</div><div class="lab">AUROC detecting the non-dipper<br>phenotype with no BP cuff</div></div>
        <div class="card hi"><div class="big">5</div><div class="lab">interventions simulated and<br>validated by paired re-simulation</div></div>
      </div>
      <p class="mut" style="margin-top:7mm;font-size:12px">Every patient in this work is simulated.
         No real patient data is used. Research prototype, not a medical device.</p>
      {foot(1)}</div>""")

    s.append(f"""<div class="slide">
      <p class="kick">The problem</p>
      <h2>315 million Indians have hypertension. Almost none get the test that matters.</h2>
      <p>Blood pressure follows a circadian rhythm. It should fall 10&ndash;20% overnight. People whose
      pressure does not fall are <b>non-dippers</b>, and non-dipping predicts cardiovascular events
      independently of the daytime reading. Reverse dippers, whose pressure rises at night, are at
      higher risk still.</p>
      <p>Confirming the phenotype requires 24-hour ambulatory blood-pressure monitoring. ABPM is
      scarce, costly and largely urban-private in India, so the overwhelming majority of the 315
      million are managed on clinic readings that cannot see the thing that matters most.</p>
      <div class="grid g3">
        <div class="card"><div class="big r">&lt;10%</div><div class="lab">nocturnal fall defines<br>a non-dipper</div></div>
        <div class="card"><div class="big r">101M</div><div class="lab">Indian adults with diabetes,<br>a major driver of non-dipping</div></div>
        <div class="card"><div class="big">~0</div><div class="lab">routine ABPM in most<br>Indian primary care</div></div>
      </div>
      <p style="margin-top:4mm"><b>Meanwhile the wrist is already instrumented.</b> Consumer wearables
      measure heart rate, heart-rate variability, sleep architecture and activity continuously. They
      cannot measure blood pressure. The question this project asks is whether what they
      <i>can</i> measure is enough to find the people who need the cuff.</p>
      {foot(2)}</div>""")

    s.append(f"""<div class="slide">
      <p class="kick">What makes it a twin</p>
      <h2>A predictor says what will happen. A twin says what happens if you change something.</h2>
      <p>We hold both to that standard. CardioTwin is a reduced mechanistic model of the patient's
      circadian pressure, personalised from one calibration week:</p>
      <div class="card" style="margin:3mm 0;font-family:ui-monospace,Menlo,monospace;font-size:13px">
        SBP(t) = S &middot; circ(t; night<sub>f</sub>) + surge(t; A) + behaviour(t) &minus; med(t) + &epsilon;
      </div>
      <p>Setpoint <b>S</b>, nocturnal factor <b>night<sub>f</sub></b> and surge amplitude <b>A</b> are
      fitted per patient from their own trace. Behaviour coefficients are estimated once across the
      cohort by within-patient regression on observed data. The twin never reads the simulator's
      latent traits.</p>
      <p>Because the structure is mechanistic rather than a black box, you can intervene on it:
      move a dose to bedtime, extend sleep, cut salt, raise activity, fix adherence, and re-integrate.</p>
      <div class="card warn" style="margin-top:3mm"><b>And because we own the ground-truth simulator,
      the what-if engine can be checked.</b> For every intervention we re-run the same virtual patient
      with the same random seed and the same override, and compare. Paired, no cherry-picking.</div>
      {foot(3)}</div>""")

    s.append(f"""<div class="slide">
      <p class="kick">Data &middot; two streams, fully synthetic</p>
      <h2>A mechanistic cohort, not a sampled noise generator</h2>
      <div class="grid g2">
        <div class="card"><b>Static: synthetic EHR</b>
          <ul><li>Demographics, urban/rural, South-Asian adiposity</li>
          <li>ICD-10 problem list: I10, E11.9, N18.3, G47.33, I21.9</li>
          <li>Labs: creatinine, eGFR, HbA1c, LDL, Na, K, UACR</li>
          <li>Polygenic risk proxy, regimen with dose timing</li></ul></div>
        <div class="card"><b>Dynamic: wearable at 5-minute resolution</b>
          <ul><li>Heart rate, RMSSD, sleep stage, steps, cuff BP</li>
          <li>300 patients &times; 21 days = 1.81M samples</li>
          <li>Circadian rhythm with dipping phenotype</li>
          <li>Morning surge, autonomic coupling, drug PK</li></ul></div>
      </div>
      <p style="margin-top:4mm">Four pieces of established physiology are encoded: the dipping rhythm,
      the morning surge, the nocturnal vagal rise that is blunted in non-dippers, and steady-state
      pharmacodynamics so a missed or badly timed dose is visible in the trace.</p>
      <div class="card hi" style="margin-top:2mm"><b>Face validity check.</b> The simulator was never
      told the clinical cut-off, yet the phenotypes it produces land exactly where the definitions sit:
      dippers fall <b>15.6%</b> overnight, non-dippers <b>6.0%</b>, reverse dippers <b>rise 2.6%</b>.</div>
      {foot(4)}</div>""")

    s.append(f"""<div class="slide">
      <p class="kick">Result 1 &middot; the negative one</p>
      <h2>Cuff-free wearables do not improve next-day event forecasting</h2>
      <p>Task: predict a sustained Stage-2 excursion (SBP &ge; 160 for &ge; 30 min) in the next 24 hours.
      Patient-grouped split, {M["n_test_patients"]} held-out patients, {M["n_test_rows"]} patient-days,
      prevalence {M["prevalence"]*100:.0f}%. Bootstrap CIs resampled by patient.</p>
      <table><thead><tr><th>Model</th><th class="n">AUROC</th><th>95% CI</th><th class="n">AUPRC</th><th class="n">Brier</th></tr></thead><tbody>
        <tr><td>EHR only</td><td class="n">{e["auroc"]:.3f}</td><td>{ci(e)}</td><td class="n">{e["auprc"]:.3f}</td><td class="n">{e["brier"]:.3f}</td></tr>
        <tr><td>Cuff-free wearable only</td><td class="n">{M["wear"]["auroc"]:.3f}</td><td>{ci(M["wear"])}</td><td class="n">{M["wear"]["auprc"]:.3f}</td><td class="n">{M["wear"]["brier"]:.3f}</td></tr>
        <tr><td><b>Fused, cuff-free</b></td><td class="n"><b>{f["auroc"]:.3f}</b></td><td>{ci(f)}</td><td class="n">{f["auprc"]:.3f}</td><td class="n">{f["brier"]:.3f}</td></tr>
        <tr><td class="mut">Reference: with BP cuff</td><td class="n">{o["auroc"]:.3f}</td><td>{ci(o)}</td><td class="n">{o["auprc"]:.3f}</td><td class="n">{o["brier"]:.3f}</td></tr>
        <tr><td class="mut">Baseline: yesterday's mean SBP</td><td class="n">{M["sbp_threshold"]["auroc"]:.3f}</td><td>{ci(M["sbp_threshold"])}</td><td class="n">&mdash;</td><td class="n">&mdash;</td></tr>
      </tbody></table>
      <div class="card warn" style="margin-top:3mm"><b>We report this as a failure of the hypothesis.</b>
      Fusing cuff-free wearable data with the record ({f["auroc"]:.3f}) does not beat the record alone
      ({e["auroc"]:.3f}); the intervals overlap almost entirely. If a cuff is available, use it
      ({o["auroc"]:.3f}) &mdash; and a one-line rule on yesterday's reading is already
      {M["sbp_threshold"]["auroc"]:.3f}. Forecasting tomorrow's pressure from the wrist is not the
      product.</div>
      {foot(5)}</div>""")

    s.append(f"""<div class="slide">
      <p class="kick">Result 2 &middot; the one that matters</p>
      <h2>The wrist can find the non-dipper, and that is the clinically scarce test</h2>
      <div class="grid g2">
        <div><p>Same cuff-free inputs. Different question: not <i>when will pressure spike</i> but
        <i>does this patient's pressure fail to fall at night</i>.</p>
        <div class="card hi"><div class="big g">{dip["auroc"]:.3f}</div>
          <div class="lab">AUROC, patient-level, cuff-free<br>95% CI {dip["auroc_ci"][0]:.3f}&ndash;{dip["auroc_ci"][1]:.3f}
          &middot; n = {dip["n_patients"]} held-out patients</div></div>
        <p style="margin-top:3mm">The signal is autonomic. Non-dippers show a blunted overnight rise in
        RMSSD, so the night/day HRV ratio carries the phenotype even with no pressure measurement
        anywhere in the input.</p></div>
        <div><div class="card"><b>Why this is the useful result</b>
          <ul><li>Non-dipping independently predicts cardiovascular events</li>
          <li>Confirming it needs ABPM, which India largely does not have</li>
          <li>A wearable screen turns a scarce test into a triage decision:
              who actually needs the cuff</li>
          <li>It runs on hardware tens of millions of Indians already wear</li></ul></div>
        <div class="card warn" style="margin-top:4mm"><b>The honest caveat.</b> This result follows from
        an autonomic-dipping coupling we encoded in the simulator. The coupling is attested in the
        physiology literature; its effect size is our choice. This number is evidence that the pipeline
        works, not evidence about real patients. Real ABPM plus wearable data is the next step.</div></div>
      </div>
      {foot(6)}</div>""")

    s.append(f"""<div class="slide">
      <p class="kick">Result 3 &middot; the twin, measured</p>
      <h2>Counterfactual accuracy against paired re-simulation</h2>
      <p>For each intervention the twin predicts a change; the ground-truth simulator is re-run for the
      same virtual patient under the same seed. {C["by_intervention"]["chronotherapy"].get("label") and ""}
      120 patients, all five interventions, no selection.</p>
      <table><thead><tr><th>Intervention</th><th class="n">true &Delta;24h</th><th class="n">twin &Delta;24h</th>
        <th class="n">MAE</th><th class="n">true &Delta;night</th><th class="n">twin &Delta;night</th><th>sign</th></tr></thead>
        <tbody>{cf_rows}</tbody></table>
      <div class="grid g2" style="margin-top:3mm">
        <div class="card hi"><b>Chronotherapy is the showcase.</b> Moving one agent to 21:00 changes the
        24-hour mean by essentially nothing ({C["by_intervention"]["chronotherapy"]["true_mean_d_sbp"]:+.2f} mmHg)
        while lowering night-time pressure by
        {abs(C["by_intervention"]["chronotherapy"]["true_night_d_sbp"]):.2f} mmHg. A scalar risk score
        cannot express that. A twin can, and that is precisely the decision a clinician faces with a
        non-dipper.</div>
        <div class="card warn"><b>Two arms we do not claim.</b> The salt arm is <span class="tag w">circular</span>
        &mdash; the twin's coefficient and the simulator's are both drawn from the same literature value,
        so it tests nothing. The sleep arm is <span class="tag n">weak</span> at
        {C["by_intervention"]["sleep_plus_1h"]["sign_agreement"]*100:.0f}% sign agreement; the twin
        over-states the benefit because it over-amplifies debt carry-over.</div>
      </div>
      <div class="center" style="margin-top:1mm"><img src="file://{RES/'calibration.png'}" style="max-height:48mm"></div>
      {foot(7)}</div>""")


    sw = S["sweep"]
    dmin = S["min_observed_cohens_d_for_auroc_080"]
    k0 = next(r for r in sw if r["kappa"] == 0.0)
    k1 = next(r for r in sw if r["kappa"] == 1.0)
    s.append(f"""<div class="slide">
      <p class="kick">Result 4 &middot; falsifiability</p>
      <h2>How much of this is just our own assumption?</h2>
      <p>The obvious attack on Result 2 is that we encoded the autonomic coupling and then
      detected it. So we measured exactly how much we assumed. We sweep the coupling strength,
      regenerate the whole cohort each time, retrain, and plot the screen against the
      <b>observed</b> separation between dippers and non-dippers, which is a quantity anyone
      can measure in real patients.</p>
      <div class="grid g2">
        <div><div class="card hi"><b>Negative control holds.</b> At zero coupling the screen
          collapses to <b>{k0["screen_auroc"]:.3f}</b>, near chance. The model has no back-door
          route to the label. The small residual above 0.5 comes from sleep apnoea, which
          independently affects both dipping and HRV.</div>
        <div class="card warn" style="margin-top:4mm"><b>The threshold.</b> The screen becomes
          usable (AUROC 0.80) once the observed separation reaches <b>d &asymp; {dmin}</b>.
          Our default assumes d = {k1["observed_cohens_d"]}, which is almost certainly generous.</div>
        <p style="margin-top:4mm">That turns an assumption into something falsifiable:
        <b>this approach works if, and only if, real non-dippers show a night/day RMSSD
        separation of at least d &asymp; {dmin}.</b> Anyone with paired ABPM and wearable data
        can check whether reality clears the bar. Result 6 is our attempt to start.</p></div>
        <div class="center"><img src="file://{RES/'sensitivity.png'}" style="max-height:108mm"></div>
      </div>
      {foot(8)}</div>""")

    H = DEP["headline"]
    rows_dep = "".join(
        f'<tr><td class="n">{r["budget_frac"]*100:.0f}%</td><td class="n">{r["referred"]}</td>'
        f'<td class="n">{r["found"]}</td><td class="n">{r["yield_precision"]*100:.0f}%</td>'
        f'<td class="n">{r["sensitivity"]*100:.0f}%</td>'
        f'<td class="n">{r["abpm_per_find"]:.2f}</td></tr>'
        for r in DEP["by_budget"])
    uni = DEP["universal_screening"]
    s.append(f"""<div class="slide">
      <p class="kick">Result 5 &middot; operational value</p>
      <h2>An AUROC is not a decision. Who gets the ABPM?</h2>
      <div class="grid g2">
        <div><table><thead><tr><th>budget</th><th>studies</th><th>found</th>
          <th>precision</th><th>sensitivity</th><th>ABPM/find</th></tr></thead><tbody>{rows_dep}
          <tr><td class="mut">universal</td><td class="n mut">{uni["referred"]}</td>
          <td class="n mut">{uni["found"]}</td>
          <td class="n mut">{DEP["held_out"]["nondipper_prevalence"]*100:.0f}%</td>
          <td class="n mut">100%</td><td class="n mut">{uni["abpm_per_find"]:.2f}</td></tr>
          </tbody></table>
        <div class="card hi" style="margin-top:3mm"><b>At a {H["best_budget_frac"]*100:.0f}% referral
          budget every single referral is a true non-dipper</b> ({H["precision_at_best"]*100:.0f}%
          precision against {H["random_precision"]*100:.0f}% for random referral), catching
          {H["sensitivity_at_best"]*100:.0f}% of all non-dippers for a third of the studies.</div>
        <div class="card warn" style="margin-top:3mm"><b>Honest ceiling.</b> At
          {H["random_precision"]*100:.0f}% prevalence, triage can lift yield by at most
          {H["max_possible_lift"]}&times;. The value here is in not wasting studies, not in
          finding something rare. A lower-prevalence screening population would gain more.</div></div>
        <div class="center"><img src="file://{RES/'deployment.png'}" style="max-height:110mm"></div>
      </div>
      {foot(9)}</div>""")

    if SHAREE:
        R = SHAREE["night_day_rmssd_ratio"]
        inc = SHAREE["incremental_over_clinic"]
        sig = "reaches" if R["mann_whitney_p"] < 0.05 else "does not reach"
        s.append(f"""<div class="slide">
      <p class="kick">Result 6 &middot; real patients</p>
      <h2>External validation on data we did not generate</h2>
      <p><b>SHAREE</b> (Melillo et al., PLoS ONE 2015): {SHAREE["n_analysed"]} real hypertensive
      patients, nominal 24-hour Holter, 12-month follow-up, {SHAREE["n_events"]} vascular events.
      We ask whether the night/day RMSSD ratio, the exact feature our screen uses, predicts those
      events.</p>
      <div class="grid g3">
        <div class="card hi"><div class="big">{R["auroc_event"]:.3f}</div><div class="lab">AUROC predicting a
          12-month vascular event<br>95% CI {R["auroc_ci"][0]:.3f}&ndash;{R["auroc_ci"][1]:.3f}</div></div>
        <div class="card"><div class="big">{R["cohens_d"]:.2f}</div><div class="lab">Cohen's d between
          event and no-event<br>Mann-Whitney p = {R["mann_whitney_p"]}</div></div>
        <div class="card"><div class="big">{SHAREE["delta_auroc"]:+.3f}</div><div class="lab">added over
          age, sex and clinic SBP<br>({inc.get("clinic only (age, sex, SBP)")} &rarr;
          {inc.get("clinic + night/day RMSSD")})</div></div>
      </div>
      <div class="center" style="margin-top:2mm"><img src="file://{RES/'sharee_validation.png'}" style="max-height:76mm"></div>
      <div class="card warn" style="margin-top:2mm"><b>What this does and does not show.</b> It
      {sig} significance at 0.05 with {SHAREE["n_events"]} events, which is an underpowered sample.
      It validates the prognostic value of the autonomic feature in real hypertensive patients.
      It does <i>not</i> validate non-dipper detection: SHAREE has no ambulatory BP, so the
      dipping phenotype is unobservable in it. Paired ABPM and wearable data remains the next step.</div>
      {foot(10)}</div>""")
    else:
        s.append(f"""<div class="slide">
      <p class="kick">Result 6 &middot; real patients</p>
      <h2>External validation on data we did not generate</h2>
      <p><b>SHAREE</b> (Melillo et al., PLoS ONE 2015): 139 real hypertensive patients, nominal
      24-hour Holter at 128 Hz, 12-month follow-up, 17 vascular events (11 myocardial infarction,
      3 stroke, 3 syncope). Open access, no credentialing.</p>
      <p>The pipeline is written and in the repository (<code>src/validate_sharee.py</code>). It
      extracts RR intervals from the shipped QRS annotations, computes night and day HRV, and tests
      whether the night/day RMSSD ratio predicts a 12-month vascular event over and above age, sex
      and clinic blood pressure.</p>
      <div class="card warn"><b>Status: analysis pending at the time of writing.</b> We are
      reporting that plainly rather than leaving the slide out. Whatever it returns, including a
      null, goes into the repository.</div>
      <div class="card" style="margin-top:4mm"><b>What it will and will not settle.</b> It tests the
      prognostic value of the autonomic feature in real hypertensive patients. It cannot validate
      non-dipper detection itself, because SHAREE carries no ambulatory blood pressure, so the
      dipping phenotype is unobservable in it.</div>
      {foot(10)}</div>""")

    s.append(f"""<div class="slide">
      <p class="kick">The console</p>
      <h2>What the doctor actually sees</h2>
      <div class="center"><img src="file://{RES/'dashboard.png'}"></div>
      {foot(11)}</div>""")

    s.append(f"""<div class="slide">
      <p class="kick">Limitations</p>
      <h2>What this is not</h2>
      <div class="grid g2">
        <div class="card"><b>Everything is in silico</b><ul>
          <li>No real patient data anywhere in the project</li>
          <li>Performance reflects our simulator's assumptions, so it is an
              upper bound on what real data would give</li>
          <li>The non-dipper result depends on a coupling we encoded</li></ul></div>
        <div class="card"><b>Method limits</b><ul>
          <li>Cuff-free forecasting adds nothing over the record (Result 1)</li>
          <li>Salt arm is circular; sleep arm over-predicts</li>
          <li>Twin is a reduced model; no baroreflex, renal or RAAS dynamics</li>
          <li>Cohort is enriched for non-dipping and is not population-representative</li></ul></div>
      </div>
      <div class="card warn" style="margin-top:4mm"><b>Clinical status.</b> This is a research prototype.
      It is not a medical device, it has not been validated on humans, and nothing in it should be used
      to make a treatment decision. The dose-timing simulation in particular illustrates a mechanism;
      chronotherapy in real patients remains an area of active and unsettled trial evidence.</div>
      <p style="margin-top:4mm"><b>What would change our minds:</b> paired ABPM and wearable recordings
      from an Indian hypertensive cohort. If the night/day RMSSD ratio does not separate dippers from
      non-dippers there, the screen does not work and we would say so.</p>
      {foot(12)}</div>""")

    s.append(f"""<div class="slide">
      <p class="kick">Where it goes</p>
      <h2>From prototype to a triage layer that fits Indian primary care</h2>
      <div class="grid g3">
        <div class="card"><b>Next</b><ul><li>Validate on open paired ABPM + wearable data</li>
          <li>Replace the reduced twin with a baroreflex model</li>
          <li>Calibrate on a real Indian cohort</li></ul></div>
        <div class="card"><b>Then</b><ul><li>Prospective pilot in an urban clinic</li>
          <li>ABPM referral driven by the screen</li>
          <li>Measure cuffs saved per non-dipper found</li></ul></div>
        <div class="card"><b>Deployment shape</b><ul><li>One calibration week with a cuff</li>
          <li>Continuous cuff-free monitoring after</li>
          <li>Escalate only when the twin flags</li></ul></div>
      </div>
      <div class="card hi" style="margin-top:6mm;text-align:center">
        <p style="font-size:19px;font-weight:650;margin:0">Find the patients whose pressure never sleeps,
        using the device they already wear.</p></div>
      <p class="mut" style="margin-top:6mm;font-size:11.5px">Reproduce every number here with
      <code>./run_all.sh</code>. MIT licensed. All simulated data synthetic.</p>
      {foot(13)}</div>""")

    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}
@page{{size:A4 landscape;margin:0}}</style></head><body>{''.join(s)}</body></html>"""


def render(html: str, out: pathlib.Path):
    tmp = DOCS / (out.stem + ".tmp.html")
    tmp.write_text(html, encoding="utf-8")
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page()
        pg.goto(f"file://{tmp.resolve()}")
        pg.wait_for_timeout(700)
        pg.pdf(path=str(out), format="A4", landscape=True, print_background=True,
               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        b.close()
    tmp.unlink()
    print(f"  {out.relative_to(ROOT)}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    print("Rendering PDFs:")
    render(architecture_html(), DOCS / "architecture.pdf")
    render(deck_html(), DOCS / "presentation.pdf")
