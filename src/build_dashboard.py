"""
Build the clinician-facing CardioTwin dashboard.

Emits a single self-contained HTML file (no server, no build step, no CDN)
so it opens from the filesystem, works on GitHub Pages, and records cleanly
in a screen capture.

What a doctor sees:
  * a panel of virtual patients ranked by next-24h risk
  * the selected patient's 7-day blood-pressure trace with sleep shaded,
    so the dipping phenotype is visible at a glance
  * the autonomic view: nocturnal HRV rise against the dipping ratio
  * a what-if panel that re-runs the twin live for five interventions and
    reports the predicted change in 24-h mean and night-time SBP
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from config import EXCURSION_SBP
from features import feature_columns, split_dynamic
from twin import CardioTwin, INTERVENTIONS, estimate_behaviour_coefficients

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_SHOW = 24          # patients carried into the dashboard payload
TRACE_DAYS = 7


def build_payload() -> dict:
    ehr = pd.read_csv(ROOT / "data" / "ehr.csv")
    daily = pd.read_csv(ROOT / "data" / "daily_summary.csv")
    ts = pd.read_parquet(ROOT / "data" / "wearables.parquet")
    table = pd.read_csv(ROOT / "data" / "model_table.csv")
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    cf = json.loads((ROOT / "results" / "counterfactual_summary.json").read_text())
    coeffs = estimate_behaviour_coefficients(daily)

    # risk score per patient = mean predicted risk on their last few days,
    # recomputed here from the stored model table using the fitted threshold
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupShuffleSplit
    from train_eval import _model
    static, dynamic = feature_columns(table)
    cuffless, _ = split_dynamic(dynamic)
    cols = static + cuffless
    y = table["y"].to_numpy()
    groups = table["patient_id"].to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.30,
                                    random_state=20261020).split(table, y, groups))
    clf = CalibratedClassifierCV(_model(), method="isotonic", cv=3)
    clf.fit(table[cols].to_numpy(float)[tr], y[tr])
    table = table.copy()
    table["risk"] = clf.predict_proba(table[cols].to_numpy(float))[:, 1]
    table["is_test"] = False
    table.loc[table.index[te], "is_test"] = True

    latest = (table.sort_values("pred_day").groupby("patient_id").tail(1)
                   .set_index("patient_id"))

    # prefer held-out patients, highest risk first
    order = latest[latest.is_test].sort_values("risk", ascending=False)
    pick = list(order.index[:N_SHOW])

    patients = []
    for pid in pick:
        e = ehr[ehr.patient_id == pid].iloc[0]
        g = ts[ts.patient_id == pid]
        cal = g[g.day < 7]
        tw = CardioTwin(e, coeffs).fit(cal)

        last = g[g.day >= g.day.max() - (TRACE_DAYS - 1)].copy()
        step = max(len(last) // 1100, 1)                   # keep payload small
        last = last.iloc[::step]

        dsum = daily[daily.patient_id == pid].tail(TRACE_DAYS)
        dip = float(dsum["dip_ratio"].mean() * 100)
        pheno = ("Reverse dipper" if dip < 0 else
                 "Non-dipper" if dip < 10 else "Dipper")

        whatif = []
        for name, spec in INTERVENTIONS.items():
            pr = tw.predict_change(name)
            whatif.append({
                "id": name, "label": spec["label"],
                "d_mean": round(pr["d_mean_sbp"], 2),
                "d_night": round(pr["d_night_sbp"], 2),
                "validated": cf["by_intervention"].get(name, {}).get("sign_agreement", 0) >= 0.90,
                "mae": cf["by_intervention"].get(name, {}).get("mae_mmhg"),
                "circular": name == "salt_minus_3g",
            })

        dx = [d for d in [("T2DM", e.dx_t2dm), ("CKD-3", e.dx_ckd3), ("OSA", e.dx_osa),
                          ("Dyslipidaemia", e.dx_dyslipidaemia), ("Prior MI", e.dx_prior_mi),
                          ("Prior stroke", e.dx_prior_stroke), ("Obesity", e.dx_obesity)]
              if isinstance(d[1], str) and d[1].strip()]

        patients.append({
            "id": pid,
            "age": int(e.age), "sex": e.sex, "bmi": round(float(e.bmi), 1),
            "region": e.region, "urban": bool(e.urban),
            "dx": [d[0] for d in dx],
            "egfr": round(float(e.lab_egfr), 0), "hba1c": round(float(e.lab_hba1c_pct), 1),
            "uacr": round(float(e.lab_uacr_mg_g), 0), "prs": round(float(e.genetic_prs_z), 2),
            "rx": e.rx_agents if isinstance(e.rx_agents, str) else "",
            "rx_hours": e.rx_dose_hours if isinstance(e.rx_dose_hours, str) else "",
            "risk": round(float(latest.loc[pid, "risk"]), 3),
            "dip_pct": round(dip, 1), "phenotype": pheno,
            "sbp_mean": round(float(dsum.sbp_mean.mean()), 1),
            "sbp_night": round(float(dsum.sbp_night.mean()), 1),
            "surge": round(float(dsum.morning_surge.mean()), 1),
            "hrv_night_ratio": round(float(dsum.hrv_night_ratio.mean()), 2),
            "sleep_h": round(float(dsum.sleep_hours.mean()), 1),
            "steps": int(dsum.steps_total.mean()),
            "trace": {
                "t": [round(float(v), 2) for v in last.t_hours],
                "sbp": [round(float(v), 1) for v in last.sbp_mmhg],
                "hr": [round(float(v), 1) for v in last.hr_bpm],
                "rmssd": [round(float(v), 1) for v in last.rmssd_ms],
                "asleep": [int(v) for v in last.asleep],
            },
            "whatif": whatif,
        })

    return {
        "generated": "synthetic cohort, no real patient data",
        "threshold": EXCURSION_SBP,
        "operating_point": metrics.get("operating_point", {}),
        "metrics": metrics,
        "counterfactual": cf,
        "patients": patients,
    }


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CardioTwin Clinician Console</title>
<style>
:root{--ink:#16202b;--mut:#5d6b7a;--line:#dde3ea;--bg:#f4f7fa;--card:#fff;
--blue:#1b6ca8;--blue2:#e8f1f8;--red:#b3261e;--redbg:#fdeceb;--amb:#9a6700;--ambbg:#fdf6e3;
--grn:#0b6b4f;--grnbg:#eaf5f1;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
background:var(--bg);color:var(--ink)}
header{background:var(--card);border-bottom:1px solid var(--line);padding:14px 22px;
display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:650;letter-spacing:-.2px}
.sub{color:var(--mut);font-size:12.5px}
.warn{margin-left:auto;background:var(--ambbg);color:var(--amb);border:1px solid #f0e2bd;
border-radius:5px;padding:4px 10px;font-size:11.5px;font-weight:600}
main{display:grid;grid-template-columns:300px 1fr;gap:16px;padding:16px 22px;align-items:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:14px 16px}
.lbl{font-size:10.5px;font-weight:700;letter-spacing:.09em;color:var(--blue);
text-transform:uppercase;margin:0 0 9px}
ul.pts{list-style:none;margin:0;padding:0;max-height:76vh;overflow:auto}
ul.pts li{padding:9px 10px;border-radius:7px;cursor:pointer;border:1px solid transparent;
display:flex;align-items:center;gap:9px}
ul.pts li:hover{background:#f0f5fa}
ul.pts li.on{background:var(--blue2);border-color:#bcd7ea}
.pid{font-weight:650;font-size:13px}
.meta{color:var(--mut);font-size:11.5px}
.risk{margin-left:auto;font-weight:700;font-size:12.5px;padding:2px 8px;border-radius:11px}
.r-hi{background:var(--redbg);color:var(--red)} .r-md{background:var(--ambbg);color:var(--amb)}
.r-lo{background:var(--grnbg);color:var(--grn)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.chips span{display:inline-block;background:#eef2f6;border:1px solid var(--line);
border-radius:11px;padding:1px 9px;font-size:11.5px;margin:2px 4px 2px 0;color:var(--mut)}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(104px,1fr));gap:10px;margin-top:4px}
.kv div{background:#f7f9fb;border:1px solid var(--line);border-radius:7px;padding:7px 9px}
.kv b{display:block;font-size:16px;font-weight:650} .kv i{font-style:normal;color:var(--mut);font-size:10.5px}
.pheno{display:inline-block;padding:2px 9px;border-radius:11px;font-size:11.5px;font-weight:650}
.p-rev{background:var(--redbg);color:var(--red)} .p-non{background:var(--ambbg);color:var(--amb)}
.p-dip{background:var(--grnbg);color:var(--grn)}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:4px}
th,td{text-align:left;padding:6px 7px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:10.5px;letter-spacing:.05em;text-transform:uppercase}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.good{color:var(--grn);font-weight:650} .bad{color:var(--red);font-weight:650}
.tag{font-size:10px;padding:1px 6px;border-radius:9px;background:var(--grnbg);color:var(--grn);font-weight:650}
.tag.no{background:#eef2f6;color:var(--mut)} .tag.circ{background:var(--ambbg);color:var(--amb)}
footer{padding:4px 22px 26px;color:var(--mut);font-size:11.5px;max-width:1100px}
.legend{font-size:11px;color:var(--mut);margin-top:6px}
</style></head><body>
<header>
  <h1>CardioTwin &mdash; Clinician Console</h1>
  <span class="sub">Hypertension digital twin &middot; next-24h risk and intervention simulation</span>
  <span class="warn">SYNTHETIC DATA &middot; RESEARCH PROTOTYPE &middot; NOT A MEDICAL DEVICE</span>
</header>
<main>
  <div class="card"><p class="lbl">Patient panel &middot; ranked by 24h risk</p><ul class="pts" id="list"></ul></div>
  <div>
    <div class="card" id="hdr"></div>
    <div class="card" style="margin-top:14px"><p class="lbl">7-day blood pressure &middot; shaded bands are sleep</p>
      <div id="chart"></div>
      <p class="legend">Dashed line is the Stage-2 excursion threshold (160 mmHg). The nocturnal
      fall across the shaded bands is what defines the dipping phenotype.</p></div>
    <div class="grid2" style="margin-top:14px">
      <div class="card"><p class="lbl">Autonomic &amp; sleep</p><div id="auto"></div></div>
      <div class="card"><p class="lbl">What if &mdash; twin simulation</p><div id="wi"></div></div>
    </div>
  </div>
</main>
<footer id="foot"></footer>
<script>
const D = __PAYLOAD__;
const rc = r => r>=.6?'r-hi':r>=.35?'r-md':'r-lo';
const pc = p => p.startsWith('Reverse')?'p-rev':p.startsWith('Non')?'p-non':'p-dip';
let cur = 0;

function list(){
  document.getElementById('list').innerHTML = D.patients.map((p,i)=>
    `<li class="${i===cur?'on':''}" onclick="sel(${i})">
       <span><span class="pid">${p.id}</span><br><span class="meta">${p.age}${p.sex} &middot; ${p.region}</span></span>
       <span class="risk ${rc(p.risk)}">${(p.risk*100).toFixed(0)}%</span></li>`).join('');
}

function chart(p){
  const W=860,H=230,L=46,R=12,T=12,B=26, t=p.trace.t, s=p.trace.sbp;
  const x0=Math.min(...t), x1=Math.max(...t);
  const y0=Math.min(90,Math.min(...s)-6), y1=Math.max(190,Math.max(...s)+6);
  const X=v=>L+(v-x0)/(x1-x0)*(W-L-R), Y=v=>T+(1-(v-y0)/(y1-y0))*(H-T-B);
  let bands='',st=null;
  for(let i=0;i<t.length;i++){
    if(p.trace.asleep[i]&&st===null) st=t[i];
    if((!p.trace.asleep[i]||i===t.length-1)&&st!==null){
      bands+=`<rect x="${X(st)}" y="${T}" width="${Math.max(X(t[i])-X(st),1)}" height="${H-T-B}" fill="#eef3f8"/>`;st=null;}
  }
  const path=t.map((v,i)=>`${i?'L':'M'}${X(v).toFixed(1)},${Y(s[i]).toFixed(1)}`).join('');
  let gl='';for(const g of [100,130,160,190]) gl+=
    `<line x1="${L}" x2="${W-R}" y1="${Y(g)}" y2="${Y(g)}" stroke="#e4e9ee"/>
     <text x="${L-7}" y="${Y(g)+4}" font-size="10" fill="#5d6b7a" text-anchor="end">${g}</text>`;
  let dl='';for(let d=Math.ceil(x0/24)*24;d<x1;d+=24) dl+=
    `<line x1="${X(d)}" x2="${X(d)}" y1="${T}" y2="${H-B}" stroke="#eceff3"/>
     <text x="${X(d)}" y="${H-8}" font-size="10" fill="#5d6b7a" text-anchor="middle">day ${Math.round(d/24)}</text>`;
  document.getElementById('chart').innerHTML=
    `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Seven day systolic blood pressure trace">
      ${bands}${gl}${dl}
      <line x1="${L}" x2="${W-R}" y1="${Y(160)}" y2="${Y(160)}" stroke="#b3261e" stroke-dasharray="5 4"/>
      <path d="${path}" fill="none" stroke="#1b6ca8" stroke-width="1.5"/></svg>`;
}

function sel(i){
  cur=i; const p=D.patients[i]; list();
  document.getElementById('hdr').innerHTML=`
    <div style="display:flex;align-items:baseline;gap:11px;flex-wrap:wrap">
      <span style="font-size:18px;font-weight:650">${p.id}</span>
      <span class="meta">${p.age} y &middot; ${p.sex} &middot; BMI ${p.bmi} &middot; ${p.region} ${p.urban?'urban':'rural'}</span>
      <span class="pheno ${pc(p.phenotype)}">${p.phenotype} &middot; ${p.dip_pct}% nocturnal fall</span>
      <span class="risk ${rc(p.risk)}" style="margin-left:auto">24h risk ${(p.risk*100).toFixed(0)}%</span></div>
    <div class="chips" style="margin-top:9px">${p.dx.map(d=>`<span>${d}</span>`).join('')||'<span>No comorbidity coded</span>'}
      ${p.rx?p.rx.split('|').map((d,j)=>`<span>${d} @ ${(p.rx_hours||'').split('|')[j]||'?'}:00</span>`).join(''):'<span>No agent</span>'}</div>
    <div class="kv">
      <div><b>${p.sbp_mean}</b><i>24h mean SBP</i></div>
      <div><b>${p.sbp_night}</b><i>night SBP</i></div>
      <div><b>${p.surge}</b><i>morning surge</i></div>
      <div><b>${p.egfr}</b><i>eGFR</i></div>
      <div><b>${p.hba1c}</b><i>HbA1c %</i></div>
      <div><b>${p.prs}</b><i>PRS z</i></div></div>`;
  chart(p);
  document.getElementById('auto').innerHTML=`<div class="kv">
      <div><b>${p.hrv_night_ratio}</b><i>night/day RMSSD</i></div>
      <div><b>${p.sleep_h}</b><i>sleep h</i></div>
      <div><b>${(p.steps/1000).toFixed(1)}k</b><i>steps/day</i></div></div>
    <p class="legend">A night/day RMSSD ratio near or below 1.0 means the normal overnight
    vagal rise is blunted. That is the autonomic signature the cuff-free model uses to flag
    non-dipping, which would otherwise need 24-hour ambulatory monitoring.</p>`;
  document.getElementById('wi').innerHTML=`<table><thead><tr>
      <th>Intervention</th><th class="n">&Delta; 24h</th><th class="n">&Delta; night</th><th>validation</th></tr></thead><tbody>
    ${p.whatif.map(w=>`<tr><td>${w.label}</td>
      <td class="n ${w.d_mean<0?'good':'bad'}">${w.d_mean>0?'+':''}${w.d_mean.toFixed(2)}</td>
      <td class="n ${w.d_night<0?'good':'bad'}">${w.d_night>0?'+':''}${w.d_night.toFixed(2)}</td>
      <td>${w.circular?'<span class="tag circ">circular</span>'
         :w.validated?`<span class="tag">MAE ${w.mae}</span>`:'<span class="tag no">weak</span>'}</td></tr>`).join('')}
    </tbody></table>
    <p class="legend">mmHg change predicted by this patient's twin. Validation is paired
    re-simulation of the same virtual patient under the same random seed.</p>`;
}

const m=D.metrics;
document.getElementById('foot').innerHTML=
  `Cuff-free next-24h model AUROC <b>${m.fused.auroc}</b> [${m.fused.auroc_ci.join('&ndash;')}] vs EHR alone
   ${m.ehr.auroc}; with a BP cuff available ${m.oracle_bp.auroc}. Cuff-free non-dipper screen AUROC
   <b>${m.dipping_screen.auroc}</b> [${m.dipping_screen.auroc_ci.join('&ndash;')}].
   Held-out ${m.n_test_patients} patients, ${m.n_test_rows} patient-days, prevalence ${(m.prevalence*100).toFixed(0)}%.
   All patients shown are simulated. No real patient data is used anywhere in this project.`;
// Default to the most informative case on first load: a non-dipper on a
// morning regimen, where moving a dose to bedtime actually changes the
// night-time pressure. Scored rather than hard-coded so it survives a reseed.
const score = p => (p.rx ? 1 : 0) * 100
  + (p.phenotype.startsWith('Dipper') ? 0 : 40)
  + Math.abs((p.whatif.find(w => w.id === 'chronotherapy') || {d_night: 0}).d_night) * 6
  + p.risk * 10;
let d0 = 0, best = -1;
D.patients.forEach((p, i) => { const s = score(p); if (s > best) { best = s; d0 = i; } });
cur = d0; list(); sel(d0);
</script></body></html>
"""


if __name__ == "__main__":
    payload = build_payload()
    out = ROOT / "dashboard"
    out.mkdir(exist_ok=True)
    html = HTML.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    (out / "index.html").write_text(html, encoding="utf-8")
    kb = len(html.encode()) / 1024
    print(f"Dashboard -> dashboard/index.html  ({kb:.0f} KB, {len(payload['patients'])} patients, self-contained)")
