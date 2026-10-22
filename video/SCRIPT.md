# CardioTwin — 20-minute demo video script

The challenge requires a **minimum 20-minute** video explaining and demonstrating the prototype. This script now runs to roughly **26 minutes** at a normal speaking pace, so you have real margin and no longer need the filler section at the bottom.

Suggested split between two presenters: **0:00 to 12:30** and **12:30 to end**.

## Before you hit record

- [ ] Run `./run_all.sh` once so every output exists and the terminal scrollback is clean
- [ ] Open `dashboard/index.html` in a browser, full screen, zoom 110%
- [ ] Open `docs/presentation.pdf` in a second tab
- [ ] Open the repo in an editor with `src/` expanded
- [ ] Screen resolution 1920×1080, hide bookmarks bar, close notifications
- [ ] Record at 1080p. OBS Studio or Zoom local recording both work. Upload unlisted to YouTube.
- [ ] Speak slowly. Twenty minutes is longer than it feels. Silence while something runs is fine.

**Two rules.** Say "simulated" or "synthetic" whenever you show a patient. Never say the model is accurate without saying it is in-silico.

---

## 0:00 – 1:30 · Opening

**Show:** presentation.pdf slide 1

> "Hello, we're `TEAM NAME` from `COLLEGE`. This is CardioTwin, our entry to the Digital Twin Challenge 2026.
>
> One sentence: CardioTwin is a hypertension digital twin that finds the patients whose blood pressure never falls at night, using only a consumer wearable, and then lets a doctor simulate interventions on that patient before changing anything.
>
> Before I go further, the most important disclaimer. Every patient you will see in this video is simulated. There is no real patient data anywhere in this project. This is a research prototype and it is not a medical device.
>
> Three numbers frame the whole project. 315 million Indian adults have hypertension. Our cuff-free screen detects the highest-risk phenotype at an AUROC of 0.96 in held-out simulated patients. And we validate five clinical interventions by paired re-simulation, which I'll explain later because it's the part we're most proud of."

---

## 1:30 – 4:30 · The problem

**Show:** presentation.pdf slide 2

> "Blood pressure is not a single number. It follows a circadian rhythm. In a healthy person it falls between 10 and 20 percent overnight. That fall is called dipping.
>
> Some people don't dip. Their pressure stays flat overnight — we call them non-dippers. And some people's pressure actually rises at night — reverse dippers. Both groups have substantially higher cardiovascular risk than their daytime reading suggests. Non-dipping independently predicts stroke and cardiac events.
>
> Here's the problem. You cannot see dipping in a clinic. A clinic reading is taken at eleven in the morning while the patient is awake and mildly stressed. To measure dipping you need 24-hour ambulatory blood pressure monitoring — a cuff the patient wears all day and all night that inflates every 20 or 30 minutes.
>
> In India, ABPM is scarce. It's expensive, it's concentrated in urban private hospitals, and the overwhelming majority of those 315 million people will never have one. So we are managing the largest hypertensive population in the world using a measurement that is blind to their highest-risk feature.
>
> Now the other half. Tens of millions of Indians already wear a device that measures heart rate, heart rate variability, sleep stages and step count, continuously, every day. It cannot measure blood pressure. But the question we asked is: is what it *can* measure enough to tell us who needs the cuff?"

---

## 4:30 – 6:30 · What makes this a twin, not a predictor

**Show:** presentation.pdf slide 3

> "I want to be precise about the word twin, because a lot of things get called digital twins when they're really just predictive models.
>
> A predictive model answers 'what will happen'. A digital twin has to answer 'what happens if I change something'. That second question is the one a doctor actually has.
>
> So CardioTwin is built as a reduced mechanistic model of the patient's circadian pressure. This equation on screen — systolic pressure as a function of a setpoint, a circadian factor, a morning surge term, a behaviour term and a medication term.
>
> Three of those parameters are fitted to each individual patient from one calibration week of their own data: their personal setpoint, their personal nocturnal factor, and their personal morning surge amplitude.
>
> Because the model is mechanistic rather than a black box, you can intervene on it. Move a dose to bedtime and re-integrate. Add an hour of sleep and re-integrate. That's the twin.
>
> And here's the part that matters. Because we built the ground-truth simulator ourselves, we can actually check whether the twin's what-if answers are right. I'll show you that at the end."

---

## 6:30 – 10:00 · Data generation, live

**Show:** `src/generate_ehr.py` in the editor, then a terminal

> "The challenge requires fusing two data streams: static EHR and dynamic wearable data, both synthetic or open-source. Let me show you how we generate both.
>
> This is the EHR generator. For each of 300 virtual patients we produce demographics with South Asian body composition, an ICD-10 coded problem list — essential hypertension, type 2 diabetes, CKD stage 3, obstructive sleep apnoea — a full lab panel, an antihypertensive regimen with dose timing, and a polygenic risk score proxy.
>
> Note this block here. These are latent traits: dipping phenotype, medication adherence, salt load, autonomic tone. They drive the simulation and they are written to a separate file that is never used as a model input. That matters for honesty — if we fed the phenotype into the model we'd just be reading back our own answer."

**Run:** `python3 src/generate_ehr.py` — let it print

> "300 patients. Roughly 40% non-dippers, which matches treated hypertensive cohorts."

**Show:** `src/generate_wearables.py`

> "This is the physiological core. For each patient we integrate forward 21 days at five-minute resolution and emit heart rate, RMSSD, sleep stage, step count and blood pressure.
>
> Four pieces of real cardiovascular physiology are encoded here. First, the circadian rhythm with the dipping phenotype — you can see the night factors: dippers multiply by 0.845, non-dippers 0.945, reverse dippers 1.035. Second, the morning surge after waking, which is larger in non-dippers. Third — and this is the one that makes the whole project work — autonomic coupling. RMSSD rises overnight with vagal tone, and in non-dippers that rise is blunted. Fourth, steady-state drug pharmacokinetics, so a missed dose or a badly timed dose is actually visible in the trace."

**Run:** `python3 src/generate_wearables.py`

> "1.8 million samples in about four seconds."

**Run:** `python3 src/features.py` and point at the phenotype table

> "Now here is a face-validity check I want to draw attention to. We never told the simulator the clinical cut-off for non-dipping. But look at what comes out: dippers fall 15.7% overnight, non-dippers 5.9%, reverse dippers rise 2.6%. The clinical definition is a 10% fall. The simulator reproduces the definitions it was never given. That's our evidence the physiology is doing the work rather than the random number generator."

---

## 10:00 – 12:30 · Fusion and models

**Show:** `src/features.py`

> "Feature fusion. One row per patient-day. Each row joins the static EHR record to summaries of the preceding seven days of wearable signal.
>
> The dynamic features are deliberately clinical constructs rather than raw statistics. Dip ratio. Morning surge. Night-to-day RMSSD ratio. Average real variability, which is a recognised blood-pressure-variability risk metric. Sleep debt. And for each of those we take the seven-day mean, the most recent value, and the linear trend.
>
> Now the most important design decision in the project. This block here splits the wearable features into cuff-free and cuff-requiring. Anything derived from a blood pressure measurement — the dip ratio, the surge, the variability — is **held out of our headline model**. Because if the patient had a cuff, they wouldn't need us. The model we're actually proposing sees only heart rate, HRV, sleep and steps, plus the record.
>
> The classifier is XGBoost with isotonic calibration, and the split is patient-grouped, so no patient appears in both training and test. The calibration matters because the dashboard shows a probability to a clinician, and that number should mean something."

---

## 12:30 – 16:00 · Results, including the one that failed

**Show:** presentation.pdf slide 5

> "I'm going to give you the negative result first, because I think reporting it is more useful than hiding it.
>
> Task: predict a sustained Stage-2 excursion, above 160 systolic for at least half an hour, in the next 24 hours.
>
> EHR alone gets 0.753. Cuff-free wearable alone gets 0.544, barely better than chance. Fusing them gets 0.754.
>
> That is not an improvement. 0.754 against 0.753, with confidence intervals that overlap almost completely. Our hypothesis for this task was wrong. And look at the bottom row: a one-line rule that just reads yesterday's average blood pressure scores 0.888. If you have a cuff, the cuff wins, and no machine learning is needed.
>
> So: forecasting tomorrow's blood pressure from the wrist is not the product. We say that in the README and on the slide."

**Show:** presentation.pdf slide 6

> "Here's the result that does work. Same cuff-free inputs. Different question. Not *when will pressure spike* but *does this patient's pressure fail to fall at night*.
>
> AUROC 0.963, confidence interval 0.90 to 1.0, on 90 held-out simulated patients, with no blood pressure measurement anywhere in the input.
>
> The signal is autonomic. Non-dippers have a blunted overnight rise in heart rate variability, so the night-to-day RMSSD ratio carries the phenotype.
>
> Why this is the useful result: non-dipping predicts cardiovascular events, confirming it needs ABPM that India largely doesn't have, and a wearable screen turns that scarce test into a triage decision — which patients should be sent for the cuff.
>
> And the honest caveat, which is also on the slide. This result follows from an autonomic–dipping coupling that we encoded into the simulator. That coupling is real physiology and it's in the literature, but we chose its effect size. So this is evidence that the pipeline works. It is not yet evidence about real patients."

**Show:** presentation.pdf slide 7

> "Third result, and this is the twin being measured rather than asserted.
>
> For each intervention, the twin predicts a change in blood pressure. Then we re-run the ground-truth simulator for the same virtual patient, with the same random seed, with the intervention applied. Same noise, paired comparison, all 120 patients, no cherry-picking.
>
> Activity: twin predicts −0.48, truth is −0.64, mean absolute error 0.16 millimetres of mercury, sign correct 100% of the time.
>
> Now look at chronotherapy — moving one drug to 21:00 dosing. Change in 24-hour mean: plus 0.11, essentially nothing. Change in night-time pressure: minus 1.83. That is the entire clinical point of chronotherapy, and a single risk score literally cannot express it. The twin can.
>
> Two arms we explicitly do not claim. Salt is marked circular: the twin's coefficient and the simulator's coefficient come from the same literature value, so that arm tests nothing and we label it as such. And sleep is weak — 76% sign agreement, the twin over-states the benefit."

---

## 16:00 – 18:30 · Result 4, answering the obvious attack

**Show:** presentation.pdf slide 8

> "Now I want to deal with the obvious objection to what I just showed you, because a good reviewer will raise it immediately.
>
> The objection is: you encoded a coupling between the dipping phenotype and heart rate variability into your simulator, and then you showed a model can detect it. That's circular. You proved nothing.
>
> That's a fair attack, so rather than answer it in words, we measured it.
>
> We take the coupling strength, call it kappa, and sweep it. At kappa zero there is no autonomic difference between dippers and non-dippers at all. At kappa one you get our default. We regenerate the entire cohort at each setting, retrain the screen from scratch, and plot two things against each other: the observed separation between dippers and non-dippers in the data, measured as Cohen's d, and the screen's AUROC.
>
> Two results. First, the negative control. At zero coupling the screen collapses to 0.574, which is essentially chance. That matters, because it proves the model has no back-door route to the label. It is not picking up age or BMI or medication count and calling it non-dipping. Take the autonomic signal away and it has nothing. The small residual above 0.5 is sleep apnoea, which in our simulator independently affects both dipping and HRV.
>
> Second, and this is the useful number. The screen becomes usable, meaning AUROC above 0.8, once the observed separation reaches a Cohen's d of about 0.69. Our default assumes 2.27, which is almost certainly generous.
>
> So what this curve does is convert an assumption into a falsifiable threshold. We are no longer saying trust us. We are saying: this approach works if and only if real non-dippers show a night-day RMSSD separation of at least 0.69 standard deviations. Anyone with paired ambulatory BP and wearable data can go and check whether reality clears that bar. That is a much more honest kind of claim."

---

## 18:30 – 20:30 · Result 5, what it is worth in a clinic

**Show:** presentation.pdf slide 9

> "An AUROC is not a decision. The decision a health system actually faces is: we have a limited number of ambulatory BP monitors, who gets one?
>
> So we turned the screen into that decision. Rank every patient by screen score, refer the top k percent, and measure what you catch.
>
> Look at the 30 percent row. Twenty-seven studies, twenty-seven non-dippers found. One hundred percent precision. Every single ABPM you spend lands on a patient who genuinely has the phenotype. If you referred at random at that prevalence you would be right 57 percent of the time.
>
> So you can cut your ABPM spend to a third and waste nothing, at the cost of missing the harder half of cases.
>
> And now the honest ceiling, which is also on the slide. At 57 percent prevalence, no triage strategy can lift your yield by more than 1.76 times, because more than half the population already has the thing you are looking for. The value here is in not wasting studies, not in finding something rare. In a lower-prevalence screening population, say a general practice rather than a hypertension clinic, the gain would be considerably larger. We state that rather than quoting a flattering multiple."

---

## 20:30 – 22:30 · Result 6, real patients

**Show:** presentation.pdf slide 10

> "Everything so far has been simulated. So the last thing we did was test the core premise against data we did not generate.
>
> SHAREE is an open database from Melillo and colleagues, published in PLoS ONE in 2015. A hundred and thirty-nine real hypertensive patients, each with a nominal 24-hour Holter recording, followed for twelve months. Seventeen of them had a vascular event: eleven myocardial infarctions, three strokes, three syncopal episodes.
>
> We extract RR intervals from the shipped QRS annotations, compute heart rate variability separately for the night window and the day window, and form exactly the same night-day RMSSD ratio our screen depends on. Then we ask: does that ratio predict who had an event, and does it add anything on top of age, sex and clinic blood pressure?"

*(If you have run it: read the three numbers off the slide, then the caveat box. If it came back null, say so plainly: "with seventeen events this analysis is underpowered, it did not reach significance, and we are reporting that rather than dropping the analysis.")*

> "And the boundary of what this settles. It tests whether the autonomic feature has real prognostic value in real hypertensive patients. It cannot validate non-dipper detection itself, because SHAREE carries no ambulatory blood pressure, so the dipping phenotype is simply unobservable in it. Paired ABPM and wearable data is still the study that would settle it, and that is what we would do next."

---

## 22:30 – 25:00 · Dashboard demo

**Show:** `dashboard/index.html`, live

> "This is the clinician console. One self-contained HTML file, no server, no build step.
>
> On the left, the patient panel ranked by next-24-hour risk. On the right, the selected patient.
>
> Here's the seven-day blood pressure trace. The shaded bands are sleep. Watch what happens across each band — you can see the pressure fall overnight and come back up. That visual *is* the dipping phenotype, and this patient is labelled up here with their nocturnal fall percentage.
>
> Let me click a different patient." *(click a non-dipper)* "Look at the difference. The pressure barely falls across the shaded bands. This is a non-dipper, and in current practice you would need 24-hour ambulatory monitoring to know that.
>
> Down here is the autonomic panel. Night-to-day RMSSD ratio. When this is near or below 1.0, the normal overnight vagal rise is blunted — that's the signature the cuff-free model keys on.
>
> And here is the what-if panel. These are the five interventions, each one run through this patient's personalised twin. For each we show the predicted change in 24-hour mean and the predicted change in night-time pressure.
>
> Notice the validation column. Each intervention is tagged with how well it survived the paired re-simulation I just showed you. Activity, MAE 0.16. Salt, tagged circular, because we don't claim it. Sleep, tagged weak. We put the honesty in the interface, not just the paper."

---

## 25:00 – 26:30 · Limitations and close

**Show:** presentation.pdf slide 9, then slide 10

> "Limitations, plainly.
>
> Everything here is in silico. The performance numbers reflect our simulator's assumptions, which makes them an upper bound on what real data would give. Our headline result depends on a coupling we encoded. Cuff-free forecasting adds nothing over the record, as we showed. The salt counterfactual is circular. The twin is a reduced model with no baroreflex or renal dynamics. And the cohort is enriched for non-dipping, so it is not population-representative.
>
> What would change our minds: paired ambulatory BP and wearable recordings from an Indian hypertensive cohort. If the night-to-day RMSSD ratio does not separate dippers from non-dippers in real data, the screen does not work, and we would report that.
>
> Where it goes next. Validate on real paired data, replace the reduced twin with a baroreflex model, then a prospective pilot in an urban clinic measuring one thing: how many cuffs we save per non-dipper found.
>
> Everything in this video reproduces with one command, `./run_all.sh`. The repository is MIT licensed and all the data is synthetic.
>
> To close on the idea: find the patients whose pressure never sleeps, using the device they already wear. Thank you."

---

## If you are running short

The script should land near 26 minutes, but if you speak quickly, expand these in order.

1. Walk through `twin.py` line by line, especially `predict_change` and `ground_truth_change`, and explain why sharing the random seed is what makes the comparison paired.
2. Run `python3 -m pytest tests/ -q` on camera. Fifteen tests, including assertions that the simulated phenotypes match their clinical definitions.
3. Open `results/metrics.json` and `results/sensitivity.json` and read the numbers straight off the files, showing they match the slides.
4. Click through four or five more patients in the dashboard and contrast a dipper, a non-dipper and a reverse dipper side by side.

## If you are running long

Cut in this order. Do not cut Result 1 or the limitations slide, they are the credibility backbone.

1. Shorten the data-generation walkthrough from 3:30 to 2:00 by not running the scripts live, just showing the output already in the terminal.
2. Trim Result 5 to the single 30-percent row and the honest-ceiling line.
