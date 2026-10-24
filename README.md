# GlucoTwin

**A clinician-facing digital twin for glucose management in Type 2 Diabetes.**
Happiest Health Digital Twin Challenge 2026.

## Submission at a glance

| | |
|---|---|
| ▶ **Demo video** (22 min) | [Watch on Google Drive](https://drive.google.com/file/d/1YjyfhFT6rcIV6HiqQKcVzY41-ewbUZoW/view?usp=sharing) |
| ▶ **Presentation** (PDF) | [Open on Google Drive](https://drive.google.com/file/d/1aCuLv0ozmCUV_eY0Iz95sqYekWDIOeYF/view?usp=sharing) |
| ▶ **Live clinician console** | [cleopatrakilledceaser.github.io/SynapseSquad_IITGuwahati](https://cleopatrakilledceaser.github.io/SynapseSquad_IITGuwahati/) — opens in a browser, no install, no sign-in. 120 simulated patients; the **Live twin** tab runs the physiological model in your browser |
| **Architecture diagram** | [docs/architecture.pdf](docs/architecture.pdf) |
| **Team** | Synapse Squad · Jahnavi Harsha (solo) · IIT Guwahati — details in [§1](#1-team-details) |
| **Licence** | MIT ([LICENSE](LICENSE)) |

## What it does

- **Anticipates.** Forecasts glucose 30, 60 and 120 minutes ahead with an 80 % range, and the
  calibrated chance of a high (above 180 mg/dL) or a low (below 70 mg/dL).
- **Follows the patient.** A glucose–insulin model of the patient updates its state with every CGM
  reading. It runs live in the console.
- **Explores changes.** Plays a meal or a walk forward before it happens, with Indian meal presets.
  Changes are model estimates, labelled as such.
- **Shows its limits.** Every result is compared with simple baselines and shown with its
  uncertainty, including where the models are weakest.

### A patient, two hours ahead

**S-151** is a 53-year-old woman with type 2 diabetes for six years, hypertension, a BMI of 24 and
an HbA1c of 7.6 % — numbers that look only moderately off at a clinic visit. At 20:25 her glucose
is 152 mg/dL and she is about to eat dinner. The twin, which has followed every CGM reading,
answers the question her clinician would ask (console → search *S-151* → Live twin → click −5 min
four times → try the meals):

| Plan | Twin's predicted peak |
|---|---|
| Vegetarian thali (95 g carbohydrate) | 257 mg/dL |
| Thali + 30-minute walk | 220 mg/dL, still high |
| Upma instead (47 g) | 198 mg/dL, still high |
| **Upma + 30-minute walk** | **172 mg/dL** |

One change is not enough; the combination is, and the clinician sees it before the meal rather than
after the spike. These numbers are generated, not typed:
[reports/metrics/twin_example_s151.json](reports/metrics/twin_example_s151.json), and the browser's
output is checked against them in CI (`tests/twin_parity.mjs`). S-151 is a **simulated** patient from
the insulin-resistant synthetic cohort, so her rise per gram of carbohydrate is steeper than the
real-cohort median of 0.33 mg/dL per gram; the walk's size is a literature-based **assumption**
(a 20 % smaller 3-hour rise), labelled on screen. An illustration of the mechanism, not a validated
treatment effect.

## Results at a glance

Real patients, split by patient, unless marked SIMULATED. Each row names its cohort, horizon and
model; full tables and 95 % confidence intervals are in §9.

| What a clinician needs | Result |
|---|---|
| **A two-hour forecast** | Shanghai T2DM (100 real type 2 patients), +120 min, development CGM LightGBM, cross-validated: **RMSE 34.1 mg/dL** against 43.9 for repeating the last value (22 % lower), **97.0 %** in Clarke zones A+B ([baselines_shanghai_t2dm.json](reports/metrics/baselines_shanghai_t2dm.json)) |
| **An accurate next hour** | Served model, held-out test patients, +60 min: mean error **15.70 mg/dL** (CGMacros) and **15.51 mg/dL** (Shanghai T2DM) ([D-072a](docs/DECISIONS.md)) |
| **A range with its coverage measured** | Served model, +60 min: the 80 % range contains the reading **80.5 %** of the time in each cohort overall. Within groups: 79.5–83.7 % across HbA1c bands in Shanghai (all type 2), but **73.8 %** for CGMacros' 14 type 2 patients — calibrating each group separately is the next fix |
| **Warning before a high** | Shanghai T2DM: AUROC **0.82** two hours ahead. At about one unconfirmed alert per patient per day, the model catches **83 %** of episodes against 59 % for a threshold alert, with a median warning of 15 minutes (CGMacros: 49 % against 42 %) |
| **Warning before a low** | Shanghai T2DM: AUROC **0.91** one hour ahead for ranking risk, but at the same alert budget a plain threshold alert catches more lows (84 % against 62 %), so it should stay in place. CGMacros has too few lows to judge (8 patients) |
| **The size of a meal's effect** | CGMacros, real breakfasts eaten twice by the same person, +60 min: the twin's average predicted difference is **27.5 mg/dL against 33.5 observed** (44 pairs); the ML models average 18.6 against 35.3 (87 pairs, a wider set). Per pair the agreement is weak for both (r = 0.21 twin, 0.14–0.16 ML): the twin gets the average size right, not each meal |
| **What each stream adds** | CGMacros, full development model, +60 min, error increase when the stream is removed: meals **+0.92**, HbA1c and fasting glucose **+0.61**, heart rate and activity +0.12 mg/dL. The served model keeps meals, post-meal activity, HbA1c and fasting glucose; heart rate added to CGM alone changes nothing measurable |
| **Learning across clinics** | SIMULATED clinics built from the real patients (a small neural network, not the served model): federated training is within 0.09–0.12 mg/dL of pooled training at +60 min (intervals include zero); at +30 min it is 0.14–0.18 mg/dL behind in the five-clinic layouts |

### How the twin works

A forecaster predicts the next reading. A twin also holds a model of the patient that updates as
data arrives and can be asked what a change would do. Real patients (CGMacros) unless marked.

| A twin must… | How GlucoTwin does it | Measured |
|---|---|---|
| Hold a model of *this* patient | Glucose–insulin ODE (gut absorption, glucose, an insulin surrogate, insulin action), fitted to the patient's first 3 days, starting from a prior set by the health record | Before any of the patient's own data, the record-based prior lowers the twin's error by **1.4 / 2.6 / 3.6 mg/dL** at +30 / +60 / +120 min (each interval excludes zero) |
| Personalise only what the data can identify | Identifiability tested parameter by parameter | Basal glucose is reproducible between two 3-day windows (**r = 0.62**) and recovered almost exactly in simulation (**r = 0.998**). The other five parameters are not reproducible (r = 0.14–0.24), so they stay near population values and only basal glucose is shown to the clinician |
| Update with every reading | Extended Kalman filter: each CGM value corrects the whole state (carbohydrate still in the gut, insulin action, glucose) | Re-estimated every 5 minutes; the console's **Live twin** runs this filter in the browser, matching Python to within 10⁻⁶ mg/dL |
| Size a change | The twin plays a meal forward | Carbohydrate response **0.33 mg/dL per gram**, inside the 0.3–1.3 range pre-registered from the literature, near its lower end (ML models: 0.19–0.20). Average size on matched real breakfasts: see the table above |
| Help the forecast | Twin state passed to LightGBM as features | 0.55 mg/dL lower error at +60 min with 3 days of data (95 % CI 0.01 to 1.09), measured against the untuned model |

**Two engines, each where it measured best.** The tuned LightGBM gives the most accurate forecast;
the twin alone is less accurate (18.1 mg/dL at +60 min) and does not serve forecasts. The twin
carries the patient's state and plays changes forward. In 61 % of test windows the two agree on the
effect of going from 30 g to 60 g of carbohydrate (same direction, within 10 mg/dL or half the
effect). The insulin surrogate cannot represent injected insulin or sulfonylureas, so the twin does
not model medication. Next step: feed the twin's state into the served forecaster and retest.

Of the 75 entries in [docs/DECISIONS.md](docs/DECISIONS.md), 71 set a protocol before the run it
governs (one of them, D-056, is a labelled post-hoc amendment); D-055 records findings, and D-073 to
D-075 record later changes. Every comparison carries a patient-level confidence interval.

> **Research prototype, not a medical device.** Outputs are decision support for a clinician.
> They are not a diagnosis and do not prescribe treatment. Open data and clearly labelled
> synthetic data only; no patient data is stored in this repository.

## 1. Team details

| | |
|---|---|
| Team leader | Jahnavi Harsha |
| Team name | Synapse Squad |
| Team members | Solo submission |

## 2. College / incubator

Indian Institute of Technology Guwahati. Incubator: none.

## 3. Project title

**GlucoTwin: a patient-specific glucose digital twin that fuses the electronic health record with
CGM and wearable streams, forecasts glucose and adverse events, and answers "what if" questions.**

## 4. Problem statement

Type 2 Diabetes is managed between visits that are weeks or months apart. A clinician sees a
HbA1c and a few lab values, while the patient generates thousands of glucose readings, meals and
activity signals that nobody has time to read. India has one of the largest T2D populations in
the world, a small number of diabetologists per patient, and very little tooling built for
Indian diets.

## 5. Healthcare use case

A diabetologist opens one patient and sees, in one screen:

1. **Where glucose is going.** A 30/60/120-minute forecast with an honest 80 % band, and the
   probability of crossing 180 mg/dL (hyperglycaemia) or dropping below 70 mg/dL.
2. **How the last two weeks went.** The standard Ambulatory Glucose Profile, time in range and
   Glucose Management Indicator, following the 2019 international consensus.
3. **What would change it.** A what-if lab (meal size, carbohydrate, timing, a post-meal walk,
   Indian meal presets), always labelled as a model-based estimate.
4. **How much to trust it.** Data quality, how much of the patient's own history the model has
   seen, and which inputs mattered.

The clinician stays the decision maker. The tool never recommends a dose or a drug.

## 6. Technical stack

Python 3.11, NumPy, pandas, LightGBM, scikit-learn, SHAP, PyTorch (GRU baseline, CPU),
Streamlit and Plotly (dashboard), pytest, ruff, black, GitHub Actions, Docker.
Everything runs on a laptop CPU.

## 7. AI / ML model and framework

| Layer | What it is |
|---|---|
| Fusion forecaster (served) | Tuned LightGBM on causal features: recent CGM and its slope, time of day, HbA1c and fasting glucose from the record, and — where the cohort records them (CGMacros) — meal macros and post-meal activity from the wearable. Heart rate was tested and left out (+0.04 mg/dL, CI includes 0). Predicts the *change* from the current value at +30 / +60 / +120 min. Trained with an absolute-error loss to match the metric it is scored on; 16 candidates searched and the best three averaged, chosen on inner-validation patients only (D-072a). |
| Mechanistic twin | A minimal glucose-insulin ODE fitted per patient (RK4 integration, extended Kalman filter for the state, record-based prior). It is the patient state, the engine that sizes what-if effects, and a feature source (−0.55 mg/dL at +60). |
| Personalisation | The population model plus fine-tuning or a recursive-least-squares correction on the patient's own days. |
| Uncertainty | Split-conformal and CV+ intervals (80 % target), Platt/isotonic calibration of event probabilities. |
| What-if engine | Counterfactual edits to meals and activity, played through a LightGBM with a monotonicity constraint in carbohydrate — so less carbohydrate can never predict a higher peak — and through the fitted twin. |
| Sequence model | PyTorch GRU over the causal window with FiLM static conditioning and a multi-horizon head; also the backbone of the physics-informed hybrid (a GRU trained with an ODE-consistency loss). |
| Federated simulation | NumPy MLP trained with FedAvg / FedProx across simulated clinics. |

The model card is [docs/model_card.md](docs/model_card.md); design choices and every decision
taken are in [docs/DECISIONS.md](docs/DECISIONS.md).

## 8. Data

Only open or synthetic data. Nothing is committed to this repository; scripts under `data/` fetch
the open datasets, and checksums are verified.

| Data | Use | Licence |
|---|---|---|
| CGMacros (PhysioNet): 45 adults, CGM, Fitbit heart rate and activity, meal macros, labs | Main development cohort | CC BY-NC-SA 4.0 |
| Shanghai T1DM/T2DM (figshare): 100 T2DM patients, CGM, meals, labs | Second cohort, transfer | CC BY 4.0 |
| Synthetic Indian cohort (this repo's simulator): CGM, heart rate, steps, sleep stages (awake / light / deep / REM), HRV, Indian meals, simulated EHR history (diagnoses, medications, eGFR, genetic-marker dosages) | Stress test, wearable detectability, demo. `make synth-data` writes it with a datasheet | Generated; labelled **SIMULATED** everywhere |

The data card is [docs/data_card.md](docs/data_card.md).

## 9. Key results (numbers come from `reports/metrics/*.json`; served-model figures from the D-072a outcome table in `docs/DECISIONS.md`)

MAE in mg/dL at +60 min, 95 % patient-bootstrap intervals in brackets. All splits are by patient.
Every simulated number is labelled SIMULATED.

**Which number is which.** Several +60 min errors appear in this README because they answer
different questions, each on patients the model never trained on:

| Number (CGMacros, +60 min) | Model | Evaluation |
|---|---|---|
| **15.70** | Served model (tuned LightGBM, D-072a) | Held-out test patients, design A |
| 16.8–17.2 | Development fusion models (§9.1 table, model-family comparison) | Patient-grouped cross-validation |
| 16.06 → 14.94 | Population LightGBM before tuning, then fine-tuned on 7 days of the patient's own data | Common test region from day 7 (personalisation, Part 5) |
| 18.1 | Mechanistic twin alone | Same windows, from day 3 |

Clarke error-grid figures are from the development fusion models; the two-hour Shanghai RMSE
(34.1 mg/dL) is the development CGM model.

### 9.1 Forecasting and fusion (CGMacros, 45 patients)

| Model | +30 | +60 | +120 |
|---|---|---|---|
| Persistence (last value) | 13.4 | 19.5 | 26.9 |
| Per-patient autoregression | 12.2 | 17.4 | 21.8 |
| LightGBM, CGM only | 12.3 | 17.7 | 22.0 |
| LightGBM, CGM + wearable + meals | **11.6** | **16.8** | **21.2** |

- **Meals are the signal that matters.** Adding meals to CGM and wearable data lowers MAE by
  0.93 mg/dL at +60 (CI −1.18 to −0.67); in the 3 hours after a meal by 1.57.
- **What each stream adds to the full model.** Remove one stream from the full development model
  (which predicts the change in glucose) and error at +60 rises: meals +0.92 mg/dL (CI 0.66 to
  1.20), HbA1c and fasting glucose +0.61 (0.07 to 1.30), wearable heart rate and activity +0.12
  (0.04 to 0.22). The full model beats CGM alone by 0.68 mg/dL (0.30 to 1.04). Added as a block on
  top of CGM, wearable and meals, the whole record does not measurably help (+0.21, CI −0.14 to
  0.60): only HbA1c and fasting glucose carry signal, and they are what the served model keeps.
- **The record's biggest job is the cold start.** Next to an hour of CGM, short-term glucose
  is dominated by recent glucose: in Shanghai, where the record is the only extra stream, it does
  not lower +60 error. Before the patient has any history, it gives the twin a 1.4–3.6 mg/dL
  head start, and it drives triage. Heart rate added to CGM on its own moves error by +0.04
  (CI includes 0). Section 9.5 tests whether sleep, HRV and steps would be detectable.
- **In clinical error-grid terms** (development fusion model, CGMacros, +60 min): **99.1 %** of
  forecasts fall in Clarke zones A+B (82.0 % in zone A), MARD 11.8 %. Shanghai, CGM model:
  98.6 % A+B, MARD 12.2 %.
- **The tuned forecaster is the model both dashboards serve (Part 10, D-072a).** Training with an
  absolute-error loss — the metric the model is scored on — and a wider hyperparameter search
  chosen only on inner-validation patients, then re-running calibration and the what-if models for
  it. Error falls and the intervals get tighter while still covering 80 %:

| Cohort | Horizon | MAE served, mg/dL | 80 % coverage | 80 % band width, mg/dL |
|---|---|---|---|---|
| CGMacros | +30 | 11.04 → **10.88** | 81.2 → 80.3 % | 35.9 → **34.5** |
| CGMacros | +60 | 16.23 → **15.70** | 81.3 → 80.5 % | 50.6 → **47.8** |
| CGMacros | +120 | 20.98 → **19.96** | 81.6 → 80.8 % | 65.3 → **61.0** |
| Shanghai T2DM | +30 | 8.89 → **8.73** | 80.9 → 80.5 % | 28.5 → **27.4** |
| Shanghai T2DM | +60 | 15.84 → **15.51** | 81.2 → 80.5 % | 51.7 → **49.2** |
| Shanghai T2DM | +120 | 23.37 → **22.79** | 81.1 → 80.8 % | 77.1 → **74.2** |

  A sharper forecast *and* a narrower interval at the same promised coverage, in both cohorts at
  every horizon. The adoption rule and the 75 % coverage floor were written before the run
  ([D-072](docs/DECISIONS.md)); coverage never fell below 80.3 %.

### 9.2 Adverse events and alarms

- **Event discrimination (AUROC, 95 % CI) by how far ahead.** Two hours ahead, the competition's
  example, the model still ranks risk well but less sharply than at 30 minutes:

| Event | Cohort | +30 min | +60 min | +120 min |
|---|---|---|---|---|
| Glucose above 180 | CGMacros | 0.91 | 0.83 | 0.77 [0.72, 0.81] |
| Glucose above 180 | Shanghai T2DM | 0.96 | 0.89 | 0.82 [0.79, 0.85] |
| Glucose above 140 | CGMacros | 0.85 | 0.77 | 0.73 [0.68, 0.77] |
| Glucose above 140 | Shanghai T2DM | 0.92 | 0.85 | 0.80 [0.78, 0.83] |
| Glucose below 70 | CGMacros (8 patients with events) | 0.95 | 0.88 | 0.78 [0.67, 0.87] |
| Glucose below 70 | Shanghai T2DM | 0.97 | 0.91 | 0.81 [0.77, 0.86] |

  Probabilities are calibrated across each cohort, so a stated 40 % chance means the event happens
  about 40 % of the time on average (not guaranteed within every subgroup). The dashboards serve Platt-calibrated probabilities, whose calibration error at +60 min is
  at most 0.025 across both cohorts and all three events; isotonic calibration reaches 0.018.
- **Alarms at about one false alarm per day:** for hyperglycaemia the model beats both a
  "current value" alert and a trend alert (Shanghai above 180: 83 % sensitivity against 59 % and
  67 %). Alert-burden curves let a clinic choose its own sensitivity / false-alarm trade-off.
- **Intervals hold their promise:** the served model's 80 % bands cover 80.5 % in both cohorts at
  +60 min, against an 80 % target, and stay at 80.3 % or above at every horizon (D-072a). Coverage
  is reported per subgroup as well as pooled, and both dashboards show it.
- **Robustness is tested, not assumed:** sensor faults are injected into the test data and the
  causal features recomputed, and a data-quality score (completeness, recency, plausibility, rate
  of change) travels with every forecast. Section 15 records where this is weakest.

### 9.3 What-if engine

The what-if engine answers counterfactual questions a clinician actually asks: what if this meal
had less carbohydrate, what if it were an idli instead of a dosa, what if the patient walked for
thirty minutes afterwards.

Two engines run side by side: a LightGBM with a **monotonicity constraint in carbohydrate** — so
less carbohydrate never predicts a higher peak (100 % of carbohydrate checks; the walk direction
passes 856 of 857, since a walk also moves unconstrained activity features) — and the fitted
mechanistic twin. Both are scored against physiological plausibility checks
(carbohydrate monotonicity, activity direction, identity, causality, bounds, dose–response) and
against **87 matched real breakfasts**.

On those 87 matched pairs the ML models predict 18.6 to 19.2 mg/dL of the 35.3 mg/dL observed
difference at +60 min. The twin, on the 44 pairs it covers, predicts 27.5 of 33.5 mg/dL, and its
carbohydrate response (0.33 mg/dL per gram) is inside the literature range where the ML models'
(0.19–0.20) is below it. So the ML models are used for direction and ranking, the twin for size,
and the interface labels ML magnitudes as likely under-estimates beside the result. Two cautions: the
twin and ML were scored on different pair sets (the twin needs 3 days of data), and per pair the
agreement is weak for both (r = 0.21 twin, 0.14–0.16 ML), so both reproduce the average size of a
meal's effect better than any single meal's
(details in [docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md)).

### 9.4 India readiness (Part 8)

**(a) Local calibration carries a model to a new population.**
Cross-cohort transfer on a shared 15-minute grid, MAE at +60:

| Direction | Persistence | Zero-shot | Trained in-cohort | Pooled |
|---|---|---|---|---|
| CGMacros → Shanghai T2DM | 19.50 | 19.37 | 15.95 | 15.98 |
| Shanghai T2DM → CGMacros | 19.31 | 20.08 | 17.49 | 17.85 |

A model trained elsewhere is on par with repeating the last value, and 3.4 mg/dL (CI 2.9 to 4.0)
behind one trained on the target population. Pooling the data recovers that gap, and a few days of
the new patient's data close part of it (ridge calibration with 3 days: −0.83, CI −1.19 to −0.50).
**For India the recipe is to train or calibrate on local patients, which the console does.**

**(b) Synthetic Indian cohort (SIMULATED).** 100 virtual patients per setting, 10 days, 16 Indian
meal presets (carbohydrate from IFCT 2017), insulin sensitivity lowered by 0.46 log units
(JCEM 2001 clamp data). Zero-shot models trained on the real cohorts, MAE at +60:

| Insulin-resistance strength | Persistence | Zero-shot | With 3-day calibration |
|---|---|---|---|
| none (control) | 15.40 | 15.21 | 14.35 |
| mild | 16.57 | 17.04 | 15.87 |
| moderate | 16.81 | 16.86 | 15.68 |
| strong | 18.58 | 17.05 | 16.17 |

With 3 days of calibration the model beats repeating the last value in every setting (−0.7 to
−2.4, CIs mostly exclude 0). Adding synthetic patients to training did not improve results
(+0.25 to +0.35 mg/dL), so the synthetic cohort is used to stress-test and demonstrate, not to train.

**(c) Federated learning (SIMULATED clinics).** Open-cohort patients are grouped into clinics;
raw records never leave a clinic and only model updates are shared. MAE at +60:

| Layout | Persistence | Local only | Pooled (central) | FedAvg |
|---|---|---|---|---|
| 5 similar clinics | 19.97 | 19.54 | 17.10 | 17.19 |
| 5 clinics split by HbA1c | 19.14 | 19.19 | 16.90 | 17.03 |
| 2 cohorts as 2 sites | 19.77 | 17.81 | 17.83 | 17.64 |

FedAvg matches pooled training (difference 0.09 to 0.12, CIs include 0) and beats a clinic alone by
about 2 mg/dL when clinics are small. The privacy design is data minimisation: records stay at the
clinic and only model updates travel. An exploratory noise sweep (D-063a) shows formal
differential-privacy guarantees need more clinics than a handful, so none is claimed;
patient-level DP-SGD and secure aggregation are future work.

**(d) Forecasts shown in the live web console (SIMULATED patients, D-069, D-070, D-072a).** The
console's 120 virtual patients are forecast by the served model, trained on the open cohorts and
applied to a population it has never seen, with a three-day calibration on each patient's own
first days — the recipe (a) recommends. Error at every 15-minute anchor from day 4 on, mg/dL,
95 % patient-bootstrap CI:

| Forecast | +30 min | +60 min | +120 min |
|---|---|---|---|
| Repeat the last value | 11.5 [10.6, 12.5] | 17.8 [15.9, 20.0] | 26.6 [23.4, 30.2] |
| Served model, no calibration | 10.9 [10.4, 11.4] | 16.5 [15.4, 17.8] | 21.6 [19.4, 24.0] |
| Served model, 3-day calibration | **10.7** [10.2, 11.2] | **16.0** [14.9, 17.4] | **20.5** [18.3, 23.1] |

The calibrated forecast is now better than repeating the last value at **all three horizons**; at
+60 min the difference is −1.78 mg/dL [−2.91, −0.78], and the gain grows with the horizon, where
early warning matters most. The 80 % bands cover 80.5 / 81.0 / 80.7 % by patient. This is a
simulation of a model meeting a new population, not a measure of accuracy on Indian patients.

### 9.5 Sleep, HRV and step count (SIMULATED)

No open dataset pairs real CGM with real sleep staging and HRV in type 2 patients, so we built a
powered simulation and answered a sharper question than "does it help": *how much data would you
need before a sleep effect of the size the literature reports becomes visible at all?* The answer
is roughly **9,000 patient-days** — a result about the field, not only about this system, and the
reason the console presents these channels as clinical context, which is what a clinician uses
them for.

The design: *if sleep affected insulin sensitivity, could a model detect it in 10 days of data?* On a simulated cohort of 100 patients (67,000 scored windows per condition), sleep
shortens each night and lowers insulin sensitivity from 06:00 the next day. A multiplier m scales
the effect (m = 0 is a false-positive control; m = 1 is literature scale). We compare nested
models: A (CGM, time of day, meals), B (A plus heart rate and activity) and C (B plus sleep, HRV
and steps). Change in MAE from adding the sleep channel (C minus B), mg/dL, 95 % CI:

| m | Mean change in log insulin sensitivity | +30 min | +60 min | +120 min |
|---|---|---|---|---|
| 0 (control) | 0.000 | −0.03 [−0.09, +0.01] | +0.09 [+0.02, +0.17] | +0.18 [+0.06, +0.30] |
| 1 | −0.016 | +0.02 [−0.00, +0.04] | −0.04 [−0.12, +0.04] | +0.12 [−0.01, +0.25] |
| 3 | −0.048 | +0.01 [−0.01, +0.02] | +0.03 [−0.00, +0.07] | +0.15 [+0.04, +0.26] |
| 5 | −0.080 | +0.02 [−0.00, +0.04] | +0.07 [−0.01, +0.15] | +0.11 [−0.01, +0.24] |

**Result:** the control behaves correctly (no false gain), and sleep features add no detectable
forecast accuracy even at five times the literature effect (heart rate and activity: B minus A
within ±0.12). The dashboard therefore presents sleep, HRV and steps as clinical context. In
simulation the effect is small relative to meal-driven variation over 10 days; the pipeline
accepts these channels as soon as a real cohort provides them.

**Larger run (D-066): 300 simulated patients, 30 days, 777,000 scored windows per condition.**
Same design, about 9 times more patient-days. C minus B, mg/dL, 95 % CI (\* = interval excludes 0):

| m | +30 min | +60 min | +120 min |
|---|---|---|---|
| 0 (control) | +0.064 [+0.040, +0.100]\* | +0.067 [+0.047, +0.087]\* | +0.058 [+0.016, +0.113]\* |
| 1 | −0.061 [−0.097, −0.035]\* | −0.031 [−0.073, +0.011] | +0.005 [−0.045, +0.051] |

The control shows no leak. At literature scale (m = 1) the sleep channel is detectable at +30
minutes, by about 0.06 mg/dL (0.5 % of a forecast error of roughly 13 mg/dL), so the dashboard
keeps sleep, HRV and steps as context. This is a simulation result about detectability.

## 10. Dashboard

Two interfaces, both reading the same bundle so they never disagree on a number.

**Web console** — live at **https://cleopatrakilledceaser.github.io/SynapseSquad_IITGuwahati/** (source in `web/`).
A single-page clinician console. A patient list ordered by who needs
attention, then per patient: the forecast with its 80 % band, the **live twin**, the what-if lab, the glucose report
(AGP, time in range, GMI), the health record, sleep and activity context, a visit summary, and an
evidence section carrying every cohort-level result. Role views — a clinician and a care
coordinator see different fields — with a session access log (a demonstration of role-based
display, not authentication). Dark and light themes, a mg/dL ⇄
mmol/L switch, readable at tablet width. The demo runs on 120 **simulated** patients with the
models applied zero-shot.

What is live and what is precomputed, plainly:

| Part of the console | How it is produced |
|---|---|
| **Live twin** tab | **Runs in your browser.** The patient's fitted glucose–insulin model and its Kalman filter (a line-for-line port of `src/mech/model.py`) assimilate the CGM readings one by one. Press play or drag the slider to watch the state change; add a meal or a walk at any moment and the twin plays it forward. The browser reproduces the Python reference to within 10⁻⁶ mg/dL on 1,872 forecasts and on the what-if path (meals, the assumed walk, sensor gaps and the S-151 example; `node tests/twin_parity.mjs`, run in CI). Across the 120 simulated patients the twin's fitted basal glucose lands a median 1.5 mg/dL from the simulator's true value (r = 0.985). |
| Forecast, event probabilities, what-if lab | Computed in advance by the served LightGBM (`make web-real`), because the model is trained on licensed open data that is not redistributed. The Streamlit app runs the same model live on a local machine. |

On simulated patients the twin forecasts closely (7.4 / 9.9 / 15.1 mg/dL at +30 / +60 / +120 min)
because the simulator uses the same equations; on real patients it is 12.7 / 18.1 / 23.2, which is why
the served forecast is LightGBM and the twin carries the state and sizes interventions.

**Streamlit application** (`make app`) — the full working app in Happiest Health purple, with ten
pages: Overview, Live twin, Glucose report (AGP, time in range, GMI), What-if lab, Twin state,
Personalization, Input tiers, Data quality, Evidence, and a PDF export for the patient file. Runs
locally; no data leaves the machine.

Every page of both carries the decision-support disclaimer: the clinician remains the decision
maker and the system never recommends a dose or a drug.

## 11. Demo video

**▶ [Watch the demo video](https://drive.google.com/file/d/1YjyfhFT6rcIV6HiqQKcVzY41-ewbUZoW/view?usp=sharing)**

A 22-minute walkthrough: the live console, the Streamlit application, and the evidence behind
both.

| | Section | Covers |
|---|---|---|
| 0:00 | Opening | What the system does and who it is for |
| 1:44 | What the brief asked for | The four requirements and the two data streams, addressed directly |
| 3:30 | The data | CGMacros, Shanghai T2DM, and what is simulated |
| 5:10 | Architecture | Harmonisation, causal features, patient-grouped splits |
| 6:46 | Live demo — the clinician console | Triage, forecast with band, what-if lab, governance |
| 10:29 | Live demo — the Streamlit application | Twin state, input tiers, data quality, evidence, PDF export |
| 11:10 | Results | Accuracy, ablation, model families, intervals, events, transfer, privacy, sleep |
| 16:36 | How the work was done | The decision log and reproducibility |
| 18:34 | Limits, and what comes next | Each limit paired with the work that closes it |
| 20:47 | Close | What the project establishes |

## 12. Open-source licence

Code: MIT ([LICENSE](LICENSE)). Datasets keep their own licences (section 8). Model outputs derived
from CGMacros (the console's precomputed forecasts, on the `gh-pages` branch with `data.json`,
`twin.json` and `evidence.json`) follow CGMacros' CC BY-NC-SA 4.0 terms: non-commercial use.

## 13. Architecture diagram

[docs/architecture.pdf](docs/architecture.pdf) — one page: the four data streams, harmonisation,
the model layer, calibration, the clinical outputs and both interfaces, with the method,
evaluation protocol, AI/ML models, measured results, privacy design and scope.

## 14. Presentation

**▶ [Open the presentation (PDF)](https://drive.google.com/file/d/1aCuLv0ozmCUV_eY0Iz95sqYekWDIOeYF/view?usp=sharing)**

The twenty slides are in this repository as images, rendered at 3600 × 2027 from HTML and inline
SVG so that every figure is regenerated from `reports/metrics/*.json` rather than redrawn by hand.
Each slide cites the decision and the metrics file it came from in its own footer.

| # | Slide | File |
|---|---|---|
| 1 | GlucoTwin — title | [`docs/slide01_title.png`](docs/slide01_title.png) |
| 2 | The problem: the data exists, but no one can read it | [`docs/slide02_problem.png`](docs/slide02_problem.png) |
| 3 | Why a digital twin, and not another dashboard | [`docs/slide03_why_twin.png`](docs/slide03_why_twin.png) |
| 4 | The data: four streams, two real cohorts, nothing private | [`docs/slide04_data.png`](docs/slide04_data.png) |
| 5 | How the system works, end to end | [`docs/slide05_architecture.png`](docs/slide05_architecture.png) |
| 6 | The clinician console | [`docs/slide06_console.png`](docs/slide06_console.png) |
| 7 | The what-if lab | [`docs/slide07_whatif.png`](docs/slide07_whatif.png) |
| 8 | How accurate is the forecast? | [`docs/slide08_accuracy.png`](docs/slide08_accuracy.png) |
| 9 | What is each data stream actually worth? | [`docs/slide09_ablation.png`](docs/slide09_ablation.png) |
| 10 | Four model families. The simplest one won. | [`docs/slide10_models.png`](docs/slide10_models.png) |
| 11 | Can you trust the interval? | [`docs/slide11_intervals.png`](docs/slide11_intervals.png) |
| 12 | Warning before it happens | [`docs/slide12_events.png`](docs/slide12_events.png) |
| 13 | What is a patient's own history worth? | [`docs/slide13_personalisation.png`](docs/slide13_personalisation.png) |
| 14 | Taking the model to a new country | [`docs/slide14_transfer.png`](docs/slide14_transfer.png) |
| 15 | Learning across clinics without moving patient data | [`docs/slide15_federated.png`](docs/slide15_federated.png) |
| 16 | Would a sleep tracker improve the forecast? | [`docs/slide16_wearables.png`](docs/slide16_wearables.png) |
| 17 | Every choice, written down before the run | [`docs/slide17_decisionlog.png`](docs/slide17_decisionlog.png) |
| 18 | What we built, measured, and did not ship | [`docs/slide18_tried.png`](docs/slide18_tried.png) |
| 19 | What the evidence covers, and what comes next | [`docs/slide19_limits.png`](docs/slide19_limits.png) |
| 20 | What this project establishes | [`docs/slide20_establishes.png`](docs/slide20_establishes.png) |

## 15. Limitations and responsible AI

- 45 + 100 real patients: small cohorts, adults, two countries, none in India.
- The India, wearable-sleep and federated results are **simulations**; they test design, not clinical
  performance.
- The ML what-if models under-estimate the average carbohydrate effect (about half on matched
  breakfasts); their sizes are labelled as likely under-estimates, beside the twin's. Individual
  meal effects are predicted poorly by both engines (per-pair r ≤ 0.21). The walk effect is assumed.
- Sensor bias and drift are not detected by the quality score: a +20 mg/dL bias drops 80 %
  interval coverage to about 65 %. This is stated on the Data quality page.
- Interval coverage is not uniform across subgroups (73 % on T2D rows against 87 % on healthy
  rows); both dashboards report it per subgroup.
- For hypoglycaemia on Shanghai a simple threshold alert outperforms the model at one false alarm
  per day.
- Hypoglycaemia evidence is thin (8 CGMacros patients with events).
- Not validated prospectively; not a medical device. Decision support only.
- Data handling is mapped to India's DPDP Act in [docs/limitations.md](docs/limitations.md).
- More: [docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md) (Parts 4 to 7), [docs/DECISIONS.md](docs/DECISIONS.md).

## 16. Reproduce

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
make test        # 212 unit tests
make smoke       # tiny end-to-end check
make app-data    # build the dashboard bundle (needs the open datasets, see data/)
make app         # open the dashboard
```

Linux/macOS: `python3.11 -m venv .venv && . .venv/bin/activate`, then `make ... PY=python`.
A `Dockerfile` is provided for the test and experiment environment.

Make targets for each experiment: `fusion`, `mech`, `hybrid`, `personalize`, `uncertainty`,
`whatif`, `transfer`, `india`, `federated`, `federated-privacy`, `sleep-power`.

## Repository layout

```
configs/   YAML config (seed, thresholds, experiments)
data/      download scripts and instructions (raw data is never committed)
src/       data features models mech hybrid personalize uncertainty whatif
           robustness federated transfer india dash eval utils
app/       Streamlit dashboard
reports/   metrics/*.json (every number above), figures/*.png
docs/      DECISIONS, RESULTS, RESULTS_SUMMARY, PROGRESS, data card
tests/     pytest suite
```
