# GlucoTwin

**A clinician-facing digital twin for glucose management in Type 2 Diabetes.**
Happiest Health Digital Twin Challenge 2026.

**▶ Live clinician console: https://cleopatrakilledceaser.github.io/SynapseSquad_IITGuwahati/** — opens in a browser,
no install, no sign-in. 120 simulated patients.

### What this is, in five lines

- A **type 2 diabetes digital twin**: the static health record fused with CGM, wearable and meal streams.
- Forecasts glucose at **+30 / +60 / +120 min** with calibrated 80 % intervals, and the calibrated
  **probability of an adverse event** (above 180 mg/dL, below 70 mg/dL).
- Answers **"what if"** — a different meal, fewer carbohydrates, a post-meal walk — through a
  monotone-constrained model and a fitted physiological twin.
- Built and tested on **two real open cohorts (145 patients, two countries)** plus a synthetic
  Indian cohort, with **federated training** so records never leave the clinic.
- Every protocol was **pre-registered before the run that tested it** — over 70 numbered decisions
  in [docs/DECISIONS.md](docs/DECISIONS.md) — and every comparison carries a patient-level
  confidence interval.

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
| Fusion forecaster | LightGBM on causal features: recent CGM, time of day, meals, heart rate and activity, and static EHR (age, BMI, HbA1c, fasting glucose, ...). Targets are the change from the current value. |
| Mechanistic twin | A minimal glucose-insulin ODE fitted per patient (RK4 integration, extended Kalman filter for the state). It is used for what-if simulation and as a feature source. |
| Personalisation | The population model plus fine-tuning or a recursive-least-squares correction on the patient's own days. |
| Uncertainty | Split-conformal and CV+ intervals (80 % target), Platt/isotonic calibration of event probabilities. |
| What-if engine | Counterfactual edits to meals and activity, played through a monotone-in-carbohydrate LightGBM and through the twin. |
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

## 9. Key results (all numbers come from `reports/metrics/*.json`)

MAE in mg/dL at +60 min, 95 % patient-bootstrap intervals in brackets. All splits are by patient.
Every simulated number is labelled SIMULATED.

### 9.1 Forecasting and fusion (CGMacros, 45 patients)

| Model | +30 | +60 | +120 |
|---|---|---|---|
| Persistence (last value) | 13.4 | 19.5 | 26.9 |
| Per-patient autoregression | 12.2 | 17.4 | 21.8 |
| LightGBM, CGM only | 12.3 | 17.7 | 22.0 |
| LightGBM, CGM + wearable + meals | **11.6** | **16.8** | **21.2** |

- **Meals are the signal that matters.** Adding meals to CGM and wearable data lowers MAE by
  0.93 mg/dL at +60 (CI −1.18 to −0.67); in the 3 hours after a meal by 1.57.
- **Wearable heart rate and activity add almost nothing on their own** (+0.04 at +60, CI
  includes 0). We report this rather than hide it. Section 9.5 tests whether sleep, HRV and
  steps would be detectable (on simulated data, they are not).
- **Static EHR helps only when the patient has no history.** With 3 days of the patient's own
  data its effect is 0.3 mg/dL or less.
- **Tuning the forecaster gives a small, consistent gain (Part 10).** Training with an
  absolute-error-type loss (the metric we are scored on) and a wider hyperparameter search, chosen
  only on inner validation patients, lowers MAE at +60 min by 0.38 [0.17, 0.58] (CGMacros, meals +
  wearable), 0.63 [0.33, 1.01] (CGMacros, full) and 0.33 [0.15, 0.53] (Shanghai T2DM), about 2 to 4 %.
  On the single-shot holdout (9 and 19 patients) the direction agrees in every case (−0.22, −0.02,
  −0.22) but no interval excludes 0. The dashboard still serves the earlier model, whose
  uncertainty bands and what-if results were calibrated for it ([D-065a](docs/DECISIONS.md)).

### 9.2 Adverse events and alarms

- **Event discrimination (AUROC, 95 % CI) by how far ahead.** Two hours ahead, the competition's
  example, the model still ranks risk well but less sharply than at 30 minutes:

| Event | Cohort | +30 min | +60 min | +120 min |
|---|---|---|---|---|
| Glucose above 180 | CGMacros | 0.90 | 0.83 | 0.76 [0.71, 0.81] |
| Glucose above 180 | Shanghai T2DM | 0.96 | 0.90 | 0.82 [0.80, 0.85] |
| Glucose above 140 | CGMacros | 0.86 | 0.78 | 0.72 [0.67, 0.76] |
| Glucose above 140 | Shanghai T2DM | 0.92 | 0.85 | 0.80 [0.78, 0.83] |
| Glucose below 70 | CGMacros (8 patients with events) | 0.95 | 0.88 | 0.78 [0.67, 0.87] |
| Glucose below 70 | Shanghai T2DM | 0.97 | 0.90 | 0.81 [0.76, 0.85] |

  Probabilities are calibrated: after recalibration the calibration error at +60 min is at most
  0.014, so a stated 40 % chance means the event happens about 40 % of the time.
- **Alarms at about one false alarm per day:** for hyperglycaemia the model beats both a
  "current value" alert and a trend alert (Shanghai above 180: 71 % sensitivity against 59 % and
  67 %). Alert-burden curves let a clinic choose its own sensitivity / false-alarm trade-off.
- **Intervals hold their promise:** 80 % bands cover 80.8 % (CGMacros) and 79.5 % (Shanghai) at
  +60 min, against an 80 % target. Coverage is reported per subgroup as well as pooled, and both
  dashboards show it.
- **Robustness is tested, not assumed:** sensor faults are injected into the test data and the
  causal features recomputed, and a data-quality score (completeness, recency, plausibility, rate
  of change) travels with every forecast. Section 15 records where this is weakest.

### 9.3 What-if engine

The what-if engine answers counterfactual questions a clinician actually asks: what if this meal
had less carbohydrate, what if it were an idli instead of a dosa, what if the patient walked for
thirty minutes afterwards.

Two engines run side by side: a LightGBM with a **monotonicity constraint in carbohydrate** — so
less carbohydrate can never predict a higher peak, guaranteed by construction rather than hoped
for — and the fitted mechanistic twin. Both are scored against physiological plausibility checks
(carbohydrate monotonicity, activity direction, identity, causality, bounds, dose–response) and
against **87 matched real breakfasts**.

On those matched meals the models predict 18 mg/dL of the 39 mg/dL observed rise, so the interface
presents every magnitude as a **lower bound**, printed beside the result rather than buried
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

**(d) Forecasts shown in the web console (SIMULATED patients, D-069 and D-070).** The console's
120 virtual patients are forecast by the models trained on the open cohorts. Error at every
15-minute anchor from day 4 on (126,240 anchors at +60 min), mg/dL, 95 % patient-bootstrap CI:

| Forecast | +30 min | +60 min | +120 min |
|---|---|---|---|
| Repeat the last value | 11.5 [10.6, 12.5] | 17.8 [15.9, 20.0] | 26.6 [23.4, 30.2] |
| Model, no calibration | 12.3 [11.8, 12.8] | 19.0 [17.8, 20.3] | 26.3 [24.4, 28.5] |
| Model, 3-day calibration | 11.7 [11.1, 12.2] | 17.2 [16.1, 18.5] | 22.8 [20.9, 24.9] |

Calibrated minus repeat-the-last-value: +0.18 [−0.50, +0.76] at +30 min and −0.60 [−1.89, +0.56]
at +60 min (on par), and −3.84 [−5.82, −2.08] at +120 min, a clear gain where early warning
matters most. Calibration also brings the 80 % bands to 81 to 83 % average coverage at every
horizon. This is a simulation of a model meeting a new population, not a measure of accuracy on
Indian patients.

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
attention, then per patient: the forecast with its 80 % band, the what-if lab, the glucose report
(AGP, time in range, GMI), the health record, sleep and activity context, a visit summary, and an
evidence section carrying every cohort-level result. Access is role-gated — a clinician and a care
coordinator see different fields — with a session access log. Dark and light themes, a mg/dL ⇄
mmol/L switch, readable at tablet width. The demo runs on 120 **simulated** patients with the
models applied zero-shot.

**Streamlit application** (`make app`) — the full working app in Happiest Health purple, with ten
pages: Overview, Live twin, Glucose report (AGP, time in range, GMI), What-if lab, Twin state,
Personalization, Input tiers, Data quality, Evidence, and a PDF export for the patient file. Runs
locally; no data leaves the machine.

Every page of both carries the decision-support disclaimer: the clinician remains the decision
maker and the system never recommends a dose or a drug.

## 11. Demo video

**[VIDEO LINK — unlisted YouTube, to be added]**

A timed walkthrough of the live console, the Streamlit application and the evidence behind both.

## 12. Open-source licence

Code: MIT ([LICENSE](LICENSE)). Datasets keep their own licences (section 8).

## 13. Architecture diagram

[docs/architecture.pdf](docs/architecture.pdf) — one page: the four data streams, harmonisation,
the model layer, calibration, the clinical outputs and both interfaces, with the method,
evaluation protocol, AI/ML models, measured results, privacy design and scope.

## 14. Presentation

**[PRESENTATION LINK — docs/GlucoTwin_presentation.pdf, to be added]**

## 15. Limitations and responsible AI

- 45 + 100 real patients: small cohorts, adults, two countries, none in India.
- The India, wearable-sleep and federated results are **simulations**; they test design, not clinical
  performance.
- The what-if carbohydrate effect is under-estimated; magnitudes are lower bounds.
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
