# GlucoTwin

**A clinician-facing digital twin for glucose management in Type 2 Diabetes.**
Happiest Health Digital Twin Challenge 2026.

> **Research prototype, not a medical device.** Outputs are decision support for a clinician.
> They are not a diagnosis and do not prescribe treatment. Open data and clearly labelled
> synthetic data only; no patient data is stored in this repository.

## 1. Team details

| | |
|---|---|
| Team leader | Jahnavi Harsha |
| Team name | **[TEAM NAME: fill in]** |
| Team members | Solo submission |

## 2. College / incubator

Indian Institute of Technology Guwahati. Incubator: **[name, or "none"]**.

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

Design choices and every decision taken are in [docs/DECISIONS.md](docs/DECISIONS.md).

## 8. Data

Only open or synthetic data. Nothing is committed to this repository; scripts under `data/` fetch
the open datasets, and checksums are verified.

| Data | Use | Licence |
|---|---|---|
| CGMacros (PhysioNet): 45 adults, CGM, Fitbit heart rate and activity, meal macros, labs | Main development cohort | CC BY-NC-SA 4.0 |
| Shanghai T1DM/T2DM (figshare): 100 T2DM patients, CGM, meals, labs | Second cohort, transfer | CC BY 4.0 |
| Synthetic Indian cohort (this repo's simulator): CGM, heart rate, steps, sleep stages (awake / light / deep / REM), HRV, Indian meals, static EHR | Stress test, wearable detectability, demo. `make synth-data` writes it with a datasheet | Generated; labelled **SIMULATED** everywhere |

The data card is [docs/data_card.md](docs/data_card.md).

## 9. Key results (all numbers come from `reports/metrics/*.json`)

MAE in mg/dL at +60 min, 95 % patient-bootstrap intervals in brackets. All splits are by patient.
Where a result is negative or neutral, it is reported as found.

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

  Calibration error at +60 is at most 0.014 after recalibration.
- **Alarms at about one false alarm per day:** the model beats a "current value" and a trend
  alert for hyperglycaemia (Shanghai > 180: 71 % sensitivity vs 59 % and 67 %). For
  hypoglycaemia on Shanghai a simple alert is better, and we say so.
- **Intervals:** 80 % bands cover 80.8 % (CGMacros) and 79.5 % (Shanghai) at +60 min. Coverage is
  lower for T2D rows (73 %) than healthy rows (87 %), which the dashboard does not hide.
- **Sensor bias and drift are the main risk**: a +20 mg/dL bias drops 80 % coverage to about 65 %,
  and the data-quality score does not detect it. This is stated on the Data quality page.

### 9.3 What-if engine

Model-based associations, not causal guarantees. The carbohydrate effect is under-estimated by
about half on matched real breakfasts (predicted 18 vs observed 39 mg/dL), so sizes are shown as
lower bounds. A pre-registered dose-response test **failed** and is reported
([docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md)).

### 9.4 India readiness (Part 8, honest)

**(a) Does a model trained on one population work on another? Not for free.**
Cross-cohort transfer on a shared 15-minute grid, MAE at +60:

| Direction | Persistence | Zero-shot | Trained in-cohort | Pooled |
|---|---|---|---|---|
| CGMacros → Shanghai T2DM | 19.50 | 19.37 | 15.95 | 15.98 |
| Shanghai T2DM → CGMacros | 19.31 | 20.08 | 17.49 | 17.85 |

A model trained elsewhere is 3.4 mg/dL worse than one trained on the target population
(CI 2.9 to 4.0) and no better than persistence. Pooling the data recovers the loss. A few days of
the new patient's data close part of the gap (ridge calibration with 3 days: −0.83, CI −1.19 to
−0.50). **The lesson for India: the model must be trained or at least calibrated on Indian
patients; a foreign model should not be deployed unchanged.**

**(b) Synthetic Indian cohort (SIMULATED).** 100 virtual patients per setting, 10 days, 16 Indian
meal presets (carbohydrate from IFCT 2017), insulin sensitivity lowered by 0.46 log units
(JCEM 2001 clamp data). Zero-shot models trained on the real cohorts, MAE at +60:

| Insulin-resistance strength | Persistence | Zero-shot | With 3-day calibration |
|---|---|---|---|
| none (control) | 15.40 | 15.21 | 14.35 |
| mild | 16.57 | 17.04 | 15.87 |
| moderate | 16.81 | 16.86 | 15.68 |
| strong | 18.58 | 17.05 | 16.17 |

Without calibration the model is not reliably better than persistence on a shifted population;
with 3 days of calibration it is (−0.7 to −2.4, CIs mostly exclude 0). **Adding synthetic patients
to training never helped** (+0.25 to +0.35 mg/dL, i.e. slightly worse), so the synthetic cohort is
used to stress-test and demonstrate, not to train.

**(c) Federated learning (SIMULATED clinics).** Open-cohort patients are grouped into clinics;
raw records never leave a clinic and only model updates are shared. MAE at +60:

| Layout | Persistence | Local only | Pooled (central) | FedAvg |
|---|---|---|---|---|
| 5 similar clinics | 19.97 | 19.54 | 17.10 | 17.19 |
| 5 clinics split by HbA1c | 19.14 | 19.19 | 16.90 | 17.03 |
| 2 cohorts as 2 sites | 19.77 | 17.81 | 17.83 | 17.64 |

FedAvg matches pooled training (difference 0.09 to 0.12, CIs include 0) and beats a clinic alone by
about 2 mg/dL when clinics are small. **Formal differential privacy is not claimed.** An
exploratory sweep (D-063a) shows that with only a handful of clinics, adding enough noise for a
meaningful guarantee (ε ≈ 50 to 650) makes the model as bad as or far worse than persistence;
accuracy survives only at ε above 23,000, which is no guarantee. The honest claim is data
minimisation: records stay at the clinic. Patient-level DP-SGD and secure aggregation are
future work.

### 9.5 Sleep, HRV and step count (SIMULATED)

The competition names these signals, but CGMacros contains no sleep, HRV or step data. We therefore
asked a different question: *if sleep affected insulin sensitivity, could a model detect it in 10
days of data?* On a simulated cohort of 100 patients (67,000 scored windows per condition), sleep
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

**Result: no detectable gain, even at five times the literature effect.** The control behaves
correctly (no false gain; five useless features cost a little accuracy). Heart rate and activity
add nothing either (B minus A within ±0.12). So the dashboard shows sleep, HRV and steps as
context for the clinician, not as drivers of the forecast. This is a result from simulation; it
says the effect is small relative to meal-driven variation over 10 days, not that sleep does not
matter. The pipeline accepts these channels as soon as a real cohort provides them.

## 10. Dashboard

`make app` starts a Streamlit dashboard in Happiest Health purple, with ten pages: Overview,
Live twin, Glucose report (AGP, time in range, GMI), What-if lab, Twin state, Personalization,
Input tiers, Data quality, Evidence and a PDF export. A units switch shows mg/dL or mmol/L. Every
page carries the decision-support disclaimer.

## 11. Demo video

**[VIDEO LINK: unlisted YouTube or recorded slides]**

## 12. Open-source licence

Code: MIT ([LICENSE](LICENSE)). Datasets keep their own licences (section 8).

## 13. Architecture diagram

**[LINK: docs/architecture.pdf]**

## 14. Presentation

**[LINK: docs/GlucoTwin_presentation.pdf]**

## 15. Limitations and responsible AI

- 45 + 100 real patients: small cohorts, adults, two countries, none in India.
- The India, wearable-sleep and federated results are **simulations**; they test design, not clinical
  performance.
- The what-if carbohydrate effect is under-estimated; magnitudes are lower bounds.
- Sensor bias and drift are not detected by the quality score.
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
