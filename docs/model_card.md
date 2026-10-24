# Model card: GlucoTwin glucose forecaster

Research prototype for the Happiest Health Digital Twin Challenge 2026. Not a medical device.

## Intended use
- **Who:** a clinician reviewing an adult with Type 2 Diabetes or prediabetes who wears a CGM.
- **What for:** decision support: a short-horizon glucose forecast (30, 60, 120 minutes) with an
  80 % band, event probabilities (above 180, above 140, below 70 mg/dL) and model-based what-if
  estimates.
- **Not for:** diagnosis, dosing, insulin titration, unsupervised use by patients, children,
  pregnancy, Type 1 Diabetes, or any use without a clinician in the loop.

## Models
| Component | Description |
|---|---|
| Forecaster | LightGBM, one model per horizon, trained on the change from the current glucose. Inputs (served model): CGM history (causal), time of day, HbA1c and fasting glucose, plus meal macros and post-meal activity from the wearable where the cohort records them (CGMacros). Heart rate, lipids, insulin, demographics and (Shanghai) medications were tested one stream at a time and left out. |
| Mechanistic twin | Glucose-insulin ODE fitted per patient; EKF state estimate. Only basal glucose (Gb) is stable per patient; other parameters are NOT shown as patient characteristics. |
| Intervals | CV+ conformal (80 % target), served; split-conformal compared |
| What-if | Monotone-in-carbohydrate LightGBM (direction, ranking) plus the twin (size), labelled model-based |

## Training and evaluation data
CGMacros (45 adults, PhysioNet) and Shanghai T2DM (100 adults, figshare). All splits are by patient;
confidence intervals are patient-bootstrap. No patient appears in both train and test.
See [data_card.md](data_card.md).

## Performance
**Served model** (tuned LightGBM, held-out test patients, D-072a), MAE mg/dL at +30 / +60 / +120:
CGMacros 10.88 / 15.70 / 19.96; Shanghai T2DM 8.73 / 15.51 / 22.79. CV+ 80 % coverage at +60:
80.5 % in each cohort overall; 73.8 % for CGMacros' type 2 patients with pooled calibration,
79.8 % with group-wise calibration (D-076). Event AUROC at +60 (README §9.2):
0.83 (above 180) and 0.77 (above 140) on CGMacros; 0.89 / 0.85 / 0.91 (above 180 / above 140 /
below 70) on Shanghai T2DM.

**Development models** (patient-grouped cross-validation, `reports/metrics/fusion_*.json`),
CGMacros MAE at +30 / +60 / +120: persistence 13.4 / 19.5 / 26.9; LightGBM with CGM, wearable and
meals 11.6 / 16.8 / 21.2.

## Known failure modes
- **Different population.** A model trained on one cohort is about 3.4 mg/dL worse than one trained
  on the target cohort and not better than persistence (cross-cohort transfer, CGMacros to
  Shanghai T2DM). Retrain or calibrate on local patients.
- **Sensor bias and drift** lower interval coverage (80 % to about 65 % for +20 mg/dL bias). The
  quality score does not detect this.
- **Subgroups.** With pooled calibration, coverage is lower for T2D rows (73.8 %) than healthy rows
  (85.7 %); group-wise calibration (D-076) brings both to about 80 %, with wider type 2 ranges.
- **Hypoglycaemia.** Few events (8 patients on CGMacros); a simple alert beat the model on Shanghai.
- **What-if.** The ML models under-estimate the average carbohydrate effect by about half; treat their
  sizes as likely under-estimates. Neither engine predicts individual meals well (per-pair r ≤ 0.21).
  The walk effect is assumed. The twin does not model injected insulin or sulfonylureas.
- **Sleep, HRV and steps** are not in the real training data.

## Ethical considerations
Open and synthetic data only. Decision-support wording throughout. Every synthetic result is
labelled SIMULATED. No claim of causal effect or clinical benefit is made.
