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
| Forecaster | LightGBM, one model per horizon, trained on the change from the current glucose. Inputs: CGM history (causal), time of day, meals, heart rate and activity, static EHR fields. |
| Mechanistic twin | Glucose-insulin ODE fitted per patient; EKF state estimate. Only basal glucose (Gb) is stable per patient; other parameters are NOT shown as patient characteristics. |
| Intervals | Split-conformal and CV+ (80 % target) |
| What-if | Monotone-in-carbohydrate LightGBM plus the twin, labelled model-based |

## Training and evaluation data
CGMacros (45 adults, PhysioNet) and Shanghai T2DM (100 adults, figshare). All splits are by patient;
confidence intervals are patient-bootstrap. No patient appears in both train and test.
See [data_card.md](data_card.md).

## Performance (from `reports/metrics`)
CGMacros, MAE mg/dL at +30 / +60 / +120: persistence 13.4 / 19.5 / 26.9; LightGBM with CGM,
wearable and meals 11.6 / 16.8 / 21.2. Event AUROC at +60: 0.83 (above 180) and 0.78 (above 140)
on CGMacros; 0.90 / 0.85 / 0.90 (above 180 / above 140 / below 70) on Shanghai.
Interval coverage at +60: 80.8 % and 79.5 %.

## Known failure modes
- **Different population.** A model trained on one cohort is about 3.4 mg/dL worse than one trained
  on the target cohort and not better than persistence (cross-cohort transfer, CGMacros to
  Shanghai T2DM). Retrain or calibrate on local patients.
- **Sensor bias and drift** lower interval coverage (80 % to about 65 % for +20 mg/dL bias). The
  quality score does not detect this.
- **Subgroups.** Coverage is lower for T2D rows (73 %) than healthy rows (87 %).
- **Hypoglycaemia.** Few events (8 patients on CGMacros); a simple alert beat the model on Shanghai.
- **What-if.** Carbohydrate effects are under-estimated by about half; use as lower bounds.
- **Sleep, HRV and steps** are not in the real training data.

## Ethical considerations
Open and synthetic data only. Decision-support wording throughout. Every synthetic result is
labelled SIMULATED. No claim of causal effect or clinical benefit is made.
