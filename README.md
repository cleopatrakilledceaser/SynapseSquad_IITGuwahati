# GlucoTwin

A patient-specific **digital twin for glucose management** in Type 2 Diabetes / prediabetes,
built for **clinicians** (Happiest Health Digital Twin Challenge 2026).

> **Research prototype — not a medical device.** Outputs are decision support only; they are
> not a diagnosis and do not prescribe treatment.

GlucoTwin fuses static clinical data (the *prior*) with dynamic CGM + wearable + meal streams
(the *evidence*) to forecast glucose at +30/+60/+120 min with calibrated uncertainty, fast
per-patient personalization, a mechanistic+ML hybrid twin, and physiologically valid what-if
simulation.

## Status
See [docs/PROGRESS.md](docs/PROGRESS.md). Design choices: [docs/DECISIONS.md](docs/DECISIONS.md).
Project rules: [docs/RULES.md](docs/RULES.md).

## Part 4 results: mechanistic twin (honest summary)
Full tables: [docs/RESULTS.md](docs/RESULTS.md); decisions D-031 to D-037.
CGMacros: 45 patients, identical test windows from day 3 on. MAE in mg/dL at +30 / +60 / +120 min,
k = 3 days of the patient's own data:

| model | +30 | +60 | +120 |
|---|---|---|---|
| mechanistic twin | 12.7 | 18.1 | 23.2 |
| per-patient AR (same 3 days) | 12.2 | 17.4 | 21.5 |
| LightGBM (none of the patient's own days) | 11.4 | 16.6 | 20.8 |
| persistence | 13.2 | 19.4 | 26.6 |

- **The twin alone is not a competitive forecaster.** It beats persistence at +60/+120, is no
  better than AR, and is worse than LightGBM at every horizon. The Shanghai T2DM cohort shows the
  same pattern.
- **Static clinical data help only without the patient's own data:** at k = 0 the static prior
  improves MAE by 1.4 / 2.6 / 3.6; with 1 or more days its effect is 0.3 mg/dL or less.
- **Activity has no measurable effect** in this model: MAE changes by 0.12 mg/dL or less.
- **Only Gb (basal glucose) is stable per patient.** SG, SI, fc, kabs and beta are not
  reproducible between two 3-day windows (test-retest r 0.14–0.24) and must not be shown to
  clinicians as patient characteristics. SG sits at its lower bound in about 40–50% of fits.
- The simulator and what-if outputs are SYNTHETIC illustrations, not forecasts.

## Quick start (Windows, PowerShell)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
make test      # unit tests
make lint      # ruff + black --check
make smoke     # tiny synthetic end-to-end check
```
Linux/macOS: same, using `python3.11 -m venv .venv && . .venv/bin/activate`, then `make ... PY=python`.

## Make targets
`make all | data | features | models | figures | app | test | lint | smoke`

## Data
No raw data is redistributed in this repository. Download scripts / instructions live under
`data/` (added in Part 1 Step B). All splits are by patient; see `src/data/splits.py`.

## Repository layout
```
configs/   YAML config (seed, paths, thresholds, splits)
data/      raw/ interim/ processed/ (gitignored) + download instructions
src/       data features models mech hybrid personalize uncertainty whatif
           robustness federated transfer india explain eval utils
app/       Streamlit dashboard (later)
reports/   metrics/*.json, figures/*.png
docs/      RULES, PROGRESS, DECISIONS, data card
tests/     pytest suite (toy synthetic fixtures)
```

## License
Code: MIT (see `LICENSE`). Datasets remain under their own licenses.
