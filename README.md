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
