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

## Part 5 results: hybrid twin and fast personalization (honest summary)
Full tables: [docs/RESULTS.md](docs/RESULTS.md); decisions D-038 to D-044.
`make hybrid`, `make personalize`.

**Hybrid twin (CGMacros, 45 patients, same Part 4 windows; MAE mg/dL at +30 / +60 / +120):**

| model (k = 3 days) | overall | onset (rising past 140) |
|---|---|---|
| A0: LightGBM, CGM + meals + glycemic static | 11.4 / 16.8 / 21.1 | 19.5 / 24.0 / 24.7 |
| A: A0 + causal twin features | 11.2 / 16.2 / 20.2 | 19.0 / 23.9 / 25.0 |
| B: LightGBM on the twin's residual | 11.5 / 16.6 / 20.7 | 19.0 / 24.3 / 25.0 |
| C: physics-informed GRU (5 seeds) | 11.5 / 16.3 / 20.8 | 21.6 / 27.0 / 28.0 |
| twin alone | 12.7 / 18.1 / 23.2 | 23.9 / 29.6 / 31.8 |

- **The twin adds a small amount beyond A0 once it has 3 days of data:**
  - A − A0 = −0.18 / −0.55 / −0.85 mg/dL. Only +60 has a CI excluding 0; post-meal is −0.41 / −0.75.
  - At k = 0 the gain is −0.21 mg/dL at +30 only. At k = 1 there is no gain.
  - Activity features contribute about 0.1 mg/dL.
  - It does not help onset windows.
- **Hybrid B (residual) and C (physics loss) do not help.**
  - B is never better than A0 overall, and is worse at k = 1.
  - C is no better than A0 overall, and is 2–3 mg/dL worse on onsets.
  - The physics term was chosen in only 2 of 5 folds, and only at its smallest weight.
  - With λ = 0, C is equally good overall, so the physics loss itself adds nothing.
- **Neural onset-weighted loss:** a trade-off, not a fix.
  - The Part 3 GRU's onset MAE improves by 3–12 mg/dL in both cohorts.
  - Overall MAE gets worse by 1–5 mg/dL.

**Personalization (both cohorts; common test region day ≥ 7; 44 CGMacros / 71 Shanghai patients):**
- **No static head start for the ML models:**
  - At k = 0, adding static features changes +60 MAE by −0.13 (CGMacros) and −0.23 (Shanghai) mg/dL.
  - Both CIs include 0, and the sign is "slightly worse with static".
  - The crossover is therefore trivial (k\* = 0). There is no static advantage to overtake.
- **The static prior helps only the mechanistic twin at k = 0** (CGMacros: +1.2 / +2.2 / +2.6
  mg/dL head start). Its patient data overtake it at about 2 days (k\* 1.6–2.3 d).
  - But the twin is 3–5 mg/dL worse than LightGBM at every k.
- **Patient data do help, modestly:**
  - LightGBM fine-tuning on the patient's own days improves +60 MAE by 0.3 / 0.9 / 1.3 mg/dL at
    k = 1 / 3 / 5 (CGMacros, with static) and by 0.2 / 0.4 / 0.8 on Shanghai.
  - At k = 3, 30 of 44 CGMacros patients improve.
  - Continuous RLS correction (it keeps learning from every past reading) gives −0.7 to −1.1 mg/dL.
  - Per-patient models from scratch are worse than the population model at every k.
  - **Caveat on the online-learner (RLS) rows (D-043):** the RLS design was revised after
    smoke runs printed MAE for fold 0's TEST patients. Hyperparameters are still chosen on
    training patients only, but those test patients influenced the design.
    - Re-evaluated without fold 0's test patients (36 CGMacros / 56 Shanghai; saved predictions,
      no re-run; `personalize_excl_fold0.json`), +60 min, with static, vs no adaptation:
      - continuous RLS: −0.84 [−1.20, −0.54] vs −1.13 with all patients (CGMacros);
        −0.76 [−1.06, −0.47] vs −0.88 (Shanghai);
      - frozen RLS at k = 3: −0.65 [−1.14, −0.20] vs −0.88 (CGMacros); −0.40 vs −0.43 (Shanghai);
      - fine-tuning (not affected by the redesign) at k = 3: −0.69 vs −0.89 (CGMacros).
    - The direction and significance hold; magnitudes are about 0.1–0.3 mg/dL smaller.
- **The twin does not make personalization better:** hybrid − LightGBM after adaptation is within
  ±0.5 mg/dL on CGMacros. On Shanghai the hybrid is 0.2–0.5 mg/dL better without adaptation.
- **What the data do and do not support:**
  - "The wearable stream keeps it current" is supported in the narrow sense that the patient's
    own CGM history improves the model by about 1 mg/dL over days.
  - "The clinical profile gives it a head start" is NOT supported for the ML models in these two
    cohorts.

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
