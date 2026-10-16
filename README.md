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

## Part 6 results: uncertainty, alarms, robustness (honest summary)
Full tables: [docs/RESULTS.md](docs/RESULTS.md); decisions D-045 to D-049. `make uncertainty`.
Design A (headline: CV repeat 0, 44 CGMacros / 71 Shanghai test patients) and design B
(confirmatory holdout: 9 / 14 test patients). B was never used for any choice.

- **Interval coverage (80 % target, +60 min, population model, design A):**
  - split conformal 80.8 % (CGMacros) / 79.5 % (Shanghai);
  - CV+ 81.3 % / 81.2 %;
  - CQR 80.8 % / 77.0 %. CQR undercovers on Shanghai at +60 and +120 (design A CI below
    80 %).
    - Design B points the same way (75–77 %), but its CI excludes 80 % only for the
      personalized model.
  - Median 80 % width is about 50 mg/dL at +60 min.
- **Coverage is not equal across groups.** On CGMacros at +60 min, split conformal covers
  87 % of healthy rows but only 73 % of T2D rows (CV+ 87 % / 73 %). Pooled coverage hides this.
  On Shanghai, up to 17 % of patients (CQR; 4–8 % for split conformal / CV+) sit more than
  10 points below target at +60.
- **ACI is not informative here (D-049):** it reaches near-exact coverage only because 7–64 %
  of its intervals are infinite. Do not use it as reported.
- **Event probabilities:** AUROC 0.83 / 0.78 (CGMacros hyper > 180 / > 140, +60) and 0.90 / 0.85 /
  0.90 (Shanghai > 180 / > 140 / < 70). After isotonic recalibration on training patients,
  ECE is ≤ 0.014. CGMacros hypoglycemia (< 70) is limited evidence: only 8 patients have events.
- **Alarms (60-min ahead, about 1 false alarm/day, design A):**
  - Hyperglycemia: the model alert beats the naive "current value" and trend alerts.
    - CGMacros > 140: sensitivity 37 % vs 16 % / 8 %.
    - Shanghai > 180: 71 % vs 59 % / 67 %.
    - Its median lead time is 15–30 min.
  - **Hypoglycemia on Shanghai: the simple naive alert is better** (84 % vs 70 % sensitivity
    at about 1 FA/day).
  - Decision curves are positive for hyperglycemia at thresholds 5–40 %. For hypoglycemia, net
    benefit is about 0 for every system.
- **Selective prediction:** ranking by CQR width cuts the MAE-coverage area by 16–23 % vs
  random. A data-quality score alone gives −1 to +3 %.
- **Robustness (+60 min, faults after day 7):** sensor bias and drift are the main risk.
  - A +20 mg/dL bias drops 80 % coverage to 65 % / 63 %. A 20 mg/dL/day drift drops it
    to 46 % / 26 %.
  - Gaps, dropout and compression degrade more gracefully: +0.2 to +4.4 mg/dL MAE and
    ≤ 9 coverage points at the worst level.
  - The data-quality score does not detect bias or drift (its mean stays at 0.97–0.99), and
    its correlation with error is weak (Spearman 0.02–0.09; CGMacros CIs include 0). It flags
    missing data, not wrong data.
- **Item 2, patient level (next-day time above 180 / time in range):**
  - Day-1 CGM beats static labs by 3.4–7.9 percentage points (CIs exclude 0).
  - Adding static labs to day-1 CGM changes nothing (−0.2 to +0.3 points, CIs include 0).
- **Disclosure:** a robustness bug was found after the first final scoring by a consistency
  check, not by test metrics. The test windows were scored a second time; the interval / event /
  alarm outputs were bit-identical (D-049).

## Part 7 results: counterfactual what-if engine (honest summary)
Full tables: [docs/RESULTS.md](docs/RESULTS.md); decisions D-050 to D-055. `make whatif`.
What-ifs are **model-based associations, not causal guarantees**. Bands are observational
(conformal for the ML models, filter variance for the twin), with no counterfactual coverage
guarantee. CGMacros only: Shanghai has no meal macros. The final stage ran once (no forced
re-run); 43 test patients for the plausibility suite, 87 within-patient breakfast pairs.

| Effect | Learned from data or assumed? |
|---|---|
| Carbs / protein / fat / fiber / timing (ML) | learned (population A0 LightGBM) |
| Carbs / timing (twin) | learned (per-patient twin, first 3 days) |
| Post-meal walk, ML | learned (`act_post_meal`); ML only sees a walk once it has happened |
| Post-meal walk, `twin_learned` | learned Part 4 activity effect (effectively 0) |
| Post-meal walk, `twin_assumed` | **ASSUMED**: 20 % lower 3-h iAUC (literature-bounded, D-052) |
| India meal presets | **ASSUMED** recipes; macros from IFCT 2017; out of distribution |

- **Engine checks pass:** zero edits reproduce the stored forecasts exactly; edits never change
  any forecast or filter state before the edited meal (100 % of about 2,500 checks per model).
- **Carbs do not reliably raise the unconstrained model's forecast:** the peak rises across the
  0 to 120 g carb grid in only 25 % of test windows (75 % within 2 mg/dL). The monotone model
  passes by construction and costs nothing in accuracy (test MAE difference -0.16 to +0.16 mg/dL,
  CIs include 0). The twin passes 98 %.
- **Both ML models under-estimate the carb effect:** median 0.20 to 0.22 mg/dL per g of carbs,
  below the pre-registered 0.3 to 1.3 (Franc et al. 2010). This test **failed**. The twin passes
  (0.33). On matched breakfasts, the ML models predict about half of the observed peak difference
  (mono: predicted 18 vs observed 39 mg/dL). The sign is right in 91 % of pairs, but the
  correlation is weak (r = 0.39 to 0.41).
- **Walk direction:** the learned ML model lowers or keeps the forecast in 69 % of post-meal
  windows. The twin's direction pass (67 to 69 %) is limited by its structure: when glucose
  is already below basal, the activity term pulls it up.
- **What needs an owner decision:** use `mono` for what-ifs; how to show the carb-effect
  under-estimation; whether to change the twin's activity term (new pre-registration).

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
