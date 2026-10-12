# DECISIONS

Every non-obvious choice and why. Newest at the bottom.

## D-001 Split over unique patient IDs, not rows (Part 1A)
All partitions are computed over the set of unique patient IDs and rows are mapped afterwards
(`assign_rows`). This makes cross-partition leakage impossible by construction;
`assert_no_patient_overlap` is called inside every split function as a second guard.

## D-002 Stratify splits by diabetes status when available (Part 1A)
Cohorts are small (tens of patients) and imbalanced across healthy/prediabetes/T2D. Without
stratification a test set could miss a class entirely. Allocation uses largest-remainder rounding
inside each stratum; at least one training patient is guaranteed per stratum.

## D-003 Repeated K-fold implemented with KFold/StratifiedKFold over patients (Part 1A)
Because each "sample" is already a unique patient, plain (Stratified)KFold over patient IDs is
exactly grouped CV. Repeat `r` uses `seed + r` so repeats differ but stay reproducible.
Defaults (K=5, repeats=5) live in `configs/splits.yaml`; final values revisited once the patient
count is verified.

## D-004 Default holdout 60/20/20 train/calibration/test (Part 1A)
A calibration set is needed for conformal uncertainty (later part). 60/20/20 is a provisional
default in `configs/splits.yaml`; revisit after verifying participant counts in Step B.

## D-005 Tooling (Part 1A)
- Python 3.11.9 (last 3.11 release with a Windows installer) in a project venv.
- GNU Make from `ezwinports.make` (4.4.1; current) rather than `GnuWin32.Make` (3.81, 2006).
- Line length 100 for black + ruff; ruff rules E,F,W,I,B,UP,N,D,SIM,NPY,PD.
- pre-commit uses local `python -m ruff/black` hooks so versions match requirements-dev.txt.
- Ruff pydocstyle uses the Google convention.
- `np.random.seed` is called once in `set_global_seed` (noqa NPY002) only for third-party code;
  project code must use `np.random.default_rng(seed)`.

## D-006 pandas 2.2.3 and scikit-learn 1.5.2 instead of latest (Part 1A)
The latest versions (pandas 3.0.6, scikit-learn 1.9.1) installed fine, but on this dev box
**Windows Application Control blocked their compiled DLLs on import** (`pandas._libs.json`,
`sklearn.utils.murmurhash`: "An Application Control policy has blocked this file").
numpy/scipy/pyarrow/matplotlib were unaffected. pandas 2.2.3 + scikit-learn 1.5.2 install and
import cleanly, `pip check` is clean and all tests pass, so those are pinned. Pinning pandas 2.x
also avoids the pandas 3.0 copy-on-write and string-dtype behaviour changes. Revisit if the
policy is relaxed. (Likely Smart App Control / WDAC reputation-based blocking of newer, unsigned
binaries; not investigated further.)

## D-007 Known minor quirk: rounding tie-breaks favour calibration (Part 1A)
With largest-remainder rounding and a stable tie-break, ties go to calibration before test
(e.g. 20 synthetic patients in 3 strata give 12/5/3). Every patient is still in exactly one
split. The real counts per stratum will be reviewed once participant numbers are verified in
Step B; tie-breaking can be changed then if needed.

## D-008 Library compatibility on the dev box (Part 1B pre-step) — PENDING OWNER DECISION
Method: each library installed into `.venv` with `-c requirements.txt -c requirements-dev.txt`
as constraints (so it must coexist with pandas 2.2.3 / scikit-learn 1.5.2), then
`python -c "import X; print(X.__version__)"`. Script/log: `%TEMP%\gt_libtest*.{ps1,txt}`. Not added
to requirements.txt.

| Library | Version tested | Install | Import | Note |
|---|---|---|---|---|
| lightgbm | 4.7.0 | ✅ | ✅ PASS | |
| torch (CPU, download.pytorch.org/whl/cpu) | 2.14.1+cpu | ✅ | ❌ BLOCKED | `WinError 4551`: Application Control blocked `torch\lib\c10.dll` (unsigned) |
| shap | 0.51.0 | ✅ | ✅ PASS | |
| statsmodels | 0.15.0 | ✅ | ✅ PASS | |
| flwr (Flower) | 1.39.0 | ✅ | ✅ PASS | Pins `packaging<26`; downgraded packaging 26.3 to 25.0 (see below) |
| opacus | 1.6.0 | ✅ | ❌ BLOCKED | Imports torch, so same `c10.dll` block |
| streamlit | 1.59.1 | ✅ | ✅ PASS | |
| plotly | 7.1.0 | ✅ | ✅ PASS | |
| graphviz (Python pkg) | 0.21 | ✅ | ✅ PASS | Rendering also needs the Graphviz system binary (not tested) |
| python-pptx | 1.0.2 | ✅ | ✅ PASS | |
| matplotlib | 3.11.2 | ✅ | ✅ PASS | already pinned |
| nbformat | 5.11.1 | ✅ | ✅ PASS | |

Side effects found:
- Core stack unchanged after all installs (pandas 2.2.3, scikit-learn 1.5.2, numpy 2.4.6); 14/14
  tests still pass.
- `pip check` now reports `virtualenv 21.14.7 requires packaging>=26.3, have 25.0`, caused by
  flwr's `packaging<26` pin. pre-commit still runs. Must be resolved before pinning flwr (e.g. pin
  virtualenv/pre-commit to a version that accepts packaging 25, or keep flwr out of the dev env).
- `lib_lightgbm.dll` is also unsigned but allowed, so the policy looks reputation-based (Smart
  App Control style), not signature-based. Same pattern as D-006 (older pandas/sklearn passed).
- Also installed for Step B probing: xlrd 2.0.2, openpyxl 3.1.5 (needed to read Shanghai .xls/.xlsx).

Decision: superseded by D-009.

## D-009 torch 2.5.1 CPU + opacus 1.5.4 run locally; flwr kept out of the project venv (Part 1B)
Owner decision: no parts are removed; try an older torch first (Option A), fall back to Colab.
- Throwaway venv (`%TEMP%\gt_torchtest`): torch 2.5.1+cpu + opacus 1.5.2 **pass** import, a
  forward/backward pass and one Opacus DP-SGD step. The older torch DLLs are not blocked.
- opacus 1.5.2/1.5.3 require `numpy<2`, which conflicts with our numpy 2.4.6 pin; opacus 1.6.0
  requires torch>=2.6. **opacus 1.5.4** (numpy>=1.15, torch>=2.0) is the matching version.
- Project venv: torch 2.5.1+cpu + opacus 1.5.4 pass the same smoke test with numpy 2.4.6 /
  pandas 2.2.3 / scikit-learn 1.5.2; `pip check` clean; 14/14 tests pass.
- **Result: torch parts run locally.** Pinned in a separate `requirements-torch.txt` (optional
  install, CPU wheel index). Colab is not needed for now; if a later torch feature needs a newer
  version that is blocked, revisit then.
- Flower: flwr 1.39.0 pins `packaging<26`, which breaks virtualenv/pre-commit (`pip check`). flwr was
  uninstalled from the project venv and packaging restored to 26.3; `pip check` is clean. For the
  federated-learning part we will either find a flwr version compatible with packaging>=26.3 or run
  flwr in its own venv / `requirements-fl.txt`. Decide in that part; nothing pinned now.

## D-010 Reference CGM = Dexcom G6 Pro; recover native readings (Part 1B)
- CGMacros CSVs are 1-min rows with **both** CGM columns linearly interpolated (verified from the
  files: kinks of the interpolation give native steps of exactly 5 min for Dexcom and 15 min for
  Libre in all 45 participants).
- Dexcom is the reference: native 5 min, closer to the forecast horizons, and the same device
  family as BIG IDEAs (transfer). Libre is secondary (cross-sensor checks and flagged fallback).
- Harmonization (Step C) must recover native points and treat anything bridged across a native gap
  > 30 min as missing. The raw 1-min values hide gaps of up to about 9.4 days (participant 018).

## D-011 CGMacros HbA1c is %, despite the dictionary saying mmol/mol (Part 1B)
Range 4.6-8.5 with % cut-offs reproduces the page's 15/16/14 group split exactly. Stored as
`hba1c_pct`; Shanghai's true mmol/mol values are converted to % with the IFCC-NGSP master equation.

## D-012 Time of day is valid in CGMacros; calendar date and weekday are not (Part 1B)
Meal starts cluster by type (breakfast median 08:12, lunch 12:36, dinner 19:00) and HR is higher by
day than at night in 44/44 participants, so the shift is by whole days. Circadian and night-window
features are allowed. Day-of-week and date features are banned (dates shifted 1-2 years).

## D-013 No sleep stream: night-rest proxy only (Part 1B)
No sleep data in any file. A `rest_proxy` (22:00-07:00, participant-relative low HR + low
METs/Intensity) may be derived later. It is never called "sleep" in outputs.

## D-014 "Time since last meal" means "last logged meal" (Part 1B)
Snacks are logged (36/45 participants) and some days have fewer than 3 meals. Every meal-timing
feature is defined on *logged* intake, with a `meal_log_gap` flag. The protocol's ≥3 h fasted
breakfast-to-lunch and lunch-to-dinner windows are the most trustworthy segments.

## D-015 CGM reconstruction method (Part 1C; implements D-010)
`src/data/cgm_reconstruct.py` recovers real Dexcom readings from the 1-min interpolated rows:
1. **Kinks.** A linear interpolant bends only at real readings, so a change in slope per elapsed
   minute (> 0.001 mg/dL/min) marks a real reading. The slope is per elapsed minute because some
   files skip rows (007 has 870 timestamp jumps of 2-23 min).
2. **Collinear readings.** Between two anchors at most 30 min apart, points on the 5-min lattice
   are accepted if they are integers. Evidence that these interior points are real: 99.98% of
   11,915 sloped straight spans have integer values at every interior point.
3. **Every accepted reading must be an integer** (Dexcom reports integer mg/dL). All 668
   non-integer candidates were run ends where the vendor truncated an interpolation; they are
   dropped.
4. **Gridding.** Each grid time t takes the latest real reading at or before t (`merge_asof`
   backward). If that reading is older than 30 min, the point is MISSING (NaN + flag). Values are
   never taken from a future reading (unit-tested in `tests/test_cgm_reconstruct.py`).

Validation on all 45 participants (`reports/metrics/cgm_reconstruction_validation.json`):
- Re-interpolating only the recovered readings reproduces the raw file exactly (max error 0) in
  43/45.
- 015: max error 8.7 mg/dL in 0.4% of rows. These are flat "hold" segments that the method
  correctly leaves missing.
- 007: 7.7% of rows differ. Irregular rows, jittered reading phase, 49 gaps > 30 min; only 77% of
  recovered readings are exactly 5 min apart. **007 is flagged lower quality** (candidate for
  exclusion in sensitivity analyses).
- Median participant: 99.9% of consecutive real readings are exactly 5 min apart.
- 018: the 226-hour (9.4-day) bridged span is now fully MISSING (grid: 50.5% real, 49.4% missing).
- Libre (secondary) uses the same detection without the integer rule (its values are not
  integers) and only counts a Libre reading within its own 15-min period.

## D-016 "Amount consumed" > 100 %: keep logged macros, flag, never rescale (Part 1C)
Tested empirically with `data/scripts/amount_consumed_analysis.py`:
- Sample: isolated CGMacros meals (no other logged meal within 3 h) with carbs > 0 and enough real
  readings; 1,031 meals from 43 participants.
- Response: iAUC and peak rise over 0-120 min, real readings only.
- Compared: logged carbs vs carbs x amount/100.

| subset | meals (patients) | within-patient Spearman: logged carbs | carbs x amount |
|---|---|---|---|
| all | 1,031 (43) | 0.346 | 0.334 |
| amount ≤ 100 % | 965 (43) | 0.354 | 0.374 |
| amount > 100 % | 66 (15) | 0.201 | 0.272 |

Patient-bootstrap 95% CI of the difference (x amount minus logged), iAUC, 1,000 resamples, seed 42:
- all: -0.013 [-0.046, 0.021]
- ≤ 100 %: +0.020 [-0.003, 0.044]
- > 100 %: +0.069 [-0.091, 0.232]

**Inconclusive.** No definition is reliably better. The >100 % subset (where the two definitions
differ most) is small and its CI spans 0. 68% of meals are exactly 100 %.

Decision:
- Keep the dataset's logged macros as the meal fields.
- Keep `amount_pct_raw` untouched and flag `amount_gt100`. Never rescale.
- Later parts may try carbs x amount as a sensitivity feature.
- 028/029 have `amount_missing=True` (the column is absent).

## D-017 Two grids: 5-min CGMacros (primary), 15-min common (transfer) (Part 1C)
- 5-min = native Dexcom. A 15-min grid would discard 2/3 of the real readings and leave only 2
  steps before the +30 min horizon.
- 15-min = native Shanghai Libre H, the coarsest common cadence. Upsampling Shanghai to 5 min
  would invent readings.
- Outputs:
  - `cgmacros_ts5.parquet` for CGMacros modeling;
  - `common_ts15.parquet` (both cohorts, same schema) for transfer. CGMacros-15 uses the same real
    readings and the same causal rule (latest reading at or before t), not averaging.
- Wearables on a grid: mean over (t - step, t], i.e. past samples only.

## D-018 Shanghai: repeated periods, time of day, diet, units (Part 1C)
- **Repeated periods:**
  - 100 patients / 109 periods; 8 patients have more than one.
  - `patient_id` = the split unit; `period_id` = the recording period.
  - Static values come from the earliest period.
  - `days_since_start` restarts each period.
  - Periods are 0.1-154 days apart and are not concatenated.
  - Tested: all periods of a patient always land in the same partition.
- **Time of day is real:**
  - Meal logs peak at 06-07, 11 and 17 h.
  - Mean glucose is lowest at 03-04 h (114 mg/dL) and highest at 09 h (171).
  - 99/100 patients are higher at 08-11 h than at 03-05 h.
  - So clock-time features are allowed for Shanghai. Calendar dates are not.
- **Diet:** free text is parsed to timing + item count + total grams (ml counted as g). No macros
  are invented. "data not available" entries are dropped. meal_type = "unknown".
- **Medications:** class flags (insulin, metformin, sulfonylurea, glinide, AGI, DPP-4, SGLT2,
  GLP-1, TZD) from the summary text; the raw text is kept. Per-row insulin doses are not yet
  harmonized.
- **Units:**
  - HbA1c mmol/mol -> % (IFCC-NGSP);
  - cholesterol mmol/L x 38.67 and TG x 88.57 -> mg/dL;
  - insulin pmol/L / 6 -> uU/mL;
  - height m -> cm.
- **Glucose range:** 39.6-468 mg/dL (wider than CGMacros); range checks are per cohort.

## D-019 Activity harmonization and rest proxy (Part 1C)
- `activity_intensity` on a 0-3 scale (sedentary / light / moderate / vigorous):
  - Fitbit `Intensity` (11 participants) is already 0-3.
  - METs (34 participants) use standard cut-points: METs < 1.5, < 3, < 6, ≥ 6 (raw METs ÷ 10).
  - `activity_source` records which source was used.
- `rest_proxy` (CGMacros only): clock night (22-07 h) AND 30-min mean HR ≤ the participant's
  *expanding, past-only* 25th HR percentile AND no activity ≥ light in the window.
  - NA until one day of HR has been seen; NA for Shanghai.
  - Never called sleep.
- 032 (HR 48%): kept, with per-row `hr_missing`; no imputation.

## D-020 Glycemic thresholds kept at 70 / 140 / 180 mg/dL (Part 1D)
From real readings only (`reports/metrics/eda_summary.json`):

| group | median | 1st-99th pct | <70 | 70-140 | 140-180 | >180 (% time, patient-mean) |
|---|---|---|---|---|---|---|
| CGMacros healthy | 118 | 76-206 | 0.5 | 79.8 | 16.7 | 3.0 |
| CGMacros prediabetes | 129 | 80-246 | 0.4 | 66.7 | 23.9 | 9.0 |
| CGMacros T2D | 155 | 84-333 | 0.1 | 33.5 | 34.8 | 31.6 |
| Shanghai T2D | 129.6 | 57.6-298.8 | 2.5 | 54.7 | 23.1 | 19.6 |

- > 180 separates the groups strongly (3% -> 9% -> 32%) and is the international consensus
  time-in-range bound. Keep it.
- 140 sits within the healthy upper tail (healthy 95th pct = 169): healthy people spend 17% of
  time at 140-180, mostly after meals. Keep it as a **post-meal excursion** marker for
  prediabetes, not as a disease cut-off.
- < 70 is rare in CGMacros (≤ 0.5%) but present in Shanghai (2.5%). Keep the consensus 70.
  Note: Libre sensors are known to read low; the cause of Shanghai's higher rate is not verified.
  - Hypoglycemia is too rare in CGMacros for a standalone hypo model; report this honestly.
- CGMacros T2D shows less hypo than Shanghai T2D, consistent with fewer insulin users (inferred,
  not verified: CGMacros has no medication data).

## D-021 Forecast anchors, targets and event labels (Part 2)
`src/features/windows.py`, `configs/features.yaml`.
- **Anchors (forecast times t):**
  - every 15 min, on the grid;
  - only at a fresh REAL reading, since a forecast is issued when a reading arrives;
  - at least 6 h of history;
  - ≥ 70% real readings in (t - 6 h, t].
  - Eligibility uses the largest N, so N = 2/4/6 h are compared on identical rows.
  - A 15-min stride (same in both cohorts) limits near-duplicate, highly correlated windows.
- **Regression target:** `y_h` = glucose at exactly t + h, only if that grid point is a REAL
  reading; otherwise NaN and the window is dropped for that horizon.
- **Two event definitions** (strict: > 180, > 140, < 70). Both are reported.
  - `<event>_at_h`: y_h beyond the threshold ("will glucose BE high at t + h?").
  - `<event>_any_h`: ANY real reading in (t, t + h] beyond the threshold ("will an excursion OCCUR
    within h?"). This is the clinically more useful alert target.
    - A negative needs ≥ 70% real readings in the horizon (else NaN).
    - A positive seen in real readings counts even with less coverage.
- Carried/missing values never form a target or label (unit-tested: setting every carried value
  to 500 mg/dL changes no target).

## D-022 Splits and CV settings (Part 2)
- **Holdout 60/20/20 (fixed; used for the single final run and later conformal calibration):**
  - CGMacros, stratified by status: 27/9/9 patients (3 per status in calibration and in test).
  - Shanghai, stratified by HbA1c group (<7, 7-9, ≥9, unknown): 60/21/19 patients.
- **Headline CV: 5-fold × 5 repeats, patient-grouped, both cohorts.**
  - CGMacros: K = 5 gives 9 validation patients per fold (3 per status). K = 10 would leave 1-2
    patients per status per fold.
  - Repeats average out the partition luck that dominates with 45 patients.
  - Metrics: per-patient statistics averaged over repeats (each patient counts once), then a
    patient bootstrap (1,000 resamples) for 95% CIs.
- **Nested tuning:** inside each outer fold, an 80/20 stratified patient split of that fold's
  training patients is the inner validation set for every choice. The outer fold and the holdout
  test are never used for a choice.
- **Window length N:** chosen once per cohort by 5-fold patient-grouped CV on HOLDOUT-TRAIN
  patients only (LightGBM CGM + time of day, mean MAE over horizons).
- **Chronological split** (later personalization): the first 2 days of a patient's FIRST period
  are 'adapt', the rest 'eval'; later Shanghai periods are always 'eval'.
- **Shanghai:** the split unit is `patient_id`, so all periods share one assignment (tested).
- sha256 of every split file is in `data/processed/splits/splits_manifest.json`.

## D-023 Features (Part 2)
`src/features/build.py`; explicit sets in `data/processed/features/feature_sets.json`.
- All features are trailing (rows at or before t).
- **CGM features (REAL readings only):**
  - lags, deltas, LS slopes over 15/30/60 min, acceleration;
  - mean/std/min/max over 60 min and over N;
  - TIR and time > 140 over min(3 h, N);
  - real/carried fractions; minutes since the last real reading.
- **Meals:**
  - time since last logged meal;
  - macros and grams over 1/3/6 h;
  - last meal type; last meal `amount_gt100`.
  - Shanghai drops the type/amount features (constant there).
- **Wearables (CGMacros):**
  - HR/activity mean/max/std over 30/60 min and N;
  - post-meal activity (≤ 120 min after the last meal);
  - `rest_proxy_prev_night_frac` (most recent COMPLETED night, 22-07 h).
- **Time:** sin/cos of clock time, `days_since_start`. No calendar features (tested).
- **Static:**
  - CGMacros: age, sex, BMI, HbA1c, fasting glucose/insulin, lipids.
  - Shanghai: age, sex, BMI, HbA1c, diabetes duration, fasting C-peptide (added; 78/100 have it),
    9 medication-class flags.
  - Shared set: CGM + time of day + age/sex/BMI/HbA1c.
- **Missingness:** NaN is kept (LightGBM native). The statistical baselines fall back to
  persistence when their history is incomplete. No Part 2 model needs imputation; later ones fit
  imputation on training patients only.
- Baselines use `cgm_only` and `cgm_tod`. `full` and `shared` are prepared for Part 3.

## D-024 Baselines (Part 2)
- **persistence:** y = glucose(t).
- **linear_trend:** now + d·h·slope_w, with (w, d) tuned; d = 1 (plain extrapolation) is in the
  grid.
- **exp_smoothing:** damped Holt on the last 2 h; recursion verified against statsmodels Holt.
- **ar_pooled:** direct AR(p) per horizon via statsmodels OLS; p (30-180 min of history) tuned.
- **ar_per_patient:** per-patient ridge AR shrunk towards the pooled fit, updated CAUSALLY (only
  that patient's pairs whose target time is ≤ t; tested).
- **lgbm_cgm / lgbm_cgm_tod:** LightGBM on Δ = y - glucose(t); early stopping and leaves {15, 63}
  chosen on inner validation.
- **Event classifiers:** lgbm_cgm_tod features, 31 leaves.
- For events, value-based scores (persistence and the LightGBM regressor) are also ranked for
  AUROC/AUPRC. Brier is reported only for the classifiers (value scores are not probabilities).
- Forecasts are clipped to the cohort's sensor range.

## D-025 Evaluation and reliability rules (Part 2)
- MAE, RMSE, MARD, Clarke A and A+B; AUROC, AUPRC, Brier.
- **CIs:** 1,000 patient-bootstrap resamples (patients, not rows; tested). AUROC/AUPRC bootstraps
  use per-patient score histograms (2,000 bins; matches sklearn within 0.002 in tests). Point
  estimates are exact.
- **Reliability (owner rule):** "not reliable" unless ≥ 30 positive windows AND ≥ 5 patients with
  events. Hypo on CGMacros is never reported as a result (owner policy), only counted.
- **Sensitivity:** every headline table is repeated without 007 (evaluation-only exclusion). The
  018/026 gaps are missing, not filled.
- **Groups:** MAE at 60 min by diabetes status (CGMacros) and HbA1c group.

## D-026 Part 2 findings that shape later parts
- **N = 6 h in both cohorts**, but the gain over 2 h is tiny (CGMacros 16.8 vs 17.2 mg/dL mean
  MAE; Shanghai 16.6 vs 16.9). It is chosen by the rule, not because it clearly matters.
- **The per-patient causal AR matches LightGBM on CGMacros** (+120 min: 21.8 vs 22.0 mg/dL, CIs
  overlap fully). Its advantage over the pooled AR grows with days of data (0 on day 0, about
  1.3-1.6 mg/dL from day 4), as expected for a causal adaptive model. Personalization is
  therefore a real lever. On Shanghai, LightGBM CGM + time of day is best at +120 min (24.6 vs 26.4).
- **Event AUROCs are high largely because many windows are already above the threshold at t.**
  On ONSET windows only (glucose at t ≤ threshold), AUROC falls (e.g. CGMacros hyper180 within
  60 min: 0.84 classifier vs 0.78 persistence; AUPRC 0.40 vs 0.31). Future event claims must
  quote onset-only metrics (`reports/metrics/baselines_event_onset.json`).
- **Holdout test vs CV:** the CGMacros holdout test (9 patients) is about 1.5-2.5 mg/dL worse than
  CV, within its wide CIs. Headline numbers are the CV results; the holdout is a confirmation only.
