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

## D-027 Part 3 ablation protocol (configs/fusion.yaml)
- **Reuse:** cached Part 2 anchor tables (N = 6 h), the same 5×5 patient-grouped CV folds, the
  same nested inner 80/20 patient split (seed 42 + repeat) for early stopping and leaves. Part 2
  persistence / per-patient AR predictions are merged in, never rerun. The holdout test is unused
  in Part 3 (nothing here is a final-test claim).
- **Two targets for every LightGBM config:** `abs` = y(t+h); `delta` = y(t+h) - base(t), with
  base = current CGM (T2/T3 tiers: the last fingerstick). Errors are always reported on absolute
  glucose. Reconstruction is unit-tested.
- **"A_static, delta"** uses the current CGM value as its base, so it means "current glucose +
  static". It is labeled that way and is not a static-only model. The pure static-only result is
  "A_static, abs".
- **Feature groups** are disjoint and resolved by prefix (tested; dropping a group removes
  exactly its columns).
  - Time of day is in every dynamic config, so B = Part 2's lgbm_cgm_tod. Fusion gains are
    measured against the strongest pooled CGM model, not a weaker CGM-only one.
  - `act_post_meal` belongs to the meals group because it needs meal timing.
  - "Drop wearable" removes HR, activity and rest_proxy. "Drop rest_proxy" alone is a separate
    run.
- **Static groups:**
  - CGMacros: demographics/anthropometrics (age, sex, BMI), glycemic (HbA1c, fasting glucose),
    insulin (fasting insulin), lipids (TG, total cholesterol, HDL, LDL).
  - Shanghai: same demo/anthro and lipids; HbA1c (+ fasting glucose); diabetes duration;
    C-peptide (fasting C-peptide + fasting insulin, both beta-cell function); 9 medication
    flags; kidney function (creatinine, eGFR).
  - Creatinine and eGFR were added to the static table from Shanghai_T2DM_Summary.xlsx
    (78 / 77 of 100 patients). The dynamic grids were unchanged (file hashes identical).
- **Evaluation subsets:**
  - post-meal = a logged meal ≤ 180 min before t (CGMacros);
  - rising onset = glucose at t ≤ 140 and a real reading > 140 within h (regression);
  - event onset-only = windows starting below the threshold (above for hypo), as in D-026.
- **Paired comparisons:** per-patient MAE sums on IDENTICAL rows for both models, the same
  resampled patients in each of 1,000 draws. "Not distinguishable" = 95% CI includes 0.
- **CGMacros hypo** is computed and labeled "limited evidence" (19 patients with events) in every
  table, never headlined (owner).

## D-028 Neural fusion model (src/models/neural.py)
- **Architecture:** GRU over the last 6 h at 15-min resolution (24 steps), channels:
  - glucose (real only) + real-mask;
  - HR, activity;
  - meal macros logged in each 15-min bin;
  - time of day sin/cos.

  Static vector + NaN mask → FiLM (gamma, beta) on the final GRU state → MLP head with 3 outputs
  (30/60/120), delta target, L1 loss.
- **Causality:** sequences end AT t; meals are binned to the first grid row at/after the meal
  (tested by perturbing the future).
- **Fitted on training patients only:** scalers and imputation (fold mean; NaN → 0 after
  standardising, plus mask channels).
- **Configs:** B (CGM + time) and E (full; Shanghai: CGM + time + static).
- **Hidden size** {32, 64} chosen once on the holdout-train inner split (seed 0); dropout 0.1,
  lr 2e-3, batch 256, ≤ 40 epochs, patience 5 on the inner patient validation split.
- **Cost:** CV repeat 0 only (5 folds) × 5 seeds. Full 5×5 × 5 seeds would be 5× the measured
  ~105 min. Neural, LightGBM and ensemble comparisons are therefore made on repeat 0 rows only
  (paired).
- **Ensemble:** mean of LightGBM E (delta) and the seed-averaged neural E.

## D-029 Input tiers and simulated fingersticks
- **Tier 1:** full fusion (E). **Tier 2:** no CGM (wearable + meals + static + time) +
  fingersticks. **Tier 3:** static + time + fingersticks.
- **Fingersticks:** the REAL CGM value at fixed clock times, `fs4` = 07:00, 11:30, 17:30, 21:30
  (fasting, pre-lunch, pre-dinner, bedtime; a typical 4-point SMBG profile) and `fs2` = 07:00,
  17:30.
  - If no real reading exists at that clock time, the stick is skipped (no imputation).
  - Features: last value, minutes since it, previous value, last change, 24-h mean and count.
  - Only sticks with time ≤ t are used (tested).
- **Limitations:**
  - Clock times are valid but days are shifted (D-012), which does not matter here.
  - Real fingersticks have larger meter error (ISO 15197 allows ±15%) than the CGM values used,
    so tiers 2/3 are optimistic.
  - Shanghai has no wearables, so only T1 and T3 are run there.
- **Delta target in T2/T3** uses the last fingerstick as its base (training-fold mean before the
  first stick). The "last fingerstick only" persistence is reported as the tier baseline.

## D-030 Part 3 findings that shape later parts
Source: `reports/metrics/fusion_<cohort>.json` (MAE mg/dL, mean [95% patient-bootstrap CI];
paired diffs m1 - m2, < 0 = m1 better; out-of-fold, patient-grouped CV, train patients only).
- **CGMacros (45 patients).**
  - Meals are the only added stream with a clear gain: D vs C delta -0.74 / -0.93 / -0.91 at
    +30/+60/+120. The gain is larger post-meal (-0.97 / -1.57 / -1.76) and at hyper-140 onset
    (-3.14 / -2.16 / -1.17). Leave-meals-out costs +0.75 / +0.92 / +0.96.
  - Wearables add nothing at +30/+60 (ns) and slightly hurt at +120 (C - B delta +0.16).
  - Static: only the glycemic group helps (leave-out +0.33 / +0.61 / +0.90). E vs D is ns
    at all horizons.
  - Best LightGBM is `D_delta` / `E_delta` (11.7 / 17.0 / 21.8). It beats per-patient AR at
    +30/+60 (-0.46 / -0.41) and ties it at +120 (0.00 ns). At hyper-140 onset it is much better
    (-4.07 / -2.95 / -1.74).
  - Delta beats absolute targets. E_abs at +120 is worse than D_abs (+1.68).
  - Excluding patient 007 changes nothing (largest shift 0.1 mg/dL).
- **Shanghai T2DM (100 patients).** CGM-only LightGBM (`B_delta`, 9.2 / 16.5 / 24.6) beats AR
  (-0.30 / -0.69 / -1.75). Adding static (E) is slightly worse (+0.09 / +0.22 / +0.27 ns). No
  leave-one-group-out static group helps, except a tiny demo/anthro effect at +120 (+0.24).
- **Neural (GRU + FiLM, CV repeat 0, 5 seeds).**
  - `nn_B_cgm` has the best overall MAE in both cohorts:
    - CGMacros 12.1 / 16.9 / 20.9;
    - Shanghai 8.9 / 15.8 / 23.6;
    - vs LightGBM B: -0.27 / -0.60 / -0.99 on Shanghai.
  - It is clearly WORSE at hyper-140 onset (Shanghai vs LightGBM B: +1.21 / +3.08 / +2.99;
    CGMacros: +2.64 / +3.08 / +2.24). Working hypothesis: it predicts conservatively, toward
    the mean. Better on average, worse at the clinically important excursions.
  - `nn_E_full` is worse than `nn_B_cgm` (static overfits with 45-100 patients) and less stable
    across seeds (sd up to 0.5 vs ≤ 0.12).
  - Leakage probe (Shanghai):
    - the gain holds in all 5 folds and for 88 / 82 / 72 of 100 patients;
    - corrupting all glucose after the anchor left 300/300 input sequences unchanged;
    - effect size is about 3-4%, plausible rather than suspicious.
  - The ensemble (`ens_E_full`) helps on Shanghai (-0.16 / -0.36 / -0.43 vs LightGBM E) and is
    neutral on CGMacros.
- **Tiers.** Without CGM, fingersticks beat fingerstick persistence by a wide margin, but stay
  far from CGM. Examples:
  - CGMacros T2 fs4 delta: 21.7 / 22.6 / 24.0 vs T1 11.7 / 17.0 / 21.8.
  - Shanghai T3 fs4 delta: 26.8 / 28.0 / 29.1, worse than CGM persistence at +30/+60.
  - These numbers are optimistic (no meter error, D-029).
- **Events (+60 min, onset-only).** LightGBM beats value persistence:
  - hyper-180 onset AUROC: CGMacros 0.879 vs 0.775; Shanghai 0.910 vs 0.815.
  - Hypo < 70 is "limited evidence" on CGMacros (19 patients with events) and reliable on
    Shanghai (65 patients, onset AUROC 0.955).
- **Groups.** Error scales with dysglycemia: CGMacros at +60 is healthy 13.3, prediabetes 16.1,
  T2D 21.8.
- **Consequences:**
  - Part 4 uses meals + CGM (+ glycemic static) with a delta target as the ML residual on top of
    the mechanistic model. Wearables and the remaining static groups are optional inputs only.
  - Always report onset / post-meal subsets next to overall MAE. Do not select models on overall
    MAE alone.
  - The neural model is kept as a candidate, but not adopted as the default forecaster until its
    onset weakness is addressed.


## D-031 Mechanistic model structure and solver (Part 4)
- **States:** two-compartment gut (Q1, Q2, g of carbohydrate), glucose G (mg/dL), insulin
  surrogate U, insulin action X, smoothed activity E. Equations: `src/mech/model.py` docstring.
- **Gastric slowing:** k = kabs / (1 + β·r), with r = (fat + 0.5·protein + fiber) / (carbs + 1) of
  the most recent meal.
- **Insulin surrogate (insulin is not measured):** driven by softplus(G − Gb) (owner change: smooth,
  not a hard rectifier) plus a feed-forward from gut appearance. X is offset so that rest (G = Gb,
  empty gut, no activity) is an exact fixed point (tested).
  - **Limitation:** it cannot represent exogenous insulin, sulfonylureas or β-cell failure beyond
    what SI/SG absorb.
- **Activity:** G' gains −α_E·E·(G − Gb), which speeds the return to basal and never pushes
  glucose below it.
- **Parameters:**
  - Fitted on a log scale within bounds: Gb, SG, SI, fc, kabs, β.
  - Fixed: p2, τI, cR, τE, softplus width (values in `configs/mech.yaml`).
  - α_E is population level (D-033).
- **Solver:** fixed-step RK4 on the native grid (5 min; 3 sub-steps on the Shanghai 15-min grid),
  with states clipped to physical ranges.
  - Compiled with numba 0.68.0 / llvmlite 0.50.0 (pinned; verified to install and compile in the
    venv). A no-op fallback decorator keeps the code runnable with plain numpy.
  - Tests: bound corners stay finite; RK4 step halving converges.
- **Fitting:** MAP via `scipy.optimize.least_squares` (bounded) on real readings only.
  - Glucose is reset to the observed reading every 6 h, so each segment is an open-loop run from a
    known value, which avoids long-horizon drift.
  - Gaussian prior on log parameters.
- **Forecast:** an EKF assimilates CGM ≤ t, then runs open-loop with no future meals and activity
  0. Predictive variance = P_GG + R.

## D-032 Nested selection for the twin (Part 4, owner change 2)
Folds are Part 2/3 CV repeat 0, the same patient folds for the twin, the static prior, LightGBM
and AR (owner change 4). Inside each fold, everything chosen uses training patients only:
1. **α_E:** grid search over 3-day fits of training patients, scored by their forecast MAE after
   day 3.
2. **Population prior:** mean and sd (with floor) of the training patients' 3-day log estimates.
3. **Static ridge prior:** ridge alpha per parameter by leave-one-out over training patients;
   missing values imputed with the training median.
4. **EKF (Q scale, R), per k:** training patients get parameters exactly as a test patient would
   at that k. The lowest mean Gaussian NLL wins.
   - **At k = 0** the training patients use the population mean (owner change 2).
   - Selection uses a subsample of training anchors.
- **Grids widened once, before the full run:** the timing fold hit the grid edges (α_E 0.0005 =
  lowest; Q/R corner 4/64). The new grids are α_E 0.0001–0.008, q_scale {1, 4, 16, 64},
  R {25, 64, 144, 256}. This decision used training-patient scores only, and the timing fold was
  recomputed in the full run.
- **Choices made in the full run:**
  - α_E = 0.0001 in 4/5 folds and 0.00025 in 1/5, again the lowest value. The data support no
    meaningful activity effect (D-036). Fold-0 MAE across the α_E grid spans only 18.84–19.22.
  - Q scale = 16 in all folds (interior of the grid).
  - R = 25 at k = 0/3, R = 144–256 at k = 1 (noisier 1-day parameters → the filter trusts the
    model less). R = 25 is the lower grid edge; at fold 0, k = 3, its NLL differs from R = 64 by
    0.005, so the surface is flat there and the edge choice is immaterial.
- **Test patients:** parameters use only their first k days. The EKF state uses their real CGM
  ≤ t at every k, like persistence and AR. LightGBM uses none of the patient's own days
  (owner change 1); AR is refit on exactly the first k days, so it does not exist at k = 0.

## D-033 Activity input and resting levels (Part 4, owner change 3)
- **Input:** A(t) = (intensity − rest)+ + (HR − HR_rest)+ / 20 bpm. Missing parts count as 0.
- **Resting level:** a past-only expanding 25th percentile of the patient's own samples before t.
- **No past data:** until 72 samples exist, defaults are used (HR 70 bpm, intensity 0). Tested:
  values after t never change the resting level at t.
- **Ablation:** α_E is population level only. The ablation is reported separately for the METs
  (34) and Fitbit-intensity (11) patients.


## D-034 Identifiability: structural vs practical (Part 4)
Run on CGMacros with the first 3 days of each patient. Figures: `mech_parameter_recovery.png`,
`mech_profile_likelihood.png`. Metrics: `reports/metrics/mech_identifiability.json`.

| param | synthetic r(true, est) | Laplace contraction (real, median) | test-retest r (real, 45 pts) | median abs Δlog (retest) | verdict |
|---|---|---|---|---|---|
| Gb | 1.00 | 1.00 | 0.62 | 0.08 | practically identifiable |
| SG | 0.77 | 0.86 | 0.22 | 0.92 | structurally only |
| SI | 0.98 | 1.00 | 0.24 | 1.07 | structurally only |
| fc | 0.80 | 0.99 | 0.19 | 0.84 | structurally only |
| kabs | 0.83 | 0.96 | 0.18 | 0.61 | structurally only |
| β | 0.89 | 0.69 | 0.14 | 1.82 | structurally only |

- **The test-retest check was added after the first verdict.** The original 3-criterion rule
  (contraction, profile, recovery) called all 6 parameters identifiable. The k = 1 vs k = 3 fits of
  the same patients disagreed badly (log-scale r 0.24–0.43, Gb 0.75), so I treated the verdict as
  suspicious. Test-retest compares MAP fits on two disjoint 3-day windows (days 0–3 vs 3–6) per
  patient; the verdict is "practical" only if r ≥ `retest_min` = 0.5.
- **With ideal data**, all 6 parameters are recoverable: the model is structurally identifiable
  from CGM plus logged meals.
- **On real data, only Gb is stable.** For the other parameters, two windows of the same patient
  differ by a median factor of e^0.6–e^1.8 (≈ 2–6×).
- **Laplace contraction overstates information** because it assumes independent residuals, while
  real CGM residuals are autocorrelated (lag-1 0.82, D-035). It is reported but not trusted.
- **Posterior correlations (median, real):** kabs–β 0.76, fc–kabs −0.52, SG–SI −0.49, Gb–SI 0.46.
  These trade-offs plus meal-log error (timing, portions) and day-to-day physiology explain the
  poor retest agreement.
- **Consequence:** per-patient SG, SI, fc, kabs and β must NOT be shown to clinicians as patient
  characteristics. Only Gb (basal glucose) is stable enough to display, with its uncertainty.
- **Fitted parameters at a bound** (added in Part 4b; evaluated patients' out-of-fold fits, within
  1% on the log scale; source `reports/metrics/mech_identifiability.json` key `at_bound`,
  per-patient values `reports/metrics/mech_fitted_params.csv`; flag threshold 20%):

  | param | CGMacros k = 1 (lower / upper) | CGMacros k = 3 | Shanghai k = 3 |
  |---|---|---|---|
  | Gb | 2–4% / 0% | 0% / 0% | below 20% |
  | **SG** | **44–51% / 0% FLAG** | **40–44% / 0% FLAG** | **43% / 0% FLAG** |
  | SI | 0–2% / 0% | 0–2% / 0% | below 20% |
  | fc | 7–9% / 2% | 11% / 0% | below 20% |
  | kabs | 9% / 4% | 9% / 2% | below 20% |
  | **beta** | 0% / **24–27% FLAG** | 2% / 16% | not fitted (Shanghai has no macros) |

  Ranges cover the four twin variants (pop/static prior × activity on/off). These fractions were
  missing from the first Part 4 report; they motivated the Part 4b diagnosis (D-037).

## D-035 Simulator (Part 4)
- **Parameters:** a multivariate normal fitted to the log estimates of real patients
  (twin_pop_actoff_k3 fits, out-of-fold).
- **Schedules:** whole real days resampled, so meal timing, macros and activity stay consistent.
- **Noise:** white sd 1.3 + AR(1) (sd 7.2, φ = 0.82), estimated from real CGMacros readings minus a
  centred 1-h mean. This is a rough proxy, not a validated sensor-error model.
- **Labeling:** ids `synthetic_*`, `synthetic = True`, files in `data/synthetic/` (gitignored).
  Never mixed with real data or real test sets.
- **`phenotype: indian`** is a documented placeholder that raises NotImplementedError (Part 11).
- **Realism, per-patient quartiles** (real → synthetic, figure `mech_simulator_realism.png`):
  - Mean glucose median 130 → 135 mg/dL, IQR 120–152 → 116–160.
  - CV median 19 → 13%, IQR 16–23 → 9–18.
  - Time >180% median 5.0 → 1.4, but the synthetic upper tail is heavier (several patients
    > 60%).
  - Post-meal rise median 60 → 34 mg/dL.
  - Time to peak median 75 → 86 min, IQR 60–88 → 70–116.
- **Verdict:**
  - Within a patient, the simulator is too smooth: lower CV, smaller post-meal rises, later and
    more spread-out peaks.
  - Between patients, it is too spread out: the multivariate normal over poorly identified
    parameters (D-034) produces extreme combinations.
  - Acceptable for code tests and illustrating what-if mechanics; NOT a realistic synthetic cohort.
    Revisit in Part 11.

## D-036 Part 4 findings (mechanistic twin)
CGMacros: 45 patients, 29,321 identical test windows (anchors on day ≥ 3), 5 patient folds, paired
patient bootstrap 95% CIs. "ns" = CI includes 0. Metrics: `reports/metrics/mech_cgmacros.json`.

- **k = 3, overall MAE +30/+60/+120 (mg/dL):**

  | model | +30 | +60 | +120 |
  |---|---|---|---|
  | twin (pop prior, activity on) | 12.7 | 18.1 | 23.2 |
  | AR, first 3 days | 12.2 | 17.4 | 21.5 |
  | LightGBM (none of the patient's own days) | 11.4 | 16.6 | 20.8 |
  | persistence | 13.2 | 19.4 | 26.6 |

  - Twin − persistence: −0.5 ns / −1.3 / −3.4. The twin beats persistence at +60 and +120.
  - Twin − AR: +0.5 ns / +0.7 ns / +1.7. Not better than AR anywhere; worse at +120.
  - Twin − LightGBM: +1.3 / +1.5 / +2.5. Worse at every horizon, in every subset.
- **Subsets at k = 3:**
  - Hyper-140 onset: twin − LightGBM is +4.4 / +5.5 / +7.3.
  - Twin − AR at onset: −0.6 ns at +30, +4.0 at +120.
  - Post-meal 3 h: twin 17.2 / 22.1 / 24.4 vs LightGBM 15.5 / 20.6 / 21.9.
- **Adaptation days:**
  - k = 3 vs k = 0: −1.9 / −4.3 / −6.8 (helps).
  - k = 1 vs k = 0: +1.1 ns / +0.2 ns / −1.6 ns overall, and +2.1 at +30 post-meal (hurts). One day
    of data gives noisy parameters, consistent with D-034.
- **Static prior:**
  - At k = 0, the static ridge prior beats the population prior by −1.4 / −2.6 / −3.6. This is the
    only clear benefit of static data in Part 4, and it brings the twin up to persistence level.
  - At k ≥ 1 its effect is |Δ| ≤ 0.3 mg/dL.
- **Activity on vs off:** |Δ| ≤ 0.12 mg/dL in both the METs (34) and Intensity (11) subgroups at
  every k. Some differences are statistically non-zero but clinically zero. α_E was selected at the
  grid floor (D-032), so the data support no walk effect in this model.
- **Interval coverage, twin_pop_acton_k3:**
  - 80% intervals cover 89 / 88 / 88% (over-wide).
  - 95% intervals cover 96 / 95 / 95% (well calibrated).
  - 95% widths: 89 / 123 / 159 mg/dL, too wide for clinical use.
- **Shanghai T2DM** (97 patients, meal timing only, pooled 50 g dose, k = 3, no activity):
  - MAE: twin 11.4 / 18.6 / 27.3, persistence 11.4 / 19.3 / 29.8, AR 9.3 / 16.9 / 25.8,
    LightGBM 8.9 / 15.9 / 23.7.
  - The twin is worse than AR and LightGBM at every horizon; it beats persistence only at +120.
  - 80% / 95% intervals under-cover: 84 / 79 / 75% and 93 / 90 / 88%. Q/R were selected nested on
    Shanghai training patients (Q scale 16, R = 25 in all 5 folds; R is the lower grid edge). The
    NLL criterion did not produce calibrated longer-horizon intervals here; treat Shanghai
    intervals as not calibrated.
- **What-if (SIMULATED, median fitted patient):**
  - Peak 143 mg/dL for 30 g vs 160 mg/dL for 60 g.
  - Time above 140: 80 vs 145 min.
  - A 30-min walk changes the peak by < 0.1 mg/dL, as the figure states.
- **Leakage check:** the twin is worse than LightGBM and AR, so the results are not suspiciously
  good. AR's ridge lambda (50) is a fixed carry-over from Part 2 (D-024), not reselected per fold,
  so AR is not tuned more than the twin. Tests cover the guarantees:
  `test_fit_uses_only_first_k_days`, `test_ekf_forecast_causal`,
  `test_rest_level_is_past_only_with_default`, `test_simulation_causal_in_meals_and_activity`.
- **Implications for Part 5 (hybrid):**
  - The mechanistic twin alone is not a competitive forecaster.
  - Its usable contributions are a stable Gb, physiological structure for what-if scaffolding, and
    possibly its states or forecasts as LightGBM features or a residual base.
  - Per-patient kinetic parameters are not reliable (D-034).
  - Activity needs a different representation, e.g. learned in the ML part.

## D-037 Part 4b: SG / beta bound diagnosis and revised specification
**Diagnosis on TRAINING patients only** (each fold's 36 training patients, first 3 days only,
stage-1 fits; `src/mech/diagnose_bounds.py` → `reports/metrics/mech_bounds_diagnosis.json`).
No test window or evaluated patient was used to choose the revision. Ranges are over the 5 folds.
- **The SG bound is binding, not just touched.** 33–39% of training fits sit at SG = 0.001.
  - With the bound widened to 1e-5, the same share moves below 0.001 (median of those fits
    3.2–3.5e-4). None reach the new bound.
  - Data-only −2 log L improves by a median of 16–21 for those patients.
  - Fold-0 profile (SG fixed, others refit): 42% of patients are best at SG ≤ 0.001. Only 3% are
    flat between 1e-5 and 1e-3.
  - So the data push SG toward zero; the old bound was cutting fits off.
- **Not a segment-length effect:** 12-h instead of 6-h simulation segments leave the at-bound share
  unchanged (33–36%).
- **Interpretation:** SG trades off against the insulin-surrogate pathway: posterior corr(SG, SI)
  −0.49 (D-034), corr(log SG, log Gb) −0.30 to −0.42. Return to baseline is attributed to X rather
  than to glucose effectiveness. SG near zero is a model-structure trade-off, not a physiological
  estimate.
- **beta is not identifiable within a patient:**
  - Split-half (days 0–1.5 vs 1.5–3) correlation −0.20 to 0.20.
  - Widening its upper bound to 100: 6–14% go above 10, 0–3% reach 100.
  - corr(log kabs, log beta) 0.16–0.42.
  - The slowing ratio varies a lot within a patient (median CV 0.86–0.91), so the data have
    contrast, but it does not translate into a stable beta.
- **Candidate specs, split-half stability** (training patients, days < 3; mean over folds):

  | spec | Gb | SG | SI | fc | kabs | beta |
  |---|---|---|---|---|---|---|
  | original | 0.71 | 0.51 | 0.30 | 0.05 | −0.13 | 0.00 |
  | SG lo 1e-5 | 0.71 | 0.52 | 0.27 | 0.04 | −0.13 | 0.00 |
  | SG lo 1e-5 + beta fixed | 0.72 | 0.53 | 0.23 | 0.02 | −0.13 | fixed |
  | SG + beta fixed | 0.72 | fixed | 0.35 | −0.18 | −0.06 | fixed |

  None of the candidates clearly improves stability.
- **Choice (made before any test-window run):** widen SG's lower bound to 1e-5 and fix beta at the
  training-population mean (the stage-1 mean of each fold's training patients).
  - Rationale: remove a binding bound, and stop fitting a parameter with zero within-patient
    stability.
  - Fixing SG instead was rejected: it made fc unstable (−0.18).
  - Config `revision` in `configs/mech.yaml`. Outputs are tagged `_rev`; the original outputs are
    unchanged.
- **Result** (identical rows; `reports/metrics/mech_revision.json`; rev − original, * = CI excludes
  0):
  - **CGMacros k = 3:**
    - overall −0.16 / −0.22 / −0.13 (ns)
    - post-meal −0.22 / −0.28 / −0.34 (ns)
    - hyper-140 onset +0.07 / +0.18 / +0.43 (ns)
  - **CGMacros k = 1:** overall +0.27 / +0.01 / −0.14 (ns); onset +0.90* / +1.14* / +0.72 (worse).
  - **CGMacros k = 0:** overall −0.09* / −0.07 / +0.10*; onset +0.27* / +0.66* at +60/+120 (worse).
  - **Shanghai k = 3:** overall ≤ 0.03; onset +0.11* / +0.27* (worse).
  - **Coverage** (twin_pop_acton_k3): unchanged, 80%: 88–89%, 95%: 95–96%.
  - **Q/R choice:** R = 256 at k = 1 in all folds (originally 144–256); k = 3 unchanged.
- **At bounds, revised fits:**
  - SG: 0% at the new bound. However, 42–56% of CGMacros fits (47–50% on Shanghai) fall below the
    old bound 0.001; median SG is 0.0008–0.002.
  - fc 9–13% and kabs 9–13% at the lower bound; nothing above 20%.
- **Test-retest, revised** (days 0–3 vs 3–6, 45 patients; original in brackets):
  - Gb 0.67 (0.62)
  - SG 0.24 (0.22)
  - SI 0.26 (0.24)
  - fc 0.24 (0.19)
  - kabs 0.17 (0.18)

  Still only Gb is above 0.5.
- **Verdict: the revision does not help.**
  - Overall MAE changes are within ±0.3 mg/dL and not distinguishable from zero at k = 3.
  - Onset windows get slightly worse at k = 0/1 and on Shanghai.
  - Stability is essentially unchanged.
  - The bound artifact is gone, but the underlying identifiability problem (D-034) remains.
  - **The original specification stays the Part 4 default.** The revised outputs are kept for
    reference.
  - For Part 5, treat SG and beta as non-identifiable; do not expose them.

## D-038 Part 5: causal twin features and the train/test-consistent scheme
- **Twin spec:** the original Part 4 twin (D-037), α_E = 0 (activity effect ≤ 0.12 mg/dL, D-036).
  - The twin forecasts equal Part 4 `twin_pop_actoff_k<k>` (max |diff| < 2e-5 mg/dL; checked for
    every fold at k = 0/1/3 CGMacros and k = 3 Shanghai).
  - The activity input A(t) and E state are still computed and passed to the ML layer.
- **Feature kernel** `src/hybrid/twin_features.ekf_features`: the Part 4 EKF, unchanged, plus at
  each anchor t (after the update with the reading at t):
  - forecast and SD at +30/+60/+120 (deltas to the current CGM value);
  - filtered gut (Q1, Q2), insulin surrogate (U) and insulin action (X), E, and glucose
    appearance Ra;
  - filter residual, last innovation and its z-score, and its 1-h causal mean;
  - activity input and its 1-h causal mean;
  - the open-loop rollout of X, Ra and S_eff (Hybrid C only).

  Tests: equals `ekf_forecast` to 1e-9; perturbing CGM, meals, activity or slowing after t
  leaves every feature at t identical; changing data before t changes them.
- **Scheme (no parameter ever comes from a patient's own data after day k):**
  - Population prior = stage-1 (first 3 days) fits of TRAINING patients (identical to Part 4).
  - Test patients: MAP on their first k days (k = 0: prior mean), with a population-prior and a
    static-prior variant.
  - Training patients: MAP on their OWN first k days with a leave-one-out population prior, so
    their own stage-1 fit is excluded.
  - Hybrids are trained only on training-patient rows at day ≥ k.
  - Tested with a fake fitter: the leave-one-out prior equals the other patients' mean; the test
    prior equals the full training mean.
- **EKF (q, r):** the Part 4 choice where it exists; otherwise chosen on training patients'
  windows at day ≥ max(3, k) by Gaussian NLL (the Part 4 rule).
- **Grid:** k ∈ {0, 0.5, 1, 2, 3, 5, 7}, both cohorts (about 20 min), cached in
  `data/processed/hybrid/`.

## D-039 Part 5: Hybrids A0 / A / A_noact / B (CGMacros)
- **A0** = Part 3's best LightGBM content: CGM + time of day + meals + glycemic static, delta
  target, Part 3 protocol (inner patient split, leaf grid, early stopping).
  - A0 is trained on EXACTLY the same rows as A (training patients, day ≥ k), so A − A0 isolates
    the twin features.
  - The Part 4 population LightGBM (all training rows, no static) is kept as `lgbm`.
- **A** = A0 + the 18 twin features. **A_noact** = A without the activity features (tw_E, tw_act,
  tw_act_mean_1h), which separates the activity effect (owner choice 3).
- **B** = LightGBM on (truth − twin forecast) with A's features, added back to the twin forecast.
- **Test windows:** the Part 4 windows (day ≥ 3), 45 patients, 29,321 rows at every k; identical
  rows for every model within a k.

## D-040 Part 5: choosing the "best hybrid" for personalization
- The choice is made per (fold, k, with/without static) between A and B, by the mean inner-validation
  MAE on training patients. It is never based on the comparison table or on test patients.

## D-041 Part 5: Hybrid C (physics-informed GRU)
- **Network:** the Part 3 GRU + FiLM. FiLM input = glycemic static plus the twin state at t (k = 3).
  Output = the 5-min trajectory out to +120 min (24 steps).
- **Loss:**
  - L1 on real CGM readings.
  - Plus λ · mean |r|, the trapezoid residual of dG/dt = −S(G − Gb) − X·G + Ra.
  - S, X and Ra come from the twin's open-loop rollout from t, so the physics term uses only
    information available at t.
  - Test: the residual is about 0 on the twin's own trajectory and large on a shifted one.
- **Selection and runs:**
  - λ ∈ {0, 0.1, 1, 10}, chosen per fold on inner-validation patients with seed 0.
  - Then 5 seeds.
  - λ = 0 is also run, as a same-network, no-physics ablation.
  - Hidden size = Part 3 E_full choice (64).


## D-042 Part 5: neural onset-weighted loss
- **Weight:** w on (window, horizon) pairs with cgm_now < 140 and y(t + h) > 140 mg/dL, else 1.
  - It uses the training target only, as a loss weight; it is never an input.
  - Early stopping uses the weighted validation L1.
- **Choice of w:** w ∈ {3, 10}, chosen once per (cohort, config) on holdout-train patients
  (inner split, seed 0) by the mean of overall and onset validation MAE (mg/dL).
- **Kept identical to Part 3:** channels, hidden size, folds (repeat 0), inner split and seeds.
- **Code:** `NeuralForecaster.fit(w_tr, w_va)`. With None it runs the identical unweighted Part 3
  code path.

## D-043 Part 5: personalization protocol
- **k grid:** {0, 0.5, 1, 2, 3, 5, 7} days of the FIRST period.
- **Test region:** ONE common region for every k (first period, day ≥ 7), so every method and k is
  paired on identical rows.
- **Exclusion:** fewer than 96 evaluable anchors (1 day) at day ≥ 7.
  - Shanghai later periods are excluded from adaptation and test (owner choice 2).
- **Adaptation pairs:** target time t + h < day k (tested never to overlap the test region).
  - Below 24 pairs, methods 2–5 and 7 use the population model.
- **Bases:**
  - `lgbm`: A0 content, fitted on all training-patient rows.
  - `hybrid`: best of A/B (D-040), refitted per k with twin features from each patient's first
    k days.
  - Each base is run with and without static features.
- **Selection:** every adaptation hyperparameter is chosen on TRAINING patients.
  - Their forecasts are out-of-fold: inner 3-fold patient refits with the outer model's leaves and
    trees.
  - Scored on their day-≥ 7 region.
  - Tested: an inner model never predicts a patient it was fitted on.
- **Methods:**
  1. none;
  2. per-patient ridge from scratch (CGM / time / meal inputs);
  3. ridge on population residuals;
  4. shrinkage blend, w = n / (n + n0);
  5. LightGBM fine-tuning (`init_model` on the trimmed population booster; 7 leaves; lr × trees
     grid);
  6. Part 4 twin, population vs static prior;
  7. RLS on residuals with forgetting, a Page-Hinkley drift detector (boosts P) and a
     trace(P) ≤ d·p0 cap. Reported frozen at day k and continuous (a pair is used only once
     t + h ≤ the anchor; tested).
- **Grid changes made after smoke runs, before any full run (disclosure):**
  - The smoke runs (fold 0, k = 1/5) printed MAE on fold 0's TEST patients.
  - The ridge λ and n0 grids were widened because the TRAINING-patient selection picked the grid
    edge; this is legitimate.
  - The RLS change was triggered by the test-patient MAE: 29–116 vs 15 mg/dL for no adaptation,
    an obvious bug (covariance wind-up / no regularization). It was diagnosed on one fold-0 test
    patient's persistence residuals.
  - The fix (p0 grid {0.001, 0.01, 0.1} plus the trace cap) is a design correction. Within every
    fold, p0 / forgetting / PH λ are still chosen on training patients only.
  - Still, fold 0's test patients informed the RLS grid design. The RLS rows should be read with
    that caveat; fold-0 test patients are 8–15 of the 44 / 71 evaluated.
  - The crossover's "best method" excludes continuous RLS. That rule was added after seeing that
    continuous RLS uses post-k data, which makes it a reporting fix, not a model choice.
  - **Sensitivity (Part 6 housekeeping):** `python -m src.personalize.report excl_fold0` re-scores
    the saved predictions without fold 0's test patients (CGMacros 36, Shanghai 56 patients) ->
    `reports/metrics/personalize_excl_fold0.json`. +60 min, with static, vs none:
    - CGMacros: rls_cont −0.84 [−1.20, −0.54] (all patients −1.13); rls k = 3 −0.65
      [−1.14, −0.20] (−0.88); finetune k = 3 −0.69 [−1.14, −0.29] (−0.89).
    - Shanghai: rls_cont −0.76 [−1.06, −0.47] (−0.88); rls k = 3 −0.40 [−0.53, −0.26] (−0.43).
    - The conclusions hold; effects shrink by about 0.1–0.3 mg/dL. Fine-tuning shrinks by a
      similar amount, so the shrinkage reflects which patients remain rather than the RLS
      redesign alone.
- **Crossover k\*:** first k (linear interpolation) at which the best no-static adapted method
  reaches the with-static population model's k = 0 MAE.
  - The method is picked by mean MAE over k > 0 (a reporting choice).
  - Patient bootstrap (1,000 resamples) with the share of resamples with no crossover by k = 7.

## D-044 Part 5 findings (hybrid twin, onset fix, personalization)
- **Leakage and causality checks:**
  - Hybrid training rows start at day ≥ k in every fold (recorded in `hybrid_cgmacros.json`
    checks).
  - Twin features are causal (future-perturbation test) and reproduce Part 4.
  - The test windows are identical to Part 4: 45 patients, 29,321 rows per k.
  - No result was suspiciously good. The largest gain (onset-weighted neural, −4 to −9 mg/dL on
    onsets) comes with a matching overall loss, which is the expected trade-off of loss
    reweighting, not leakage.
- **Hybrid, CGMacros** (paired A − A0, mg/dL, +30/+60/+120; * = CI excludes 0):
  - k = 0: −0.21* / −0.10 / −0.26.
  - k = 1: about 0.
  - k = 3: −0.18 / −0.55* / −0.85 (post-meal −0.41* / −0.75*).
  - Without 007: same pattern.
  - The twin's value is small and needs about 3 days of patient data. It does not help onset
    windows at any k.
  - A − A_noact is ≤ 0.2 mg/dL (activity is negligible, consistent with D-036).
- **Hybrid B and C:**
  - B is never better than A0 (k = 1: +0.9* / +1.7* worse at +60/+120).
  - C (k = 3): λ chosen 0.1 in 2 folds and 0 in 3 folds.
  - C − A0 overall is ns. Onset is +2.1* / +3.0* / +3.3* (worse). C − C_lam0 is ≤ 0.1 overall
    (ns).
  - Seed SD is 0.04–0.26 mg/dL. **The physics-informed loss does not help.**
- **Neural onset fix, CGMacros** (w = 10 chosen for both configs):
  - nnw − nn onset: B −9.1* / −9.0* / −4.1*; E −5.5* / −6.1* / −3.4*.
  - nnw − nn overall: B +3.1* / +4.9* / +4.8*; E +1.5* / +2.8* / +2.2*.
  - The weakness at rising glucose can be moved, but only by trading overall accuracy. It is not
    fixed for free.
  - nnw_E beats LightGBM E on onsets (−3.1* / −3.3*) but loses overall (+2.2* / +3.6*).
- **Neural onset fix, Shanghai** (w = 10 for both configs):
  - nnw − nn onset: B −5.7* / −11.7* / −10.0*; E −5.3* / −9.1* / −6.9*.
  - nnw − nn overall: B +0.8* / +3.2* / +4.5*; E +1.4* / +3.9* / +4.3*.
  - The same trade-off as CGMacros.
- **Personalization (+60 min; CGMacros 44 / Shanghai 71 patients; day ≥ 7 region):**
  - **Static head start (ML), k = 0:** −0.13 [−0.86, 0.48] CGMacros and −0.23 [−0.50, 0.04]
    Shanghai. These are not positive, so the crossover is trivially k\* = 0 and uninformative.
  - **Twin static prior, k = 0:** CGMacros +2.17 [−0.06, 5.12], crossover k\* = 1.93 d
    [0, 3.12]; Shanghai +0.19 (ns).
  - **Best k-day adapter: LightGBM fine-tuning.**
    - CGMacros with static: −0.25* / −0.89* / −1.28* at k = 1/3/5.
    - Shanghai: −0.18 / −0.36* / −0.82*.
  - Shrinkage and frozen RLS give −0.2 to −0.9. Continuous RLS gives −0.7* to −1.1*.
  - Per-patient ridge from scratch is worse at every k (+0.5 to +3.8). Residual ridge is worse at
    k = 1 and neutral by k = 5.
  - **Twin in personalization:** hybrid − lgbm with the same method is within ±0.5 (ns) on
    CGMacros. Shanghai: hybrid none is −0.3 to −0.5* better; after fine-tuning it is ns.
- **Thesis check:**
  - "Clinical profile gives a head start" is not supported for the ML twin in either cohort. It
    holds only for the mechanistic twin's prior, which is itself not competitive.
  - "The wearable stream keeps it current" is supported for the CGM stream, by about 1 mg/dL over
    3–5 days of adaptation.
  - Activity / HR adds nothing measurable (D-036, A − A_noact).



