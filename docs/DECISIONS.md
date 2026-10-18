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




## D-045 Part 6 protocol: uncertainty, two designs declared BEFORE running (owner choice C)
- **Models.**
  - M_pop = population LightGBM A0 (CGM + time of day + meals + glycemic static, delta target;
    Part 5 `lgbm_s_none`).
  - M_ft = M_pop fine-tuned on the patient's first k = 3 days (Part 5 `lgbm_s_finetune`; lr and
    trees chosen per fold on training patients' out-of-fold rows).
  - Evaluation rows are the Part 5 rows: first period, day >= 7, eligible patients, real target.
    Horizons +30 / +60 / +120 min; targets 80 % and 95 %.
- **Design A (HEADLINE):** CV repeat 0, the same 5 folds as Part 5. Two arms per fold:
  - *Split arm* (split conformal, CQR, ACI). "Calibration patients" = the fold's training
    patients that are in the fixed holdout calibration set (configs/splits.yaml). They are
    REMOVED from model fitting, early stopping, fine-tune selection and every tuning choice.
    They only supply conformity scores.
  - *CV+ arm*. Model on all fold-training patients (the Part 5 base). Out-of-fold residuals come
    from the Part 5 inner 3-fold patient split; test intervals use the 3 inner models (CV+,
    Barber et al. 2021, Ann. Stat. 49:486, Sec. 3). Guarantee >= 1 - 2 alpha under
    exchangeability only.
  - The CV+ arm has no calibration set by design: the holdout-calibration patients are ordinary
    training patients there, exactly as in the Part 5 base. "Calibration patients are never used
    in training or tuning" applies to the split arm and is unit-tested there.
  - In design B, the CV+ arm trains on holdout train + calibration patients.
  - CV+ residuals = training patients' out-of-fold rows in the same region as test (first period,
    day >= 7, real target), for both M_pop and M_ft (fine-tuned out-of-fold).
  - M_ft CV+ uses each test patient's fine-tune of each of the 3 inner models.
  - Split-arm calibration scores = the calibration patients' region rows.
  - The ACI run starts at day 3 (end of adaptation, causal warm-up) and is scored on day >= 7.
  - Inner fold seed: 42 + fold (Part 5); 42 in design B.
  - ACI gamma grid widened to include 0.1 and 0.2, after a design-B CGMacros smoke run picked the
    old upper edge (0.05) in 11 of 12 cells. That choice was made on proper-training patients'
    objective only; no test metric was computed.
  - Robustness uses design A only (the B test set is too small to stratify faults).
  - For robustness, a missing current reading is replaced by the last real reading if it is
    <= 60 min old (stale forecast); otherwise the system gives no forecast. Availability is
    reported.
- **Design B (CONFIRMATORY):** holdout split, train 27 / cal 9 / test 9 (Shanghai 60 / 21 / 19),
  eligible patients evaluated. Same code, with the holdout as a single fold.
  - B is never used to choose a method, level or threshold. It is reported next to A.
  - Where A and B disagree, A stays the headline and the disagreement is reported.
- **Split conformal:** score |y - yhat|; quantile at rank ceil((n + 1)(1 - alpha)) of the
  calibration rows.
- **CQR** (Romano et al. 2019):
  - LightGBM quantile models (delta target, A0 features, the point model's leaves, early stopping
    on the inner-validation patients) at alpha/2 and 1 - alpha/2. Score
    max(lo - y, y - hi).
  - Personalized CQR shifts both quantiles by the fine-tune correction (M_ft - M_pop) and is
    conformalized on the calibration patients' personalized scores.
- **ACI** (Gibbs & Candès 2021): alpha_{t+1} = alpha_t + gamma (alpha - err_t), checked against
  the authors' code (github.com/isgibbs/AdaptiveConformal, ACCode.R).
  - Runs per test patient, in time order, on split-conformal scores.
  - err_t enters only after y(t + h) is observed (delayed feedback, causal).
  - alpha_t <= 0 gives an infinite interval (counted as covered; reported separately). alpha_t >= 1
    gives a zero-width interval.
  - gamma is chosen from the config grid on TRAINING patients only (out-of-fold scores, own
    patient left out), minimizing mean |per-patient coverage - target|.
- **Exchangeability caveat** (stated with every table):
  - Rows within a patient are autocorrelated, so the row-level guarantees are nominal.
  - Coverage is reported overall, per group and per patient, with 95 % patient-bootstrap CIs
    (1,000 draws).
  - Cells with < 5 patients are marked "unreliable". Every CGMacros group in B falls in this
    category.
- **Event probabilities** (CV+ arm):
  - Raw score = P(yhat + R > thr), with R the training patients' out-of-fold residuals.
  - Platt and isotonic maps are fitted on training out-of-fold rows only.
  - Events, onset-only: hyper180_any_h and hyper140_any_h (glucose at t <= thr); hypo70_any_h
    (glucose at t >= 70) on Shanghai. CGMacros hypo is computed but labeled "limited evidence".
  - Metrics: Brier, ECE (10 equal-width bins), reliability curve, AUROC.
- **Selective prediction:**
  - Signals: CQR 80 % width, the data-quality score (D-047), and their mean rank.
  - Risk = MAE on retained rows. Report the risk-coverage curve and AURC.
  - No abstention threshold is tuned on test.
- **Development rule:** print only training-patient numbers. Test windows are scored once at the
  end.


## D-046 Part 6 alert definitions and false-alarm counting (written before running)
- **Episodes (ground truth)** follow the consensus lv1 definitions (Battelino et al. 2023, Lancet
  Diabetes Endocrinol 11:42, as implemented in the cgmguru R package):
  - Hyper-180 starts at >= 15 consecutive minutes of real readings > 180 and ends after >= 15
    consecutive minutes <= 180.
  - Hyper-140 uses the same rule at 140.
  - Hypo-70 starts at >= 15 minutes < 70 and ends after >= 15 minutes >= 70.
  - "15 consecutive minutes" = a run of real readings with no missing grid point: 4 readings at
    5 min (CGMacros), 2 at 15 min (Shanghai). A missing reading breaks the run.
  - Episodes are detected in the evaluation region (first period, day >= 7).
- **Alert systems**, evaluated at every anchor (15-min stride) while the patient is NOT in an
  episode of that type ("at-risk" anchors):
  1. *Model*: P(episode onset within 60 min) >= theta. The raw score is the CV+-arm M_pop
     predictive probability P(yhat_60 + R beyond threshold). It is Platt-mapped on TRAINING
     patients' at-risk anchors (out-of-fold forecasts, all first-period days), with the alert
     truth label. The naive and trend scores get the same kind of Platt map (used for DCA).
  2. *Naive*: glucose(t) >= theta (hypo: <= theta).
  3. *Linear trend*: glucose(t) + 60 * slope_30(t) >= theta (hypo: <=).
- **Counting rules:**
  - After an alert, the same system is silenced for 60 min (refractory). Silenced alerts are not
    counted.
  - An alert at t is TRUE if an episode starts in (t, t + 60]; otherwise it is a FALSE alarm.
  - An episode is DETECTED if a counted alert falls in [onset - 60, onset). Lead time = onset -
    first such alert.
  - Episodes with no at-risk anchor in the 60 min before onset are excluded and counted
    separately.
  - Monitoring time = at-risk anchors x 15 min. FA/day = false alarms / monitoring days.
  - Sensitivity at 1 / 2 / 3 FA/day is reported two ways:
    - (i) as a curve point on test (FROC-style, no tuning);
    - (ii) with theta chosen on TRAINING out-of-fold rows, then applied to test (the realized
      FA/day is reported).
  - PPV = true alerts / counted alerts.
- **Decision curve analysis** (Vickers & Elkin 2006):
  - At-risk anchors; label = event within 60 min.
  - Net benefit = TP/N - FP/N * pt / (1 - pt), for pt in 0.05-0.50.
  - Compared: model, treat-all, treat-none, and the naive and trend scores (each mapped to a
    probability by a Platt fit on training patients).

## D-047 Part 6 robustness and data-quality score (written before running)
- **Fault injection:**
  - Faults go into test patients' grids at t >= tau = day 7. Features are recomputed with the
    Part 2 causal code. Targets remain the clean real readings.
  - Unit-tested: anchors before tau are unchanged.
- **Severity ladders** (configs/uncertainty.yaml):
  - gaps, dropout bursts, bias, drift, noise;
  - compression-lag night dips;
  - missing wearables: CGMacros only (Shanghai has no wearable stream, so n/a);
  - timestamp delay: lag only. An "early" shift would leak future values and is not simulated.
- **Data-quality score** at t, in [0, 1], = mean of four sub-scores:
  - completeness: real fraction in (t - 6 h, t];
  - recency: 1 - minutes since the last real reading / 60, clipped;
  - plausibility: 1 - fraction of the last 60 min of readings at or beyond the sensor range
    bounds;
  - rate-of-change: 1 if max |dG/dt| over 60 min is <= 4 mg/dL/min, falling linearly to 0 at 8.
    These limits are a design assumption with no verified source; they are listed in the config.
  - Wearables are excluded, because A0 uses no wearable input.
- **Test that a low score predicts high error:** Spearman correlation between the score and
  |error|, plus MAE by score bin, on pooled clean + faulted test rows, with patient-bootstrap
  CIs.

## D-048 Part 6 item 2: patient-level static + dynamic fusion (written before running)
- **Unit:** one row per patient (first period).
- **Targets:** next-day (hours 24-48) time above 180 and time in range 70-180, in % of real
  readings. A day needs >= 70 % of its grid points real.
- **Input sets:**
  - (i) static only: HbA1c, fasting glucose, age, BMI, sex (+ diabetes duration on Shanghai);
  - (ii) first-day (hours 0-24) CGM summary: mean, SD, CV, TIR, TAR180, TBR70, max; plus meal
    count and carbs (CGMacros);
  - (iii) both.
- **Model:** ridge on standardized inputs. Alpha is chosen by inner leave-one-out on training
  patients only; missing values get training medians.
- **Validation:** patient-grouped 5-fold CV x 5 repeats, predictions averaged over repeats per
  patient.
- **Report:** MAE, plus paired patient-bootstrap CIs for (iii) - (i) and (iii) - (ii). If fusion
  does not help, this is said plainly.


## D-049 Part 6: robustness bug, second scoring of test windows, ACI infinite intervals
- **Bug:** `refresh_derived` (src/robustness/faults.py) rebuilt `minutes_since_real` from the
  grid index. Part 1 stores it relative to the raw reading time, so the "clean" fault rebuild
  differed from the stored features.
  - Found after the first `report --final` by a consistency check (clean rebuild vs stored
    features), NOT by looking at test metrics.
  - Every robustness number from the first run was invalid.
  - Fix: `orig` parameter; new test `test_zero_severity_fault_is_identity`.
  - New guard in `run_job`: the clean rebuild must match the population +60 forecast within
    1e-6, otherwise `RuntimeError`. It is saved as `clean_rebuild_max_abs_diff_60`; the value is 0.0
    in all 10 design-A folds.
- **Second scoring:** design A was re-run and `report --final` was run a second time.
  - The interval / event / alarm outputs of the re-run are byte-for-byte identical to the
    first run (max |diff| = 0 on every test parquet). Only the robustness numbers changed.
  - No method, threshold or hyperparameter was changed between the two scorings.
- **ACI infinite intervals (reported, not fixed):** with the gamma chosen on training patients
  (0.05–0.2), alpha_t often drops below 0, so the interval is infinite (D-045 counts it as
  covered).
  - On test, 7–54 % (CGMacros) and 11–64 % (Shanghai) of ACI rows are infinite.
  - ACI's near-exact coverage is therefore mostly produced by uninformative intervals, and its
    "median width" is over finite rows only.
  - Cause: the gamma objective (|per-patient coverage − target|) ignores width and the number
    of infinite intervals. This is a flaw in the pre-registered design. Any change (cap alpha_t,
    penalize infinite intervals) would be chosen after seeing test results, so it is left to the
    owner. RESULTS now prints the infinite fraction in each cell.


## D-050 Part 7 what-if engine: models, edit semantics, bands (written before any Part 7 code)
- **Scope:** CGMacros only. Shanghai is excluded from meal what-ifs: its A0 model has no meal
  inputs and its meal log has no macros (D-018). Stated in the report, not dropped silently.
- **Models (per fold, CV repeat 0, Part 5/6 folds):**
  - `unc`: unconstrained A0 LightGBM (CGM + time of day + meals + glycemic static, delta target).
    Refit exactly like the Part 6 CV+ arm (`um.fit_population` on all fold-training patients,
    inner split seed 42, 2 threads; inner 3-fold models with seed 42 + fold).
  - `mono`: identical features and protocol, plus LightGBM `monotone_constraints` (owner choice
    Q2 = A): +1 on `meal_carbs_g_{1,3,6}h` and `meal_kcal_{1,3,6}h`; -1 on `act_post_meal`;
    0 elsewhere (protein, fat, fiber free). Constraint method = LightGBM default ("basic").
  - `twin`: original Part 4 specification as used in Part 5 (alpha_E = 0 fits, D-038):
    parameters from the patient's first 3 days (`twinfeatfits_*`, k = 3, prior "pop"; training
    patients use the leave-one-out population prior), EKF (q, r) of that fold at k = 3.
    `twin_learned` walk effect = Part 4 `alpha_on` of that fold (chosen on training patients).
- **Edits:** change carbs / protein / fat / fiber of a logged meal; remove a meal; move a meal;
  add a post-meal walk.
  - kcal is updated by the macro change: kcal' = max(0, kcal + 4 dCarbs + 4 dProtein + 9 dFat);
    a fiber change does not change kcal (assumption).
  - ML inputs are recomputed with the unchanged Part 2 code (`src.features.build.period_features`)
    on the grid slice (t - 8 h, t] and the period's (edited) meal log. Guard: the zero edit must
    reproduce the stored A0 feature row (max |diff| <= 1e-9, NaN-equal), else `RuntimeError`.
  - A meal moved past t is unknown at t (= removed for the ML models). The twin can play
    meals and walks AFTER t forward (open-loop rollout); the ML models cannot (their inputs are
    causal). A walk therefore enters the ML models only at forecast times after the walk.
  - Walk: grid `activity_intensity` = max(existing, 2.0) for [meal + 15, meal + 45) min (Part 4
    `whatif.walk` values, moved to `configs/whatif.yaml` unchanged). HR is not edited. For the
    twin the activity input is recomputed with the Part 4 rule (`activity_input`, rest level at t).
- **Bands** (labelled "observational band, no counterfactual coverage guarantee"):
  - Test patients (final run): CV+ (Part 6 method) from the form's 3 inner-fold models and the
    out-of-fold absolute residuals of training patients (region day >= 7, first period).
  - Training patients (development): scored by the inner-fold model that did not see them; band
    = split-conformal half-width from OOF residuals of training patients NOT in that inner fold.
  - Twin: Gaussian EKF predictive variance (mean +- z sd), labelled "filter variance, not
    conformal".
- **Amendment (engine bug found in development, no test patient involved):** a smoke run of the
  suite on two fold-0 TRAINING periods showed that recomputing on the slice (t - 8 h, t] differs
  from the stored rows by about 2e-16 (pandas rolling sums depend on where the series starts),
  which flipped LightGBM splits in 5 % of `unc` identity and causality checks. The recompute now
  uses the period's grid prefix up to t (exactly as Part 2 built the table); the zero-edit guard
  is bitwise (0 difference on 286 sampled anchors). No pass rule or range was changed.
- **Guards:** the `unc` refit must reproduce Part 6 OOF population forecasts
  (`unc_A_cgmacros_fold<f>_oof.parquet`, `oof_pop|h`) within 1e-6 (training patients only); the
  zero-edit twin forecast must equal `ekf_forecast` within 1e-9.


## D-051 Part 7 plausibility suite: tests and pass rules (accepted by the owner as written)
- **Windows:** anchors (Part 2 table) with day >= 3, a logged meal with macros and recorded amount
  in the 15 min before t (meal windows); walk windows below. At most 20 windows per patient
  (random, seed 42; config).
- **Peak:** ML = max of the +30 / +60 / +120 forecasts; twin = max of the 0..120-min trajectory.
- **Tests and pass rules:**
  - Carb monotonicity: carbs of the last meal set to 0/15/30/45/60/90/120 g; the peak never
    decreases. Tolerance 1e-9 (primary); 2 mg/dL (secondary). For `mono` this is true by
    construction (both carbs and kcal are +1 constrained) and is labelled that way.
  - Activity direction: a walk never raises the peak or the 2-h AUC (trapezoid over 0/30/60/120 for
    ML with 0 = cgm_now; full trajectory for the twin). For `mono` true by construction (owner
    Q2 = A); the `unc` pass rate is reported as "what the data learned". ML walk windows = anchors
    45-120 min after the last meal (walk completed, within the 120-min post-meal feature window);
    twin walk windows = the meal windows (walk in the future).
  - Identity: zero edit gives exactly the same forecast.
  - Causality: an edit at t_e changes no forecast before t_e (ML: anchors in the slice with
    t < t_e; twin: filter states before t_e).
  - Physiological bounds: unclipped forecasts within 40-400 mg/dL; |forecast(+30) - cgm_now| / 30 <=
    4 mg/dL/min (same unsourced design assumption as D-047).
  - Dose-response size: per window, (peak(60 g) - peak(30 g)) / 30 in [0, 2.0] mg/dL/g; median over
    windows in [0.3, 1.3]. Source: Franc et al. 2010 (Diabetes Care 33:1913, PMC2928333): breakfast
    rise = 65 x carbs / body weight + 73 mg/dL (R^2 0.20; insulin-treated T2D), i.e. 0.65-1.3
    mg/dL/g for 50-100 kg; lunch slope about half. Weak bounds.
  - Cross-model consistency: same sign of the 60 g vs 30 g peak effect for ML and twin, and
    |ML effect - twin effect| <= max(10 mg/dL, 50 % of the larger) (design assumption). Spearman
    correlation of effects also reported.
- **Report:** pass rate per test x {unc, mono, twin_learned, twin_assumed (walk tests only)} with
  95 % patient-bootstrap CIs (1000 resamples). Protein / fat / fiber edits are reported
  descriptively (median effect), with no pass rule.



## D-052 Part 7 walk effect: learned vs assumed (accepted by the owner as written)
- **Learned:** ML response to the edited `act_post_meal` and the twin with `alpha_on` (Part 4,
  bottom of its grid, effectively 0). Reported as they come out.
- **Assumed (twin only, `twin_assumed`):** alpha_E calibrated so that a 30-min walk (intensity
  2.0) starting 15 min after a 60 g meal lowers the 3-h incremental post-meal AUC of the
  "median patient" (training-population mean parameters of the fold, starting at basal) by 20 %;
  sensitivity 10 % and 35 %. Grounding: Iida et al. 2020 (PMC7538501; 15-min walks after each
  meal, 24-h iAUC about 33 % lower, n = 11 T2D) and Engeroff et al. 2023 (PMC10036272; SMD 0.55,
  direction only). Assumed output is always labelled "assumed, not learned from data", kept in
  separate JSON keys and figure panels and never mixed with learned results.


## D-053 Part 7 matched-meal validation (written before running)
- **Primary, within-patient:** CGMacros standardized breakfasts (dataset page: "protein shakes with
  varying amounts of carbohydrates, protein, fat, and fiber"). Pairs of the same patient: both
  breakfasts; protein / fat / fiber within 5 / 5 / 3 g; carbs differ by >= 20 g; clock time within
  2 h; pre-meal glucose (cgm_now) within 20 mg/dL; amount 90-110 % and recorded; no other meal from
  3 h before to 2 h after; real CGM at the anchor and at the forecast points.
- **Secondary, cross-patient:** same diabetes status, HbA1c within 0.5, same meal type, clock time
  within 1 h, cgm_now within 15, protein / fat within 5 g, carbs differ >= 20 g (more confounded).
- **Anchor** = first Part 2 anchor at or after the meal, at most 15 min after it. Pairs are ordered so
  B has more carbs.
- **Predicted difference** = mean over the two contexts of f(carbs_B) - f(carbs_A) (edit-based).
  **Observed** = (y_B(+h) - cgm_B) - (y_A(+h) - cgm_A). Primary +60 min; +30, +120 and peak secondary.
- **Metrics:** Pearson, Spearman, sign agreement, calibration slope + intercept (OLS observed on
  predicted), mean predicted vs observed; 95 % CIs from 1000 patient-bootstrap resamples.
- **Twin:** only pairs with both meals at day >= 3.
- **Implementation details fixed before running:** the matched "peak" is max over +30 / +60 / +120
  for every model (comparable with the observed readings; D-051's trajectory peak is for the
  plausibility suite only). Cross-patient pairs: at most one pair per (patient A, patient B), the
  closest in clock time; bootstrap clusters = the lower-carb patient of each pair. The amount /
  clean-window / real-CGM rules of the primary design also apply to the secondary design.
- **Development:** training patients scored by inner-fold models. **Final:** every patient scored
  by the fold model that never saw it.
- **Limits written in advance:** meal order not shown to be randomized; day-to-day variation,
  previous dinner, sleep and medication not controlled; two carb levels give a two-point slope;
  liquid shakes (may not transfer to mixed / Indian meals); the model was trained on other
  patients' identical shakes, so this tests "a new patient, a familiar meal design".
- **Final run once:** `python -m src.whatif.run final` writes `data/processed/whatif/FINAL_DONE`;
  a second run requires `--force`, is recorded in the outputs and is disclosed to the owner.
  Test-patient metrics are never computed or printed by the development stage.


## D-054 Part 7 India meal library (owner choice Q1 = A)
- About 15 presets in `src/india/meals.json`: our own ASSUMED raw-ingredient recipes (grams), the
  IFCT 2017 codes used (Longvah et al., NIN / ICMR) and per-portion carbs / protein / fat / fiber
  / kcal. No IFCT table and no per-100 g rows are stored in the repo.
- `python -m src.india.library build|check` fetches the IFCT 2017 CSV at run time (ifct2017
  GitHub package, AGPL-3.0 since April 2025; not redistributed, cached only under the gitignored
  `data/external/`) and recomputes / cross-checks the per-portion numbers. Ingredients not
  traceable to IFCT are marked "approximate assumption".
- A top-level disclaimer says the values are assumptions, not clinical values. Model outputs on
  these meals carry an out-of-distribution note (the model was trained on US meals).



## D-055 Part 7 findings and disclosures (written AFTER the single final run)
- **Runs:** dev stage (fold-0 training patients, 236 s); final stage once (514 s, 5 folds in
  parallel; lock `runs: 1, forced_reruns: 0`). Guards: `unc` refit = Part 6 OOF forecasts
  (max |diff| 0.0 in all 5 folds); zero-edit features bitwise identical; zero-edit twin = Part 4
  `ekf_forecast` (1e-13).
- **Engine bug before results (D-050 amendment):** slice-based recompute differed by ~2e-16 and
  flipped trees; fixed to the full period prefix during development, no rule changed.
- **Pre-registered tests that FAILED (reported, not tuned away):**
  - Median dose response of both ML forms is below the D-051 range (final: unc 0.22, mono 0.20
    mg/dL per g vs 0.3-1.3); the twin passes (0.33). The matched breakfasts agree: the ML forms
    predict about half of the observed within-patient peak carb effect.
  - `unc` carb monotonicity holds in only about a quarter of windows (75 % within 2 mg/dL).
  - Activity direction is about 68 % for `unc` (learned) and both twin variants. For the twin
    this is structural: the activity term (SG + aE E) pulls G towards Gb from either side, so a
    walk RAISES predicted glucose when G is below basal (dev check on 120 training windows:
    all 23 twin_assumed failures started below Gb, none of the 57 windows above Gb failed).
    Median peak rise among failures (final): twin_learned 0.03, twin_assumed 1.4, unc 0.34 mg/dL.
- **Monotone constraint cost:** test MAE mono - unc is +0.04 / +0.16 / -0.16 mg/dL at
  +30 / +60 / +120 (CIs include 0).
- **Owner decision needed (not changed here):** whether to use `mono` as the what-if model,
  how to present the ML under-estimation of carb effects, and whether the twin's activity term
  should be one-sided (a model change that would need a new pre-registration).
- **Re-running `make whatif`** stops at the final stage (lock) unless `--force` is passed; any
  forced run is recorded in the lock and must be disclosed.


## D-056 POST-HOC amendment: one-sided twin activity term; owner display decisions (written before running)
- **Status:** POST-HOC (decided after seeing the Part 7 test results, D-055). NOT pre-registered.
  The original Part 7 results stay unchanged in the report; every D-056 number is labelled
  "post-hoc, not pre-registered".
- **Owner decisions (Part 7 review):** (1) the monotone model (`mono`) is the default what-if
  model; (2) magnitudes are NOT rescaled: the app shows direction and ranking plus the fixed note
  "Model-based estimate. In matched real breakfasts the models predicted about half the observed
  rise; treat sizes as lower bounds.", with the twin's magnitude shown alongside, labelled;
  (3) the twin's activity term is fixed as below.
- **Kernel change (src/mech/model.py, opt-in flag `act_onesided`, default 0 = original):**
  original G' = -(SG + aE E)(G - Gb) - X G + Ra; D-056 G' = -SG (G - Gb) - aE E max(G - Gb, 0)
  - X G + Ra. E >= 0, so activity only adds glucose uptake and has no effect at or below basal.
  With the flag 0 (all Parts 4-6 configs) the kernel is unchanged.
- **Alphas:** `twin_learned_d056` uses the Part 4 `alpha_on` of the fold (same choice as the
  original); `twin_assumed_d056` is re-calibrated with the D-056 kernel to the same D-052 target
  (20 % 3-h iAUC reduction; 10 / 35 % sensitivity) on the fold's training-population means.
- **Checks, in this order:**
  1. Training windows only: fold-0 training patients, the same sampled meal windows as the Part 7
     dev stage, twin walk tests only (activity direction + walk AUC effect).
  2. Test windows: the final stage is locked, so ONLY the twin walk tests are re-run on the same
     test windows with the lock override (`--force`). This is **forced re-run 1** of the Part 7
     final stage and is recorded in the lock file. No ML or carb test is re-run.
  3. Parts 4 and 5 forecasts: Part 5 twin features use alpha_E = 0, so they cannot change
     (checked by recomputing the features with the D-056 kernel; expected bitwise identical). Part 4:
     the full CGMacros twin protocol is re-run with the D-056 kernel into separate files
     (alpha selection on training patients, test forecasts) and compared with the stored Part 4
     forecasts (max |difference| per variant; MAE difference). `actoff` variants must be identical;
     `acton` variants are expected to change by a negligible amount (alpha_E at the grid bottom).
- **Pass rule:** none (post-hoc). Results are reported as they come out.
- **Training-window check done (before the test-window run):** fold-0 training patients, the
  same 668 windows as the Part 7 dev stage (identical per period). Walk direction pass 100 % for
  both `twin_learned_d056` and `twin_assumed_d056`; max peak rise 0.0 mg/dL, max walk AUC effect
  0.0. D-056 alphas equal the originals (calibration starts at basal, where the two kernels agree).
- **Known indirect effect (synthetic check, disclosed):** activity never raises the DIRECT
  glucose rate, but lowering glucose also lowers the insulin surrogate, so a later part of the
  trajectory can sit slightly higher (up to 0.86 mg/dL at alpha 0.02, 3-4 h after a walk that
  started at 200 mg/dL). The D-051 metrics (peak, AUC over 0-120 min) never increase in the
  synthetic checks. Not changed (owner rule is about the activity term itself).
- **Part 5 check done:** twin features recomputed with the D-056 kernel at alpha_E = 0 for 24
  fold-0 patient x level combinations: max |D-056 - original kernel| = 0.0 (bitwise). Part 5
  results cannot change. (The recomputed features match the stored Part 5 files to 1.5e-5, a
  floating-point difference present for both kernels.)
- **Test-window run = forced re-run 1 (disclosed):** `python -m src.whatif.d056 final --force`
  (owner-approved), twin walk tests only, the same 830 test windows as Part 7 (identical per
  period). Lock file: `runs: 2, forced_reruns: 1`. No ML, carb or matched-meal result was
  re-run or changed; the original Part 7 numbers stay in the report.
  Result (post-hoc): walk direction 100 % [100, 100] for `twin_learned_d056` and
  `twin_assumed_d056` (43 patients; max peak rise 0.0; median walk AUC effect -2.9 / -218
  mg/dL x min). Original kernel: 67.2 % / 69.4 %.
- **Part 4 re-run (expectation NOT met, disclosed):** the full CGMacros Part 4 twin protocol was
  re-run with the D-056 kernel (separate files `data/processed/whatif/d056_part4_twin_fold*`).
  - `actoff` variants: bitwise identical (max |diff| 0.0), as expected.
  - `acton` variants: CHANGED. With the D-056 kernel the training-patient alpha_E selection
    moved off the grid bottom (0.0001-0.00025 -> 0.002 in folds 0, 1, 2, 4 and 0.004 in fold 3),
    because activity no longer harms forecasts when glucose is below basal. Test MAE at +60
    changes by -0.9 to +0.1 mg/dL per variant (pop k0 22.29 -> 21.38; pop k3 18.11 -> 18.20);
    single forecasts differ by up to 293 mg/dL (median 0.9; 12 % of forecasts by > 5 mg/dL).
    The Part 4 conclusions used the `actoff` twin as the main model and found no activity
    benefit; `acton` k3 (the Part 4 comparison level) is not better with D-056 (18.20 vs
    18.11). The stored Part 4 results are NOT replaced.
  - Part 5 hybrids and Part 6 use alpha_E = 0 (bitwise unchanged, checked above).
  - Consequence for Part 7: `twin_learned_d056` uses the ORIGINAL Part 4 `alpha_on` (as
    pre-specified in this entry), not the re-selected value.


## D-057 Part 9 step 1: explainability (SHAP + templates, no LLM)
- **Model explained:** the Part 5/6 population LightGBM (A0: CGM + time of day + meals + glycemic
  static, delta target), one booster per horizon. TreeSHAP (LightGBM `pred_contrib`, restricted to
  the `n_iter` trees the model uses) explains the predicted CHANGE in mg/dL: base value + sum of
  contributions = predicted delta (checked on 60 training-patient rows: max |diff| 2e-13 at every
  horizon; the `shap` package gives identical values, max |diff| 0.0).
- **Template rules (`src/explain/templates.py`, `configs/explain.yaml`):** a feature is quoted
  only if (i) the model really uses it (positive split gain in the used trees), (ii) |SHAP| >= 1
  mg/dL at that horizon, (iii) the push points the way the sentence says (towards the event for
  moderate / high / very high risk, away from it for low risk). One reason per topic (trend, meal,
  level, activity, time, profile, data), at most two clauses; clinically meaningful topics are
  listed before time of day / profile / data. Every number in a clause is read from the row; a
  feature without a template raises `KeyError` (tested for every used feature).
- **Wording of probabilities** (display choice, not a model result): < 10 % low, < 30 % moderate,
  < 60 % high, otherwise very high. The probability itself is the Part 6 calibrated event
  probability (Platt on training out-of-fold rows); not available is shown as "not available".
- **Limits:** SHAP explains the model, not the physiology: a quoted feature is what moved the
  forecast, not a cause. Time of day and HbA1c can appear as reasons when nothing else stands out.
  Explanations exist for the population model only (not for the monotone model or the twin).
- **Fonts (Part 9 design):** Atkinson Hyperlegible (Braille Institute) and Source Serif 4 (Adobe),
  both SIL Open Font License 1.1, bundled locally as woff2 in `app/static/fonts` (latin subsets
  fetched once from the fontsource packages at build time, never at run time).

## D-058 Part 9 step 3: app data bundle (`make app-data`, `app/demo`, offline)
Status: written before the first run of the builder; the guard values below are recorded in
`app/demo/manifest.json` by each build, not claimed here.
- **Owner decisions (checkpoint 1):** a display-only very-high band at 250 mg/dL and a very-low
  band at 54 mg/dL (no risk is computed for them; models and alerts keep 70, 140, 180); damson
  plum with Atkinson Hyperlegible and Source Serif 4; what-ifs run LIVE from saved models.
- **What is shipped:** the fold-0 population models (unconstrained and monotone forms, full model
  plus the three inner-fold models of the Part 6 CV+ arm, trimmed to the trees they use) as
  LightGBM text, their out-of-fold residuals, Platt maps and alert thresholds from the fold's
  TRAINING patients, stored cohort numbers copied from the Part 3/5/6/7 metric files, SHAP values
  and template sentences, and six evidence figures. No raw or processed dataset is committed
  (`app/demo/` is gitignored; CGMacros is CC BY-NC-SA).
- **Which patients:** fold-0 test patients that have Part 6 test rows, chosen by a rule fixed in
  `configs/app.yaml` before looking at any result (first patient per diabetes status by sorted id,
  then the next T2D patient), plus one SYNTHETIC patient from the Part 4 simulator (the simulated
  patient whose Gb is closest to the cohort median), labelled synthetic on every page.
- **No outcome is ever used for a displayed patient:** the builder reads the test patients'
  anchor rows only after dropping `y_*` and `*_any_*` columns (`cgm_slope_30` is a trailing-slope input, kept); every forecast is from a
  fold model that never saw the patient; calibration and alert thresholds are fitted on training
  patients only. A guard fails the build if an outcome column appears in any output frame.
- **Live what-if:** meal and activity inputs are recomputed on the period prefix with the Part 2
  code (Part 7 edit semantics); CGM, time and static inputs cannot change. The shipped runtime
  (`src/appdata/rt.py`) needs no LightGBM and no training data; the build checks it reproduces the
  in-memory forecasts and CV+ bands (`rtform_max_abs_diff`, limit 1e-6 mg/dL).
- **Twin:** needs the fitted per-patient state, so its what-if trajectories are precomputed on a
  (carbs, slowing-ratio) grid per window, once with the learned activity effect and once with the
  assumed walk effect (D-052), both with the one-sided activity term (D-056). Values between grid
  nodes are bilinear interpolations and are labelled as such; the build measures the interpolation
  error against the engine at random off-grid points (`interp_max_abs_diff`, limit 5 mg/dL).
  The synthetic patient has no fitted twin: "not available".
- **Alerts in the replay are causal:** the at-risk flag uses only readings up to the anchor, which
  differs slightly from Part 6 (its episode flags knew the confirmation 15 min later). This is a
  display rule and is stated in the app.
- **Confidence label** (uncertain / moderate / high) is a DESIGN CHOICE, not a validated rule:
  interval wider than 1.5 x the fold's median CV+ width at +60, or data quality below 0.70, or a
  sensor reading older than 60 min.
- **Removed:** `src/dash/live.py` duplicated `src/appdata/rt.py` and imported LightGBM; the
  calibration and patient-state code moved into `rt.py`.
- **Not run yet:** the builder has not been executed at the time of writing.

## D-059 Part 9 step 3: UI rules (apply to every page)
- Language is decision support: never "diagnosis" or "prescribe". A footer on every page states
  that GlucoTwin is a research prototype, not a medical device.
- Only the baseline glucose Gb of the twin is shown (the only identifiable parameter, test-retest
  r 0.62; the others 0.14 to 0.24); SG, SI, fc, kabs and beta are never displayed.
- Every what-if result carries the fixed note: "Model-based estimate, not a causal guarantee. In
  matched real breakfasts the models predicted about half the observed rise; treat sizes as lower
  bounds." The monotone model is the default; the twin and the unconstrained model are alternatives.
- Interpolated twin values, assumed inputs (synthetic patient profile, macro ratios) and display
  thresholds are labelled as such.
- Data-quality text says the score detects missing data, not wrong data (bias and drift are not
  detected, Part 6). Personalization text says the gain is about 1 mg/dL after 3 to 5 days of the
  patient's own CGM and that heart rate and activity add nothing measurable (Parts 4 and 5).
- Colour is never the only carrier of meaning: each glucose state has a pattern and a label.

## D-060: Dashboard visual redesign after first screenshots (Part 9)

Context: first screenshots looked plain, the Material icon font printed as text ("_arrow_right"), a heading was duplicated, the overview URL was wrong in the screenshot script, and Atkinson's slashed zero looked odd in numbers.
Decision: dark damson sidebar with icons, white rounded cards, three number tiles per patient (glucose now, +60 min, chance of the headline event), word-plus-dot chips for risk and alerts, softer zone patterns (pattern and label still on every zone), system UI font stack instead of Atkinson (offline, native look, no slashed zero), Source Serif 4 kept only for the decision sentence. Streamlit toolbar and Deploy button hidden.
Unchanged: every wording rule, fixed what-if note, disclaimer on every page, decision-support language, display of only Gb, no invented numbers. This departs from the flat, text-only look in docs/DESIGN.md section 2 at the user's request.

## D-061: Part 8a cross-cohort transfer protocol (written before any run)

Question: how much accuracy is lost when a model trained on one open cohort forecasts another, and what recovers it?
Setup: CGMacros on the common 15-min grid (`cgmacros_15` tables) and Shanghai T2DM, both directions, 6 h windows, +30/+60/+120 min. LightGBM, delta target, Part 2 protocol (inner patient split for early stopping).
Features: only columns present in both tables: `cgm_*` (shared names), time of day (sin, cos; `days_since_start` excluded because recording lengths differ), and optionally the five static fields both cohorts have (age, sex, BMI, HbA1c, fasting glucose). No wearable or meal features (Shanghai has none).
Methods, all scored on the same rows (first period of each target patient, anchors at day >= 3, patients with >= 96 such anchors, real y at +60):
1. persistence (last CGM value);
2. in-cohort: target-cohort CV repeat 0 with the same features and the saved split files;
3. zero-shot: trained on all source patients only;
4. zero-shot + few-shot correction with the target patient's first k = 1 or 3 days (shrunk mean residual "offset", or ridge on four features). Only pairs whose target time is before day k are used (unit-tested). Hyperparameters are fixed in `configs/transfer.yaml` before running and are not tuned on target patients;
5. pooled: all source patients plus the target training patients of each CV fold.
Reported: MAE with patient-bootstrap 95% intervals; paired differences (zero-shot vs in-cohort = domain-shift penalty; pooled vs in-cohort; few-shot vs zero-shot); for CGMacros as target also the T2D-only subgroup (Shanghai is T2D only, so this is the fair comparison).
Not done: South Asian subgroup (neither dataset records ethnicity; see data_card.md). Event (hypo / hyper) transfer is not run. A negative result will be reported as found.

## D-062: Part 8b India synthetic cohort, stress test and augmentation (written before any run)

Question: if South Asian physiology differs from the training cohorts in the assumed ways, how does a model trained on real data degrade, and does adding synthetic data to training help real patients?
Why synthetic: no open dataset records ethnicity and has CGM, meals and wearables together, and the competition rules require anonymized, open-source or synthetic data. AI-READI was checked (D-063 note below) and needs a controlled-access request plus an Azure storage container; it has no confirmed meal log and its ethnicity field is controlled.
Cohort: 100 simulated patients x 10 days per strength. Generator: Part 4 mechanistic simulator; the log-parameter covariance comes from the real CGMacros twin fits (k = 3 days), only the MEAN is shifted. Shifts at strength 1.0: insulin sensitivity ln(4.7/7.5) = -0.46 (clamp study, healthy Asian Indians vs Caucasians, J Clin Endocrinol Metab 2001; the mapping to the model parameter is an ASSUMPTION), fasting glucose parameter +0.05 and absorption rate +0.15 (both ASSUMPTIONS). Strengths: control 0, mild 0.5, moderate 1.0, strong 1.5.
Inputs: 16 assumed Indian meal presets (src/india/meals.json; composition from IFCT 2017 tables, recipes assumed), a documented wearable generator (walking bouts, then activity, heart rate and intensity), static fields derived from the simulated CGM (HbA1c from eAG = 28.7 x A1C - 46.7, Nathan 2008) plus assumed age, sex and BMI draws. Noise and activity gain come from stored Part 4 results.
A. Stress test: LightGBM trained on all real CGMacros patients (same features that exist in both tables: cgm, time of day, wearable, meal, five static), zero-shot and offset few-shot (k = 1, 3) on each synthetic cohort; same evaluation rows rule as D-061; MAE, bias, paired differences with patient-bootstrap intervals; compared with persistence and with the in-cohort CGMacros reference on the same features.
B. Augmentation: CGMacros CV repeat 0; train on real patients only, real + control cohort, real + moderate cohort; evaluate on the real validation patients only. Fixed hyperparameters; no tuning on evaluation patients.
Every number involving a synthetic cohort is labelled SIMULATED. It supports claims about robustness under assumed shifts, never claims about accuracy on Indian patients. A negative result will be reported as found.
Known limits: the cohort skews to the diabetic range (the fitted parameters come from CGMacros, which includes healthy participants but the shifts push towards impaired glucose handling); meal presets are assumed recipes; wearable signals are simple functions of the simulated activity; carbohydrate energy share is compared with ICMR-INDIAB (62%) in the cohort summary.
Note on AI-READI: public CGM and Garmin files exist, but access requires CILogon, ethics training, a licence agreement, a publicly listed research description and an Azure container. Not pursued; listed as future work.

## D-063: Part 8c federated learning simulation (written before any run)

Question: can clinics that cannot share patient records still train a shared glucose forecaster, and what does it cost in accuracy compared with pooling the data? This is a SIMULATION on one machine: the "clinics" are groups of patients from the two open cohorts (no network, no real hospitals).
Data and features: CGMacros and Shanghai T2DM on the common 15-min grid, the shared features of D-061 (`cgm_*`, time of day, five common static fields), 6 h windows, targets delta at +30/+60/+120 min (glucose at +h minus now). Evaluation rows follow D-061 (first period, anchors from day 3, patients with at least 96 evaluable anchors).
Model: a small MLP (64-32, ReLU) written in NumPy so it runs anywhere and is unit-tested. It is NOT claimed to beat the LightGBM models of Parts 2-3; the comparison is federated against centralized and local-only versions of the SAME model. Inputs are standardized with statistics aggregated across clinics from sums only (a federated-analytics step; aggregate statistics are shared, not records); missing values become 0 after standardization.
Clinic layouts (5 clinics except where noted): (1) IID: patients of both cohorts dealt at random; (2) non-IID by HbA1c: patients sorted by HbA1c and cut into contiguous groups; (3) by dataset: 2 clinics, one per cohort (cross-silo). In each clinic 20% of patients are held out for testing (5 random splits; same hold-out for every method); training uses only the other patients.
Methods: persistence; local-only (each clinic alone, tested on its held-out patients); centralized (all training patients pooled); FedAvg (sample-weighted, all clinics every round); FedProx (proximal weight mu); FedAvg with client-level differential privacy (update clipping at norm C, Gaussian noise z*C on the sum, equal weights). Every method makes the same number of passes over each patient's training data (rounds x local epochs), so centralized training takes more optimizer steps than one client does. All hyperparameters (rounds, local epochs, learning rate, mu, C, z) are fixed in `configs/federated.yaml` before the first run and are not tuned on test patients.
Privacy accounting: epsilon of the DP runs is reported with delta = 1e-5 from the Renyi-DP accounting of the Gaussian mechanism under full participation (no subsampling amplification claimed). With only 2 to 5 clients, client-level DP is expected to cost a lot of accuracy; the result will be reported as found. The DP here protects a clinic's whole contribution; it is not patient-level DP.
Reported: MAE per method, layout and horizon with 95% patient-bootstrap intervals (resampling patient-split pairs); federated minus centralized and federated minus local-only paired differences; epsilon table.
Not done (listed deferred): patient-level DP-SGD (per-example clipping); secure aggregation; communication cost beyond counting rounds and parameters; real multi-site deployment; heterogeneous model architectures; client dropout. Mapping to the DPDP Act (purpose limitation, data minimisation, no raw-record transfer) is in docs, not a legal opinion.

## D-064: Part 8d simulated sleep, HRV and step channel, and a detectability (power) analysis (written before any run)

Why: the organizers' page names sleep stages, heart-rate variability and step count as example dynamic streams. Our real cohorts have none of them (CGMacros has heart rate and activity only), and Parts 4 and 5 found that heart rate and activity add nothing measurable. This part shows that the pipeline CAN ingest such channels and asks how large a sleep effect would have to be before the model could use it. It does not claim that sleep helps real patients.
Generator (all SYNTHETIC, India cohort of D-062): one night of sleep before each day: duration ~ N(6.8, 1.0) h clipped to [3.5, 9.5], efficiency ~ N(0.88, 0.05), deep-sleep fraction ~ N(0.17, 0.04) (independent of glucose, an ASSUMPTION, so these carry no signal), nightly HRV (RMSSD) = 32 ms x exp(-0.05 x deficit + noise) where deficit = 7 h minus duration (the HRV link to short sleep is an ASSUMPTION), and a step count derived from the simulated activity. A night's summary becomes known at 06:00 on the day after it (causal).
Glucose effect (the only path from sleep to glucose): from 06:00 each day insulin sensitivity SI is multiplied by exp(-beta x m x deficit), where deficit = 7 h minus last night's duration (never negative: longer sleep has no effect), beta = 0.03 per hour (ASSUMPTION, between 0.023 and 0.045 per hour implied by Buxton et al., Diabetes 2010: 7 nights at 5 h versus 10 h in bed, n = 20 healthy men, insulin sensitivity -11 +/- 5.5% by clamp and -20 +/- 24% by IVGTT) and m = effect multiplier. Conditions m = 0 (no effect, the false-positive control), 1 (literature scale), 3 and 5 (stronger than the literature, to find the point where the channel becomes usable). Patients, parameters, meals and sleep draws are identical across conditions (paired design); only m changes.
Test: within each condition, patient-level 5-fold CV (seeded random folds) of LightGBM (D-061 fit protocol, +30/+60/+120 min, 15-min grid, 6 h windows) with three nested feature sets: A = CGM, time of day, meal features; B = A + heart rate and activity; C = B + sleep, HRV and step features. Evaluation rows: day >= 3 with a real target. Reported: MAE, paired patient-bootstrap difference B - A and C - B with 95% intervals.
Reading the result: if C - B is indistinguishable from 0 at m = 1 and negative only at larger m, the honest statement is "at literature-scale effects the sleep channel is not detectable in 10 days of data; the dashboard shows it as context only". If C - B is negative at m = 0 something is leaking and the run is a bug. A negative result will be reported as found. No claim about real patients.

### D-063a: exploratory noise sweep (added AFTER the first federated run; not pre-registered)

The two noise multipliers fixed in D-063 (z = 0.5 and 1.0) both ruined the model (about +20 to +155 mg/dL at +60 min, epsilon 50 to 141), which is the outcome D-063 said to expect with 2 to 5 clinics. To show where accuracy survives and what privacy that buys, a sweep over smaller multipliers z = 0.01, 0.03, 0.1, 0.2 is run with everything else unchanged (same clip, rounds, splits, seeds, same FedAvg reference). It is labelled EXPLORATORY in the metrics file and in every figure because it was chosen after seeing the first result. It is a privacy-utility curve, not a tuned model: no setting is selected on test patients and no setting is proposed for use. Expected reading: accuracy returns only when the noise is so small that epsilon is in the thousands or more, i.e. client-level differential privacy is not usable with a handful of clinics; a real deployment would need many more clients or patient-level DP-SGD (still deferred).

## D-065: Part 10 tuned LightGBM (protocol written before any run)

Why: the shipped forecaster uses a squared-error loss and tunes only the number of leaves, while it is scored on absolute error. A wider search and an absolute-error-type loss may lower MAE. Because Parts 2 to 9 looked at the same patients many times, the protocol below limits how much a gain can be due to selection.

Candidates (`configs/improve.yaml`): 16 candidates, the first one is exactly the Part 2/3 setting, the rest are seeded random draws over loss (squared, absolute, Huber with alpha 10 and 25), leaves, minimum child samples, learning rate, column and row subsampling and L1/L2 penalties. The candidate list is identical for every fold and horizon.

Selection: inside each outer training fold, on the INNER validation patients only (the Part 2 inner split). Score = MAE. The 3 best candidates are averaged. Test patients are never used for any choice, so the search cannot do worse than the status quo on inner validation but may do worse on test.

Feature sets: CGMacros D_cgm_wear_meal and E_full, Shanghai T2DM E_full (fusion.yaml), delta target. Nothing else changes.

Design A (cv): outer CV repeat 0 (5 patient-grouped folds), status quo refitted on the same folds, paired patient-bootstrap difference tuned minus status quo at +30/+60/+120.
Design B (holdout): the holdout test patients (never used for any choice in Parts 2 to 9; used in Part 6 only for evaluation), run ONCE after the design A result is recorded, train = holdout train patients, validation = holdout calibration patients. The runner refuses a second run.

Decision rule (fixed now): the tuned model becomes the default forecaster only if (1) on design A its +60 min difference has a CI entirely below 0 in at least one cohort, (2) its point difference is below 0 at all three horizons for every cohort and config, and (3) the holdout +60 point difference has the same sign. Otherwise the status quo stays and the result is reported as found. Either way the numbers go in the README.

What this does not do: it does not change event classifiers, intervals, the twin or the what-if engine. If tuned forecasts are adopted, intervals and event scores are NOT recomputed in this part; that would be a new, separately documented step.

## D-066: Part 8d scale-up: would more simulated patients and days reveal the sleep effect? (written before the run)

Why: the D-064 run (100 patients, 10 days) found no detectable gain from sleep, HRV and steps at any multiplier. A fair next question is whether that is only a lack of data. Same generator, same features, same CV, same analysis; only the size changes: 300 patients, 30 days, multipliers 0 (control) and 1 (literature scale). Seed 212 (different draws from D-064).

Reading (fixed now): if C minus B at m = 1 is still indistinguishable from 0, the honest statement is that, at literature-scale effects, the sleep channel does not improve 30 to 120 minute forecasts even with about 9 times more patient-days, because the mean glucose shift it causes (about 0.3 mg/dL at m = 1 in D-064) is far smaller than meal-driven variation. If C minus B is below 0 with a CI excluding 0 at m = 1 and not at m = 0, it is detectable at that scale. m = 0 below 0 would be a leak and a bug. This is a SIMULATION result and says nothing about real patients.

Not done and why: ingesting AI-READI (access barriers, D-0xx) and attaching invented sleep or HRV to real patients (it would carry no real signal).

