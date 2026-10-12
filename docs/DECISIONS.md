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
