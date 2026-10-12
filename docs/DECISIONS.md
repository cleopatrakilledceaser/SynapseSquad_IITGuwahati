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
