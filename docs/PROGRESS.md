# PROGRESS

## Part 1: Scaffold + Data
| Step | Status |
|---|---|
| A. Scaffold, patient-level splits, tests, git init | ✅ done, approved (4ac154e) |
| B-pre. Library compatibility | ✅ done: torch 2.5.1+cpu + opacus 1.5.4 run locally (`requirements-torch.txt`); flwr kept out of the venv (D-008/D-009) |
| B. Dataset verification | ✅ done, approved (tag `part-1b-done`) |
| C. Harmonization + QC | ✅ done: `make data` (about 20 s) -> data/processed/*.parquet (D-015 to D-019) |
| D. EDA + data card + integrity tests | ✅ done, approved (tag `part-1-done`) |

## Part 2: Windows, features, splits, baselines
| Step | Status |
|---|---|
| Windows + labels (D-021) | ✅ `src/features/windows.py` |
| Causal features (D-023) | ✅ `src/features/build.py`; `make features` builds N = 2/4/6 h tables + splits |
| Splits (D-022) | ✅ holdout, 5×5 patient-grouped CV, chronological; saved with sha256 |
| Evaluation (D-025) | ✅ `src/eval/metrics.py`, `src/eval/report.py` (patient bootstrap) |
| Baselines (D-024) | ✅ 7 forecasters + LightGBM event classifiers; `make models` (about 70 min) |
| Leakage tests | ✅ causality per feature group, real-only targets, no calendar, patient bootstrap, shuffle |
| Status | Done (tag `part-2-done`). Awaiting owner review. No fusion / mechanistic code yet |

## Environment (dev box)
- Project `C:\Projects\glucotwin`, venv `.venv` (Python 3.11.9). GNU Make 4.4.1 (ezwinports).
- Windows Application Control blocks newer unsigned DLLs (pandas 3, sklearn 1.9, torch 2.14), so
  older versions are pinned (D-006, D-009). `pip check` clean.
- Docker: Dockerfile written but **NOT built/verified locally** (Docker not installed).
- CI is defined; its first run happens on push to origin.
- Also installed in the venv (not yet pinned): lightgbm, shap, statsmodels, streamlit, plotly,
  graphviz, python-pptx, nbformat, xlrd, openpyxl. Pin them in the part that first uses them.

## Data on disk (gitignored; reproduce with `python data/scripts/download.py ...`)
- `data/raw/cgmacros/`: full v1.0.0 release, SHA256-verified.
- `data/raw/shanghai/diabetes_datasets.zip`: MD5-verified.
- `data/raw/bigideas/`: metadata only. The 4.7 GB zip is not downloaded.
- `data/interim/cgmacros_raw_coverage.csv`, `cgmacros_native_probe.csv`: verification outputs.

## Key findings carried into Step C
- CGM columns are interpolated to 1 min: recover native 5-min Dexcom readings and mask gaps
  > 30 min (D-010).
- HbA1c is in % (D-011). Time of day is valid; dates and weekday are not (D-012). No sleep data (D-013).
- Schema varies across files (9 column sets); meal-type labels need normalising; `Amount Consumed`
  ranges 0-900 and is absent in 028/029.

## Part 1 C/D outputs
- `data/processed/`: cgmacros_ts5, common_ts15, meals, static (+ manifest.json).
- `reports/metrics/`: qc_summary, cgm_reconstruction_validation, amount_consumed_analysis,
  eda_summary (.json) + per-patient coverage CSVs.
- `reports/figures/`: 6 EDA figures.
- Known data-quality flags for later parts:
  - 007 (irregular file, lower-quality reconstruction);
  - 018 (CGM gap of 9.4 days);
  - 026 (40-h gap);
  - 032 (48% HR coverage);
  - 028/029 (no amount column).
- Disk: 711.7 GB free on C: (BIG IDEAs needs about 39 GB including the zip). Not downloaded.

## Part 2 outputs
- `data/processed/features/` (gitignored): `<cohort>_N{2,4,6}.parquet`, `feature_sets.json`,
  `window_counts.json`.
- `data/processed/splits/` (gitignored): holdout / cv / chrono + `splits_manifest.json`.
- `data/processed/predictions/` (gitignored): out-of-fold CV and holdout-test predictions.
- `reports/metrics/`: `baselines_cgmacros.json`, `baselines_shanghai_t2dm.json`,
  `baselines_checks.json`, per-patient CSVs.
- `reports/figures/`: `baselines_error_vs_horizon.png`, `baselines_clarke_grid.png`,
  `baselines_per_patient_spread.png`.
- Pinned in requirements.txt: lightgbm 4.7.0, statsmodels 0.15.0.

## Next steps
1. Owner reviews the Part 2 report.
2. Part 3 (fusion ablation) uses the `full` / `shared` feature sets and the same splits/evaluation.
   The Part 2 baselines are the bar to beat (per-patient AR is the strongest at +120 min on CGMacros).
