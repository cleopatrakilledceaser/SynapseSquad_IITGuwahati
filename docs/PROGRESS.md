# PROGRESS

## Part 1: Scaffold + Data
| Step | Status |
|---|---|
| A. Scaffold, patient-level splits, tests, git init | ✅ done, approved (4ac154e) |
| B-pre. Library compatibility | ✅ done: torch 2.5.1+cpu + opacus 1.5.4 run locally (`requirements-torch.txt`); flwr kept out of the venv (D-008/D-009) |
| B. Dataset verification | ✅ done (tag `part-1b-done`); see docs/data_card.md. Awaiting owner review |
| C. Harmonization + QC | ⏳ next, after owner approval |
| D. EDA + data card figures + integrity tests | ⏳ blocked on C |

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

## Next steps
1. Owner reviews the Part 1B report (open questions: Amount Consumed > 100; whether to download
   BIG IDEAs).
2. Step C: harmonized schema + QC; choose 5- vs 15-min grid.
