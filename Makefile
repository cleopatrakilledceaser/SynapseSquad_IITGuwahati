# GlucoTwin Makefile. Works on Linux/CI and on Windows with GNU Make (ezwinports) + PowerShell venv.
# `make all` rebuilds everything end-to-end.

ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
else
PY ?= python
endif

.PHONY: india transfer app-data app-check all setup data features models fusion mech hybrid personalize uncertainty whatif figures app test lint format smoke clean help

help:
	@echo "Targets: setup data features models figures app test lint format smoke all"

all: data features models figures test

setup:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

data:
	$(PY) data/scripts/download.py cgmacros shanghai
	$(PY) -m src.data.make_dataset
	$(PY) data/scripts/qc_summary.py
	$(PY) data/scripts/validate_cgm_reconstruction.py all
	$(PY) data/scripts/amount_consumed_analysis.py

features:
	$(PY) -m src.features.make_features
	$(PY) -m src.features.make_splits

models:
	$(PY) -W ignore -m src.models.run_baselines
	$(PY) -m src.eval.report

fusion:  ## Part 3 (about 4 h CPU, resumable: cached runs in data/processed/fusion are skipped)
	$(PY) -W ignore -u -m src.models.run_fusion all
	$(PY) -W ignore -m src.eval.fusion_report

mech:  ## Part 4 mechanistic twin (about 9.5 min elapsed, folds in parallel; cached in data/processed/mech)
	$(PY) -W ignore -u -m src.mech.run_mech all
	$(PY) -W ignore -m src.mech.report
	$(PY) -W ignore -m src.mech.report rev

hybrid:  ## Part 5.1 twin features + hybrids A0/A/B/C + neural onset fix (cached in data/processed/hybrid; about 2.5 h CPU)
	$(PY) -W ignore -u -m src.hybrid.run_hybrid all
	$(PY) -W ignore -m src.hybrid.report

personalize:  ## Part 5.2 cold-start personalization, both cohorts (needs make hybrid features; about 2 h CPU)
	$(PY) -W ignore -u -m src.personalize.run all
	$(PY) -W ignore -m src.personalize.report

uncertainty:  ## Part 6 intervals / calibration / alarms / robustness + patient fusion (about 35 min CPU, resumable)
	$(PY) -W ignore -u -m src.uncertainty.run all all
	$(PY) -W ignore -m src.uncertainty.report --final
	$(PY) data/scripts/make_results_md.py


whatif:  ## Part 7 what-if engine: dev (fold-0 training) + final (run ONCE; lock file) + report (about 13 min CPU)
	$(PY) -m src.india.library check
	$(PY) -W ignore -u -m src.whatif.run dev
	$(PY) -W ignore -u -m src.whatif.run final
	$(PY) -W ignore -u -m src.whatif.d056 dev
	@echo D-056 test-window walk run (forced re-run 1) and Part 4 re-run: python -m src.whatif.d056 final --force / part4
	$(PY) -W ignore -m src.whatif.report --final
	$(PY) data/scripts/make_results_md.py

transfer:  ## Part 8a cross-cohort transfer (both directions; needs make features)
	$(PY) -W ignore -u -m src.transfer.run
	$(PY) -W ignore -m src.transfer.report

india:  ## Part 8b India stress test + augmentation (SYNTHETIC cohorts; needs make features)
	$(PY) -W ignore -u -m src.india.stress

figures:
	$(PY) -m src.eval.eda

app-data:
	python -m src.appdata.build

app-check:
	python -m src.appdata.build --check

app:
	streamlit run app/main.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests data/scripts
	$(PY) -m black --check src tests data/scripts

format:
	$(PY) -m ruff check --fix src tests data/scripts
	$(PY) -m black src tests data/scripts

download:
	$(PY) data/scripts/download.py cgmacros shanghai bigideas

smoke:
	$(PY) -m src.smoke
