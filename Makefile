# GlucoTwin Makefile. Works on Linux/CI and on Windows with GNU Make (ezwinports) + PowerShell venv.
# `make all` rebuilds everything end-to-end.

ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
else
PY ?= python
endif

.PHONY: all setup data features models fusion mech figures app test lint format smoke clean help

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

figures:
	$(PY) -m src.eval.eda

app:
	@echo "[app] not yet implemented"

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
