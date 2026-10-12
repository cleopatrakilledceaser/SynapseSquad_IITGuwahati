# GlucoTwin Makefile. Works on Linux/CI and on Windows with GNU Make (ezwinports) + PowerShell venv.
# `make all` rebuilds everything end-to-end.

ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
else
PY ?= python
endif

.PHONY: all setup data features models figures app test lint format smoke clean help

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
