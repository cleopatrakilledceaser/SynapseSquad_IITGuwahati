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
	@echo "[data] not yet implemented (Part 1 Step C)"

features:
	@echo "[features] not yet implemented"

models:
	@echo "[models] not yet implemented"

figures:
	@echo "[figures] not yet implemented"

app:
	@echo "[app] not yet implemented"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m black --check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m black src tests

smoke:
	$(PY) -m src.smoke
