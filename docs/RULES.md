# GlucoTwin — Permanent Rules (apply to every part)

Source: project brief from the owner (Part 1 kickoff). Read this, `docs/PROGRESS.md`
and `docs/DECISIONS.md` at the start of every session.

## Working style
- Senior ML engineer with healthcare/clinical-informatics experience, pair-building with a solo owner.
- Work in small verifiable steps. Do ONE PART at a time. At the end of a part: STOP, report, and wait for approval.

## Project
GlucoTwin is a patient-specific digital twin for glucose management in Type 2 Diabetes/prediabetes,
for CLINICIANS (Happiest Health Digital Twin Challenge 2026). It fuses static clinical data (the
"prior") with dynamic CGM + wearable + meal streams (the "evidence") to forecast glucose at
+30/+60/+120 min with calibrated uncertainty, fast per-patient personalization, a mechanistic+ML
hybrid twin, and physiologically valid what-if simulation. Planned parts: fusion ablation,
mechanistic ODE twin, hybrid, personalization, conformal uncertainty, what-ifs, robustness,
federated learning, cross-cohort transfer, an India synthetic cohort, a Streamlit dashboard, and
submission docs.

## Engineering
1. Python 3.11, pinned requirements.txt (+ requirements-dev.txt), global seed 42, all config in
   configs/*.yaml (no magic numbers).
2. Typed code, docstrings, ruff + black, pytest, GitHub Actions CI (lint + tests + tiny smoke run
   on a toy fixture).
3. Dockerfile + Makefile; `make all` rebuilds everything; also make data / features / models /
   figures / app / test.
4. Every experiment writes metrics to reports/metrics/<name>.json and figures to
   reports/figures/<name>.png (300 dpi, labeled axes with units, colorblind-safe palette). Log git
   hash, config, seed, data hash with each metrics file.
5. Maintain docs/PROGRESS.md (status/next steps) and docs/DECISIONS.md (every non-obvious choice
   and why). Read both at the start of every session. This rules block lives in docs/RULES.md and
   .clinerules.

## Scientific
1. ALL splits are by PATIENT. No patient in more than one of train/calibration/test. Unit-test it.
2. NO leakage: features at time t use only data at or before t. Unit-test it.
3. Never tune on test. Tune with patient-grouped CV on train only. With few patients, use repeated
   patient-grouped CV and report mean + 95% bootstrap CI over patients (not rows).
4. Report honestly. If something does not help, say so. Suspiciously good numbers = assume leakage
   and investigate.
5. Only open-access or synthetic data. Synthetic data is always labeled synthetic and never mixed
   into a real test set.
6. Verify dataset facts from the real files/pages at runtime; never rely on memory; cite the source
   (URL or file) for each fact. Real data wins over the brief if they conflict; report the conflict.
7. Never redistribute raw datasets; scripts/instructions only. Respect dataset licenses.
8. No LLM chatbot, no cloud infra, no databases, no auth.

## Clinical
- Units mg/dL; hypo < 70, hyper > 180, secondary hyper > 140 (confirm against data).
- UI language is decision support, never "diagnosis"/"prescribe".
- It is a research prototype, not a medical device.

## Token economy
Never print or open large files in full. Use head, shape, dtypes, describe(), value_counts, small
samples. Keep command output short.

## STOP-AND-TELL-ME triggers (do not work around, do not guess)
- login/registration/credential wall
- download needing manual action
- license/terms conflict
- missing stream we need (CGM, wearable, meals, static labs)
- file contents differ from documentation
- a test failing twice
- any doubt affecting scientific validity

When stopping, state what is wrong, what you tried, and 2-3 options with a recommendation.

## Environment notes (owner instructions, Part 1)
- Project lives at C:\Projects\glucotwin (standalone; never touch the shared Cline chat workspace).
- Use the project venv only: `.venv\Scripts\python` (Python 3.11). All installs go in the venv.
- Pin only versions verified to install cleanly in the 3.11 venv.
