# HANDOFF (working notes for the next Claude session; delete before final submission)

Project: GlucoTwin, clinician-facing T2D glucose digital twin, Happiest Health Digital Twin Challenge 2026.
Owner: Jahnavi Harsha (solo, IIT Guwahati). Repo: cleopatrakilledceaser/glucotwin, branch `wip-part9`.
Windows + PowerShell, venv `.venv`, repo at `C:\Projects\glucotwin`. Read `docs/DECISIONS.md` (D-001 to D-062) and `docs/PROGRESS.md` first.

## How to work with the owner
- Give steps one at a time, plainly, one command at a time. Do not cut scope or decide for her; list anything deferred.
- Do not raise or ask about timing questions.
- Commit trailers on every commit:
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01M2M5UiL7jkag7dt9S3PLnp` (or the new session's own line).
- Never put phone number or email in the repo. No raw or processed data in git. No invented numbers.
- Stop and tell her at any login, credential or payment wall (for example AI-READI needs CILogon, training and an Azure container: NOT pursued, future work).
- Language: decision support, never "diagnosis" or "prescribe". Display only the twin's Gb. Fixed what-if note:
  "Model-based estimate, not a causal guarantee. In matched real breakfasts the models predicted about half the observed rise; treat sizes as lower bounds."
  Disclaimer footer on every page. Display-only glucose bands at 54 and 250 mg/dL. Monotone model is the default for what-ifs.
- Every number from a synthetic cohort is labelled SIMULATED.

## Competition rules (from the organizers' page)
Required: one condition (ours T2D); fusion of static EHR-like data + dynamic wearable data (synthetic or open allowed); a model that predicts an adverse event (e.g. glucose spike 2 h ahead); conceptual doctor dashboard.
README must contain: team details and college/incubator; project title; problem statement and healthcare use case; technical stack, AI/ML model and framework; video of AT LEAST 20 minutes (PowerPoint recording or unlisted YouTube); open-source licence; architecture diagram (PDF/PPT); presentation (PDF/PPT) with details and outcome.
Folder/repo name: `TeamName_CollegeName`; repo must be PUBLIC; the submission form takes only the leader's name, phone, email and repo link.
Phases: top 5 chosen from the submission by a jury; top 10 present virtually; final top 5 at the Bengaluru summit.

## Done
Parts 1-7 (data, features, fusion, mechanistic twin, personalization, uncertainty, what-if), Part 9 Streamlit app (9 pages, purple/damson style), Part 8a transfer code (`src/transfer`), Part 8b India cohort generator + stress test code (`src/india`, D-062).
Part 8a and 8b are written and unit-tested but NOT yet run on real data.

## Do next (in order)
1. Run `python -W ignore -u -m src.transfer.run` then `python -m src.transfer.report`; read `reports/metrics/transfer.json`; fix errors.
2. Run `python -W ignore -u -m src.india.stress` (writes `reports/metrics/india_stress.json`); interpret honestly (negative results are reported as found).
3. Part 8c: federated learning simulation (manual FedAvg / FedProx in torch; clinics IID, non-IID by HbA1c, by dataset; compare centralized and local-only; optional DP; DPDP mapping). Write the decision (D-063) BEFORE running.
4. Wearables: simulated sleep + HRV channel (labelled synthetic); ablation CGM only / + wearable / + static with intervals.
5. Meals: cite carbohydrate and glycaemic load of the 16 Indian presets from IFCT 2017 and the International GI Tables; keep the response labelled model-based.
6. Dashboard: mmol/L toggle, time-in-range day view, risk-sorted triage list, Indian meal plate picker, tier toggle, guided demo mode. Needs the Happiest Health purple style (ask for a website screenshot if needed).
7. Deliverables: model card, datasheet, limitations, DPDP mapping, architecture diagram PDF, 12-15 slide presentation, final README (see `README_DRAFT.md` if present), 20+ minute video script, licence, rename repo to `TeamName_CollegeName`, make public, clean-machine and Docker test, merge `wip-part9` into main.
8. Pin streamlit/plotly versions in requirements (`pip freeze`).

## Before every push
`python -m black src tests data/scripts` then `python -m ruff check src tests data/scripts --fix` then `python -m pytest -q`. CI fails on lint otherwise (and the owner gets emails). Tests that need `data/processed` must skip when it is absent.

## Open decisions for the owner
ACI decision (D-049); D-055 owner decisions; thesis rewording proposal; whether to apply for AI-READI (not required).
