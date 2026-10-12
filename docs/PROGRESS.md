# PROGRESS

## Part 1: Scaffold + Data
| Step | Status |
|---|---|
| A. Scaffold, patient-level splits, tests, git init | ✅ done, approved (commit 4ac154e) |
| B-pre. Library compatibility test | ⚠️ done: torch + opacus BLOCKED by Windows Application Control (DECISIONS D-008). STOPPED for owner decision |
| B. Verify datasets (CGMacros + secondary) | ⏸ partially started, see "Step B work already done" below |
| C. Harmonization + QC | ⏳ blocked on B + data download |
| D. EDA + data card + integrity tests | ⏳ blocked on C |

## Environment (dev box)
- Project: `C:\Projects\glucotwin`, venv `.venv` (Python 3.11.9, installed via winget/python.org installer).
- GNU Make 4.4.1 via `winget install ezwinports.make`.
- **Docker: Dockerfile written but NOT built/verified locally** (Docker not installed).
- CI (GitHub Actions) defined but not yet run: no remote repository configured.
- Windows Application Control blocks pandas 3.x / scikit-learn 1.9 DLLs, so older versions are
  pinned (DECISIONS D-006).
- `make` was added to the user PATH by winget; open a new shell (or restart VS Code) for it to be found.

## Verified at end of Step A
`make lint` ✅ · `make test` ✅ 14 passed · `make smoke` ✅

## Step B work already done (started in parallel before the pre-step results came in)
Not yet written into data_card.md; findings so far, with sources:
- CGMacros v1.0.0 (https://physionet.org/content/cgmacros/1.0.0/): Open Access, CC BY-NC-SA 4.0
  (LICENSE.txt in the release is 0 bytes, so the license comes from the project page), 627.1 MB zip,
  45 participants (15 healthy, 16 prediabetes, 14 T2D), about 10 days each, Libre Pro 15-min +
  Dexcom G6 Pro 5-min, Fitbit Sense (HR, calories, METs), dates shifted by +/-N days (365<N<720).
  The data dictionary (DataDictionary_Bio.csv) labels HbA1c as "mmol/mol" with range 4.6-8.5,
  while the project page defines groups by HbA1c in %.
- CGMacros zip download from the PhysioNet open S3 mirror is PAUSED at about 187/627 MB. Resume
  with `curl -C -` (same URL) and verify against SHA256SUMS.txt.
- Shanghai T1DM/T2DM (figshare collection 6310860, article 21600933, CC BY 4.0, no login) was
  downloaded to data/raw/shanghai/ and MD5 verified. Counted from files: T2DM 100 patients /
  109 periods; T1DM 12 patients / 16 periods; CGM every 15 min; T2DM period length 2.6/12.7/13.9 days
  (min/median/max); diet is free text; no wearable data.
- BIG IDEAs v1.1.3 (PhysioNet): Open Access, ODC-By 1.0, 34.1 GB uncompressed / 4.7 GB zip,
  n=16 (normal glucose range), Dexcom G6 5-min + Empatica E4, food logs with macros.
- AI-READI v3.0.0: 2280 participants, controlled access through FAIRhub, link typically within
  2 weeks (aireadi.org/faq). Treated as future work.

## Next steps
1. Owner decides torch/opacus workaround (D-008 options).
2. Resume Step B: finish CGMacros download + SHA256, run the 5 file checks, write download scripts,
   data_card.md, then the final report.
