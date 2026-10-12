# PROGRESS

## Part 1: Scaffold + Data
| Step | Status |
|---|---|
| A. Scaffold, patient-level splits, tests, git init | ✅ done (awaiting owner approval) |
| B. Verify datasets (CGMacros + secondary) | ⏳ not started — wait for owner go-ahead |
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

## Next steps
1. Owner approves Step A.
2. Step B: verify CGMacros page/license/files, then secondary datasets; write download scripts.
