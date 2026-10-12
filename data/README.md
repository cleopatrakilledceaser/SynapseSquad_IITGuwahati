# Data

Raw datasets are **never committed or redistributed** (see `.gitignore` and docs/RULES.md).

- `raw/`       downloaded originals (gitignored)
- `interim/`   intermediate files (gitignored)
- `processed/` harmonized parquet produced by `make data` (gitignored)
- `external/`  any other third-party files (gitignored)

## Download (no login needed for any of these)
```powershell
.venv\Scripts\python data\scripts\download.py cgmacros shanghai bigideas
```
- **cgmacros**: PhysioNet CGMacros v1.0.0 (CC BY-NC-SA 4.0: non-commercial, share-alike), about
  627 MB, fetched from PhysioNet's open S3 mirror and checked against `SHA256SUMS.txt`.
- **shanghai**: figshare 21600933 (CC BY 4.0), 3.6 MB, checked by MD5.
- **bigideas**: PhysioNet BIG IDEAs v1.1.3 (ODC-By). Metadata only by default; add `--full` for
  the 4.7 GB zip (34 GB unpacked).
- **AI-READI**: controlled access via FAIRhub (future work). Do not attempt without an approved
  agreement.

## Verification
`.venv\Scripts\python data\scripts\cgmacros_checks.py` prints the five Part 1B checks (reads the
zip in place). Results are in `docs/data_card.md`.
