# Data card (Part 1B: verification, before harmonization)

Every fact has a source. Sources: **[P]** = project page; **[F]** = checked from the downloaded
files with `data/scripts/cgmacros_checks.py` (or the probe noted next to the fact). Raw data is
never committed or redistributed. Verified 2026-10-13.

## 1. CGMacros v1.0.0 (primary)
Project page [P]: https://physionet.org/content/cgmacros/1.0.0/ (DOI 10.13026/3z8q-x658)

| Item | Value | Source |
|---|---|---|
| Access | "Open Access". "Anyone can access the files, as long as they conform to the terms of the specified license." No login was needed (downloaded anonymously) | [P], [F] |
| License | **CC BY-NC-SA 4.0** (non-commercial, share-alike). `LICENSE.txt` in the release is **0 bytes** (sha256 e3b0c442... = empty file), so the license comes from the project page | [P], [F] |
| Citation | Cite the PhysioNet record, Das et al. (Sci Data), and the standard PhysioNet citation | [P] |
| Size | Zip 626.7 MB (657,187,340 B); 627.9 MB uncompressed total; all 7 files match `SHA256SUMS.txt` | [P], [F] |
| Participants | 45: 15 healthy (HbA1c <5.7%), 16 prediabetes (5.7-6.4%), 14 T2D (>6.4%). Folder IDs 001-049 with **024, 025, 037, 040 absent** | [P], [F] |
| Recruitment | Sansum Diabetes Research Institute, Santa Barbara CA, 2021-2024; ages 18-69; BMI 21-46 | [P] |
| Days per participant | Page: "approximately ten days". Files: **min 7.2 / median 10.6 / max 19.2** days (first to last row) | [P], [F] |
| CGM devices | Abbott FreeStyle Libre Pro (upper arm, 15-min sampling) + Dexcom G6 Pro (abdomen, 5-min sampling), both blinded | [P] |
| CGM in files | 1-min rows. **Both CGM columns are linearly interpolated to 1 min.** Native cadence recovered from the interpolation kinks: Dexcom 5 min, Libre 15 min (all 45 participants). Glucose range 40-400 (Libre max 405) mg/dL | [F] `gt_native_probe` |
| Wearable | Fitbit Sense: `HR` (bpm, 1-min average) in all 45; `Calories (Activity)` in 44 (missing in 007); `METs` (x10) in 34. **The other 11 (011, 026, 027, 031-035, 038, 042, 046) have `Intensity` instead of METs.** `Steps` exists only in 007. No sleep column anywhere | [P], [F] |
| Meals | Logged with MyFitnessPal, meal photos via WhatsApp. Breakfast = protein shakes, lunch = Chipotle, dinner = free choice. Fields: `Meal Type`, `Calories`, `Carbs`, `Protein`, `Fat`, `Fiber`, `Amount Consumed` (%), `Image path`. Meal time = photo timestamp | [P], [F] |
| Static (bio.csv, 45x24) | Age, Gender, BMI, Body weight (lb), Height (in), Self-identify, A1c, Fasting GLU, Insulin, Triglycerides, Cholesterol, HDL, Non HDL, LDL (Cal), VLDL (Cal), Cho/HDL Ratio, lab collection time, 3 fingerstick glucose readings + times | [F] |
| Other | microbes.csv (1,979 bacteria present/absent), gut_health_test.csv (22 Viome scores), meal photos (3,500+ zip entries) | [P], [F] |
| De-identification | "dates ... have been time shifted by +/- N days (365<N<720)". Dictionary: "Randomly shifted time stamps" | [P], DataDictionary_CGMacros-00X.csv |

### 1.1 Check 1: is time of day preserved? **Yes** [F]
The shift is by whole days, so clock time survives:
- Meal starts cluster by meal type (clock hours, 10th/50th/90th percentile):
  - Breakfast 6.6 / **8.2** / 10.0
  - Lunch 11.5 / **12.6** / 14.2
  - Dinner 16.6 / **19.0** / 21.1
  - Snack 14.9 / 17.7 / 22.1
- Day/night heart rate: mean HR at 10:00-20:00 is above 01:00-05:00 in **44/44** participants with
  HR in both windows (45 have HR; one has no 01:00-05:00 readings), median +17.7 bpm. Cohort HR by hour falls from 72 bpm at midnight to a low of 66 at
  03:00-04:00, then plateaus at 84-85 from 09:00 to 19:00.

Time-of-day features (circadian encoding, night windows) are valid. Calendar dates and weekday
are **not** (shifted 1-2 years), so day-of-week features must not be used.

### 1.2 Check 2: units [F]
- **HbA1c is %, not mmol/mol**, despite the dictionary label. `A1c PDL (Lab)` ranges 4.6-8.5.
  Binning it with the page's % cut-offs gives exactly 15 / 16 / 14 (<5.7 / 5.7-6.4 / >6.4), which
  matches the page's group counts. In mmol/mol, 4.6-8.5 would be physiologically impossible.
  Harmonized: `hba1c_pct`, plus `hba1c_mmol_mol = 10.929 * (pct - 2.15)` (IFCC-NGSP master equation).
- Weight is lb (116.8-284.6) and height is in (59-72). BMI recomputed with 1 lb = 0.45359237 kg
  and 1 in = 0.0254 m matches the reported BMI exactly (median and max |diff| = 0.00), which confirms
  both units. Harmonized: `weight_kg`, `height_cm`.
- Glucose is mg/dL (40-400; Libre max 405 in participant 030, above the dictionary's stated max of 400).
- Lipids are mg/dL; insulin is µU/mL (dictionary). The dictionary flags LDL = 800 and VLDL = 400 as
  error codes, so set them to NaN in harmonization.

### 1.3 Check 3: coverage per participant [F]
- CGM: every participant has both Libre and Dexcom; none lacks a CGM.
- Dexcom fraction NaN per file: median 3.7%, max 24%. Libre max 1.5%.
- **Interpolation bridges long gaps.** The native-point probe found Dexcom gaps up to 13,545 min
  (participant 018) and 2,431 min (026), and Libre up to 13,512 min (018) and 944 min (015). These
  were filled by straight lines, so harmonization must mark rows inside a native gap > 30 min as
  missing, not trust them.
- Overlap of Dexcom and Libre spans: 7.2 / 9.9 / 19.1 days (min/median/max).
- HR: present in all 45; row coverage median 93%, min 48% (**032** < 50%).
- METs missing in 11 participants (they have `Intensity` instead, see table).
- Meals: present for all 45 (17-64 meal rows each).
- Duplicate timestamps 0; backwards time steps 0.
- **Schema varies across files** (9 distinct column sets):
  - `Unnamed: 0` index column in 9 files;
  - `Amount Consumed` has a trailing space in 1 file and is **missing in 028 and 029**;
  - `RecordIndex` and `Steps` exist only in 007; `Sugar` only in 013.
  - The loader must normalise column names and treat absent columns as missing.

### 1.4 Check 4: sleep [F]
There is **no sleep data** (no sleep-like column in any file; none listed on the page). Because
time of day is preserved (1.1), a proxy is feasible:
- Night-rest window = 22:00-07:00 clock time with sustained low HR (participant's rolling HR
  below their own 25th percentile) and low METs/Intensity.
- Flag it `rest_proxy` and never label it as sleep.

### 1.5 Check 5: meal logging [F]
- Meal rows: 1,706 in total. Calories, Carbs, Protein and Fat are never missing; Fiber is missing once.
- **Meal Type labels are inconsistent**: breakfast/Breakfast, lunch/Lunch, dinner/Dinner, snack /
  Snack / Snacks / "snack 1". They need case-folding and mapping.
- **Snacks are logged**: 343 snack rows across 36/45 participants.
- 3.54 meals per participant-day (median 3, max 10). Days with fewer than 3 logged meals exist,
  which points to **some unlogged intake** (or partial first/last days).
- `Amount Consumed` (% of the meal eaten) ranges 0-900. Values > 100 appear in 16 participants
  (e.g. 007: dinners at 200/300/400). These are probably multiple servings, but this is
  **unconfirmed**. Amount is missing only where the column is absent: 028 (34 meals) and 029 (30).
- Implication: "time since last meal" is only reliable as "time since last *logged* meal". The
  protocol asked for ≥3 h gaps with only water or black coffee between breakfast and lunch and
  between lunch and dinner [P], so those windows are the most trustworthy. Evening and snacking
  periods are less reliable.

### 1.6 Reference CGM signal: **Dexcom G6 Pro** (decision D-010)
- Native 5-min cadence vs Libre's 15 min, which suits +30/60/120 min forecasting better.
- Its 5-min gap structure is recoverable.
- It is the same device family as BIG IDEAs (Dexcom G6), which helps transfer.
- Libre stays as a secondary signal: used for cross-sensor sanity checks, and as a flagged
  fallback where Dexcom has gaps. Harmonization must recover native points rather than use the
  1-min interpolated values.


## 2. Shanghai T1DM / T2DM (secondary, transfer)
Sources: Zhao et al., Sci Data 10:35 (2023), https://doi.org/10.1038/s41597-023-01940-7 **[A]**;
figshare collection 6310860, article 21600933 (API) **[G]**; files [F].

| Item | Value | Source |
|---|---|---|
| Access | Open, no login (figshare download link) | [G], [F] |
| License | CC BY 4.0 | [G] |
| Size | `diabetes_datasets.zip` 3.6 MB, MD5 OK | [G], [F] |
| Version | figshare article v5 (2023-10-30); `note.txt` says this version adds 188 CGM readings to T2DM | [G], [F] |
| Participants | T1DM **12 patients / 16 periods**; T2DM **100 patients / 109 periods** (92 with 1 period, 7 with 2, 1 with 3) | [F] |
| Count conflict | Paper text: T2DM "94 patients with 1 period ..., 6 patients with 2 periods, and 1 patient with 3" = 101 patients. **The files show 92/7/1 = 100.** The files win | [A] vs [F] |
| Status | All T2DM summary rows are `T2DM`. **No prediabetes or healthy participants** | [F] |
| CGM | FreeStyle Libre H, **15-min** (modal step 15 min in all 109 T2DM periods) | [A], [F] |
| Days per period (T2DM) | min 2.6 / median 12.7 / max 13.9 | [F] |
| Wearable | **None** (no HR, steps or activity) | [A], [F] |
| Diet | Free-text dishes with grams (`Dietary intake`, plus a Chinese-language column). **No macros.** 3,808 non-empty entries, some "data not available" | [F] |
| Medication | Insulin s.c./i.v., CSII bolus/basal, non-insulin agents (per-row); agents also in summary | [A], [F] |
| Static (summary, 109 x 33) | age 22-97 (median 63); sex 50 F / 59 M (coded 1/2); height m; weight kg; BMI 17.1-36.7; diabetes duration; complications; comorbidities; FPG; 2-h PPG; C-peptide; insulin; **HbA1c in mmol/mol (23.5-144.8, i.e. about 4.3-15.4%; 8 rows "/")**; glycated albumin; lipids mmol/L; creatinine; eGFR; uric acid; BUN; hypoglycemia y/n | [F] |
| Units to harmonize | HbA1c mmol/mol -> % (`pct = mmol/mol / 10.929 + 2.15`); lipids mmol/L -> mg/dL (cholesterol x 38.67, TG x 88.57); height m -> cm | [F] |

## 3. BIG IDEAs Lab Glycemic Variability and Wearable Device Data v1.1.3 (secondary)
Source: https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/ **[B]**; Demographics.csv [F].
Only metadata was downloaded; the 4.7 GB zip was deliberately not fetched.

| Item | Value | Source |
|---|---|---|
| Access | Open Access, no login | [B] |
| License | Open Data Commons Attribution License v1.0 (ODC-By) | [B] |
| Size | Zip 4.7 GB; 34.1 GB uncompressed. Participant 001: ACC 838 MB, BVP 1.3 GB, Dexcom 151 KB, EDA 89 MB, HR 12 MB, IBI 10 MB, TEMP 75 MB, Food_Log 6.7 KB | [B] |
| Participants | 16, "elevated blood glucose in the normal range"; screened for HbA1c in the "high normal and prediabetic range", ages 35-65 | [B] |
| HbA1c (Demographics.csv) | 5.3-6.4%: **9/16 ≥ 5.7 (prediabetes range)**, 7 < 5.7, no T2D; 7 M / 9 F | [F] |
| Days | 8-10 | [B] |
| CGM | Dexcom G6, 5-min, mg/dL | [B] |
| Wearable | Empatica E4: PPG/BVP 64 Hz, HR (1/s), IBI, EDA 4 Hz, skin temp 4 Hz, 3-axis ACC | [B] |
| Meals | Food_Log: date, time_of_day, time_begin, time_end, logged_food, amount, unit, searched_food, calorie, total_carb, dietary_fiber, sugar, protein, total_fat. Standardized breakfast every other day | [B] |
| Static | Only gender + HbA1c in the files (no age, BMI or other labs per participant) | [F] |
| Time | "all data is time-shifted (by date)"; v1.1.3 "Updated misaligned food log dates" | [B] |

## 4. AI-READI (future work)
- v3.0.0: 2,280 participants, 15 modalities, 3.82 TB [https://docs.aireadi.org/].
- Controlled set via the FAIRhub access page. After the access process, the download link
  typically arrives "within 2 weeks" [https://aireadi.org/faq].
- Medication data is protected and "not currently available for general access" [FAQ].
- **Not usable now**: requires an access agreement, which we will not bypass.
- The step-by-step access requirements could not be read as static text (the docs pages render
  with JavaScript); the owner must read FAIRhub directly.

## 5. Other openly accessible CGM datasets checked
- The Glucose-ML index (https://github.com/Augmented-Health-Lab/Glucose-ML-Project,
  `5_Tables-from-paper/Table_2.csv`) lists open T2D/prediabetes CGM sets: Hall 2018 (57;
  T2D/PreD/ND), Colas 2019 (208; T2D/ND), Park 2025 (38; T2D/PreD/ND).
- **None of these has been verified from its own source.** This is secondary-source information
  only, so they go to future work.
- PhysioCGM (10 participants, T1D, ECG/PPG; github.com/PSI-TAMU/PhysioCGM) is T1D-only and out of
  scope.

## 6. Harmonized data (Part 1 Steps C-D)
Built by `make data` into `data/processed/` (gitignored); `manifest.json` records the git hash,
config and raw-zip hashes. Schema: `src/data/schema.py`. Decisions: D-015 to D-020.

| File | Rows | Patients | Content |
|---|---|---|---|
| `cgmacros_ts5.parquet` | 141,300 | 45 | CGMacros, 5-min grid (primary) |
| `common_ts15.parquet` | 159,621 | 145 (45 + 100) | Both cohorts, 15-min grid (transfer) |
| `meals.parquet` | 4,769 | 145 | 1,706 CGMacros meals with macros; 3,063 Shanghai meals (timing + grams only) |
| `static.parquet` | 145 | 145 | Units: HbA1c %, kg, cm, mg/dL; status; Shanghai medication flags |

QC after harmonization (CGM = Dexcom for CGMacros, Libre H for Shanghai;
`reports/metrics/qc_summary.json`). "Within wear" = between a period's first and last real
reading. Most CGMacros "missing" is the Fitbit/Libre tail after the 10-day Dexcom sensor ended.

| Cohort / status | Patients (periods) | Rows (grid) | % real | % carried ≤ 30 min | % missing | % missing within wear | % HR | % activity |
|---|---|---|---|---|---|---|---|---|
| CGMacros healthy | 15 | 48,418 (5-min) | 87.0 | 0.6 | 12.4 | 6.2 | 81.7 | 90.2 |
| CGMacros prediabetes | 16 | 49,151 (5-min) | 88.9 | 1.7 | 9.4 | 1.5 | 90.4 | 94.6 |
| CGMacros T2D | 14 | 43,731 (5-min) | 90.9 | 0.5 | 8.6 | 0.2 | 89.1 | 95.9 |
| Shanghai T2D | 100 (109) | 112,490 (15-min) | 99.98 | 0.01 | 0.02 | 0.02 | 0 | 0 |

- Per-patient CGMacros:
  - Missing CGM median 8.2% (max 49.4%, participant 018).
  - Within-wear missing is highest in 018 (49.2%), 026 (14.8%), 007 (8.8%); everyone else is ≤ 1%.
  - HR coverage median 91.7% (min 48.1%, 032).
- Full per-patient tables: `reports/metrics/per_patient_coverage_*.csv`.
- Validation of the CGM reconstruction (all 45): `reports/metrics/cgm_reconstruction_validation.json`.

Figures (`reports/figures/`, 300 dpi, Okabe-Ito palette):
- `glucose_distribution.png`
- `time_in_range.png`
- `meal_responses.png`
- `missingness_map.png`
- `days_and_wearable_coverage.png`
- `cgm_reconstruction_before_after.png`

## 7. Synthetic data
None yet. Any synthetic cohort (e.g. India) will be labeled `synthetic=True` and never mixed into a
real test set.

