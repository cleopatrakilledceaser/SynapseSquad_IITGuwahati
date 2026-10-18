"""Export the GlucoTwin synthetic Indian T2D dataset (D-067). ALL DATA ARE SYNTHETIC.

Run: ``python -W ignore -u -m src.india.export`` (a few minutes). Writes CSV files and a
DATASHEET.md under data/processed/india/synthetic_dataset/. Nothing is committed to git; the
dataset is reproduced from the seed in configs/india.yaml.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.india import cohort as ic
from src.india import ehr
from src.utils.config import PROJECT_ROOT, load_config

OUT = PROJECT_ROOT / "data" / "processed" / "india" / "synthetic_dataset"


SENSOR_RANGE = (40.0, 400.0)  # Dexcom reporting range (configs/data.yaml)
MAX_PATIENT_MEAN = 300.0  # mg/dL; a profile above this needs urgent care, not a routine clinic list


def plausible(
    c: dict[str, pd.DataFrame], n_keep: int, icfg: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """Apply the CGM reporting range and keep the first ``n_keep`` patients with a plausible mean.

    HbA1c and diabetes status are recomputed from the sensor-range glucose so the profile stays
    consistent with what a CGM would have reported.
    """
    g = c["grid"].copy()
    g["glucose"] = g["glucose"].clip(*SENSOR_RANGE)
    means = g.groupby("patient_id", sort=False)["glucose"].mean()
    keep = list(means.index[means <= MAX_PATIENT_MEAN])[:n_keep]
    out = {k: v[v["patient_id"].isin(keep)].reset_index(drop=True) for k, v in c.items()}
    out["grid"] = g[g["patient_id"].isin(keep)].reset_index(drop=True)
    s = icfg["static"]
    st = out["static"].set_index("patient_id")
    a1c = (means[keep] - s["eag_intercept"]) / s["eag_slope"]
    st["hba1c_pct"] = a1c
    st["diabetes_status"] = pd.cut(
        a1c, s["hba1c_bins"], labels=s["hba1c_labels"], right=False
    ).astype(str)
    out["static"] = st.reset_index()
    return out


def with_history(static: pd.DataFrame, seed: int, ecfg: dict[str, Any]) -> pd.DataFrame:
    """Static profile plus the simulated EHR history (own random stream per patient)."""
    rows = []
    for i, r in enumerate(static.to_dict("records")):
        rows.append(ehr.history(np.random.default_rng([seed, i, 9]), r, ecfg))
    return pd.concat([static.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def tables(c: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The five public tables (no simulator internals such as true glucose or parameters)."""
    g = c["grid"]
    cgm = pd.DataFrame({
        "patient_id": g["patient_id"], "timestamp": g["t"], "glucose_mgdl": g["glucose"].round(1),
        "heart_rate_bpm": g["hr"].round(1), "activity_intensity": g["activity_intensity"].round(3),
        "steps_5min": g["steps"].astype(int),
    })  # fmt: skip
    cols = [
        "patient_id",
        "t",
        "meal_type",
        "preset",
        "carbs_g",
        "protein_g",
        "fat_g",
        "fiber_g",
        "kcal",
    ]
    meals = c["meals"][cols].rename(columns={"t": "timestamp"})
    s = c["sleep"]
    sleep = pd.DataFrame({
        "patient_id": s["patient_id"], "wake_time": s["wake_t"],
        "total_sleep_h": s["sleep_h"].round(2), "efficiency": s["sleep_eff"].round(3),
        "awake_min": s["sleep_awake_min"].round(1), "light_min": s["sleep_light_min"].round(1),
        "deep_min": s["sleep_deep_min"].round(1), "rem_min": s["sleep_rem_min"].round(1),
        "hrv_rmssd_ms": s["hrv_rmssd"].round(1),
    })  # fmt: skip
    return {
        "cgm_wearables": cgm,
        "meals": meals,
        "sleep_nightly": sleep,
        "static_ehr": c["static"].drop(columns=["synthetic"], errors="ignore"),
    }


def datasheet(t: dict[str, pd.DataFrame], ecfg: dict[str, Any]) -> str:
    """Plain-language datasheet: what it is, how it was made, what is assumed, what it is not."""
    g, st = t["cgm_wearables"], t["static_ehr"]
    return f"""# GlucoTwin synthetic Indian T2D dataset (SYNTHETIC)

**Every value in this dataset is simulated. It contains no real person's data.**

| Item | Value |
|---|---|
| Patients | {g['patient_id'].nunique()} virtual adults; diabetes status per patient is in `static_ehr.csv` (derived from simulated HbA1c) |
| Days per patient | {ecfg['days']} |
| Rows | {len(g)} CGM / wearable rows (5-minute), {len(t['meals'])} meals, {len(t['sleep_nightly'])} nights |
| Seed | {ecfg['seed']} (regenerate with `python -m src.india.export`) |
| Mean glucose | {g['glucose_mgdl'].mean():.1f} mg/dL; share above 180: {100 * (g['glucose_mgdl'] > 180).mean():.1f} % |
| HbA1c (from CGM, simulated) | mean {st['hba1c_pct'].mean():.2f} % |

## Tables
- `cgm_wearables.csv`: 5-minute glucose (with sensor noise, limited to the CGM range 40 to 400 mg/dL), heart rate, activity intensity, steps.
- `meals.csv`: Indian meal presets (idli, dosa, poha, dal, rice, roti, ...) with carbohydrate, protein, fat, fibre, kcal.
- `sleep_nightly.csv`: one row per night: total sleep, efficiency, minutes awake / light / deep / REM, HRV (RMSSD).
- `static_ehr.csv`: age, sex, BMI, HbA1c, fasting glucose, diabetes status (derived from the simulated CGM), plus a simulated history: diabetes duration, diagnoses (hypertension, dyslipidaemia, CKD stage 3+, retinopathy, neuropathy, coronary disease), medications, eGFR, family history and genetic-marker dosages (TCF7L2, FTO, SLC30A8). The history is assumed and has NO effect on the simulated glucose.

## How it was made
A mechanistic glucose-insulin simulator (the Part 4 twin) with patient parameters drawn from the
parameters fitted to real CGMacros patients, shifted towards the lower insulin sensitivity
reported for South Asians (JCEM 2001 clamp study; the size of the shift is an assumption).
Meal macros of the presets come from IFCT 2017. Sleep, HRV, steps and stages are generated by
simple assumed distributions.

## What is assumed (not measured)
- Sleep duration, efficiency, stage fractions and HRV distributions, and the link of short sleep to
  lower insulin sensitivity (strength multiplier {ecfg['sleep_mult']:g}, literature-scale).
- Insulin sensitivity shift, basal glucose shift and absorption shift for the Indian setting.
- Meal timing and portion variability.

## Selection rule
Patients whose mean glucose over the study exceeds 300 mg/dL are left out (the simulator can draw
such profiles; they would not be in a routine clinic list). Glucose is clipped to the CGM reporting
range and HbA1c is recomputed from the clipped values.

## Intended use
Demonstration of the GlucoTwin pipeline and dashboard, stress tests of models, and unit tests.
## Not for
Training or validating a clinical model, estimating real effect sizes, or any claim about real
patients. Earlier experiments (README section 9.4) show that adding such synthetic patients to
training did not improve real-patient forecasts.
"""


def main() -> None:
    """Generate the exported synthetic cohort and write its tables and datasheet."""
    icfg, mc = load_config("india"), load_config("mech")
    e = icfg["export"]
    c = ic.generate(icfg, mc, icfg["strengths"][e["strength"]], "exp", e["seed"],
                    int(e["n_patients"] * 1.25), e["days"], sleep_mult=e["sleep_mult"])  # fmt: skip
    c = plausible(c, e["n_patients"], icfg)
    c["static"] = with_history(c["static"], e["seed"], icfg["ehr"])
    t = tables(c)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in t.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"  {name}: {len(df)} rows", flush=True)
    (OUT / "DATASHEET.md").write_text(datasheet(t, e), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
