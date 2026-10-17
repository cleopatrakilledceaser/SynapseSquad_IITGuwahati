"""The one SYNTHETIC demo patient (D-058): a Part 4 simulator patient turned into model inputs.

The record comes from ``data/synthetic/synthetic_mech_cohort.parquet`` (Part 4 simulator, labelled
synthetic). The simulator has carbs and an activity input but no macros, meal types or heart rate:
- every simulated carb event becomes a meal with ASSUMED protein / fat / fibre per gram of carbs
  (configs/app.yaml demo.synthetic.macro_ratios);
- ``activity_intensity`` is set to the simulator's activity input (an approximation);
- the static profile (HbA1c, fasting glucose) is ASSUMED (configs/app.yaml).
The fold-0 models never saw this patient. The twin is not run for it (no fitted twin: "not
available"); its true simulated Gb is shown on the twin-state page and labelled simulated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.build import period_features
from src.utils.config import PROJECT_ROOT

SYN = PROJECT_ROOT / "data" / "synthetic"


def pick_patient(params: pd.DataFrame, rule: str) -> str:
    """Simulated patient whose true Gb is closest to the cohort median (rule median_gb)."""
    if rule != "median_gb":
        raise ValueError(f"unknown rule {rule}")
    med = params["Gb"].median()
    return str(params.loc[(params["Gb"] - med).abs().idxmin(), "patient_id"])


def build(
    cfg: dict[str, Any], fcfg: dict[str, Any], clip: tuple[float, float], n_h: int
) -> dict[str, Any]:
    """Grid, meals, anchor rows (15-min stride) and the true Gb of the synthetic patient."""
    sc = cfg["synthetic"]
    ts = pd.read_parquet(SYN / "synthetic_mech_cohort.parquet")
    prm = pd.read_parquet(SYN / "synthetic_mech_params.parquet")
    pid = pick_patient(prm, sc["rule"])
    s = ts[ts.patient_id == pid].sort_values("step").reset_index(drop=True)
    t0 = pd.Timestamp(sc["start_datetime"])
    n = len(s)
    g = pd.DataFrame(
        {
            "patient_id": pid,
            "period_id": pid,
            "t": t0 + pd.to_timedelta(s["minutes"].to_numpy(float), unit="m"),
            "days_since_start": s["minutes"].to_numpy(float) / 1440.0,
            "glucose": np.clip(s["glucose"].to_numpy(float), *clip),
            "glucose_is_real": True,
            "glucose_is_carried": False,
            "glucose_missing": False,
            "minutes_since_real": 0.0,
            "hr": np.nan,
            "hr_missing": True,
            "activity_intensity": s["activity_input"].to_numpy(float),
            "activity_missing": False,
            "rest_proxy": False,
        }
    )
    ev = s[s["carbs_g"] > 0]
    r = sc["macro_ratios"]
    meals = pd.DataFrame(
        {
            "patient_id": pid,
            "period_id": pid,
            "t": g["t"].to_numpy()[ev.index.to_numpy()],
            "meal_type": "unknown",
            "carbs_g": ev["carbs_g"].to_numpy(float),
            "protein_g": ev["carbs_g"].to_numpy(float) * r["protein_g"],
            "fat_g": ev["carbs_g"].to_numpy(float) * r["fat_g"],
            "fiber_g": ev["carbs_g"].to_numpy(float) * r["fiber_g"],
            "amount_gt100": False,
            "amount_missing": False,
            "macros_available": True,
        }
    ).reset_index(drop=True)
    meals["kcal"] = 4 * meals["carbs_g"] + 4 * meals["protein_g"] + 9 * meals["fat_g"]
    meals["food_grams_total"] = np.nan
    f = period_features(g, meals, 5, n_h, fcfg)
    f["static_hba1c_pct"] = sc["static_hba1c_pct"]
    f["static_fasting_glucose_mgdl"] = sc["static_fasting_glucose_mgdl"]
    f["cgm_now"] = g["glucose"].to_numpy()
    rows = pd.concat([g[["patient_id", "period_id", "t", "days_since_start"]], f], axis=1)
    rows = rows.loc[:, ~rows.columns.duplicated()]
    stride = 15 // 5
    rows = rows.iloc[::stride].reset_index(drop=True)
    gb = float(prm.loc[prm.patient_id == pid, "Gb"].iloc[0])
    return {"patient_id": pid, "grid": g, "meals": meals, "rows": rows, "gb_true": gb, "n": n}
