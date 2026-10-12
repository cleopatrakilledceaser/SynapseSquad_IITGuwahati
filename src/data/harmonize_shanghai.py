"""Harmonize Shanghai T2DM (time series, meals as timing + grams only, static) (Part 1 Step C).

- CGM values are native 15-min Libre H readings (not interpolated), so every row is real.
- Repeated recording periods of the same patient share one ``patient_id`` (the split unit) and
  are distinguished by ``period_id``; ``days_since_start`` restarts per period.
- Diet is free text: only timing, item count and total grams are parsed. No macros are invented.
- No wearable stream: hr/activity are NaN with missing flags; ``rest_proxy`` is NA.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.cgm_reconstruct import make_grid, to_grid

DATASET = "shanghai_t2dm"
HBA1C_A, HBA1C_B = 10.929, 2.15
CHOL_MMOL_TO_MGDL = 38.67
TG_MMOL_TO_MGDL = 88.57
GLUCOSE_MMOL_TO_MGDL = 18.0
INSULIN_PMOL_TO_UU = 1 / 6.0  # 1 uU/mL = 6 pmol/L
GRAMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(g|ml)\b", re.IGNORECASE)


def period_members(z: zipfile.ZipFile) -> list[tuple[str, str, str]]:
    """Return (patient raw id, period id, member) for every T2DM recording period."""
    out = []
    for n in z.namelist():
        m = re.match(r"Shanghai_T2DM/((\d+)_(\d+)_\d+)\.xlsx?$", n)
        if m:
            out.append((m.group(2), m.group(1), n))
    return sorted(out)


def read_period(z: zipfile.ZipFile, member: str) -> pd.DataFrame:
    """Read one period; columns are matched by position/prefix (5 header variants exist)."""
    df = pd.read_excel(io.BytesIO(z.read(member)))
    cols = [str(c).strip() for c in df.columns]
    pick = {
        "t": 0,
        "cgm": next(i for i, c in enumerate(cols) if c.startswith("CGM")),
        "diet": next(i for i, c in enumerate(cols) if c.startswith("Dietary intake")),
    }
    out = pd.DataFrame(
        {
            "t": pd.to_datetime(df.iloc[:, pick["t"]], errors="coerce"),
            "cgm": pd.to_numeric(df.iloc[:, pick["cgm"]], errors="coerce"),
            "diet": df.iloc[:, pick["diet"]].astype("string"),
        }
    )
    return out.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)


def parse_meal_text(text: str) -> tuple[int, float]:
    """Return (n items, total grams) from free text like 'Rice 100 g / Beef 15 g' (ml as g)."""
    amounts = [float(a) for a, _ in GRAMS_RE.findall(text)]
    return len(amounts), float(sum(amounts)) if amounts else float("nan")


def build_period(
    pid: str, period: str, df: pd.DataFrame, step: int, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Harmonized grid and meal rows for one recording period."""
    c = cfg["cgm"]
    real = df.dropna(subset=["cgm"])
    grid = make_grid(df["t"].iloc[0], df["t"].iloc[-1], step)
    g = to_grid(
        real["t"],
        real["cgm"],
        grid,
        step,
        c["max_gap_min"],
        c["session_gap_min"],
        c["early_wear_hours"],
    ).drop(columns=["source_time"])
    g["glucose_secondary"] = np.nan
    g["hr"] = np.nan
    g["hr_missing"] = True
    g["activity_intensity"] = np.nan
    g["activity_source"] = "none"
    g["activity_missing"] = True
    g["rest_proxy"] = pd.array([pd.NA] * len(g), dtype="boolean")
    g["patient_id"] = f"{DATASET}_{pid}"
    g["dataset"] = DATASET
    g["period_id"] = f"{DATASET}_{period}"
    g["minutes_since_start"] = (g["t"] - g["t"].iloc[0]).dt.total_seconds() / 60
    g["days_since_start"] = g["minutes_since_start"] / 1440

    d = df.dropna(subset=["diet"])
    d = d[~d["diet"].str.contains("not available", case=False, na=False)]
    parsed = d["diet"].map(parse_meal_text)
    meals = pd.DataFrame(
        {
            "t": d["t"],
            "meal_type": "unknown",
            "amount_pct_raw": np.nan,
            "amount_missing": True,
            "amount_gt100": False,
            "macros_available": False,
            "food_items": [p[0] for p in parsed],
            "food_grams_total": [p[1] for p in parsed],
            "patient_id": f"{DATASET}_{pid}",
            "dataset": DATASET,
            "period_id": f"{DATASET}_{period}",
        }
    )
    meals["minutes_since_start"] = (meals["t"] - df["t"].iloc[0]).dt.total_seconds() / 60
    return g, meals


AGENT_CLASSES: dict[str, tuple[str, ...]] = {
    "med_insulin": ("insulin", "novolin", "humulin", "gansulin", "degludec", "detemir", "glarg"),
    "med_metformin": ("metformin",),
    "med_sulfonylurea": ("gliclazide", "glimepiride", "gliquidone", "glipizide", "glibenclamide"),
    "med_glinide": ("repaglinide", "nateglinide"),
    "med_agi": ("acarbose", "voglibose", "miglitol"),
    "med_dpp4": ("sitagliptin", "linagliptin", "saxagliptin", "vildagliptin", "alogliptin"),
    "med_sglt2": ("dapagliflozin", "canagliflozin", "empagliflozin"),
    "med_glp1": ("liraglutide", "dulaglutide", "semaglutide", "exenatide"),
    "med_tzd": ("pioglitazone", "rosiglitazone"),
}


def build_static(z: zipfile.ZipFile, periods: pd.DataFrame) -> pd.DataFrame:
    """One row per patient (first period's summary row), units converted, medication flags."""
    s = pd.read_excel(io.BytesIO(z.read("Shanghai_T2DM_Summary.xlsx")))
    s.columns = [str(c).strip() for c in s.columns]
    s["pid"] = s["Patient Number"].astype(str).str.split("_").str[0]
    s["period"] = s["Patient Number"].astype(str).str.split("_").str[1].astype(int)
    first = s.sort_values(["pid", "period"]).groupby("pid", as_index=False).first()
    num = lambda c: pd.to_numeric(first[c], errors="coerce")  # noqa: E731
    a1c_mmol = num("HbA1c (mmol/mol)")
    height_cm = num("Height (m)") * 100
    weight = num("Weight (kg)")
    out = pd.DataFrame(
        {
            "patient_id": DATASET + "_" + first["pid"],
            "dataset": DATASET,
            "diabetes_status": "t2d",
            "age_years": num("Age (years)"),
            "sex": num("Gender (Female=1, Male=2)").map({1: "F", 2: "M"}).astype("string"),
            "ethnicity": pd.Series(pd.NA, index=first.index, dtype="string"),
            "height_cm": height_cm,
            "weight_kg": weight,
            "bmi": weight / (height_cm / 100) ** 2,
            "bmi_reported": num("BMI (kg/m2)"),
            "hba1c_pct": a1c_mmol / HBA1C_A + HBA1C_B,
            "hba1c_mmol_mol": a1c_mmol,
            "fasting_glucose_mgdl": num("Fasting Plasma Glucose (mg/dl)"),
            "fasting_insulin": num("Fasting Insulin (pmol/L)") * INSULIN_PMOL_TO_UU,
            "triglycerides_mgdl": num("Triglyceride (mmol/L)") * TG_MMOL_TO_MGDL,
            "cholesterol_mgdl": num("Total Cholesterol (mmol/L)") * CHOL_MMOL_TO_MGDL,
            "hdl_mgdl": num("High-Density Lipoprotein Cholesterol (mmol/L)") * CHOL_MMOL_TO_MGDL,
            "ldl_mgdl": num("Low-Density Lipoprotein Cholesterol (mmol/L)") * CHOL_MMOL_TO_MGDL,
            "diabetes_duration_years": num("Duration of diabetes (years)"),
            "synthetic": False,
        }
    )
    agents = first["Hypoglycemic Agents"].astype(str).str.lower()
    for col, keys in AGENT_CLASSES.items():
        out[col] = agents.apply(lambda a, k=keys: any(x in a for x in k))
    out["hypoglycemic_agents_raw"] = first["Hypoglycemic Agents"].astype("string").str.strip()
    counts = periods.groupby("patient_id").size()
    out["n_periods"] = out["patient_id"].map(counts).astype("Int64")
    return out


def harmonize(zip_path: Path, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Return {'ts15','meals','static'} frames for Shanghai T2DM (native 15-min grid only)."""
    z = zipfile.ZipFile(zip_path)
    step = cfg["grid"]["common_min"]
    ts, meals, per = [], [], []
    for pid, period, member in period_members(z):
        df = read_period(z, member)
        g, m = build_period(pid, period, df, step, cfg)
        ts.append(g)
        meals.append(m)
        per.append({"patient_id": f"{DATASET}_{pid}", "period_id": f"{DATASET}_{period}"})
    periods = pd.DataFrame(per)
    return {
        "ts15": pd.concat(ts, ignore_index=True),
        "meals": pd.concat(meals, ignore_index=True),
        "static": build_static(z, periods),
    }
