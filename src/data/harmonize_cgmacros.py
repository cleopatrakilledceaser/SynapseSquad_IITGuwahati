"""Harmonize CGMacros (time series, meals, static) into the common schema (Part 1 Step C)."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.cgm_reconstruct import detect_native, make_grid, to_grid

DATASET = "cgmacros"
LB_TO_KG = 0.45359237
IN_TO_CM = 2.54
HBA1C_A, HBA1C_B = 10.929, 2.15  # IFCC (mmol/mol) = 10.929 * (NGSP% - 2.15)
DROP_COLUMNS = ("Unnamed: 0", "RecordIndex", "Image path")


def participant_members(z: zipfile.ZipFile) -> dict[str, str]:
    """Map raw participant id ('001') to the per-participant CSV member in the zip."""
    out = {}
    for n in z.namelist():
        m = re.search(r"CGMacros-(\d{3})\.csv$", n)
        if m and "__MACOSX" not in n:
            out[m.group(1)] = n
    return dict(sorted(out.items()))


def read_participant(z: zipfile.ZipFile, member: str) -> pd.DataFrame:
    """Read one raw CSV and normalise its 9 column variants (strip names, drop index columns)."""
    df = pd.read_csv(io.BytesIO(z.read(member)), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df["t"] = pd.to_datetime(df["Timestamp"])
    return df.sort_values("t").reset_index(drop=True)


def normalize_meal_type(s: pd.Series) -> pd.Series:
    """Case-fold meal labels; every snack variant ('Snack', 'Snacks', 'snack 1') -> 'snack'."""
    x = s.astype("string").str.strip().str.lower()
    x = x.mask(x.str.startswith("snack", na=False), "snack")
    return x.where(x.isin(["breakfast", "lunch", "dinner", "snack"]), "unknown")


def minute_activity_level(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.Series, str]:
    """Per-minute activity level 0..3 from METs (x10) or Fitbit Intensity, plus its source."""
    if "METs" in df.columns and df["METs"].notna().any():
        mets = pd.to_numeric(df["METs"], errors="coerce") / cfg["activity"]["mets_scale"]
        cuts = [-np.inf, *cfg["activity"]["mets_cutpoints"], np.inf]
        lvl = pd.cut(mets, cuts, labels=False, right=False).astype("float64")
        return lvl, "mets"
    if "Intensity" in df.columns:
        return pd.to_numeric(df["Intensity"], errors="coerce").astype("float64"), "intensity"
    return pd.Series(np.nan, index=df.index), "none"


def trailing_mean(t: pd.Series, x: pd.Series, grid: pd.DatetimeIndex, step: int) -> pd.Series:
    """Mean of ``x`` over each grid window (g - step, g]: uses only samples at or before g."""
    s = pd.Series(x.to_numpy(), index=pd.DatetimeIndex(t)).dropna()
    r = s.resample(f"{step}min", label="right", closed="right").mean()
    return r.reindex(grid)


def build_timeseries(df: pd.DataFrame, pid: str, step: int, cfg: dict[str, Any]) -> pd.DataFrame:
    """Build the harmonized grid for one participant at ``step`` minutes."""
    c = cfg["cgm"]
    t = df["t"]
    minutes = ((t - t.iloc[0]).dt.total_seconds() // 60).astype(int).to_numpy()
    grid = make_grid(t.iloc[0], t.iloc[-1], step)
    out: dict[str, Any] = {}
    dex = pd.to_numeric(df["Dexcom GL"], errors="coerce").to_numpy()
    nat = detect_native(
        minutes, dex, c["dexcom_period_min"], c["max_gap_min"], c["kink_tol"], c["integer_tol"]
    )
    g = to_grid(
        t[nat], dex[nat], grid, step, c["max_gap_min"], c["session_gap_min"], c["early_wear_hours"]
    )
    lib = pd.to_numeric(df["Libre GL"], errors="coerce").to_numpy()
    lnat = detect_native(
        minutes, lib, c["libre_period_min"], c["max_gap_min"], c["kink_tol"], c["integer_tol"]
    )
    lg = to_grid(t[lnat], lib[lnat], grid, step, c["libre_period_min"], 10**9, 0)
    hr = trailing_mean(t, pd.to_numeric(df.get("HR"), errors="coerce"), grid, step)
    lvl, src = minute_activity_level(df, cfg)
    act = trailing_mean(t, lvl, grid, step)
    out = g.drop(columns=["source_time"]).copy()
    out["glucose_secondary"] = lg["glucose"].to_numpy()
    out["hr"] = hr.to_numpy()
    out["hr_missing"] = out["hr"].isna()
    out["activity_intensity"] = act.to_numpy()
    out["activity_source"] = src
    out["activity_missing"] = out["activity_intensity"].isna()
    out["rest_proxy"] = rest_proxy(out, step, cfg)
    out["patient_id"] = f"{DATASET}_{pid}"
    out["dataset"] = DATASET
    out["period_id"] = f"{DATASET}_{pid}"
    out["minutes_since_start"] = (out["t"] - out["t"].iloc[0]).dt.total_seconds() / 60
    out["days_since_start"] = out["minutes_since_start"] / 1440
    return out


def rest_proxy(g: pd.DataFrame, step: int, cfg: dict[str, Any]) -> pd.Series:
    """Night-rest proxy (NOT sleep): clock night, sedentary, HR below a causal personal quantile.

    The HR threshold is the expanding (past-only) quantile of the participant's own HR, so the
    proxy at t uses no data after t. NA until one day of HR has been seen or when HR is missing.
    """
    rp = cfg["rest_proxy"]
    win = max(1, rp["window_min"] // step)
    hr_win = g["hr"].rolling(win, min_periods=win).mean()
    thr = g["hr"].expanding(min_periods=24 * 60 // step).quantile(rp["hr_quantile"])
    act_win = g["activity_intensity"].rolling(win, min_periods=win).max()
    h = g["t"].dt.hour
    night = (h >= rp["night_start_hour"]) | (h < rp["night_end_hour"])
    val = night & (hr_win <= thr) & (act_win.fillna(np.inf) < 1)
    known = hr_win.notna() & thr.notna()
    return val.astype("boolean").where(known, pd.NA)


def build_meals(df: pd.DataFrame, pid: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per logged meal; raw amount kept as logged, values >100 % flagged (never altered)."""
    m = df.loc[df["Meal Type"].notna()].copy() if "Meal Type" in df.columns else df.iloc[0:0]
    amount = pd.to_numeric(m.get("Amount Consumed"), errors="coerce") if len(m) else None
    out = pd.DataFrame(
        {
            "t": m["t"],
            "meal_type": normalize_meal_type(m["Meal Type"]),
            "kcal": pd.to_numeric(m.get("Calories"), errors="coerce"),
            "carbs_g": pd.to_numeric(m.get("Carbs"), errors="coerce"),
            "protein_g": pd.to_numeric(m.get("Protein"), errors="coerce"),
            "fat_g": pd.to_numeric(m.get("Fat"), errors="coerce"),
            "fiber_g": pd.to_numeric(m.get("Fiber"), errors="coerce"),
        }
    )
    if amount is None or "Amount Consumed" not in df.columns:
        out["amount_pct_raw"] = np.nan
    else:
        out["amount_pct_raw"] = amount.to_numpy()
    out["amount_missing"] = out["amount_pct_raw"].isna()
    out["amount_gt100"] = out["amount_pct_raw"] > cfg["meals"]["amount_plausible_max"]
    out["macros_available"] = True
    out["patient_id"] = f"{DATASET}_{pid}"
    out["dataset"] = DATASET
    out["period_id"] = f"{DATASET}_{pid}"
    out["minutes_since_start"] = (out["t"] - df["t"].iloc[0]).dt.total_seconds() / 60
    return out


def build_static(z: zipfile.ZipFile, cfg: dict[str, Any]) -> pd.DataFrame:
    """Static table from bio.csv: HbA1c in % (+ mmol/mol), kg, cm, BMI recomputed, status."""
    name = next(n for n in z.namelist() if n.lower().endswith("bio.csv") and "__MACOSX" not in n)
    b = pd.read_csv(io.BytesIO(z.read(name)))
    b.columns = [c.strip() for c in b.columns]
    num = lambda c: pd.to_numeric(b[c], errors="coerce")  # noqa: E731
    a1c = num("A1c PDL (Lab)")
    weight_kg = num("Body weight") * LB_TO_KG
    height_cm = num("Height") * IN_TO_CM
    clin = cfg["clinical"]
    status = np.select(
        [a1c >= clin["hba1c_t2d_pct"], a1c >= clin["hba1c_prediabetes_pct"]],
        ["t2d", "prediabetes"],
        default="healthy",
    )
    ldl = num("LDL (Cal)").mask(lambda s: s == 800)  # dictionary: 800 = calculation error
    return pd.DataFrame(
        {
            "patient_id": [f"{DATASET}_{int(s):03d}" for s in b["subject"]],
            "dataset": DATASET,
            "diabetes_status": status,
            "age_years": num("Age"),
            "sex": b["Gender"].astype("string").str.strip().str.upper().str[0],
            "ethnicity": b["Self-identify"].astype("string").str.strip(),
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": weight_kg / (height_cm / 100) ** 2,
            "bmi_reported": num("BMI"),
            "hba1c_pct": a1c,
            "hba1c_mmol_mol": HBA1C_A * (a1c - HBA1C_B),
            "fasting_glucose_mgdl": num("Fasting GLU - PDL (Lab)"),
            "fasting_insulin": num("Insulin"),
            "triglycerides_mgdl": num("Triglycerides"),
            "cholesterol_mgdl": num("Cholesterol"),
            "hdl_mgdl": num("HDL"),
            "ldl_mgdl": ldl,
            "n_periods": 1,
            "synthetic": False,
        }
    )


def harmonize(zip_path: Path, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Return {'ts5','ts15','meals','static'} frames for CGMacros."""
    z = zipfile.ZipFile(zip_path)
    ts5, ts15, meals = [], [], []
    for pid, member in participant_members(z).items():
        df = read_participant(z, member)
        ts5.append(build_timeseries(df, pid, cfg["grid"]["primary_min"], cfg))
        ts15.append(build_timeseries(df, pid, cfg["grid"]["common_min"], cfg))
        meals.append(build_meals(df, pid, cfg))
    return {
        "ts5": pd.concat(ts5, ignore_index=True),
        "ts15": pd.concat(ts15, ignore_index=True),
        "meals": pd.concat(meals, ignore_index=True),
        "static": build_static(z, cfg),
    }
