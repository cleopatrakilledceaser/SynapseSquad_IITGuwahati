"""Harmonized schema definitions shared by all cohorts (Part 1 Step C).

Time series (one row per patient per grid step), static table (one row per patient) and meal
table (one row per logged meal). Calendar-date features are deliberately absent: dates in both
cohorts are shifted, so only ``t`` (a timestamp kept for ordering), ``days_since_start`` and
clock-time fields derived later are allowed (DECISIONS D-012).
"""

from __future__ import annotations

TIMESERIES_COLUMNS: dict[str, str] = {
    "patient_id": "string",  # "<dataset>_<raw id>"
    "dataset": "string",  # cgmacros | shanghai_t2dm
    "period_id": "string",  # recording period (Shanghai has repeats); == patient_id for CGMacros
    "t": "datetime64[ns]",  # date-shifted timestamp; ordering only, never a feature
    "minutes_since_start": "float64",
    "days_since_start": "float64",
    "glucose": "float64",  # mg/dL; real reading at t, else causal carry (<= max_gap), else NaN
    "glucose_is_real": "bool",  # a real sensor reading sits exactly on this grid step
    "glucose_is_carried": "bool",  # value carried forward from an earlier real reading
    "glucose_missing": "bool",  # no real reading within max_gap minutes before t
    "minutes_since_real": "float64",
    "cgm_session": "Int64",  # increments after a gap >= session_gap_min
    "early_wear": "bool",  # first hours of a CGM session
    "glucose_secondary": "float64",  # CGMacros Libre (real readings only), mg/dL
    "hr": "float64",  # bpm, NaN if missing
    "hr_missing": "bool",
    "activity_intensity": "float64",  # 0 sedentary, 1 light, 2 moderate, 3 vigorous
    "activity_source": "string",  # mets | intensity | none
    "activity_missing": "bool",
    "rest_proxy": "boolean",  # CGMacros only; NA elsewhere. NOT sleep.
    "meal_start": "bool",  # a logged meal starts within this grid step
    "minutes_since_logged_meal": "float64",
}

STATIC_COLUMNS: dict[str, str] = {
    "patient_id": "string",
    "dataset": "string",
    "diabetes_status": "string",  # healthy | prediabetes | t2d
    "age_years": "float64",
    "sex": "string",  # F | M
    "ethnicity": "string",
    "height_cm": "float64",
    "weight_kg": "float64",
    "bmi": "float64",  # recomputed from kg / m^2
    "bmi_reported": "float64",
    "hba1c_pct": "float64",
    "hba1c_mmol_mol": "float64",
    "fasting_glucose_mgdl": "float64",
    "fasting_insulin": "float64",  # CGMacros uU/mL; Shanghai pmol/L converted to uU/mL
    "triglycerides_mgdl": "float64",
    "cholesterol_mgdl": "float64",
    "hdl_mgdl": "float64",
    "ldl_mgdl": "float64",
    "diabetes_duration_years": "float64",
    "fasting_c_peptide_nmol_l": "float64",  # Shanghai only
    "n_periods": "Int64",
    "synthetic": "bool",
}

MEAL_COLUMNS: dict[str, str] = {
    "patient_id": "string",
    "dataset": "string",
    "period_id": "string",
    "t": "datetime64[ns]",
    "minutes_since_start": "float64",
    "meal_type": "string",  # breakfast | lunch | dinner | snack | unknown
    "kcal": "float64",
    "carbs_g": "float64",
    "protein_g": "float64",
    "fat_g": "float64",
    "fiber_g": "float64",
    "amount_pct_raw": "float64",  # as logged; never altered
    "amount_gt100": "bool",
    "amount_missing": "bool",
    "macros_available": "bool",  # False for Shanghai (no macros; none invented)
    "food_grams_total": "float64",  # Shanghai: summed grams from free text
    "food_items": "Int64",
}

FORBIDDEN_FEATURE_PATTERNS: tuple[str, ...] = (
    r"date",
    r"weekday",
    r"day_?of_?week",
    r"\bdow\b",
    r"month",
    r"^year$|_year$|^year_",  # calendar year; durations like `age_years` are allowed
    r"week",
    r"\bdoy\b",
    r"day_?of_?year",
    r"is_weekend",
)
"""Regexes that must not match any harmonized column name (calendar features, D-012)."""


def is_calendar_column(name: str) -> bool:
    """True if ``name`` looks like a calendar-date feature (forbidden: dates are shifted)."""
    import re

    return any(re.search(p, name.lower()) for p in FORBIDDEN_FEATURE_PATTERNS)


DIABETES_STATUSES: tuple[str, ...] = ("healthy", "prediabetes", "t2d")
MEAL_TYPES: tuple[str, ...] = ("breakfast", "lunch", "dinner", "snack", "unknown")


def conform(df, columns: dict[str, str]):  # type: ignore[no-untyped-def]
    """Return ``df`` with exactly ``columns`` in order, cast to the declared dtypes.

    Missing columns are added as all-NA (never silently imputed).
    """
    import numpy as np
    import pandas as pd

    df = df.reset_index(drop=True)
    out = pd.DataFrame(index=df.index)
    for name, dtype in columns.items():
        col = df[name] if name in df.columns else pd.Series([pd.NA] * len(df), index=df.index)
        if dtype.startswith("datetime"):
            out[name] = pd.to_datetime(col)
        elif dtype == "bool":
            out[name] = col.astype("boolean").fillna(False).astype(bool)
        elif dtype == "float64":
            out[name] = pd.to_numeric(col, errors="coerce").astype("float64").fillna(np.nan)
        else:
            out[name] = col.astype(dtype)
    return out
