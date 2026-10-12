"""Integrity tests on the harmonized parquet files (skipped when `make data` has not been run)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.harmonize_cgmacros import normalize_meal_type
from src.data.schema import (
    DIABETES_STATUSES,
    MEAL_COLUMNS,
    MEAL_TYPES,
    STATIC_COLUMNS,
    TIMESERIES_COLUMNS,
    is_calendar_column,
)
from src.data.splits import patient_train_cal_test_split
from src.utils.config import load_config

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
CFG = load_config("data")
needs_data = pytest.mark.skipif(
    not (PROC / "static.parquet").exists(), reason="processed data not built (run `make data`)"
)


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / f"{name}.parquet")


# ---------- schema-level tests (always run) ----------
def test_schema_has_no_calendar_columns() -> None:
    for cols in (TIMESERIES_COLUMNS, STATIC_COLUMNS, MEAL_COLUMNS):
        for c in cols:
            assert not is_calendar_column(c), c


@pytest.mark.parametrize(
    "bad", ["date", "meal_date", "weekday", "day_of_week", "month", "year", "week_index", "doy"]
)
def test_calendar_detector_catches(bad: str) -> None:
    assert is_calendar_column(bad)


@pytest.mark.parametrize("ok", ["age_years", "diabetes_duration_years", "days_since_start", "t"])
def test_calendar_detector_allows(ok: str) -> None:
    assert not is_calendar_column(ok)


def test_meal_type_normalization() -> None:
    raw = pd.Series(["Breakfast", "lunch", "Snack", "Snacks", "snack 1", " Dinner ", None, "x"])
    out = normalize_meal_type(raw).tolist()
    assert out == ["breakfast", "lunch", "snack", "snack", "snack", "dinner", "unknown", "unknown"]


# ---------- data tests ----------
@needs_data
@pytest.mark.parametrize("name", ["cgmacros_ts5", "common_ts15"])
def test_timeseries_schema_and_no_calendar(name: str) -> None:
    df = _load(name)
    assert list(df.columns) == list(TIMESERIES_COLUMNS)
    assert not [c for c in df.columns if is_calendar_column(c)]


@needs_data
@pytest.mark.parametrize("name", ["cgmacros_ts5", "common_ts15"])
def test_monotonic_and_unique(name: str) -> None:
    df = _load(name)
    assert not df.duplicated(["period_id", "t"]).any()
    assert df.groupby("period_id")["t"].apply(lambda s: s.is_monotonic_increasing).all()


@needs_data
@pytest.mark.parametrize("name", ["cgmacros_ts5", "common_ts15"])
def test_glucose_physiological_and_flags(name: str) -> None:
    df = _load(name)
    c = CFG["cgm"]
    for ds, (lo, hi) in c["glucose_range_mgdl"].items():
        g = df.loc[df.dataset == ds, "glucose"].dropna()
        assert g.between(lo, hi).all(), ds
    flags = df[["glucose_is_real", "glucose_is_carried", "glucose_missing"]].sum(axis=1)
    assert (flags == 1).all(), "exactly one CGM state per row"
    assert df.loc[df.glucose_missing, "glucose"].isna().all()
    assert df.loc[~df.glucose_missing, "glucose"].notna().all()
    assert df.loc[~df.glucose_missing, "minutes_since_real"].max() <= c["max_gap_min"]


@needs_data
def test_grid_steps() -> None:
    for name, step in (("cgmacros_ts5", 5), ("common_ts15", 15)):
        df = _load(name)
        d = df.groupby("period_id")["t"].diff().dropna().dt.total_seconds() / 60
        assert (d == step).all(), name


@needs_data
def test_static_units_and_status() -> None:
    s = _load("static")
    assert list(s.columns)[: len(STATIC_COLUMNS)] == list(STATIC_COLUMNS)
    assert s.patient_id.is_unique
    assert s.diabetes_status.isin(DIABETES_STATUSES).all()
    assert s.hba1c_pct.dropna().between(3.5, 18).all(), "HbA1c must be in %"
    assert s.height_cm.dropna().between(120, 220).all(), "height must be cm"
    assert s.weight_kg.dropna().between(30, 200).all(), "weight must be kg"
    diff = (s.bmi - s.bmi_reported).abs().dropna()
    assert (diff < 1.0).mean() > 0.95, "BMI recomputed from kg/cm should match reported"
    cg = s[s.dataset == "cgmacros"]
    assert cg.diabetes_status.value_counts().to_dict() == {
        "prediabetes": 16,
        "healthy": 15,
        "t2d": 14,
    }
    assert (s[s.dataset == "shanghai_t2dm"].diabetes_status == "t2d").all()


@needs_data
def test_patient_counts() -> None:
    s = _load("static")
    assert (s.dataset == "cgmacros").sum() == 45
    assert (s.dataset == "shanghai_t2dm").sum() == 100
    ts = _load("common_ts15")
    sh = ts[ts.dataset == "shanghai_t2dm"]
    assert sh.patient_id.nunique() == 100 and sh.period_id.nunique() == 109


@needs_data
def test_repeated_periods_never_split_across_partitions() -> None:
    ts = _load("common_ts15")
    sh = ts[ts.dataset == "shanghai_t2dm"]
    multi = sh.groupby("patient_id").period_id.nunique()
    assert (multi > 1).sum() == 8
    split = patient_train_cal_test_split(
        sh["patient_id"], {"train": 0.6, "calibration": 0.2, "test": 0.2}, 42
    )
    lookup = {p: k for k, ids in split.as_dict().items() for p in ids}
    per_period = sh.drop_duplicates("period_id").assign(part=lambda d: d.patient_id.map(lookup))
    assert per_period.groupby("patient_id").part.nunique().max() == 1


@needs_data
def test_meals_schema_and_no_invented_macros() -> None:
    m = _load("meals")
    assert list(m.columns) == list(MEAL_COLUMNS)
    assert m.meal_type.isin(MEAL_TYPES).all()
    sh = m[m.dataset == "shanghai_t2dm"]
    assert not sh.macros_available.any()
    assert sh[["kcal", "carbs_g", "protein_g", "fat_g", "fiber_g"]].isna().all().all()
    cg = m[m.dataset == "cgmacros"]
    assert (cg.amount_gt100 == (cg.amount_pct_raw > 100)).all(), "flag must mirror raw value"
    assert cg.loc[cg.patient_id.isin(["cgmacros_028", "cgmacros_029"]), "amount_missing"].all()


@needs_data
def test_rest_proxy_only_cgmacros() -> None:
    ts = _load("common_ts15")
    assert ts.loc[ts.dataset == "shanghai_t2dm", "rest_proxy"].isna().all()
    assert not any("sleep" in c for c in ts.columns)
