"""Glucose unit switch (Part 9): mg/dL is unchanged; mmol/L converts numbers in text and charts."""

from __future__ import annotations

import pandas as pd
import pytest

from src.dash import charts, units
from src.dash import text as T

CFG = {"zones": {"very_low": 54, "low": 70, "high": 180, "very_high": 250},
       "y_range": [40, 300], "history_hours": 6, "forecast_min": 120}  # fmt: skip


@pytest.fixture(autouse=True)
def reset():
    units.set_unit(units.MGDL)
    yield
    units.set_unit(units.MGDL)


def test_mgdl_mode_is_identity():
    s = "Forecast at +60 min: 168 mg/dL (80% interval 150 to 186 mg/dL)."
    assert units.localize(s) == s and units.fmt(168.4) == "168 mg/dL" and units.num(54) == "54"
    assert T.mgdl(168.4) == "168 mg/dL" and T.mgdl(None) == T.NA


def test_mmol_conversion_values():
    units.set_unit(units.MMOLL)
    assert units.num(180) == "10.0" and units.num(70) == "3.9" and units.num(54) == "3.0"
    assert units.fmt(250) == "13.9 mmol/L" and T.mgdl(180) == "10.0 mmol/L"
    assert T.mgdl_num(180) == "10.0"


def test_localize_text_single_range_signed_and_untouched_words():
    units.set_unit(units.MMOLL)
    out = units.localize("Forecast at +60 min: 180 mg/dL (80% interval 144 to 198 mg/dL).")
    assert out == "Forecast at +60 min: 10.0 mmol/L (80% interval 8.0 to 11.0 mmol/L)."
    assert (
        units.localize("Change +18 mg/dL, below 70 mg/dL") == "Change +1.0 mmol/L, below 3.9 mmol/L"
    )
    assert units.localize("62% chance, 3 days") == "62% chance, 3 days"  # no unit, no change
    assert units.localize(42) == 42


def test_roundtrip_error_is_small():
    units.set_unit(units.MMOLL)
    for v in (40, 54, 70, 126, 180, 250, 400):
        assert float(units.num(v)) * units.FACTOR == pytest.approx(
            v, abs=units.FACTOR * 0.05 + 0.01
        )


def test_axis_ticks_only_in_mmol():
    assert units.axis_ticks(40, 300) is None
    units.set_unit(units.MMOLL)
    vals, labels = units.axis_ticks(40, 300)
    assert vals[0] == 40 and labels[vals.index(180)] == "10.0" and max(vals) <= 300


def _fig():
    try:
        import plotly  # noqa: F401
    except ImportError:
        pytest.skip("plotly not installed")
    now = pd.Timestamp("2026-01-05 12:00")
    hist = pd.DataFrame({"t": pd.date_range(now - pd.Timedelta(hours=3), now, freq="15min")})
    hist["glucose"] = 150.0
    fc = {h: {"mean": 160.0, "lo": 140.0, "hi": 180.0} for h in (30, 60, 120)}
    meals = pd.DataFrame({"t": [now - pd.Timedelta(hours=1)], "carbs_g": [60.0]})
    return charts.hero_figure(hist, now, 150.0, fc, meals, CFG)


def test_hero_chart_default_labels_unchanged_and_mmol_labels_converted():
    texts = [a.text for a in _fig().layout.annotations]
    assert "Very low, below 54" in texts and "In range, 70 to 180" in texts
    assert "<b>Now 150 mg/dL</b>" in texts
    units.set_unit(units.MMOLL)
    f = _fig()
    texts = [a.text for a in f.layout.annotations]
    assert "In range, 3.9 to 10.0" in texts and "<b>Now 8.3 mmol/L</b>" in texts
    assert f.layout.yaxis.title.text == "Glucose (mmol/L)"
    assert list(f.layout.yaxis.tickvals)[:2] == [40, 54]
    hist_trace = next(t for t in f.data if t.name == "Glucose history")
    assert "mmol/L" in hist_trace.hovertemplate and hist_trace.customdata[0] == pytest.approx(
        150 / 18.016
    )
