"""Glucose report numbers (Part 9): zones, GMI, variability, coverage, AGP profile."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dash import agp

Z = {"very_low": 54, "low": 70, "high": 180, "very_high": 250}


def test_shares_sum_to_100_and_cuts_consistent():
    g = pd.Series([40, 60, 100, 150, 200, 300, 120, 180, 70, 54], dtype=float)
    s = agp.shares(g, Z)
    assert s["very_low"] + s["low"] + s["range"] + s["high"] + s["very_high"] == pytest.approx(100)
    assert s["below70"] == pytest.approx(s["very_low"] + s["low"])
    assert s["above180"] == pytest.approx(s["high"] + s["very_high"])
    assert s["range"] == pytest.approx(50.0)  # 100, 150, 120, 180, 70 of 10
    assert agp.shares(pd.Series([], dtype=float), Z) == {}


def test_gmi_known_values():
    assert agp.gmi(154) == pytest.approx(7.0, abs=0.01)  # 3.31 + 0.02392 x 154
    assert agp.gmi(100) == pytest.approx(5.70, abs=0.01)


def test_cv_percent():
    assert agp.cv_percent(pd.Series([100.0, 100.0, 100.0])) == 0.0
    assert agp.cv_percent(pd.Series([90.0, 110.0])) == pytest.approx(10.0)


def test_targets_direction():
    assert agp.meets("range", 71.0) and not agp.meets("range", 70.0)
    assert agp.meets("below70", 3.9) and not agp.meets("below70", 4.0)
    assert agp.meets("above250", 4.0) and not agp.meets("above250", 5.0)


def _df(days=3, step=5, seed=0):
    t = pd.date_range("2026-01-05", periods=days * 1440 // step, freq=f"{step}min")
    rng = np.random.default_rng(seed)
    g = 130 + 30 * np.sin(2 * np.pi * (t.hour * 60 + t.minute) / 1440) + rng.normal(0, 5, len(t))
    return pd.DataFrame({"t": t, "glucose": g})


def test_coverage_full_and_with_gaps():
    d = _df()
    c = agp.coverage(d)
    assert c["fraction"] == pytest.approx(1.0) and c["days"] == pytest.approx(3 - 5 / 1440)
    gap = d.drop(d.index[100:200])
    assert agp.coverage(gap)["fraction"] < 1.0


def test_agp_profile_percentiles_ordered_and_follow_daily_shape():
    p = agp.agp_profile(_df(days=6))
    assert len(p) == 48 and (p["q05"] <= p["q25"]).all() and (p["q75"] <= p["q95"]).all()
    assert (p["q25"] <= p["q50"]).all() and (p["q50"] <= p["q75"]).all()
    peak = p.loc[p["q50"].idxmax(), "minute"]
    assert 240 <= peak <= 480  # sine peaks about 06:00


def test_report_bundle():
    r = agp.report(_df(), Z)
    assert r["n"] == len(_df()) and r["gmi"] == pytest.approx(agp.gmi(r["mean"]))
    assert r["shares"]["range"] > 90
