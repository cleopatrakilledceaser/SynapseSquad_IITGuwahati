"""Unit tests for CGM reconstruction on synthetic data (no real data needed; runs in CI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cgm_reconstruct import detect_native, make_grid, reinterpolation_error, to_grid

PERIOD, MAX_GAP, KINK, INT_TOL = 5, 30, 1e-3, 1e-6


def _interpolated(real_minutes: np.ndarray, real_values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Mimic the vendor file: 1-min rows, linear interpolation between real readings."""
    minutes = np.arange(real_minutes[0], real_minutes[-1] + 1)
    return minutes, np.interp(minutes, real_minutes, real_values)


@pytest.fixture()
def synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integer readings every 5 min with one 6-hour dropout bridged by interpolation."""
    rng = np.random.default_rng(42)
    first = np.arange(0, 600, PERIOD)
    second = np.arange(600 + 360, 1500, PERIOD)
    rm = np.concatenate([first, second])
    rv = np.round(120 + np.cumsum(rng.normal(0, 3, rm.size)))
    rv[10:13] = rv[9] + np.array([2, 4, 6])  # collinear real readings (no kink)
    minutes, values = _interpolated(rm, rv)
    return rm, rv, minutes, values


def test_detects_exactly_the_real_readings(synthetic) -> None:  # noqa: ANN001
    rm, _, minutes, values = synthetic
    nat = detect_native(minutes, values, PERIOD, MAX_GAP, KINK, INT_TOL)
    assert set(minutes[nat]) == set(rm)


def test_bridge_interior_never_marked_real(synthetic) -> None:  # noqa: ANN001
    _, _, minutes, values = synthetic
    nat = detect_native(minutes, values, PERIOD, MAX_GAP, KINK, INT_TOL)
    inside = (minutes > 595) & (minutes < 960)
    assert not nat[inside].any()


def test_reinterpolation_reproduces_raw(synthetic) -> None:  # noqa: ANN001
    _, _, minutes, values = synthetic
    nat = detect_native(minutes, values, PERIOD, MAX_GAP, KINK, INT_TOL)
    err, frac = reinterpolation_error(minutes, values, nat, MAX_GAP)
    assert err < 1e-9 and frac == 0.0


def test_grid_gap_is_missing_not_filled(synthetic) -> None:  # noqa: ANN001
    rm, rv, _, _ = synthetic
    t0 = pd.Timestamp("2021-01-01 00:00")
    times = pd.Series(t0 + pd.to_timedelta(rm, unit="min"))
    grid = make_grid(times.iloc[0], times.iloc[-1], 5)
    g = to_grid(times, pd.Series(rv), grid, 5, MAX_GAP, 120, 24)
    mins = (g["t"] - t0).dt.total_seconds() / 60
    in_gap = (mins > 595 + MAX_GAP) & (mins < 960)
    assert g.loc[in_gap, "glucose_missing"].all()
    assert g.loc[in_gap, "glucose"].isna().all()
    assert g.loc[~g.glucose_missing, "minutes_since_real"].max() <= MAX_GAP


def test_no_future_leak_in_grid() -> None:
    """Every grid value at t must come from a real reading taken at or before t."""
    rng = np.random.default_rng(42)
    t0 = pd.Timestamp("2021-01-01 00:03")  # off-grid readings
    rm = np.sort(rng.choice(np.arange(0, 3000), 400, replace=False))
    times = pd.Series(t0 + pd.to_timedelta(rm, unit="min"))
    vals = pd.Series(rng.integers(60, 250, rm.size).astype(float))
    grid = make_grid(times.min(), times.max(), 5)
    g = to_grid(times, vals, grid, 5, MAX_GAP, 120, 24)
    present = g[~g.glucose_missing]
    assert (present["source_time"] <= present["t"]).all()
    lookup = dict(zip(times, vals, strict=True))
    assert np.allclose(present["glucose"], present["source_time"].map(lookup))


def test_future_change_does_not_alter_past() -> None:
    """Changing readings after time T must not change any grid value at or before T."""
    rng = np.random.default_rng(42)
    t0 = pd.Timestamp("2021-01-01")
    rm = np.arange(0, 2000, 5)
    times = pd.Series(t0 + pd.to_timedelta(rm, unit="min"))
    v1 = pd.Series(rng.integers(60, 250, rm.size).astype(float))
    v2 = v1.copy()
    cut = 1000
    v2[rm > cut] += 50
    grid = make_grid(times.min(), times.max(), 5)
    g1 = to_grid(times, v1, grid, 5, MAX_GAP, 120, 24)
    g2 = to_grid(times, v2, grid, 5, MAX_GAP, 120, 24)
    past = (g1["t"] - t0).dt.total_seconds() / 60 <= cut
    pd.testing.assert_series_equal(g1.loc[past, "glucose"], g2.loc[past, "glucose"])


def test_non_integer_run_end_rejected() -> None:
    minutes = np.arange(0, 21)
    values = np.interp(minutes, [0, 5, 10, 15, 20], [100, 104, 103, 107, 108.4])
    nat = detect_native(minutes, values, PERIOD, MAX_GAP, KINK, INT_TOL)
    assert not nat[-1]
