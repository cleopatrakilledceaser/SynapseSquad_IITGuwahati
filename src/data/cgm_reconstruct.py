"""Recover real CGM readings from linearly interpolated 1-minute rows; map them to a grid causally.

CGMacros stores Dexcom (5-min) and Libre (15-min) on a 1-minute grid after linear interpolation,
which also bridges long dropouts (up to about 9 days). :func:`detect_native` recovers real readings:

1. A linear interpolant bends only at real readings, so every kink (|second difference| >
   ``kink_tol``) is real, as are the first and last values of each non-missing run.
2. Collinear real readings (no kink) are recovered between consecutive anchors ``a < b`` only
   if ``b - a`` is a multiple of the period and ``<= max_span``, and the value at ``a + k*period``
   is an integer (Dexcom reports integer mg/dL). Points in longer straight stretches are never
   accepted, so a bridged dropout never becomes "real" data.

:func:`to_grid` gives each grid time ``t`` the latest real reading at or before ``t``
(``merge_asof`` backward) and marks it MISSING if that reading is older than ``max_gap_min``.
No grid value is ever computed from a reading taken after ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_native(
    minutes: np.ndarray,
    values: np.ndarray,
    period: int,
    max_span: int,
    kink_tol: float,
    integer_tol: float | None,
) -> np.ndarray:
    """Return a boolean mask of real (native) readings in a 1-minute interpolated series.

    Args:
        minutes: Integer minute index per row, strictly increasing.
        values: Glucose per row in mg/dL (NaN allowed).
        period: Native sensor period in minutes (5 for Dexcom, 15 for Libre).
        max_span: Longest straight stretch (minutes) whose interior can hold real readings.
        kink_tol: |second difference| above which a row is a bend, i.e. a real reading.
        integer_tol: If given, collinear candidates must be integers within this tolerance.

    Returns:
        Boolean array aligned with ``values``; True where the row is a real reading.
    """
    native = np.zeros(len(values), dtype=bool)
    idx = np.flatnonzero(~np.isnan(values))
    if idx.size == 0:
        return native
    row_of_minute = {int(m): i for i, m in zip(idx, minutes[idx], strict=True)}
    # Rows may be absent for a few minutes (e.g. participant 007); the vendor interpolation is
    # still linear in *time* across them, so runs only break at long time jumps.
    breaks = np.flatnonzero(np.diff(minutes[idx]) > max(max_span, period)) + 1
    for run in np.split(idx, breaks):
        native[run[0]] = native[run[-1]] = True
        if run.size < 3:
            continue
        slope = np.diff(values[run]) / np.diff(minutes[run])
        native[run[1:-1][np.abs(np.diff(slope)) > kink_tol]] = True
        anchors = run[native[run]]
        for a, b in zip(anchors[:-1], anchors[1:], strict=True):
            span = int(minutes[b] - minutes[a])
            if span <= period or span > max_span or span % period:
                continue
            for k in range(1, span // period):
                j = row_of_minute.get(int(minutes[a]) + k * period)
                if j is None:
                    continue
                if integer_tol is None or abs(values[j] - round(values[j])) <= integer_tol:
                    native[j] = True
    if integer_tol is not None:
        # Run ends are where the vendor's interpolation was truncated; a non-integer value there
        # is an interpolated point, not a reading (verified: all 668 non-integer candidates in
        # CGMacros Dexcom were run ends). Require integers for every accepted reading.
        native &= np.abs(values - np.round(values)) <= integer_tol
    return native


def make_grid(start: pd.Timestamp, end: pd.Timestamp, step_min: int) -> pd.DatetimeIndex:
    """Clock-aligned grid from ``ceil(start)`` to ``ceil(end)`` with ``step_min`` spacing."""
    freq = f"{step_min}min"
    return pd.date_range(start.ceil(freq), end.ceil(freq), freq=freq)


def reinterpolation_error(
    minutes: np.ndarray, values: np.ndarray, native: np.ndarray, max_span: int
) -> tuple[float, float]:
    """Error (mg/dL) of re-interpolating detected native points back onto the raw 1-minute rows.

    Only rows strictly between two consecutive native readings at most ``max_span`` minutes apart
    are scored (elsewhere the raw value is a bridge or truncated tail, which we discard on purpose).
    If the detected set is complete there, interpolation reproduces the raw series exactly.

    Returns:
        (max absolute error, fraction of scored rows with error > 0.01 mg/dL).
    """
    nat_idx = np.flatnonzero(native)
    if nat_idx.size < 2:
        return float("nan"), float("nan")
    rebuilt = np.interp(minutes, minutes[nat_idx], values[nat_idx])
    scored = np.zeros(len(values), dtype=bool)
    for a, b in zip(nat_idx[:-1], nat_idx[1:], strict=True):
        if minutes[b] - minutes[a] <= max_span:
            scored[a + 1 : b] = True
    scored &= ~np.isnan(values)
    if not scored.any():
        return 0.0, 0.0
    err = np.abs(rebuilt[scored] - values[scored])
    return float(err.max()), float(np.mean(err > 0.01))


def to_grid(
    real_times: pd.Series,
    real_values: pd.Series,
    grid: pd.DatetimeIndex,
    step_min: int,
    max_gap_min: int,
    session_gap_min: int,
    early_wear_hours: float,
) -> pd.DataFrame:
    """Map real readings onto ``grid`` using only readings at or before each grid time.

    Args:
        real_times: Timestamps of real readings (any order).
        real_values: Glucose at those times (mg/dL).
        grid: Target grid (sorted).
        step_min: Grid step in minutes.
        max_gap_min: A grid point whose latest real reading is older than this is MISSING.
        session_gap_min: A gap between real readings this long starts a new CGM session.
        early_wear_hours: Grid points this soon after a session start are flagged ``early_wear``.

    Returns:
        Frame aligned with ``grid``: glucose, flags, ``source_time`` (time of the reading used),
        ``minutes_since_real`` and ``cgm_session``.
    """
    real = (
        pd.DataFrame({"source_time": pd.to_datetime(real_times), "v": np.asarray(real_values)})
        .dropna()
        .sort_values("source_time")
        .drop_duplicates("source_time")
    )
    gap = real["source_time"].diff().dt.total_seconds().div(60)
    real["cgm_session"] = (gap.isna() | (gap >= session_gap_min)).cumsum().astype("Int64")
    real["session_start"] = real.groupby("cgm_session")["source_time"].transform("min")
    m = pd.merge_asof(
        pd.DataFrame({"t": grid}),
        real,
        left_on="t",
        right_on="source_time",
        direction="backward",
    )
    since = (m["t"] - m["source_time"]).dt.total_seconds().div(60)
    missing = since.isna() | (since > max_gap_min)
    is_real = ~missing & (since < step_min)
    since_session = (m["t"] - m["session_start"]).dt.total_seconds()
    return pd.DataFrame(
        {
            "t": m["t"],
            "glucose": m["v"].where(~missing),
            "glucose_is_real": is_real,
            "glucose_is_carried": ~missing & ~is_real,
            "glucose_missing": missing,
            "minutes_since_real": since,
            "source_time": m["source_time"].where(~missing),
            "cgm_session": m["cgm_session"].where(~missing),
            "early_wear": ~missing & (since_session < early_wear_hours * 3600),
        }
    )
