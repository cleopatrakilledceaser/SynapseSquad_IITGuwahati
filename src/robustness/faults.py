"""Fault injection on one period's regular grid (D-047).

Every fault modifies ONLY grid rows with ``days_since_start >= tau`` (tested). The grid keeps the
Part 1 schema; after a fault the derived columns (``glucose_is_real``, ``glucose_missing``,
``glucose_is_carried``, ``minutes_since_real``) are made consistent again, and features are then
recomputed with the unchanged causal Part 2 code (``src.features.build.period_features``).
Timestamp faults are delays only (values arrive late); an "early" shift would leak the future.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def refresh_derived(g: pd.DataFrame, step: int, orig: pd.DataFrame | None = None) -> pd.DataFrame:
    """Make missing / carried flags and minutes since the last real reading consistent again.

    Part 1 stores ``minutes_since_real`` relative to the RAW reading time (e.g. 1.0 at a real
    5-min grid point), so it is never recomputed from the grid index: rows that are real in both
    ``orig`` and ``g`` keep the stored value; other rows get the stored value of the last real row
    plus the grid distance (identity when nothing changed; tested).
    """
    real = g["glucose_is_real"].to_numpy(bool)
    g["glucose_missing"] = ~real & ~g["glucose_is_carried"].to_numpy(bool)
    ref = orig if orig is not None else g
    was_real = ref["glucose_is_real"].to_numpy(bool)
    if "minutes_since_real" in ref:
        base = ref["minutes_since_real"].to_numpy(float)
    else:  # fresh grid without the column: real readings sit exactly on grid points
        base = np.zeros(len(g))
    idx = np.arange(len(g))
    last = np.maximum.accumulate(np.where(real, idx, -1))
    anchor = np.where(real & was_real, base, 0.0)
    since = np.where(last >= 0, anchor[np.maximum(last, 0)] + (idx - last) * step, np.nan)
    since = np.where(real & was_real, base, since)
    g["minutes_since_real"] = since.astype(float)
    return g


def _drop(g: pd.DataFrame, mask: np.ndarray) -> None:
    """Remove readings (become missing, never carried)."""
    g.loc[mask, "glucose"] = np.nan
    g.loc[mask, "glucose_is_real"] = False
    g.loc[mask, "glucose_is_carried"] = False


def inject(
    g: pd.DataFrame,
    fault: str,
    level: float,
    tau_days: float,
    step: int,
    spec: dict[str, Any],
    clip: tuple[float, float],
    seed: int,
) -> pd.DataFrame:
    """Return a faulted copy of one period grid (sorted by t); rows before tau untouched."""
    g = g.copy().reset_index(drop=True)
    orig = g.copy()
    rng = np.random.default_rng(seed)
    after = g["days_since_start"].to_numpy(float) >= tau_days
    real = g["glucose_is_real"].to_numpy(bool)
    mins = g["minutes_since_start"].to_numpy(float)
    t0 = tau_days * 1440.0
    glu = g["glucose"].to_numpy(float).copy()
    if fault == "gaps":
        per = spec["every_h"] * 60.0
        pos = (mins - t0) % per
        _drop(g, after & (pos < level))
    elif fault == "dropout":
        n = int(after.sum())
        drop = np.zeros(len(g), bool)
        idx = np.flatnonzero(after)
        target = int(round(level * n))
        while drop[idx].sum() < target and n > 0:
            s = idx[rng.integers(0, n)]
            ln = max(1, int(rng.geometric(1.0 / spec["mean_burst"])))
            drop[s : min(s + ln, idx[-1] + 1)] = True
        _drop(g, drop & after)
    elif fault in ("bias", "drift", "noise", "compression"):
        m = after & real
        if fault == "bias":
            glu[m] += level
        elif fault == "drift":
            glu[m] += level * (mins[m] - t0) / 1440.0
        elif fault == "noise":
            glu[m] += rng.normal(0.0, level, int(m.sum()))
        else:
            hrs = (g["t"].dt.hour + g["t"].dt.minute / 60.0).to_numpy(float)
            lo_h, hi_h = spec["night_hours"]
            day = np.floor(mins / 1440.0)
            for d in np.unique(day[after]):
                start_h = rng.uniform(lo_h, hi_h - spec["dip_min"] / 60.0)
                w = (day == d) & (hrs >= start_h) & (hrs < start_h + spec["dip_min"] / 60.0)
                glu[w & m] -= level
        g["glucose"] = np.where(m, np.clip(glu, *clip), g["glucose"].to_numpy(float))
    elif fault == "timestamp_delay":
        k = int(round(level / step))
        if k * step != level:
            raise ValueError(f"delay {level} min is not a multiple of the {step}-min grid")
        src_g = g["glucose"].to_numpy(float)
        src_r = g["glucose_is_real"].to_numpy(bool)
        src_c = g["glucose_is_carried"].to_numpy(bool)
        i = np.flatnonzero(after)
        j = i - k
        ok = j >= 0
        g.loc[i[ok], "glucose"] = src_g[j[ok]]
        g.loc[i[ok], "glucose_is_real"] = src_r[j[ok]]
        g.loc[i[ok], "glucose_is_carried"] = src_c[j[ok]]
    elif fault == "wearable_missing":
        day = np.floor(mins / 1440.0)
        days = np.unique(day[after])
        n_off = int(round(level * len(days)))
        off = rng.choice(days, n_off, replace=False) if n_off else np.array([])
        m = after & np.isin(day, off)
        for c in ("hr", "activity_intensity"):
            g.loc[m, c] = np.nan
        for c in ("hr_missing", "activity_missing"):
            if c in g:
                g.loc[m, c] = True
    else:
        raise ValueError(f"unknown fault {fault}")
    return refresh_derived(g, step, orig)
