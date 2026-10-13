"""Per-period input arrays for the mechanistic twin (Part 4).

For each recording period on its native grid (CGMacros 5 min, Shanghai 15 min):
    y      real CGM readings (NaN where the grid value is not a real reading)
    carbs  g of carbohydrate of meals logged in (t_{i-1}, t_i]  (meal enters at the next grid time)
    slow   slowing ratio (fat + w*protein + fiber) / (carbs + 1) of the most recent meal <= t_i
    act    activity input above the patient's own rest at t_i (D-033)
All inputs at index i use only data at or before t_i (tested in tests/test_mech.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Record:
    """One period's arrays (aligned to the grid)."""

    patient_id: str
    period_id: str
    t: np.ndarray  # datetime64[ns]
    days: np.ndarray  # days since period start
    y: np.ndarray
    carbs: np.ndarray
    slow: np.ndarray
    act: np.ndarray
    activity_source: str


def rest_level(x: pd.Series, q: float, min_n: int, default: float) -> np.ndarray:
    """Past-only expanding q-quantile of x (samples strictly before t; NaN ignored).

    Until ``min_n`` past non-missing samples exist, ``default`` is used (no future data).
    """
    past = x.shift(1)
    lvl = past.expanding(min_periods=1).quantile(q)
    n = past.notna().cumsum()
    return np.where(n >= min_n, lvl.to_numpy(dtype=float), default)


def activity_input(g: pd.DataFrame, acfg: dict[str, Any]) -> np.ndarray:
    """A(t) = (intensity - rest)+ + (HR - HR_rest)+ / hr_scale; missing parts count as 0."""
    q, n = acfg["rest_quantile"], acfg["min_rest_samples"]
    a = np.zeros(len(g))
    if "activity_intensity" in g:
        it = g["activity_intensity"].astype(float)
        it_rest = rest_level(it, q, n, acfg["intensity_rest_default"])
        a += np.nan_to_num(np.maximum(it.to_numpy() - it_rest, 0.0))
    if "hr" in g:
        hr = g["hr"].astype(float)
        hr_rest = rest_level(hr, q, n, acfg["hr_rest_default_bpm"])
        a += np.nan_to_num(np.maximum(hr.to_numpy() - hr_rest, 0.0) / acfg["hr_scale_bpm"])
    return a


def meal_arrays(
    t: np.ndarray, meals: pd.DataFrame, w_protein: float, pooled_dose: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """(carbs, slow) on the grid. A meal at time m enters at the first grid time >= m.

    ``pooled_dose``: if given (Shanghai, no macros), every logged meal is ``pooled_dose`` g of
    carbohydrate with slowing ratio 0.
    """
    n = len(t)
    carbs, ratio_at = np.zeros(n), np.full(n, np.nan)
    if len(meals):
        mt = meals["t"].to_numpy().astype("datetime64[ns]")
        idx = np.searchsorted(t, mt, side="left")
        ok = idx < n
        step = t[1] - t[0] if n > 1 else np.timedelta64(5, "m")
        ok &= ~((idx == 0) & (mt < t[0] - step))  # drop meals well before the record
        if pooled_dose is not None:
            c = np.full(len(meals), pooled_dose)
            r = np.zeros(len(meals))
        else:
            c = meals["carbs_g"].fillna(0).to_numpy(dtype=float)
            slow_g = (
                meals["fat_g"].fillna(0)
                + w_protein * meals["protein_g"].fillna(0)
                + meals["fiber_g"].fillna(0)
            ).to_numpy(dtype=float)
            r = slow_g / (c + 1.0)
        np.add.at(carbs, idx[ok], c[ok])
        for i, rr in zip(idx[ok], r[ok], strict=True):  # most recent meal wins at equal index
            ratio_at[i] = rr
    slow = pd.Series(ratio_at).ffill().fillna(0.0).to_numpy()
    return carbs, slow


def build_records(
    grid: pd.DataFrame,
    meals: pd.DataFrame,
    acfg: dict[str, Any],
    w_protein: float,
    use_activity: bool = True,
    pooled_dose: float | None = None,
) -> dict[str, Record]:
    """Records keyed by period_id (grid must be one cohort, sorted by period and time)."""
    out = {}
    mg = dict(tuple(meals.groupby("period_id")))
    for pid, g in grid.groupby("period_id", sort=True):
        g = g.sort_values("t")
        t = g["t"].to_numpy().astype("datetime64[ns]")
        carbs, slow = meal_arrays(t, mg.get(pid, meals.iloc[:0]), w_protein, pooled_dose)
        act = activity_input(g, acfg) if use_activity else np.zeros(len(g))
        src = g["activity_source"].dropna()
        out[pid] = Record(
            patient_id=str(g["patient_id"].iloc[0]),
            period_id=str(pid),
            t=t,
            days=g["days_since_start"].to_numpy(dtype=float),
            y=g["glucose"].where(g["glucose_is_real"]).to_numpy(dtype=float),
            carbs=carbs,
            slow=slow,
            act=act,
            activity_source=str(src.iloc[0]) if len(src) else "none",
        )
    return out


def anchor_index(rec: Record, times: np.ndarray) -> np.ndarray:
    """Grid indices of anchor times (must be exact grid times)."""
    idx = np.searchsorted(rec.t, times.astype("datetime64[ns]"))
    if (idx >= len(rec.t)).any() or (rec.t[np.minimum(idx, len(rec.t) - 1)] != times).any():
        raise ValueError(f"anchors not on the grid of {rec.period_id}")
    return idx
