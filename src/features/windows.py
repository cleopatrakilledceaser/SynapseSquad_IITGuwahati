"""Forecast anchors, regression targets and event labels (Part 2).

An *anchor* is a grid time ``t`` at which a forecast is issued. Anchors must sit on a real reading
(``anchor_requires_real``), on the stride lattice, and have enough real readings in the input
window ``(t - N, t]`` for the largest N compared. Targets are built ONLY from real readings:

- ``y_h``: glucose at exactly ``t + h`` if that grid point is a real reading, else NaN.
- ``<event>_at_h``: ``y_h`` beyond the threshold (NaN when ``y_h`` is NaN).
- ``<event>_any_h``: any real reading in ``(t, t + h]`` beyond the threshold. NaN when fewer than
  ``min_real_frac_horizon`` of the horizon's grid points are real (a negative needs coverage).
  A positive seen in the real readings is kept even with lower coverage.

Carried and missing values never enter a target (unit-tested).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _future_any(flag: np.ndarray, real: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """For each i: (any flag in rows i+1..i+k, count of real rows in i+1..i+k)."""
    n = len(flag)
    cf = np.concatenate([[0], np.cumsum(flag.astype(int))])
    cr = np.concatenate([[0], np.cumsum(real.astype(int))])
    i = np.arange(n)
    hi = np.minimum(i + k, n - 1) + 1
    lo = np.minimum(i + 1, n)
    return (cf[hi] - cf[lo]) > 0, cr[hi] - cr[lo]


def period_targets(g: pd.DataFrame, step: int, cfg: dict[str, Any]) -> pd.DataFrame:
    """Targets/labels for one period's grid (sorted, regular). Index aligned with ``g``."""
    w, ev = cfg["windows"], cfg["events"]
    real = g["glucose_is_real"].to_numpy()
    glu = g["glucose"].where(g["glucose_is_real"]).to_numpy()  # real-only glucose
    out: dict[str, np.ndarray] = {}
    n = len(g)
    for h in w["horizons_min"]:
        k = h // step
        y = np.full(n, np.nan)
        y[: n - k] = glu[k:] if k < n else np.nan
        out[f"y_{h}"] = y
        for name, spec in ev.items():
            thr = spec["threshold"]
            beyond = glu > thr if spec["direction"] == "above" else glu < thr
            beyond = np.where(np.isnan(glu), False, beyond)
            at = np.where(
                np.isnan(y), np.nan, (y > thr) if spec["direction"] == "above" else (y < thr)
            )
            out[f"{name}_at_{h}"] = at
            anyb, n_real = _future_any(beyond, real, k)
            enough = n_real >= w["min_real_frac_horizon"] * k
            valid = (enough | anyb) & (np.arange(n) + k < n)
            out[f"{name}_any_{h}"] = np.where(valid, anyb.astype(float), np.nan)
    return pd.DataFrame(out, index=g.index)


def anchor_mask(g: pd.DataFrame, step: int, cfg: dict[str, Any], stride: int) -> pd.DataFrame:
    """Anchor eligibility flags and input real fraction for each grid row of one period."""
    w = cfg["windows"]
    n_max = max(w["input_hours"]) * 60 // step
    real = g["glucose_is_real"].astype(int)
    frac = real.rolling(n_max, min_periods=1).sum() / n_max
    on_stride = (g["minutes_since_start"].round().astype(int) % stride) == 0
    history_ok = g["minutes_since_start"] >= (n_max - 1) * step
    return pd.DataFrame(
        {
            "on_stride": on_stride,
            "anchor_real": g["glucose_is_real"],
            "history_ok": history_ok,
            "input_real_frac": frac,
            "input_ok": frac >= w["min_real_frac_input"],
        },
        index=g.index,
    )


def build_anchor_table(
    grid: pd.DataFrame, step: int, stride: int, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """All valid anchors with targets for every period, plus drop counts per reason."""
    parts, counts = [], {
        "candidates_on_stride": 0,
        "dropped_anchor_not_real": 0,
        "dropped_insufficient_history": 0,
        "dropped_input_real_frac": 0,
    }
    for _, g in grid.groupby("period_id", sort=True):
        g = g.sort_values("t")
        m = anchor_mask(g, step, cfg, stride)
        tg = period_targets(g, step, cfg)
        s = m["on_stride"]
        counts["candidates_on_stride"] += int(s.sum())
        counts["dropped_anchor_not_real"] += int((s & ~m.anchor_real).sum())
        counts["dropped_insufficient_history"] += int((s & m.anchor_real & ~m.history_ok).sum())
        counts["dropped_input_real_frac"] += int(
            (s & m.anchor_real & m.history_ok & ~m.input_ok).sum()
        )
        keep = s & m.anchor_real & m.history_ok & m.input_ok
        cols = ["patient_id", "period_id", "dataset", "t", "minutes_since_start", "glucose"]
        part = pd.concat(
            [g.loc[keep, cols], m.loc[keep, ["input_real_frac"]], tg.loc[keep]], axis=1
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True), counts
