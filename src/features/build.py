"""Causal features at each grid time t (Part 2). Every feature uses only rows at or before t.

All rolling operations are trailing (pandas ``rolling`` on a sorted regular grid, window ending at
t). Feature names never encode calendar information (dates are shifted, D-012). CGM features use
REAL readings only; carried values appear only via explicit ``*_frac_carried`` counts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _rolling_slope(y: pd.Series, k: int, min_points: int, step: int) -> pd.Series:
    """Least-squares slope (mg/dL per min) over the last ``k`` rows; NaN-aware, trailing."""
    x = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    ok = y.notna()
    xv, yv = x.where(ok), y.where(ok)
    n = ok.astype(float).rolling(k, min_periods=1).sum()
    sx = xv.rolling(k, min_periods=1).sum()
    sy = yv.rolling(k, min_periods=1).sum()
    sxx = (xv * xv).rolling(k, min_periods=1).sum()
    sxy = (xv * yv).rolling(k, min_periods=1).sum()
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den.replace(0, np.nan)
    return (slope / step).where(n >= min_points)


def cgm_features(g: pd.DataFrame, step: int, n_hours: int, cfg: dict[str, Any]) -> pd.DataFrame:
    """CGM features for one period's sorted grid, for input window ``n_hours``."""
    c = cfg["cgm"]
    y = g["glucose"].where(g["glucose_is_real"])  # real readings only
    nwin = n_hours * 60 // step
    f: dict[str, pd.Series] = {"cgm_now": y}
    for lag in c["lags_min"]:
        if lag // step <= nwin and lag % step == 0:
            f[f"cgm_lag_{lag}"] = y.shift(lag // step)
    for d in c["delta_min"]:
        if d // step <= nwin and d % step == 0:
            f[f"cgm_delta_{d}"] = y - y.shift(d // step)
    for wmin in c["slope_windows_min"]:
        k = wmin // step + 1
        if k - 1 <= nwin and k >= 2:
            f[f"cgm_slope_{wmin}"] = _rolling_slope(y, k, min(c["slope_min_points"], k), step)
    a = c["accel_window_min"] // step
    if f"cgm_slope_{c['accel_window_min']}" in f:
        s = f[f"cgm_slope_{c['accel_window_min']}"]
        f["cgm_accel"] = (s - s.shift(a)) / c["accel_window_min"]
    windows = sorted({m for m in c["rolling_windows_min"] if m <= n_hours * 60} | {n_hours * 60})
    for wmin in windows:
        k = wmin // step
        r = y.rolling(k, min_periods=max(2, k // 2))
        tag = "N" if wmin == n_hours * 60 else str(wmin)
        f[f"cgm_mean_{tag}"] = r.mean()
        f[f"cgm_std_{tag}"] = r.std()
        f[f"cgm_min_{tag}"] = r.min()
        f[f"cgm_max_{tag}"] = r.max()
    k = min(c["tir_window_min"], n_hours * 60) // step
    inr = ((y >= c["range_low"]) & (y <= c["range_high"])).astype(float).where(y.notna())
    f["cgm_tir_3h"] = inr.rolling(k, min_periods=max(2, k // 2)).mean()
    f["cgm_tabove140_3h"] = (
        (y > c["range_secondary"])
        .astype(float)
        .where(y.notna())
        .rolling(k, min_periods=max(2, k // 2))
        .mean()
    )
    f["cgm_frac_real_N"] = g["glucose_is_real"].astype(float).rolling(nwin, min_periods=1).mean()
    f["cgm_frac_carried_N"] = (
        g["glucose_is_carried"].astype(float).rolling(nwin, min_periods=1).mean()
    )
    f["cgm_minutes_since_real"] = g["minutes_since_real"]
    return pd.DataFrame(f, index=g.index)


def time_features(g: pd.DataFrame) -> pd.DataFrame:
    """Cyclical clock time (valid: only whole days were shifted) and days since start."""
    mins = g["t"].dt.hour * 60 + g["t"].dt.minute
    ang = 2 * np.pi * mins / 1440
    return pd.DataFrame(
        {
            "tod_sin": np.sin(ang),
            "tod_cos": np.cos(ang),
            "days_since_start": g["days_since_start"],
        },
        index=g.index,
    )


def meal_features(g: pd.DataFrame, meals: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Meal features from logged meals at or before t (``merge_asof`` backward + trailing sums).

    A meal logged exactly at t is included (it is known at t). Shanghai has no macros, so only the
    timing / grams features are non-NaN there (no macros are invented).
    """
    mc = cfg["meals"]
    t = g["t"]
    out = pd.DataFrame(index=g.index)
    m = meals.sort_values("t")
    if m.empty:
        out["meal_minutes_since_last"] = np.nan
        return out
    last = (
        pd.merge_asof(
            pd.DataFrame({"t": t, "_i": g.index}).sort_values("t"),
            m.rename(columns={"t": "meal_t"}).assign(t=lambda d: d["meal_t"]),
            on="t",
            direction="backward",
        )
        .set_index("_i")
        .reindex(g.index)
    )
    out["meal_minutes_since_last"] = (t - last["meal_t"]).dt.total_seconds() / 60
    for mt in mc["meal_types"]:
        out[f"meal_last_type_{mt}"] = (
            (last["meal_type"] == mt).astype(float).where(last["meal_t"].notna())
        )
    out["meal_last_amount_gt100"] = last["amount_gt100"].astype(float).where(last["meal_t"].notna())
    ts_ns = t.to_numpy().astype("datetime64[ns]").astype(np.int64)
    mt_ns = m["t"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    for hrs in mc["windows_h"]:
        lo_ns = ts_ns - np.int64(hrs * 3600 * 10**9)
        hi_idx = np.searchsorted(mt_ns, ts_ns, side="right")  # meals with time <= t
        lo_idx = np.searchsorted(mt_ns, lo_ns, side="right")  # meals with time > t - hrs
        cnt = hi_idx - lo_idx
        out[f"meal_count_{hrs}h"] = cnt.astype(float)
        cols = list(mc["macros"]) + ["food_grams_total"]
        for col in cols:
            v = m[col].to_numpy(dtype=float)
            if np.all(np.isnan(v)):
                continue
            cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(v))])
            out[f"meal_{col}_{hrs}h"] = cs[hi_idx] - cs[lo_idx]
    return out


def wearable_features(
    g: pd.DataFrame, step: int, n_hours: int, cfg: dict[str, Any], minutes_since_meal: pd.Series
) -> pd.DataFrame:
    """HR / activity aggregates (trailing), post-meal activity, previous-night rest_proxy."""
    w = cfg["wearable"]
    out = pd.DataFrame(index=g.index)
    if g["hr"].notna().sum() == 0 and g["activity_intensity"].notna().sum() == 0:
        return out  # cohort without wearables (Shanghai): no wearable features at all
    windows = sorted({m for m in w["windows_min"] if m <= n_hours * 60} | {n_hours * 60})
    for wmin in windows:
        k = wmin // step
        tag = "N" if wmin == n_hours * 60 else str(wmin)
        hr = g["hr"].rolling(k, min_periods=1)
        out[f"hr_mean_{tag}"] = hr.mean()
        out[f"hr_max_{tag}"] = hr.max()
        out[f"hr_std_{tag}"] = hr.std()
        act = g["activity_intensity"].rolling(k, min_periods=1)
        out[f"act_mean_{tag}"] = act.mean()
        out[f"act_max_{tag}"] = act.max()
        out[f"hr_frac_missing_{tag}"] = (
            g["hr_missing"].astype(float).rolling(k, min_periods=1).mean()
        )
    out["hr_now"] = g["hr"]
    out["act_now"] = g["activity_intensity"]
    # post-meal activity: mean activity since the last logged meal, if that meal is recent
    pm = w["post_meal_window_min"]
    k = pm // step
    act_mean_pm = g["activity_intensity"].rolling(k, min_periods=1).mean()
    out["act_post_meal"] = act_mean_pm.where(minutes_since_meal <= pm)
    # previous-night rest_proxy summary: fraction of the most recent COMPLETED night (22-07 h)
    out["rest_proxy_prev_night_frac"] = _prev_night_rest(g, w)
    return out


def _prev_night_rest(g: pd.DataFrame, w: dict[str, Any]) -> pd.Series:
    """Fraction of rest_proxy==True over the last night that ended at or before t."""
    t = g["t"]
    h = t.dt.hour
    night = (h >= w["night_start_hour"]) | (h < w["night_end_hour"])
    # a night is labeled by the date (shifted, used only internally for grouping) it ends on
    night_end = (t + pd.to_timedelta(24 - w["night_start_hour"], unit="h")).dt.floor("D")
    rp = g["rest_proxy"].astype("float")
    summary = rp.where(night).groupby(night_end.where(night)).mean()
    end_time = summary.index + pd.to_timedelta(w["night_end_hour"], unit="h")
    s = pd.Series(summary.to_numpy(), index=end_time).sort_index()
    m = (
        pd.merge_asof(
            pd.DataFrame({"t": t, "_i": g.index}).sort_values("t"),
            pd.DataFrame({"t": s.index, "v": s.to_numpy()}),
            on="t",
            direction="backward",
        )
        .set_index("_i")
        .reindex(g.index)
    )
    return m["v"]


def static_features(static: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Static per-patient features (``sex_male`` derived from ``sex``; med flags as 0/1)."""
    s = static.set_index("patient_id").copy()
    s["sex_male"] = (s["sex"] == "M").astype(float).where(s["sex"].notna())
    out = pd.DataFrame(index=s.index)
    for c in cols:
        out[f"static_{c}"] = pd.to_numeric(s[c].astype("float64"), errors="coerce")
    return out


def period_features(
    g: pd.DataFrame,
    meals: pd.DataFrame,
    step: int,
    n_hours: int,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """All dynamic features for one sorted period grid (CGM, time, meals, wearables)."""
    cg = cgm_features(g, step, n_hours, cfg)
    tm = time_features(g)
    ml = meal_features(g, meals, cfg)
    since = (
        ml["meal_minutes_since_last"]
        if "meal_minutes_since_last" in ml
        else pd.Series(np.nan, index=g.index)
    )
    wr = wearable_features(g, step, n_hours, cfg, since)
    return pd.concat([cg, tm, ml, wr], axis=1)
