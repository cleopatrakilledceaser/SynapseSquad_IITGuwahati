"""Synthetic patients from the mechanistic twin (Part 4). EVERYTHING here is SYNTHETIC.

- Parameters: multivariate normal on log scale, fitted to real per-patient MAP estimates
  (twin_pop_actoff_k3 fits from the CV run), clipped to the config bounds.
- Schedules: meal / activity days resampled from real CGMacros records (whole days, so meal
  timing, macros and activity stay realistic together).
- Sensor noise: Gaussian white noise + AR(1) drift (configurable; default estimated by the caller).
- ``phenotype``: "default" only. "indian" is a documented placeholder (Part 11), not implemented.
Outputs carry ``synthetic = True`` and ids prefixed ``synthetic_``; never mixed with real data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.mech.fit import log_bounds
from src.mech.inputs import Record
from src.mech.model import FIT_PARAMS, bounds_array, initial_state, simulate, theta_from_log

STEPS_PER_DAY = 288  # 5-min grid


def param_distribution(fits: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of log parameters across real patients."""
    lg = np.log(fits[list(FIT_PARAMS)].to_numpy(dtype=float))
    return lg.mean(0), np.cov(lg, rowvar=False)


def day_library(recs: list[Record]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Complete real days (carbs, slow, act), each 288 steps."""
    lib = []
    for r in recs:
        for d in range(len(r.t) // STEPS_PER_DAY):
            s = slice(d * STEPS_PER_DAY, (d + 1) * STEPS_PER_DAY)
            lib.append((r.carbs[s], r.slow[s], r.act[s]))
    return lib


def ar1_noise(
    n: int, white_sd: float, ar_sd: float, phi: float, rng: np.random.Generator
) -> np.ndarray:
    """White noise + stationary AR(1) drift with marginal sd ``ar_sd``."""
    e = rng.normal(0, ar_sd * np.sqrt(1 - phi**2), n)
    d = np.empty(n)
    d[0] = rng.normal(0, ar_sd)
    for i in range(1, n):
        d[i] = phi * d[i - 1] + e[i]
    return d + rng.normal(0, white_sd, n)


def simulate_cohort(
    fits: pd.DataFrame,
    recs: list[Record],
    mc: dict[str, Any],
    alpha: float,
    noise: dict[str, float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(timeseries, params) for ``simulate.n_patients`` SYNTHETIC patients of ``simulate.days``."""
    sc, sim = mc["solver"], mc["simulate"]
    if sim["phenotype"] != "default":
        raise NotImplementedError("phenotype 'indian' is a documented placeholder (Part 11)")
    rng = np.random.default_rng(seed)
    mu, cov = param_distribution(fits)
    lo, hi = log_bounds(mc["params"])
    lib = day_library(recs)
    rows, prm = [], []
    for i in range(sim["n_patients"]):
        lp = np.clip(rng.multivariate_normal(mu, cov), lo, hi)
        days = [lib[j] for j in rng.integers(0, len(lib), sim["days"])]
        carbs, slow, act = (np.concatenate([d[k] for d in days]) for k in range(3))
        th = theta_from_log(lp, mc["params"]["fixed"], alpha)
        x0 = initial_state(float(np.exp(lp[0])), th)
        x = simulate(th, x0, carbs, slow, act, sc["dt_min"], sc["n_sub"], bounds_array(sc))
        g = x[:, 2] + ar1_noise(len(x), noise["white_sd"], noise["ar_sd"], noise["phi"], rng)
        g = np.clip(g, *mc["solver"]["g_bounds"])
        pid = f"synthetic_{i:03d}"
        rows.append(
            pd.DataFrame(
                {
                    "patient_id": pid,
                    "step": np.arange(len(g)),
                    "minutes": np.arange(len(g)) * sc["dt_min"],
                    "glucose": g,
                    "glucose_true": x[:, 2],
                    "carbs_g": carbs,
                    "activity_input": act,
                    "synthetic": True,
                }
            )
        )
        prm.append(
            {"patient_id": pid, "synthetic": True} | dict(zip(FIT_PARAMS, np.exp(lp), strict=True))
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(prm)


def realism_stats(g: pd.Series, pid: pd.Series, carbs: pd.Series | None = None) -> pd.DataFrame:
    """Per-patient mean, SD, CV, %<70, %70-140, %140-180, %>180, and post-meal peak / timing."""
    d = pd.DataFrame({"g": g.to_numpy(), "pid": pid.to_numpy()})
    if carbs is not None:
        d["c"] = carbs.to_numpy()
    out = []
    for p, x in d.groupby("pid", sort=False):
        v = x["g"].dropna().to_numpy()
        row = {
            "patient_id": p,
            "mean": v.mean(),
            "sd": v.std(),
            "cv_pct": 100 * v.std() / v.mean(),
            "pct_lt70": 100 * (v < 70).mean(),
            "pct_70_140": 100 * ((v >= 70) & (v <= 140)).mean(),
            "pct_140_180": 100 * ((v > 140) & (v <= 180)).mean(),
            "pct_gt180": 100 * (v > 180).mean(),
        }
        if carbs is not None:
            gg, cc = x["g"].to_numpy(), x["c"].to_numpy()
            peaks, tpk = [], []
            for i in np.flatnonzero(cc >= 20):
                w = gg[i : i + 37]  # 3 h after the meal (5-min grid)
                if len(w) == 37 and not np.isnan(w[0]) and np.isnan(w).mean() < 0.3:
                    peaks.append(np.nanmax(w) - w[0])
                    tpk.append(5 * int(np.nanargmax(w)))
            row["meal_peak_rise"] = float(np.median(peaks)) if peaks else np.nan
            row["meal_time_to_peak_min"] = float(np.median(tpk)) if tpk else np.nan
        out.append(row)
    return pd.DataFrame(out)
