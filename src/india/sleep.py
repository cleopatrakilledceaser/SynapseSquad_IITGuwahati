"""Simulated sleep, HRV and step channel (Part 8d, D-064). ALL SYNTHETIC.

Only the sleep DURATION affects glucose: from the wake time of each night, insulin sensitivity SI
is multiplied by ``exp(-beta * m * deficit)`` (deficit = baseline - duration, never negative).
Efficiency and deep-sleep fraction are independent draws (no signal by construction); HRV is a
noisy function of the deficit. The simulation is run in segments between wake times with exact
state continuity, so with ``m = 0`` it reproduces the single-run simulation bit for bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.mech.model import bounds_array, initial_state, rk4_step, simulate, theta_from_log

STEP_MIN = 5.0


def draw_nights(
    rng: np.random.Generator, n_nights: int, scfg: dict[str, Any]
) -> dict[str, np.ndarray]:
    """One night of sleep before each day: duration, efficiency, deep fraction, HRV (RMSSD, ms)."""

    def tn(c: dict[str, float]) -> np.ndarray:
        return np.clip(rng.normal(c["mean"], c["sd"], n_nights), c["lo"], c["hi"])

    dur = tn(scfg["duration"])
    eff, deep = tn(scfg["efficiency"]), tn(scfg["deep_frac"])
    h = scfg["hrv"]
    deficit = np.maximum(scfg["baseline_h"] - dur, 0.0)
    hrv = h["mean_ms"] * np.exp(
        -h["per_hour_deficit"] * deficit + rng.normal(0, h["noise_sd_log"], n_nights)
    )
    return {"sleep_h": dur, "sleep_eff": eff, "sleep_deep": deep, "hrv_rmssd": hrv}


def si_shift(sleep_h: np.ndarray, scfg: dict[str, Any], mult: float) -> np.ndarray:
    """Log shift of SI after each night: -beta * mult * deficit (<= 0)."""
    deficit = np.maximum(scfg["baseline_h"] - sleep_h, 0.0)
    return -scfg["si_beta_per_hour"] * mult * deficit


def wake_steps(n_nights: int, scfg: dict[str, Any]) -> list[int]:
    """Grid index (5-min) at which each night's effect starts (wake time of day k)."""
    return [int((scfg["wake_minute"] + 1440 * k) // STEP_MIN) for k in range(n_nights)]


def simulate_segmented(
    lp: np.ndarray, fixed: Any, alpha: float, carbs: np.ndarray, slow: np.ndarray, act: np.ndarray,
    solver: dict[str, Any], shifts: np.ndarray, si_index: int, bounds: tuple[np.ndarray, np.ndarray],
    edges: list[int],
) -> np.ndarray:  # fmt: skip
    """Open-loop trajectory with a per-segment SI shift.

    ``edges`` are the start indices of the segments after the first; ``shifts[j]`` is the log
    shift of segment j + 1 (segment 0 always has none).
    """
    n = carbs.shape[0]
    bnd, dt, nsub = bounds_array(solver), solver["dt_min"], solver["n_sub"]
    lo, hi = bounds
    out = np.empty((n, 6))
    starts = [0, *edges]
    ends = [*edges, n]
    for j, (s, e) in enumerate(zip(starts, ends, strict=True)):
        if e <= s:
            continue
        lpj = lp.copy()
        if j > 0:
            lpj[si_index] += shifts[j - 1]
            lpj = np.clip(lpj, lo, hi)
        th = theta_from_log(lpj, fixed, alpha)
        if j == 0:
            x0 = initial_state(float(np.exp(lp[0])), th)
        else:
            x0 = rk4_step(out[s - 1], th, slow[s - 1], act[s - 1], dt, nsub, bnd)
        out[s:e] = simulate(th, x0, carbs[s:e], slow[s:e], act[s:e], dt, nsub, bnd)
    return out


def steps_from_activity(
    rng: np.random.Generator, act: np.ndarray, scfg: dict[str, Any]
) -> np.ndarray:
    """Steps per 5 minutes from the simulated activity input (zero when the input is zero)."""
    s = scfg["steps"]
    raw = act * s["per_activity_unit_5min"] + rng.normal(0.0, s["noise_sd"], len(act))
    return np.where(act > 0, np.maximum(np.round(raw), 0.0), 0.0)
