"""Mechanistic twin under edits (Part 7, D-050 / D-052).

The Part 4 EKF (same kernels: ``step_jacobian``, ``initial_state``, ``clip_state``) filters the
record up to the anchor with the (edited) inputs at or before t, then rolls the state forward
with PLANNED future inputs: meals moved / placed after t and post-meal walks. With no planned
inputs the rollout equals ``ekf_forecast`` exactly (slowing ratio held, activity 0; tested).
Future walk input = (walk intensity - intensity rest level at t)+ (HR is not edited).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.mech.inputs import Record, activity_input, meal_arrays, rest_level
from src.mech.model import (
    IG,
    STATES,
    bounds_array,
    clip_state,
    initial_state,
    njit,
    simulate,
    step_jacobian,
    theta_from_log,
)


@njit(cache=True)
def filter_rollout(
    th: np.ndarray,
    y: np.ndarray,
    carbs: np.ndarray,
    slow: np.ndarray,
    act: np.ndarray,
    ia: int,
    fc: np.ndarray,
    fs: np.ndarray,
    fa: np.ndarray,
    qd: np.ndarray,
    r_obs: float,
    p0: np.ndarray,
    dt: float,
    nsub: int,
    bnd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EKF over indices <= ia, then open-loop rollout of len(fc) - 1 steps.

    Rollout step j-1 -> j uses slowing ratio fs[j-1] and activity fa[j-1]; fc[j] g of carbs enter
    at step j. Returns (G mean (hmax+1,), predictive var (hmax+1,), filtered G at every index <= ia
    (NaN before the first real reading)).
    """
    n = ia + 1
    hmax = fc.shape[0] - 1
    hist = np.full(n, np.nan)
    gm = np.full(hmax + 1, np.nan)
    gv = np.full(hmax + 1, np.nan)
    first = 0
    while first < n and np.isnan(y[first]):
        first += 1
    if first >= n:
        return gm, gv, hist
    x = initial_state(y[first], th)
    pm = np.diag(p0)
    for i in range(first, n):
        if i > first:
            x, f = step_jacobian(x, th, slow[i - 1], act[i - 1], dt, nsub, bnd)
            pm = f @ pm @ f.T + qd
        x[0] += carbs[i]
        if not np.isnan(y[i]):
            s = pm[IG, IG] + r_obs
            kg = pm[:, IG] / s
            x = x + kg * (y[i] - x[IG])
            pm = pm - np.outer(kg, kg) * s
            pm = 0.5 * (pm + pm.T)
            clip_state(x, bnd)
        hist[i] = x[IG]
    xf = x.copy()
    pf = pm.copy()
    gm[0] = xf[IG]
    gv[0] = pf[IG, IG] + r_obs
    for j in range(1, hmax + 1):
        xf, f = step_jacobian(xf, th, fs[j - 1], fa[j - 1], dt, nsub, bnd)
        pf = f @ pf @ f.T + qd
        xf[0] += fc[j]
        gm[j] = xf[IG]
        gv[j] = pf[IG, IG] + r_obs
    return gm, gv, hist


@dataclass
class TwinSpec:
    """Twin of one patient: log parameters, EKF (q, r) and the Part 4 config."""

    logp: np.ndarray
    q_scale: float
    r: float
    mc: dict[str, Any]

    def theta(self, alpha: float) -> np.ndarray:
        """Kernel parameter vector with activity effect alpha."""
        return theta_from_log(self.logp, self.mc["params"]["fixed"], alpha)


def run(
    spec: TwinSpec,
    rec: Record,
    g: pd.DataFrame,
    meals: pd.DataFrame,
    walks: list[tuple[pd.Timestamp, pd.Timestamp]],
    t: pd.Timestamp,
    alpha: float,
    wc: dict[str, Any],
    rollout_min: float,
    act: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Twin forecast at anchor t (`act`: precomputed Part 4 activity input of `g`).

    ``meals`` = meals known at t plus PLANNED meals after t (the caller drops real future meals);
    ``g`` = period grid with walks applied (walk minutes before t enter the filter through the
    Part 4 activity rule); ``walks`` = walk intervals, their part after t is played forward.
    """
    mc, sc = spec.mc, spec.mc["solver"]
    tt = rec.t
    dt = float((tt[1] - tt[0]) / np.timedelta64(1, "m"))
    ia = int(np.searchsorted(tt, np.datetime64(t, "ns")))
    if ia >= len(tt) or tt[ia] != np.datetime64(t, "ns"):
        raise ValueError("anchor not on the grid")
    carbs, slow = meal_arrays(tt, meals, mc["params"]["slow_protein_weight"], None)
    if alpha <= 0:
        act = np.zeros(len(tt))
    elif act is None:
        act = activity_input(g, mc["activity"])
    hmax = int(round(rollout_min / dt))
    fc, fs, fa = np.zeros(hmax + 1), np.full(hmax + 1, slow[ia]), np.zeros(hmax + 1)
    fut = carbs[ia + 1 : ia + hmax + 1]
    fc[1 : 1 + len(fut)] = fut
    fsl = slow[ia : ia + hmax + 1]
    fs[: len(fsl)] = fsl
    if alpha > 0 and walks:
        ac = mc["activity"]
        it = g["activity_intensity"].astype(float).reset_index(drop=True)
        rest = rest_level(it, ac["rest_quantile"], ac["min_rest_samples"],
                          ac["intensity_rest_default"])[ia]  # fmt: skip
        lvl = max(float(wc["intensity"]) - float(rest), 0.0)
        steps = pd.DatetimeIndex(tt[ia : ia + hmax + 1])
        for s, e in walks:
            on = np.asarray((steps >= s) & (steps < e))
            fa[: len(on)] = np.where(on, lvl, fa[: len(on)])
    e = mc["ekf"]
    qd = np.diag([spec.q_scale * e["q_base"][s] for s in STATES])
    p0 = np.array([e["p0"][s] for s in STATES])
    gm, gv, hist = filter_rollout(
        spec.theta(alpha), rec.y, carbs, slow, act, ia, fc, fs, fa, qd, spec.r, p0, dt,
        sc["n_sub"], bounds_array(sc),
    )  # fmt: skip
    return {"g": gm, "var": gv, "hist": hist, "dt": np.array([dt])}


def at_horizons(out: dict[str, np.ndarray], horizons: list[int]) -> dict[int, tuple[float, float]]:
    """(mean, sd) at each horizon (minutes) of a rollout."""
    dt = float(out["dt"][0])
    idx = {h: int(round(h / dt)) for h in horizons}
    return {h: (float(out["g"][i]), float(np.sqrt(out["var"][i]))) for h, i in idx.items()}


def iauc(gl: np.ndarray, dt: float) -> float:
    """Incremental AUC above the starting value (trapezoid of (G - G0)+), mg/dL x min."""
    v = np.maximum(gl - gl[0], 0.0)
    return float(np.sum(0.5 * (v[1:] + v[:-1])) * dt)


def walk_reduction(logp: np.ndarray, mc: dict, wc: dict, ac: dict, alpha: float) -> float:
    """Relative 3-h iAUC reduction by the walk for a patient starting at basal (D-052)."""
    sc = mc["solver"]
    dt = float(sc["dt_min"])
    n = int(ac["iauc_hours"] * 60 / dt) + 1
    th = theta_from_log(logp, mc["params"]["fixed"], alpha)
    carbs = np.zeros(n)
    carbs[0] = ac["meal_g"]
    slow = np.full(n, float(ac["slow_ratio"]))
    res = []
    for walk in (False, True):
        a = np.zeros(n)
        if walk:
            s = int(wc["start_after_meal_min"] / dt)
            a[s : s + int(wc["duration_min"] / dt)] = wc["intensity"]
        x0 = initial_state(float(np.exp(logp[0])), th)
        x = simulate(th, x0, carbs, slow, a, dt, sc["n_sub"], bounds_array(sc))
        res.append(iauc(x[:, IG], dt))
    return 1.0 - res[1] / res[0] if res[0] > 0 else float("nan")


def calibrate_alpha(logp: np.ndarray, mc: dict, wc: dict, ac: dict, target: float) -> float:
    """alpha_E giving a ``target`` iAUC reduction (bisection; NaN if not reached)."""
    lo, hi = 0.0, float(ac["alpha_max"])
    if not walk_reduction(logp, mc, wc, ac, hi) >= target:
        return float("nan")
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if walk_reduction(logp, mc, wc, ac, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
