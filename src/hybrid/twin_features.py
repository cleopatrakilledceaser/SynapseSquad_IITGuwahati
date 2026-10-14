"""Causal twin-derived features at each forecast time t (Part 5, D-038).

``ekf_features`` is the Part 4 EKF (src.mech.model.ekf_forecast, unchanged) extended to also
return, at every anchor t (after the update with the reading at t): the filtered state
(Q1, Q2, G, U, X, E), the innovation y_i - G_pred,i and its variance at every grid step, and the
open-loop rollout of (X, Ra, S_eff = SG + aE * E) from t to t + hmax (physics loss of Hybrid C).
Forecast mean / variance equal ``ekf_forecast`` (tested). Everything at anchor index i uses inputs
with index <= i only (tested by perturbing the future).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.mech.inputs import Record, anchor_index
from src.mech.model import (
    IG,
    STATES,
    bounds_array,
    clip_state,
    initial_state,
    njit,
    step_jacobian,
    theta_from_log,
)

HORIZONS = [30, 60, 120]


@njit(cache=True)
def _ra(x: np.ndarray, th: np.ndarray, r: float) -> float:
    """Glucose appearance fc * k * Q2 (mg/dL/min) at state x and slowing ratio r."""
    k = th[4] / (1.0 + th[5] * r)
    return th[3] * k * x[1]


@njit(cache=True)
def ekf_features(
    th: np.ndarray,
    y: np.ndarray,
    carbs: np.ndarray,
    slow: np.ndarray,
    act: np.ndarray,
    anchors: np.ndarray,
    hsteps: np.ndarray,
    qd: np.ndarray,
    r_obs: float,
    p0: np.ndarray,
    dt: float,
    nsub: int,
    bnd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EKF over a record -> (mean, var, state, innov, innov_var, traj).

    mean / var: (n_anchors, n_h) exactly as ``ekf_forecast``; state: (n_anchors, 6) filtered state
    at the anchor; innov / innov_var: (n,) per grid step (NaN where no real reading);
    traj: (n_anchors, hmax + 1, 3) rollout of X, Ra, S_eff (index 0 = the anchor state).
    """
    n = y.shape[0]
    na = anchors.shape[0]
    nh = hsteps.shape[0]
    hmax = hsteps[nh - 1]
    mean = np.full((na, nh), np.nan)
    var = np.full((na, nh), np.nan)
    state = np.full((na, 6), np.nan)
    innov = np.full(n, np.nan)
    ivar = np.full(n, np.nan)
    traj = np.full((na, hmax + 1, 3), np.nan)
    first = 0
    while first < n and np.isnan(y[first]):
        first += 1
    x = initial_state(y[first] if first < n else th[0], th)
    pm = np.diag(p0)
    ai = 0
    while ai < na and anchors[ai] < first:
        ai += 1
    for i in range(first, n):
        if i > first:
            x, f = step_jacobian(x, th, slow[i - 1], act[i - 1], dt, nsub, bnd)
            pm = f @ pm @ f.T + qd
        x[0] += carbs[i]
        if not np.isnan(y[i]):
            s = pm[IG, IG] + r_obs
            innov[i] = y[i] - x[IG]
            ivar[i] = s
            kg = pm[:, IG] / s
            x = x + kg * (y[i] - x[IG])
            pm = pm - np.outer(kg, kg) * s
            pm = 0.5 * (pm + pm.T)
            clip_state(x, bnd)
        while ai < na and anchors[ai] == i:
            state[ai] = x
            traj[ai, 0, 0] = x[4]
            traj[ai, 0, 1] = _ra(x, th, slow[i])
            traj[ai, 0, 2] = th[1] + th[6] * x[5]
            xf = x.copy()
            pf = pm.copy()
            hj = 0
            for j in range(1, hmax + 1):
                xf, f = step_jacobian(xf, th, slow[i], 0.0, dt, nsub, bnd)
                pf = f @ pf @ f.T + qd
                traj[ai, j, 0] = xf[4]
                traj[ai, j, 1] = _ra(xf, th, slow[i])
                traj[ai, j, 2] = th[1] + th[6] * xf[5]
                if j == hsteps[hj]:
                    mean[ai, hj] = xf[IG]
                    var[ai, hj] = pf[IG, IG] + r_obs
                    hj += 1
            ai += 1
    return mean, var, state, innov, ivar, traj


def _causal_mean(v: np.ndarray, w: int) -> np.ndarray:
    """Mean of the non-NaN values in [i - w + 1, i] (NaN if none): uses no future values."""
    return pd.Series(v).rolling(w, min_periods=1).mean().to_numpy()


def record_features(
    rec: Record,
    anchors_t: np.ndarray,
    logp: np.ndarray,
    mc: dict[str, Any],
    alpha: float,
    q_scale: float,
    r: float,
    nsub: int,
    hcfg: dict[str, Any],
    keep_traj: bool = False,
) -> pd.DataFrame:
    """Twin features (tw_*) at the given anchor times of one record (sorted grid times)."""
    e, sc = mc["ekf"], mc["solver"]
    th = theta_from_log(logp, mc["params"]["fixed"], alpha)
    qd = np.diag([q_scale * e["q_base"][s] for s in STATES])
    p0 = np.array([e["p0"][s] for s in STATES])
    dt = float((rec.t[1] - rec.t[0]) / np.timedelta64(1, "m"))
    hs = np.array([int(round(h / dt)) for h in HORIZONS])
    ai = anchor_index(rec, anchors_t)
    m, v, st, inn, iv, traj = ekf_features(
        th, rec.y, rec.carbs, rec.slow, rec.act, ai, hs, qd, r, p0, dt, nsub, bounds_array(sc)
    )
    w_in = max(1, int(round(hcfg["innov_window_min"] / dt)))
    w_ac = max(1, int(round(hcfg["act_window_min"] / dt)))
    inn_last = pd.Series(inn).ffill().to_numpy()  # most recent innovation at or before i
    z = pd.Series(inn / np.sqrt(iv)).ffill().to_numpy()
    o = pd.DataFrame({"period_id": rec.period_id, "t": anchors_t})
    for j, h in enumerate(HORIZONS):
        o[f"tw_mu_{h}"] = m[:, j]
        o[f"tw_sd{h}"] = np.sqrt(v[:, j])
    for k, name in enumerate(STATES):
        o[f"tw_{name}" if name != "G" else "tw_Gf"] = st[:, k]
    o["tw_Ra"] = traj[:, 0, 1]
    o["tw_innov"] = inn_last[ai]
    o["tw_innov_z"] = z[ai]
    o["tw_innov_mean_1h"] = _causal_mean(inn, w_in)[ai]
    o["tw_act"] = rec.act[ai]
    o["tw_act_mean_1h"] = _causal_mean(rec.act, w_ac)[ai]
    if keep_traj:
        cols = {}
        for j in range(traj.shape[1]):
            cols[f"tr_X_{j}"] = traj[:, j, 0]
            cols[f"tr_Ra_{j}"] = traj[:, j, 1]
            cols[f"tr_S_{j}"] = traj[:, j, 2]
        cols["tr_Gb"] = np.full(len(o), th[0])
        o = pd.concat([o, pd.DataFrame(cols)], axis=1)
    return o


def add_derived(tab: pd.DataFrame) -> pd.DataFrame:
    """Forecast deltas and filter residual relative to the current CGM value (cgm_now)."""
    tab = tab.copy()
    for h in HORIZONS:
        tab[f"tw_d{h}"] = tab[f"tw_mu_{h}"] - tab["cgm_now"]
    tab["tw_gres"] = tab["tw_Gf"] - tab["cgm_now"]
    return tab
