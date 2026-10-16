"""Mechanistic glucose model (Part 4): gut + minimal model + insulin surrogate + activity.

Numba-compiled kernels with a pure-numpy fallback (same code, D-031).

States (mg/dL, g, 1/min):
    Q1, Q2  two-compartment gut (g of carbohydrate)
    G       plasma/interstitial glucose (mg/dL; observed by CGM, sensor lag ignored)
    U       insulin SURROGATE (mg/dL-equivalent; insulin is not measured, see limitation)
    X       remote insulin action (1/min)
    E       smoothed activity above the patient's own rest (dimensionless)

Equations (k = kabs / (1 + beta * r), r = slowing ratio of the most recent meal):
    Q1' = -k Q1                                (meals enter Q1 as impulses at their time)
    Q2' =  k (Q1 - Q2)
    Ra  =  fc k Q2                             glucose appearance, mg/dL/min
    G'  = -(SG + aE E)(G - Gb) - X G + Ra      activity speeds return to basal
    U'  = (-U + softplus_s(G - Gb) + cR Ra) / tauI
    X'  =  p2 (-X + SI (U - U0)),  U0 = softplus_s(0)  => rest fixed point G = Gb, X = 0
    E'  = (A - E) / tauE                       A = activity input above rest

Limitation: U is a glucose- and meal-driven surrogate of insulin secretion. It cannot represent
exogenous insulin, sulfonylurea effects or beta-cell failure beyond what SI/SG absorb.
"""

from __future__ import annotations

import math

import numpy as np

try:  # numba verified to import and compile on the dev box (D-031); fallback keeps it portable
    from numba import njit
except ImportError:  # pragma: no cover

    def njit(*args, **kwargs):  # type: ignore[no-untyped-def]
        """No-op decorator used when numba is unavailable."""
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


STATES = ("Q1", "Q2", "G", "U", "X", "E")
FIT_PARAMS = ("Gb", "SG", "SI", "fc", "kabs", "beta")
THETA = FIT_PARAMS + ("aE", "p2", "tauI", "cR", "tauE", "soft_s")
IG = 2  # index of G in the state vector
LN2 = math.log(2.0)


def make_theta(
    fit: dict[str, float], fixed: dict[str, float], alpha_e: float, onesided: bool = False
) -> np.ndarray:
    """Full parameter vector in kernel order (THETA).

    ``onesided`` (D-056, post-hoc, opt-in): appends a 13th element = 1, which switches the
    activity term to -aE E max(G - Gb, 0). Default: the original 12-element vector (Parts 4-6).
    """
    v = {**fit, **fixed, "aE": alpha_e}
    th = [float(v[k]) for k in THETA]
    onesided = onesided or bool(fixed.get("act_onesided", 0))  # config route (D-056 re-runs)
    return np.array(th + [1.0] if onesided else th)


def theta_from_log(
    logp: np.ndarray, fixed: dict[str, float], alpha_e: float, onesided: bool = False
) -> np.ndarray:
    """Kernel vector from log fitted parameters (FIT_PARAMS order)."""
    fit = dict(zip(FIT_PARAMS, np.exp(logp), strict=True))
    return make_theta(fit, fixed, alpha_e, onesided)


def bounds_array(solver: dict) -> np.ndarray:
    """[g_lo, g_hi, x_lo, x_hi] state bounds from configs/mech.yaml solver section."""
    return np.array([*solver["g_bounds"], *solver["x_bounds"]], dtype=float)


@njit(cache=True)
def softplus(z: float, s: float) -> float:
    """Smooth max(z, 0) with width s (numerically stable)."""
    u = z / s
    if u > 30.0:
        return z
    return s * math.log1p(math.exp(u))


@njit(cache=True)
def deriv(x: np.ndarray, th: np.ndarray, r: float, a: float, out: np.ndarray) -> None:
    """Right-hand side of the ODE at state x with slowing ratio r and activity input a."""
    gb, sg, si, fc, kabs, beta = th[0], th[1], th[2], th[3], th[4], th[5]
    ae, p2, taui, cr, taue, s = th[6], th[7], th[8], th[9], th[10], th[11]
    k = kabs / (1.0 + beta * r)
    ra = fc * k * x[1]
    out[0] = -k * x[0]
    out[1] = k * (x[0] - x[1])
    if th.shape[0] > 12 and th[12] > 0.0:
        # D-056 (post-hoc): activity only adds uptake above basal; no effect at or below Gb
        out[2] = -sg * (x[2] - gb) - ae * x[5] * max(x[2] - gb, 0.0) - x[4] * x[2] + ra
    else:
        out[2] = -(sg + ae * x[5]) * (x[2] - gb) - x[4] * x[2] + ra
    out[3] = (-x[3] + softplus(x[2] - gb, s) + cr * ra) / taui
    out[4] = p2 * (-x[4] + si * (x[3] - s * LN2))
    out[5] = (a - x[5]) / taue


@njit(cache=True)
def clip_state(x: np.ndarray, bnd: np.ndarray) -> None:
    """Keep states in their physical ranges (in place)."""
    x[0] = max(x[0], 0.0)
    x[1] = max(x[1], 0.0)
    x[2] = min(max(x[2], bnd[0]), bnd[1])
    x[3] = max(x[3], 0.0)
    x[4] = min(max(x[4], bnd[2]), bnd[3])
    x[5] = max(x[5], 0.0)


@njit(cache=True)
def rk4_step(
    x: np.ndarray, th: np.ndarray, r: float, a: float, dt: float, nsub: int, bnd: np.ndarray
) -> np.ndarray:
    """Advance one grid step (dt minutes, nsub RK4 sub-steps); inputs held constant."""
    h = dt / nsub
    y = x.copy()
    k1 = np.empty(6)
    k2 = np.empty(6)
    k3 = np.empty(6)
    k4 = np.empty(6)
    for _ in range(nsub):
        deriv(y, th, r, a, k1)
        deriv(y + 0.5 * h * k1, th, r, a, k2)
        deriv(y + 0.5 * h * k2, th, r, a, k3)
        deriv(y + h * k3, th, r, a, k4)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        clip_state(y, bnd)
    return y


@njit(cache=True)
def initial_state(g0: float, th: np.ndarray) -> np.ndarray:
    """Rest state at glucose g0: empty gut, surrogate at its quasi-steady level, X = 0."""
    x = np.zeros(6)
    x[2] = g0
    x[3] = softplus(g0 - th[0], th[11])
    return x


@njit(cache=True)
def simulate(
    th: np.ndarray,
    x0: np.ndarray,
    carbs: np.ndarray,
    slow: np.ndarray,
    act: np.ndarray,
    dt: float,
    nsub: int,
    bnd: np.ndarray,
) -> np.ndarray:
    """Open-loop trajectory (n, 6). Row i = state at grid time i AFTER meal impulses at i.

    Causality: carbs[i] (meals in (t_{i-1}, t_i]) enter at i; slow[i] / act[i] are known at t_i
    and drive the step i -> i+1. Nothing after t_i affects row i.
    """
    n = carbs.shape[0]
    out = np.empty((n, 6))
    x = x0.copy()
    x[0] += carbs[0]
    out[0] = x
    for i in range(1, n):
        x = rk4_step(x, th, slow[i - 1], act[i - 1], dt, nsub, bnd)
        x[0] += carbs[i]
        out[i] = x
    return out


@njit(cache=True)
def sim_segments(
    th: np.ndarray,
    g_obs: np.ndarray,
    carbs: np.ndarray,
    slow: np.ndarray,
    act: np.ndarray,
    seg: int,
    dt: float,
    nsub: int,
    bnd: np.ndarray,
) -> np.ndarray:
    """Simulated G with G reset to the observed reading at the start of every ``seg`` steps.

    Used for fitting: each segment is an open-loop forecast from a known glucose value (other
    states carried over), which matches the forecasting task and avoids long-horizon drift.
    """
    n = carbs.shape[0]
    g = np.empty(n)
    first = 0
    while first < n and np.isnan(g_obs[first]):
        first += 1
    g0 = g_obs[first] if first < n else th[0]
    x = initial_state(g0, th)
    x[0] += carbs[0]
    g[0] = x[2]
    for i in range(1, n):
        x = rk4_step(x, th, slow[i - 1], act[i - 1], dt, nsub, bnd)
        x[0] += carbs[i]
        if i % seg == 0 and not np.isnan(g_obs[i]):
            x[2] = g_obs[i]
        g[i] = x[2]
    return g


@njit(cache=True)
def step_jacobian(
    x: np.ndarray, th: np.ndarray, r: float, a: float, dt: float, nsub: int, bnd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One-step map f(x) and its forward-difference Jacobian F = df/dx."""
    fx = rk4_step(x, th, r, a, dt, nsub, bnd)
    jac = np.empty((6, 6))
    for j in range(6):
        e = 1e-6 + 1e-5 * abs(x[j])
        xp = x.copy()
        xp[j] += e
        jac[:, j] = (rk4_step(xp, th, r, a, dt, nsub, bnd) - fx) / e
    return fx, jac


@njit(cache=True)
def ekf_forecast(
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
) -> tuple[np.ndarray, np.ndarray]:
    """Extended Kalman filter over a record + forecasts at each (sorted) anchor index.

    Measurement: y_i = G_i + v, v ~ N(0, r_obs), used only where y is not NaN (real readings).
    At anchor i (after the update with y_i) the state is propagated open-loop to max(hsteps)
    with NO future inputs: no new meals, activity input 0, slowing ratio held at slow[i].
    Returns forecast mean and predictive variance (P_GG + r_obs), shape (n_anchors, n_h).
    """
    n = y.shape[0]
    na = anchors.shape[0]
    nh = hsteps.shape[0]
    hmax = hsteps[nh - 1]
    mean = np.full((na, nh), np.nan)
    var = np.full((na, nh), np.nan)
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
            kg = pm[:, IG] / s
            x = x + kg * (y[i] - x[IG])
            pm = pm - np.outer(kg, kg) * s
            pm = 0.5 * (pm + pm.T)
            clip_state(x, bnd)
        while ai < na and anchors[ai] == i:
            xf = x.copy()
            pf = pm.copy()
            hj = 0
            for j in range(1, hmax + 1):
                xf, f = step_jacobian(xf, th, slow[i], 0.0, dt, nsub, bnd)
                pf = f @ pf @ f.T + qd
                if j == hsteps[hj]:
                    mean[ai, hj] = xf[IG]
                    var[ai, hj] = pf[IG, IG] + r_obs
                    hj += 1
            ai += 1
    return mean, var
