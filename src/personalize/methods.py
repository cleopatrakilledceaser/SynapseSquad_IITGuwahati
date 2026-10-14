"""Personalization methods (Part 5.2, D-043): small, causal, per-patient adapters.

All functions take ONLY the evaluated patient's adaptation pairs (anchors whose TARGET time
t + h is before day k of the first period, real reading at t + h) and return corrections for the
evaluation anchors. Nothing here sees data after day k except ``rls_run`` in continuous mode,
which uses a pair only once its target time t + h is <= the anchor time it predicts (causal).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.mech.model import njit

MIN_PER_DAY = 1440.0


def adapt_mask(days: np.ndarray, h: int, k: float, y: np.ndarray) -> np.ndarray:
    """Adaptation pairs at level k: target time (day + h / 1440) < k and real target."""
    return (days + h / MIN_PER_DAY < k) & ~np.isnan(y)


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """Centered ridge (unpenalized intercept): returns (intercept, coefficients)."""
    xm, ym = x.mean(0), y.mean()
    xc = x - xm
    beta = np.linalg.solve(xc.T @ xc + lam * np.eye(x.shape[1]), xc.T @ (y - ym))
    return float(ym - xm @ beta), beta


def ridge_predict(model: tuple[float, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Prediction of a ``ridge_fit`` model."""
    return model[0] + x @ model[1]


def shrink_weight(n: int, n0: float) -> float:
    """Blend weight of the per-patient model: n / (n + n0) (0 with no data)."""
    return n / (n + n0) if n > 0 else 0.0


@njit(cache=True)
def rls_run(
    x: np.ndarray,
    resid: np.ndarray,
    t_anchor: np.ndarray,
    h: float,
    limit: float,
    lam: float,
    p0: float,
    ph_delta: float,
    ph_lambda: float,
    boost: float,
) -> tuple[np.ndarray, int]:
    """Recursive least squares on residuals with forgetting + Page-Hinkley drift boost.

    x: (n, d) features at anchors sorted by time t_anchor (min); resid: (n,) y(t + h) - pop
    forecast (NaN = no pair). Pair s becomes available at t_anchor[s] + h and is used only if that
    time is <= the anchor being predicted AND < ``limit`` (frozen-at-k mode: limit = k days;
    continuous mode: limit = inf). Returns (correction at every anchor, number of drift alarms).
    Drift: Page-Hinkley on the absolute a-priori residual; an alarm multiplies P by ``boost``.
    Regularization: theta starts at 0 (= the population model) with P = p0 * I (prior variance in
    units of the noise variance, standardized features); trace(P) is capped at d * p0, so
    forgetting / drift boosts never make the learner less regularized than its prior (no
    covariance wind-up).
    """
    n, d = x.shape
    theta = np.zeros(d)
    pm = np.eye(d) * p0
    pmax = d * p0
    corr = np.zeros(n)
    tau = t_anchor + h
    order = np.argsort(tau)
    j = 0
    m_ph, m_min, mean_abs, n_seen, alarms = 0.0, 0.0, 0.0, 0, 0
    for i in range(n):
        while j < n:
            s = order[j]
            if tau[s] > t_anchor[i] or tau[s] >= limit:
                break
            j += 1
            if np.isnan(resid[s]):
                continue
            xs = x[s]
            e = resid[s] - xs @ theta
            px = pm @ xs
            g = px / (lam + xs @ px)
            theta = theta + g * e
            pm = (pm - np.outer(g, px)) / lam
            n_seen += 1
            mean_abs += (abs(e) - mean_abs) / n_seen
            m_ph += abs(e) - mean_abs - ph_delta
            m_min = min(m_min, m_ph)
            if m_ph - m_min > ph_lambda:
                pm = pm * boost
                m_ph, m_min = 0.0, 0.0
                alarms += 1
            tr = np.trace(pm)
            if tr > pmax:
                pm = pm * (pmax / tr)
        corr[i] = x[i] @ theta
    return corr, alarms


def finetune_params(base: dict[str, Any], ft: dict[str, Any], lr: float) -> dict[str, Any]:
    """LightGBM training parameters for continued boosting on the adaptation pairs."""
    return {
        "objective": "regression",
        "learning_rate": lr,
        "num_leaves": ft["num_leaves"],
        "min_child_samples": ft["min_child_samples"],
        "subsample": base["subsample"],
        "subsample_freq": base["subsample_freq"],
        "colsample_bytree": base["colsample_bytree"],
        "reg_lambda": base["reg_lambda"],
        "num_threads": ft["threads"],
        "seed": 42,
        "verbose": -1,
        "feature_pre_filter": False,
    }


def finetune(
    init_str: str, n_init: int, xa: Any, ya: np.ndarray, params: dict[str, Any], t_max: int
) -> Any:
    """Continue boosting the (trimmed) population booster for ``t_max`` trees on (xa, ya).

    ``ya`` is the population model's target (y - base). Returns the new booster; the extra
    contribution of the first T new trees is ``predict(x, start_iteration=n_init,
    num_iteration=T)`` (raw scale, added to the population forecast).
    """
    import lightgbm as lgb

    init = lgb.Booster(model_str=init_str)
    ds = lgb.Dataset(xa, ya, free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=t_max, init_model=init)
