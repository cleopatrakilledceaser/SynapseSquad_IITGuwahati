"""Conformal prediction primitives (D-045): split conformal, CQR scores, CV+, ACI.

All functions are pure numpy (ACI kernel JIT-compiled when numba is available). Scores are
absolute residuals |y - yhat| (split / CV+) or CQR scores max(lo - y, y - hi).
"""

from __future__ import annotations

import math

import numpy as np

from src.mech.model import njit


def conformal_rank(n: int, alpha: float) -> int:
    """Finite-sample rank ceil((n + 1)(1 - alpha)) (1-based); > n means an infinite quantile."""
    return int(math.ceil((n + 1) * (1.0 - alpha) - 1e-12))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile of the calibration scores (inf if too few scores)."""
    s = np.sort(np.asarray(scores, float)[~np.isnan(scores)])
    r = conformal_rank(len(s), alpha)
    if r > len(s) or len(s) == 0:
        return float("inf")
    return float(s[max(r, 1) - 1])


def cqr_scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CQR conformity scores max(lo - y, y - hi) (Romano et al. 2019)."""
    return np.maximum(lo - y, y - hi)


def _kth_union(m: np.ndarray, res: list[np.ndarray], r: int, iters: int = 60) -> np.ndarray:
    """Per test point, the r-th smallest value of the union over k of {m[:, k] + res[k]}.

    ``res[k]`` sorted ascending; bisection on the value with vectorized counting.
    """
    lo = np.min(m, 1) + min(float(x[0]) for x in res) - 1.0
    hi = np.max(m, 1) + max(float(x[-1]) for x in res) + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cnt = np.zeros(len(mid))
        for k, rk in enumerate(res):
            cnt += np.searchsorted(rk, mid - m[:, k], side="right")
        ok = cnt >= r
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return hi


def cvplus_interval(
    m: np.ndarray, res: list[np.ndarray], alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """CV+ interval (Barber et al. 2021, Sec. 3) for K-fold residuals.

    Args:
        m: (n_test, K) predictions of the K fold models (model k did not see fold k).
        res: K arrays of absolute out-of-fold residuals of fold k's training rows.
        alpha: miscoverage level.

    Lower = floor(alpha (n + 1))-th smallest of {mu_-k(x) - R_i}; upper =
    ceil((1 - alpha)(n + 1))-th smallest of {mu_-k(x) + R_i}; n = total residual count.
    """
    res = [np.sort(np.asarray(r, float)[~np.isnan(r)]) for r in res]
    n = sum(len(r) for r in res)
    r_up = conformal_rank(n, alpha)
    r_lo = int(math.floor(alpha * (n + 1) + 1e-12))
    up = np.full(len(m), np.inf) if r_up > n else _kth_union(m, res, r_up)
    # r_lo-th smallest of {m - R} = -((n - r_lo + 1)-th smallest of {-m + R})
    lo = np.full(len(m), -np.inf) if r_lo < 1 else -_kth_union(-m, res, n - r_lo + 1)
    return lo, up


@njit(cache=True)
def aci_run(
    scores: np.ndarray,
    t_anchor: np.ndarray,
    h: float,
    ref_sorted: np.ndarray,
    alpha: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive conformal inference (Gibbs & Candes 2021) for ONE patient, delayed feedback.

    scores: (n,) conformity scores at anchors sorted by t_anchor (min); NaN = no target.
    The error of anchor s is used (alpha_t <- alpha_t + gamma (alpha - err_s)) only once its
    target time t_s + h <= the anchor being predicted (causal). Returns (q, alpha_t) per anchor;
    q = inf when alpha_t <= 0 (always covered), 0 when alpha_t >= 1.
    """
    n = len(scores)
    m = len(ref_sorted)
    q = np.empty(n)
    at = np.empty(n)
    a = alpha
    tau = t_anchor + h
    order = np.argsort(tau)
    j = 0
    for i in range(n):
        while j < n:
            s = order[j]
            if tau[s] > t_anchor[i]:
                break
            j += 1
            if np.isnan(scores[s]):
                continue
            err = 1.0 if scores[s] > q[s] else 0.0
            a = a + gamma * (alpha - err)
        at[i] = a
        if a <= 0.0:
            q[i] = np.inf
        elif a >= 1.0:
            q[i] = 0.0
        else:
            r = int(math.ceil((m + 1) * (1.0 - a) - 1e-12))
            q[i] = np.inf if r > m else ref_sorted[max(r, 1) - 1]
    return q, at
