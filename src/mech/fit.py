"""Per-patient MAP fitting, priors (population / static ridge) and Laplace posteriors (Part 4).

MAP objective (log parameters p, free subset only):
    sum_i ((G_sim_i(p) - y_i) / sigma)^2 + sum_j ((p_j - mu_j) / sd_j)^2
over REAL readings in the patient's first k days, with G_sim from 6-h segments reset to the
observed reading (model.sim_segments). Bounds are on the log scale (configs/mech.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from src.mech.inputs import Record
from src.mech.model import FIT_PARAMS, sim_segments, theta_from_log


@dataclass
class Prior:
    """Gaussian prior on log parameters (FIT_PARAMS order)."""

    mu: np.ndarray
    sd: np.ndarray


@dataclass
class FitResult:
    """MAP estimate (log scale), Laplace covariance of the free parameters, diagnostics."""

    logp: np.ndarray
    cov: np.ndarray | None
    free: list[str]
    cost: float
    n_obs: int
    rmse: float
    success: bool
    extra: dict[str, Any] = field(default_factory=dict)


def config_prior(pcfg: dict[str, Any]) -> Prior:
    """Weak prior from configs/mech.yaml (stage 1)."""
    f = pcfg["fit"]
    return Prior(
        np.log([f[k]["mean"] for k in FIT_PARAMS]), np.array([f[k]["sd"] for k in FIT_PARAMS])
    )


def log_bounds(pcfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Log-scale lower / upper bounds (FIT_PARAMS order)."""
    f = pcfg["fit"]
    return np.log([f[k]["lo"] for k in FIT_PARAMS]), np.log([f[k]["hi"] for k in FIT_PARAMS])


def first_days(rec: Record, k: float) -> int:
    """Number of grid points in the first k days of the record."""
    return int(np.searchsorted(rec.days, k, side="left"))


def fit_patient(
    rec: Record,
    k: float,
    prior: Prior,
    mcfg: dict[str, Any],
    alpha_e: float,
    free: list[str] | None = None,
    base_logp: np.ndarray | None = None,
    nsub: int | None = None,
) -> FitResult:
    """MAP fit on the record's first k days (real readings only). Non-free params = base_logp."""
    pc, fc, sc = mcfg["params"], mcfg["fit"], mcfg["solver"]
    dt = float((rec.t[1] - rec.t[0]) / np.timedelta64(1, "m"))
    nsub = nsub or sc["n_sub"]
    bnd = np.array([*sc["g_bounds"], *sc["x_bounds"]], dtype=float)
    free = free or list(FIT_PARAMS)
    fi = np.array([FIT_PARAMS.index(p) for p in free])
    lo, hi = log_bounds(pc)
    base = (base_logp if base_logp is not None else prior.mu).copy()
    n = first_days(rec, k)
    y = rec.y[:n]
    obs = ~np.isnan(y)
    if obs.sum() < 10:  # nothing to fit: return the prior centre
        return FitResult(base, None, free, float("nan"), int(obs.sum()), float("nan"), False)
    seg = max(1, int(round(fc["segment_hours"] * 60 / dt)))
    sig = fc["sigma_obs"]
    carbs, slow, act = rec.carbs[:n], rec.slow[:n], rec.act[:n]

    def resid(z: np.ndarray) -> np.ndarray:
        lp = base.copy()
        lp[fi] = z
        th = theta_from_log(lp, pc["fixed"], alpha_e)
        g = sim_segments(th, y, carbs, slow, act, seg, dt, nsub, bnd)
        return np.concatenate([(g[obs] - y[obs]) / sig, (z - prior.mu[fi]) / prior.sd[fi]])

    z0 = np.clip(base[fi], lo[fi] + 1e-6, hi[fi] - 1e-6)
    r = least_squares(resid, z0, bounds=(lo[fi], hi[fi]), method="trf", max_nfev=fc["max_nfev"])
    lp = base.copy()
    lp[fi] = r.x
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac)
    except np.linalg.LinAlgError:
        cov = None
    nobs = int(obs.sum())
    rmse = float(np.sqrt(np.mean((r.fun[:nobs] * sig) ** 2)))
    return FitResult(lp, cov, free, float(r.cost), nobs, rmse, bool(r.success))


def population_prior(logps: np.ndarray, sd_floor: float) -> Prior:
    """Mean / sd of training patients' log estimates (sd floored)."""
    return Prior(logps.mean(axis=0), np.maximum(logps.std(axis=0, ddof=1), sd_floor))


def contraction(fr: FitResult, prior: Prior) -> np.ndarray:
    """1 - posterior var / prior var per free parameter (Laplace); NaN if no covariance."""
    if fr.cov is None:
        return np.full(len(fr.free), np.nan)
    fi = [FIT_PARAMS.index(p) for p in fr.free]
    return 1.0 - np.diag(fr.cov) / prior.sd[fi] ** 2
