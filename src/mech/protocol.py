"""Part 4 fold protocol: everything that is CHOSEN uses training patients only (D-032).

Per outer fold (Part 2/3 CV repeat 0, identical to LightGBM and the static prior):
  1. alpha_E (POPULATION activity effect, never per patient): for each grid value, MAP fits of
     TRAINING patients on their first 3 days (config prior) -> EKF forecasts on their windows
     after day 3 -> lowest mean MAE wins. "off" variant: alpha_E = 0.
  2. Population prior = mean / sd of training patients' 3-day log estimates.
     Static prior = ridge static -> log params on training patients (alpha by LOO).
  3. EKF (q_scale, r) per k: training patients get parameters exactly as an evaluated patient
     would at that k (k = 0: population mean, no fit; k >= 1: MAP on first k days with the
     population prior); the grid value with the lowest mean Gaussian NLL on their windows wins.
  4. Evaluated patients: parameters from ONLY their first k days (none at k = 0). The EKF state
     uses their real CGM readings <= t at every k (nowcasting), like persistence and AR.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from src.mech.fit import Prior, config_prior, fit_patient, population_prior
from src.mech.inputs import Record, anchor_index
from src.mech.model import FIT_PARAMS, STATES, bounds_array, ekf_forecast, theta_from_log
from src.mech.prior import StaticPrior, static_matrix

HORIZONS = [30, 60, 120]


@dataclass
class Ctx:
    """Cohort inputs for the protocol."""

    cohort: str
    mc: dict[str, Any]
    by_pat: dict[str, list[Record]]  # patient -> records sorted by start time
    anchors: pd.DataFrame  # anchor table (patient_id, period_id, t, days_since_start, y_h)
    static: pd.DataFrame
    nsub: int
    free: list[str]
    base_logp: np.ndarray  # log values used for non-free parameters


def ekf_patient(
    c: Ctx, pid: str, logp: np.ndarray, alpha: float, q_scale: float, r: float, stride: int = 1
) -> pd.DataFrame:
    """EKF forecasts (mean, var) at the patient's anchors with day >= test_from_day."""
    e, sc = c.mc["ekf"], c.mc["solver"]
    th = theta_from_log(logp, c.mc["params"]["fixed"], alpha)
    qd = np.diag([q_scale * e["q_base"][s] for s in STATES])
    p0 = np.array([e["p0"][s] for s in STATES])
    out = []
    for rec in c.by_pat[pid]:
        a = c.anchors[c.anchors.period_id == rec.period_id]
        a = a[a.days_since_start >= c.mc["test_from_day"]].iloc[::stride]
        if a.empty:
            continue
        dt = float((rec.t[1] - rec.t[0]) / np.timedelta64(1, "m"))
        hs = np.array([int(round(h / dt)) for h in HORIZONS])
        ai = anchor_index(rec, a["t"].to_numpy())
        bnd = bounds_array(sc)
        m, v = ekf_forecast(
            th, rec.y, rec.carbs, rec.slow, rec.act, ai, hs, qd, r, p0, dt, c.nsub, bnd
        )
        o = a[["patient_id", "period_id", "t"]].copy()
        for j, h in enumerate(HORIZONS):
            o[f"mu_{h}"], o[f"var_{h}"] = m[:, j], v[:, j]
        out.append(o)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def merge_free(c: Ctx, mu: np.ndarray) -> np.ndarray:
    """Prior centre for free parameters, fixed base values for the others."""
    lp = c.base_logp.copy()
    fi = [FIT_PARAMS.index(p) for p in c.free]
    lp[fi] = mu[fi]
    return lp


def fit_pids(
    c: Ctx, pids: list[str], k: float, prior_of: Callable[[str], Prior], alpha: float
) -> dict[str, np.ndarray]:
    """Log parameters per patient: MAP on the first k days of its first record (k=0: prior)."""
    out = {}
    for p in pids:
        pr = prior_of(p)
        start = merge_free(c, pr.mu)
        if k <= 0:
            out[p] = start
            continue
        fr = fit_patient(c.by_pat[p][0], k, pr, c.mc, alpha, c.free, start, nsub=c.nsub)
        out[p] = fr.logp
    return out


def score(c: Ctx, f: pd.DataFrame, metric: str) -> float:
    """Mean over horizons of MAE or Gaussian NLL on the given forecasts."""
    ycols = [f"y_{h}" for h in HORIZONS]
    d = f.merge(c.anchors[["period_id", "t"] + ycols], on=["period_id", "t"])
    vals = []
    for h in HORIZONS:
        y, m, v = d[f"y_{h}"], d[f"mu_{h}"], d[f"var_{h}"]
        ok = y.notna() & m.notna()
        y, m, v = y[ok], m[ok], v[ok]
        if metric == "mae":
            vals.append(float(np.mean(np.abs(m - y))))
        else:
            vals.append(float(np.mean(0.5 * np.log(2 * np.pi * v) + (y - m) ** 2 / (2 * v))))
    return float(np.mean(vals))


def pooled_forecasts(
    c: Ctx, pids: list[str], lps: dict[str, np.ndarray], alpha: float, q: float, r: float
) -> pd.DataFrame:
    """Strided EKF forecasts for several (training) patients."""
    stride = c.mc["ekf"]["train_eval_stride"]
    return pd.concat([ekf_patient(c, p, lps[p], alpha, q, r, stride) for p in pids])


def select_alpha(c: Ctx, train: list[str]) -> tuple[float, dict[str, float]]:
    """Population alpha_E by training-patient forecast MAE (default Q/R, 3-day fits)."""
    e, cp = c.mc["ekf"], config_prior(c.mc["params"])
    q, r = e["default"]["q_scale"], e["default"]["r"]
    scores = {}
    for a in c.mc["activity"]["alpha_grid"]:
        lps = fit_pids(c, train, c.mc["fit"]["stage1_days"], lambda _p: cp, a)
        scores[str(a)] = score(c, pooled_forecasts(c, train, lps, a, q, r), "mae")
    return float(min(scores, key=scores.get)), scores


def select_qr(
    c: Ctx, train: list[str], lps: dict[str, np.ndarray], alpha: float
) -> tuple[float, float, dict[str, float]]:
    """(q_scale, r) minimizing mean Gaussian NLL on training patients' windows."""
    e = c.mc["ekf"]
    scores = {}
    for q in e["q_scale_grid"]:
        for r in e["r_grid"]:
            scores[f"{q}|{r}"] = score(c, pooled_forecasts(c, train, lps, alpha, q, r), "nll")
    q, r = (float(v) for v in min(scores, key=scores.get).split("|"))
    return q, r, scores


def fix_to_population(c: Ctx, pop: Prior, fix: list[str]) -> Ctx:
    """Ctx where the parameters in ``fix`` are not fitted but set to the population mean (D-037).

    ``pop`` comes from TRAINING patients' stage-1 fits, so no evaluated patient is involved.
    """
    fix = [p for p in fix if p in c.free]
    if not fix:
        return c
    base = c.base_logp.copy()
    fi = [FIT_PARAMS.index(p) for p in fix]
    base[fi] = pop.mu[fi]
    return replace(c, free=[p for p in c.free if p not in fix], base_logp=base)


def run_fold(
    c0: Ctx,
    fold: int,
    train: list[str],
    test: list[str],
    ks: list[int],
    fix: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """All twin variants for one fold -> (forecasts long table, choices, test-patient params).

    ``fix``: parameters set to the training-population mean after stage 1 instead of being fitted
    per patient (Part 4b revision, D-037). None = original specification.
    """
    c = c0
    mc, cp = c.mc, config_prior(c.mc["params"])
    floor, feats = mc["fit"]["prior_sd_floor"], mc["prior"]["static_features"]
    choices: dict[str, Any] = {"fold": fold, "n_train": len(train), "n_test": len(test)}
    preds, fits = [], []
    variants: dict[str, float | None] = {"on": None, "off": 0.0}
    if c.cohort != "cgmacros":
        variants = {"off": 0.0}  # no wearable stream
    s_test = dict(zip(test, static_matrix(c.static, test, feats), strict=True))
    for act, alpha in variants.items():
        c = c0  # stage 1 / alpha_E always fit every free parameter
        if alpha is None:
            alpha, choices["alpha_scores"] = select_alpha(c, train)
        choices[f"alpha_{act}"] = alpha
        st1 = fit_pids(c, train, mc["fit"]["stage1_days"], lambda _p: cp, alpha)
        tr_lp = np.array([st1[p] for p in train])
        pop = population_prior(tr_lp, floor)
        sp = StaticPrior(mc["prior"]["ridge_alphas"], floor).fit(
            static_matrix(c.static, train, feats), tr_lp
        )
        choices[f"ridge_alpha_{act}"] = dict(zip(FIT_PARAMS, sp.chosen_alpha, strict=True))
        choices[f"pop_prior_{act}"] = {"mu": pop.mu.tolist(), "sd": pop.sd.tolist()}
        c = fix_to_population(c0, pop, fix or [])
        choices[f"free_{act}"] = list(c.free)
        pop_of: Callable[[str], Prior] = lambda _p, _pop=pop: _pop  # noqa: E731
        static_of: Callable[[str], Prior] = lambda p, _sp=sp: _sp.prior_for(s_test[p])  # noqa: E731
        for k in ks:
            tr_k = fit_pids(c, train, k, pop_of, alpha)
            q, r, qs = select_qr(c, train, tr_k, alpha)
            choices[f"qr_{act}_k{k}"] = {"q_scale": q, "r": r, "scores": qs}
            inits = {"pop": pop_of, "static": static_of}
            for init, prior_of in inits.items():
                lps = fit_pids(c, test, k, prior_of, alpha)
                name = f"twin_{init}_act{act}_k{k}"
                for p in test:
                    f = ekf_patient(c, p, lps[p], alpha, q, r)
                    f["model"] = name
                    preds.append(f)
                    row = {"patient_id": p, "model": name, "fold": fold, "k": k}
                    fits.append(row | dict(zip(FIT_PARAMS, np.exp(lps[p]), strict=True)))
    return pd.concat(preds, ignore_index=True), choices, pd.DataFrame(fits)
