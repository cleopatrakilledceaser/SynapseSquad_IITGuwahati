"""Forecasting baselines (Part 2).

Each baseline exposes ``fit(train_tab, grid, val=..., horizons=...)`` and
``predict(tab, grid, horizons) -> {horizon: np.ndarray}`` on anchor tables. Inputs are causal
features or the period's grid history at or before each anchor; no baseline reads values after t.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


def history_matrix(tab: pd.DataFrame, grid: pd.DataFrame, n_lags: int) -> np.ndarray:
    """Real-reading glucose at t, t-1, ..., t-(n_lags-1) grid steps for each anchor (NaN if absent).

    Built by positional lookup within each period's sorted grid; only rows at or before t.
    """
    out = np.full((len(tab), n_lags), np.nan)
    for pid, rows in tab.groupby("period_id").indices.items():
        g = grid[grid.period_id == pid]
        v = g["glucose"].where(g["glucose_is_real"]).to_numpy()
        pos = np.searchsorted(g["t"].to_numpy(), tab["t"].to_numpy()[rows])
        for j in range(n_lags):
            p = pos - j
            ok = p >= 0
            col = np.full(len(rows), np.nan)
            col[ok] = v[p[ok]]
            out[rows, j] = col
    return out


@dataclass
class Persistence:
    """y(t+h) = glucose(t)."""

    name: str = "persistence"

    def fit(self, tab: pd.DataFrame, grid: pd.DataFrame, **_: Any) -> Persistence:
        return self

    def predict(self, tab: pd.DataFrame, grid: pd.DataFrame, horizons: list[int]) -> dict:
        return {h: tab["cgm_now"].to_numpy(dtype=float) for h in horizons}


@dataclass
class LinearTrend:
    """y(t+h) = glucose(t) + d * h * slope_w; (w, d) tuned per horizon on inner validation."""

    windows: list[int]
    dampings: list[float]
    name: str = "linear_trend"
    best_w: dict = field(default_factory=dict)

    def fit(self, tab, grid, val=None, horizons=None, **_):  # type: ignore[no-untyped-def]
        for h in horizons:
            errs = {}
            for w in self.windows:
                for d in self.dampings:
                    pred = val["cgm_now"] + d * h * val[f"cgm_slope_{w}"].fillna(0.0)
                    errs[(w, d)] = float(np.nanmean(np.abs(pred - val[f"y_{h}"])))
            self.best_w[h] = min(errs, key=errs.get)
        return self

    def predict(self, tab, grid, horizons):  # type: ignore[no-untyped-def]
        out = {}
        for h in horizons:
            w, d = self.best_w[h]
            out[h] = (tab["cgm_now"] + d * h * tab[f"cgm_slope_{w}"].fillna(0.0)).to_numpy()
        return out


def holt_forecast(
    x: np.ndarray, alpha: float, beta: float, phi: float, steps: np.ndarray
) -> np.ndarray:
    """Vectorised damped-trend Holt filter over rows of ``x`` (oldest -> newest; NaN = skip)."""
    level = x[:, 0].copy()
    trend = np.zeros(len(x))
    for j in range(1, x.shape[1]):
        obs = x[:, j]
        ok = ~np.isnan(obs) & ~np.isnan(level)
        new_level = alpha * obs + (1 - alpha) * (level + phi * trend)
        new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = np.where(ok, new_level, np.where(np.isnan(level), obs, level + phi * trend))
        trend = np.where(ok, new_trend, phi * trend)
    out = {}
    for k in steps:
        damp = np.sum(phi ** np.arange(1, k + 1))
        out[int(k)] = level + damp * trend
    return out  # type: ignore[return-value]


@dataclass
class HoltES:
    """Damped-trend exponential smoothing on each anchor's own recent history (tuned globally).

    The recursion matches statsmodels Holt(damped_trend=True) with fixed parameters (tested).
    """

    history_min: int
    alphas: list[float]
    betas: list[float]
    phis: list[float]
    step: int = 5
    name: str = "exp_smoothing"
    best: dict = field(default_factory=dict)

    def _hist(self, tab, grid):  # type: ignore[no-untyped-def]
        return history_matrix(tab, grid, self.history_min // self.step + 1)[:, ::-1]

    def fit(self, tab, grid, val=None, horizons=None, **_):  # type: ignore[no-untyped-def]
        x = self._hist(val, grid)
        steps = np.array([h // self.step for h in horizons])
        for h in horizons:
            best, best_err = None, np.inf
            for a in self.alphas:
                for b in self.betas:
                    for p in self.phis:
                        f = holt_forecast(x, a, b, p, steps)[h // self.step]
                        e = float(np.nanmean(np.abs(f - val[f"y_{h}"].to_numpy())))
                        if e < best_err:
                            best, best_err = (a, b, p), e
            self.best[h] = best
        return self

    def predict(self, tab, grid, horizons):  # type: ignore[no-untyped-def]
        x = self._hist(tab, grid)
        out = {}
        for h in horizons:
            a, b, p = self.best[h]
            out[h] = holt_forecast(x, a, b, p, np.array([h // self.step]))[h // self.step]
        return out


def _ar_design(x_hist: np.ndarray, p: int) -> np.ndarray:
    """Lag matrix [1, y_t, y_{t-1}, ..., y_{t-p+1}] from newest-first history."""
    return np.column_stack([np.ones(len(x_hist)), x_hist[:, :p]])


@dataclass
class PooledAR:
    """Direct AR(p) per horizon, pooled over training patients, fitted with statsmodels OLS."""

    orders_min: list[int]
    step: int = 5
    max_rows: int = 300000
    seed: int = 42
    name: str = "ar_pooled"
    coef: dict = field(default_factory=dict)
    order: dict = field(default_factory=dict)

    def _fit_one(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ok = ~np.isnan(x).any(axis=1) & ~np.isnan(y)
        xs, ys = x[ok], y[ok]
        if len(ys) > self.max_rows:
            pick = np.random.default_rng(self.seed).choice(len(ys), self.max_rows, replace=False)
            xs, ys = xs[pick], ys[pick]
        return sm.OLS(ys, xs).fit().params

    def fit(self, tab, grid, val=None, horizons=None, **_):  # type: ignore[no-untyped-def]
        pmax = max(self.orders_min) // self.step
        xt = history_matrix(tab, grid, pmax)
        xv = history_matrix(val, grid, pmax)
        for h in horizons:
            errs = {}
            for om in self.orders_min:
                p = om // self.step
                c = self._fit_one(_ar_design(xt, p), tab[f"y_{h}"].to_numpy())
                pred = _ar_design(xv, p) @ c
                pred = np.where(np.isnan(pred), val["cgm_now"].to_numpy(), pred)
                errs[om] = float(np.nanmean(np.abs(pred - val[f"y_{h}"].to_numpy())))
            self.order[h] = min(errs, key=errs.get)
            p = self.order[h] // self.step
            self.coef[h] = self._fit_one(_ar_design(xt, p), tab[f"y_{h}"].to_numpy())
        return self

    def predict(self, tab, grid, horizons):  # type: ignore[no-untyped-def]
        pmax = max(self.order.values()) // self.step
        x = history_matrix(tab, grid, pmax)
        out = {}
        for h in horizons:
            p = self.order[h] // self.step
            pred = _ar_design(x, p) @ self.coef[h]
            out[h] = np.where(np.isnan(pred), tab["cgm_now"].to_numpy(), pred)  # fallback
        return out


@dataclass
class PatientAR:
    """Per-patient direct AR, updated CAUSALLY and shrunk towards the pooled AR.

    For an anchor at time t of patient i, the coefficients are a ridge solution fitted ONLY on
    patient i's training pairs (x_s, y_{s+h}) with s + h <= t, i.e. whose target was already
    observed at t, with a ridge prior centred on the pooled coefficients:
        beta_i(t) = (X'X + lambda I)^-1 (X'y + lambda beta_pooled).
    With fewer than ``min_pairs`` such pairs the pooled coefficients are used. No future data.
    """

    pooled: PooledAR
    ridge_lambda: float
    min_pairs: int
    step: int = 5
    name: str = "ar_per_patient"

    def fit(self, tab, grid, val=None, horizons=None, **_):  # type: ignore[no-untyped-def]
        self.pooled.fit(tab, grid, val=val, horizons=horizons)
        return self

    def _patient_grid_pairs(self, g: pd.DataFrame, h: int, p: int) -> tuple[np.ndarray, ...]:
        """All (x_s, y_{s+h}, time of target) pairs in one period's grid (real readings only)."""
        v = g["glucose"].where(g["glucose_is_real"]).to_numpy()
        k = h // self.step
        n = len(v)
        idx = np.arange(p - 1, n - k)
        x = np.column_stack([np.ones(len(idx))] + [v[idx - j] for j in range(p)])
        y = v[idx + k]
        t_target = g["t"].to_numpy()[idx + k]
        ok = ~np.isnan(x).any(axis=1) & ~np.isnan(y)
        return x[ok], y[ok], t_target[ok]

    def predict(self, tab, grid, horizons):  # type: ignore[no-untyped-def]
        base = self.pooled.predict(tab, grid, horizons)
        out = {h: base[h].copy() for h in horizons}
        lam = self.ridge_lambda
        for h in horizons:
            p = self.pooled.order[h] // self.step
            b0 = self.pooled.coef[h]
            x_hist = history_matrix(tab, grid, p)
            design = _ar_design(x_hist, p)
            for pid, rows in tab.groupby("patient_id").indices.items():
                g = grid[grid.patient_id == pid]
                xs, ys, ts = [], [], []
                for _, gp in g.groupby("period_id"):
                    x, y, tt = self._patient_grid_pairs(gp.sort_values("t"), h, p)
                    xs.append(x)
                    ys.append(y)
                    ts.append(tt)
                if not xs:
                    continue
                x, y, tt = np.vstack(xs), np.concatenate(ys), np.concatenate(ts)
                order = np.argsort(tt)
                x, y, tt = x[order], y[order], tt[order]
                anchor_t = tab["t"].to_numpy()[rows]
                n_avail = np.searchsorted(tt, anchor_t, side="right")  # pairs with target <= t
                # cumulative sufficient statistics => exact causal ridge at each anchor
                xx = np.cumsum(np.einsum("ni,nj->nij", x, x), axis=0)
                xy = np.cumsum(x * y[:, None], axis=0)
                eye = np.eye(x.shape[1])
                for r, na in zip(rows, n_avail, strict=True):
                    if na < self.min_pairs or np.isnan(design[r]).any():
                        continue
                    a = xx[na - 1] + lam * eye
                    b = xy[na - 1] + lam * b0
                    out[h][r] = design[r] @ np.linalg.solve(a, b)
        return out
