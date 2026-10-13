"""Comparators on the Part 4 test windows (day >= 3), refit per fold on training patients.

- ``ar_k{k}``: pooled AR (order chosen on an inner patient split of the training patients),
  then a per-patient ridge refit on EXACTLY the patient's first k days (pairs whose TARGET time
  is within the first k days of the first period), shrunk to the pooled coefficients. The
  coefficients are frozen after day k (no causal updating), so AR and twin see the same data.
- LightGBM: population model, no data of the evaluated patient (src.models.fusion_lgbm).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.baselines import PatientAR, PooledAR, _ar_design, history_matrix
from src.models.experiment import inner_split

HORIZONS = [30, 60, 120]


def ar_first_k(
    tab: pd.DataFrame,
    grid: pd.DataFrame,
    train: list[str],
    test: list[str],
    ks: list[int],
    strata: dict[str, str],
    bc: dict[str, Any],
    lam: float,
    min_pairs: int,
    step: int,
    test_from_day: float,
) -> pd.DataFrame:
    """Predictions ar_k<k>|h for test patients' anchors with day >= test_from_day."""
    in_tr, in_va = inner_split(train, strata, bc, 42)
    pooled = PooledAR(bc["ar"]["order_min_grid"], step=step, max_rows=bc["ar"]["max_fit_rows"])
    pooled.fit(
        tab[tab.patient_id.isin(in_tr)],
        grid,
        val=tab[tab.patient_id.isin(in_va)],
        horizons=HORIZONS,
    )
    helper = PatientAR(pooled, lam, min_pairs, step=step)
    ev = tab[tab.patient_id.isin(test) & (tab.days_since_start >= test_from_day)]
    ev = ev.reset_index(drop=True)
    out = ev[["patient_id", "period_id", "t"]].copy()
    for h in HORIZONS:
        p = pooled.order[h] // step
        b0 = pooled.coef[h]
        design = _ar_design(history_matrix(ev, grid, p), p)
        for k in ks:
            pred = np.full(len(ev), np.nan)
            for pid, rows in ev.groupby("patient_id").indices.items():
                g = grid[grid.patient_id == pid]
                first = g[g.period_id == g.sort_values("t").period_id.iloc[0]].sort_values("t")
                x, y, tt = helper._patient_grid_pairs(first, h, p)
                ok = tt < first["t"].iloc[0].to_datetime64() + np.timedelta64(int(k * 1440), "m")
                x, y = x[ok], y[ok]
                beta = (
                    np.linalg.solve(x.T @ x + lam * np.eye(len(b0)), x.T @ y + lam * b0)
                    if len(y) >= min_pairs
                    else b0
                )
                pred[rows] = design[rows] @ beta
            pred = np.where(np.isnan(pred), ev["cgm_now"].to_numpy(), pred)  # Part 2 fallback
            out[f"ar_k{k}|{h}"] = pred
    return out
