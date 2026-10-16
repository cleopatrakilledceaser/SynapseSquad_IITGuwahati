"""Causal data-quality score per grid time (D-047): completeness, recency, plausibility, ROC.

Every sub-score at row i uses rows <= i only (trailing windows; tested).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def dq_components(
    g: pd.DataFrame, step: int, cfg: dict[str, Any], clip: tuple[float, float]
) -> pd.DataFrame:
    """Sub-scores in [0, 1] and their mean ``dq`` for one sorted period grid."""
    real = g["glucose_is_real"].astype(bool)
    y = g["glucose"].where(real)
    n6 = int(cfg["completeness_window_h"] * 60 // step)
    comp = real.astype(float).rolling(n6, min_periods=1).sum() / n6
    since = g["minutes_since_real"].astype(float)
    rec = (1.0 - since / cfg["recency_full_min"]).clip(0.0, 1.0).fillna(0.0)
    nw = max(int(cfg["plaus_window_min"] // step), 1)
    at_bound = ((y <= clip[0]) | (y >= clip[1])).astype(float).where(real)
    plaus = 1.0 - at_bound.rolling(nw, min_periods=1).mean().fillna(0.0)
    # rate of change between consecutive REAL readings (mg/dL/min), trailing max
    idx = pd.Series(np.arange(len(g)), index=g.index).where(real).ffill()
    prev_val = y.ffill().shift(1)
    prev_idx = idx.shift(1)
    dt = (pd.Series(np.arange(len(g)), index=g.index) - prev_idx) * step
    roc = ((y - prev_val).abs() / dt).where(real)
    rmax = roc.rolling(nw, min_periods=1).max().fillna(0.0)
    ok, bad = cfg["roc_ok_mgdl_min"], cfg["roc_bad_mgdl_min"]
    rs = (1.0 - (rmax - ok) / (bad - ok)).clip(0.0, 1.0)
    out = pd.DataFrame(
        {"dq_complete": comp, "dq_recency": rec, "dq_plaus": plaus, "dq_roc": rs}, index=g.index
    )
    out["dq"] = out.mean(axis=1)
    return out
