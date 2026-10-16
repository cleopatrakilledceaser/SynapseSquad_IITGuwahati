"""Faulted test frames (D-047): inject -> recompute causal features -> same anchors / targets.

Targets and the anchor set are the CLEAN ones (what really happened); only model inputs change.
Stale rule: if the anchor's own reading was removed, the last real reading <= stale_max_min old
is used as ``cgm_now`` (else the system gives no forecast; ``available`` = False).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.build import period_features
from src.robustness.faults import inject
from src.robustness.quality import dq_components

TARGET_PREFIXES = ("y_", "hyper180_", "hyper140_", "hypo70_")


def rebuild(
    clean_rows: pd.DataFrame,
    g: pd.DataFrame,
    meals: pd.DataFrame,
    step: int,
    n_h: int,
    fcfg: dict[str, Any],
    dqc: dict[str, Any],
    clip: tuple[float, float],
    stale_max: float,
) -> pd.DataFrame:
    """Recompute features of ``clean_rows`` (one period, its anchors) from grid ``g``."""
    g = g.reset_index(drop=True)
    f = period_features(g, meals, step, n_h, fcfg)
    f = pd.concat([f, dq_components(g, step, dqc, clip)], axis=1)
    real = g["glucose_is_real"].astype(bool)
    last = g["glucose"].where(real).ffill()
    f["_last_real"] = last.to_numpy()
    f["_since"] = g["minutes_since_real"].to_numpy()
    f["t"] = g["t"].to_numpy()
    keep = [c for c in clean_rows.columns if c.startswith(TARGET_PREFIXES) or c not in f.columns]
    keep = list(dict.fromkeys(["t", *keep]))
    d = clean_rows[keep].merge(f, on="t", how="left", validate="one_to_one")
    miss = d["cgm_now"].isna()
    stale = miss & (d["_since"] <= stale_max)
    d.loc[stale, "cgm_now"] = d.loc[stale, "_last_real"]
    d["available"] = ~miss | stale
    return d.drop(columns=["_last_real", "_since"])


def faulted_frame(
    te: pd.DataFrame,
    grid: pd.DataFrame,
    meals: pd.DataFrame,
    fault: str,
    level: float,
    spec: dict[str, Any],
    ctx: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Faulted copy of the test frame ``te`` (first-period rows of test patients)."""
    parts = []
    for pid, rows in te.groupby("period_id"):
        g = grid[grid.period_id == pid].sort_values("t")
        gf = inject(g, fault, level, ctx["tau"], ctx["step"], spec, ctx["clip"], seed)
        m = meals[meals.period_id == pid]
        parts.append(
            rebuild(
                rows,
                gf,
                m,
                ctx["step"],
                ctx["n_h"],
                ctx["fcfg"],
                ctx["dqc"],
                ctx["clip"],
                ctx["stale_max"],
            )
        )
    return pd.concat(parts, ignore_index=True)
