"""Edits of a meal log / activity grid and causal ML input recompute (Part 7, D-050).

Edits never touch glucose. The ML inputs that depend on meals or activity (``meal_*`` and
``act_post_meal``) are recomputed with the unchanged Part 2 code
(``src.features.build.period_features``) on the period grid prefix up to t. CGM, time-of-day
and static inputs cannot change under an edit, so they are taken from the stored anchor row.
A zero edit must reproduce the stored row (guarded in ``check_identity``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.features.build import meal_features, period_features, wearable_features

MACROS = ("carbs_g", "protein_g", "fat_g", "fiber_g")


@dataclass(frozen=True)
class Edit:
    """One edit of a logged meal (``meal`` = index label in the meals table).

    kind: ``macros`` (absolute grams in ``values``), ``remove``, ``move`` (``shift_min``) or
    ``walk`` (post-meal walk; timing / intensity from configs/whatif.yaml ``walk``).
    """

    kind: str
    meal: Any
    values: tuple[tuple[str, float], ...] = ()
    shift_min: float = 0.0


def set_macros(meal: Any, **grams: float) -> Edit:
    """Set macros of a meal to absolute values in grams (kcal follows, D-050)."""
    bad = [k for k in grams if k not in MACROS]
    if bad:
        raise ValueError(f"unknown macros {bad}")
    return Edit("macros", meal, tuple(sorted(grams.items())))


def remove_meal(meal: Any) -> Edit:
    """Remove a logged meal."""
    return Edit("remove", meal)


def move_meal(meal: Any, minutes: float) -> Edit:
    """Move a logged meal by ``minutes`` (positive = later)."""
    return Edit("move", meal, shift_min=float(minutes))


def add_walk(meal: Any) -> Edit:
    """Add a post-meal walk after a logged meal."""
    return Edit("walk", meal)


def new_kcal(kcal: float, old: dict[str, float], new: dict[str, float], kpg: dict) -> float:
    """New kcal = max(0, kcal + sum_k kcal_per_g[k] * (new_k - old_k)); NaN kcal stays NaN."""
    if pd.isna(kcal):
        return float("nan")
    d = sum(kpg[k] * (new[k] - (0.0 if pd.isna(old[k]) else old[k])) for k in new)
    return max(0.0, float(kcal) + d)


def apply_meals(meals: pd.DataFrame, edits: Sequence[Edit], kcal_per_g: dict) -> pd.DataFrame:
    """Edited copy of one period's meal log (walk edits do not change the log)."""
    m = meals.copy()
    for e in edits:
        if e.kind == "macros":
            new = dict(e.values)
            old = {k: m.loc[e.meal, k] for k in new}
            m.loc[e.meal, "kcal"] = new_kcal(m.loc[e.meal, "kcal"], old, new, kcal_per_g)
            for k, v in new.items():
                m.loc[e.meal, k] = float(v)
        elif e.kind == "remove":
            m = m.drop(index=e.meal)
        elif e.kind == "move":
            m.loc[e.meal, "t"] = m.loc[e.meal, "t"] + pd.Timedelta(minutes=e.shift_min)
        elif e.kind != "walk":
            raise ValueError(f"unknown edit {e.kind}")
    return m.sort_values("t", kind="stable")


def walk_interval(meal_t: pd.Timestamp, wc: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) of the walk after a meal at ``meal_t``."""
    s = meal_t + pd.Timedelta(minutes=wc["start_after_meal_min"])
    return s, s + pd.Timedelta(minutes=wc["duration_min"])


def apply_walks(
    g: pd.DataFrame, meals: pd.DataFrame, edits: Sequence[Edit], wc: dict
) -> pd.DataFrame:
    """Grid copy with activity_intensity = max(existing, intensity) during each walk."""
    walks = [e for e in edits if e.kind == "walk"]
    if not walks:
        return g
    g = g.copy()
    for e in walks:
        s, en = walk_interval(meals.loc[e.meal, "t"], wc)
        w = ((g["t"] >= s) & (g["t"] < en)).to_numpy()
        cur = g.loc[w, "activity_intensity"].to_numpy(float)
        g.loc[w, "activity_intensity"] = np.fmax(cur, float(wc["intensity"]))
        if "activity_missing" in g:
            g.loc[w, "activity_missing"] = False
    return g


def edited_columns(cols: Sequence[str]) -> list[str]:
    """Model inputs that an edit can change (meal features and post-meal activity)."""
    return [c for c in cols if c.startswith("meal_") or c == "act_post_meal"]


def recompute(
    rows: pd.DataFrame,
    g: pd.DataFrame,
    meals: pd.DataFrame,
    cols: Sequence[str],
    ctx: dict[str, Any],
) -> pd.DataFrame:
    """Copies of ``rows`` (anchor rows of ONE period) with edit-dependent inputs recomputed.

    ``g`` = the period's (possibly walk-edited) grid; ``meals`` = its (possibly edited) meal log;
    ctx: fcfg, step, n_h.
    """
    out = rows.copy()
    # period prefix up to max(t), exactly as Part 2 built the table (pandas rolling sums depend
    # on where the series starts, so a shorter slice differs by ~1e-16 and can flip a split)
    gs = g[g["t"] <= rows["t"].max()].reset_index(drop=True)
    # same Part 2 functions as period_features (CGM / time features cannot change under an edit)
    ml = meal_features(gs, meals, ctx["fcfg"])
    since = ml.get("meal_minutes_since_last", pd.Series(np.nan, index=gs.index))
    wr = wearable_features(gs, ctx["step"], ctx["n_h"], ctx["fcfg"], since)
    f = pd.concat([ml, wr], axis=1)
    f.index = pd.DatetimeIndex(gs["t"])
    f = f.reindex(pd.DatetimeIndex(rows["t"]))
    for c in edited_columns(cols):
        out[c] = f[c].to_numpy(float) if c in f else np.nan
    return out


def recompute_full(
    rows: pd.DataFrame, g: pd.DataFrame, meals: pd.DataFrame, cols: Sequence[str], ctx: dict
) -> pd.DataFrame:
    """Reference path: every feature from ``period_features`` (tests check it equals recompute)."""
    out = rows.copy()
    # period prefix up to max(t), exactly as Part 2 built the table (pandas rolling sums depend
    # on where the series starts, so a shorter slice differs by ~1e-16 and can flip a split)
    gs = g[g["t"] <= rows["t"].max()].reset_index(drop=True)
    f = period_features(gs, meals, ctx["step"], ctx["n_h"], ctx["fcfg"])
    f.index = pd.DatetimeIndex(gs["t"])
    f = f.reindex(pd.DatetimeIndex(rows["t"]))
    for c in edited_columns(cols):
        out[c] = f[c].to_numpy(float) if c in f else np.nan
    return out


def max_diff(a: pd.DataFrame, b: pd.DataFrame, cols: Sequence[str]) -> float:
    """Max |a - b| over cols (NaN-equal; NaN vs number = inf)."""
    x, y = a[list(cols)].to_numpy(float), b[list(cols)].to_numpy(float)
    nan_x, nan_y = np.isnan(x), np.isnan(y)
    if (nan_x != nan_y).any():
        return float("inf")
    d = np.abs(np.where(nan_x, 0.0, x - y))
    return float(d.max()) if d.size else 0.0


def check_identity(
    rows: pd.DataFrame, g: pd.DataFrame, meals: pd.DataFrame, cols: Sequence[str], ctx: dict
) -> float:
    """Zero-edit recompute vs stored rows; raises if above ctx['tol'] (D-050 guard)."""
    d = max_diff(recompute(rows, g, meals, cols, ctx), rows, edited_columns(cols))
    if not d <= ctx["tol"]:
        raise RuntimeError(f"zero-edit features differ from stored rows by {d}")
    return d
