"""Part 3 data: feature-group resolution, static re-join, causal simulated fingersticks.

Reuses the cached Part 2 anchor tables (data/processed/features/<cohort>_N<h>.parquet). Static
columns are re-joined from static.parquet so new static fields (e.g. Shanghai kidney function)
are available without rebuilding the dynamic features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import is_calendar_column
from src.features.build import static_features
from src.utils.config import PROJECT_ROOT

PROC = PROJECT_ROOT / "data" / "processed"


def resolve_group(spec: dict[str, Any], cols: list[str]) -> list[str]:
    """Columns of one feature group (prefix / exact / static), in table order."""
    out = []
    prefixes, exclude = tuple(spec.get("prefix", [])), spec.get("exclude", [])
    for c in cols:
        by_prefix = bool(prefixes) and c.startswith(prefixes) and c not in exclude
        if by_prefix or c in spec.get("exact", []):
            out.append(c)
    out += [f"static_{s}" for s in spec.get("static", []) if f"static_{s}" in cols]
    return list(dict.fromkeys(out))


def resolve_groups(cohort: str, cfg: dict[str, Any], cols: list[str]) -> dict[str, list[str]]:
    """All feature groups of a cohort -> columns. Groups are disjoint (asserted)."""
    groups = {g: resolve_group(s, cols) for g, s in cfg["groups"][cohort].items()}
    seen: dict[str, str] = {}
    for g, cs in groups.items():
        for c in cs:
            if c in seen:
                raise ValueError(f"column {c} in groups {seen[c]} and {g}")
            seen[c] = g
    return groups


def expand(names: list[str], groups: dict[str, list[str]]) -> list[str]:
    """Group names (with 'static' = every static_* group) -> group names."""
    out = []
    for n in names:
        out += [g for g in groups if g.startswith("static_")] if n == "static" else [n]
    return out


def config_columns(names: list[str], groups: dict[str, list[str]]) -> list[str]:
    """Columns of a config (list of group names); no calendar features allowed."""
    cols = [c for g in expand(names, groups) for c in groups[g]]
    bad = [c for c in cols if is_calendar_column(c)]
    if bad:
        raise ValueError(f"calendar features: {bad}")
    return cols


def drop_groups(cols: list[str], drop: list[str], groups: dict[str, list[str]]) -> list[str]:
    """Remove exactly the columns of the dropped groups."""
    gone = {c for g in drop for c in groups[g]}
    return [c for c in cols if c not in gone]


def all_static_columns(cfg: dict[str, Any], cohort: str) -> list[str]:
    """Every static column named in any static_* group of the cohort."""
    return [
        c
        for g, s in cfg["groups"][cohort].items()
        if g.startswith("static_")
        for c in s.get("static", [])
    ]


def fingerstick_features(
    grid: pd.DataFrame, anchors: pd.DataFrame, times: list[str], lookback_h: float
) -> pd.DataFrame:
    """Causal simulated fingerstick features for each anchor.

    A fingerstick at clock time c on each day of a period equals the REAL CGM reading at c (if the
    grid point at c is real; otherwise that stick is skipped). Only sticks with time <= t are used:
    last value, minutes since it, previous value, mean / count over ``lookback_h`` hours,
    and the change between the last two.
    """
    clock = {(int(s[:2]), int(s[3:])) for s in times}
    g = grid[["period_id", "t", "glucose", "glucose_is_real"]]
    hm = list(zip(g["t"].dt.hour, g["t"].dt.minute, strict=True))
    is_stick = np.array([x in clock for x in hm]) & g["glucose_is_real"].to_numpy()
    sticks = g.loc[is_stick, ["period_id", "t", "glucose"]].sort_values(["period_id", "t"])
    out = pd.DataFrame(index=anchors.index)
    names = (
        "fs_last",
        "fs_minutes_since",
        "fs_prev",
        "fs_delta_last2",
        "fs_mean_lb",
        "fs_count_lb",
    )
    cols = {k: np.full(len(anchors), np.nan) for k in names}
    lb = np.int64(lookback_h * 3600 * 10**9)
    for pid, rows in anchors.groupby("period_id").indices.items():
        s = sticks[sticks.period_id == pid]
        if s.empty:
            continue
        st = s["t"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        sv = s["glucose"].to_numpy()
        csum = np.r_[0.0, np.cumsum(sv)]
        at = anchors["t"].to_numpy()[rows].astype("datetime64[ns]").astype(np.int64)
        hi = np.searchsorted(st, at, side="right")  # sticks with time <= t
        lo = np.searchsorted(st, at - lb, side="right")  # sticks with time > t - lookback
        has = hi > 0
        last = np.where(has, sv[np.maximum(hi - 1, 0)], np.nan)
        prev = np.where(hi > 1, sv[np.maximum(hi - 2, 0)], np.nan)
        cols["fs_last"][rows] = last
        cols["fs_minutes_since"][rows] = np.where(
            has, (at - st[np.maximum(hi - 1, 0)]) / 6e10, np.nan
        )
        cols["fs_prev"][rows] = prev
        cols["fs_delta_last2"][rows] = last - prev
        n = hi - lo
        cols["fs_count_lb"][rows] = n
        cols["fs_mean_lb"][rows] = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)
    for k, v in cols.items():
        out[k] = v
    return out


def load_table(cohort: str, n_h: int, cfg: dict[str, Any]) -> pd.DataFrame:
    """Cached Part 2 anchor table with static columns re-joined from static.parquet."""
    tab = pd.read_parquet(PROC / "features" / f"{cohort}_N{n_h}.parquet")
    tab = tab.drop(columns=[c for c in tab.columns if c.startswith("static_")])
    static = pd.read_parquet(PROC / "static.parquet")
    st = static_features(static, all_static_columns(cfg, cohort))
    return tab.merge(st, left_on="patient_id", right_index=True, how="left")


def load_grid(cohort: str) -> pd.DataFrame:
    """The cohort's sorted grid (5-min CGMacros, 15-min Shanghai)."""
    name = "cgmacros_ts5.parquet" if cohort == "cgmacros" else "common_ts15.parquet"
    g = pd.read_parquet(PROC / name)
    return g[g.dataset == cohort].sort_values(["period_id", "t"]).reset_index(drop=True)
