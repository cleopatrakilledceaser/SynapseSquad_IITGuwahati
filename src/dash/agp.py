"""Standard glucose-report numbers (Part 9): time in ranges, GMI, variability, AGP profile.

Pure pandas / numpy. Definitions follow the international consensus on CGM metrics (Battelino et
al., Diabetes Care 2019; variability target from Danne et al., Diabetes Care 2017) and the
glucose management indicator of Bergenstal et al. (Diabetes Care 2018): GMI (%) = 3.31 + 0.02392 x
mean glucose (mg/dL). The targets are population consensus values, not a prescription for one
patient.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

QUANTILES = (5, 25, 50, 75, 95)
# (key, label, direction, limit in % of readings, glucose cut shown in the label)
TARGETS = [
    ("range", "In range (70 to 180 mg/dL)", "above", 70.0),
    ("above180", "Above 180 mg/dL", "below", 25.0),
    ("above250", "Above 250 mg/dL", "below", 5.0),
    ("below70", "Below 70 mg/dL", "below", 4.0),
    ("below54", "Below 54 mg/dL", "below", 1.0),
]
CV_TARGET = 36.0  # coefficient of variation, percent
MIN_DAYS = 14  # consensus: at least 14 days with 70% of readings


def shares(glucose: pd.Series, zones: dict[str, float]) -> dict[str, float]:
    """Percent of readings in each zone and the cumulative cuts used by the consensus."""
    g = glucose.dropna().to_numpy(float)
    if len(g) == 0:
        return {}
    pct = lambda m: float(100.0 * np.mean(m))  # noqa: E731
    return {
        "very_low": pct(g < zones["very_low"]),
        "low": pct((g >= zones["very_low"]) & (g < zones["low"])),
        "range": pct((g >= zones["low"]) & (g <= zones["high"])),
        "high": pct((g > zones["high"]) & (g <= zones["very_high"])),
        "very_high": pct(g > zones["very_high"]),
        "below70": pct(g < zones["low"]),
        "below54": pct(g < zones["very_low"]),
        "above180": pct(g > zones["high"]),
        "above250": pct(g > zones["very_high"]),
    }


def gmi(mean_mgdl: float) -> float:
    """Glucose management indicator in percent HbA1c units."""
    return 3.31 + 0.02392 * float(mean_mgdl)


def cv_percent(glucose: pd.Series) -> float:
    """Coefficient of variation of the readings in percent."""
    g = glucose.dropna().to_numpy(float)
    return float(100.0 * g.std(ddof=0) / g.mean()) if len(g) > 1 and g.mean() > 0 else float("nan")


def meets(key: str, value: float) -> bool:
    """Whether a share meets the consensus target for ``key``."""
    _, _, direction, limit = next(t for t in TARGETS if t[0] == key)
    return value > limit if direction == "above" else value < limit


def coverage(df: pd.DataFrame) -> dict[str, float]:
    """Days covered and the fraction of expected readings that are present."""
    t = pd.to_datetime(df["t"]).sort_values()
    if len(t) < 2:
        return {"days": 0.0, "fraction": 0.0}
    step = t.diff().dropna().median()
    span = t.iloc[-1] - t.iloc[0]
    expected = span / step + 1
    return {
        "days": float(span / pd.Timedelta(days=1)),
        "fraction": float(min(len(t) / expected, 1.0)),
    }


def agp_profile(df: pd.DataFrame, bin_min: int = 30) -> pd.DataFrame:
    """Percentiles of glucose by time of day (all days overlaid), one row per bin."""
    t = pd.to_datetime(df["t"])
    minute = (t.dt.hour * 60 + t.dt.minute) // bin_min * bin_min
    rows = []
    for m, g in df.groupby(minute.to_numpy())["glucose"]:
        v = g.dropna().to_numpy(float)
        if len(v) < 3:
            continue
        rows.append(
            {
                "minute": int(m),
                "n": len(v),
                **{f"q{q:02d}": float(np.percentile(v, q)) for q in QUANTILES},
            }
        )
    return pd.DataFrame(rows).sort_values("minute").reset_index(drop=True)


def report(df: pd.DataFrame, zones: dict[str, float]) -> dict[str, Any]:
    """All report numbers for one patient's readings (columns t, glucose)."""
    g = df["glucose"]
    mean = float(g.mean())
    return {"mean": mean, "gmi": gmi(mean), "cv": cv_percent(g), "shares": shares(g, zones),
            "coverage": coverage(df), "n": int(g.notna().sum())}  # fmt: skip
