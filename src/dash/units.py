"""Glucose units for the dashboard (Part 9): mg/dL (default) or mmol/L.

One switch per Streamlit session (a context variable set at the top of every run). All model
numbers stay in mg/dL internally; only what the clinician reads is converted. The factor is
18.016 mg/dL per mmol/L. With mg/dL selected every function returns its input unchanged, so the
default screen is identical to the one without this module.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Any

MGDL, MMOLL = "mg/dL", "mmol/L"
FACTOR = 18.016  # mg/dL per mmol/L
UNITS = (MGDL, MMOLL)
_UNIT: ContextVar[str] = ContextVar("glucotwin_unit", default=MGDL)


def set_unit(unit: str) -> None:
    """Select the display unit for the current run."""
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit}")
    _UNIT.set(unit)


def unit() -> str:
    """The current display unit."""
    return _UNIT.get()


def conv(v: float) -> float:
    """Value in the display unit (input in mg/dL)."""
    return v / FACTOR if unit() == MMOLL else v


def num(v: float, signed: bool = False) -> str:
    """Number in the display unit: whole mg/dL, one decimal mmol/L."""
    dp = 1 if unit() == MMOLL else 0
    x = conv(float(v))
    return f"{x:+.{dp}f}" if signed else f"{x:.{dp}f}"


def fmt(v: float, signed: bool = False) -> str:
    """'168 mg/dL' or '9.3 mmol/L'."""
    return f"{num(v, signed)} {unit()}"


_RANGE = re.compile(r"(?<![\w.])([-+]?\d+(?:\.\d+)?)(\s+to\s*)([-+]?\d+(?:\.\d+)?)\s*mg/dL")
_ONE = re.compile(r"(?<![\w.])([-+]?\d+(?:\.\d+)?)\s*mg/dL")


def _one(m: re.Match[str]) -> str:
    s = m.group(1)
    return f"{num(float(s), signed=s[0] in '+-')} {MMOLL}"


def _range(m: re.Match[str]) -> str:
    a, b = m.group(1), m.group(3)
    return f"{num(float(a), signed=a[0] in '+-')}{m.group(2)}{num(float(b), signed=b[0] in '+-')} {MMOLL}"


def localize(s: Any) -> Any:
    """Rewrite '<number> mg/dL' (and 'a to b mg/dL') in text; non-strings pass through."""
    if not isinstance(s, str) or unit() == MGDL:
        return s
    return _ONE.sub(_one, _RANGE.sub(_range, s))


def axis_ticks(lo: float, hi: float) -> tuple[list[float], list[str]] | None:
    """Tick positions (mg/dL) and labels for a mmol/L axis, or None for mg/dL."""
    if unit() == MGDL:
        return None
    vals = [v for v in (40, 54, 70, 100, 140, 180, 250, 300, 350) if lo <= v <= hi]
    return vals, [f"{v / FACTOR:.1f}" for v in vals]
