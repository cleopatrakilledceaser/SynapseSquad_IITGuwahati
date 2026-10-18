"""Template-based plain-language "why" for event risks and forecasts (Part 9, D-057). No LLM.

A sentence is assembled from fixed templates. A feature is quoted only if (i) the model really
uses it (``shap_values.used_features``), (ii) its SHAP value moves the +h forecast by at least
``min_abs_shap_mgdl`` and (iii) the push points the way the sentence says (towards the event for
moderate / high risk, away from it for low risk). Every quoted number is read from the row.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

WEAK_TOPICS = ("time", "profile", "data")
NAMES = {"carbs_g": "carbs", "protein_g": "protein", "fat_g": "fat", "fiber_g": "fibre"}


def feature_topic(f: str) -> str:
    """Topic of a model feature (one reason per topic is quoted)."""
    if re.match(r"cgm_(slope|delta|accel)", f):
        return "trend"
    if f.startswith("meal_"):
        return "meal"
    if f.startswith(("act_", "hr_")):
        return "activity"
    if f in ("tod_sin", "tod_cos", "days_since_start"):
        return "time"
    if f.startswith("static_"):
        return "profile"
    if f in ("cgm_frac_real_N", "cgm_frac_carried_N", "cgm_minutes_since_real"):
        return "data"
    return "level"


def _num(row: Mapping[str, Any], k: str) -> float:
    v = row.get(k, float("nan"))
    return float(v) if v is not None else float("nan")


def _clock(row: Mapping[str, Any]) -> str:
    ang = math.atan2(_num(row, "tod_sin"), _num(row, "tod_cos"))
    mins = int(round((ang % (2 * math.pi)) / (2 * math.pi) * 1440)) % 1440
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _meal_clause(f: str, row: Mapping[str, Any]) -> str:
    since = _num(row, "meal_minutes_since_last")
    m = re.match(r"meal_(carbs_g|protein_g|fat_g|fiber_g|kcal|count)_(\d)h", f)
    if m:
        kind, w, v = m.group(1), int(m.group(2)), _num(row, f)
        if kind == "count":
            return f"{v:.0f} meal{'s' if v != 1 else ''} logged in the last {w} h"
        if kind == "kcal":
            return f"{v:.0f} kcal eaten in the last {w} h"
        one = _num(row, f"meal_count_{w}h") == 1
        if kind == "carbs_g" and one and not math.isnan(since):
            return f"the last meal had {v:.0f} g carbs {since:.0f} min ago"
        return f"{v:.0f} g {NAMES[kind]} eaten in the last {w} h"
    if f == "meal_minutes_since_last":
        return f"the last meal was {_num(row, f):.0f} min ago"
    if f.startswith("meal_last_type_"):
        return f"the last meal was logged as {f.removeprefix('meal_last_type_')}"
    return "the last meal was larger than the logged amount"  # meal_last_amount_gt100


def clause(f: str, row: Mapping[str, Any]) -> str:
    """One plain-language clause for feature ``f`` with its value taken from ``row``."""
    v = _num(row, f)
    m = re.match(r"cgm_(lag|delta|slope)_(\d+)$", f)
    if f == "cgm_now":
        return f"glucose is {v:.0f} mg/dL now"
    if m and m.group(1) == "lag":
        return f"glucose was {v:.0f} mg/dL {m.group(2)} min ago"
    if m and m.group(1) == "delta":
        way = "risen" if v >= 0 else "fallen"
        return f"glucose has {way} {abs(v):.0f} mg/dL in the last {m.group(2)} min"
    if m and m.group(1) == "slope":
        way = "rising" if v >= 0 else "falling"
        return f"glucose is {way} about {abs(v) * 15:.0f} mg/dL per 15 min"
    if f == "cgm_accel":
        return "the rise is speeding up" if v >= 0 else "the rise is slowing down"
    stat = re.match(r"cgm_(mean|std|min|max)_(60|N)$", f)
    if stat:
        span = "hour" if stat.group(2) == "60" else "6 hours"
        word = {
            "mean": "average glucose",
            "std": "glucose variability (SD)",
            "min": "the lowest glucose",
            "max": "the highest glucose",
        }[stat.group(1)]
        return f"{word} over the last {span} is {v:.0f} mg/dL"
    fixed = {
        "cgm_tir_3h": f"{100 * v:.0f}% of the last 3 hours were in range (70 to 180 mg/dL)",
        "cgm_tabove140_3h": f"{100 * v:.0f}% of the last 3 hours were above 140 mg/dL",
        "cgm_frac_real_N": f"{100 * v:.0f}% of the last 6 h has sensor readings",
        "cgm_frac_carried_N": f"{100 * v:.0f}% of the last 6 h was filled in",
        "cgm_minutes_since_real": f"the last sensor reading is {v:.0f} min old",
        "act_post_meal": f"activity since the last meal averaged {v:.1f} intensity units",
        "tod_sin": f"it is {_clock(row)} (time of day)",
        "tod_cos": f"it is {_clock(row)} (time of day)",
        "days_since_start": f"this is day {v:.1f} of monitoring",
        "static_hba1c_pct": f"HbA1c is {v:.1f} %",
        "static_fasting_glucose_mgdl": f"fasting glucose is {v:.0f} mg/dL",
    }
    if f in fixed:
        return fixed[f]
    if f.startswith("meal_"):
        return _meal_clause(f, row)
    raise KeyError(f"no template for feature {f}")


def risk_word(p: float, bounds: list[float], words: list[str]) -> str:
    """Wording of a calibrated probability."""
    return words[int(np.searchsorted(bounds, p, side="right"))]


def pick_reasons(
    shap_row: Mapping[str, float], used: set[str], direction: int, cfg: dict[str, Any]
) -> list[tuple[str, float]]:
    """Top features (one per topic) whose SHAP pushes in ``direction`` (+1 up, -1 down)."""
    cand = [
        (f, s)
        for f, s in shap_row.items()
        if f in used and direction * s >= cfg["min_abs_shap_mgdl"] and not math.isnan(s)
    ]
    # clinically meaningful topics first; time of day, profile, data only if nothing else qualifies
    cand.sort(key=lambda x: (feature_topic(x[0]) in WEAK_TOPICS, -abs(x[1])))
    out, seen = [], set()
    for f, s in cand:
        t = feature_topic(f)
        if t in seen:
            continue
        seen.add(t)
        out.append((f, s))
        if len(out) >= cfg["max_reasons"]:
            break
    return out


def explain_event(
    event: str,
    horizon: int,
    prob: float | None,
    row: Mapping[str, Any],
    shap_row: Mapping[str, float],
    used: set[str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Sentence + the structured reasons behind it for one event / horizon / row."""
    ev = cfg["events"][event]
    if prob is None or math.isnan(prob):
        text = f"Risk of glucose {ev['label']} in {horizon} min: not available."
        return {"sentence": text, "reasons": [], "risk": None}
    rl = cfg["risk_levels"]
    word = risk_word(prob, rl["bounds"], rl["words"])
    up = ev["direction"] == "above"
    towards = word != rl["words"][0]  # low risk: quote what pushes AWAY from the event
    direction = (1 if up else -1) * (1 if towards else -1)
    reasons = pick_reasons(shap_row, used, direction, cfg)
    head = f"Risk of glucose {ev['label']} in {horizon} min is {word} ({100 * prob:.0f}%)"
    if reasons:
        sentence = f"{head}: {' and '.join(clause(f, row) for f, _ in reasons)}."
    else:
        sentence = f"{head}: no single input stands out."
    return {
        "sentence": sentence,
        "risk": word,
        "reasons": [
            {"feature": f, "shap_mgdl": float(s), "topic": feature_topic(f), "text": clause(f, row)}
            for f, s in reasons
        ],
    }


def trend_word(slope_per_min: float) -> str:
    """'rising' / 'falling' / 'steady' (within 3 mg/dL per 15 min) from a slope in mg/dL per min."""
    per15 = 15 * slope_per_min
    return "rising" if per15 > 3 else "falling" if per15 < -3 else "steady"
