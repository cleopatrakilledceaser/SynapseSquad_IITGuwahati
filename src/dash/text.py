"""Plain-language strings of the dashboard (Part 9, D-059). Pure functions, no Streamlit.

Every number in a sentence is read from the replay row or the stored cohort statistics; a missing
number is written "not available". Language is decision support: never "diagnosis" or
"prescribe".
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.dash import units
from src.explain.templates import risk_word, trend_word

NA = "not available"
EVENT_TEXT = {
    "hyper180": ("above", 180, "above 180 mg/dL"),
    "hyper140": ("above", 140, "above 140 mg/dL"),
    "hypo70": ("below", 70, "below 70 mg/dL"),
}
DISCLAIMER = (
    "Research prototype for decision support. Not a medical device. Not for diagnosis or treatment."
)
WHATIF_NOTE = (
    "Model-based estimate, not a causal guarantee. In matched real breakfasts the models predicted "
    "about half the observed rise; treat sizes as lower bounds."
)
CONF_BARS = {"uncertain": 1, "moderate": 2, "high": 3}


def isnum(v: Any) -> bool:
    """True for a finite number."""
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def clock(ts: Any) -> str:
    """HH:MM of a timestamp."""
    return pd.Timestamp(ts).strftime("%H:%M")


def mgdl(v: Any) -> str:
    """'168 mg/dL' (or '9.3 mmol/L' when selected) or 'not available'."""
    return units.fmt(float(v)) if isnum(v) else NA


def mgdl_num(v: Any) -> str:
    """The number alone in the display unit, or 'not available'."""
    return units.num(float(v)) if isnum(v) else NA


def pct(p: Any) -> str:
    """'62%' or 'not available'."""
    return f"{100 * float(p):.0f}%" if isnum(p) else NA


def trend_of(row: Mapping[str, Any]) -> str:
    """'rising' / 'falling' / 'steady' from the trailing 30-min CGM slope, else 'not available'."""
    s = row.get("cgm_slope_30")
    return trend_word(float(s)) if isnum(s) else NA


def readouts(row: Mapping[str, Any], horizon: int = 60) -> list[dict[str, Any]]:
    """Event readouts at +horizon: probability, or why there is none."""
    out = []
    now = row.get("cgm_now")
    for ev, (direction, thr, label) in EVENT_TEXT.items():
        p = row.get(f"p_{ev}|{horizon}")
        if isnum(p):
            out.append({"event": ev, "label": label, "p": float(p), "note": ""})
            continue
        beyond = isnum(now) and ((now > thr) if direction == "above" else (now < thr))
        note = f"already {label.split(' ')[0]} {thr} mg/dL now" if beyond else NA
        out.append({"event": ev, "label": label, "p": None, "note": note})
    return out


def risk_label(p: Any, cfg: Mapping[str, Any]) -> str:
    """Wording of a calibrated probability (display choice from configs/app.yaml)."""
    if not isnum(p):
        return NA
    rw = cfg["risk_words"]
    return risk_word(float(p), rw["bounds"], rw["words"])


def alert_state(row: Mapping[str, Any]) -> dict[str, Any]:
    """Which alerts are ON at this anchor (hyper180, hypo70)."""
    on = [e for e in ("hyper180", "hypo70") if bool(row.get(f"active_{e}", False))]
    fired = [e for e in ("hyper180", "hypo70") if bool(row.get(f"fired_{e}", False))]
    return {"on": on, "fired_now": fired}


def alert_evidence(stats: Mapping[str, Any], event: str, rate: int) -> dict[str, Any] | None:
    """Stored Part 6 operating point of the model alert at ``rate`` false alarms per day."""
    cell = stats.get("alarms", {}).get(event, {}).get("model", {}).get(str(rate))
    if not cell:
        return None
    return {
        "sensitivity": cell.get("sensitivity"),
        "lead_median_min": cell.get("lead_median_min"),
        "episodes": cell.get("episodes_eligible"),
        "fa_per_day": cell.get("fa_per_day"),
    }


def confidence(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Confidence label, bar count and the reasons when it is 'uncertain' (design choice)."""
    c = cfg["confidence"]
    label = str(row.get("confidence", "uncertain"))
    reasons: list[str] = []
    ratio, dq, stale = row.get("width_ratio"), row.get("dq"), row.get("stale_min")
    if isnum(ratio) and float(ratio) > c["wide_ratio"]:
        reasons.append("the forecast interval is wider than usual for this model")
    if isnum(dq) and float(dq) < c["dq_uncertain_below"]:
        reasons.append("the sensor data are incomplete or implausible")
    if isnum(stale) and float(stale) > c["stale_min"]:
        reasons.append(f"the last sensor reading is {float(stale):.0f} min old")
    return {
        "label": label,
        "bars": CONF_BARS.get(label, 1),
        "reasons": reasons if label == "uncertain" else [],
    }


def forecast_line(row: Mapping[str, Any], horizon: int = 60) -> str:
    """Forecast sentence, e.g. 'Forecast at +60 min: 168 mg/dL (80% interval 150 to 186 mg/dL).'."""
    p, lo, hi = row.get(f"pred|{horizon}"), row.get(f"lo80|{horizon}"), row.get(f"hi80|{horizon}")
    if not (isnum(p) and isnum(lo) and isnum(hi)):
        return f"Forecast at +{horizon} min: {NA}."
    return (
        f"Forecast at +{horizon} min: {float(p):.0f} mg/dL "
        f"(80% interval {float(lo):.0f} to {float(hi):.0f} mg/dL)."
    )


def decision(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The decision sentence of an anchor and the event it is about.

    Rule (a display choice): the event with an alert ON (high-glucose alert first), else the more
    likely of 'above 180' and 'below 70' at +60 min.
    """
    ev60 = {r["event"]: r for r in readouts(row, 60)}
    alerts = alert_state(row)["on"]
    cand = ["hyper180", "hypo70"]
    if alerts:
        event = alerts[0]
    else:
        beyond = [e for e in cand if ev60[e]["p"] is None and ev60[e]["note"].startswith("already")]
        probs = {e: (ev60[e]["p"] if ev60[e]["p"] is not None else -1.0) for e in cand}
        event = beyond[0] if beyond else max(cand, key=lambda e: probs[e])
    r = ev60[event]
    trend = trend_of(row)
    head = f"Next 2 hours: glucose {trend}." if trend != NA else "Next 2 hours:"
    if r["p"] is None:
        body = (
            f"Glucose is {r['note']}."
            if r["note"].startswith("already")
            else f"The chance of glucose going {EVENT_TEXT[event][2]} within 60 min is {NA}."
        )
        word = NA
    else:
        word = risk_label(r["p"], cfg)
        body = f"{pct(r['p'])} chance of going {EVENT_TEXT[event][2]} within 60 min ({word} risk)."
    return {
        "event": event,
        "sentence": f"{head} {body}".replace(":  ", ": "),
        "risk_word": word,
        "p": r["p"],
    }


def uncertain_banner(conf: Mapping[str, Any]) -> str | None:
    """Banner that replaces the alert when the forecast should not be relied on."""
    if conf["label"] != "uncertain":
        return None
    why = "; ".join(conf["reasons"]) if conf["reasons"] else "the confidence checks did not pass"
    return f"Uncertain. Do not rely on this forecast: {why}."


def explain_sentences(explain: pd.DataFrame, t: Any, horizon: int = 60) -> list[str]:
    """The template sentences of an anchor at one horizon (all three events)."""
    e = explain[(explain["t"] == pd.Timestamp(t)) & (explain["horizon"] == horizon)]
    order = {k: i for i, k in enumerate(EVENT_TEXT)}
    e = e.assign(_o=e["event"].map(order)).sort_values("_o")
    return [str(s) for s in e["sentence"]]


def profile_lines(meta: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Static profile as (label, value) pairs; synthetic values are marked as assumed."""
    syn = bool(meta.get("synthetic"))
    suffix = " (assumed)" if syn else ""
    status = {"t2d": "type 2 diabetes range", "prediabetes": "prediabetes range",
              "healthy": "normal range", "synthetic": "synthetic"}.get(str(meta.get("status")), NA)  # fmt: skip
    return [
        ("HbA1c", (f"{meta['hba1c_pct']:.1f} %{suffix}" if isnum(meta.get("hba1c_pct")) else NA)),
        ("Fasting glucose",
         (f"{meta['fasting_glucose_mgdl']:.0f} mg/dL{suffix}" if isnum(meta.get("fasting_glucose_mgdl")) else NA)),  # fmt: skip
        ("Glycemic status by HbA1c", status),
    ]


LIMIT = "The score detects missing data, not wrong data: a sensor that reads too high or drifts is not detected."


def quality_banner(row: Mapping[str, Any]) -> str:
    """Data-quality banner of an anchor: what the score does and does not detect."""
    dq = row.get("dq")
    if not isnum(dq):
        return f"Sensor data quality: {NA}. {LIMIT}"
    d = float(dq)
    level = "complete" if d >= 0.95 else "mostly complete" if d >= 0.70 else "incomplete"
    return f"Sensor data {level} (quality score {d:.2f}). {LIMIT}"


def short_status(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    """One status line for the patient list: trend and the headline risk."""
    d = decision(row, cfg)
    trend = trend_of(row)
    head = trend.capitalize() if trend != NA else "Trend not available"
    r = next(x for x in readouts(row, 60) if x["event"] == d["event"])
    if r["p"] is None:
        return (
            f"{head}. Glucose is {r['note']}."
            if r["note"].startswith("already")
            else f"{head}. Risk {NA}."
        )
    return f"{head}. {pct(r['p'])} chance {EVENT_TEXT[d['event']][2]} within 60 min."
