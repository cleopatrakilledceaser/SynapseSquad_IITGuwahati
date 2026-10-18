"""Tables and numbers of the reference pages, read from the stored cohort statistics (D-059).

Everything is copied from ``stats.json.gz`` (itself copied from the Part 3 / 5 / 6 / 7 metric
files by ``src.appdata.export.cohort_stats``); nothing is recomputed. Pure pandas, no Streamlit.
A key that is missing gives "not available" instead of an invented number.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

NA = "not available"
H = ("30", "60", "120")


def _g(d: Any, *keys: Any, default: Any = None) -> Any:
    """Nested get that tolerates int / str keys (JSON turns int keys into strings)."""
    for k in keys:
        if isinstance(d, Mapping):
            d = d[k] if k in d else d.get(str(k), None)
        else:
            return default
        if d is None:
            return default
    return d


def ci_text(v: Any, ci: Any, digits: int = 1, unit: str = "") -> str:
    """'16.9 [15.2, 18.7] mg/dL'."""
    if v is None:
        return NA
    s = f"{float(v):.{digits}f}"
    if ci and len(ci) == 2:
        s += f" [{float(ci[0]):.{digits}f}, {float(ci[1]):.{digits}f}]"
    return f"{s}{unit}"


# --------------------------------------------------------------------------- input tiers

TIER_ROWS = [
    ("persistence", "Last CGM value carried forward (reference)", "CGM only"),
    ("lgbm_E_full_delta", "Full inputs", "CGM, time of day, wearable, meals, profile"),
    ("lgbm_T2_fs4_delta", "No CGM, four fingersticks a day", "time, wearable, meals, profile"),
    ("lgbm_T2_fs2_delta", "No CGM, two fingersticks a day", "time, wearable, meals, profile"),
    ("lgbm_T3_fs4_delta", "Time and profile only, four fingersticks a day", "time, profile"),
    ("lgbm_T3_fs2_delta", "Time and profile only, two fingersticks a day", "time, profile"),
]


def tier_table(stats: Mapping[str, Any]) -> pd.DataFrame:
    """Mean absolute error (mg/dL) with 95% patient-bootstrap interval per tier and horizon."""
    rows = []
    by = {(r["model"], str(r["horizon_min"])): r for r in stats.get("tiers", [])}
    for model, name, inputs in TIER_ROWS:
        row = {"Inputs available": name, "What the model sees": inputs}
        for h in H:
            r = by.get((model, h))
            row[f"+{h} min"] = ci_text(r["mae"], r.get("mae_ci")) if r else NA
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- personalization

CURVE_MODELS = {
    "lgbm_s_none": "Population model, no personalization",
    "lgbm_s_finetune": "Population model fine-tuned on the patient's first k days",
}


def personalization_curve(stats: Mapping[str, Any], horizon: str = "60") -> pd.DataFrame:
    """Long table (model label, k days, MAE, ci_lo, ci_hi) of the Part 5 cold-start curve."""
    rows = []
    for r in _g(stats, "personalize", "curves", horizon, default=[]):
        if r["model"] in CURVE_MODELS:
            ci = r.get("mae_ci") or [None, None]
            rows.append({"model": CURVE_MODELS[r["model"]], "k": float(r["k"]), "mae": r["mae"],
                         "lo": ci[0], "hi": ci[1]})  # fmt: skip
    return pd.DataFrame(rows)


def personalization_gain(stats: Mapping[str, Any], horizon: str = "60") -> dict[str, Any]:
    """MAE gain (mg/dL, negative = better) of fine-tuning vs none after 3 and 5 days."""
    cur = personalization_curve(stats, horizon)
    out: dict[str, Any] = {}
    if cur.empty:
        return out
    base, ft = (cur[cur["model"] == v] for v in CURVE_MODELS.values())
    for k in (1.0, 3.0, 5.0):
        b, f = base[base["k"] == k], ft[ft["k"] == k]
        if len(b) and len(f):
            out[k] = float(f["mae"].iloc[0] - b["mae"].iloc[0])
    return out


def head_start(stats: Mapping[str, Any]) -> dict[str, Any]:
    """The Part 5 head-start result (static profile as a prior): value, CI, plain note."""
    h = _g(stats, "personalize", "head_start", default={}) or {}
    return {
        "mgdl": h.get("head_start_mgdl"),
        "ci": h.get("head_start_ci"),
        "positive": h.get("head_start_positive"),
        "note": h.get("note"),
    }


# --------------------------------------------------------------------------- data quality


def robustness_table(stats: Mapping[str, Any]) -> pd.DataFrame:
    """Split-conformal 80% coverage at +60 min and MAE under sensor faults (Part 6)."""
    rows = []
    for c in stats.get("drift_cells", []):
        f, lvl = c["fault"], c.get("level")
        label = (
            "No fault" if f == "clean"
            else f"Constant offset of {lvl:g} mg/dL" if f == "bias"
            else f"Drift of {lvl:g} mg/dL per day" if f == "drift"
            else f"{f} {lvl}"
        )  # fmt: skip
        cov, ci = c.get("cov_scp_pop_80_60"), c.get("cov_ci_scp_pop_80_60")
        rows.append({
            "Fault": label,
            "80% interval coverage": (ci_text(100 * cov, [100 * ci[0], 100 * ci[1]], 0, " %") if cov is not None and ci else NA),
            "Data-quality score": (f"{c['dq_mean']:.2f}" if c.get("dq_mean") is not None else NA),
            "Mean error": ci_text(c.get("mae_60"), c.get("mae_ci_60"), 1, " mg/dL"),
        })  # fmt: skip
    if not rows:
        return pd.DataFrame(
            columns=["Fault", "80% interval coverage", "Data-quality score", "Mean error"]
        )
    order = {"No fault": 0}
    return pd.DataFrame(rows).sort_values(
        "Fault", key=lambda s: s.map(lambda x: order.get(x, 1)), kind="stable"
    )


# --------------------------------------------------------------------------- what-if evidence

TEST_LABEL = {
    "carb_monotonicity": "More carbohydrate never lowers the predicted peak",
    "activity_direction": "A walk never raises the predicted peak or area",
    "causality": "A change after the forecast time changes nothing before it",
    "bounds": "Forecasts stay within 40 to 400 mg/dL and a plausible rate of change",
    "dose_window": "Peak rise per gram of carbohydrate stays in a plausible range",
    "cross_model": "The machine-learning and twin models agree on the carbohydrate effect",
}
MODEL_LABEL = {
    "mono": "Monotone model (default)", "unc": "Unconstrained model",
    "twin_learned": "Twin, learned walk effect", "twin_assumed": "Twin, assumed walk effect",
}  # fmt: skip


def plausibility_table(stats: Mapping[str, Any]) -> pd.DataFrame:
    """Part 7 pass rates (test patients) for the tests that have a plain-language label."""
    rows = []
    for r in _g(stats, "plausibility", "final", "pass_rates", default=[]):
        if r["test"] not in TEST_LABEL:
            continue
        rows.append({
            "Test": TEST_LABEL[r["test"]], "Model": MODEL_LABEL.get(r["model"], r["model"]),
            "Pass rate": ci_text(100 * r["pass_rate"], [100 * r["ci"][0], 100 * r["ci"][1]], 0, " %"),
            "By construction": "yes" if "construction" in str(r.get("note", "")) else "no",
        })  # fmt: skip
    return pd.DataFrame(rows)


def dose_table(stats: Mapping[str, Any]) -> pd.DataFrame:
    """Median predicted peak rise per gram of carbohydrate against the literature range."""
    rows = []
    for r in _g(stats, "plausibility", "final", "dose_median", default=[]):
        lo, hi = r["range"]
        rows.append({
            "Model": MODEL_LABEL.get(r["model"], r["model"]),
            "Median rise per gram": f"{r['median_mgdl_per_g']:.2f} mg/dL",
            "Expected range": f"{lo:.1f} to {hi:.1f} mg/dL",
            "Within range": "yes" if r["pass"] else "no",
        })  # fmt: skip
    return pd.DataFrame(rows)


def matched_summary(stats: Mapping[str, Any]) -> pd.DataFrame:
    """Matched real breakfasts: does the predicted peak difference follow the observed one?"""
    rows = []
    for model in ("mono", "unc", "twin_learned"):
        m = _g(stats, "matched", f"within|{model}|peak")
        if not m:
            continue
        rows.append({
            "Model": MODEL_LABEL.get(model, model),
            "Breakfast pairs": f"{m['n_pairs']} from {m['n_clusters']} patients",
            "Direction right": ci_text(100 * m["sign_agreement"]["value"], [100 * x for x in m["sign_agreement"]["ci"]], 0, " %"),
            "Mean predicted peak difference": f"{m['mean_pred']['value']:.0f} mg/dL",
            "Mean observed peak difference": f"{m['mean_obs']['value']:.0f} mg/dL" if "mean_obs" in m else NA,
        })  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- alerts and events


def alarm_table(stats: Mapping[str, Any], rate: str = "2") -> pd.DataFrame:
    """Alert operating points (about ``rate`` false alarms a day) of the three alert systems."""
    name = {"model": "GlucoTwin model", "naive": "Current glucose value", "trend": "Current trend"}
    ev = {"hyper180": "Above 180 mg/dL", "hypo70": "Below 70 mg/dL"}
    rows = []
    for e, el in ev.items():
        for s, sl in name.items():
            c = _g(stats, "alarms", e, s, rate)
            if not c:
                continue
            rows.append({
                "Event": el, "Alert based on": sl,
                "Episodes caught": f"{100 * c['sensitivity']:.0f} % of {c['episodes_eligible']}"
                + (" (limited evidence)" if c["episodes_eligible"] < 20 else ""),
                "False alarms per day": f"{c['fa_per_day']:.1f}",
                "Median warning": f"{c['lead_median_min']:.0f} min" if c.get("lead_median_min") is not None else NA,
            })  # fmt: skip
    return pd.DataFrame(rows)


def auroc_text(stats: Mapping[str, Any]) -> list[str]:
    """One line per event: AUROC of the calibrated probability at +60 min."""
    lab = {
        "hyper180_60": "above 180 mg/dL",
        "hyper140_60": "above 140 mg/dL",
        "hypo70_60": "below 70 mg/dL",
    }
    out = []
    for k, label in lab.items():
        v = _g(stats, "events_auroc", k)
        if v:
            note = " (only 8 patients had events: limited evidence)" if k == "hypo70_60" else ""
            out.append(f"Glucose {label} within 60 min: {ci_text(v['value'], v['ci'], 2)}{note}")
    return out


def coverage_text(stats: Mapping[str, Any]) -> str:
    """Pooled 80% interval coverage at +60 (CV+) and its unequal-group caveat."""
    c = next(
        (
            r
            for r in stats.get("coverage", [])
            if r["method"] == "cvp" and str(r["horizon_min"]) == "60"
        ),
        None,
    )
    if not c:
        return NA
    return (
        f"80% intervals contained the true glucose {100 * c['coverage']:.0f} % of the time at +60 min "
        f"(95% interval {100 * c['coverage_ci'][0]:.0f} to {100 * c['coverage_ci'][1]:.0f} %), "
        f"median width {c['median_width']:.0f} mg/dL. In the worst-covered patient the share was "
        f"{100 * c['per_patient_min']:.0f} %."
    )


LIMITATIONS = [
    "Coverage is not equal across groups: on CGMacros the intervals covered 87 % of healthy "
    "rows but 73 % of type 2 diabetes rows (Part 6).",
    "A sensor that reads too high or drifts is not detected by the data-quality score, and "
    "coverage falls sharply under such faults.",
    "The learned models understate the effect of carbohydrate: in matched real breakfasts they "
    "predicted about half of the observed rise, and the pre-registered dose-response test failed "
    "for both machine-learning models.",
    "The twin adds little on top of the machine-learning model, and only the basal glucose Gb of "
    "the twin is identifiable from the data.",
    "The static clinical profile did not improve the machine-learning forecasts; heart rate and "
    "activity add nothing measurable.",
    "Low-glucose results rest on few events (8 patients in CGMacros) and are limited evidence.",
    "All demonstration patients come from one open dataset of 45 people; no prospective or "
    "clinical validation was done. This is a research prototype, not a medical device.",
]
