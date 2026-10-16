"""Plausibility suite (D-051): per-window tests for unc / mono / twin_learned / twin_assumed.

Windows: meal windows (a logged meal with macros + recorded amount in the 15 min before t, day >=
min_day) and ML walk windows (last meal 45-120 min before t). At most N per patient (seed 42).
Every record is (test, model, patient_id, pass, value); no outcome (y) is ever read here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.hybrid.twin_features import HORIZONS
from src.whatif import edits as ed
from src.whatif.engine import FoldCtx, Period, edited_rows, ml_forecast, twin_run

ML = ("unc", "mono")
BY_CONSTRUCTION = {("carb_monotonicity", "mono"), ("activity_direction", "mono")}


def last_meal(per: Period, t: pd.Timestamp, lo_min: float, hi_min: float) -> Any:
    """Index of the latest meal in [t - hi, t - lo] with macros + amount, else None."""
    m = per.meals[per.meals["t"] <= t]
    if m.empty:
        return None
    r = m.iloc[-1]
    age = (t - r["t"]).total_seconds() / 60
    ok = lo_min <= age <= hi_min and bool(r["macros_available"]) and not bool(r["amount_missing"])
    return m.index[-1] if ok and pd.notna(r["carbs_g"]) else None


def windows(per: Period, pc: dict, rng: np.random.Generator, kind: str) -> list[tuple]:
    """Sampled (t, meal) windows of one period ('meal' or 'walk')."""
    lo, hi = (0.0, pc["meal_lookback_min"]) if kind == "meal" else tuple(pc["walk_window_min"])
    rows = per.rows[per.rows["days_since_start"] >= pc["min_day"]]
    out = [(t, m) for t in rows["t"] if (m := last_meal(per, t, lo, hi)) is not None]
    k = pc["max_windows_per_patient"]
    if len(out) > k:
        out = [out[i] for i in sorted(rng.choice(len(out), k, replace=False))]
    return out


def ml_batch(fc: FoldCtx, per: Period, row: pd.DataFrame, sets: list[list]) -> dict:
    """Unclipped ML forecasts (form -> (n_sets, 3)) for several edit sets at one anchor."""
    d = pd.concat([edited_rows(fc, per, row, s) for s in sets], ignore_index=True)
    return ml_forecast(fc, d, per.patient_id)


def ml_peak_auc(f: np.ndarray, now: float) -> tuple[np.ndarray, np.ndarray]:
    """Peak (max of +30/+60/+120) and trapezoid AUC over 0/30/60/120 min (0 = cgm_now)."""
    x = np.array([0.0, *HORIZONS])
    v = np.column_stack([np.full(len(f), now), f])
    return f.max(1), np.trapezoid(v, x, axis=1)


def tw_peak_auc(r: dict) -> tuple[float, float]:
    """Peak and AUC of a twin trajectory (0..rollout)."""
    g, dt = r["g"], float(r["dt"][0])
    return float(np.max(g)), float(np.trapezoid(g, dx=dt))


def rec(out: list, test: str, model: str, pid: str, ok: Any, value: float = np.nan) -> None:
    """Append one record (ok None = descriptive only)."""
    out.append({"test": test, "model": model, "patient_id": pid, "pass": ok, "value": value})


def twin_alphas(fc: FoldCtx) -> dict[str, float]:
    """Twin variants of the suite: learned (Part 4 alpha_on) and assumed (D-052 main value)."""
    a = fc.wc["twin"]["assumed"]
    return {"twin_learned": fc.alpha_on, "twin_assumed": fc.alpha_assumed[f"{a['reduction']:g}"]}


def carb_sets(meal: Any, pc: dict) -> list[list]:
    """Edit sets: carb grid, then zero edit (no edits), then protein / fat / fiber edits."""
    return [[ed.set_macros(meal, carbs_g=float(c))] for c in pc["carb_grid_g"]]


def macro_sets(per: Period, meal: Any, pc: dict) -> list[list]:
    """+x g protein / fat / fiber of the meal (descriptive edits)."""
    r = per.meals.loc[meal]
    return [
        [ed.set_macros(meal, **{k: float(np.nan_to_num(r[k])) + float(v)})]
        for k, v in pc["macro_edits_g"].items()
    ]


def bounds_ok(f: np.ndarray, now: float, pc: dict) -> bool:
    """All forecasts within bounds and |f(+30) - now| / 30 within the rate limit."""
    lo, hi = pc["bounds_mgdl"]
    rate = np.abs(np.atleast_2d(f)[:, 0] - now) / HORIZONS[0]
    return bool(np.all((f >= lo) & (f <= hi)) and np.all(rate <= pc["max_rate_mgdl_min"]))


def dose(peaks: np.ndarray, pc: dict) -> float:
    """(peak(high g) - peak(low g)) / (high - low) from peaks on the carb grid."""
    g = list(pc["carb_grid_g"])
    lo, hi = pc["dose_low_g"], pc["dose_high_g"]
    return float((peaks[g.index(hi)] - peaks[g.index(lo)]) / (hi - lo))


def ml_meal_window(fc: FoldCtx, per: Period, t: pd.Timestamp, meal: Any, out: list) -> dict:
    """ML tests at one meal window; returns form -> dose (for cross-model consistency)."""
    pc, pid = fc.wc["plausibility"], per.patient_id
    row = per.rows[per.rows["t"] == t]
    now = float(row["cgm_now"].iloc[0])
    sets = carb_sets(meal, pc) + [[]] + macro_sets(per, meal, pc)
    f = ml_batch(fc, per, row, sets)
    stored = ml_forecast(fc, row, pid)
    ng, doses = len(pc["carb_grid_g"]), {}
    for n in ML:
        pk, _ = ml_peak_auc(f[n], now)
        steps = np.diff(pk[:ng])
        rec(out, "carb_monotonicity", n, pid, bool(np.all(steps >= -pc["mono_tol"])), steps.min())
        rec(
            out, "carb_monotonicity_2mgdl", n, pid, bool(np.all(steps >= -pc["mono_tol_secondary"]))
        )
        rec(out, "identity", n, pid, bool(np.array_equal(f[n][ng], stored[n][0])))
        rec(out, "bounds", n, pid, bounds_ok(f[n][: ng + 1], now, pc))
        d = dose(pk, pc)
        lo, hi = pc["dose_window_range"]
        rec(out, "dose_window", n, pid, bool(lo <= d <= hi), d)
        doses[n] = d
        for i, k in enumerate(pc["macro_edits_g"]):
            rec(out, f"edit_{k}", n, pid, None, float(pk[ng + 1 + i] - pk[ng]))
    return doses


def ml_causality(fc: FoldCtx, per: Period, t: pd.Timestamp, meal: Any, out: list) -> None:
    """Edits of the meal (carbs 120 g, removal, walk) change no ML forecast before the meal."""
    te = per.meals.loc[meal, "t"]
    before = per.rows[(per.rows["t"] < te) & (per.rows["t"] >= te - pd.Timedelta(hours=2))]
    if before.empty:
        return
    base = ml_forecast(fc, before, per.patient_id)
    for s in ([ed.set_macros(meal, carbs_g=120.0)], [ed.remove_meal(meal)], [ed.add_walk(meal)]):
        f = ml_forecast(fc, edited_rows(fc, per, before, s), per.patient_id)
        for n in ML:
            rec(out, "causality", n, per.patient_id, bool(np.array_equal(f[n], base[n])))


def ml_walk_window(fc: FoldCtx, per: Period, t: pd.Timestamp, meal: Any, out: list) -> None:
    """ML walk test: a completed post-meal walk does not raise the peak or the 2-h AUC."""
    row = per.rows[per.rows["t"] == t]
    now = float(row["cgm_now"].iloc[0])
    f = ml_batch(fc, per, row, [[], [ed.add_walk(meal)]])
    for n in ML:
        pk, auc = ml_peak_auc(f[n], now)
        ok = bool(pk[1] <= pk[0] + 1e-9 and auc[1] <= auc[0] + 1e-9)
        rec(out, "activity_direction", n, per.patient_id, ok, float(pk[1] - pk[0]))


def tw_bounds_ok(r: dict, pc: dict) -> bool:
    """Twin trajectory within bounds; |G(+30) - G(0)| / 30 within the rate limit."""
    g, dt = r["g"], float(r["dt"][0])
    lo, hi = pc["bounds_mgdl"]
    i30 = int(round(HORIZONS[0] / dt))
    return bool(
        np.all((g >= lo) & (g <= hi))
        and abs(g[i30] - g[0]) / HORIZONS[0] <= pc["max_rate_mgdl_min"]
    )


def twin_meal_window(fc: FoldCtx, per: Period, t: pd.Timestamp, meal: Any, out: list) -> float:
    """Twin (learned alpha) meal tests at one window; returns the 30 -> 60 g peak effect / 30."""
    pc, pid, al = fc.wc["plausibility"], per.patient_id, fc.alpha_on
    runs = [twin_run(fc, per, t, s, al) for s in carb_sets(meal, pc)]
    pk = np.array([tw_peak_auc(r)[0] for r in runs])
    steps = np.diff(pk)
    rec(
        out,
        "carb_monotonicity",
        "twin_learned",
        pid,
        bool(np.all(steps >= -pc["mono_tol"])),
        steps.min(),
    )
    rec(
        out,
        "carb_monotonicity_2mgdl",
        "twin_learned",
        pid,
        bool(np.all(steps >= -pc["mono_tol_secondary"])),
    )
    base = twin_run(fc, per, t, [], al)
    r0 = per.meals.loc[meal]
    same = twin_run(fc, per, t, [ed.set_macros(meal, carbs_g=float(r0["carbs_g"]))], al)
    rec(out, "identity", "twin_learned", pid, bool(np.array_equal(same["g"], base["g"])))
    rec(out, "bounds", "twin_learned", pid, all(tw_bounds_ok(r, pc) for r in runs))
    d = dose(pk, pc)
    lo, hi = pc["dose_window_range"]
    rec(out, "dose_window", "twin_learned", pid, bool(lo <= d <= hi), d)
    p0 = tw_peak_auc(base)[0]
    for s, k in zip(macro_sets(per, meal, pc), pc["macro_edits_g"], strict=True):
        rec(
            out,
            f"edit_{k}",
            "twin_learned",
            pid,
            None,
            tw_peak_auc(twin_run(fc, per, t, s, al))[0] - p0,
        )
    # causality: filter states before the meal are unchanged by edits of that meal
    te = np.datetime64(per.meals.loc[meal, "t"], "ns")
    nb = int(np.searchsorted(per.rec.t, te, side="left"))
    for s in ([ed.set_macros(meal, carbs_g=120.0)], [ed.remove_meal(meal)], [ed.add_walk(meal)]):
        h = twin_run(fc, per, t, s, al)["hist"]
        rec(
            out,
            "causality",
            "twin_learned",
            pid,
            bool(np.array_equal(h[:nb], base["hist"][:nb], equal_nan=True)),
        )
    return d


def twin_walk_window(fc: FoldCtx, per: Period, t: pd.Timestamp, meal: Any, out: list) -> None:
    """Twin walk test (walk after t, played forward): peak and AUC do not increase."""
    for name, al in twin_alphas(fc).items():
        if not np.isfinite(al):
            continue
        p0, a0 = tw_peak_auc(twin_run(fc, per, t, [], al))
        p1, a1 = tw_peak_auc(twin_run(fc, per, t, [ed.add_walk(meal)], al))
        ok = bool(p1 <= p0 + 1e-9 and a1 <= a0 + 1e-9)
        rec(out, "activity_direction", name, per.patient_id, ok, p1 - p0)
        rec(out, "walk_effect_auc", name, per.patient_id, None, a1 - a0)


def cross(ml: dict, d_tw: float, pid: str, pc: dict, out: list) -> None:
    """Cross-model consistency of the 30 -> 60 g peak effect (mg/dL) between ML and twin."""
    span = pc["dose_high_g"] - pc["dose_low_g"]
    b = d_tw * span
    for n, d in ml.items():
        a = d * span
        tol = max(pc["cross_abs_tol"], pc["cross_rel_tol"] * max(abs(a), abs(b)))
        ok = bool(np.sign(a) == np.sign(b) and abs(a - b) <= tol)
        rec(out, "cross_model", n, pid, ok, a - b)
        out[-1]["ml_effect"], out[-1]["twin_effect"] = a, b


def run_period(fc: FoldCtx, per: Period, seed: int) -> list[dict]:
    """Every test on the sampled windows of one period."""
    pc, out = fc.wc["plausibility"], []
    rng = np.random.default_rng(seed)
    has_twin = per.patient_id in fc.twins
    for t, meal in windows(per, pc, rng, "meal"):
        doses = ml_meal_window(fc, per, t, meal, out)
        ml_causality(fc, per, t, meal, out)
        if has_twin:
            d_tw = twin_meal_window(fc, per, t, meal, out)
            cross(doses, d_tw, per.patient_id, pc, out)
            twin_walk_window(fc, per, t, meal, out)
    for t, meal in windows(per, pc, rng, "walk"):
        ml_walk_window(fc, per, t, meal, out)
    for r in out:
        r["by_construction"] = (r["test"], r["model"]) in BY_CONSTRUCTION
    return out
