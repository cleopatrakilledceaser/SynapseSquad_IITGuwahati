"""Real forecasts and what-ifs for the console (D-069). ALL PATIENTS ARE SIMULATED.

Run: ``python -W ignore -u -m src.web.real_forecast`` after ``make web-data`` and ``make app-data``.
Rewrites ``web/data.json`` in place (not committed).

For each SIMULATED T2D patient of ``src.web.build_data`` this does what ``src.appdata.synthetic``
does for the Part 9 synthetic demo patient: the simulated record becomes model inputs (5-min period
grid, meal log with the Indian presets' macros, heart rate and activity of the cohort table; the
static profile is the simulated HbA1c and fasting glucose), then the shipped fold-0 models
(``src.appdata.rt``, ``app/demo/models``, ``calib.npz``) run at the patient's "now" anchor. The
models never saw these patients: results are zero-shot on simulated patients.

Replaced in data.json (same keys): ``fc`` (point forecast and CV+ 80 % band, clamped as in Part 9),
``ev`` (calibrated event probabilities; an event without a shipped calibration map is dropped and
listed in ``ev_not_available``) and ``forecast_source`` = "model". Added: ``whatif_meta`` and per
patient ``whatif`` (every value from ``rt.edited_forecast`` with the Part 7 edit semantics).

The MAE of the point forecast against the simulated future and the persistence MAE are printed in
the run output only: one forecast per patient is an illustration, not an accuracy claim.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.appdata import rt
from src.appdata import synthetic as sy
from src.dash import whatif as dw
from src.utils.config import PROJECT_ROOT, load_config
from src.web import build_data as bd

APP = PROJECT_ROOT / "app" / "demo"
OUT = bd.OUT
HZ = rt.HORIZONS
ALPHA80 = 0.2
CLIP = (40.0, 405.0)  # input glucose clip of the app builder (src.appdata.build.CLIP)
CARBS_GRID = list(range(0, 151, 10))
MAX_WINDOWS = 3
WINDOW_GAP_MIN = 125  # the meal is at least this long before "now" (its 2 h are in the history)
WINDOW_LEAD_MIN = 180  # the what-if chart draws 3 h of history before the meal
# console key -> (calibrated event, horizon); an event with no shipped map is dropped
EVENTS = {
    "hi180_60": ("hyper180", 60),
    "hi180_120": ("hyper180", 120),
    "hi250_120": ("hyper250", 120),
    "lo70_60": ("hypo70", 60),
}
ZERO_SHOT = "Model trained on open data, applied zero-shot to simulated patients."
NOTES = [
    ZERO_SHOT,
    "All results are SIMULATED. Error against the simulated future is not an accuracy claim.",
    "Twin what-ifs are not offered for these patients: no twin was fitted to them.",
    f"Carbohydrate edits are tested from {CARBS_GRID[0]} to {CARBS_GRID[-1]} g; larger meals "
    "are clamped to the grid edge. Protein, fat and fibre stay as logged.",
]


@dataclass
class Shipped:
    """The app bundle's models, calibration and what-if context (no LightGBM, no training data)."""

    forms: dict[str, rt.RtForm]
    cal: rt.Calibration
    ectx: dict[str, Any]
    kcal_per_g: dict[str, float]
    walk: dict[str, Any]
    cols: list[str]
    clamp: tuple[float, float]


def load_shipped(root: Path = APP) -> Shipped:
    """Read ``app/demo`` (written by ``make app-data``)."""
    ctx = json.loads((root / "context.json").read_text("utf-8"))
    meta = json.loads((root / "models" / "models.json").read_text("utf-8"))
    clamp = tuple(float(v) for v in load_config("app")["display"]["clamp"])
    return Shipped(
        rt.load_forms(root / "models"),
        rt.load_calibration(root),
        ctx["ectx"],
        ctx["kcal_per_g"],
        ctx["walk"],
        list(meta["cols"]),
        clamp,
    )


def model_inputs(
    grid: pd.DataFrame, meals: pd.DataFrame, hba1c: float, fasting: float, sh: Shipped
) -> pd.DataFrame:
    """Anchor rows of every grid step (as ``src.appdata.synthetic.build``); missing inputs NaN."""
    g = grid.reset_index(drop=True).copy()
    g["glucose"] = np.clip(g["glucose"].to_numpy(float), *CLIP)
    rows = sy.anchor_rows(
        g,
        meals,
        sh.ectx["fcfg"],
        int(sh.ectx["n_h"]),
        float(hba1c),
        float(fasting),
        int(sh.ectx["step"]),
    )
    for c in sh.cols:
        if c not in rows.columns:
            rows[c] = np.nan
    return rows


def windows(meals: pd.DataFrame, now: pd.Timestamp) -> list[dict[str, Any]]:
    """Up to 3 most recent logged meals of the last 24 h that end >= 125 min before now."""
    day0 = now - pd.Timedelta(hours=24)
    m = (meals["t"] - day0).dt.total_seconds() / 60
    ok = (m >= WINDOW_LEAD_MIN) & (meals["t"] <= now - pd.Timedelta(minutes=WINDOW_GAP_MIN))
    sel = meals[ok].sort_values("t").tail(MAX_WINDOWS)
    return [{"meal_idx": i, "t": r["t"], "m": int(round(m[i]))} for i, r in sel.iterrows()]


def with_carbs(base: dw.Scenario, carbs: float) -> dw.Scenario:
    """The logged meal with ``carbs`` g carbohydrate; protein, fat and fibre stay as logged.

    D-069 also asked for protein / fat / fibre scaled per gram of carbohydrate, but those inputs
    are not monotone-constrained, so scaling them breaks the monotone model's non-decreasing
    carbohydrate curve that D-069 requires (checked on a simulated patient). A carbs-only edit is
    what ``src.dash.whatif.edits_for`` applies for a carbohydrate change.
    """
    return replace(base, carbs_g=float(carbs))


def _mean(
    form: rt.RtForm,
    st: rt.PatientState,
    w: dict[str, Any],
    s: dw.Scenario | None,
    clamp: tuple[float, float],
) -> list[float]:
    edits = [] if s is None else dw.edits_for(w, s)
    return rt.edited_forecast(form, st, w["t"], edits, ALPHA80, clamp)["mean"]


def whatif_window(
    sh: Shipped, st: rt.PatientState, w: dict[str, Any], presets: list[dict[str, Any]]
) -> dict[str, Any]:
    """One what-if window: logged meal, carbohydrate grid and presets, walk off / on, both forms."""
    meal = st.meals.loc[w["meal_idx"]]
    base = dw.Scenario(*(float(meal[c]) for c in dw.MACROS))
    preset_meals = [dw.Scenario(*(float(p["macros"][k]) for k in dw.MACROS)) for p in presets]
    models = {}
    for name in dw.ML_FORMS:

        def fc(s: dw.Scenario | None, f: rt.RtForm = sh.forms[name]) -> dict[str, float]:
            v = _mean(f, st, w, s, sh.clamp)
            return {str(h): round(float(v[i]), 1) for i, h in enumerate(HZ)}

        def walk_pair(s: dw.Scenario) -> list[dict[str, float]]:
            return [fc(s), fc(replace(s, walk=True))]

        grid = [walk_pair(with_carbs(base, c)) for c in CARBS_GRID]
        pre = [walk_pair(s) for s in preset_meals]
        models[name] = {
            "base": fc(None),
            "logged_walk": fc(replace(base, walk=True)),
            "pred": {str(h): [[g[k][str(h)] for g in grid] for k in (0, 1)] for h in HZ},
            "presets": [{"pred": {str(h): [q[0][str(h)], q[1][str(h)]] for h in HZ}} for q in pre],
        }
    carbs = float(meal["carbs_g"])
    label = (
        f"{pd.Timestamp(meal['t']).strftime('%H:%M')} {meal['meal_type']}: "
        f"{meal.get('preset_label', 'logged meal')}, {carbs:.0f} g carbohydrate"
    )
    out = {"label": label, "m": w["m"], "base_macros": {"carbs": round(carbs, 1)}, "models": models}
    if carbs > CARBS_GRID[-1]:
        out["note"] = (
            f"Logged meal has {carbs:.0f} g carbohydrate, above the tested range; "
            f"carbohydrate edits are clamped to {CARBS_GRID[-1]} g."
        )
    return out


def patient(
    sh: Shipped,
    pid: str,
    grid: pd.DataFrame,
    meals: pd.DataFrame,
    hba1c: float,
    fasting: float,
    now: pd.Timestamp,
    presets: list[dict[str, Any]],
) -> dict[str, Any]:
    """``fc``, ``ev``, ``whatif`` and raw point forecasts of one simulated patient at ``now``."""
    g = grid[grid["t"] <= now].reset_index(drop=True)
    m = meals[meals["t"] <= now].sort_values("t").reset_index(drop=True)
    if len(g) == 0 or g["t"].iloc[-1] != now:
        raise ValueError(f"{pid}: 'now' is not on the grid")
    rows = model_inputs(g, m, hba1c, fasting, sh)
    wins = windows(m, now)
    keep = {now, *(w["t"] for w in wins)}
    st = rt.PatientState(
        pid,
        rows[rows["t"].isin(keep)].reset_index(drop=True),
        g,
        m,
        sh.ectx,
        sh.kcal_per_g,
        sh.walk,
        sh.cols,
    )
    f = rt.edited_forecast(sh.forms["unc"], st, now, [], ALPHA80, sh.clamp)
    fc = {
        str(h): {
            "p": round(f["mean"][i], 1),
            "lo": round(f["lo"][i], 1),
            "hi": round(f["hi"][i], 1),
        }
        for i, h in enumerate(HZ)
    }
    ev = {}
    for key, (e, h) in EVENTS.items():
        if e not in sh.cal.events:
            continue
        p = float(sh.cal.prob(e, h, np.array([f["raw_mean"][HZ.index(h)]]))[0])
        if np.isfinite(p):
            ev[key] = round(p, 3)
    wi = {"windows": [whatif_window(sh, st, w, presets) for w in wins]}
    return {
        "fc": fc,
        "ev": ev,
        "whatif": wi,
        "point": {str(h): f["mean"][i] for i, h in enumerate(HZ)},
    }


def preset_list() -> list[dict[str, Any]]:
    """India presets for the What-if lab (src.india.library, D-054)."""
    return [
        {
            "key": k,
            "label": p["label"],
            "portion": p["portion"],
            "macros": {c: float(p["macros"][c]) for c in dw.MACROS},
        }
        for k, p in dw.presets().items()
    ]


def missing_events(sh: Shipped) -> dict[str, str]:
    """Console events with no shipped calibration map (dropped, not invented)."""
    out = {}
    for key, (e, h) in EVENTS.items():
        if e not in sh.cal.events or sh.cal.platt.get(f"{e}|{h}") is None:
            out[key] = f"no calibrated '{e}' map at +{h} min in the shipped models; not shown"
    return out


# --------------------------------------------------------------------------- run

_W: dict[str, Any] = {}


def _init(inputs: dict[str, Any]) -> None:
    _W.update(inputs, sh=load_shipped())


def _one(job: tuple[str, str]) -> tuple[str, dict[str, Any]]:
    wid, now = job
    pid = _W["ids"][wid]
    c = _W["c"]
    s = _W["static"].set_index("patient_id").loc[pid]
    meals = c["meals"][c["meals"]["patient_id"] == pid].copy()
    meals["preset_label"] = meals["preset"].map(_W["labels"]).fillna("logged meal")
    out = patient(
        _W["sh"],
        pid,
        c["grid"][c["grid"]["patient_id"] == pid],
        meals,
        float(s["hba1c_pct"]),
        float(s["fasting_glucose_mgdl"]),
        pd.Timestamp(now),
        _W["presets"],
    )
    return wid, out


def mae_report(data: dict[str, Any], points: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """MAE of the point forecast and of persistence vs the simulated future (run output only)."""
    rows = []
    for h in HZ:
        k = h // bd.STEP - 1
        ok = [p for p in data["patients"] if len(p["future"]) > k]
        y = np.array([p["future"][k] for p in ok])
        model = np.array([points[p["id"]][str(h)] for p in ok])
        pers = np.array([p["hist"][-1] for p in ok])
        rows.append(
            {
                "h": h,
                "n": len(ok),
                "model": float(np.mean(np.abs(model - y))),
                "persistence": float(np.mean(np.abs(pers - y))),
            }
        )
    return rows


def main(workers: int | None = None) -> None:
    """Rewrite web/data.json with model forecasts and what-ifs (SIMULATED patients)."""
    t0 = time.time()
    data = json.loads(OUT.read_text("utf-8"))
    sh = load_shipped()
    presets = preset_list()
    c, static, _ = bd.simulate()
    ids = {bd.web_id(p): p for p in static["patient_id"]}
    jobs = [(p["id"], p["now"]) for p in data["patients"]]
    inputs = {
        "c": c,
        "static": static,
        "ids": ids,
        "presets": presets,
        "labels": {p["key"]: p["label"] for p in presets},
    }
    print(
        f"[web-real] {len(jobs)} SIMULATED patients; zero-shot fold-0 models from {APP}", flush=True
    )
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(inputs,)) as pool:
        res = dict(pool.map(_one, jobs))
    points, dropped = {}, missing_events(sh)
    for p in data["patients"]:
        r = res[p["id"]]
        p["fc"], p["ev"], p["whatif"] = r["fc"], r["ev"], r["whatif"]
        points[p["id"]] = r["point"]
    data["forecast_source"] = "model"
    data["forecast_note"] = ZERO_SHOT
    data.pop("residual_sd", None)  # belonged to the development placeholder
    data["ev_not_available"] = dropped
    data["whatif_meta"] = {
        "carbs_grid": CARBS_GRID,
        "models": dict(dw.ML_FORMS),
        "presets": presets,
        "walk_min": int(sh.walk["duration_min"]),
        "notes": NOTES,
    }
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    n_win = sum(len(p["whatif"]["windows"]) for p in data["patients"])
    bad = sum(
        any(np.any(np.diff(lst) < 0) for h in w["models"]["mono"]["pred"].values() for lst in h)
        for p in data["patients"]
        for w in p["whatif"]["windows"]
    )
    print(
        f"[web-real] wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.0f} s; "
        f"{n_win} what-if windows; monotone-model windows with a decreasing carbs curve: {bad}"
    )
    for k, v in dropped.items():
        print(f"[web-real] dropped {k}: {v}")
    print(
        "[web-real] SIMULATED point-forecast MAE (mg/dL) vs the simulated future, one 'now' per "
        "patient; illustration only, not shown on the page:"
    )
    for r in mae_report(data, points):
        flag = "  <-- model WORSE than persistence" if r["model"] > r["persistence"] else ""
        print(
            f"  +{r['h']:>3} min  n={r['n']:>3}  model {r['model']:6.1f}  "
            f"persistence {r['persistence']:6.1f}{flag}"
        )


if __name__ == "__main__":
    main()
