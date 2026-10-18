"""Real forecasts and what-ifs for the console (D-069, D-070). ALL PATIENTS ARE SIMULATED.

Run: ``python -W ignore -u -m src.web.real_forecast`` after ``make web-data`` and ``make app-data``.
Rewrites ``web/data.json`` in place (not committed).

For each SIMULATED T2D patient of ``src.web.build_data`` this does what ``src.appdata.synthetic``
does for the Part 9 synthetic demo patient: the simulated record becomes model inputs (5-min period
grid, meal log with the Indian presets' macros, heart rate and activity of the cohort table; the
static profile is the simulated HbA1c and fasting glucose), then the shipped fold-0 models
(``src.appdata.rt``, ``app/demo/models``, ``calib.npz``) run on it. The models never saw these
patients.

D-070 calibration: the shipped models run on every 15-min anchor of each patient's record. The
few-shot offset of D-062 (``src.transfer.core.fewshot``, method ``offset``, k = 3 days: shrunk mean
residual of the pairs whose reading falls in days 1 to 3) is subtracted from every forecast of that
patient from day 4 on, per form and horizon: the "now" forecast, its band and all what-if values.
The 80 % band is the calibrated point plus the pooled 10th / 90th percentile of the calibration-day
residuals (after the offset) of all patients, per horizon. ``forecast_eval`` scores persistence, the
zero-shot model and the calibrated model on every anchor from day 4 on (patient bootstrap).

Replaced in data.json (same keys): ``fc``, ``ev`` (calibrated event probabilities from the
calibrated forecast; an event without a shipped calibration map is dropped and listed in
``ev_not_available``), ``forecast_source`` = "model". Added: ``forecast_tag``, ``forecast_note``,
``forecast_eval``, ``whatif_meta`` and per patient ``whatif`` (every value from
``rt.edited_forecast`` with the Part 7 edit semantics, minus the patient's offset).
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
from src.eval.metrics import bootstrap_point, paired_bootstrap_diff, point_stats
from src.personalize.methods import adapt_mask
from src.transfer import core
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
STRIDE = 3  # 15-min anchors on the 5-min grid
K_DAYS = 3.0  # D-062 few-shot k: calibration pairs read in days 1 to 3
EVAL_FROM_DAY = 3.0  # scored anchors: day 4 onward
N_BOOT, BOOT_SEED, CI = 2000, 7, 0.95
BAND_Q = (10, 90)
FORMS = tuple(dw.ML_FORMS)
METHODS = {
    "persistence": "Persistence",
    "zeroshot": "Model without calibration",
    "calibrated": "Model with 3-day calibration",
}
FORECAST_TAG = "Calibrated forecast"
FORECAST_NOTE = (
    "Model trained on open data, applied to simulated patients with an offset calibrated on each "
    "patient's first 3 days (D-070). Simulated patients, not clinical evidence."
)
BAND_NOTE = (
    " The 80 % range is the calibrated forecast plus the 10th to 90th percentile of the "
    "calibration-day errors pooled over all simulated patients."
)


def compare_note(res: dict[str, Any]) -> str:
    """One neutral sentence: calibrated against persistence at each horizon, from the intervals."""
    parts = []
    for h in HZ:
        lo, hi = res["diff"][h]["diff_mae_ci"]
        word = "better" if hi < 0 else "worse" if lo > 0 else "on par"
        parts.append(f"{word} at +{h} min")
    return (
        " Compared with assuming glucose stays where it is (persistence), the calibrated "
        "forecast is " + ", ".join(parts) + "."
    )


NOTES = [
    FORECAST_NOTE,
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
    off: dict[int, float],
    clamp: tuple[float, float],
) -> list[float]:
    """Calibrated what-if point forecast: raw forecast minus the patient's offset, clamped."""
    edits = [] if s is None else dw.edits_for(w, s)
    raw = rt.edited_forecast(form, st, w["t"], edits, ALPHA80, clamp)["raw_mean"]
    return [float(np.clip(raw[i] - off[h], *clamp)) for i, h in enumerate(HZ)]


def whatif_window(
    sh: Shipped,
    st: rt.PatientState,
    w: dict[str, Any],
    presets: list[dict[str, Any]],
    offs: dict[str, dict[int, float]],
) -> dict[str, Any]:
    """One what-if window: logged meal, carbohydrate grid and presets, walk off / on, both forms."""
    meal = st.meals.loc[w["meal_idx"]]
    base = dw.Scenario(*(float(meal[c]) for c in dw.MACROS))
    preset_meals = [dw.Scenario(*(float(p["macros"][k]) for k in dw.MACROS)) for p in presets]
    models = {}
    for name in FORMS:

        def fc(s: dw.Scenario | None, nm: str = name) -> dict[str, float]:
            v = _mean(sh.forms[nm], st, w, s, offs[nm], sh.clamp)
            return {str(h): round(float(v[i]), 1) for i, h in enumerate(HZ)}

        def walk_pair(s: dw.Scenario, nm: str = name) -> list[dict[str, float]]:
            return [fc(s, nm), fc(replace(s, walk=True), nm)]

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


# --------------------------------------------------------------------------- D-070 calibration


def anchor_table(
    sh: Shipped, grid: pd.DataFrame, meals: pd.DataFrame, hba1c: float, fasting: float
) -> pd.DataFrame:
    """Every 15-min anchor of one record: reading, later readings, raw forecasts, CV+ band.

    ``pred_{form}_{h}`` is the zero-shot model output before the clamp, ``zlo_{h}`` / ``zhi_{h}``
    the zero-shot CV+ 80 % band of the ``unc`` form, ``y_{h}`` the simulated reading h min later.
    """
    g = grid.sort_values("t").reset_index(drop=True)
    rows = model_inputs(g, meals, hba1c, fasting, sh)
    reading = g["glucose"].to_numpy(float)
    idx = np.arange(0, len(g), STRIDE)
    r = rows.iloc[idx].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "patient_id": r["patient_id"].to_numpy(),
            "t": r["t"].to_numpy(),
            "days_since_start": r["days_since_start"].to_numpy(float),
            "reading": reading[idx],
        }
    )
    step = int(sh.ectx["step"])
    for h in HZ:
        j = idx + h // step
        out[f"y_{h}"] = np.where(j < len(g), reading[np.minimum(j, len(g) - 1)], np.nan)
        for name in FORMS:
            out[f"pred_{name}_{h}"] = sh.forms[name].predict(r, h)
        out[f"zlo_{h}"], out[f"zhi_{h}"] = sh.forms["unc"].band(r, h, ALPHA80)
    return out


def offsets(tab: pd.DataFrame) -> dict[str, dict[int, float]]:
    """D-062 few-shot offset (k = 3 days) of one patient per form and horizon (to subtract).

    ``core.fewshot`` adds w * mean(reading - forecast) over the pairs read in days 1 to 3
    (``adapt_mask``), w = n / (n + n0), and no correction below ``min_pairs`` pairs.
    """
    cfg = load_config("transfer")["fewshot"]
    out: dict[str, dict[int, float]] = {}
    for name in FORMS:
        out[name] = {}
        for h in HZ:
            col = f"pred_{name}_{h}"
            d = tab[col].to_numpy(float) - core.fewshot(tab, col, h, K_DAYS, "offset", cfg)
            out[name][h] = float(d[0]) if len(d) else 0.0
    return out


def calibration_residuals(
    tab: pd.DataFrame, off: dict[str, dict[int, float]]
) -> dict[int, np.ndarray]:
    """Reading minus calibrated ``unc`` forecast on the calibration pairs (days 1 to 3)."""
    days = tab["days_since_start"].to_numpy(float)
    out = {}
    for h in HZ:
        y = tab[f"y_{h}"].to_numpy(float)
        m = adapt_mask(days, h, K_DAYS, y)
        out[h] = y[m] - (tab[f"pred_unc_{h}"].to_numpy(float)[m] - off["unc"][h])
    return out


def band_quantiles(residuals: list[dict[int, np.ndarray]]) -> dict[int, tuple[float, float]]:
    """Pooled 10th / 90th percentile of the calibration-day residuals of all patients."""
    out = {}
    for h in HZ:
        r = np.concatenate([x[h] for x in residuals])
        lo, hi = (float(v) for v in np.percentile(r, BAND_Q))
        if not lo <= 0.0 <= hi:
            raise RuntimeError(f"+{h}: calibration residual band [{lo}, {hi}] does not contain 0")
        out[h] = (lo, hi)
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
    """Anchor table, offsets, raw "now" forecast and calibrated what-ifs of one patient."""
    tab = anchor_table(sh, grid, meals, hba1c, fasting)
    off = offsets(tab)
    g = grid[grid["t"] <= now].reset_index(drop=True)
    m = meals[meals["t"] <= now].sort_values("t").reset_index(drop=True)
    if len(g) == 0 or g["t"].iloc[-1] != now:
        raise ValueError(f"{pid}: 'now' is not on the grid")
    if (now - g["t"].iloc[0]) / pd.Timedelta(days=1) < EVAL_FROM_DAY:
        raise ValueError(f"{pid}: 'now' is inside the calibration days")
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
    return {
        "table": tab,
        "offsets": off,
        "residuals": calibration_residuals(tab, off),
        "raw_now": [float(v) for v in f["raw_mean"]],
        "whatif": {"windows": [whatif_window(sh, st, w, presets, off) for w in wins]},
    }


def now_outputs(
    sh: Shipped, r: dict[str, Any], q: dict[int, tuple[float, float]]
) -> tuple[dict[str, Any], dict[str, float]]:
    """Calibrated ``fc`` (point and pooled-residual band, clamped) and event probabilities."""
    fc, cal = {}, {}
    for i, h in enumerate(HZ):
        p = r["raw_now"][i] - r["offsets"]["unc"][h]
        cal[h] = p
        fc[str(h)] = {
            "p": round(float(np.clip(p, *sh.clamp)), 1),
            "lo": round(float(np.clip(p + q[h][0], *sh.clamp)), 1),
            "hi": round(float(np.clip(p + q[h][1], *sh.clamp)), 1),
        }
    ev = {}
    for key, (e, h) in EVENTS.items():
        if e not in sh.cal.events:
            continue
        pr = float(sh.cal.prob(e, h, np.array([cal[h]]))[0])
        if np.isfinite(pr):
            ev[key] = round(pr, 3)
    return fc, ev


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


# --------------------------------------------------------------------------- D-070 evaluation


def scored(
    tables: list[pd.DataFrame],
    offs: dict[str, dict[str, dict[int, float]]],
    q: dict[int, tuple[float, float]],
    clamp: tuple[float, float],
) -> pd.DataFrame:
    """Anchors from day 4 on with persistence, zero-shot and calibrated forecasts and bands."""
    out = []
    for t in tables:
        d = t[t["days_since_start"] >= EVAL_FROM_DAY].copy()
        off = offs[str(d["patient_id"].iloc[0])]["unc"]
        for h in HZ:
            raw = d[f"pred_unc_{h}"].to_numpy(float)
            d[f"persistence_{h}"] = d["reading"]
            d[f"zeroshot_{h}"] = np.clip(raw, *clamp)
            d[f"calibrated_{h}"] = np.clip(raw - off[h], *clamp)
            d[f"clo_{h}"] = np.clip(raw - off[h] + q[h][0], *clamp)
            d[f"chi_{h}"] = np.clip(raw - off[h] + q[h][1], *clamp)
            d[f"zlo_{h}"] = np.clip(d[f"zlo_{h}"], *clamp)
            d[f"zhi_{h}"] = np.clip(d[f"zhi_{h}"], *clamp)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def evaluate(ev: pd.DataFrame) -> dict[str, Any]:
    """Patient-bootstrap MAE, paired calibrated - persistence difference, coverage by patient."""
    res: dict[str, Any] = {"mae": {}, "diff": {}, "coverage": {}}
    for h in HZ:
        st = {m: point_stats(ev, f"y_{h}", f"{m}_{h}") for m in METHODS}
        res["mae"][h] = {m: bootstrap_point(s, N_BOOT, BOOT_SEED, CI) for m, s in st.items()}
        res["diff"][h] = paired_bootstrap_diff(
            st["calibrated"], st["persistence"], N_BOOT, BOOT_SEED, CI
        )
        d = ev[ev[f"y_{h}"].notna()]
        y = d[f"y_{h}"]
        for band, (lo, hi) in {"calibrated": ("clo", "chi"), "zeroshot": ("zlo", "zhi")}.items():
            by_p = ((y >= d[f"{lo}_{h}"]) & (y <= d[f"{hi}_{h}"])).groupby(d["patient_id"]).mean()
            res["coverage"].setdefault(band, {})[h] = {
                "mean": float(by_p.mean()),
                "median": float(by_p.median()),
                "p10": float(by_p.quantile(0.1)),
                "p90": float(by_p.quantile(0.9)),
                "n_patients": int(len(by_p)),
            }
    d60 = res["diff"][60]
    res["passed"] = bool(d60["diff_mae"] < 0 and d60["diff_mae_ci"][1] < 0)
    res["n_rows"] = int(ev["y_60"].notna().sum())
    res["n_patients"] = int(ev["patient_id"].nunique())
    return res


def _cell(r: dict[str, Any]) -> str:
    lo, hi = r["mae_ci"]
    return f"{r['mae']:.1f} [{lo:.1f}, {hi:.1f}]"


def eval_table(res: dict[str, Any]) -> dict[str, Any]:
    """``forecast_eval`` for the console (Evidence, India readiness). SIMULATED."""
    cov = res["coverage"]

    def pcts(band: str) -> str:
        return ", ".join(f"+{h} min {100 * cov[band][h]['mean']:.0f} %" for h in HZ)

    d60 = res["diff"][60]
    verdict = compare_note(res).strip()
    note = (
        "Share of readings inside the 80 % range, mean over patients: calibrated "
        f"{pcts('calibrated')}; without calibration {pcts('zeroshot')}. Calibrated minus "
        f"persistence at +60 min: {d60['diff_mae']:+.1f} mg/dL [{d60['diff_mae_ci'][0]:+.1f}, "
        f"{d60['diff_mae_ci'][1]:+.1f}]. {verdict} Simulated patients, not clinical evidence."
    )
    return {
        "title": "Forecast error on the console's simulated patients (simulated)",
        "sub": (
            f"Mean absolute error, mg/dL, 95 % patient-bootstrap interval. {res['n_patients']} "
            f"simulated patients, every 15 minutes from day 4 on ({res['n_rows']:,} anchors at "
            "+60 min). Calibration: an offset from each patient's first 3 days (D-062, D-070)."
        ),
        "cols": ["Forecast", *(f"+{h} min" for h in HZ)],
        "rows": [[lab, *(_cell(res["mae"][h][m]) for h in HZ)] for m, lab in METHODS.items()],
        "note": note,
    }


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


def report(data: dict[str, Any], result: dict[str, Any], q: dict, offs: dict) -> None:
    """Run output: band, offsets, the evaluation table, paired differences, coverage, rule."""
    print("[web-real] pooled calibration-residual band, mg/dL: " + json.dumps(q))
    off = pd.DataFrame([{h: o["unc"][h] for h in HZ} for o in offs.values()])
    print("[web-real] offsets subtracted (unc), mg/dL, mean / min / max over patients:")
    for h in HZ:
        print(f"  +{h:>3} min  {off[h].mean():+6.1f} {off[h].min():+6.1f} {off[h].max():+6.1f}")
    t = data["forecast_eval"]
    print(f"[web-real] SIMULATED forecast_eval: {t['sub']}")
    for row in [t["cols"], *t["rows"]]:
        print("  " + " | ".join(row))
    for h in HZ:
        d = result["diff"][h]
        lo, hi = d["diff_mae_ci"]
        print(f"  calibrated - persistence +{h} min: {d['diff_mae']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    for band, cv in result["coverage"].items():
        for h in HZ:
            v = cv[h]
            print(
                f"  80 % band coverage by patient, {band} +{h} min: mean {v['mean']:.3f}, "
                f"median {v['median']:.3f}, 10th-90th pct {v['p10']:.3f} to {v['p90']:.3f}"
            )
    print(f"[web-real] D-070 decision rule {'PASSED' if result['passed'] else 'NOT MET'}")


def main(workers: int | None = None) -> None:
    """Rewrite web/data.json with calibrated forecasts, what-ifs and ``forecast_eval``."""
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
    print(f"[web-real] {len(jobs)} SIMULATED patients; fold-0 models from {APP}", flush=True)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(inputs,)) as pool:
        res = dict(pool.map(_one, jobs))
    q = band_quantiles([res[w]["residuals"] for w, _ in jobs])
    for p in data["patients"]:
        r = res[p["id"]]
        p["fc"], p["ev"] = now_outputs(sh, r, q)
        p["whatif"] = r["whatif"]
    offs = {ids[w]: res[w]["offsets"] for w, _ in jobs}
    result = evaluate(scored([res[w]["table"] for w, _ in jobs], offs, q, sh.clamp))
    data["forecast_source"] = "model"
    data["forecast_tag"] = FORECAST_TAG
    data["forecast_note"] = FORECAST_NOTE + BAND_NOTE + compare_note(result)
    data["forecast_eval"] = eval_table(result)
    data.pop("residual_sd", None)  # belonged to the development placeholder
    data["ev_not_available"] = missing_events(sh)
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
    for k, v in data["ev_not_available"].items():
        print(f"[web-real] dropped {k}: {v}")
    report(data, result, q, offs)


if __name__ == "__main__":
    main()
