"""Live what-if inference from the shipped fold models (Part 9, D-058). No training data needed.

``RtForm`` loads the boosters exported by ``src.appdata.export`` (population model, inner-fold CV+
ensemble, residuals) and reproduces ``src.whatif.models.Form.predict`` / ``band`` (checked at
build time: ``make app-data`` stores the max difference). The edit semantics are the Part 7 ones
(``src.whatif.edits``): meals are edited in the log, meal / activity inputs are recomputed on the
period prefix with the Part 2 code, CGM, time and static inputs cannot change.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.uncertainty import conformal as cf
from src.uncertainty.events import exceed_prob
from src.whatif import edits as ed

HORIZONS = (30, 60, 120)


def _load_booster(path: Path) -> lgb.Booster:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return lgb.Booster(model_str=fh.read())


@dataclass
class RtForm:
    """One form (unc or mono) of the fold: population model + CV+ ensemble."""

    name: str
    cols: list[str]
    base: dict[int, str]
    pop: dict[int, lgb.Booster]
    inner: dict[int, dict[int, lgb.Booster]]
    res: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)

    def predict(self, df: pd.DataFrame, h: int) -> np.ndarray:
        """Absolute forecast (mg/dL) of the full model, which never saw the patient."""
        return df[self.base[h]].to_numpy(float) + self.pop[h].predict(df[self.cols])

    def band(self, df: pd.DataFrame, h: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """CV+ interval of the Part 6 arm (inner-fold models + their out-of-fold residuals)."""
        js = sorted(self.inner)
        m = np.column_stack(
            [df[self.base[h]].to_numpy(float) + self.inner[j][h].predict(df[self.cols]) for j in js]
        )
        return cf.cvplus_interval(m, [self.res[h][j] for j in js], alpha)


def load_forms(root: Path) -> dict[str, RtForm]:
    """Load both forms from ``root`` (``app/demo/models``)."""
    meta = json.loads((root / "models.json").read_text("utf-8"))
    res = np.load(root / "residuals.npz")
    forms = {}
    for name, fm in meta["forms"].items():
        pop = {h: _load_booster(root / f"{name}_{h}_pop.txt.gz") for h in HORIZONS}
        inner = {
            j: {h: _load_booster(root / f"{name}_{h}_in{j}.txt.gz") for h in HORIZONS}
            for j in fm["inner"]
        }
        r = {h: {j: res[f"{name}_{h}_{j}"] for j in fm["inner"]} for h in HORIZONS}
        forms[name] = RtForm(
            name, meta["cols"], {int(h): b for h, b in fm["base"].items()}, pop, inner, r
        )
    return forms


@dataclass
class Calibration:
    """Platt maps + OOF residuals of the fold's training patients (Part 6 event probabilities)."""

    res: dict[int, np.ndarray]
    platt: dict[str, dict[str, float] | None]
    events: dict[str, tuple[float, bool]]

    def prob(self, event: str, h: int, pred: np.ndarray) -> np.ndarray:
        """Calibrated P(event within h min) from a forecast; NaN if no map exists."""
        mp = self.platt.get(f"{event}|{h}")
        if mp is None:
            return np.full(np.shape(pred), np.nan)
        thr, above = self.events[event]
        s = np.clip(exceed_prob(np.asarray(pred, float), self.res[h], thr, above), 1e-4, 1 - 1e-4)
        z = mp["coef"] * np.log(s / (1 - s)) + mp["intercept"]
        return 1.0 / (1.0 + np.exp(-z))


def load_calibration(root: Path) -> Calibration:
    """Load ``calib.npz`` / ``calib.json`` written by the app-data builder."""
    meta = json.loads((root / "calib.json").read_text("utf-8"))
    r = np.load(root / "calib.npz")
    return Calibration(
        {h: r[f"res_{h}"] for h in HORIZONS},
        meta["platt"],
        {k: (v[0], bool(v[1])) for k, v in meta["events"].items()},
    )


@dataclass
class PatientState:
    """What a live what-if needs for one demo patient (shipped under app/demo/patients)."""

    pid: str
    rows: pd.DataFrame  # stored anchor rows (60 model inputs, no outcomes)
    grid: pd.DataFrame  # period grid prefix up to the last replay anchor
    meals: pd.DataFrame  # meal log up to the last replay anchor
    ectx: dict[str, Any]
    kcal_per_g: dict[str, float]
    walk: dict[str, Any]
    cols: list[str]


def edited_forecast(
    form: RtForm,
    st: PatientState,
    t: pd.Timestamp,
    edits: list[ed.Edit],
    alpha: float = 0.2,
    clamp: tuple[float, float] = (40.0, 400.0),
) -> dict[str, Any]:
    """Forecast, CV+ band and edited feature row at anchor ``t`` under ``edits`` (clamped)."""
    row = st.rows[st.rows["t"] == t]
    if len(row) != 1:
        raise ValueError("t is not an anchor of this patient")
    m = ed.apply_meals(st.meals, edits, st.kcal_per_g)
    g = ed.apply_walks(st.grid, st.meals, edits, st.walk)
    d = ed.recompute(row, g, m, st.cols, st.ectx)
    out: dict[str, Any] = {"row": d, "mean": [], "lo": [], "hi": [], "raw_mean": []}
    for h in HORIZONS:
        p = float(form.predict(d, h)[0])
        lo, hi = form.band(d, h, alpha)
        out["raw_mean"].append(p)
        out["mean"].append(float(np.clip(p, *clamp)))
        out["lo"].append(float(np.clip(lo[0], *clamp)))
        out["hi"].append(float(np.clip(hi[0], *clamp)))
    return out
