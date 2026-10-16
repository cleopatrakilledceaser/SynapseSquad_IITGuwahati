"""What-if engine (Part 7, D-050): edit a window, re-forecast with unc / mono / twin + bands.

``whatif(fc, period, t, edits)`` returns forecasts at +30 / +60 / +120 min for each model with
observational bands (ML: conformal, twin: filter variance). Development uses the inner-fold model
that did not see the patient (``fc.dev = True``); the final stage uses the fold model (test
patients). Everything is causal: ML inputs use data at or before t; only the twin plays planned
meals / walks after t forward.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.hybrid.twin_features import HORIZONS
from src.mech.inputs import Record
from src.utils.config import PROJECT_ROOT
from src.whatif import edits as ed
from src.whatif import twin as tw
from src.whatif.models import Form

PROC = PROJECT_ROOT / "data" / "processed"


@dataclass
class Period:
    """One recording period: grid, meal log, anchor rows, twin record."""

    patient_id: str
    period_id: str
    g: pd.DataFrame
    meals: pd.DataFrame
    rows: pd.DataFrame
    rec: Record


@dataclass
class FoldCtx:
    """Everything needed to run what-ifs in one fold."""

    fold: int
    forms: dict[str, Form]
    twins: dict[str, tw.TwinSpec]
    alpha_on: float
    alpha_assumed: dict[str, float]
    ectx: dict[str, Any]
    wc: dict[str, Any]
    dev: bool
    cols: list[str]
    extra: dict[str, Any] = field(default_factory=dict)


def twin_specs(fold: int, pids: Sequence[str], role: str, mc: dict, k: float) -> dict:
    """TwinSpec per patient from the Part 5 twin fits (prior pop, level k) and EKF (q, r)."""
    fits = pd.read_parquet(PROC / "hybrid" / f"twinfeatfits_cgmacros_fold{fold}.parquet")
    ch = json.loads((PROC / "hybrid" / f"twinfeat_cgmacros_fold{fold}.json").read_text())
    qr = ch[f"qr_k{k:g}"]
    f = fits[(fits.k == k) & (fits.role == role) & (fits.prior == "pop")]
    f = f[f.patient_id.isin(pids)].set_index("patient_id")
    lp = [str(i) for i in range(6)]
    return {
        p: tw.TwinSpec(f.loc[p, lp].to_numpy(float), float(qr["q_scale"]), float(qr["r"]), mc)
        for p in f.index
    }


def part4_choices(fold: int) -> dict[str, Any]:
    """Part 4 twin choices of a fold (alpha_on, population prior)."""
    return json.loads((PROC / "mech" / f"twin_cgmacros_fold{fold}.json").read_text())


def ml_forecast(fc: FoldCtx, d: pd.DataFrame, pid: str) -> dict[str, np.ndarray]:
    """Unclipped ML forecasts, form -> (n_rows, n_horizons)."""
    who = pid if fc.dev else None
    return {
        n: np.column_stack([f.predict(d, h, who) for h in HORIZONS]) for n, f in fc.forms.items()
    }


def ml_bands(fc: FoldCtx, d: pd.DataFrame, pid: str, a: float) -> dict[str, np.ndarray]:
    """Observational bands, form -> (n_rows, n_horizons, 2) [lo, hi]."""
    who = pid if fc.dev else None
    out = {}
    for n, f in fc.forms.items():
        b = [np.column_stack(f.band(d, h, a, who)) for h in HORIZONS]
        out[n] = np.stack(b, axis=1)
    return out


def edited_rows(
    fc: FoldCtx, per: Period, rows: pd.DataFrame, edits: Sequence[ed.Edit]
) -> pd.DataFrame:
    """Anchor rows with ML inputs recomputed under the edits (meal log + walks)."""
    m = ed.apply_meals(per.meals, edits, fc.wc["kcal_per_g"])
    g = ed.apply_walks(per.g, per.meals, edits, fc.wc["walk"])
    return ed.recompute(rows, g, m, fc.cols, fc.ectx)


def twin_run(
    fc: FoldCtx, per: Period, t: pd.Timestamp, edits: Sequence[ed.Edit], alpha: float
) -> dict[str, np.ndarray]:
    """Twin rollout at t: meals known at t + planned (edited) meals after t, planned walks."""
    spec = fc.twins[per.patient_id]
    known = per.meals.index[per.meals["t"] <= t]
    m = ed.apply_meals(per.meals, edits, fc.wc["kcal_per_g"])
    planned = [e.meal for e in edits if e.kind in ("macros", "move")]
    # real meals after t are unknown at t (dropped); edited meals are planned (kept)
    m = m[m.index.isin(known) | m.index.isin(planned)]
    walks = [ed.walk_interval(per.meals.loc[e.meal, "t"], fc.wc["walk"]) for e in edits
             if e.kind == "walk"]  # fmt: skip
    g = ed.apply_walks(per.g, per.meals, edits, fc.wc["walk"])
    act = None if walks else per.rec.act  # meal edits do not change the activity input
    wk, ro = fc.wc["walk"], fc.wc["twin"]["rollout_min"]
    return tw.run(spec, per.rec, g, m, walks, t, alpha, wk, ro, act)


def whatif(
    fc: FoldCtx, per: Period, t: pd.Timestamp, edits: Sequence[ed.Edit], alphas: Sequence[float]
) -> dict[str, Any]:
    """Forecasts + bands at +30/+60/+120 for every model under ``edits`` at anchor t."""
    row = per.rows[per.rows["t"] == t]
    if len(row) != 1:
        raise ValueError("t is not an anchor of this period")
    d = edited_rows(fc, per, row, edits)
    out: dict[str, Any] = {"t": t, "cgm_now": float(row["cgm_now"].iloc[0])}
    pf = ml_forecast(fc, d, per.patient_id)
    for a in alphas:
        b = ml_bands(fc, d, per.patient_id, a)
        for n in fc.forms:
            out.setdefault(n, {"mean": pf[n][0].tolist()})[f"band{round(100 * (1 - a))}"] = b[n][
                0
            ].tolist()
    if per.patient_id not in fc.twins:
        return out
    z = {0.2: fc.wc["twin"]["z80"], 0.05: fc.wc["twin"]["z95"]}
    # twin = Part 5 spec (alpha_E 0); twin_learned = Part 4 alpha_on; twin_assumed = D-052 (20 %)
    variants = {"twin": 0.0, "twin_learned": fc.alpha_on}
    variants["twin_assumed"] = fc.alpha_assumed[f"{fc.wc['twin']['assumed']['reduction']:g}"]
    for name, al in variants.items():
        if not np.isfinite(al):
            continue
        r = twin_run(fc, per, t, edits, al)
        hz = tw.at_horizons(r, HORIZONS)
        out[name] = {"mean": [hz[h][0] for h in HORIZONS], "traj": r["g"].tolist(), "alpha": al}
        for a in alphas:
            out[name][f"band{round(100 * (1 - a))}"] = [
                [hz[h][0] - z[a] * hz[h][1], hz[h][0] + z[a] * hz[h][1]] for h in HORIZONS
            ]
    return out
