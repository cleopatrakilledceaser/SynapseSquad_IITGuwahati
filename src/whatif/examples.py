"""Example what-if scenarios for the report (D-050 / D-054). Training patient of fold 0 only.

1. CGMacros window: last meal set to 30 g vs 60 g carbs; with / without a post-meal walk. ML at a
   meal window (walk unknown at t), twin plays the walk forward (learned and ASSUMED alpha).
2. India presets: the same window with the meal replaced by preset macros (roti vs rice ...).
   Out of distribution: the models were trained on US meals (CGMacros).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.hybrid.twin_features import HORIZONS
from src.india.library import load_library
from src.whatif import context as cx
from src.whatif import edits as ed
from src.whatif.engine import whatif
from src.whatif.plausibility import windows

OOD = "OUT OF DISTRIBUTION: trained on US meals (CGMacros); preset macros are assumptions"


def pick_window(fc: Any, data: dict, wc: dict) -> tuple[Any, Any, Any]:
    """First fold-0 training patient (sorted) with a twin and a meal window (seed 42)."""
    rng = np.random.default_rng(wc["seed"])
    for pid in sorted(fc.twins):
        for rec in data["recs"][pid]:
            per = cx.period(data, rec.period_id)
            w = windows(per, wc["plausibility"], rng, "meal")
            if w:
                return per, w[len(w) // 2][0], w[len(w) // 2][1]
    raise RuntimeError("no example window")


def _clean(o: dict) -> dict:
    return {k: v for k, v in o.items() if k not in ("t",)} | {"t": str(o["t"])}


def build(wc: dict) -> dict[str, Any]:
    """Scenario outputs (means, bands, twin trajectories) with labels."""
    data = cx.load_data(wc)
    fc = cx.fold_ctx(data, wc, wc["dev_folds"][0], dev=True)
    per, t, meal = pick_window(fc, data, wc)
    alphas = wc["alphas"]
    out: dict[str, Any] = {
        "label": "model-based / SIMULATED what-if; observational bands, no counterfactual "
        "coverage guarantee; twin bands = filter variance (not conformal)",
        "patient": "fold-0 training patient (inner-fold models; never a test patient)",
        "horizons_min": HORIZONS,
        "meal_logged": {k: float(per.meals.loc[meal, k]) for k in ed.MACROS},
        "minutes_after_meal": float((t - per.meals.loc[meal, "t"]).total_seconds() / 60),
        "scenarios": {},
        "india": {"note": OOD, "scenarios": {}},
    }
    for c in wc["examples"]["carbs_g"]:
        for walk in (False, True):
            edits = [ed.set_macros(meal, carbs_g=float(c))] + ([ed.add_walk(meal)] if walk else [])
            key = f"{c:g} g carbs, {'walk' if walk else 'no walk'}"
            out["scenarios"][key] = _clean(whatif(fc, per, t, edits, alphas))
    lib = load_library()
    for name in wc["examples"]["india_presets"]:
        m = lib["presets"][name]["macros"]
        e = [ed.set_macros(meal, **{k: float(m[k]) for k in ed.MACROS})]
        r = _clean(whatif(fc, per, t, e, alphas))
        r["preset"] = {"label": lib["presets"][name]["label"], "macros": m}
        out["india"]["scenarios"][name] = r
    return out
