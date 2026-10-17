"""What-if computations of the dashboard (Part 9, D-058). Pure numpy / pandas, no Streamlit.

Machine-learning what-ifs run LIVE from the shipped fold models (``src.appdata.rt``) with the Part 7
edit semantics: the meal log and walk are edited, meal and activity inputs are recomputed on the
period prefix with the Part 2 code, CGM, time and profile inputs cannot change.

Twin what-ifs are read from the grid precomputed at build time (carbs x slowing ratio, walk on or
off, two walk-effect variants) by bilinear interpolation. Values between grid nodes are
interpolated and are labelled as such. The twin baseline (the logged meal, no walk) is also read
from the grid, so it is interpolated too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.appdata import rt
from src.appdata.grid_interp import interp
from src.dash.bundle import HORIZONS, PatientData
from src.india.library import load_library
from src.utils.config import load_config
from src.whatif import edits as ed

ML_FORMS = {"mono": "Monotone model (default)", "unc": "Unconstrained model"}
TWIN_VARIANTS = {"twin_learned": "Twin, learned walk effect", "twin_assumed": "Twin, assumed walk effect"}
MODEL_LABELS = {**ML_FORMS, **TWIN_VARIANTS}
MACROS = ("carbs_g", "protein_g", "fat_g", "fiber_g")


@dataclass(frozen=True)
class Scenario:
    """A what-if: replace the logged meal by these macros (grams) and optionally add a walk."""

    carbs_g: float
    protein_g: float
    fat_g: float
    fiber_g: float
    walk: bool = False


def presets() -> dict[str, dict[str, Any]]:
    """India meal presets with per-portion macros (our own assumed recipes, D-054)."""
    lib = load_library()
    out = {}
    for key, p in lib["presets"].items():
        if "macros" in p:
            out[key] = {"label": p["label"], "portion": p["portion"], "macros": p["macros"],
                        "source": p.get("source", "")}  # fmt: skip
    return out


def logged_scenario(pdata: PatientData, window: dict[str, Any]) -> Scenario:
    """The patient's logged meal as a (no-walk) scenario."""
    m = pdata.meals.loc[window["meal_idx"]]
    return Scenario(*(float(m[c]) for c in MACROS), walk=False)


def slow_ratio(s: Scenario) -> float:
    """Slowing ratio the twin uses: (fat + w x protein + fibre) / (carbs + 1)."""
    w = float(load_config("mech")["params"]["slow_protein_weight"])
    return float((s.fat_g + w * s.protein_g + s.fiber_g) / (s.carbs_g + 1.0))


def patient_state(pdata: PatientData, ctx: dict[str, Any], cols: list[str]) -> rt.PatientState:
    """Live what-if state of a patient from the bundle."""
    return rt.PatientState(
        pid=pdata.pid,
        rows=pdata.replay,
        grid=pdata.grid,
        meals=pdata.meals,
        ectx=ctx["ectx"],
        kcal_per_g=ctx["kcal_per_g"],
        walk=ctx["walk"],
        cols=cols,
    )


def edits_for(window: dict[str, Any], s: Scenario) -> list[ed.Edit]:
    """Part 7 edits realizing a scenario on the window's meal."""
    m = window["meal_idx"]
    out = [ed.set_macros(m, carbs_g=s.carbs_g, protein_g=s.protein_g, fat_g=s.fat_g, fiber_g=s.fiber_g)]
    return out + ([ed.add_walk(m)] if s.walk else [])


def ml_run(
    form: rt.RtForm, state: rt.PatientState, window: dict[str, Any], s: Scenario,
    clamp: tuple[float, float], alpha: float = 0.2,
) -> dict[str, Any]:  # fmt: skip
    """Baseline (logged meal, no walk) and what-if forecasts of one ML form at the window."""
    t = pd.Timestamp(window["t"])
    base = rt.edited_forecast(form, state, t, [], alpha, clamp)
    new = rt.edited_forecast(form, state, t, edits_for(window, s), alpha, clamp)
    return {
        "kind": "ml",
        "base": _pack(base["mean"], base["lo"], base["hi"]),
        "new": _pack(new["mean"], new["lo"], new["hi"]),
        "interpolated": False,
    }


def _pack(mean: list[float], lo: list[float] | None, hi: list[float] | None) -> dict[int, dict[str, float | None]]:
    return {
        h: {"mean": float(mean[i]), "lo": None if lo is None else float(lo[i]),
            "hi": None if hi is None else float(hi[i])}
        for i, h in enumerate(HORIZONS)
    }  # fmt: skip


def twin_run(
    pdata: PatientData, window: dict[str, Any], variant: str, s: Scenario,
    base: Scenario, ctx: dict[str, Any], clamp: tuple[float, float],
) -> dict[str, Any] | None:  # fmt: skip
    """Twin baseline and what-if from the precomputed grid (None if the patient has no twin)."""
    arr = pdata.twin.get(f"w{window['k']}")
    if arr is None:
        return None
    grid = ctx["twin_grid"]
    cs, rs = grid["carbs_g"], grid["slow_ratio"]
    iv = list(ctx["variants"]).index(variant)

    def traj(sc: Scenario) -> tuple[np.ndarray, bool]:
        return interp(arr[iv, int(sc.walk)], cs, rs, sc.carbs_g, slow_ratio(sc))

    b, ib = traj(Scenario(base.carbs_g, base.protein_g, base.fat_g, base.fiber_g, False))
    n, in_ = traj(s)
    dt = 120.0 / (arr.shape[-1] - 1)
    idx = {h: int(round(h / dt)) for h in HORIZONS}

    def pack(g: np.ndarray) -> dict[int, dict[str, float | None]]:
        return {h: {"mean": float(np.clip(g[i], *clamp)), "lo": None, "hi": None} for h, i in idx.items()}

    return {
        "kind": "twin", "base": pack(b), "new": pack(n), "interpolated": bool(ib or in_),
        "base_traj": np.clip(b, *clamp), "new_traj": np.clip(n, *clamp), "dt": dt,
        "clamped_inputs": bool(s.carbs_g > cs[-1] or slow_ratio(s) > rs[-1]),
    }  # fmt: skip


def comparison_table(res: dict[str, Any]) -> pd.DataFrame:
    """Baseline and what-if at the three horizons, in mg/dL."""
    rows = []
    for h in HORIZONS:
        b, n = res["base"][h], res["new"][h]
        row = {"Horizon": f"+{h} min", "Baseline": f"{b['mean']:.0f} mg/dL", "What-if": f"{n['mean']:.0f} mg/dL",
               "Change": f"{n['mean'] - b['mean']:+.0f} mg/dL"}  # fmt: skip
        if n.get("lo") is not None:
            row["What-if 80% interval"] = f"{n['lo']:.0f} to {n['hi']:.0f} mg/dL"
        else:
            row["What-if 80% interval"] = "not available for the twin"
        rows.append(row)
    return pd.DataFrame(rows)


def window_label(pdata: PatientData, w: dict[str, Any]) -> str:
    """'09:33 meal, 66 g carbs (22 min in)'."""
    m = pdata.meals.loc[w["meal_idx"]]
    return f"{pd.Timestamp(m['t']).strftime('%a %H:%M')} meal, {float(m['carbs_g']):.0f} g carbs ({w['offset_min']:.0f} min in)"
