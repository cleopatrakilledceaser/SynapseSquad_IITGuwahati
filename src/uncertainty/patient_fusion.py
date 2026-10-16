"""Item 2 (D-048): next-day TAR180 / TIR from static vs first-day CGM+meals vs both.

One row per patient (first period). Ridge with alpha chosen by inner leave-one-out on the
training patients of each outer fold; patient-grouped 5-fold CV x 5 repeats (seed 42 + r).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from src.features.fusion_data import load_grid
from src.uncertainty.common import PROC


def day_stats(g: pd.DataFrame, lo_h: float, hi_h: float, min_frac: float) -> dict | None:
    """CGM summary of hours [lo_h, hi_h) of a period (None if < min_frac real)."""
    m = (g.minutes_since_start >= lo_h * 60) & (g.minutes_since_start < hi_h * 60)
    d = g[m]
    if len(d) == 0 or d.glucose_is_real.mean() < min_frac:
        return None
    y = d.glucose[d.glucose_is_real.astype(bool)].to_numpy(float)
    return {
        "mean": y.mean(),
        "sd": y.std(),
        "cv": y.std() / y.mean(),
        "tir": np.mean((y >= 70) & (y <= 180)) * 100,
        "tar180": np.mean(y > 180) * 100,
        "tbr70": np.mean(y < 70) * 100,
        "max": y.max(),
    }


def patient_table(cohort: str, pc: dict) -> pd.DataFrame:
    """Per-patient static, first-day dynamic and next-day target columns."""
    grid = load_grid(cohort)
    meals = pd.read_parquet(PROC / "meals.parquet")
    static = pd.read_parquet(PROC / "static.parquet").set_index("patient_id")
    static["sex_male"] = (static["sex"] == "M").astype(float).where(static["sex"].notna())
    rows = []
    for pid, g in grid.groupby("patient_id"):
        first = g.groupby("period_id").t.min().idxmin()
        g = g[g.period_id == first].sort_values("t")
        a = day_stats(g, *pc["first_day_h"], pc["min_real_frac"])
        b = day_stats(g, *pc["next_day_h"], pc["min_real_frac"])
        if a is None or b is None:
            continue
        r: dict[str, Any] = {"patient_id": pid}
        r |= {f"d1_{k}": v for k, v in a.items()}
        if cohort == "cgmacros":
            mm = meals[meals.period_id == first]
            mins = (mm.t - g.t.min()).dt.total_seconds() / 60
            sel = (mins >= pc["first_day_h"][0] * 60) & (mins < pc["first_day_h"][1] * 60)
            r["d1_meal_count"] = float(sel.sum())
            r["d1_carbs_g"] = float(mm.loc[sel, "carbs_g"].fillna(0).sum())
        for c in pc["static"][cohort]:
            r[f"s_{c}"] = static.loc[pid, c]
        r["y_tar180"], r["y_tir"] = b["tar180"], b["tir"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("patient_id")


def cv_predict(x: pd.DataFrame, y: np.ndarray, pc: dict, seed: int) -> np.ndarray:
    """Patient-grouped CV ridge predictions (training medians / scaling / LOO alpha)."""
    out = np.zeros(len(y))
    for tr, te in KFold(pc["n_splits"], shuffle=True, random_state=seed).split(x):
        xt, xv = x.iloc[tr].astype(float), x.iloc[te].astype(float)
        med = xt.median()
        xt, xv = xt.fillna(med), xv.fillna(med)
        mu, sd = xt.mean(), xt.std().replace(0, 1).fillna(1)
        m = RidgeCV(alphas=pc["ridge_alpha_grid"]).fit((xt - mu) / sd, y[tr])
        out[te] = m.predict((xv - mu) / sd)
    return out


def run(cohort: str, pc: dict, bc: dict) -> dict[str, Any]:
    """MAE of (i) static, (ii) dynamic, (iii) both, + paired patient-bootstrap differences."""
    t = patient_table(cohort, pc)
    sets = {
        "static": [c for c in t if c.startswith("s_")],
        "dynamic": [c for c in t if c.startswith("d1_")],
        "both": [c for c in t if c.startswith(("s_", "d1_"))],
    }
    res: dict[str, Any] = {"n_patients": int(len(t))}
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(t), (bc["n_boot"], len(t)))
    for tgt in ("y_tar180", "y_tir"):
        y = t[tgt].to_numpy(float)
        err = {}
        for name, cols in sets.items():
            p = np.mean([cv_predict(t[cols], y, pc, 42 + r) for r in range(pc["n_repeats"])], 0)
            err[name] = np.abs(y - p)
        base = np.abs(y - y.mean())  # reference: training-free mean (optimistic for a baseline)
        r_t: dict[str, Any] = {"mae": {k: float(v.mean()) for k, v in err.items()}}
        r_t["mae"]["cohort_mean_reference"] = float(base.mean())
        for a, b in (("both", "static"), ("both", "dynamic"), ("dynamic", "static")):
            d = err[a][idx].mean(1) - err[b][idx].mean(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            r_t[f"{a}-{b}"] = {
                "diff": float(err[a].mean() - err[b].mean()),
                "ci": [float(lo), float(hi)],
                "distinguishable_from_zero": bool(lo > 0 or hi < 0),
            }
        r_t["target_mean"], r_t["target_sd"] = float(y.mean()), float(y.std())
        res[tgt] = r_t
    return res
