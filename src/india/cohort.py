"""Synthetic Indian-phenotype cohort ``synthetic_india_v1`` (Part 8b, D-062). ALL SYNTHETIC.

The Part 4 mechanistic simulator with (i) the mean of the log parameters shifted towards a
"lean but insulin-resistant" phenotype (every shift is cited or marked ASSUMPTION in
configs/india.yaml), (ii) Indian meal days built from the assumed-recipe library, (iii) a simple
documented wearable generator, and (iv) static fields derived from the simulated CGM (HbA1c from
the mean glucose via the ADAG eAG equation). Strength 0 gives a CONTROL cohort (same diet, real
parameter distribution) so the effect of the physiological shift can be isolated from the diet.
Outputs carry ``synthetic = True``; ids start with the cohort name. Never mixed with real data.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.india.library import load_library
from src.mech.fit import log_bounds
from src.mech.model import FIT_PARAMS, bounds_array, initial_state, simulate, theta_from_log
from src.mech.simulate import ar1_noise
from src.utils.config import PROJECT_ROOT

STEPS_PER_DAY = 288
DT_MIN = 5.0


def load_fits(icfg: dict[str, Any]) -> pd.DataFrame:
    """Real per-patient twin fits (CGMacros, k = 3 days) from the stored Part 4 table."""
    s = icfg["fit_source"]
    d = pd.read_csv(PROJECT_ROOT / s["csv"])
    d = d[(d["model"] == s["model"]) & (d["cohort"] == s["cohort"])]
    if d.empty:
        raise ValueError("no fits found for the configured model / cohort")
    return d


def load_noise_and_alpha(icfg: dict[str, Any]) -> tuple[dict[str, float], float]:
    """Sensor-noise parameters and the population activity gain stored by Part 4."""
    m = json.loads((PROJECT_ROOT / icfg["noise_source"]["json"]).read_text(encoding="utf-8"))
    met = m["metrics"]
    alpha = float(np.median([c["alpha_on"] for c in met["fold_choices"]]))
    return met["simulator"]["noise"], alpha


def shifted_mean(mu: np.ndarray, shifts: dict[str, float], strength: float) -> np.ndarray:
    """Log-parameter mean with the configured log shifts scaled by ``strength``."""
    out = mu.copy()
    for name, d in shifts.items():
        out[FIT_PARAMS.index(name)] += strength * float(d)
    return out


def meal_day(
    rng: np.random.Generator, lib: dict[str, Any], mcfg: dict[str, Any], patient_factor: float
) -> list[dict[str, Any]]:
    """One day of meals: [{step, key, scale, macros}] with macros scaled to the eaten portion."""
    out = []
    for slot, s in mcfg["slots"].items():
        if rng.random() >= s["p"]:
            continue
        minute = float(np.clip(rng.normal(*s["time"]), 5, 1435))
        key = str(rng.choice(s["pool"]))
        scale = s.get("portion", 1.0) * patient_factor * float(rng.lognormal(0, mcfg["portion_sd_meal"]))
        mac = {k: float(v) * scale for k, v in lib["presets"][key]["macros"].items()}
        out.append({"slot": slot, "step": int(minute // DT_MIN), "minute": minute, "key": key,
                    "scale": scale, **mac})  # fmt: skip
    return sorted(out, key=lambda m: m["step"])


def meal_inputs(
    days: list[list[dict[str, Any]]], w_protein: float
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """(carbs, slow, meal list with absolute steps) for consecutive days.

    A meal enters at its grid step; the slowing ratio (fat + w x protein + fibre) / (carbs + 1)
    is that of the most recent meal (as ``src.mech.inputs.meal_arrays``).
    """
    n = len(days) * STEPS_PER_DAY
    carbs, ratio = np.zeros(n), np.full(n, np.nan)
    meals = []
    for d, day in enumerate(days):
        for m in day:
            i = d * STEPS_PER_DAY + m["step"]
            carbs[i] += m["carbs_g"]
            ratio[i] = (m["fat_g"] + w_protein * m["protein_g"] + m["fiber_g"]) / (m["carbs_g"] + 1.0)
            meals.append({**m, "abs_step": i})
    slow = pd.Series(ratio).ffill().fillna(0.0).to_numpy()
    return carbs, slow, meals


def activity_days(rng: np.random.Generator, n_days: int, acfg: dict[str, Any]) -> np.ndarray:
    """Activity input A(t) (mean about 0.2 in daytime, 1-2 in walking bouts) on the 5-min grid."""
    a = np.zeros(n_days * STEPS_PER_DAY)
    base = acfg["daytime_base"]
    for d in range(n_days):
        o = d * STEPS_PER_DAY
        a[o + int(base["from_min"] // DT_MIN) : o + int(base["to_min"] // DT_MIN)] = base["level"]
        for slot in ("morning", "evening"):
            s = acfg[slot]
            if rng.random() < s["p"]:
                start = int(np.clip(rng.normal(*s["time"]), 0, 1439) // DT_MIN)
                length = int(rng.uniform(*s["duration"]) // DT_MIN)
                a[o + start : o + min(start + length, STEPS_PER_DAY)] = rng.uniform(*s["level"])
    return a


def wearables(
    rng: np.random.Generator, act: np.ndarray, acfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(heart rate, activity intensity 0-3, rest proxy) derived from the activity input."""
    h = acfg["hr"]
    minute = (np.arange(len(act)) * DT_MIN) % 1440
    night = (minute < 360) | (minute >= 1320)
    hr = h["rest_bpm"] + h["per_unit"] * act - h["night_dip"] * night + rng.normal(0, h["noise_sd"], len(act))
    intensity = np.clip(np.round(act * acfg["intensity_per_unit"]), 0, 3)
    return hr, intensity, night & (act < 0.05)


def static_from_cgm(
    rng: np.random.Generator, g: np.ndarray, scfg: dict[str, Any]
) -> dict[str, Any]:
    """Static profile: assumed demographics, HbA1c from mean glucose (eAG), fasting glucose."""
    a1c = (float(np.mean(g)) - scfg["eag_intercept"]) / scfg["eag_slope"]
    minute = (np.arange(len(g)) * DT_MIN) % 1440
    fasting = float(np.median(g[(minute >= 240) & (minute < 360)]))
    age = float(np.clip(rng.normal(scfg["age"]["mean"], scfg["age"]["sd"]), scfg["age"]["lo"], scfg["age"]["hi"]))
    bmi = float(np.clip(rng.normal(scfg["bmi"]["mean"], scfg["bmi"]["sd"]), scfg["bmi"]["lo"], scfg["bmi"]["hi"]))
    bins, labels = scfg["hba1c_bins"], scfg["hba1c_labels"]
    status = next(lab for lo, hi, lab in zip(bins[:-1], bins[1:], labels, strict=True) if lo <= a1c < hi)
    return {"age_years": age, "sex": "M" if rng.random() < scfg["male_fraction"] else "F",
            "bmi": bmi, "hba1c_pct": a1c, "fasting_glucose_mgdl": fasting, "diabetes_status": status}  # fmt: skip


def generate(
    icfg: dict[str, Any], mc: dict[str, Any], strength: float, tag: str, seed: int,
    n_patients: int | None = None, days: int | None = None,
    fits: pd.DataFrame | None = None, noise: dict[str, float] | None = None,
    alpha: float | None = None,
) -> dict[str, pd.DataFrame]:  # fmt: skip
    """Simulate one cohort: dict with ``grid``, ``meals``, ``static``, ``params`` (all SYNTHETIC)."""
    fits = load_fits(icfg) if fits is None else fits
    if noise is None or alpha is None:
        noise, alpha = load_noise_and_alpha(icfg)
    n_pat, n_days = n_patients or icfg["n_patients"], days or icfg["days"]
    rng = np.random.default_rng(seed)
    lg = np.log(fits[list(FIT_PARAMS)].to_numpy(dtype=float))
    mu, cov = shifted_mean(lg.mean(0), icfg["shifts"], strength), np.cov(lg, rowvar=False)
    lo, hi = log_bounds(mc["params"])
    lib = load_library()
    sc, w_p = mc["solver"], mc["params"]["slow_protein_weight"]
    t0 = pd.Timestamp(icfg["start_datetime"])
    grids, meal_rows, static_rows, prm = [], [], [], []
    for i in range(n_pat):
        pid = f"{icfg['name']}_{tag}_{i:03d}"
        lp = np.clip(rng.multivariate_normal(mu, cov), lo, hi)
        pf = float(rng.lognormal(0, icfg["meals"]["portion_sd_patient"]))
        day_meals = [meal_day(rng, lib, icfg["meals"], pf) for _ in range(n_days)]
        carbs, slow, meals = meal_inputs(day_meals, w_p)
        act = activity_days(rng, n_days, icfg["activity"])
        th = theta_from_log(lp, mc["params"]["fixed"], alpha)
        x = simulate(th, initial_state(float(np.exp(lp[0])), th), carbs, slow, act,
                     sc["dt_min"], sc["n_sub"], bounds_array(sc))  # fmt: skip
        n = len(x)
        g = np.clip(x[:, 2] + ar1_noise(n, noise["white_sd"], noise["ar_sd"], noise["phi"], rng), *sc["g_bounds"])
        hr, inten, rest = wearables(rng, act, icfg["activity"])
        t = t0 + pd.to_timedelta(np.arange(n) * DT_MIN, unit="m")
        grids.append(pd.DataFrame({
            "patient_id": pid, "period_id": pid, "dataset": icfg["name"], "t": t,
            "days_since_start": np.arange(n) * DT_MIN / 1440.0, "glucose": g, "glucose_true": x[:, 2],
            "glucose_is_real": True, "glucose_is_carried": False, "glucose_missing": False,
            "minutes_since_real": 0.0, "hr": hr, "hr_missing": False, "activity_intensity": inten,
            "activity_missing": False, "rest_proxy": rest, "synthetic": True,
        }))  # fmt: skip
        for m in meals:
            meal_rows.append({
                "patient_id": pid, "period_id": pid, "t": t[m["abs_step"]], "meal_type": m["slot"],
                "carbs_g": m["carbs_g"], "protein_g": m["protein_g"], "fat_g": m["fat_g"],
                "fiber_g": m["fiber_g"], "kcal": m["kcal"], "amount_gt100": False,
                "amount_missing": False, "macros_available": True, "food_grams_total": np.nan,
                "preset": m["key"], "synthetic": True,
            })  # fmt: skip
        static_rows.append({"patient_id": pid, "dataset": icfg["name"], "synthetic": True,
                            **static_from_cgm(rng, g, icfg["static"])})  # fmt: skip
        prm.append({"patient_id": pid, "synthetic": True, "strength": strength,
                    **dict(zip(FIT_PARAMS, np.exp(lp), strict=True))})  # fmt: skip
    return {"grid": pd.concat(grids, ignore_index=True), "meals": pd.DataFrame(meal_rows),
            "static": pd.DataFrame(static_rows), "params": pd.DataFrame(prm)}  # fmt: skip


def summary(c: dict[str, pd.DataFrame], icfg: dict[str, Any]) -> dict[str, Any]:
    """Per-cohort realism summary: CGM statistics, diet energy shares, post-meal rise."""
    g = c["grid"]
    v = g["glucose"]
    pp = g.groupby("patient_id")["glucose"]
    m = c["meals"]
    kcal_carb, kcal = 4 * m["carbs_g"].sum(), (4 * m["carbs_g"] + 4 * m["protein_g"] + 9 * m["fat_g"]).sum()
    d = {"n_patients": int(g["patient_id"].nunique()), "n_days": float(g["days_since_start"].max()),
         "mean_glucose": float(v.mean()), "patient_mean_quartiles": pp.mean().quantile([0.25, 0.5, 0.75]).tolist(),
         "cv_pct_median": float((100 * pp.std() / pp.mean()).median()),
         "pct_gt180": float(100 * (v > 180).mean()), "pct_lt70": float(100 * (v < 70).mean()),
         "pct_70_180": float(100 * ((v >= 70) & (v <= 180)).mean()),
         "carb_energy_share": float(kcal_carb / kcal), "carb_energy_share_reference": icfg["meals"]["carb_energy_share_reference"],
         "mean_meal_carbs_g": float(m["carbs_g"].mean()),
         "hba1c_quartiles": c["static"]["hba1c_pct"].quantile([0.25, 0.5, 0.75]).tolist(),
         "status_counts": c["static"]["diabetes_status"].value_counts().to_dict()}  # fmt: skip
    return d
