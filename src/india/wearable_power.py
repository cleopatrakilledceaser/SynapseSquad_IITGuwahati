"""Part 8d: can the model use a sleep / HRV / step channel? A detectability analysis (D-064).

ALL DATA ARE SYNTHETIC. For each effect multiplier m the same 100 simulated patients are generated
(same parameters, meals and sleep draws); only the strength of the sleep effect on insulin
sensitivity changes. Patient-level 5-fold CV with nested feature sets A (CGM, time of day, meals),
B (A + heart rate and activity) and C (B + sleep, HRV, steps). m = 0 is the no-effect control.

Run: ``python -W ignore -u -m src.india.wearable_power`` (about 20 to 40 minutes).
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import pandas as pd

from src.features.make_features import build_cohort
from src.india import cohort as ic
from src.india import stress
from src.transfer import core
from src.transfer.run import fit_model, predict_into
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.provenance import write_metrics

OUT = PROJECT_ROOT / "data" / "processed" / "india"
SLEEP_COLS = ["sleep_h_prev", "sleep_eff_prev", "sleep_deep_prev", "hrv_prev", "steps_1h"]


def add_sleep_features(tab: pd.DataFrame, c: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join causal sleep / HRV / step features to an anchor table (keys patient_id, t).

    A night's summary is used only from its wake time on; before the first wake time the
    features are missing. ``steps_1h`` is the sum over the trailing hour of 5-min step counts.
    """
    out = tab.copy()
    out["_i"] = np.arange(len(out))
    nights = c["sleep"].rename(
        columns={"sleep_h": "sleep_h_prev", "sleep_eff": "sleep_eff_prev",
                 "sleep_deep": "sleep_deep_prev", "hrv_rmssd": "hrv_prev"}  # fmt: skip
    )[["patient_id", "wake_t", *SLEEP_COLS[:4]]]
    left = out.sort_values("t")
    merged = pd.merge_asof(left, nights.sort_values("wake_t"), left_on="t", right_on="wake_t",
                           by="patient_id", direction="backward")  # fmt: skip
    g = c["grid"][["patient_id", "t", "steps"]].copy()
    g["steps_1h"] = g.groupby("patient_id")["steps"].transform(lambda s: s.rolling(12, min_periods=1).sum())
    merged = merged.merge(g[["patient_id", "t", "steps_1h"]], on=["patient_id", "t"], how="left")
    merged = merged.sort_values("_i").drop(columns=["wake_t"])
    return merged.reset_index(drop=True)


def feature_sets(cols: list[str]) -> dict[str, list[str]]:
    """Nested feature sets A (CGM, tod, meals), B (+ wearable), C (+ sleep / HRV / steps)."""
    cgm = [x for x in cols if x.startswith("cgm_")]
    tod = ["tod_sin", "tod_cos"]
    meals = [x for x in cols if x.startswith("meal_")] + ["act_post_meal"]
    wear = [x for x in cols if x.startswith(("hr_", "act_")) and x != "act_post_meal"]
    wear += ["rest_proxy_prev_night_frac"]
    a = [x for x in cgm + tod + meals if x in cols]
    b = a + [x for x in wear if x in cols]
    return {"A": a, "B": b, "C": b + [x for x in SLEEP_COLS if x in cols]}


def folds_of(pids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    """Seeded random patient folds."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(sorted(pids))
    return [list(map(str, f)) for f in np.array_split(order, n_folds)]


def run_condition(
    mult: float, icfg: dict[str, Any], mc: dict[str, Any], fcfg: dict[str, Any], bc: dict[str, Any],
    pre: dict[str, Any],
) -> dict[str, Any]:  # fmt: skip
    """Generate one condition, run the nested-feature-set CV, return summary metrics."""
    sc = icfg["sleep"]
    scfg = {**icfg["stress"], "n_hours": sc["n_hours"]}
    c = ic.generate(icfg, mc, icfg["main_strength_value"], "pwr", sc["seed"], sc["n_patients"],
                    sc["days"], pre["fits"], pre["noise"], pre["alpha"], sleep_mult=mult)  # fmt: skip
    f2 = copy.deepcopy(fcfg)
    f2["cohorts"][stress.SYN] = {**fcfg["cohorts"]["cgmacros_15"], "dataset": icfg["name"]}
    f2["static"][stress.SYN] = list(scfg["static"])
    grid = ic.to_grid(c, 15)
    tabs, _ = build_cohort(stress.SYN, f2, grid, c["meals"], c["static"])
    tab = add_sleep_features(tabs[sc["n_hours"]], c)
    sets = feature_sets(list(tab.columns))
    hz = sc["horizons_min"]
    status = c["static"].set_index("patient_id")["diabetes_status"]
    strata = {p: f"{stress.SYN}|{s}" for p, s in status.items()}
    clip = tuple(fcfg["glucose_clip_mgdl"])
    ev = core.eval_table(tab, sc["test_from_day"], 1)
    ev = ev[ev["days_since_start"] >= sc["test_from_day"]].copy()
    for h in hz:
        ev[f"pred_persistence_{h}"] = ev["cgm_now"]
        for s in sets:
            ev[f"pred_{s}_{h}"] = np.nan
    all_p = sorted(tab["patient_id"].unique())
    for k, test_p in enumerate(folds_of(all_p, sc["n_folds"], sc["seed"])):
        train_p = [p for p in all_p if p not in set(test_p)]
        rows = ev.index[ev["patient_id"].isin(set(test_p))]
        for s, cols in sets.items():
            m = fit_model(tab, train_p, strata, cols, bc, hz, sc["seed"] + k)
            predict_into(ev, rows, m, s, hz, clip)
        print(f"    m={mult:g} fold {k} done", flush=True)
    models = ["persistence", *sets]
    pairs = [("B", "A"), ("C", "B"), ("C", "A")]
    boot = bc["bootstrap"]
    ev.to_parquet(OUT / f"sleep_power_m{mult:g}.parquet", index=False)
    return {
        "n_patients": int(ev["patient_id"].nunique()),
        "mean_glucose": float(c["grid"]["glucose_true"].mean()),
        "pct_gt180": float(100 * (c["grid"]["glucose_true"] > 180).mean()),
        "mean_si_log_shift": float(c["sleep"]["si_log_shift"].mean()),
        "n_features": {k: len(v) for k, v in sets.items()},
        "mae": core.mae_rows(ev, models, hz, boot),
        "diff": core.diff_rows(ev, pairs, hz, boot),
    }  # fmt: skip


def main() -> None:
    """Run all effect multipliers and write reports/metrics/india_sleep_power.json."""
    icfg, mc = load_config("india"), load_config("mech")
    fcfg, bc = load_config("features"), load_config("baselines")
    icfg = {**icfg, "main_strength_value": icfg["strengths"][icfg["main_strength"]]}
    fits = ic.load_fits(icfg)
    noise, alpha = ic.load_noise_and_alpha(icfg)
    pre = {"fits": fits, "noise": noise, "alpha": alpha}
    OUT.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {"SYNTHETIC": True, "multipliers": icfg["sleep"]["multipliers"]}
    for mult in icfg["sleep"]["multipliers"]:
        t0 = time.time()
        metrics[f"m{mult:g}"] = run_condition(mult, icfg, mc, fcfg, bc, pre)
        print(f"  multiplier {mult:g}: {time.time() - t0:.0f}s", flush=True)
    write_metrics("india_sleep_power", metrics, icfg, icfg["sleep"]["seed"], "synthetic")
    print("wrote reports/metrics/india_sleep_power.json")


if __name__ == "__main__":
    main()
