"""Fit every baseline on one (outer-train, outer-eval) patient split and return predictions.

Inside the outer-train patients, a patient-level inner split (configs/baselines.yaml inner_split)
provides the validation set for ALL tuning (trend window, Holt parameters, AR order, LightGBM
early stopping and leaf count). Outer-eval patients are only predicted, never used for a choice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.splits import patient_train_cal_test_split
from src.models.baselines import HoltES, LinearTrend, PatientAR, Persistence, PooledAR
from src.models.lgbm import LGBMForecaster

KEYS = ["patient_id", "period_id", "t"]


def inner_split(
    train_pids: list[str], strata: dict[str, str], bcfg: dict[str, Any], seed: int
) -> tuple[list[str], list[str]]:
    """Patient-level inner train / validation split of the outer-train patients."""
    sp = patient_train_cal_test_split(
        train_pids, bcfg["inner_split"], seed, {p: strata[p] for p in train_pids}
    )
    return sorted(sp.train), sorted(sp.calibration)


def event_score_from_value(value: np.ndarray, event: str, fcfg: dict[str, Any]) -> np.ndarray:
    """Monotone [0,1] ranking score from a glucose forecast (for AUROC/AUPRC only, not Brier)."""
    name = event.split("_")[0]
    spec = fcfg["events"][name]
    z = (value - spec["threshold"]) / 10.0
    z = z if spec["direction"] == "above" else -z
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_predict(
    tab: pd.DataFrame,
    grid: pd.DataFrame,
    train_pids: list[str],
    eval_pids: list[str],
    strata: dict[str, str],
    feature_sets: dict[str, list[str]],
    step: int,
    clip: tuple[float, float],
    bcfg: dict[str, Any],
    fcfg: dict[str, Any],
    seed: int,
    with_events: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit all baselines on ``train_pids`` and predict ``eval_pids``. Returns (preds, choices)."""
    horizons = fcfg["windows"]["horizons_min"]
    in_tr, in_va = inner_split(train_pids, strata, bcfg, seed)
    tr = tab[tab.patient_id.isin(in_tr)]
    va = tab[tab.patient_id.isin(in_va)]
    full_tr = tab[tab.patient_id.isin(train_pids)]
    ev = tab[tab.patient_id.isin(eval_pids)].reset_index(drop=True)
    h_cfg, ar_cfg, par_cfg = bcfg["holt"], bcfg["ar"], bcfg["patient_ar"]
    models: list[Any] = [
        Persistence(),
        LinearTrend(bcfg["linear_trend"]["window_min_grid"], bcfg["linear_trend"]["damping_grid"]),
        HoltES(
            h_cfg["history_min"],
            h_cfg["alpha_grid"],
            h_cfg["beta_grid"],
            h_cfg["phi_grid"],
            step=step,
        ),
        PooledAR(ar_cfg["order_min_grid"], step=step, max_rows=ar_cfg["max_fit_rows"], seed=seed),
        PatientAR(
            PooledAR(
                ar_cfg["order_min_grid"], step=step, max_rows=ar_cfg["max_fit_rows"], seed=seed
            ),
            par_cfg["ridge_lambda"],
            par_cfg["min_pairs"],
            step=step,
        ),
        LGBMForecaster(feature_sets["cgm_only"], bcfg["lightgbm"], name="lgbm_cgm", seed=seed),
        LGBMForecaster(feature_sets["cgm_tod"], bcfg["lightgbm"], name="lgbm_cgm_tod", seed=seed),
    ]
    out = ev[KEYS + [c for c in ev.columns if c.startswith(("y_", "hyper", "hypo"))]].copy()
    out["cgm_now"] = ev["cgm_now"].to_numpy()
    choices: dict[str, Any] = {}
    for m in models:
        # statistical baselines are tuned on inner-val and (for AR) refitted on inner-train only;
        # LightGBM early-stops on inner-val.
        is_lgbm = isinstance(m, LGBMForecaster)
        (
            m.fit(
                tr, grid, val=va, horizons=horizons, events=with_events and m.name == "lgbm_cgm_tod"
            )
            if is_lgbm
            else m.fit(
                full_tr if isinstance(m, Persistence) else tr, grid, val=va, horizons=horizons
            )
        )
        preds = m.predict(ev, grid, horizons)
        for h in horizons:
            out[f"pred_{m.name}_{h}"] = np.clip(preds[h], *clip)
        for attr in ("best_w", "best", "order", "chosen"):
            if hasattr(m, attr) and getattr(m, attr):
                choices[m.name] = {str(k): v for k, v in getattr(m, attr).items()}
        if is_lgbm and m.classifiers:
            for col, p in m.predict_events(ev).items():
                out[f"prob_{m.name}_{col}"] = p
    for h in horizons:
        for name in ("hyper180", "hyper140", "hypo70"):
            for d in ("any", "at"):
                out[f"score_persistence_{name}_{d}_{h}"] = event_score_from_value(
                    out["cgm_now"].to_numpy(), f"{name}_{d}_{h}", fcfg
                )
                out[f"score_lgbm_cgm_tod_reg_{name}_{d}_{h}"] = event_score_from_value(
                    out[f"pred_lgbm_cgm_tod_{h}"].to_numpy(), f"{name}_{d}_{h}", fcfg
                )
    return out, choices
