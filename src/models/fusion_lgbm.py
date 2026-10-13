"""Part 3 LightGBM runs: one (cohort, config, target) over the Part 2 repeated patient CV.

Identical protocol to Part 2 (D-022): outer folds from data/processed/splits/<cohort>_cv.csv,
inner 80/20 stratified patient split of the outer-train patients (seed = 42 + repeat) for early
stopping and the leaf grid. Outer-validation patients are only predicted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.models.experiment import inner_split
from src.models.lgbm import LGBMForecaster
from src.utils.config import PROJECT_ROOT

PROC = PROJECT_ROOT / "data" / "processed"
KEYS = ["patient_id", "period_id", "t"]


def folds(cohort: str, repeats: list[int]) -> list[tuple[int, int, list[str], list[str]]]:
    """(repeat, fold, train_pids, val_pids) from the saved Part 2 CV assignment."""
    cv = pd.read_csv(PROC / "splits" / f"{cohort}_cv.csv")
    allp = set(cv.patient_id)
    out = []
    for (r, f), g in cv[cv.repeat.isin(repeats)].groupby(["repeat", "fold"]):
        out.append((int(r), int(f), sorted(allp - set(g.patient_id)), sorted(g.patient_id)))
    return out


def fit_fold(
    tab: pd.DataFrame,
    cols: list[str],
    fold: tuple[int, int, list[str], list[str]],
    strata: dict[str, str],
    bcfg: dict[str, Any],
    horizons: list[int],
    target: str,
    base: str,
    clip: tuple[float, float],
    events: list[str] | None,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, Any], Any]:
    """Fit on one outer fold; return predictions for the outer-validation patients."""
    rep, fno, tr_p, ev_p = fold
    seed = 42 + rep
    in_tr, in_va = inner_split(tr_p, strata, bcfg, seed)
    if tab[base].isna().any():
        # delta base missing (tiers: before the first fingerstick) -> training-patient mean
        tab = tab.copy()
        tab[base] = tab[base].fillna(float(tab.loc[tab.patient_id.isin(in_tr), base].mean()))
    lcfg = {
        **bcfg["lightgbm"],
        "params": {**bcfg["lightgbm"]["params"], "n_jobs": n_jobs},
        "classifier_events": events or [],
    }
    m = LGBMForecaster(cols, lcfg, seed=seed, target=target, base=base)
    m.fit(
        tab[tab.patient_id.isin(in_tr)],
        val=tab[tab.patient_id.isin(in_va)],
        horizons=horizons,
        events=bool(events),
    )
    ev = tab[tab.patient_id.isin(ev_p)]
    out = ev[KEYS].copy()
    out["repeat"], out["fold"] = rep, fno
    for h, p in m.predict(ev, horizons=horizons).items():
        out[f"pred_{h}"] = np.clip(p, *clip)
    for col, p in m.predict_events(ev).items():
        out[f"prob_{col}"] = p
    return out, m.chosen, m


def run_config(
    tab: pd.DataFrame,
    cols: list[str],
    cohort: str,
    repeats: list[int],
    strata: dict[str, str],
    bcfg: dict[str, Any],
    horizons: list[int],
    target: str,
    base: str,
    clip: tuple[float, float],
    events: list[str] | None = None,
    n_parallel: int = 4,
) -> pd.DataFrame:
    """Out-of-fold predictions for every (repeat, fold); folds run in parallel processes."""
    fl = folds(cohort, repeats)
    need = list(
        dict.fromkeys(
            KEYS
            + cols
            + [base]
            + [f"y_{h}" for h in horizons]
            + [f"{e}_{h}" for e in (events or []) for h in horizons]
        )
    )
    sub = tab[need]
    res = Parallel(n_jobs=n_parallel)(
        delayed(fit_fold)(
            sub,
            cols,
            f,
            strata,
            bcfg,
            horizons,
            target,
            base,
            clip,
            events,
            max(1, 8 // n_parallel),
        )
        for f in fl
    )
    return pd.concat([r[0] for r in res], ignore_index=True)
