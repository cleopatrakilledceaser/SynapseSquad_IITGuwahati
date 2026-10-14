"""Population models and evaluation frames for personalization (Part 5.2, D-043).

Rules (tested in tests/test_personalize.py):
  * population models are fitted on TRAINING patients only (outer fold, CV repeat 0);
  * training patients get OUT-OF-FOLD population forecasts from an inner 3-fold patient split
    (fixed hyperparameters of the outer model) -> all adaptation choices use those, never a test
    patient;
  * adaptation / evaluation use each patient's FIRST period only; the common test region is
    day >= test_from_day (7) for every k; patients need >= min_test_anchors evaluable anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.hybrid.models import fit_variant
from src.hybrid.twin_features import HORIZONS


@dataclass
class PopH:
    """One horizon of a population model: forecast = base column + booster (first n_iter trees)."""

    est: Any
    base: str
    n_iter: int
    cols: list[str]
    leaves: int

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        """Booster output (delta to the base column)."""
        return self.est.predict(df[self.cols], num_iteration=self.n_iter)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Absolute glucose forecast (mg/dL)."""
        return df[self.base].to_numpy(float) + self.raw(df)

    def model_str(self) -> str:
        """Booster text trimmed to the used trees (init_model for fine-tuning)."""
        return self.est.booster_.model_to_string(num_iteration=self.n_iter)


def first_period(tab: pd.DataFrame) -> pd.DataFrame:
    """Rows of each patient's first (earliest) period only."""
    start = tab.groupby("period_id")["t"].transform("min")
    first = tab.assign(_s=start).groupby("patient_id")["_s"].transform("min")
    return tab[start == first].copy()


def eligible(fp: pd.DataFrame, test_from: float, min_n: int) -> list[str]:
    """Patients with >= min_n anchors (real y at +60 min) at day >= test_from (first period)."""
    ev = fp[(fp.days_since_start >= test_from) & fp["y_60"].notna()]
    n = ev.groupby("patient_id").size()
    return sorted(n[n >= min_n].index)


def fit_pop(
    tr: pd.DataFrame, va: pd.DataFrame, cols: list[str], residual: bool, bc: dict, n_jobs: int
) -> tuple[dict[int, PopH], dict[str, Any]]:
    """Outer population model (Part 3 protocol: leaf grid + early stopping on inner-val)."""
    models, chosen = fit_variant(tr, va, cols, residual, bc, n_jobs)
    out = {}
    for h, m in models.items():
        est = m.models[h]
        n_it = int(est.best_iteration_ or est.n_estimators)
        leaves = int(m.chosen[f"reg_{h}"]["num_leaves"])
        out[h] = PopH(est, m.base, n_it, cols, leaves)
    return out, chosen


def fit_fixed(
    tr: pd.DataFrame, pop: dict[int, PopH], bc: dict, n_jobs: int, seed: int = 42
) -> dict[int, PopH]:
    """Refit with the outer model's leaves / trees (no early stopping) on other patients' rows."""
    out = {}
    for h, p in pop.items():
        y = tr[f"y_{h}"].to_numpy(float) - tr[p.base].to_numpy(float)
        ok = ~np.isnan(y)
        params = {**bc["lightgbm"]["params"], "n_jobs": n_jobs, "random_state": seed}
        params |= {"num_leaves": p.leaves, "n_estimators": max(p.n_iter, 1)}
        est = lgb.LGBMRegressor(**params).fit(tr.loc[ok, p.cols], y[ok])
        out[h] = PopH(est, p.base, max(p.n_iter, 1), p.cols, p.leaves)
    return out


def inner_folds(pids: list[str], n: int, seed: int) -> dict[str, int]:
    """Deterministic patient -> inner fold (0..n-1) assignment."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(sorted(pids))
    return {p: i % n for i, p in enumerate(perm)}


def oof_forecasts(
    rows_fit: pd.DataFrame,
    rows_pred: pd.DataFrame,
    pop: dict[int, PopH],
    fmap: dict[str, int],
    bc: dict,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[int, dict[int, str]]]:
    """Out-of-fold forecasts for training patients (+ booster strings per inner fold / h).

    Inner fold j's model is fitted on ``rows_fit`` of patients NOT in fold j and predicts
    ``rows_pred`` of fold-j patients only.
    """
    parts, strings = [], {}
    for j in sorted(set(fmap.values())):
        inj = [p for p, f in fmap.items() if f == j]
        m = fit_fixed(rows_fit[~rows_fit.patient_id.isin(inj)], pop, bc, n_jobs)
        d = rows_pred[rows_pred.patient_id.isin(inj)].copy()
        for h in HORIZONS:
            d[f"pop_{h}"] = m[h].predict(d)
        parts.append(d)
        strings[j] = {h: m[h].model_str() for h in HORIZONS}
    return pd.concat(parts), strings
