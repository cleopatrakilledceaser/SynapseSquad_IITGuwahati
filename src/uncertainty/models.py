"""Point / quantile models for Part 6 (D-045): M_pop (A0 LightGBM) and M_ft (k-day fine-tune).

Reuses the Part 5 population code (``src.personalize.data``), so the CV+ arm's model is exactly
the Part 5 ``lgbm_s`` base. Every fit sees only the patients passed in.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.hybrid.twin_features import HORIZONS
from src.models.experiment import inner_split
from src.personalize import methods as pm
from src.personalize.data import PopH, fit_fixed, fit_pop, inner_folds


def fit_population(
    tab: pd.DataFrame, pids: list[str], cols: list[str], strata: dict, bc: dict, threads: int
) -> tuple[dict[int, PopH], list[str], list[str]]:
    """Part 5 population model on ``pids`` (inner split seed 42 for leaves / early stopping)."""
    in_tr, in_va = inner_split(pids, strata, bc, 42)
    rows = tab[tab.patient_id.isin(pids)]
    tr, va = rows[rows.patient_id.isin(in_tr)], rows[rows.patient_id.isin(in_va)]
    pop, _ = fit_pop(tr, va, cols, False, bc, threads)
    return pop, in_tr, in_va


def inner_models(
    tab: pd.DataFrame,
    pids: list[str],
    pop: dict[int, PopH],
    n_folds: int,
    seed: int,
    bc: dict,
    threads: int,
) -> tuple[dict[str, int], dict[int, dict[int, PopH]]]:
    """Inner patient folds and fixed-hyperparameter refits without each fold (Part 5 rule)."""
    fmap = inner_folds(pids, n_folds, seed)
    rows = tab[tab.patient_id.isin(pids)]
    mods = {}
    for j in sorted(set(fmap.values())):
        out = [p for p, f in fmap.items() if f == j]
        mods[j] = fit_fixed(rows[~rows.patient_id.isin(out)], pop, bc, threads)
    return fmap, mods


def oof_predict(df: pd.DataFrame, fmap: dict[str, int], mods: dict, h: int) -> np.ndarray:
    """Out-of-fold forecast of every row (from its patient's inner-fold model)."""
    out = np.full(len(df), np.nan)
    pid = df["patient_id"].to_numpy()
    for j, m in mods.items():
        sel = np.isin(pid, [p for p, f in fmap.items() if f == j])
        if sel.any():
            out[sel] = m[h].predict(df[sel])
    return out


def select_ft(df: pd.DataFrame, h: int, preds: dict, test_from: float) -> tuple[float, int]:
    """(lr, trees) with the lowest patient-weighted MAE on rows day >= test_from (Part 5 rule)."""
    y, pid = df[f"y_{h}"].to_numpy(float), df["patient_id"].to_numpy()
    ev = (df["days_since_start"].to_numpy(float) >= test_from) & ~np.isnan(y)
    best, key = np.inf, None
    for k_, p in preds.items():
        ok = ev & ~np.isnan(p)
        s = float(pd.Series(np.abs(y[ok] - p[ok])).groupby(pid[ok]).mean().mean())
        if s < best:
            best, key = s, k_
    return float(key[0]), int(key[1])


def finetune_frame(
    df: pd.DataFrame,
    h: int,
    k: float,
    model_of: dict[str, PopH],
    pc: dict,
    bc: dict,
    grid: tuple[float, int] | None = None,
) -> tuple[dict[tuple[float, int], np.ndarray], dict[str, dict[float, Any]]]:
    """Fine-tuned forecasts (rows day >= k) of every patient for each (lr, trees) grid point.

    ``model_of[p]`` is a population model that never saw patient p. Adaptation pairs = target
    time before day k (Part 5 ``adapt_mask``). Rows before day k and patients with fewer than
    min_pairs pairs keep the population forecast. Also returns the boosters (for re-use on
    faulted data, whose adaptation days are untouched). ``grid`` = one chosen (lr, trees).
    """
    ft = pc["finetune"]
    lrs, trees = ft["lr_grid"], ft["trees_grid"]
    if grid is not None:
        lrs, trees = [grid[0]], [grid[1]]
    out = {(lr, t): np.full(len(df), np.nan) for lr in lrs for t in trees}
    boosters: dict[str, dict[float, Any]] = {}
    y = df[f"y_{h}"].to_numpy(float)
    days = df["days_since_start"].to_numpy(float)
    for p, rows in df.groupby("patient_id").indices.items():
        mp = model_of[p]
        d = df.iloc[rows]
        popf = mp.predict(d)
        for key in out:
            out[key][rows] = popf
        a = pm.adapt_mask(days[rows], h, k, y[rows])
        e = days[rows] >= k
        boosters[p] = {}
        if a.sum() < pc["ridge"]["min_pairs"] or not e.any():
            continue
        base = d[mp.base].to_numpy(float)
        for lr in lrs:
            prm = pm.finetune_params(bc["lightgbm"]["params"], ft, lr)
            b = pm.finetune(
                mp.model_str(), mp.n_iter, d.loc[a, mp.cols], y[rows][a] - base[a], prm, max(trees)
            )
            boosters[p][lr] = b
            for t in trees:
                extra = b.predict(d.loc[e, mp.cols], start_iteration=mp.n_iter, num_iteration=t)
                out[(lr, t)][rows[e]] = popf[e] + extra
    return out, boosters


def ft_apply(
    df: pd.DataFrame, mp: PopH, booster: dict[float, Any], lr: float, trees: int, k: float
) -> np.ndarray:
    """Re-apply a fitted fine-tune booster of one patient to (possibly faulted) rows."""
    popf = mp.predict(df)
    if lr not in booster:
        return popf
    e = df["days_since_start"].to_numpy(float) >= k
    out = popf.copy()
    if e.any():
        out[e] += booster[lr].predict(
            df.loc[e, mp.cols], start_iteration=mp.n_iter, num_iteration=trees
        )
    return out


def fit_quantiles(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    pop: dict[int, PopH],
    qs: list[float],
    bc: dict,
    threads: int,
) -> dict[tuple[int, float], tuple[Any, int]]:
    """LightGBM quantile models (delta target, point model's leaves, early stopping on va)."""
    out = {}
    for h in HORIZONS:
        p = pop[h]
        yt = tr[f"y_{h}"].to_numpy(float) - tr[p.base].to_numpy(float)
        yv = va[f"y_{h}"].to_numpy(float) - va[p.base].to_numpy(float)
        ot, ov = ~np.isnan(yt), ~np.isnan(yv)
        for q in qs:
            prm = {**bc["lightgbm"]["params"], "n_jobs": threads, "random_state": 42}
            prm |= {"num_leaves": p.leaves, "objective": "quantile", "alpha": q}
            m = lgb.LGBMRegressor(**prm).fit(
                tr.loc[ot, p.cols],
                yt[ot],
                eval_set=[(va.loc[ov, p.cols], yv[ov])],
                eval_metric="quantile",
                callbacks=[
                    lgb.early_stopping(bc["lightgbm"]["early_stopping_rounds"], verbose=False)
                ],
            )
            out[(h, q)] = (m, int(m.best_iteration_ or prm["n_estimators"]))
    return out


def predict_quantile(df: pd.DataFrame, qm: tuple[Any, int], p: PopH) -> np.ndarray:
    """Absolute-glucose quantile forecast (base + booster)."""
    m, n = qm
    return df[p.base].to_numpy(float) + m.predict(df[p.cols], num_iteration=n)
