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
from src.personalize.data import PopEns, PopH, fit_fixed, fit_pop, fit_pop_tuned, inner_folds


def tuning_config() -> dict | None:
    """D-065 ``tuning`` section if ``point_model: tuned`` (D-072), else None (status quo)."""
    from src.utils.config import load_config

    if load_config("uncertainty").get("point_model", "status_quo") != "tuned":
        return None
    return load_config("improve")["tuning"]


def fit_population(
    tab: pd.DataFrame,
    pids: list[str],
    cols: list[str],
    strata: dict,
    bc: dict,
    threads: int,
    tcfg: dict | None = None,
) -> tuple[dict[int, Any], list[str], list[str]]:
    """Population model on ``pids`` (inner split seed 42 for selection / early stopping).

    ``tcfg`` None: Part 5 status quo (leaf grid). Otherwise the D-065 tuned search (D-072).
    """
    in_tr, in_va = inner_split(pids, strata, bc, 42)
    rows = tab[tab.patient_id.isin(pids)]
    tr, va = rows[rows.patient_id.isin(in_tr)], rows[rows.patient_id.isin(in_va)]
    if tcfg is None:
        pop, _ = fit_pop(tr, va, cols, False, bc, threads)
    else:
        pop, _ = fit_pop_tuned(tr, va, cols, bc, tcfg, threads)
    return pop, in_tr, in_va


def ft_fit(mp: Any, xa: pd.DataFrame, ya: np.ndarray, prm: dict, t_max: int) -> Any:
    """Fine-tune booster(s): one per member of a ``PopEns`` (each from its own trimmed booster)."""
    if isinstance(mp, PopEns):
        return [
            pm.finetune(m.model_str(), m.n_iter, xa[m.cols], ya, prm, t_max) for m in mp.members
        ]
    return pm.finetune(mp.model_str(), mp.n_iter, xa[mp.cols], ya, prm, t_max)


def ft_extra(mp: Any, b: Any, x: pd.DataFrame, trees: int) -> np.ndarray:
    """Contribution of the first ``trees`` fine-tune trees (mean over members of a ``PopEns``)."""
    if isinstance(mp, PopEns):
        return np.mean(
            [
                bi.predict(x[m.cols], start_iteration=m.n_iter, num_iteration=trees)
                for bi, m in zip(b, mp.members, strict=True)
            ],
            axis=0,
        )
    return b.predict(x[mp.cols], start_iteration=mp.n_iter, num_iteration=trees)


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
            b = ft_fit(mp, d.loc[a], y[rows][a] - base[a], prm, max(trees))
            boosters[p][lr] = b
            for t in trees:
                out[(lr, t)][rows[e]] = popf[e] + ft_extra(mp, b, d.loc[e], t)
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
        out[e] += ft_extra(mp, booster[lr], df.loc[e], trees)
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
