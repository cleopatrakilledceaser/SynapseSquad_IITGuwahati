"""Stage S2 (Part 5): Hybrid A0 / A / A-minus-activity / B on CGMacros (D-039).

Per outer fold (CV repeat 0) and twin level k in comparison_k:
  * training rows: TRAINING patients' anchors with day >= k, twin features from parameters fitted
    on their own first k days (src.hybrid.build); early stopping / leaf grid on the Part 3 inner
    patient split of the training patients (seed 42), never on test patients.
  * test rows: test patients' anchors with day >= test_from_day (the Part 4 windows), twin
    features from their first k days (population prior).
  A0 = CGM + time of day + meals + glycemic static (delta target), on EXACTLY the same rows as A,
  so A - A0 isolates the twin features. A = A0 + twin features. A_noact = A without the activity
  features. B = LightGBM on the residual y - twin forecast (base = twin forecast), added back.
Inner-validation MAE (training patients) is stored for choosing the best hybrid (D-040).
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.features.fusion_data import config_columns, load_table, resolve_groups
from src.features.make_splits import strata_for
from src.hybrid.build import OUT, PROC
from src.hybrid.twin_features import HORIZONS, add_derived
from src.models.experiment import inner_split
from src.models.fusion_lgbm import KEYS, folds
from src.models.lgbm import LGBMForecaster
from src.utils.config import load_config


def base_columns(cohort: str, hc: dict[str, Any], tab: pd.DataFrame) -> list[str]:
    """A0 feature columns of a cohort (configs/hybrid.yaml lgbm.groups)."""
    fc = load_config("fusion")
    g = hc["lgbm"]["groups"][cohort]
    return config_columns(g["dynamic"] + g["static"], resolve_groups(cohort, fc, list(tab.columns)))


def twin_table(cohort: str, fold: int, k: float, role: str, prior: str = "pop") -> pd.DataFrame:
    """Twin features of one fold / level / role (keys period_id, t)."""
    f = pd.read_parquet(OUT / f"twinfeat_{cohort}_fold{fold}.parquet")
    f = f[(f.k == k) & (f.role == role) & (f.prior == prior)]
    return f.drop(columns=["k", "role", "prior", "patient_id"])


def hybrid_rows(
    tab: pd.DataFrame,
    cohort: str,
    fold: int,
    k: float,
    pids: list[str],
    role: str,
    min_day: float,
    prior: str = "pop",
) -> pd.DataFrame:
    """Anchor rows of ``pids`` with day >= min_day joined with their twin features at level k."""
    d = tab[tab.patient_id.isin(pids) & (tab.days_since_start >= min_day)]
    d = d.merge(twin_table(cohort, fold, k, role, prior), on=["period_id", "t"], how="left")
    return add_derived(d)


def lgbm_cfg(bc: dict[str, Any], n_jobs: int) -> dict[str, Any]:
    """Part 2/3 LightGBM settings with the given thread count and no classifiers."""
    return {
        **bc["lightgbm"],
        "params": {**bc["lightgbm"]["params"], "n_jobs": n_jobs},
        "classifier_events": [],
    }


def fit_variant(
    tr: pd.DataFrame, va: pd.DataFrame, cols: list[str], residual: bool, bc: dict, n_jobs: int
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Per-horizon LightGBM; residual=True: target y - tw_mu_h (base = twin forecast)."""
    models, chosen = {}, {}
    for h in HORIZONS:
        base = f"tw_mu_{h}" if residual else "cgm_now"
        m = LGBMForecaster(cols, lgbm_cfg(bc, n_jobs), seed=42, target="delta", base=base)
        m.fit(tr, val=va, horizons=[h], events=False)
        models[h] = m
        p = m.predict(va, horizons=[h])[h]
        ok = va[f"y_{h}"].notna().to_numpy() & ~np.isnan(p)
        chosen[f"reg_{h}"] = m.chosen[f"reg_{h}"]
        chosen[f"inner_val_mae_{h}"] = float(np.mean(np.abs(p[ok] - va[f"y_{h}"].to_numpy()[ok])))
    return models, chosen


def variants(cohort: str, hc: dict[str, Any], tab: pd.DataFrame) -> dict[str, tuple]:
    """Variant name -> (feature columns, residual target?)."""
    a0 = base_columns(cohort, hc, tab)
    tw = list(hc["lgbm"]["twin_features"])
    act = set(hc["lgbm"]["activity_features"])
    return {
        "A0": (a0, False),
        "A": (a0 + tw, False),
        "A_noact": (a0 + [c for c in tw if c not in act], False),
        "B": (a0 + tw, True),
    }


def fold_job(cohort: str, fold: int, train: list[str], test: list[str], k: float) -> None:
    """All hybrid variants for one (fold, k) -> parquet + choices json."""
    path = OUT / f"hyb_{cohort}_fold{fold}_k{k:g}.parquet"
    if path.exists():
        return
    t0 = time.time()
    hc, fc, bc, sc = (load_config(n) for n in ("hybrid", "fusion", "baselines", "splits"))
    tab = load_table(cohort, fc["n_hours"], fc)
    strata = strata_for(pd.read_parquet(PROC / "static.parquet"), cohort, sc)
    in_tr, in_va = inner_split(train, strata, bc, 42 + hc["cv_repeat"])
    rows_tr = hybrid_rows(tab, cohort, fold, k, train, "train", k)
    tr, va = rows_tr[rows_tr.patient_id.isin(in_tr)], rows_tr[rows_tr.patient_id.isin(in_va)]
    ev = hybrid_rows(tab, cohort, fold, k, test, "test", hc["test_from_day"])
    clip = tuple(load_config("data")["cgm"]["glucose_range_mgdl"][cohort])
    out = ev[KEYS].copy()
    out["fold"], out["k"] = fold, k
    for h in HORIZONS:
        out[f"twin|{h}"] = np.clip(ev[f"tw_mu_{h}"].to_numpy(), *clip)
    ch: dict[str, Any] = {"fold": fold, "k": k, "n_train_rows": len(tr), "n_val_rows": len(va)}
    ch["train_min_day"] = float(rows_tr.days_since_start.min())
    for name, (cols, resid) in variants(cohort, hc, tab).items():
        models, chosen = fit_variant(tr, va, cols, resid, bc, 2)
        ch[name] = chosen
        for h, m in models.items():
            out[f"{name}|{h}"] = np.clip(m.predict(ev, horizons=[h])[h], *clip)
    ch["seconds"] = round(time.time() - t0, 1)
    (OUT / f"hyb_{cohort}_fold{fold}_k{k:g}.json").write_text(json.dumps(ch, indent=1))
    out.to_parquet(path, index=False)
    print(f"  {path.name}: {len(out)} rows, {ch['seconds']:.0f}s", flush=True)


def stage_lgbm(n_jobs: int = 4) -> None:
    """Hybrid A0 / A / A_noact / B for every fold x comparison k (CGMacros)."""
    hc = load_config("hybrid")
    jobs = [
        ("cgmacros", f, tr, te, k)
        for _, f, tr, te in folds("cgmacros", [hc["cv_repeat"]])
        for k in hc["comparison_k"]
    ]
    Parallel(n_jobs=n_jobs)(delayed(fold_job)(*j) for j in jobs)
