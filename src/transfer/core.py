"""Part 8a helpers (D-061): pure pandas / numpy, no LightGBM.

Feature sets shared by two cohorts, the common evaluation region, few-shot corrections of a
zero-shot forecast with a target patient's first k days, and MAE tables with patient-bootstrap
intervals. Few-shot corrections use only pairs whose TARGET time is before day k (causal).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.eval.metrics import bootstrap_point, paired_bootstrap_diff, point_stats
from src.personalize.methods import adapt_mask, ridge_fit, ridge_predict, shrink_weight

KEYS = ["patient_id", "period_id", "t"]


def feature_sets(
    cols_a: list[str], cols_b: list[str], cfg: dict[str, Any]
) -> tuple[dict[str, list[str]], list[str]]:
    """Feature sets built only from columns present in BOTH cohorts' tables.

    Returns (sets, dropped_cgm_columns). Static columns are the common static list; time-of-day
    columns come from the config. Raises if a required column is missing in either table.
    """
    shared = set(cols_a) & set(cols_b)
    cgm = sorted(c for c in shared if c.startswith("cgm_"))
    dropped = sorted({c for c in cols_a + cols_b if c.startswith("cgm_")} - set(cgm))
    tod = list(cfg["tod_columns"])
    static = [f"static_{s}" for s in cfg["common_static"]]
    for c in tod + static:
        if c not in shared:
            raise ValueError(f"column {c} is missing in one of the cohorts' tables")
    parts = {"cgm": cgm, "tod": tod, "static": static}
    return {
        n: [c for g in groups for c in parts[g]] for n, groups in cfg["feature_sets"].items()
    }, dropped


def first_period(tab: pd.DataFrame) -> pd.DataFrame:
    """Rows of each patient's first (earliest) period only."""
    start = tab.groupby("period_id")["t"].transform("min")
    first = tab.assign(_s=start).groupby("patient_id")["_s"].transform("min")
    return tab[start == first].copy()


def eval_table(tab: pd.DataFrame, test_from: float, min_n: int) -> pd.DataFrame:
    """First-period rows of patients with >= min_n evaluable anchors at day >= test_from."""
    fp = first_period(tab)
    ev = fp[(fp["days_since_start"] >= test_from) & fp["y_60"].notna()]
    n = ev.groupby("patient_id").size()
    keep = n[n >= min_n].index
    out = fp[fp["patient_id"].isin(keep)]
    return out.sort_values(["patient_id", "t"]).reset_index(drop=True)


def fewshot(
    tab: pd.DataFrame, pred_col: str, h: int, k: float, method: str, cfg: dict[str, Any]
) -> np.ndarray:
    """Zero-shot forecast corrected with each patient's first k days (``offset`` or ``ridge``)."""
    out = tab[pred_col].to_numpy(float).copy()
    y = tab[f"y_{h}"].to_numpy(float)
    res = y - out
    days = tab["days_since_start"].to_numpy(float)
    feats = cfg["ridge_features"]
    for _, idx in tab.groupby("patient_id").indices.items():
        m = adapt_mask(days[idx], h, k, y[idx])
        n = int(m.sum())
        if n < cfg["min_pairs"]:
            continue
        r = res[idx]
        if method == "offset":
            out[idx] += shrink_weight(n, cfg["offset_n0"]) * float(r[m].mean())
        elif method == "ridge":
            x = tab.iloc[idx][feats].to_numpy(float)
            with np.errstate(all="ignore"):
                mu = np.nanmean(np.where(m[:, None], x, np.nan), axis=0)
            mu = np.where(np.isnan(mu), 0.0, mu)
            x = np.where(np.isnan(x), mu, x)
            sd = x[m].std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            z = (x - mu) / sd
            out[idx] += ridge_predict(ridge_fit(z[m], r[m], cfg["ridge_lambda"]), z)
        else:
            raise ValueError(method)
    return out


def mae_rows(
    df: pd.DataFrame, models: list[str], horizons: list[int], boot: dict[str, Any]
) -> list[dict[str, Any]]:
    """MAE with patient-bootstrap CI per model and horizon (rows with a real y only)."""
    rows = []
    for h in horizons:
        for m in models:
            st = point_stats(df, f"y_{h}", f"pred_{m}_{h}")
            r = bootstrap_point(st, boot["n_boot"], 42, boot["ci"])
            rows.append({"model": m, "horizon_min": h, "mae": r["mae"], "mae_ci": r["mae_ci"],
                         "n_patients": r["n_patients"], "n_rows": r["n_rows"]})  # fmt: skip
    return rows


def diff_rows(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    horizons: list[int],
    boot: dict[str, Any],
) -> list[dict[str, Any]]:
    """Paired patient bootstrap of MAE(model 1) - MAE(model 2) for each pair and horizon."""
    rows = []
    for a, b in pairs:
        for h in horizons:
            s1 = point_stats(df, f"y_{h}", f"pred_{a}_{h}")
            s2 = point_stats(df, f"y_{h}", f"pred_{b}_{h}")
            r = paired_bootstrap_diff(s1, s2, boot["n_boot"], 42, boot["ci"])
            rows.append({"model_1": a, "model_2": b, "horizon_min": h, **r})
    return rows
