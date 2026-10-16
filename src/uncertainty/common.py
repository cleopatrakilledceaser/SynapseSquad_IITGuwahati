"""Shared loading / bootstrap helpers for the Part 6 reports."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.fusion_data import load_table
from src.features.make_splits import strata_for
from src.utils.config import PROJECT_ROOT, load_config

PROC = PROJECT_ROOT / "data" / "processed"
UNC = PROC / "uncertainty"
KEYS = ["patient_id", "period_id", "t"]
HORIZONS = [30, 60, 120]
COHORTS = {"cgmacros": "CGMacros", "shanghai_t2dm": "Shanghai T2DM"}
METHODS = {
    "scp": "split conformal",
    "cqr": "CQR",
    "aci": "ACI",
    "cvp": "CV+",
}
LABEL_COLS = [f"{e}_any_{h}" for e in ("hyper180", "hyper140", "hypo70") for h in HORIZONS] + [
    "cgm_slope_30"
]


def groups(cohort: str) -> dict[str, str]:
    """Reporting group per patient: CGMacros diabetes status, Shanghai HbA1c group."""
    s = pd.read_parquet(PROC / "static.parquet")
    if cohort == "cgmacros":
        return s[s.dataset == cohort].set_index("patient_id")["diabetes_status"].to_dict()
    return strata_for(s, cohort, load_config("splits"))


def load(design: str, cohort: str, kind: str) -> pd.DataFrame:
    """Concatenate job outputs of one kind ('test' | 'oof' | 'rob') over folds (+ labels)."""
    files = sorted(UNC.glob(f"unc_{design}_{cohort}_fold*_{kind}.parquet"))
    if not files:
        raise FileNotFoundError(f"no {kind} outputs for {design} {cohort}")
    parts = []
    for f in files:
        d = pd.read_parquet(f)
        d["fold"] = int(f.stem.split("_fold")[1].split("_")[0])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    if kind in ("test", "oof"):
        fc = load_config("fusion")
        tab = load_table(cohort, fc["n_hours"], fc)[KEYS + LABEL_COLS]
        d = d.merge(tab, on=KEYS, how="left", validate="many_to_one")
    d["group"] = d["patient_id"].map(groups(cohort)).fillna("unknown")
    return d


def ci(v: np.ndarray, level: float) -> list[float]:
    """Percentile interval of bootstrap draws."""
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return [float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))]


def boot_ratio(
    num: pd.Series, den: pd.Series, n_boot: int, seed: int, level: float
) -> dict[str, Any]:
    """Pooled ratio sum(num) / sum(den) with a patient-bootstrap CI (index = patient)."""
    a, b = num.to_numpy(float), den.to_numpy(float)
    est = float(a.sum() / b.sum()) if b.sum() > 0 else float("nan")
    if len(a) == 0:
        return {"value": est, "ci": [float("nan")] * 2, "n_patients": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    draws = a[idx].sum(1) / np.maximum(b[idx].sum(1), 1e-12)
    return {"value": est, "ci": ci(draws, level), "n_patients": int(len(a))}


def boot_paired_diff(
    e1: pd.Series, e2: pd.Series, n: pd.Series, n_boot: int, seed: int, level: float
) -> dict[str, Any]:
    """Paired patient bootstrap of pooled mean(e1) - mean(e2) (per-patient sums, same rows)."""
    a, b, c = e1.to_numpy(float), e2.to_numpy(float), n.to_numpy(float)
    est = float((a.sum() - b.sum()) / c.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    d = (a[idx].sum(1) - b[idx].sum(1)) / c[idx].sum(1)
    lo, hi = ci(d, level)
    return {"diff": est, "ci": [lo, hi], "distinguishable_from_zero": bool(lo > 0 or hi < 0)}
