"""The two interval arms of one (design, cohort, fold) job (D-045).

* split arm: models on PROPER training patients (fold training minus calibration patients);
  calibration patients only supply conformity scores; ACI gamma and fine-tune (lr, trees) are
  chosen on proper-training patients' out-of-fold rows.
* CV+ arm: models on all fold-training patients (= Part 5 base) + inner 3-fold patient OOF.
Frames are FIRST-period rows; ``region`` = day >= test_from_day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.uncertainty import conformal as cf

LVL = {0.2: "80", 0.05: "95"}


def _quantile_levels(alphas: list[float]) -> list[float]:
    return sorted({a / 2 for a in alphas} | {1 - a / 2 for a in alphas})


def tmin(df: pd.DataFrame) -> np.ndarray:
    """Anchor time in minutes since the period start."""
    return df["days_since_start"].to_numpy(float) * 1440.0


def aci_patients(
    df: pd.DataFrame,
    score: np.ndarray,
    pred: np.ndarray,
    h: int,
    ref: np.ndarray,
    alpha: float,
    gamma: float,
    start_day: float,
) -> tuple[np.ndarray, np.ndarray]:
    """ACI interval (lo, hi) for every row of ``df`` (rows before start_day: NaN)."""
    lo, hi = np.full(len(df), np.nan), np.full(len(df), np.nan)
    days, t = df["days_since_start"].to_numpy(float), tmin(df)
    for _, rows in df.groupby("patient_id").indices.items():
        r = rows[days[rows] >= start_day]
        r = r[np.argsort(t[r], kind="stable")]
        if len(r) == 0:
            continue
        q, _ = cf.aci_run(score[r], t[r], float(h), ref, alpha, gamma)
        lo[r], hi[r] = pred[r] - q, pred[r] + q
    return lo, hi


def choose_gamma(
    df: pd.DataFrame,
    score: np.ndarray,
    h: int,
    alpha: float,
    grid: list[float],
    start_day: float,
    test_from: float,
) -> tuple[float, dict[str, float]]:
    """Gamma minimizing mean |per-patient coverage - (1 - alpha)| on TRAINING patients' OOF rows.

    Reference scores for patient p = region scores of the OTHER training patients.
    """
    days, pid = df["days_since_start"].to_numpy(float), df["patient_id"].to_numpy()
    reg = (days >= test_from) & ~np.isnan(score)
    idx = df.groupby("patient_id").indices
    refs = {p: np.sort(score[reg & (pid != p)]) for p in idx}
    t = tmin(df)
    obj = {}
    for g in grid:
        dev = []
        for p, rows in idx.items():
            r = rows[days[rows] >= start_day]
            r = r[np.argsort(t[r], kind="stable")]
            if len(r) == 0 or not reg[r].any():
                continue
            q, _ = cf.aci_run(score[r], t[r], float(h), refs[p], alpha, g)
            m = reg[r]
            dev.append(abs(np.mean(score[r][m] <= q[m]) - (1 - alpha)))
        obj[str(g)] = float(np.mean(dev))
    best = min(obj, key=obj.get)
    return float(best), obj
