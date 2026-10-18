"""Event probabilities, alert decisions, data quality and confidence for the app (D-058).

Everything reuses Part 6 code. Calibration maps (Platt) and alert thresholds are fitted on the
fold's TRAINING patients (out-of-fold forecasts) and applied to rows of patients the fold models
never saw. No outcome label of a displayed patient is read or stored.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.uncertainty import alarm_eval as ae
from src.uncertainty.alarms import episodes
from src.uncertainty.events import Platt, exceed_prob

EV = ae.EV


def event_maps(oof: pd.DataFrame, ev: str, h: int, col: str) -> tuple[np.ndarray, Platt | None]:
    """(sorted OOF residuals, Platt map) of one event / horizon from the fold's training rows."""
    thr, above = EV[ev]
    res = ae.oof_residuals(oof, col, h)
    mo = ae.onset_mask(oof, ev, h) & ~np.isnan(oof[col].to_numpy(float))
    yo = oof.loc[mo, f"{ev}_any_{h}"].to_numpy(int)
    if yo.sum() < 5 or yo.sum() == len(yo) or len(res) == 0:
        return res, None
    so = exceed_prob(oof.loc[mo, col].to_numpy(float), res, thr, above)
    return res, Platt().fit(so, yo)


def probabilities(
    pred: np.ndarray, now: np.ndarray, ev: str, res: np.ndarray, platt: Platt | None
) -> np.ndarray:
    """Calibrated P(event within h) for rows not already beyond the threshold, else NaN."""
    thr, above = EV[ev]
    out = np.full(len(pred), np.nan)
    if platt is None:
        return out
    onset = (now <= thr) if above else (now >= thr)
    ok = onset & ~np.isnan(pred)
    if ok.any():
        out[ok] = platt.predict(exceed_prob(pred[ok], res, thr, above))
    return out


def alert_theta(
    oof: pd.DataFrame, ev: str, res: np.ndarray, epi: dict, ac: dict, rate: float
) -> float | None:
    """Model-score alert threshold chosen on training OOF rows at ``rate`` false alarms per day."""
    h = ac["horizon_min"]
    o = oof[(oof.days_since_start >= 1)].reset_index(drop=True)
    o = o[~o[f"oof_pop|{h}"].isna()].reset_index(drop=True)
    o = o[ae.at_risk(o, epi)].reset_index(drop=True)
    if len(o) == 0:
        return None
    yo = ae.alert_label(o, epi, h)
    if yo.sum() < 5:
        return None
    so = ae.scores(o, f"oof_pop|{h}", res, ev, ac)["model"]
    cur = ae.sweep(o, so, epi, ac, ae.threshold_grid(so, 60))
    return ae.theta_at_rate(cur, rate)


def causal_at_risk(
    g: pd.DataFrame, ev: str, idx: np.ndarray, step: int, minutes: int
) -> np.ndarray:
    """Anchors (grid positions ``idx``) at which an alert of type ``ev`` may fire (causal).

    Uses only the readings up to the anchor: no episode in progress, glucose not beyond the
    threshold.

    Part 6 excluded anchors inside an episode with the episode flags of the whole record (which
    know about the confirmation 15 min later). The live display cannot, so this rule is causal and
    slightly different; it is a display rule and is described as such in the app.
    """
    thr, above = EV[ev]
    glu, real = g["glucose"].to_numpy(float), g["glucose_is_real"].to_numpy(bool)
    n = int(math.ceil(minutes / step)) + 1
    out = np.zeros(len(idx), bool)
    for k, i in enumerate(idx):
        _, ins = episodes(glu[: i + 1], real[: i + 1], thr, above, n, n)
        now = glu[i]
        beyond = bool(now > thr) if above else bool(now < thr)
        out[k] = not ins[-1] and not beyond and not np.isnan(now)
    return out


def alert_track(
    t: pd.Series, score: np.ndarray, at_risk: np.ndarray, theta: float | None, refractory_min: float
) -> tuple[np.ndarray, np.ndarray]:
    """(fired, active) per anchor. Fired = at risk, score >= theta, outside the refractory time.

    Active = an alert fired within the last ``refractory_min`` minutes (shown as ON in the app).
    """
    n = len(t)
    fired, active = np.zeros(n, bool), np.zeros(n, bool)
    if theta is None:
        return fired, active
    last = None
    for i, ti in enumerate(pd.DatetimeIndex(t)):
        quiet = last is None or (ti - last).total_seconds() / 60 >= refractory_min
        if at_risk[i] and quiet and score[i] >= theta:
            fired[i], last = True, ti
        active[i] = last is not None and (ti - last).total_seconds() / 60 < refractory_min
    return fired, active


def confidence(
    width60: np.ndarray,
    ref_width: float,
    dq: np.ndarray,
    stale_min: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """(label, width ratio). DESIGN CHOICE, not a validated rule (configs/app.yaml display).

    uncertain: interval wider than ``wide_ratio`` x the fold's median width, or data quality below
    ``dq_uncertain_below``, or the last sensor reading older than ``stale_min``.
    high: width at most the median and data quality in the top Part 6 bin. Otherwise moderate.
    """
    ratio = width60 / ref_width
    bad = (
        (ratio > cfg["wide_ratio"])
        | (dq < cfg["dq_uncertain_below"])
        | (stale_min > cfg["stale_min"])
    )
    good = (ratio <= 1.0) & (dq >= cfg["dq_high_from"])
    label = np.where(bad | np.isnan(ratio), "uncertain", np.where(good, "high", "moderate"))
    return label, ratio
