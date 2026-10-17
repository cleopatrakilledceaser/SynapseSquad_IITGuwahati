"""Replay tables of the demo patients: forecasts, CV+ bands, event risks, alerts, quality (D-058).

All forecasts come from fold models that never saw the patient. Calibration maps and alert
thresholds come from the fold's TRAINING patients only. Outcome columns (``y_*``, ``*_any_*``) of a
displayed patient are never read or stored.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.appdata import events as ev
from src.hybrid.twin_features import HORIZONS
from src.robustness.quality import dq_components
from src.uncertainty import alarm_eval as ae

KEYS = ["patient_id", "period_id", "t"]


def pick_demo(pids: list[str], status: dict[str, str], cfg: dict[str, Any]) -> list[str]:
    """First patient (sorted id) per diabetes status, then extra T2D patients (rule in app.yaml)."""
    ids = sorted(pids)
    out: list[str] = []
    for s in ("t2d", "prediabetes", "healthy"):
        out += [p for p in ids if status[p] == s][: cfg["per_status"]]
    out += [p for p in ids if status[p] == "t2d" and p not in out][: cfg["extra_t2d"]]
    return sorted(out)


def replay_rows(rows: pd.DataFrame, cfg: dict[str, Any], start_day: float) -> pd.DataFrame:
    """Anchors of the replay window: from the first anchor at/after the start day."""
    r = rows[rows["days_since_start"] >= start_day].sort_values("t")
    r = r.iloc[: int(cfg["replay_hours"] * 4)]
    if len(r) < 8:
        raise RuntimeError("replay window has fewer than 8 anchors")
    return r.reset_index(drop=True)


def forecast_block(f: Any, rows: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Point forecast and CV+ band per horizon (the Part 6 CV+ arm of the population model)."""
    out = {}
    for h in HORIZONS:
        out[f"pred|{h}"] = f.predict(rows, h, None)
        lo, hi = f.band(rows, h, alpha, None)
        out[f"lo80|{h}"], out[f"hi80|{h}"] = lo, hi
    return pd.DataFrame(out, index=rows.index)


class Calib:
    """Event maps and alert thresholds fitted on the fold's training patients (Part 6 rules)."""

    def __init__(self, oof: pd.DataFrame, ac: dict[str, Any], events: list[str], rate: float):
        """Fit maps on ``oof`` (training rows) and thresholds at ``rate`` false alarms per day."""
        self.ac, self.rate = ac, rate
        self.maps = {
            (e, h): ev.event_maps(oof, e, h, f"oof_pop|{h}") for e in ev.EV for h in HORIZONS
        }
        h = ac["horizon_min"]
        res = ae.oof_residuals(oof, f"oof_pop|{h}", h)
        self.res60 = res
        self.theta: dict[str, float | None] = {}
        for e in events:
            epi = ae.grid_episodes("cgmacros", e, ac, 5)
            self.theta[e] = ev.alert_theta(oof, e, res, epi, ac, rate)

    def probs(self, fc: pd.DataFrame, now: np.ndarray) -> pd.DataFrame:
        """Calibrated P(event within h) per event / horizon (NaN if already beyond)."""
        cols = {}
        for (e, h), (res, platt) in self.maps.items():
            cols[f"p_{e}|{h}"] = ev.probabilities(
                fc[f"pred|{h}"].to_numpy(float), now, e, res, platt
            )
        return pd.DataFrame(cols, index=fc.index)


def quality_at(
    g: pd.DataFrame, t: pd.Series, dq_cfg: dict, clip: tuple[float, float]
) -> pd.DataFrame:
    """Causal data-quality score and minutes since the last real reading at each anchor time."""
    q = dq_components(g.reset_index(drop=True), 5, dq_cfg, clip)
    pos = pd.DatetimeIndex(g["t"]).get_indexer(pd.DatetimeIndex(t))
    if (pos < 0).any():
        raise RuntimeError("anchor time not on the grid")
    out = q.iloc[pos][["dq", "dq_complete", "dq_recency", "dq_plaus", "dq_roc"]].reset_index(
        drop=True
    )
    out["stale_min"] = g["minutes_since_real"].to_numpy(float)[pos]
    out["grid_pos"] = pos
    return out


def alert_block(
    cal: Calib,
    g: pd.DataFrame,
    rows: pd.DataFrame,
    fc: pd.DataFrame,
    pos: np.ndarray,
    events: list[str],
) -> pd.DataFrame:
    """Alert score, causal at-risk flag, fired and active flags per alert event (+60 min score)."""
    ac, out = cal.ac, {}
    for e in events:
        thr, above = ae.EV[e]
        res = cal.res60
        score = ev.exceed_prob(fc["pred|60"].to_numpy(float), res, thr, above)
        risk = ev.causal_at_risk(g.reset_index(drop=True), e, pos, 5, ac["episode_min_minutes"])
        fired, active = ev.alert_track(rows["t"], score, risk, cal.theta[e], ac["refractory_min"])
        out[f"score_{e}"], out[f"atrisk_{e}"] = score, risk
        out[f"fired_{e}"], out[f"active_{e}"] = fired, active
    return pd.DataFrame(out, index=rows.index)
