"""Event probabilities (calibrated on TRAINING out-of-fold rows) and alarm burden (D-045/D-046).

Per (design, cohort, fold): residual distribution, Platt / isotonic maps and alert thresholds
are fitted on that fold's training patients' OOF rows (``*_oof.parquet``), then applied to the
fold's test rows. Nothing here is tuned on test rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.fusion_data import load_grid
from src.uncertainty.alarms import alert_stats, episodes
from src.uncertainty.events import Isotonic, Platt, exceed_prob

EV = {"hyper180": (180.0, True), "hyper140": (140.0, True), "hypo70": (70.0, False)}


def onset_mask(d: pd.DataFrame, ev: str, h: int) -> np.ndarray:
    """Onset-only windows: glucose at t not beyond the threshold, label available."""
    thr, above = EV[ev]
    now = d["cgm_now"].to_numpy(float)
    lab = d[f"{ev}_any_{h}"].to_numpy(float)
    return ~np.isnan(lab) & ((now <= thr) if above else (now >= thr))


def oof_residuals(oof: pd.DataFrame, col: str, h: int) -> np.ndarray:
    """Sorted training OOF residuals y - yhat in the evaluation region (day >= 7)."""
    r = oof[f"y|{h}"].to_numpy(float) - oof[col].to_numpy(float)
    reg = oof["days_since_start"].to_numpy(float) >= 7
    return np.sort(r[reg & ~np.isnan(r)])


def event_probs(
    oof: pd.DataFrame, te: pd.DataFrame, ev: str, h: int, col_oof: str, col_te: str
) -> dict[str, np.ndarray] | None:
    """Raw / Platt / isotonic probabilities on test onset rows (maps fitted on OOF onset rows)."""
    thr, above = EV[ev]
    res = oof_residuals(oof, col_oof, h)
    mo = onset_mask(oof, ev, h) & ~np.isnan(oof[col_oof].to_numpy(float))
    mt = onset_mask(te, ev, h)
    yo = oof.loc[mo, f"{ev}_any_{h}"].to_numpy(int)
    if yo.sum() < 5 or yo.sum() == len(yo) or len(res) == 0:
        return None
    so = exceed_prob(oof.loc[mo, col_oof].to_numpy(float), res, thr, above)
    st = exceed_prob(te.loc[mt, col_te].to_numpy(float), res, thr, above)
    return {
        "mask": mt,
        "y": te.loc[mt, f"{ev}_any_{h}"].to_numpy(int),
        "raw": st,
        "platt": Platt().fit(so, yo).predict(st),
        "isotonic": Isotonic().fit(so, yo).predict(st),
    }


def grid_episodes(cohort: str, ev: str, ac: dict, step: int) -> dict[str, Any]:
    """Episode onsets (minutes since period start) and in-episode flags per period."""
    thr, above = EV[ev]
    n = int(np.ceil(ac["episode_min_minutes"] / step)) + 1  # readings spanning >= 15 min
    out, inside = {}, {}
    for pid, g in load_grid(cohort).groupby("period_id"):
        g = g.sort_values("t")
        on, ins = episodes(
            g["glucose"].to_numpy(float), g["glucose_is_real"].to_numpy(bool), thr, above, n, n
        )
        mins = g["minutes_since_start"].to_numpy(float).round(3)
        out[pid] = mins[on]
        inside[pid] = pd.Series(ins, index=mins)
    return {"onsets": out, "inside": inside}


def at_risk(d: pd.DataFrame, epi: dict) -> np.ndarray:
    """Anchors not inside an episode of this type."""
    ok = np.ones(len(d), bool)
    mins = (d["days_since_start"].to_numpy(float) * 1440.0).round(3)
    for pid, rows in d.groupby("period_id").indices.items():
        s = epi["inside"].get(pid)
        if s is not None:
            ok[rows] = ~s.reindex(mins[rows], method="nearest").to_numpy(bool)
    return ok


def alert_label(d: pd.DataFrame, epi: dict, horizon: float) -> np.ndarray:
    """1 if an episode starts in (t, t + horizon] (the alert truth, D-046)."""
    y = np.zeros(len(d), int)
    mins = d["days_since_start"].to_numpy(float) * 1440.0
    for pid, rows in d.groupby("period_id").indices.items():
        on = epi["onsets"].get(pid, np.array([]))
        y[rows] = [int(np.any((on > x) & (on <= x + horizon))) for x in mins[rows]]
    return y


def scores(d: pd.DataFrame, pred_col: str, res: np.ndarray, ev: str, ac: dict) -> dict:
    """Raw scores of the three alert systems (larger = more alarming)."""
    thr, above = EV[ev]
    now = d["cgm_now"].to_numpy(float)
    slope = d[ac["trend_slope_feature"]].fillna(0.0).to_numpy(float)
    trend = now + ac["horizon_min"] * slope
    sign = 1.0 if above else -1.0
    return {
        "model": exceed_prob(d[pred_col].to_numpy(float), res, thr, above),
        "naive": sign * now,
        "trend": sign * trend,
    }


def burden(d: pd.DataFrame, score: np.ndarray, theta: float, epi: dict, ac: dict) -> dict:
    """Pooled alarm counts of one system / threshold over the at-risk anchors in ``d``."""
    keys = ("n_alerts", "n_true", "n_false", "episodes_eligible", "episodes_detected")
    tot: dict[str, Any] = {k: 0 for k in keys} | {"leads": [], "monitor_days": 0.0}
    mins = d["days_since_start"].to_numpy(float) * 1440.0
    pid_of = d["patient_id"].to_numpy()
    per = {}
    for per_id, rows in d.groupby("period_id").indices.items():
        o = rows[np.argsort(mins[rows])]
        on = epi["onsets"].get(per_id, np.array([]))
        on = on[(on > mins[o].min()) & (on <= mins[o].max() + ac["horizon_min"])]
        st = alert_stats(mins[o], score[o], on, theta, ac["horizon_min"], ac["refractory_min"])
        st["monitor_days"] = len(o) * 15.0 / 1440.0
        per[pid_of[o[0]]] = st
        for k in tot:
            tot[k] = tot[k] + st[k]
    return {"total": tot, "per_patient": per}


def summarize_burden(b: dict) -> dict[str, Any]:
    """Sensitivity, FA/day, PPV and lead-time summary from pooled counts."""
    t = b["total"]
    lead = np.asarray(t["leads"], float)

    def q(p: float) -> float:
        return float(np.quantile(lead, p)) if len(lead) else float("nan")

    return {
        "sensitivity": t["episodes_detected"] / max(t["episodes_eligible"], 1),
        "fa_per_day": t["n_false"] / max(t["monitor_days"], 1e-9),
        "ppv": t["n_true"] / t["n_alerts"] if t["n_alerts"] else float("nan"),
        "episodes_eligible": int(t["episodes_eligible"]),
        "episodes_detected": int(t["episodes_detected"]),
        "n_alerts": int(t["n_alerts"]),
        "monitor_days": float(t["monitor_days"]),
        "lead_median_min": q(0.5),
        "lead_q25_min": q(0.25),
        "lead_q75_min": q(0.75),
    }


def threshold_grid(score: np.ndarray, n: int) -> np.ndarray:
    """Candidate thresholds: quantiles of the score (ascending)."""
    q = np.linspace(0.0, 1.0, n) ** 0.25  # dense near the top (low alarm rates)
    return np.unique(np.nanquantile(score, q))


def sweep(d: pd.DataFrame, score: np.ndarray, epi: dict, ac: dict, grid: np.ndarray) -> list:
    """(theta, summary) over a threshold grid (FROC-style curve)."""
    return [(float(th), summarize_burden(burden(d, score, float(th), epi, ac))) for th in grid]


def theta_at_rate(curve: list, rate: float) -> float:
    """Lowest threshold with FA/day <= rate on a sweep (higher sensitivity at that budget)."""
    ok = [th for th, s in curve if s["fa_per_day"] <= rate]
    return float(min(ok)) if ok else float(curve[-1][0])


def sens_at_rate(curve: list, rate: float) -> dict[str, Any]:
    """Best sensitivity among curve points with FA/day <= rate (test FROC point; no tuning)."""
    ok = [s for _, s in curve if s["fa_per_day"] <= rate]
    if not ok:
        return {"sensitivity": 0.0, "fa_per_day": float("nan"), "ppv": float("nan")}
    best = max(ok, key=lambda s: s["sensitivity"])
    return {k: best[k] for k in ("sensitivity", "fa_per_day", "ppv", "lead_median_min")}
