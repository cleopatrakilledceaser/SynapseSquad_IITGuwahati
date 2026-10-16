"""Episodes (consensus lv1) and alert counting (D-046)."""

from __future__ import annotations

from typing import Any

import numpy as np


def episodes(
    glu: np.ndarray, real: np.ndarray, thr: float, above: bool, n_start: int, n_end: int
) -> tuple[np.ndarray, np.ndarray]:
    """Episodes on one regular grid.

    Start: first index of >= n_start consecutive REAL readings beyond thr. End: confirmed after
    >= n_end consecutive real readings not beyond thr (missing readings break both runs).
    Returns (onset indices, in_episode flag per grid point).
    """
    beyond = (glu > thr) if above else (glu < thr)
    n = len(glu)
    inside = np.zeros(n, bool)
    onsets: list[int] = []
    state, run, start = False, 0, -1
    for i in range(n):
        if not real[i]:
            run, start = 0, -1
            inside[i] = state
            continue
        if not state:
            if beyond[i]:
                start = i if run == 0 else start
                run += 1
                if run >= n_start:
                    state, run = True, 0
                    onsets.append(start)
                    inside[start : i + 1] = True
            else:
                run, start = 0, -1
        else:
            inside[i] = True
            run = run + 1 if not beyond[i] else 0
            if run >= n_end:
                state, run = False, 0
    return np.asarray(onsets, int), inside


def alert_stats(
    t_anchor: np.ndarray,
    score: np.ndarray,
    onset_t: np.ndarray,
    theta: float,
    horizon: float,
    refractory: float,
) -> dict[str, Any]:
    """Alerts of one system for ONE patient at its at-risk anchors (sorted by time, minutes).

    Counted alert: score >= theta and >= refractory min after the last counted alert. TRUE if an
    onset falls in (t, t + horizon]. Episode eligible if an at-risk anchor lies in
    [onset - horizon, onset); detected if a counted alert does; lead = onset - first such alert.
    """
    alerts, last = [], -np.inf
    for t, s in zip(t_anchor, score, strict=True):
        if s >= theta and t - last >= refractory:
            alerts.append(t)
            last = t
    a = np.asarray(alerts, float)
    true = np.array([np.any((onset_t > x) & (onset_t <= x + horizon)) for x in a], bool)
    leads: list[float] = []
    detected = eligible = 0
    for o in onset_t:
        if not np.any((t_anchor >= o - horizon) & (t_anchor < o)):
            continue
        eligible += 1
        w = a[(a >= o - horizon) & (a < o)]
        if len(w):
            detected += 1
            leads.append(float(o - w.min()))
    return {
        "n_alerts": int(len(a)),
        "n_true": int(true.sum()),
        "n_false": int((~true).sum()) if len(a) else 0,
        "episodes_eligible": int(eligible),
        "episodes_detected": int(detected),
        "leads": leads,
    }
