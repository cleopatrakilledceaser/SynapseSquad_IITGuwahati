"""Event probabilities, calibration, episodes and alert evaluation (D-045 / D-046)."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def exceed_prob(
    pred: np.ndarray, resid_sorted: np.ndarray, thr: float, above: bool = True
) -> np.ndarray:
    """P(pred + R beyond thr) under the empirical residual distribution R = y - yhat (sorted)."""
    n = len(resid_sorted)
    if above:  # P(R > thr - pred)
        return 1.0 - np.searchsorted(resid_sorted, thr - pred, side="right") / n
    return np.searchsorted(resid_sorted, thr - pred, side="left") / n  # P(R < thr - pred)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Platt:
    """Logistic map on the logit of a score in [0, 1] (or on a raw score if ``raw``)."""

    def __init__(self, raw: bool = False) -> None:  # noqa: D107
        self.raw = raw
        self.m = LogisticRegression(C=1e6, max_iter=1000)

    def _x(self, s: np.ndarray) -> np.ndarray:
        return (s if self.raw else _logit(s)).reshape(-1, 1)

    def fit(self, s: np.ndarray, y: np.ndarray) -> Platt:
        """Fit on training-patient scores / labels."""
        self.m.fit(self._x(s), y.astype(int))
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        """Calibrated probability."""
        return self.m.predict_proba(self._x(s))[:, 1]


class Isotonic:
    """Isotonic calibration map (clipped outside the training range)."""

    def __init__(self) -> None:  # noqa: D107
        self.m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, s: np.ndarray, y: np.ndarray) -> Isotonic:
        """Fit on training-patient scores / labels."""
        self.m.fit(s, y.astype(float))
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        """Calibrated probability."""
        return self.m.predict(s)


def ece(p: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Expected calibration error with equal-width bins."""
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / max(len(p), 1))


def reliability_curve(p: np.ndarray, y: np.ndarray, bins: int) -> dict[str, list[float]]:
    """Mean predicted vs observed frequency per equal-width bin (bins with rows only)."""
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    out: dict[str, list[float]] = {"pred": [], "obs": [], "n": []}
    for k in range(bins):
        m = b == k
        if m.any():
            out["pred"].append(float(p[m].mean()))
            out["obs"].append(float(y[m].mean()))
            out["n"].append(int(m.sum()))
    return out


def net_benefit(p: np.ndarray, y: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Decision-curve net benefit TP/N - FP/N * pt / (1 - pt) at each threshold probability."""
    n = len(y)
    out = []
    for pt in pts:
        pos = p >= pt
        tp = np.sum(pos & (y == 1))
        fp = np.sum(pos & (y == 0))
        out.append(tp / n - fp / n * pt / (1 - pt))
    return np.asarray(out)
