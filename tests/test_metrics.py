"""Tests for metrics: Clarke zones, histogram AUC vs sklearn, bootstrap resamples PATIENTS."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.eval.metrics import (
    auc_from_hist,
    bootstrap_event,
    bootstrap_point,
    clarke_zones,
    event_hist,
    point_stats,
)


def test_clarke_known_points() -> None:
    ref = np.array([100, 100, 50, 250, 60, 200, 300])
    pred = np.array([110, 150, 60, 50, 250, 300, 150])
    assert list(clarke_zones(ref, pred)) == ["A", "B", "A", "E", "E", "B", "D"]


def test_hist_auc_matches_sklearn() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 5000)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, 5000), 0, 1)
    df = pd.DataFrame({"y": y, "p": p, "patient_id": rng.integers(0, 30, 5000).astype(str)})
    _, hp, hn, _ = event_hist(df, "y", "p", 4000)
    a, ap = auc_from_hist(hp.sum(0), hn.sum(0))
    assert abs(a - roc_auc_score(y, p)) < 2e-3
    assert abs(ap - average_precision_score(y, p)) < 1e-2


def test_bootstrap_resamples_patients_not_rows() -> None:
    """One patient with many rows and a large error: a ROW bootstrap would give a narrow CI;
    a PATIENT bootstrap must sometimes omit that patient entirely, so its CI is much wider."""
    rows = []
    for i in range(10):
        n, err = (5000, 50.0) if i == 0 else (50, 5.0)
        rows.append(
            pd.DataFrame({"patient_id": f"p{i}", "y": 100.0, "p": 100.0 + err}, index=range(n))
        )
    df = pd.concat(rows, ignore_index=True)
    st = point_stats(df, "y", "p")
    assert len(st) == 10  # sufficient statistics are per patient
    b = bootstrap_point(st, 500, 42, 0.95)
    lo, hi = b["mae_ci"]
    assert lo < 10 < 40 < hi  # patient 0 present or absent drives the spread
    ev = df.assign(e=(np.arange(len(df)) % 2).astype(float), q=0.5)
    r = bootstrap_event(ev, "e", "q", 50, 42, 0.95, 100)
    assert r["n_patients"] == 10
