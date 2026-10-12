"""Baseline correctness tests on synthetic series (no real data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt

from src.models.baselines import PatientAR, PooledAR, holt_forecast


def test_holt_matches_statsmodels() -> None:
    rng = np.random.default_rng(42)
    y = 120 + np.cumsum(rng.normal(0, 2, 40))
    a, b, phi = 0.5, 0.2, 0.9
    ours = holt_forecast(y[None, :], a, b, phi, np.array([6, 12]))
    fit = Holt(
        y, damped_trend=True, initialization_method="known", initial_level=y[0], initial_trend=0.0
    ).fit(smoothing_level=a, smoothing_trend=b, damping_trend=phi, optimized=False)
    sm_fc = fit.forecast(12)
    assert np.isclose(ours[6][0], sm_fc[5], atol=1e-8)
    assert np.isclose(ours[12][0], sm_fc[11], atol=1e-8)


def _toy_grid(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2021-01-01", periods=n, freq="5min")
    v = 130 + 30 * np.sin(np.arange(n) / 40) + rng.normal(0, 3, n)
    return pd.DataFrame(
        {"patient_id": "p1", "period_id": "p1", "t": t, "glucose": v, "glucose_is_real": True}
    )


def test_patient_ar_is_causal() -> None:
    """Changing glucose AFTER an anchor's time must not change that anchor's forecast."""
    g = _toy_grid()
    anchors = g.iloc[1500:1510].copy()
    anchors["cgm_now"] = anchors["glucose"]
    for h in (30, 60):
        anchors[f"y_{h}"] = g["glucose"].shift(-(h // 5)).iloc[1500:1510].to_numpy()
    train = g.iloc[100:1400].assign(cgm_now=lambda d: d.glucose)
    for h in (30, 60):
        train[f"y_{h}"] = g["glucose"].shift(-(h // 5)).iloc[100:1400].to_numpy()
    model = PatientAR(PooledAR([30], step=5), ridge_lambda=50.0, min_pairs=50, step=5)
    model.fit(train, g, val=train, horizons=[30, 60])
    p1 = model.predict(anchors, g, [30, 60])
    g2 = g.copy()
    g2.loc[1510:, "glucose"] += 80.0  # perturb the future of every anchor (t <= index 1509)
    p2 = model.predict(anchors, g2, [30, 60])
    for h in (30, 60):
        assert np.allclose(p1[h], p2[h])
        # sanity: the per-patient fit actually differs from the pooled fit (it is used)
        assert not np.allclose(p1[h], model.pooled.predict(anchors, g, [h])[h])
