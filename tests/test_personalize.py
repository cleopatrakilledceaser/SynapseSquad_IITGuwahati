"""Part 5.2 personalization tests: adaptation windows, online-learner causality, population
models without evaluated-patient data, first-period rule, identical folds (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.personalize import methods as pm
from src.personalize.data import eligible, first_period, inner_folds
from src.utils.config import load_config

PC = load_config("personalize")


def test_adapt_mask_targets_before_k_and_never_in_test_region():
    days = np.arange(0, 10, 1 / 96)
    y = np.ones_like(days)
    for k in PC["k_grid"]:
        for h in (30, 60, 120):
            a = pm.adapt_mask(days, h, k, y)
            assert np.all(days[a] + h / 1440 < k)  # target time before day k
            assert not np.any(a & (days >= PC["test_from_day"]))  # no overlap with test region
    assert not pm.adapt_mask(days, 30, 0, y).any()  # k = 0: no adaptation pairs


def test_test_region_after_all_adaptation_days():
    assert PC["test_from_day"] >= max(PC["k_grid"])


def _rls_inputs(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 15.0
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    resid = 5.0 + 3.0 * x[:, 1] + rng.normal(0, 1, n)
    return x, resid, t


def test_rls_causal_future_residuals_do_not_change_past_corrections():
    """Continuous RLS uses a pair only once t_s + h <= t_i: corrections ignore later targets."""
    x, r, t = _rls_inputs()
    h, cut = 60.0, 200
    c1, _ = pm.rls_run(x, r, t, h, np.inf, 0.995, 100.0, 2.0, 50.0, 10.0)
    r2 = r.copy()
    # residuals whose TARGET time is after anchor `cut` are changed
    late = t + h > t[cut]
    r2[late] = 1000.0
    c2, _ = pm.rls_run(x, r2, t, h, np.inf, 0.995, 100.0, 2.0, 50.0, 10.0)
    np.testing.assert_array_equal(c1[: cut + 1], c2[: cut + 1])
    assert not np.allclose(c1[cut + 10 :], c2[cut + 10 :])


def test_rls_frozen_mode_uses_only_pairs_before_limit():
    x, r, t = _rls_inputs()
    h, limit = 60.0, 1440.0  # first day
    c1, _ = pm.rls_run(x, r, t, h, limit, 0.995, 100.0, 2.0, 50.0, 10.0)
    r2 = r.copy()
    r2[t + h >= limit] = -500.0
    c2, _ = pm.rls_run(x, r2, t, h, limit, 0.995, 100.0, 2.0, 50.0, 10.0)
    np.testing.assert_array_equal(c1, c2)
    after = t >= limit
    np.testing.assert_allclose(
        c1[after] - (x[after] @ np.linalg.lstsq(x[after], c1[after], rcond=None)[0]), 0, atol=1e-8
    )  # frozen: linear in x, fixed theta


def test_rls_learns_offset_and_drift_alarm():
    x, r, t = _rls_inputs(800)
    c, _ = pm.rls_run(x, r, t, 15.0, np.inf, 0.995, 100.0, 2.0, 50.0, 10.0)
    assert np.mean(np.abs(r[-100:] - c[-100:])) < 2.0
    r2 = r.copy()
    r2[400:] += 60.0  # abrupt drift
    _, alarms = pm.rls_run(x, r2, t, 15.0, np.inf, 0.995, 100.0, 2.0, 50.0, 10.0)
    assert alarms >= 1


def test_ridge_and_shrink():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 3))
    y = 2.0 + x @ np.array([1.0, -1.0, 0.5])
    m = pm.ridge_fit(x, y, 1e-6)
    np.testing.assert_allclose(pm.ridge_predict(m, x), y, atol=1e-4)
    assert pm.shrink_weight(0, 96) == 0.0 and pm.shrink_weight(96, 96) == pytest.approx(0.5)


def test_first_period_and_eligibility():
    t0 = pd.Timestamp("2024-01-01")
    rows = []
    for per, start in (("p_a", t0), ("p_b", t0 + pd.Timedelta(days=30))):
        for i in range(10 * 96):
            rows.append(
                {
                    "patient_id": "p",
                    "period_id": per,
                    "t": start + pd.Timedelta(minutes=15 * i),
                    "days_since_start": i / 96,
                    "y_60": 1.0,
                }
            )
    df = pd.DataFrame(rows)
    fp = first_period(df)
    assert set(fp.period_id) == {"p_a"}
    assert eligible(fp, 7, 96) == ["p"]
    assert eligible(fp, 7, 300) == []


def test_oof_forecasts_never_fit_on_predicted_patient(monkeypatch):
    """Training-patient forecasts come from inner models that did not see that patient."""
    import src.personalize.data as pdata

    fitted_on: list[set[str]] = []

    class Dummy:
        def __init__(self, pids: set[str]) -> None:
            self.pids = pids

        def predict(self, d: pd.DataFrame) -> np.ndarray:
            assert not set(d.patient_id) & self.pids  # predicted patients not in fit
            return np.zeros(len(d))

        def model_str(self) -> str:
            return ""

    def fake_fit_fixed(rows, pop, bc, n_jobs, seed=42):
        s = set(rows.patient_id)
        fitted_on.append(s)
        return {h: Dummy(s) for h in (30, 60, 120)}

    monkeypatch.setattr(pdata, "fit_fixed", fake_fit_fixed)
    pids = [f"p{i}" for i in range(9)]
    rows = pd.DataFrame({"patient_id": np.repeat(pids, 3)})
    fmap = inner_folds(pids, 3, 0)
    out, _ = pdata.oof_forecasts(rows, rows, {}, fmap, {}, 1)
    assert len(fitted_on) == 3 and sorted(out.patient_id) == sorted(rows.patient_id)


def test_inner_folds_partition_training_patients_only():
    pids = [f"p{i}" for i in range(36)]
    f = inner_folds(pids, 3, 42)
    assert set(f) == set(pids) and set(f.values()) == {0, 1, 2}
    assert f == inner_folds(list(reversed(pids)), 3, 42)  # deterministic
