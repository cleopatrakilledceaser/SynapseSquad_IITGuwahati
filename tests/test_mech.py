"""Part 4 mechanistic twin tests (synthetic inputs only; no real data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mech.fit import config_prior, fit_patient, log_bounds
from src.mech.identify import synthetic_record
from src.mech.inputs import Record, activity_input, meal_arrays, rest_level
from src.mech.model import (
    FIT_PARAMS,
    bounds_array,
    ekf_forecast,
    initial_state,
    simulate,
    theta_from_log,
)
from src.mech.simulate import ar1_noise
from src.utils.config import load_config

MC = load_config("mech")
BND = bounds_array(MC["solver"])
FIX = MC["params"]["fixed"]
N_DAY = 288


def _theta(alpha: float = 0.002, **over: float) -> np.ndarray:
    lp = config_prior(MC["params"]).mu.copy()
    for k, v in over.items():
        lp[FIT_PARAMS.index(k)] = np.log(v)
    return theta_from_log(lp, FIX, alpha)


def _run(th: np.ndarray, carbs: np.ndarray, act: np.ndarray | None = None, g0: float = 115.0):
    n = len(carbs)
    act = np.zeros(n) if act is None else act
    return simulate(th, initial_state(g0, th), carbs, np.zeros(n), act, 5.0, 1, BND)


def _toy_record(days: int = 3, seed: int = 0) -> Record:
    """SYNTHETIC record: 3 meals/day, some activity, all readings observed."""
    rng = np.random.default_rng(seed)
    n = days * N_DAY
    carbs = np.zeros(n)
    for d in range(days):
        for hh, c in ((8, 50), (13, 70), (19, 60)):
            carbs[d * N_DAY + hh * 12] = c + rng.integers(-10, 10)
    act = np.zeros(n)
    act[(np.arange(n) % N_DAY >= 17 * 12) & (np.arange(n) % N_DAY < 17 * 12 + 6)] = 2.0
    t = (np.datetime64("2026-01-01T00:00") + np.arange(n) * np.timedelta64(5, "m")).astype(
        "datetime64[ns]"
    )
    return Record("synthetic_toy", "synthetic_toy", t, np.arange(n) / N_DAY, np.full(n, 110.0),
                  carbs, np.full(n, 0.5), act, "mets")  # fmt: skip


def test_relaxes_to_basal_without_inputs() -> None:
    """No meals, no activity: glucose returns to Gb from above and below."""
    th = _theta(Gb=110.0)
    for g0 in (200.0, 70.0):
        g = _run(th, np.zeros(N_DAY * 2), g0=g0)[:, 2]
        assert abs(g[-1] - 110.0) < 2.0


def test_rest_is_a_fixed_point() -> None:
    """At G = Gb with empty gut and no activity, the state does not move."""
    th = _theta(Gb=120.0)
    x = _run(th, np.zeros(N_DAY), g0=120.0)
    assert np.abs(x[:, 2] - 120.0).max() < 1e-6


def test_more_carbs_higher_peak() -> None:
    th = _theta()
    peaks = []
    for c in (20.0, 40.0, 80.0):
        carbs = np.zeros(N_DAY)
        carbs[10] = c
        peaks.append(_run(th, carbs)[:, 2].max())
    assert peaks[0] < peaks[1] < peaks[2]


def test_activity_does_not_raise_peak() -> None:
    th = _theta(alpha=0.004)
    carbs = np.zeros(N_DAY)
    carbs[10] = 60.0
    act = np.zeros(N_DAY)
    act[13:19] = 2.0
    assert _run(th, carbs, act)[:, 2].max() <= _run(th, carbs)[:, 2].max() + 1e-9


def test_slowing_lowers_and_delays_peak() -> None:
    th = _theta(beta=2.0)
    carbs = np.zeros(N_DAY)
    carbs[10] = 60.0
    x0 = initial_state(115.0, th)
    fast = simulate(th, x0, carbs, np.zeros(N_DAY), np.zeros(N_DAY), 5.0, 1, BND)[:, 2]
    slow = simulate(th, x0, carbs, np.full(N_DAY, 2.0), np.zeros(N_DAY), 5.0, 1, BND)[:, 2]
    assert slow.max() < fast.max() and slow.argmax() > fast.argmax()


@pytest.mark.parametrize("corner", ["lo", "hi"])
def test_solver_stable_at_bound_corners(corner: str) -> None:
    """Every fitted parameter at its lower / upper bound: finite, within state bounds."""
    lo, hi = log_bounds(MC["params"])
    th = theta_from_log(lo if corner == "lo" else hi, FIX, 0.008)
    carbs = np.zeros(N_DAY * 2)
    carbs[::60] = 150.0
    x = _run(th, carbs, np.full(len(carbs), 3.0), g0=300.0)
    assert np.isfinite(x).all()
    assert x[:, 2].min() >= BND[0] and x[:, 2].max() <= BND[1]


def test_rk4_step_halving_converges() -> None:
    """dt = 5 min vs 1-min sub-steps agree closely (numerical accuracy of the grid step)."""
    th = _theta()
    carbs = np.zeros(N_DAY)
    carbs[10] = 60.0
    z = np.zeros(N_DAY)
    a = simulate(th, initial_state(115.0, th), carbs, z, z, 5.0, 1, BND)[:, 2]
    b = simulate(th, initial_state(115.0, th), carbs, z, z, 5.0, 5, BND)[:, 2]
    assert np.abs(a - b).max() < 0.5


def test_simulation_causal_in_meals_and_activity() -> None:
    """Changing meals / activity after step j leaves the trajectory up to j unchanged."""
    th = _theta()
    carbs = np.zeros(N_DAY)
    carbs[[10, 100]] = [50.0, 70.0]
    act = np.zeros(N_DAY)
    act[50:60] = 1.0
    base = _run(th, carbs, act)[:, 2]
    c2, a2 = carbs.copy(), act.copy()
    c2[150:] += 80.0
    a2[150:] += 3.0
    alt = _run(th, c2, a2)[:, 2]
    assert np.array_equal(base[:150], alt[:150])
    assert not np.array_equal(base[150:], alt[150:])


def test_ekf_forecast_causal() -> None:
    """Forecasts at anchor i do not change when readings / meals / activity after i change."""
    rec = _toy_record()
    th = _theta()
    rng = np.random.default_rng(1)
    y = 110 + 20 * np.sin(np.arange(len(rec.y)) / 40) + rng.normal(0, 3, len(rec.y))
    y[::7] = np.nan
    anchors = np.array([300, 400, 500])
    hs = np.array([6, 12, 24])
    qd = np.diag([1.0, 1.0, 4.0, 4.0, 1e-7, 0.0])
    p0 = np.array([100.0, 100.0, 100.0, 100.0, 1e-6, 0.01])

    def run(yy, cc, aa):  # type: ignore[no-untyped-def]
        return ekf_forecast(th, yy, cc, rec.slow, aa, anchors, hs, qd, 25.0, p0, 5.0, 1, BND)

    m0, v0 = run(y, rec.carbs, rec.act)
    y2, c2, a2 = y.copy(), rec.carbs.copy(), rec.act.copy()
    y2[501:] += 50
    c2[501:] += 40
    a2[501:] += 2
    m1, v1 = run(y2, c2, a2)
    assert np.array_equal(m0, m1) and np.array_equal(v0, v1)
    y3 = y.copy()
    y3[450:] += 50  # changes before anchor 500 must matter
    m2, _ = run(y3, rec.carbs, rec.act)
    assert not np.array_equal(m0[2], m2[2]) and np.array_equal(m0[:2], m2[:2])
    assert (v0 > 0).all()


def test_meal_arrays_enter_at_next_grid_time() -> None:
    t = (np.datetime64("2026-01-01T08:00") + np.arange(12) * np.timedelta64(5, "m")).astype(
        "datetime64[ns]"
    )
    meals = pd.DataFrame(
        {
            "t": pd.to_datetime(["2026-01-01 08:12", "2026-01-01 08:30"]),
            "carbs_g": [40.0, 20.0],
            "protein_g": [10.0, 0.0],
            "fat_g": [10.0, 0.0],
            "fiber_g": [5.0, 0.0],
        }
    )
    carbs, slow = meal_arrays(t, meals, 0.5, None)
    assert carbs[3] == 40.0 and carbs[6] == 20.0 and carbs.sum() == 60.0
    assert slow[2] == 0.0 and slow[3] == pytest.approx(20.0 / 41.0) and slow[6] == 0.0


def test_rest_level_is_past_only_with_default() -> None:
    """Resting level uses samples strictly before t; default until min_n samples exist."""
    x = pd.Series(np.r_[np.full(10, 60.0), np.full(10, 120.0)])
    r = rest_level(x, 0.25, 5, 70.0)
    assert (r[:5] == 70.0).all()
    assert r[10] == 60.0  # the jump at index 10 is not yet seen
    x2 = x.copy()
    x2[15:] = 500.0
    assert np.array_equal(rest_level(x2, 0.25, 5, 70.0)[:16], r[:16])


def test_activity_input_is_above_rest_and_nonnegative() -> None:
    acfg = MC["activity"]
    n = 200
    g = pd.DataFrame({"activity_intensity": np.r_[np.zeros(150), np.full(50, 2.0)],
                      "hr": np.r_[np.full(150, 65.0), np.full(50, 105.0)]})  # fmt: skip
    a = activity_input(g, acfg)
    assert (a >= 0).all() and a[:150].max() < 1e-9 and a[160] == pytest.approx(2.0 + 40 / 20)
    assert len(a) == n


def test_parameter_recovery_on_synthetic_record() -> None:
    """Gb and fc recover within 25% from 3 synthetic days with sensor noise."""
    rec = _toy_record()
    lo, hi = log_bounds(MC["params"])
    true = config_prior(MC["params"]).mu.copy()
    true[FIT_PARAMS.index("Gb")] = np.log(130.0)
    true[FIT_PARAMS.index("fc")] = np.log(6.0)
    syn = synthetic_record(rec, true, MC, 0.002, 5.0, np.random.default_rng(42))
    assert syn.patient_id.endswith("_synthetic")
    fr = fit_patient(syn, 3, config_prior(MC["params"]), MC, 0.002)
    est = np.exp(fr.logp)
    assert abs(est[FIT_PARAMS.index("Gb")] / 130.0 - 1) < 0.25
    assert abs(est[FIT_PARAMS.index("fc")] / 6.0 - 1) < 0.25
    assert ((fr.logp >= lo - 1e-9) & (fr.logp <= hi + 1e-9)).all()


def test_fit_uses_only_first_k_days() -> None:
    """Changing readings after day k does not change the k-day fit."""
    rec = _toy_record()
    rng = np.random.default_rng(3)
    rec.y = 110 + rng.normal(0, 10, len(rec.y))
    pr = config_prior(MC["params"])
    a = fit_patient(rec, 1, pr, MC, 0.002)
    rec.y[N_DAY:] += 80
    b = fit_patient(rec, 1, pr, MC, 0.002)
    assert np.array_equal(a.logp, b.logp)


def test_noise_and_simulation_reproducible_by_seed() -> None:
    a = ar1_noise(500, 3.0, 5.0, 0.9, np.random.default_rng(42))
    b = ar1_noise(500, 3.0, 5.0, 0.9, np.random.default_rng(42))
    c = ar1_noise(500, 3.0, 5.0, 0.9, np.random.default_rng(43))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    rec = _toy_record()
    true = config_prior(MC["params"]).mu
    s1 = synthetic_record(rec, true, MC, 0.002, 5.0, np.random.default_rng(7))
    s2 = synthetic_record(rec, true, MC, 0.002, 5.0, np.random.default_rng(7))
    assert np.array_equal(s1.y, s2.y)


def test_fixed_params_held_at_population_value() -> None:
    """Revised spec (D-037): fixed params equal the training-population value; others are fitted."""
    from src.mech.fit import Prior
    from src.mech.protocol import Ctx, fix_to_population

    pr = config_prior(MC["params"])
    c = Ctx("toy", MC, {}, pd.DataFrame(), pd.DataFrame(), 1, list(FIT_PARAMS), pr.mu.copy())
    pop = Prior(pr.mu + 0.3, pr.sd)
    c2 = fix_to_population(c, pop, ["beta"])
    assert c2.free == [p for p in FIT_PARAMS if p != "beta"] and c.free == list(FIT_PARAMS)
    ib = FIT_PARAMS.index("beta")
    assert c2.base_logp[ib] == pop.mu[ib]
    rec = _toy_record()
    fr = fit_patient(rec, 1, pr, MC, 0.002, c2.free, c2.base_logp)
    assert fr.logp[ib] == pop.mu[ib]
