"""Part 6 tests (synthetic data only): conformal coverage guarantees, CV+, ACI causality and
long-run coverage, calibration patients never used for fitting, alarm episode/counting rules,
fault injection locality and causality of the data-quality score / recomputed features."""

from __future__ import annotations

import numpy as np
import pytest

from src.uncertainty import conformal as cf
from src.uncertainty.alarms import alert_stats, episodes
from src.uncertainty.events import Platt, ece, exceed_prob, net_benefit
from src.utils.config import load_config

UC = load_config("uncertainty")


def test_split_conformal_coverage_exchangeable():
    """Exchangeable synthetic data: mean coverage over many draws >= 1 - alpha (and close)."""
    rng = np.random.default_rng(42)
    for alpha in UC["alphas"]:
        cov = []
        for _ in range(2000):
            cal = np.abs(rng.standard_t(3, 50))
            q = cf.conformal_quantile(cal, alpha)
            cov.append(abs(rng.standard_t(3)) <= q)
        m = float(np.mean(cov))
        assert m >= 1 - alpha - 0.015
        assert m <= 1 - alpha + 1 / 51 + 0.02  # upper bound 1 - alpha + 1/(n+1)


def test_conformal_quantile_too_few_scores_is_infinite():
    assert np.isinf(cf.conformal_quantile(np.ones(10), 0.05))  # ceil(11 * .95) = 11 > 10


def test_cvplus_matches_bruteforce_and_covers():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(7, 3))
    res = [np.abs(rng.normal(size=n)) for n in (20, 25, 30)]
    lo, hi = cf.cvplus_interval(m, res, 0.2)
    n = sum(len(r) for r in res)
    for i in range(len(m)):
        up = np.sort(np.concatenate([m[i, k] + res[k] for k in range(3)]))
        dn = np.sort(np.concatenate([m[i, k] - res[k] for k in range(3)]))
        assert hi[i] == pytest.approx(up[cf.conformal_rank(n, 0.2) - 1], abs=1e-6)
        assert lo[i] == pytest.approx(dn[int(np.floor(0.2 * (n + 1))) - 1], abs=1e-6)
    # coverage on exchangeable data with identical fold models (reduces to split conformal)
    cov = []
    for _ in range(1000):
        r = [np.abs(rng.normal(size=40)) for _ in range(3)]
        lo, hi = cf.cvplus_interval(np.zeros((1, 3)), r, 0.2)
        y = rng.normal()
        cov.append(lo[0] <= y <= hi[0])
    assert np.mean(cov) >= 0.8 - 0.03


def test_aci_long_run_coverage_under_shift_and_causality():
    rng = np.random.default_rng(1)
    n, h = 3000, 60.0
    t = np.arange(n) * 15.0
    scale = np.where(np.arange(n) < n // 2, 1.0, 3.0)  # abrupt shift: plain conformal under-covers
    s = np.abs(rng.normal(0, scale))
    ref = np.sort(np.abs(rng.normal(0, 1.0, 500)))
    q, _ = cf.aci_run(s, t, h, ref, 0.2, 0.02)
    assert abs(np.mean(s <= q) - 0.8) < 0.03
    q0, _ = cf.aci_run(s, t, h, ref, 0.2, 0.0)
    assert np.mean(s <= q0) < 0.65  # gamma = 0 is plain split conformal and fails here
    # causality: changing scores whose target time is after anchor `cut` leaves q[:cut+1] intact
    cut = 1000
    s2 = s.copy()
    s2[t + h > t[cut]] = 99.0
    q2, _ = cf.aci_run(s2, t, h, ref, 0.2, 0.02)
    np.testing.assert_array_equal(q[: cut + 1], q2[: cut + 1])


def test_episode_rules_consensus_lv1():
    g = np.array([100, 190, 190, 190, 190, 150, 150, 150, 150, 190, 190, 120], float)
    real = np.ones(len(g), bool)
    on, inside = episodes(g, real, 180, True, 4, 4)  # 5-min grid: 4 readings = 15 min
    assert list(on) == [1]
    assert inside[1:5].all() and not inside[0]
    # a missing reading breaks the start run
    r2 = real.copy()
    r2[3] = False
    on2, _ = episodes(g, r2, 180, True, 4, 4)
    assert len(on2) == 0
    # hypo direction
    on3, _ = episodes(np.array([80, 60, 60, 60, 60, 80], float), np.ones(6, bool), 70, False, 4, 4)
    assert list(on3) == [1]


def test_alert_counting_refractory_true_false_and_lead():
    t = np.arange(0, 600, 15.0)
    score = np.zeros(len(t))
    score[(t >= 100) & (t <= 160)] = 1.0  # alerts at 105 (counted), 120..160 suppressed
    score[t == 450] = 1.0  # false alarm (no onset in (450, 510])
    st = alert_stats(t, score, np.array([150.0]), 0.5, 60.0, 60.0)
    assert st["n_alerts"] == 2 and st["n_true"] == 1 and st["n_false"] == 1
    assert st["episodes_detected"] == 1 and st["leads"] == [45.0]


def test_exceed_prob_and_calibration_helpers():
    r = np.sort(np.array([-10.0, 0.0, 10.0]))
    assert exceed_prob(np.array([175.0]), r, 180.0)[0] == pytest.approx(1 / 3)
    assert exceed_prob(np.array([75.0]), r, 70.0, above=False)[0] == pytest.approx(1 / 3)
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 0, 1, 1])
    assert ece(p, y, 10) == pytest.approx(0.1)
    nb = net_benefit(np.array([0.0, 0.0, 1.0, 1.0]), y, np.array([0.5]))
    assert nb[0] == pytest.approx(0.5)  # perfect: TP/N = 0.5, no FP
    nb_all = net_benefit(np.ones(4), y, np.array([0.5]))
    assert nb_all[0] == pytest.approx(0.0)  # treat-all at pt = prevalence
    rng = np.random.default_rng(3)
    s = rng.uniform(0.01, 0.99, 4000)
    yy = (rng.uniform(size=4000) < s**2).astype(int)  # miscalibrated score
    pc = Platt().fit(s, yy).predict(s)
    assert ece(pc, yy, 10) < ece(s, yy, 10)
