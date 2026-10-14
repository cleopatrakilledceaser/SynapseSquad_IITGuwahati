"""Part 5 hybrid tests: twin-feature kernel equivalence and causality, train/test consistency,
hybrid row rules, folds, PINN physics residual (synthetic inputs; no real data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.hybrid.twin_features import add_derived, ekf_features, record_features
from src.mech.fit import config_prior
from src.mech.inputs import Record
from src.mech.model import STATES, bounds_array, ekf_forecast, theta_from_log
from src.utils.config import load_config

MC = load_config("mech")
HC = load_config("hybrid")
BND = bounds_array(MC["solver"])
N_DAY = 288


def _record(n_days: float = 2.0, seed: int = 0) -> Record:
    rng = np.random.default_rng(seed)
    n = int(n_days * N_DAY)
    t = np.datetime64("2024-01-01T00:00") + np.arange(n) * np.timedelta64(5, "m")
    carbs = np.zeros(n)
    idx = np.array([96, 156, 228, 384, 444])
    carbs[idx[idx < n]] = np.array([40, 60, 70, 50, 80])[idx < n]
    y = 110 + 25 * np.sin(np.arange(n) / 40.0) + rng.normal(0, 4, n)
    y[rng.random(n) < 0.05] = np.nan
    act = np.clip(rng.normal(0, 1, n), 0, None)
    slow = np.full(n, 0.3)
    return Record("p1", "p1_0", t, np.arange(n) / N_DAY, y, carbs, slow, act, "synthetic")


def _args(rec: Record, alpha: float = 0.002):
    th = theta_from_log(config_prior(MC["params"]).mu, MC["params"]["fixed"], alpha)
    e = MC["ekf"]
    qd = np.diag([4.0 * e["q_base"][s] for s in STATES])
    p0 = np.array([e["p0"][s] for s in STATES])
    anchors = np.arange(10, len(rec.y) - 30, 7)
    hs = np.array([6, 12, 24])
    return th, rec.y, rec.carbs, rec.slow, rec.act, anchors, hs, qd, 25.0, p0, 5.0, 1, BND


def test_ekf_features_matches_part4_forecast():
    rec = _record()
    args = _args(rec)
    m0, v0 = ekf_forecast(*args)
    m1, v1, st, inn, iv, traj = ekf_features(*args)
    np.testing.assert_allclose(m1, m0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(v1, v0, rtol=0, atol=1e-9)
    assert st.shape == (len(args[5]), 6) and traj.shape == (len(args[5]), 25, 3)
    assert np.isnan(inn[np.isnan(rec.y)]).all()


def test_record_features_causal_under_future_perturbation():
    """Changing CGM, meals and activity AFTER t leaves every tw_* feature at t unchanged."""
    rec = _record()
    cut = 300
    ts = rec.t[np.arange(20, cut + 1, 9)]
    lp = config_prior(MC["params"]).mu
    a = record_features(rec, ts, lp, MC, 0.002, 4.0, 25.0, 1, HC["twin"], keep_traj=True)
    rec2 = _record()
    rng = np.random.default_rng(5)
    rec2.y[cut + 1 :] = rng.normal(250, 40, len(rec2.y) - cut - 1)
    rec2.carbs[cut + 1 :] = 100.0
    rec2.act[cut + 1 :] = 9.0
    rec2.slow[cut + 1 :] = 3.0
    b = record_features(rec2, ts, lp, MC, 0.002, 4.0, 25.0, 1, HC["twin"], keep_traj=True)
    cols = [c for c in a.columns if c.startswith(("tw_", "tr_"))]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_record_features_change_with_past():
    """Sanity: features DO respond to data before t (the test above is not vacuous)."""
    rec = _record()
    ts = rec.t[[200]]
    lp = config_prior(MC["params"]).mu
    a = record_features(rec, ts, lp, MC, 0.002, 4.0, 25.0, 1, HC["twin"])
    rec.y[150:200] += 40
    b = record_features(rec, ts, lp, MC, 0.002, 4.0, 25.0, 1, HC["twin"])
    assert abs(a.tw_mu_60.iloc[0] - b.tw_mu_60.iloc[0]) > 1.0


def test_ode_residual_near_zero_on_twin_rollout():
    """The physics term is ~0 for the twin's own open-loop G (same equation, RK4 vs trapezoid)."""
    from src.hybrid.pinn import ode_residual
    from src.mech.model import initial_state, simulate

    th = theta_from_log(config_prior(MC["params"]).mu, MC["params"]["fixed"], 0.0)
    x0 = initial_state(120.0, th)
    x0[1] = 30.0  # carbohydrate in the gut -> non-trivial rise
    n = 25
    traj = simulate(th, x0, np.zeros(n), np.zeros(n), np.zeros(n), 5.0, 1, BND)
    k = th[4] / (1.0 + th[5] * 0.0)
    ra = th[3] * k * traj[:, 1]
    s = np.full(n, th[1])
    r = ode_residual(
        traj[None, 1:, 2], traj[None, 0, 2], traj[None, :, 4], ra[None], s[None], th[:1], 5.0
    )
    assert np.max(np.abs(r)) < 0.05  # mg/dL/min; glucose changes by ~1 mg/dL/min here
    bad = ode_residual(
        traj[None, 1:, 2] + 20.0, traj[None, 0, 2], traj[None, :, 4], ra[None], s[None], th[:1], 5.0
    )
    assert np.max(np.abs(bad)) > 1.0


def test_dense_targets_real_only_and_aligned():
    from src.hybrid.pinn import dense_targets

    t = pd.date_range("2024-01-01", periods=10, freq="5min")
    grid = pd.DataFrame(
        {
            "period_id": "a",
            "t": t,
            "glucose": np.arange(10.0) + 100,
            "glucose_is_real": [True] * 4 + [False] + [True] * 5,
        }
    )
    anc = pd.DataFrame({"period_id": ["a"], "t": [t[2]]})
    d = dense_targets(grid, anc, 4)
    np.testing.assert_array_equal(d[0], [103.0, np.nan, 105.0, 106.0])


def test_fold_features_train_test_consistency(monkeypatch):
    """Training AND test patients' twin parameters at level k use only their first k days;
    training patients' prior excludes their own stage-1 fit (leave-one-out)."""
    import src.hybrid.build as hb
    from src.mech.protocol import Ctx

    seen: list[tuple[str, float, object]] = []

    def fake_fit_pids(c, pids, k, prior_of, alpha):
        out = {}
        for p in pids:
            pr = prior_of(p)
            seen.append((p, k, pr))
            out[p] = pr.mu.copy() + (0.01 if k > 0 else 0.0)
        return out

    def fake_select_qr(c, train, lps, alpha):
        return 1.0, 25.0, {}

    monkeypatch.setattr(hb, "fit_pids", fake_fit_pids)
    monkeypatch.setattr(hb, "select_qr", fake_select_qr)
    monkeypatch.setattr(hb, "_part4_choice", lambda *a: {})
    recs = {f"p{i}": [_record(1.0, seed=i)] for i in range(5)}
    for i, (p, rr) in enumerate(recs.items()):
        rr[0].patient_id, rr[0].period_id = p, f"{p}_0"
        rr[0].y = rr[0].y + 5 * i
    anchors = pd.concat(
        [pd.DataFrame({"patient_id": p, "period_id": f"{p}_0", "t": r[0].t[20::50],
                       "days_since_start": r[0].days[20::50]}) for p, r in recs.items()]
    )  # fmt: skip
    static = pd.DataFrame({"patient_id": list(recs), "sex": ["M", "F", "M", "F", "M"]})
    for i, f in enumerate(MC["prior"]["static_features"]):
        if f != "sex_male":
            static[f] = np.arange(5.0) + i
    c = Ctx("cgmacros", MC, recs, anchors, static, 1,
            ["Gb", "SG", "SI", "fc", "kabs", "beta"], config_prior(MC["params"]).mu)  # fmt: skip
    hc = {**HC, "k_grid": [0, 0.5]}
    train, test = ["p0", "p1", "p2", "p3"], ["p4"]
    st1_vals = {}

    def fit_stage1(c_, pids, k, prior_of, alpha):  # distinct stage-1 estimate per patient
        if k == MC["fit"]["stage1_days"]:
            out = {p: prior_of(p).mu + 0.1 * (i + 1) for i, p in enumerate(pids)}
            st1_vals.update(out)
            seen.extend((p, k, prior_of(p)) for p in pids)
            return out
        return fake_fit_pids(c_, pids, k, prior_of, alpha)

    monkeypatch.setattr(hb, "fit_pids", fit_stage1)
    feat, _, fits, ch = hb.fold_features(c, 0, train, test, hc)
    st1 = [s for s in seen if s[1] == MC["fit"]["stage1_days"]]
    assert {s[0] for s in st1} == set(train)  # stage 1 / priors: training patients only
    for p, k, pr in seen:
        if k in (0, 0.5) and p in train:  # leave-one-out: mean of the OTHER training patients
            others = np.mean([st1_vals[q] for q in train if q != p], axis=0)
            np.testing.assert_allclose(pr.mu, others)
    pop_mu = np.mean([st1_vals[q] for q in train], axis=0)
    for k in (0, 0.5):  # test patient: one population-prior fit + one static-prior fit per k
        pri = [pr.mu for p, kk, pr in seen if p in test and kk == k]
        assert len(pri) == 2
        np.testing.assert_allclose(pri[0], pop_mu)  # full training-population prior
        assert not np.allclose(pri[1], pop_mu)  # static ridge prior (training patients only)
    assert set(feat.loc[feat.role == "test", "patient_id"]) == {"p4"}
    assert set(feat.loc[feat.role == "train", "patient_id"]) == set(train)
    assert set(fits.k) == {0, 0.5}


def test_hybrid_training_rows_start_at_day_k():
    from src.hybrid.models import hybrid_rows

    tab = pd.DataFrame(
        {"patient_id": "p", "period_id": "p_0", "t": pd.date_range("2024-01-01", periods=20,
         freq="6h"), "days_since_start": np.arange(20) / 4, "cgm_now": 100.0}
    )  # fmt: skip
    import src.hybrid.models as hm

    tw = tab[["period_id", "t"]].copy()
    for h in (30, 60, 120):
        tw[f"tw_mu_{h}"] = 100.0
    tw["tw_Gf"] = 100.0
    orig = hm.twin_table
    hm.twin_table = lambda *a, **k: tw
    try:
        d = hybrid_rows(tab, "cgmacros", 0, 3, ["p"], "train", 3)
    finally:
        hm.twin_table = orig
    assert d.days_since_start.min() >= 3


def test_folds_identical_to_saved_cv_repeat0():
    """Every Part 5 stage takes folds from the saved Part 2 assignment (same as Parts 2-4)."""
    from pathlib import Path

    from src.models.fusion_lgbm import folds

    p = Path(__file__).resolve().parents[1] / "data" / "processed" / "splits" / "cgmacros_cv.csv"
    if not p.exists():
        pytest.skip("processed splits not available (CI without data)")
    cv = pd.read_csv(p)
    cv0 = cv[cv.repeat == 0]
    for _, f, tr, te in folds("cgmacros", [0]):
        assert sorted(cv0.loc[cv0.fold == f, "patient_id"]) == te
        assert not set(tr) & set(te)


def test_add_derived_deltas():
    df = pd.DataFrame({"cgm_now": [100.0], "tw_Gf": [104.0]})
    for h in (30, 60, 120):
        df[f"tw_mu_{h}"] = 100.0 + h
    out = add_derived(df)
    assert out.tw_d60.iloc[0] == 60.0 and out.tw_gres.iloc[0] == 4.0
