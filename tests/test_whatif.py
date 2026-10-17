"""Part 7 what-if tests (synthetic data, labeled synthetic): kcal rule, zero-edit identity,
causality of ML inputs and twin, monotone constraints, run-once lock, India library schema."""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.india import library as lib
from src.mech.inputs import Record
from src.mech.model import STATES, bounds_array, ekf_forecast, theta_from_log
from src.utils.config import load_config
from src.whatif import edits as ed
from src.whatif import models as wm
from src.whatif import run as wr
from src.whatif import twin as tw

WC = load_config("whatif")
FC = load_config("features")
MC = load_config("mech")
STEP = 5
COLS = [
    "meal_minutes_since_last",
    "meal_count_1h",
    "meal_carbs_g_1h",
    "meal_kcal_1h",
    "meal_carbs_g_3h",
    "meal_protein_g_3h",
    "act_post_meal",
]


def _grid(days: float = 2.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic 5-min grid (Part 1 schema subset)."""
    rng = np.random.default_rng(seed)
    n = int(days * 1440 / STEP)
    t = pd.date_range("2026-01-01", periods=n, freq=f"{STEP}min")
    mins = np.arange(n) * float(STEP)
    return pd.DataFrame(
        {
            "t": t,
            "minutes_since_start": mins,
            "days_since_start": mins / 1440.0,
            "glucose": 110 + 20 * np.sin(mins / 200.0) + rng.normal(0, 3, n),
            "glucose_is_real": np.ones(n, bool),
            "glucose_is_carried": np.zeros(n, bool),
            "minutes_since_real": np.zeros(n),
            "hr": rng.normal(70, 4, n),
            "hr_missing": np.zeros(n, bool),
            "activity_intensity": rng.integers(0, 2, n).astype(float),
            "activity_missing": np.zeros(n, bool),
            "rest_proxy": rng.uniform(size=n) > 0.5,
        }
    )


def _meals() -> pd.DataFrame:
    t = pd.to_datetime(["2026-01-01 08:00", "2026-01-01 12:30", "2026-01-01 19:00"])
    return pd.DataFrame(
        {
            "t": t,
            "meal_type": ["breakfast", "lunch", "dinner"],
            "kcal": [400.0, 700.0, 600.0],
            "carbs_g": [50.0, 80.0, 60.0],
            "protein_g": [20.0, 30.0, 25.0],
            "fat_g": [10.0, 25.0, 20.0],
            "fiber_g": [3.0, 8.0, 5.0],
            "amount_gt100": [False, False, False],
            "food_grams_total": [np.nan] * 3,
        },
        index=[10, 11, 12],
    )


def _ctx() -> dict:
    return {"fcfg": FC, "step": STEP, "n_h": 6, "slice_hours": 8, "tol": 0.0}


def _rows(g: pd.DataFrame, meals: pd.DataFrame, times: list[str]) -> pd.DataFrame:
    """'Stored' rows built with the full-period reference path."""
    base = pd.DataFrame({"t": pd.to_datetime(times)})
    for c in COLS:
        base[c] = np.nan
    return ed.recompute_full(base, g, meals, COLS, _ctx())


def test_kcal_follows_macro_edits() -> None:
    m = _meals()
    kpg = WC["kcal_per_g"]
    e = ed.apply_meals(m, [ed.set_macros(10, carbs_g=80.0, fat_g=5.0, fiber_g=10.0)], kpg)
    assert e.loc[10, "kcal"] == pytest.approx(400 + 4 * 30 + 9 * -5)  # fiber adds no kcal
    e2 = ed.apply_meals(m, [ed.set_macros(10, carbs_g=0.0, protein_g=0.0, fat_g=0.0)], kpg)
    assert e2.loc[10, "kcal"] >= 0.0
    assert ed.apply_meals(m, [ed.remove_meal(11)], kpg).index.tolist() == [10, 12]


def test_zero_edit_identity_and_fast_equals_full() -> None:
    g, m = _grid(), _meals()
    rows = _rows(g, m, ["2026-01-01 08:15", "2026-01-01 13:00", "2026-01-01 20:00"])
    assert ed.check_identity(rows, g, m, COLS, _ctx()) == 0.0
    e = ed.apply_meals(m, [ed.set_macros(11, carbs_g=10.0)], WC["kcal_per_g"])
    fast = ed.recompute(rows, g, e, COLS, _ctx())
    full = ed.recompute_full(rows, g, e, COLS, _ctx())
    assert ed.max_diff(fast, full, COLS) == 0.0
    assert fast.loc[1, "meal_carbs_g_1h"] == 10.0


def test_edits_do_not_change_inputs_before_the_meal() -> None:
    g, m = _grid(), _meals()
    rows = _rows(g, m, ["2026-01-01 11:00", "2026-01-01 12:25", "2026-01-01 13:30"])
    for edit in (ed.set_macros(11, carbs_g=150.0), ed.remove_meal(11), ed.add_walk(11)):
        em = ed.apply_meals(m, [edit], WC["kcal_per_g"])
        eg = ed.apply_walks(g, m, [edit], WC["walk"])
        new = ed.recompute(rows, eg, em, COLS, _ctx())
        assert ed.max_diff(new.iloc[:2], rows.iloc[:2], COLS) == 0.0  # anchors before 12:30
    walk = ed.recompute(rows, ed.apply_walks(g, m, [ed.add_walk(11)], WC["walk"]), m, COLS, _ctx())
    assert walk.loc[2, "act_post_meal"] > rows.loc[2, "act_post_meal"]


def test_moved_meal_after_t_is_unknown_to_ml() -> None:
    g, m = _grid(), _meals()
    rows = _rows(g, m, ["2026-01-01 12:45"])
    em = ed.apply_meals(m, [ed.move_meal(11, 60)], WC["kcal_per_g"])
    new = ed.recompute(rows, g, em, COLS, _ctx())
    assert new.loc[0, "meal_carbs_g_1h"] == 0.0


def _record(g: pd.DataFrame, meals: pd.DataFrame) -> Record:
    from src.mech.inputs import meal_arrays

    t = g["t"].to_numpy().astype("datetime64[ns]")
    carbs, slow = meal_arrays(t, meals, MC["params"]["slow_protein_weight"], None)
    return Record("syn", "syn_p1", t, g["days_since_start"].to_numpy(), g["glucose"].to_numpy(),
                  carbs, slow, np.zeros(len(t)), "none")  # fmt: skip


def _spec() -> tw.TwinSpec:
    mu = np.log([115.0, 0.01, 0.0002, 4.0, 0.03, 0.5])
    return tw.TwinSpec(mu, 16.0, 25.0, MC)


def test_twin_zero_edit_equals_part4_ekf() -> None:
    g, m = _grid(), _meals()
    rec, spec = _record(g, m), _spec()
    t = pd.Timestamp("2026-01-01 14:00")
    known = m[m.t <= t]
    out = tw.run(spec, rec, g, known, [], t, 0.0, WC["walk"], 120.0)
    ia = int(np.searchsorted(rec.t, np.datetime64(t, "ns")))
    e = MC["ekf"]
    qd = np.diag([spec.q_scale * e["q_base"][s] for s in STATES])
    p0 = np.array([e["p0"][s] for s in STATES])
    carbs = rec.carbs.copy()
    carbs[ia + 1 :] = 0.0
    mean, var = ekf_forecast(spec.theta(0.0), rec.y, carbs, rec.slow, rec.act, np.array([ia]),
                             np.array([6, 12, 24]), qd, spec.r, p0, 5.0, 1,
                             bounds_array(MC["solver"]))  # fmt: skip
    hz = tw.at_horizons(out, [30, 60, 120])
    for j, h in enumerate((30, 60, 120)):
        assert hz[h][0] == pytest.approx(mean[0, j], abs=1e-9)
        assert hz[h][1] ** 2 == pytest.approx(var[0, j], abs=1e-9)


def test_twin_future_edits_do_not_change_past_states_and_carbs_raise_peak() -> None:
    g, m = _grid(), _meals()
    rec, spec = _record(g, m), _spec()
    t = pd.Timestamp("2026-01-01 12:40")
    peaks, hist = [], []
    for c in (0.0, 40.0, 120.0):
        em = ed.apply_meals(m, [ed.set_macros(11, carbs_g=c)], WC["kcal_per_g"])
        out = tw.run(spec, rec, g, em[em.t <= t], [], t, 0.0, WC["walk"], 120.0)
        peaks.append(out["g"].max())
        hist.append(out["hist"])
    assert peaks[0] <= peaks[1] <= peaks[2]
    nb = int(np.searchsorted(rec.t, np.datetime64("2026-01-01T12:30", "ns")))
    for h in hist[1:]:
        np.testing.assert_array_equal(h[:nb], hist[0][:nb])


def test_assumed_walk_lowers_iauc_and_calibrates() -> None:
    ac = WC["twin"]["assumed"]
    mu = _spec().logp
    assert tw.walk_reduction(mu, MC, WC["walk"], ac, 0.0) == pytest.approx(0.0, abs=1e-12)
    a = tw.calibrate_alpha(mu, MC, WC["walk"], ac, 0.2)
    assert np.isfinite(a) and 0 < a <= ac["alpha_max"]
    assert tw.walk_reduction(mu, MC, WC["walk"], ac, a) == pytest.approx(0.2, abs=1e-3)


def test_monotone_constraints_hold_under_random_perturbations() -> None:
    rng = np.random.default_rng(0)
    cols = ["meal_carbs_g_1h", "meal_kcal_1h", "meal_protein_g_1h", "act_post_meal", "cgm_now"]
    mcfg = {"increasing": ["meal_carbs_g_1h", "meal_kcal_1h"], "decreasing": ["act_post_meal"]}
    assert wm.constraint_vector(cols, mcfg) == [1, 1, 0, -1, 0]
    with pytest.raises(ValueError):
        wm.constraint_vector(["cgm_now"], mcfg)
    x = pd.DataFrame(rng.uniform(0, 100, (3000, len(cols))), columns=cols)
    y = (
        np.sin(x.meal_carbs_g_1h / 10) * 20
        - np.cos(x.act_post_meal / 7) * 5
        + rng.normal(0, 3, 3000)
    )
    mono = wm.constraint_vector(cols, mcfg)
    m = lgb.LGBMRegressor(n_estimators=80, num_leaves=15, verbose=-1, monotone_constraints=mono)
    m = m.fit(x, y)
    base = x.sample(300, random_state=1).reset_index(drop=True)
    for c, sign in (("meal_carbs_g_1h", 1), ("meal_kcal_1h", 1), ("act_post_meal", -1)):
        up = base.copy()
        up[c] = up[c] + rng.uniform(0, 50, len(up))
        assert np.all(sign * (m.predict(up) - m.predict(base)) >= -1e-9)


def test_final_stage_runs_once_unless_forced(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(wr, "OUT", tmp_path)
    monkeypatch.setattr(wr, "LOCK", tmp_path / "FINAL_DONE")
    assert wr.check_lock(False) == 0
    wr.write_lock(0, False)
    with pytest.raises(RuntimeError, match="disclosed"):
        wr.check_lock(False)
    n = wr.check_lock(True)
    wr.write_lock(n, True)
    info = json.loads((tmp_path / "FINAL_DONE").read_text())
    assert info == {"runs": 2, "forced_reruns": 1}


def test_india_library_schema() -> None:
    data = lib.load_library()
    assert len(data["presets"]) >= 15
    for p in data["presets"].values():
        assert p["source"] in lib.SOURCES
        assert set(p["macros"]) == set(lib.MACROS)
        assert not any(k.startswith("per_100") for k in p)
    bad = json.loads(json.dumps(data))
    next(iter(bad["presets"].values()))["per_100g"] = {}
    with pytest.raises(ValueError, match="per-100"):
        lib.validate(bad)
    bad = json.loads(json.dumps(data))
    bad["disclaimer"] = "values"
    with pytest.raises(ValueError):
        lib.validate(bad)


def test_dev_stage_uses_training_patients_only() -> None:
    from pathlib import Path

    splits = Path(__file__).resolve().parents[1] / "data" / "processed" / "splits"
    if not (splits / "cgmacros_cv.csv").exists():
        pytest.skip("split files not built (run `make data`)")
    from src.models.fusion_lgbm import folds

    for _, f, tr, te in folds("cgmacros", [WC["cv_repeat"]]):
        if f in WC["dev_folds"]:
            assert not set(tr) & set(te)
    assert theta_from_log is not None  # kernel import used by the twin


def test_d056_onesided_activity_never_raises_glucose_and_default_unchanged() -> None:
    from src.mech.model import deriv, initial_state, simulate

    mu = _spec().logp
    fix = MC["params"]["fixed"]
    assert len(theta_from_log(mu, fix, 0.01)) == 12  # default: original kernel (Parts 4-6)
    th1 = theta_from_log(mu, fix, 0.02, onesided=True)
    th0 = theta_from_log(mu, fix, 0.02)
    gb = float(np.exp(mu[0]))
    # direct term: at or below basal activity changes nothing; above basal it only lowers dG/dt
    for g in (60.0, gb, gb + 40.0):
        x = np.array([10.0, 5.0, g, 3.0, 0.001, 2.0])
        with_act, no_act = np.empty(6), np.empty(6)
        deriv(x, th1, 0.5, 0.0, with_act)
        x0 = x.copy()
        x0[5] = 0.0
        deriv(x0, th1, 0.5, 0.0, no_act)
        if g <= gb:
            assert with_act[2] == no_act[2]
        else:
            assert with_act[2] < no_act[2]
    bnd = bounds_array(MC["solver"])
    n = 72
    a = np.zeros(n)
    a[3:9] = 2.0
    z = np.zeros(n)
    rebound = 0.0
    for g0 in (80.0, gb, 200.0):
        base = simulate(th1, initial_state(g0, th1), z, z, z, 5.0, 1, bnd)[:, 2]
        walk = simulate(th1, initial_state(g0, th1), z, z, a, 5.0, 1, bnd)[:, 2]
        assert walk.max() <= base.max() + 1e-12  # peak (D-051 metric) never rises
        assert np.trapezoid(walk, dx=5.0) <= np.trapezoid(base, dx=5.0) + 1e-9  # nor the AUC
        rebound = max(rebound, float(np.max(walk - base)))
    # indirect only: lower glucose -> lower insulin surrogate -> shallower later dip (D-056)
    assert rebound < 1.0
    g = [simulate(t, initial_state(80.0, t), z, z, a, 5.0, 1, bnd)[:, 2] for t in (th0, th1)]
    assert g[0].max() > g[1].max()  # the original kernel pushes up below basal; D-056 does not
    np.testing.assert_array_equal(th1[:12], th0)
