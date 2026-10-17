"""Part 8d (D-064): sleep channel keeps the legacy simulation intact and acts only as specified."""

from __future__ import annotations

import numpy as np
import pytest

from src.india import cohort as ic
from src.india import sleep as sl
from src.utils.config import PROJECT_ROOT, load_config


def _need_inputs():
    if not (PROJECT_ROOT / "reports" / "metrics" / "mech_fitted_params.csv").exists():
        pytest.skip("mechanistic fits not available")


def gen(mult, n=3, days=4, seed=5):
    icfg, mc = load_config("india"), load_config("mech")
    return ic.generate(icfg, mc, 1.0, "t", seed, n_patients=n, days=days, sleep_mult=mult)


def test_zero_multiplier_reproduces_legacy_simulation_exactly():
    _need_inputs()
    a, b = gen(None), gen(0.0)
    assert np.array_equal(
        a["grid"]["glucose_true"].to_numpy(), b["grid"]["glucose_true"].to_numpy()
    )
    assert np.array_equal(a["grid"]["glucose"].to_numpy(), b["grid"]["glucose"].to_numpy())
    assert "sleep" not in a and "steps" not in a["grid"].columns
    assert "steps" in b["grid"].columns and len(b["sleep"]) == 3 * 4


def test_positive_multiplier_lowers_sensitivity_and_only_after_first_wake():
    _need_inputs()
    a, b = gen(0.0, n=4, days=5), gen(4.0, n=4, days=5)
    assert (b["sleep"]["si_log_shift"] <= 0).all()
    assert (a["sleep"]["sleep_h"].to_numpy() == b["sleep"]["sleep_h"].to_numpy()).all()  # paired
    scfg = load_config("india")["sleep"]
    first = sl.wake_steps(1, scfg)[0]
    for pid, ga in a["grid"].groupby("patient_id"):
        gb = b["grid"][b["grid"].patient_id == pid]
        assert np.array_equal(
            ga["glucose_true"].to_numpy()[:first], gb["glucose_true"].to_numpy()[:first]
        )
    # lower insulin sensitivity: glucose is higher on average over the cohort
    assert b["grid"]["glucose_true"].mean() > a["grid"]["glucose_true"].mean()


def test_draws_in_range_and_deficit_never_negative():
    scfg = load_config("india")["sleep"]
    n = sl.draw_nights(np.random.default_rng(0), 500, scfg)
    assert (
        n["sleep_h"].min() >= scfg["duration"]["lo"]
        and n["sleep_h"].max() <= scfg["duration"]["hi"]
    )
    assert (n["hrv_rmssd"] > 0).all()
    shift = sl.si_shift(np.array([5.0, 7.0, 9.0]), scfg, 2.0)
    assert shift[0] < 0 and shift[1] == 0 and shift[2] == 0


def test_hrv_is_lower_after_short_sleep_on_average():
    scfg = load_config("india")["sleep"]
    n = sl.draw_nights(np.random.default_rng(1), 4000, scfg)
    short, long_ = n["hrv_rmssd"][n["sleep_h"] < 6], n["hrv_rmssd"][n["sleep_h"] > 7.5]
    assert short.mean() < long_.mean()


def test_steps_zero_when_no_activity_and_scale_with_activity():
    scfg = load_config("india")["sleep"]
    act = np.array([0.0, 0.0, 1.0, 2.0])
    s = sl.steps_from_activity(np.random.default_rng(0), act, scfg)
    assert s[0] == 0 and s[1] == 0 and s[3] > s[2] > 0


def test_sleep_features_are_causal_and_aligned():
    _need_inputs()
    from src.india import stress
    from src.india import wearable_power as wp

    icfg, mc, fcfg = load_config("india"), load_config("mech"), load_config("features")
    c = ic.generate(icfg, mc, 1.0, "t", 5, n_patients=2, days=5, sleep_mult=1.0)
    scfg = {**icfg["stress"], "n_hours": 6}
    tab = stress.synthetic_table(c, icfg, fcfg, scfg)
    out = wp.add_sleep_features(tab, c)
    assert len(out) == len(tab) and (out["t"].to_numpy() == tab["t"].to_numpy()).all()
    nights = c["sleep"]
    for pid, d in out.groupby("patient_id"):
        nn = nights[nights.patient_id == pid].sort_values("wake_t")
        first = nn["wake_t"].iloc[0]
        assert d.loc[d["t"] < first, "sleep_h_prev"].isna().all()  # nothing before the first wake
        late = d[d["t"] >= nn["wake_t"].iloc[1]].iloc[0]
        assert late["sleep_h_prev"] == pytest.approx(nn["sleep_h"].iloc[1])
    sets = wp.feature_sets(list(out.columns))
    assert set(sets["A"]) < set(sets["B"]) < set(sets["C"]) and "steps_1h" in sets["C"]
    assert not any(x.startswith(("y_", "hyper", "hypo")) for x in sets["C"])


def test_folds_cover_all_patients_once():
    from src.india import wearable_power as wp

    pids = [f"p{i}" for i in range(23)]
    f = wp.folds_of(pids, 5, 1)
    assert sorted(x for fold in f for x in fold) == sorted(pids) and len(f) == 5
