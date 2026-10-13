"""Part 3 tests: delta reconstruction, group dropping, causal fingersticks and sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.fusion_data import (
    config_columns,
    drop_groups,
    fingerstick_features,
    resolve_groups,
)
from src.models.lgbm import LGBMForecaster
from src.models.neural import CHANNELS, build_sequences
from src.utils.config import load_config
from tests.test_features_causal import STEP, _grid, _meals

FCFG = load_config("fusion")


def _cols(cohort: str) -> list[str]:
    """Synthetic table columns mimicking the real feature tables."""
    base = [
        "cgm_now",
        "cgm_lag_5",
        "cgm_slope_30",
        "tod_sin",
        "tod_cos",
        "days_since_start",
        "hr_mean_30",
        "act_mean_30",
        "act_post_meal",
        "rest_proxy_prev_night_frac",
        "meal_carbs_g_1h",
        "meal_minutes_since_last",
    ]
    stat = [
        c for g, s in FCFG["groups"][cohort].items() if g.startswith("static_") for c in s["static"]
    ]
    return base + [f"static_{c}" for c in stat] + ["y_30", "patient_id"]


def test_delta_reconstruction_equals_absolute() -> None:
    rng = np.random.default_rng(0)
    tab = pd.DataFrame(
        {
            "cgm_now": rng.uniform(60, 300, 50),
            "fs_last": rng.uniform(60, 300, 50),
            "y_60": rng.uniform(60, 300, 50),
        }
    )
    tab.loc[3, "y_60"] = np.nan
    for base in ("cgm_now", "fs_last"):
        m = LGBMForecaster(["cgm_now"], {}, target="delta", base=base)
        enc = m.encode(tab, 60)
        assert np.allclose(m.decode(tab, enc), tab["y_60"], equal_nan=True)
        assert np.isnan(enc[3])
    a = LGBMForecaster(["cgm_now"], {}, target="abs")
    assert np.allclose(a.decode(tab, a.encode(tab, 60)), tab["y_60"], equal_nan=True)


@pytest.mark.parametrize("cohort", ["cgmacros", "shanghai_t2dm"])
def test_drop_group_removes_exactly_that_group(cohort: str) -> None:
    cols = _cols(cohort)
    groups = resolve_groups(cohort, FCFG, cols)
    full = config_columns(FCFG["ablation"][cohort]["E_full"], groups)
    for name, drop in FCFG["logo"][cohort].items():
        kept = drop_groups(full, drop, groups)
        removed = set(full) - set(kept)
        expected = {c for g in drop for c in groups[g]}
        assert removed == expected, name
        assert expected, f"{name} drops nothing"
    if cohort == "cgmacros":
        assert "act_post_meal" in groups["meals"]  # post-meal activity belongs to meals
        assert "act_post_meal" not in groups["wearable"]
    else:
        assert not [c for c in full if c.startswith(("meal_", "hr_", "act_"))]


def test_groups_are_disjoint_and_cover_static() -> None:
    cols = _cols("cgmacros")
    g = resolve_groups("cgmacros", FCFG, cols)
    stat = [c for c in cols if c.startswith("static_")]
    assert sorted(stat) == sorted(c for k, v in g.items() if k.startswith("static_") for c in v)


def test_fingersticks_never_use_future_readings() -> None:
    g = _grid(n=2000)
    anchors = g.loc[g.glucose_is_real & (g.index > 400), ["period_id", "t"]].iloc[::7]
    times = FCFG["fingerstick"]["schedules"]["fs4"]
    base = fingerstick_features(g, anchors, times, 24)
    assert base["fs_last"].notna().mean() > 0.9
    assert (base["fs_minutes_since"].dropna() >= 0).all()
    for cut_i in (600, 1100, 1500):
        cut_t = g["t"].iloc[cut_i]
        g2 = g.copy()
        g2.loc[g2.t > cut_t, "glucose"] += 99
        pert = fingerstick_features(g2, anchors, times, 24)
        past = (anchors["t"] <= cut_t).to_numpy()
        pd.testing.assert_frame_equal(base[past], pert[past])
    # the last fingerstick equals the real CGM value at a scheduled clock time <= t
    a = anchors.iloc[50]
    sched = {(int(s[:2]), int(s[3:])) for s in times}
    prior = g[(g.t <= a.t) & g.glucose_is_real & g.t.map(lambda x: (x.hour, x.minute) in sched)]
    assert base.loc[a.name, "fs_last"] == prior.iloc[-1]["glucose"]


def test_sequences_are_causal() -> None:
    g = _grid(n=1500)
    m = _meals(g)
    anchors = g.loc[g.glucose_is_real & (g.index > 200), ["period_id", "t"]].iloc[::11]
    anchors = anchors.reset_index(drop=True)
    chans = CHANNELS["cgm"] + CHANNELS["wearable"] + CHANNELS["meals"] + CHANNELS["tod"]
    base = build_sequences(g, m, anchors, 4, STEP, 15, chans)
    for cut_i in (500, 900):
        cut_t = g["t"].iloc[cut_i]
        g2, m2 = g.copy(), m.copy()
        fut = g2.t > cut_t
        g2.loc[fut, "glucose"] += 77
        g2.loc[fut, "hr"] = 200.0
        m2.loc[m2.t > cut_t, "carbs_g"] = 999.0
        pert = build_sequences(g2, m2, anchors, 4, STEP, 15, chans)
        past = (anchors["t"] <= cut_t).to_numpy()
        assert np.array_equal(base[past], pert[past], equal_nan=True)
    # last time step is the anchor row itself
    i = 5
    row = g[g.t == anchors.t[i]].iloc[0]
    assert base[i, -1, 0] == row.glucose
