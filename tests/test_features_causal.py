"""Leakage tests for windows, labels and features (synthetic data; run in CI).

(b) causality: perturb every input stream strictly AFTER time t; every feature at times <= t must
    be unchanged (all feature groups: CGM, time, meals, wearables, rest_proxy).
(c) targets come only from real readings (never carried / missing).
(d) no calendar features in any feature set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import is_calendar_column
from src.features.build import period_features
from src.features.make_features import feature_sets
from src.features.windows import build_anchor_table, period_targets
from src.utils.config import load_config

CFG = load_config("features")
STEP = 5


def _grid(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2021-03-01 06:00", periods=n, freq=f"{STEP}min")
    glu = np.round(130 + 40 * np.sin(np.arange(n) / 30) + rng.normal(0, 4, n))
    real = rng.uniform(size=n) > 0.1
    carried = ~real & (rng.uniform(size=n) > 0.5)
    missing = ~real & ~carried
    g = pd.DataFrame(
        {
            "patient_id": "p1",
            "period_id": "p1",
            "dataset": "cgmacros",
            "t": t,
            "glucose": np.where(missing, np.nan, glu),
            "glucose_is_real": real,
            "glucose_is_carried": carried,
            "glucose_missing": missing,
            "minutes_since_real": np.where(real, 0.0, 10.0),
            "hr": 70 + 10 * rng.normal(size=n),
            "hr_missing": False,
            "activity_intensity": rng.integers(0, 4, n).astype(float),
            "rest_proxy": pd.array(rng.uniform(size=n) > 0.5, dtype="boolean"),
        }
    )
    g["minutes_since_start"] = np.arange(n) * STEP
    g["days_since_start"] = g["minutes_since_start"] / 1440
    return g


def _meals(g: pd.DataFrame) -> pd.DataFrame:
    idx = [50, 200, 420, 700, 900, 1100]
    return pd.DataFrame(
        {
            "t": g["t"].iloc[idx].to_numpy(),
            "meal_type": ["breakfast", "lunch", "dinner", "snack", "breakfast", "lunch"],
            "carbs_g": [40.0, 60, 80, 20, 30, 50],
            "protein_g": [20.0, 30, 25, 5, 10, 20],
            "fat_g": [10.0, 20, 30, 5, 10, 15],
            "fiber_g": [2.0, 5, 6, 1, 2, 4],
            "kcal": [350.0, 600, 700, 150, 250, 450],
            "amount_gt100": [False, False, True, False, False, False],
            "food_grams_total": np.nan,
            "period_id": "p1",
        }
    )


@pytest.mark.parametrize("n_hours", [2, 4, 6])
def test_features_unchanged_by_future_perturbation(n_hours: int) -> None:
    g, m = _grid(), None
    m = _meals(g)
    base = period_features(g, m, STEP, n_hours, CFG)
    rng = np.random.default_rng(7)
    for cut in (300, 600, 901):
        g2, m2 = g.copy(), m.copy()
        fut = g2.index > cut
        g2.loc[fut, "glucose"] = g2.loc[fut, "glucose"] + 77
        g2.loc[fut, "glucose_is_real"] = rng.uniform(size=fut.sum()) > 0.5
        g2.loc[fut, "hr"] = 200.0
        g2.loc[fut, "activity_intensity"] = 3.0
        g2.loc[fut, "rest_proxy"] = ~g2.loc[fut, "rest_proxy"]
        late = m2["t"] > g["t"].iloc[cut]
        m2.loc[late, "carbs_g"] = 999.0
        m2.loc[late, "t"] = m2.loc[late, "t"] - pd.Timedelta(minutes=1)  # still after t
        extra = pd.DataFrame({**m2.iloc[[0]].to_dict("list"), "t": [g["t"].iloc[cut + 1]]})
        m2 = pd.concat([m2, extra], ignore_index=True)
        pert = period_features(g2, m2, STEP, n_hours, CFG)
        past = g.index <= cut
        groups = {
            "cgm": [c for c in base if c.startswith("cgm_")],
            "time": ["tod_sin", "tod_cos", "days_since_start"],
            "meal": [c for c in base if c.startswith("meal_")],
            "wearable": [c for c in base if c.startswith(("hr_", "act_"))],
            "rest_proxy": [c for c in base if c.startswith("rest_proxy")],
        }
        for name, cols in groups.items():
            assert cols, f"no features in group {name}"
            pd.testing.assert_frame_equal(
                base.loc[past, cols],
                pert.loc[past, cols],
                check_exact=False,
                rtol=1e-9,
                obj=f"group {name} at cut {cut}",
            )


def test_causality_test_detects_a_leak() -> None:
    """Negative control: a deliberately leaky feature (centred rolling mean) must be caught."""
    g = _grid()
    leaky = g["glucose"].rolling(7, center=True, min_periods=1).mean()
    g2 = g.copy()
    g2.loc[g2.index > 600, "glucose"] += 77
    leaky2 = g2["glucose"].rolling(7, center=True, min_periods=1).mean()
    past = g.index <= 600
    assert not np.allclose(leaky[past].to_numpy(), leaky2[past].to_numpy(), equal_nan=True)


def test_targets_only_from_real_readings() -> None:
    g = _grid()
    tg = period_targets(g, STEP, CFG)
    real_vals = g["glucose"].where(g["glucose_is_real"]).to_numpy()
    for h in CFG["windows"]["horizons_min"]:
        k = h // STEP
        y = tg[f"y_{h}"].to_numpy()
        src_real = np.r_[g["glucose_is_real"].to_numpy()[k:], np.zeros(k, bool)]
        assert np.all(np.isnan(y[~src_real])), "target taken from a non-real reading"
        assert np.allclose(y[src_real], real_vals[k:][src_real[: len(real_vals) - k]])
    # making every carried value extreme must not change any target or label
    g2 = g.copy()
    g2.loc[g2.glucose_is_carried, "glucose"] = 500.0
    pd.testing.assert_frame_equal(tg, period_targets(g2, STEP, CFG))


def test_anchor_requires_real_reading_and_counts() -> None:
    g = _grid()
    anchors, counts = build_anchor_table(g, STEP, 15, CFG)
    real_t = set(g.loc[g.glucose_is_real, "t"])
    assert set(anchors["t"]).issubset(real_t)
    assert counts["candidates_on_stride"] == sum(
        v for k, v in counts.items() if k.startswith("dropped")
    ) + len(anchors)


def test_no_calendar_features_in_sets() -> None:
    g = _grid()
    f = period_features(g, _meals(g), STEP, 4, CFG)
    cols = list(f.columns) + [f"static_{c}" for c in CFG["static"]["cgmacros"]]
    for name, fs in feature_sets(cols, "cgmacros", CFG).items():
        assert not [c for c in fs if is_calendar_column(c)], name
    assert not [c for c in f.columns if is_calendar_column(c)]
