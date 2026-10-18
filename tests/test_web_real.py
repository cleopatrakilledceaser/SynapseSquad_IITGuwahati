"""D-069 forecasts on a SIMULATED patient: ordered bands, probabilities, monotone, determinism."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.web import real_forecast as rf

pytestmark = pytest.mark.skipif(
    not (rf.APP / "context.json").exists(), reason="needs the app bundle (make app-data)"
)

MEALS = ((8 * 60, "breakfast", 60.0), (13 * 60, "lunch", 80.0), (20 * 60, "dinner", 70.0))


def sim_patient(days: int = 3, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """A small SIMULATED record in the cohort table's layout (5-min grid, three meals a day)."""
    rng = np.random.default_rng(seed)
    n = days * 288
    t = pd.Timestamp("2026-01-05") + pd.to_timedelta(np.arange(n) * 5, unit="m")
    minute = (np.arange(n) * 5) % 1440
    g = 140 + 25 * np.sin(2 * np.pi * minute / 1440) + rng.normal(0, 4, n)
    grid = pd.DataFrame(
        {
            "patient_id": "sim",
            "period_id": "sim",
            "t": t,
            "days_since_start": np.arange(n) * 5 / 1440,
            "glucose": g,
            "glucose_is_real": True,
            "glucose_is_carried": False,
            "glucose_missing": False,
            "minutes_since_real": 0.0,
            "hr": 75 + rng.normal(0, 3, n),
            "hr_missing": False,
            "activity_intensity": np.abs(rng.normal(0.3, 0.2, n)),
            "activity_missing": False,
            "rest_proxy": False,
        }
    )
    rows = []
    for d in range(days):
        for mm, kind, carbs in MEALS:
            rows.append(
                {
                    "patient_id": "sim",
                    "period_id": "sim",
                    "t": t[0] + pd.Timedelta(minutes=d * 1440 + mm),
                    "meal_type": kind,
                    "carbs_g": carbs,
                    "protein_g": 0.2 * carbs,
                    "fat_g": 0.25 * carbs,
                    "fiber_g": 0.1 * carbs,
                    "kcal": 4 * 1.2 * carbs + 9 * 0.25 * carbs,
                    "amount_gt100": False,
                    "amount_missing": False,
                    "macros_available": True,
                    "food_grams_total": np.nan,
                    "preset_label": "test meal",
                }
            )
    now = t[0] + pd.Timedelta(minutes=(days - 1) * 1440 + 22 * 60)
    return grid, pd.DataFrame(rows), now


@pytest.fixture(scope="module")
def shipped() -> rf.Shipped:
    return rf.load_shipped()


def run(sh: rf.Shipped) -> dict:
    grid, meals, now = sim_patient()
    return rf.patient(sh, "sim", grid, meals, 6.9, 128.0, now, rf.preset_list()[:2])


@pytest.fixture(scope="module")
def out(shipped: rf.Shipped) -> dict:
    return run(shipped)


def test_bands_ordered_and_clamped(out, shipped):
    lo_c, hi_c = shipped.clamp
    for h in ("30", "60", "120"):
        f = out["fc"][h]
        assert lo_c <= f["lo"] <= f["p"] <= f["hi"] <= hi_c, (h, f)


def test_event_probabilities_in_unit_interval_and_none_invented(out, shipped):
    assert out["ev"], "no event probability computed"
    assert all(0.0 <= v <= 1.0 for v in out["ev"].values())
    assert not set(out["ev"]) & set(rf.missing_events(shipped))


def test_monotone_model_non_decreasing_in_carbs(out):
    wins = out["whatif"]["windows"]
    assert wins, "no what-if window"
    for w in wins:
        assert w["m"] >= rf.WINDOW_LEAD_MIN
        for h, (no_walk, walk) in w["models"]["mono"]["pred"].items():
            assert len(no_walk) == len(walk) == len(rf.CARBS_GRID)
            assert np.all(np.diff(no_walk) >= 0) and np.all(np.diff(walk) >= 0), (h, w["label"])


def test_deterministic(out, shipped):
    again = run(shipped)
    assert json.dumps(again, sort_keys=True) == json.dumps(out, sort_keys=True)
