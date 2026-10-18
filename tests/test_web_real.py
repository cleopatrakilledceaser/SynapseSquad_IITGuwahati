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
    """Calibrated outputs of the 5-day simulated patient (band from its own calibration days)."""
    grid, meals, now = sim_patient(days=5)
    r = rf.patient(sh, "sim", grid, meals, 6.9, 128.0, now, rf.preset_list()[:2])
    q = rf.band_quantiles([r["residuals"]])
    r["fc"], r["ev"] = rf.now_outputs(sh, r, q)
    r["q"] = q
    return r


@pytest.fixture(scope="module")
def out(shipped: rf.Shipped) -> dict:
    return run(shipped)


def test_calibrated_bands_ordered_and_clamped(out, shipped):
    lo_c, hi_c = shipped.clamp
    for h in ("30", "60", "120"):
        f = out["fc"][h]
        assert lo_c <= f["lo"] <= f["p"] <= f["hi"] <= hi_c, (h, f)
    assert all(lo <= 0 <= hi for lo, hi in out["q"].values())


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


def test_offset_uses_days_1_to_3_only(out):
    tab = out["table"]
    base = rf.offsets(tab)
    assert any(abs(v) > 0 for o in base.values() for v in o.values())
    later = tab.copy()
    days = later["days_since_start"].to_numpy(float)
    for h in rf.HZ:
        outside = days + h / 1440 >= rf.K_DAYS  # reading after day 3: not a calibration pair
        later.loc[outside, f"y_{h}"] += 100.0
        for name in rf.FORMS:
            later.loc[days >= rf.K_DAYS, f"pred_{name}_{h}"] -= 50.0
    assert rf.offsets(later) == base
    early = tab.copy()
    early.loc[days < 1.0, "y_60"] += 100.0  # a calibration-day reading does move the offset
    assert all(rf.offsets(early)[n][60] != base[n][60] for n in rf.FORMS)


def test_forecast_eval_table(out, shipped):
    q, off = out["q"], {"sim": out["offsets"]}
    res = rf.evaluate(rf.scored([out["table"]], off, q, shipped.clamp))
    t = rf.eval_table(res)
    assert [r[0] for r in t["rows"]] == list(rf.METHODS.values())
    assert all(len(r) == len(t["cols"]) == 4 for r in t["rows"])
    assert "Simulated" in t["note"] and "simulated" in t["title"]


def test_deterministic(out, shipped):
    again = run(shipped)

    def key(r: dict) -> str:
        r = {k: v for k, v in r.items() if k not in ("table", "residuals")}
        return json.dumps(r, sort_keys=True, default=str)

    assert key(again) == key(out)
    assert again["table"].equals(out["table"])
