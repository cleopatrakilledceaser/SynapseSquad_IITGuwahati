"""Part 6 robustness tests (synthetic grid): every fault touches only rows at or after tau;
recomputed features and the data-quality score at anchors before tau are unchanged (causal);
calibration patients are never part of a fitting set; ACI/DQ do not look ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import period_features
from src.robustness.faults import inject, refresh_derived
from src.robustness.quality import dq_components
from src.uncertainty.run import check_roles, jobs, roles
from src.utils.config import load_config

UC = load_config("uncertainty")
FC = load_config("features")
STEP = 5
CLIP = (40.0, 405.0)


def _grid(days: float = 9.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic 5-min grid in the Part 1 schema (labeled synthetic)."""
    rng = np.random.default_rng(seed)
    n = int(days * 1440 / STEP)
    t = pd.date_range("2026-01-01", periods=n, freq=f"{STEP}min")
    mins = np.arange(n) * float(STEP)
    glu = 120 + 40 * np.sin(mins / 300.0) + rng.normal(0, 5, n)
    real = rng.uniform(size=n) > 0.05
    g = pd.DataFrame(
        {
            "patient_id": "syn_001",
            "dataset": "synthetic",
            "period_id": "syn_001_p1",
            "t": t,
            "minutes_since_start": mins,
            "days_since_start": mins / 1440.0,
            "glucose": np.where(real, glu, np.nan),
            "glucose_is_real": real,
            "glucose_is_carried": np.zeros(n, bool),
            "hr": rng.normal(70, 5, n),
            "hr_missing": np.zeros(n, bool),
            "activity_intensity": rng.integers(0, 3, n).astype(float),
            "activity_missing": np.zeros(n, bool),
            "rest_proxy": rng.uniform(size=n) > 0.5,
        }
    )
    return refresh_derived(g, STEP)


FAULTS = [(f, float(s["levels"][-1])) for f, s in UC["robustness"]["faults"].items()]


@pytest.mark.parametrize(("fault", "level"), FAULTS)
def test_fault_only_affects_rows_at_or_after_tau(fault: str, level: float):
    g = _grid()
    tau = UC["test_from_day"]
    spec = UC["robustness"]["faults"][fault]
    gf = inject(g, fault, level, tau, STEP, spec, CLIP, seed=1)
    before = g["days_since_start"] < tau
    cols = [
        "glucose",
        "glucose_is_real",
        "glucose_is_carried",
        "minutes_since_real",
        "hr",
        "activity_intensity",
    ]
    pd.testing.assert_frame_equal(g.loc[before, cols], gf.loc[before, cols])
    assert not g.loc[~before, cols].equals(gf.loc[~before, cols])  # the fault did something


@pytest.mark.parametrize(("fault", "level"), FAULTS)
def test_features_and_dq_before_tau_unchanged(fault: str, level: float):
    g = _grid()
    tau = UC["test_from_day"]
    gf = inject(g, fault, level, tau, STEP, UC["robustness"]["faults"][fault], CLIP, seed=2)
    empty = pd.DataFrame(
        columns=[
            "t",
            "meal_type",
            "amount_gt100",
            "carbs_g",
            "protein_g",
            "fat_g",
            "fiber_g",
            "kcal",
            "food_grams_total",
        ]
    )
    f0 = pd.concat(
        [period_features(g, empty, STEP, 6, FC), dq_components(g, STEP, UC["dq"], CLIP)], axis=1
    )
    f1 = pd.concat(
        [period_features(gf, empty, STEP, 6, FC), dq_components(gf, STEP, UC["dq"], CLIP)], axis=1
    )
    before = (g["days_since_start"] < tau).to_numpy()
    pd.testing.assert_frame_equal(f0[before], f1[before])


def test_zero_severity_fault_is_identity():
    """A no-op fault must leave the grid (incl. raw-time minutes_since_real) unchanged."""
    g = _grid(seed=5)
    g["minutes_since_real"] = np.where(g["glucose_is_real"], 1.0, g["minutes_since_real"] + 1.0)
    for fault in ("noise", "bias", "drift"):
        gf = inject(g, fault, 0.0, 7.0, STEP, UC["robustness"]["faults"][fault], CLIP, seed=3)
        pd.testing.assert_frame_equal(g.reset_index(drop=True), gf, check_dtype=False)


def test_dq_score_is_causal_and_drops_with_gaps():
    g = _grid(seed=4)
    d0 = dq_components(g, STEP, UC["dq"], CLIP)
    cut = 1500
    g2 = g.copy()
    g2.loc[cut + 1 :, "glucose"] = 999.0  # future values changed wildly
    d1 = dq_components(refresh_derived(g2, STEP), STEP, UC["dq"], CLIP)
    pd.testing.assert_frame_equal(d0.iloc[: cut + 1], d1.iloc[: cut + 1])
    assert d0["dq"].between(0, 1).all()
    gg = inject(g, "gaps", 240.0, 7.0, STEP, UC["robustness"]["faults"]["gaps"], CLIP, 0)
    d2 = dq_components(gg, STEP, UC["dq"], CLIP)
    after = (g["days_since_start"] >= 7).to_numpy()
    assert d2["dq"][after].mean() < d0["dq"][after].mean()


@pytest.mark.parametrize("design", ["A", "B"])
def test_calibration_patients_never_in_fit_sets(design: str):
    from pathlib import Path

    splits = Path(__file__).resolve().parents[1] / "data" / "processed" / "splits"
    if not (splits / "cgmacros_cv.csv").exists():
        pytest.skip("split files not built (run `make data`)")
    for cohort in ("cgmacros", "shanghai_t2dm"):
        for d, c, f, tr, te in jobs([design], [cohort], UC):
            r = roles(d, c, f, tr, te)
            check_roles(r)
            assert r["cal"], "split arm needs calibration patients"
            assert not set(r["cal"]) & set(r["proper"])
            assert not set(r["test"]) & (set(r["cvplus"]) | set(r["cal"]))


def test_check_roles_raises_on_overlap():
    with pytest.raises(AssertionError):
        check_roles({"proper": ["a", "b"], "cal": ["b"], "cvplus": ["a", "b"], "test": ["c"]})
