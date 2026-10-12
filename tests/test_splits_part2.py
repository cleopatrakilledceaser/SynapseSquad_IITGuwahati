"""Split tests on the saved Part 2 assignments (skipped when splits are not built)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.splits import chronological_within_patient

SPL = Path(__file__).resolve().parents[1] / "data" / "processed" / "splits"
needs = pytest.mark.skipif(not (SPL / "cgmacros_holdout.csv").exists(), reason="run make features")
COHORTS = ("cgmacros", "shanghai_t2dm")


@needs
@pytest.mark.parametrize("cohort", COHORTS)
def test_holdout_no_patient_overlap(cohort: str) -> None:
    h = pd.read_csv(SPL / f"{cohort}_holdout.csv")
    assert h.patient_id.is_unique
    parts = [set(h.loc[h.split == s, "patient_id"]) for s in ("train", "calibration", "test")]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])


@needs
@pytest.mark.parametrize("cohort", COHORTS)
def test_cv_each_patient_validated_once_per_repeat(cohort: str) -> None:
    cv = pd.read_csv(SPL / f"{cohort}_cv.csv")
    assert (cv.groupby(["repeat", "patient_id"]).size() == 1).all()
    n = cv.patient_id.nunique()
    assert (cv.groupby("repeat").patient_id.nunique() == n).all()


@needs
def test_shanghai_multi_period_patients_stay_together() -> None:
    chrono = pd.read_parquet(SPL / "shanghai_t2dm_chrono.parquet")
    multi = chrono.groupby("patient_id").period_id.nunique()
    multi = multi[multi > 1].index
    assert len(multi) == 8
    h = pd.read_csv(SPL / "shanghai_t2dm_holdout.csv").set_index("patient_id")["split"]
    cv = pd.read_csv(SPL / "shanghai_t2dm_cv.csv")
    for p in multi:
        assert p in h.index  # split unit is the patient, so all periods share one assignment
        assert (cv.patient_id == p).sum() == cv.repeat.nunique()
    rows = chrono.assign(split=chrono.patient_id.map(h))
    assert rows.groupby("patient_id").split.nunique().max() == 1


def test_chronological_adapt_precedes_eval() -> None:
    df = pd.DataFrame(
        {
            "patient_id": ["a"] * 6 + ["b"] * 4,
            "period_id": ["a_0"] * 4 + ["a_1"] * 2 + ["b_0"] * 4,
            "days_since_start": [0.1, 1.0, 2.5, 3.0, 0.1, 0.5, 0.2, 1.9, 2.1, 5.0],
        }
    )
    lab = chronological_within_patient(df, 2)
    assert lab.tolist() == [
        "adapt",
        "adapt",
        "eval",
        "eval",
        "eval",
        "eval",
        "adapt",
        "adapt",
        "eval",
        "eval",
    ]
