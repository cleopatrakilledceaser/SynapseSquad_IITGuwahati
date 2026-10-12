"""Tests proving patient-level splits never leak a patient across partitions."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from src.data.splits import (
    assert_no_patient_overlap,
    assign_rows,
    patient_train_cal_test_split,
    repeated_patient_group_kfold,
)
from src.utils.config import load_config

SPLITS_CFG = load_config("splits")
FRACTIONS = SPLITS_CFG["holdout"]
SEED = SPLITS_CFG["seed"]


def _strata(static: pd.DataFrame) -> dict[str, str]:
    return dict(zip(static["patient_id"], static["diabetes_status"], strict=True))


@pytest.mark.parametrize("stratified", [False, True])
def test_holdout_zero_overlap_and_full_coverage(toy_static: pd.DataFrame, stratified: bool) -> None:
    strata = _strata(toy_static) if stratified else None
    split = patient_train_cal_test_split(toy_static["patient_id"], FRACTIONS, SEED, strata)
    assert not split.train & split.calibration
    assert not split.train & split.test
    assert not split.calibration & split.test
    assert split.train | split.calibration | split.test == set(toy_static["patient_id"])
    assert len(split.calibration) > 0 and len(split.test) > 0


def test_holdout_rows_follow_patients(toy_timeseries: pd.DataFrame) -> None:
    split = patient_train_cal_test_split(toy_timeseries["patient_id"], FRACTIONS, SEED)
    labels = assign_rows(toy_timeseries, split)
    per_patient = labels.groupby(toy_timeseries["patient_id"]).nunique()
    assert (per_patient == 1).all(), "a patient's rows landed in more than one split"
    sets = {name: set(toy_timeseries.loc[labels == name, "patient_id"]) for name in split.as_dict()}
    assert_no_patient_overlap(*sets.values())


def test_holdout_deterministic(toy_static: pd.DataFrame) -> None:
    a = patient_train_cal_test_split(toy_static["patient_id"], FRACTIONS, SEED)
    b = patient_train_cal_test_split(toy_static["patient_id"], FRACTIONS, SEED)
    c = patient_train_cal_test_split(toy_static["patient_id"], FRACTIONS, SEED + 1)
    assert a == b
    assert a != c


def test_holdout_stratification_preserves_classes(toy_static: pd.DataFrame) -> None:
    strata = _strata(toy_static)
    split = patient_train_cal_test_split(toy_static["patient_id"], FRACTIONS, SEED, strata)
    for part in split.as_dict().values():
        assert {strata[p] for p in part} == set(strata.values())


@pytest.mark.parametrize(
    "bad",
    [
        {"train": 0.5, "calibration": 0.2, "test": 0.2},
        {"train": 0.0, "calibration": 0.5, "test": 0.5},
        {"train": 1.2, "calibration": -0.1, "test": -0.1},
        {"train": 0.8, "test": 0.2},
    ],
)
def test_holdout_rejects_bad_fractions(toy_static: pd.DataFrame, bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        patient_train_cal_test_split(toy_static["patient_id"], bad, SEED)


@pytest.mark.parametrize("stratified", [False, True])
def test_repeated_kfold_disjoint_and_each_patient_validated_once(
    toy_static: pd.DataFrame, stratified: bool
) -> None:
    cv = SPLITS_CFG["cv"]
    strata = _strata(toy_static) if stratified else None
    folds = list(
        repeated_patient_group_kfold(
            toy_static["patient_id"], cv["n_splits"], cv["n_repeats"], SEED, strata
        )
    )
    assert len(folds) == cv["n_splits"] * cv["n_repeats"]
    all_pids = set(toy_static["patient_id"])
    for r in range(cv["n_repeats"]):
        repeat_folds = [f for f in folds if f.repeat == r]
        counts = Counter(p for f in repeat_folds for p in f.val)
        assert set(counts) == all_pids
        assert set(counts.values()) == {1}
        for f in repeat_folds:
            assert not f.train & f.val
            assert f.train | f.val == all_pids
    first_val = [f.val for f in folds if f.repeat == 0]
    second_val = [f.val for f in folds if f.repeat == 1]
    assert first_val != second_val, "repeats should reshuffle patients"


def test_kfold_rejects_bad_k(toy_static: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        list(repeated_patient_group_kfold(toy_static["patient_id"], 1, 1, SEED))
    with pytest.raises(ValueError):
        list(repeated_patient_group_kfold(toy_static["patient_id"], 10_000, 1, SEED))


def test_overlap_guard_detects_leak() -> None:
    with pytest.raises(ValueError, match="Patient leakage"):
        assert_no_patient_overlap({"a", "b"}, {"c"}, {"b"})


def test_assign_rows_rejects_unknown_patient(toy_timeseries: pd.DataFrame) -> None:
    split = patient_train_cal_test_split(["toy_000", "toy_001", "toy_002"], FRACTIONS, SEED)
    with pytest.raises(ValueError):
        assign_rows(toy_timeseries, split)
