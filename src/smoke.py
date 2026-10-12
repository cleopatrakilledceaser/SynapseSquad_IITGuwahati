"""Tiny end-to-end smoke run on a synthetic toy fixture (used by CI).

Builds a synthetic patient list, makes the configured patient-level split and CV folds, and
checks there is no patient overlap. Uses no real data.
"""

from __future__ import annotations

from src.data.splits import (
    assert_no_patient_overlap,
    patient_train_cal_test_split,
    repeated_patient_group_kfold,
)
from src.utils.config import load_config
from src.utils.seed import set_global_seed

SMOKE_N_PATIENTS = 20
SMOKE_STATUSES = ("healthy", "prediabetes", "t2d")


def main() -> None:
    """Run the smoke check and print a one-line summary."""
    base = load_config("base")
    cfg = load_config("splits")
    set_global_seed(base["seed"])
    pids = [f"synthetic_{i:03d}" for i in range(SMOKE_N_PATIENTS)]
    strata = {p: SMOKE_STATUSES[i % len(SMOKE_STATUSES)] for i, p in enumerate(pids)}
    split = patient_train_cal_test_split(pids, cfg["holdout"], cfg["seed"], strata)
    assert_no_patient_overlap(split.train, split.calibration, split.test)
    train_ids = sorted(split.train)
    folds = list(
        repeated_patient_group_kfold(
            train_ids, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], cfg["seed"]
        )
    )
    print(
        f"smoke OK: train={len(split.train)} cal={len(split.calibration)} "
        f"test={len(split.test)} folds={len(folds)} (synthetic)"
    )


if __name__ == "__main__":
    main()
