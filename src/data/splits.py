"""Patient-level data splitting.

Every split in GlucoTwin is made over *unique patient IDs*, never over rows, so no patient's
data can appear in more than one of train / calibration / test, or on both sides of a CV fold.
Row-level frames are mapped onto patient-level splits with :func:`assign_rows`.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

SPLIT_NAMES: tuple[str, str, str] = ("train", "calibration", "test")


@dataclass(frozen=True)
class PatientSplit:
    """Disjoint sets of patient IDs for train, calibration and test."""

    train: frozenset[Hashable]
    calibration: frozenset[Hashable]
    test: frozenset[Hashable]

    def as_dict(self) -> dict[str, frozenset[Hashable]]:
        """Return the split as ``{split_name: patient_ids}``."""
        return {"train": self.train, "calibration": self.calibration, "test": self.test}


@dataclass(frozen=True)
class CVFold:
    """One fold of repeated patient-grouped cross-validation."""

    repeat: int
    fold: int
    train: frozenset[Hashable]
    val: frozenset[Hashable]


def assert_no_patient_overlap(*groups: Sequence[Hashable] | frozenset[Hashable]) -> None:
    """Raise ``ValueError`` if any patient ID appears in more than one group.

    Args:
        *groups: Collections of patient IDs (e.g. train, calibration, test).

    Raises:
        ValueError: If two groups share at least one patient ID.
    """
    sets = [set(g) for g in groups]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            shared = sets[i] & sets[j]
            if shared:
                sample = sorted(map(str, shared))[:5]
                raise ValueError(
                    f"Patient leakage: groups {i} and {j} share {len(shared)} patient(s), "
                    f"e.g. {sample}"
                )


def _unique_patients(patient_ids: Sequence[Hashable]) -> list[Hashable]:
    """Return unique patient IDs in a deterministic (sorted-by-string) order."""
    unique = list(dict.fromkeys(patient_ids))
    if not unique:
        raise ValueError("No patient IDs supplied.")
    return sorted(unique, key=str)


def _validate_fractions(fractions: Mapping[str, float]) -> tuple[float, float, float]:
    """Validate split fractions and return them as (train, calibration, test)."""
    missing = [k for k in SPLIT_NAMES if k not in fractions]
    if missing:
        raise ValueError(f"Missing split fractions: {missing}")
    tr, cal, te = (float(fractions[k]) for k in SPLIT_NAMES)
    if min(tr, cal, te) < 0:
        raise ValueError(f"Split fractions must be non-negative, got {(tr, cal, te)}")
    if tr <= 0:
        raise ValueError("Train fraction must be > 0.")
    if not np.isclose(tr + cal + te, 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {tr + cal + te:.4f}")
    return tr, cal, te


def _allocate_counts(n: int, fracs: tuple[float, float, float]) -> tuple[int, int, int]:
    """Split ``n`` items into integer counts by largest-remainder rounding.

    Guarantees at least one training item whenever ``n >= 1``.
    """
    raw = np.array(fracs) * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for idx in order[:remainder]:
        counts[idx] += 1
    if n >= 1 and counts[0] == 0:
        donor = int(np.argmax(counts))
        counts[donor] -= 1
        counts[0] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def patient_train_cal_test_split(
    patient_ids: Sequence[Hashable],
    fractions: Mapping[str, float],
    seed: int,
    strata: Mapping[Hashable, Hashable] | None = None,
) -> PatientSplit:
    """Split unique patients into disjoint train / calibration / test sets.

    Args:
        patient_ids: Patient IDs (duplicates allowed, e.g. one per row).
        fractions: Mapping with keys ``train``, ``calibration``, ``test`` summing to 1.
        seed: Random seed for reproducibility.
        strata: Optional ``{patient_id: label}`` (e.g. diabetes status). Allocation is then
            done within each stratum so class proportions are preserved.

    Returns:
        A :class:`PatientSplit` whose three sets are disjoint and cover all patients.
    """
    fracs = _validate_fractions(fractions)
    patients = _unique_patients(patient_ids)
    rng = np.random.default_rng(seed)

    groups: dict[Hashable, list[Hashable]] = {}
    if strata is None:
        groups["__all__"] = patients
    else:
        missing = [p for p in patients if p not in strata]
        if missing:
            raise ValueError(f"Strata missing for {len(missing)} patient(s), e.g. {missing[:5]}")
        for p in patients:
            groups.setdefault(strata[p], []).append(p)

    out: dict[str, list[Hashable]] = {name: [] for name in SPLIT_NAMES}
    for label in sorted(groups, key=str):
        members = list(groups[label])
        rng.shuffle(members)
        n_tr, n_cal, _ = _allocate_counts(len(members), fracs)
        out["train"].extend(members[:n_tr])
        out["calibration"].extend(members[n_tr : n_tr + n_cal])
        out["test"].extend(members[n_tr + n_cal :])

    split = PatientSplit(
        train=frozenset(out["train"]),
        calibration=frozenset(out["calibration"]),
        test=frozenset(out["test"]),
    )
    assert_no_patient_overlap(split.train, split.calibration, split.test)
    return split


def repeated_patient_group_kfold(
    patient_ids: Sequence[Hashable],
    n_splits: int,
    n_repeats: int,
    seed: int,
    strata: Mapping[Hashable, Hashable] | None = None,
) -> Iterator[CVFold]:
    """Yield repeated K-fold splits over unique patients (patient-grouped CV).

    Intended for tuning on the *training* patients only. Within each repeat every patient is
    on the validation side of exactly one fold.

    Args:
        patient_ids: Patient IDs (duplicates allowed).
        n_splits: Number of folds K (``2 <= K <= n_patients``).
        n_repeats: Number of independent reshuffled repeats (``>= 1``).
        seed: Base random seed; repeat ``r`` uses ``seed + r``.
        strata: Optional ``{patient_id: label}`` for stratified folds.

    Yields:
        :class:`CVFold` objects with disjoint ``train`` / ``val`` patient sets.
    """
    patients = _unique_patients(patient_ids)
    if n_splits < 2 or n_splits > len(patients):
        raise ValueError(f"n_splits must be in [2, {len(patients)}], got {n_splits}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    arr = np.array(patients, dtype=object)
    labels = None if strata is None else np.array([str(strata[p]) for p in patients])

    for r in range(n_repeats):
        if labels is None:
            iterator = KFold(n_splits=n_splits, shuffle=True, random_state=seed + r).split(arr)
        else:
            iterator = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed + r
            ).split(arr, labels)
        for f, (tr_idx, va_idx) in enumerate(iterator):
            fold = CVFold(
                repeat=r,
                fold=f,
                train=frozenset(arr[tr_idx].tolist()),
                val=frozenset(arr[va_idx].tolist()),
            )
            assert_no_patient_overlap(fold.train, fold.val)
            yield fold


def assign_rows(
    df: pd.DataFrame, split: PatientSplit, patient_col: str = "patient_id"
) -> pd.Series:
    """Map each row of ``df`` to its patient's split name.

    Args:
        df: Row-level frame containing ``patient_col``.
        split: Patient-level split.
        patient_col: Name of the patient ID column.

    Returns:
        A string Series aligned with ``df.index`` with values in :data:`SPLIT_NAMES`.

    Raises:
        ValueError: If any row's patient is not part of the split.
    """
    lookup = {p: name for name, ids in split.as_dict().items() for p in ids}
    labels = df[patient_col].map(lookup)
    if labels.isna().any():
        unknown = df.loc[labels.isna(), patient_col].unique()[:5].tolist()
        raise ValueError(f"Rows belong to patients absent from the split, e.g. {unknown}")
    return labels.astype(str)
