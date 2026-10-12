"""Create and save all patient-level split assignments (Part 2), with a content hash.

Outputs (data/processed/splits/, gitignored):
- ``<cohort>_holdout.csv``   patient_id -> train | calibration | test (stratified)
- ``<cohort>_cv.csv``        patient_id, repeat, fold (the fold in which the patient is validation)
- ``<cohort>_chrono.parquet`` per-anchor 'adapt' / 'eval' labels for the personalization experiment
- ``splits_manifest.json``   settings + sha256 of every file
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from src.data.splits import (
    assert_no_patient_overlap,
    chronological_within_patient,
    patient_train_cal_test_split,
    repeated_patient_group_kfold,
)
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.provenance import git_hash

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROC / "splits"


def strata_for(static: pd.DataFrame, cohort: str, scfg: dict) -> dict[str, str]:
    """Per-patient stratum label (CGMacros: diabetes status; Shanghai: HbA1c group)."""
    s = static[static.dataset == cohort].set_index("patient_id")
    if scfg["cohorts"][cohort]["strata"] == "diabetes_status":
        return s["diabetes_status"].astype(str).to_dict()
    grp = pd.cut(
        s["hba1c_pct"], scfg["hba1c_group_bins_pct"], labels=scfg["hba1c_group_labels"], right=False
    )
    return grp.astype(str).replace("nan", "unknown").to_dict()


def _sha(path) -> str:  # type: ignore[no-untyped-def]
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Build and save holdout, repeated CV and chronological assignments for both cohorts."""
    scfg = load_config("splits")
    fcfg = load_config("features")
    static = pd.read_parquet(PROC / "static.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"git_hash": git_hash(), "config": scfg, "files": {}, "summary": {}}
    for cohort, cc in scfg["cohorts"].items():
        strata = strata_for(static, cohort, scfg)
        pids = sorted(strata)
        hold = patient_train_cal_test_split(pids, scfg["holdout"], scfg["seed"], strata)
        assert_no_patient_overlap(hold.train, hold.calibration, hold.test)
        h = pd.DataFrame(
            [(p, k) for k, ids in hold.as_dict().items() for p in sorted(ids)],
            columns=["patient_id", "split"],
        )
        h["stratum"] = h.patient_id.map(strata)
        h.to_csv(OUT / f"{cohort}_holdout.csv", index=False)
        rows = []
        for f in repeated_patient_group_kfold(
            pids, cc["n_splits"], cc["n_repeats"], scfg["seed"], strata
        ):
            rows += [(p, f.repeat, f.fold) for p in sorted(f.val)]
        cv = pd.DataFrame(rows, columns=["patient_id", "repeat", "fold"])
        cv.to_csv(OUT / f"{cohort}_cv.csv", index=False)
        anchors = pd.read_parquet(
            PROC / "features" / f"{cohort}_N{max(fcfg['windows']['input_hours'])}.parquet",
            columns=["patient_id", "period_id", "t", "days_since_start"],
        )
        anchors["chrono"] = chronological_within_patient(
            anchors, scfg["chronological"]["default_adapt_days"]
        )
        anchors[["patient_id", "period_id", "t", "chrono"]].to_parquet(
            OUT / f"{cohort}_chrono.parquet", index=False
        )
        manifest["summary"][cohort] = {
            "holdout": h.pivot_table(
                index="split",
                columns="stratum",
                values="patient_id",
                aggfunc="count",
                fill_value=0,
            ).to_dict(),
            "cv": {
                "n_splits": cc["n_splits"],
                "n_repeats": cc["n_repeats"],
                "strata": cc["strata"],
            },
            "chrono_counts": anchors["chrono"].value_counts().to_dict(),
        }
        for fn in (f"{cohort}_holdout.csv", f"{cohort}_cv.csv", f"{cohort}_chrono.parquet"):
            manifest["files"][fn] = _sha(OUT / fn)
        print(
            cohort,
            h.split.value_counts().to_dict(),
            "| chrono:",
            anchors["chrono"].value_counts().to_dict(),
        )
    (OUT / "splits_manifest.json").write_text(json.dumps(manifest, indent=1, default=str), "utf-8")


if __name__ == "__main__":
    main()
