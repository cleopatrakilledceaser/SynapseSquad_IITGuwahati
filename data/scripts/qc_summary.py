"""Post-harmonization QC summary per cohort and diabetes status -> reports/metrics/qc_summary.json.

Run after ``make data``: ``python data/scripts/qc_summary.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.provenance import file_hash, write_metrics  # noqa: E402

PROC = ROOT / "data" / "processed"


def within_wear(ts: pd.DataFrame) -> pd.Series:
    """True for rows between the first and last real CGM reading of each period (sensor worn).

    CGMacros files continue after the 10-day Dexcom sensor ends (Libre/Fitbit keep recording), so
    trailing rows are missing by design; this mask separates dropouts from end-of-wear.
    """
    real_t = ts["t"].where(ts["glucose_is_real"])
    first = real_t.groupby(ts["period_id"]).transform("min")
    last = real_t.groupby(ts["period_id"]).transform("max")
    return (ts["t"] >= first) & (ts["t"] <= last)


def summarise(ts: pd.DataFrame, static: pd.DataFrame, grid: str) -> pd.DataFrame:
    """Per (dataset, status) summary of patients, rows and CGM real/carried/missing percentages."""
    d = ts.merge(static[["patient_id", "diabetes_status"]], on="patient_id", how="left")
    d["missing_in_wear"] = d["glucose_missing"].where(within_wear(d))
    g = d.groupby(["dataset", "diabetes_status"])
    out = pd.DataFrame(
        {
            "patients": g["patient_id"].nunique(),
            "periods": g["period_id"].nunique(),
            "rows": g.size(),
            "pct_real": 100 * g["glucose_is_real"].mean(),
            "pct_carried": 100 * g["glucose_is_carried"].mean(),
            "pct_missing": 100 * g["glucose_missing"].mean(),
            "pct_missing_within_wear": 100 * g["missing_in_wear"].mean(),
            "pct_hr_present": 100 * (1 - g["hr_missing"].mean()),
            "pct_activity_present": 100 * (1 - g["activity_missing"].mean()),
        }
    ).round(2)
    out["grid"] = grid
    return out


def per_patient(ts: pd.DataFrame) -> pd.DataFrame:
    """Per-patient CGM gap % and wearable coverage %."""
    ts = ts.assign(missing_in_wear=ts["glucose_missing"].where(within_wear(ts)))
    g = ts.groupby("patient_id")
    return pd.DataFrame(
        {
            "days": g["days_since_start"].max().round(2),
            "pct_cgm_missing": (100 * g["glucose_missing"].mean()).round(2),
            "pct_cgm_missing_within_wear": (100 * g["missing_in_wear"].mean()).round(2),
            "pct_cgm_real": (100 * g["glucose_is_real"].mean()).round(2),
            "pct_hr_present": (100 * (1 - g["hr_missing"].mean())).round(2),
            "pct_activity_present": (100 * (1 - g["activity_missing"].mean())).round(2),
        }
    )


def main() -> None:
    """Print and write the QC summary."""
    static = pd.read_parquet(PROC / "static.parquet")
    ts5 = pd.read_parquet(PROC / "cgmacros_ts5.parquet")
    ts15 = pd.read_parquet(PROC / "common_ts15.parquet")
    s = pd.concat([summarise(ts5, static, "5min"), summarise(ts15, static, "15min")])
    print(s.to_string())
    pp5 = per_patient(ts5)
    pp15 = per_patient(ts15[ts15.dataset == "shanghai_t2dm"])
    pp5.to_csv(ROOT / "reports" / "metrics" / "per_patient_coverage_cgmacros_5min.csv")
    pp15.to_csv(ROOT / "reports" / "metrics" / "per_patient_coverage_shanghai_15min.csv")
    print("CGMacros per-patient (5-min):")
    print(pp5.describe().loc[["min", "50%", "max"]].to_string())
    print("Shanghai per-patient (15-min):")
    print(pp15.describe().loc[["min", "50%", "max"]].to_string())
    cfg = load_config("data")
    write_metrics(
        "qc_summary",
        {
            "by_cohort_status": s.reset_index().to_dict("records"),
            "cgmacros_per_patient": pp5.reset_index().to_dict("records"),
            "shanghai_per_patient": pp15.reset_index().to_dict("records"),
        },
        cfg,
        cfg["seed"],
        file_hash(sorted(PROC.glob("*.parquet"))),
    )


if __name__ == "__main__":
    main()
