"""Validate CGM reconstruction (D-010) on selected CGMacros participants and write metrics.

For each participant it reports, for Dexcom:
- n real readings detected, how many came from kinks vs collinear recovery;
- re-interpolation error: interpolating ONLY detected real points back onto the 1-minute rows
  must reproduce the raw file (max |error| about 0 means the detected set fully explains the data);
- integer check: real Dexcom readings must be integers;
- spacing: fraction of consecutive real readings exactly 5 min apart;
- longest gap between real readings, and how many 5-min grid points are MISSING (not filled).

Run: ``python data/scripts/validate_cgm_reconstruction.py 001 018 026``
"""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.cgm_reconstruct import (  # noqa: E402
    detect_native,
    make_grid,
    reinterpolation_error,
    to_grid,
)
from src.utils.config import load_config  # noqa: E402


def validate(pid: str, cfg: dict) -> dict:
    """Return validation metrics for one participant."""
    c = cfg["cgm"]
    z = zipfile.ZipFile(ROOT / cfg["paths"]["cgmacros_zip"])
    name = next(n for n in z.namelist() if n.endswith(f"CGMacros-{pid}.csv") and "MACOSX" not in n)
    df = pd.read_csv(io.BytesIO(z.read(name)), usecols=["Timestamp", "Dexcom GL"])
    ts = pd.to_datetime(df["Timestamp"])
    minutes = ((ts - ts.iloc[0]).dt.total_seconds() // 60).astype(int).to_numpy()
    v = pd.to_numeric(df["Dexcom GL"], errors="coerce").to_numpy()
    period = c["dexcom_period_min"]
    kinks_only = detect_native(minutes, v, period, 0, c["kink_tol"], c["integer_tol"])
    native = detect_native(minutes, v, period, c["max_gap_min"], c["kink_tol"], c["integer_tol"])
    real_t, real_v = ts[native], v[native]
    sp = real_t.diff().dt.total_seconds().div(60).dropna()
    grid = make_grid(ts.iloc[0], ts.iloc[-1], cfg["grid"]["primary_min"])
    g = to_grid(
        real_t,
        pd.Series(real_v),
        grid,
        cfg["grid"]["primary_min"],
        c["max_gap_min"],
        c["session_gap_min"],
        c["early_wear_hours"],
    )
    raw_on_grid = pd.Series(v, index=ts).reindex(grid)
    hidden = raw_on_grid.notna().to_numpy() & g["glucose_missing"].to_numpy()
    err_max, err_frac = reinterpolation_error(minutes, v, native, c["max_gap_min"])
    return {
        "pid": pid,
        "rows_1min": int(len(v)),
        "rows_with_value": int((~np.isnan(v)).sum()),
        "real_readings": int(native.sum()),
        "from_kinks_or_run_ends": int(kinks_only.sum()),
        "from_collinear_recovery": int(native.sum() - kinks_only.sum()),
        "reinterp_max_abs_err_mgdl": round(err_max, 4),
        "reinterp_rows_err_gt_0p01_frac": round(err_frac, 5),
        "spacing_mode_counts": sp.value_counts().head(4).to_dict(),
        "real_non_integer": int(np.sum(np.abs(real_v - np.round(real_v)) > c["integer_tol"])),
        "spacing_exactly_5min_frac": round(float((sp == period).mean()), 4),
        "longest_gap_min": float(sp.max()),
        "gaps_over_30min": int((sp > c["max_gap_min"]).sum()),
        "grid_points": int(len(g)),
        "grid_real_pct": round(100 * float(g["glucose_is_real"].mean()), 2),
        "grid_missing_pct": round(100 * float(g["glucose_missing"].mean()), 2),
        "raw_interpolated_values_now_missing": int(hidden.sum()),
        "max_minutes_since_real_when_present": float(
            g.loc[~g["glucose_missing"], "minutes_since_real"].max()
        ),
    }


def main() -> None:
    """Validate the requested participants and write reports/metrics/cgm_reconstruction.json."""
    cfg = load_config("data")
    pids = sys.argv[1:] or ["001", "018", "026"]
    if pids == ["all"]:
        z = zipfile.ZipFile(ROOT / cfg["paths"]["cgmacros_zip"])
        pids = sorted(
            m.group(1)
            for n in z.namelist()
            if (m := re.search(r"CGMacros-(\d{3})\.csv$", n)) and "MACOSX" not in n
        )
    res = [validate(p, cfg) for p in pids]
    table = pd.DataFrame(res).set_index("pid").drop(columns=["spacing_mode_counts"])
    if len(res) > 6:
        print(table.describe().loc[["min", "50%", "max"]].round(4).T.to_string())
        print(
            "reinterp err > 0.01 mg/dL in >1% of rows:",
            table.index[table.reinterp_rows_err_gt_0p01_frac > 0.01].tolist(),
        )
    else:
        print(table.T.to_string())
    from src.utils.provenance import write_metrics

    write_metrics(
        "cgm_reconstruction_validation",
        {"participants": res},
        cfg,
        cfg["seed"],
        "sha256:" + _sha_from_sums(),
    )
    print(json.dumps({"written": "reports/metrics/cgm_reconstruction_validation.json"}))


def _sha_from_sums() -> str:
    """Use PhysioNet's published SHA256 for the zip (already verified by download.py)."""
    sums = (ROOT / "data" / "raw" / "cgmacros" / "SHA256SUMS.txt").read_text().splitlines()
    return next(line.split()[0] for line in sums if line.endswith("CGMacros_dateshifted365.zip"))


if __name__ == "__main__":
    main()
