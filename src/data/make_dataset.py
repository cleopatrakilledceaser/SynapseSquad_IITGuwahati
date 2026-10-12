"""Build all harmonized parquet files (``make data``).

Outputs in data/processed/ (gitignored):
- ``cgmacros_ts5.parquet``   CGMacros on the native 5-min Dexcom grid (primary modeling grid)
- ``common_ts15.parquet``    CGMacros + Shanghai T2DM on the common 15-min grid (transfer)
- ``meals.parquet``          one row per logged meal (both cohorts)
- ``static.parquet``         one row per patient (both cohorts)
- ``manifest.json``          row counts, git hash, config, raw-data hashes
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data import harmonize_cgmacros, harmonize_shanghai
from src.data.schema import MEAL_COLUMNS, STATIC_COLUMNS, TIMESERIES_COLUMNS, conform
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.provenance import file_hash, git_hash
from src.utils.seed import set_global_seed

STATIC_EXTRA = ("hypoglycemic_agents_raw",)


def _static(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate static tables; keep schema columns plus medication flags."""
    df = pd.concat(frames, ignore_index=True)
    base = conform(df, STATIC_COLUMNS)
    extra = [c for c in df.columns if c.startswith("med_") or c in STATIC_EXTRA]
    for c in extra:
        col = df[c]
        base[c] = col.astype("boolean") if c.startswith("med_") else col.astype("string")
    return base


def main() -> None:
    """Harmonize both cohorts and write parquet + manifest."""
    cfg = load_config("data")
    set_global_seed(cfg["seed"])
    out = PROJECT_ROOT / cfg["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)
    cg_zip = PROJECT_ROOT / cfg["paths"]["cgmacros_zip"]
    sh_zip = PROJECT_ROOT / cfg["paths"]["shanghai_zip"]

    cg = harmonize_cgmacros.harmonize(cg_zip, cfg)
    sh = harmonize_shanghai.harmonize(sh_zip, cfg)

    files = {
        "cgmacros_ts5": conform(cg["ts5"], TIMESERIES_COLUMNS),
        "common_ts15": conform(pd.concat([cg["ts15"], sh["ts15"]]), TIMESERIES_COLUMNS),
        "meals": conform(pd.concat([cg["meals"], sh["meals"]]), MEAL_COLUMNS),
        "static": _static([cg["static"], sh["static"]]),
    }
    for name, df in files.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {df.shape}")
    manifest = {
        "git_hash": git_hash(),
        "seed": cfg["seed"],
        "config": cfg,
        "raw_hashes": {"cgmacros_zip": file_hash([cg_zip]), "shanghai_zip": file_hash([sh_zip])},
        "rows": {k: int(len(v)) for k, v in files.items()},
        "patients": {k: int(v["patient_id"].nunique()) for k, v in files.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), "utf-8")


if __name__ == "__main__":
    main()


__all__ = ["main", "Path"]
