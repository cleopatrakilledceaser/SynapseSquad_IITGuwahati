"""Build anchor tables with targets + causal features per cohort and input length N.

Outputs (data/processed/features/, gitignored):
- ``<cohort>_N<h>.parquet``  anchors x (ids, targets, features) for N in configs windows.input_hours
- ``feature_sets.json``       explicit feature lists per cohort / set (cgm_only, cgm_tod, full,
                              shared) and N
- ``window_counts.json``      candidate / dropped / kept anchors and per-target availability

The same anchors are used for every N (eligibility uses the largest N), so N is compared on
identical evaluation rows.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.data.schema import is_calendar_column
from src.features.build import period_features, static_features
from src.features.windows import build_anchor_table
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.provenance import file_hash, git_hash
from src.utils.seed import set_global_seed

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROC / "features"
ID_COLS = ["patient_id", "period_id", "dataset", "t", "minutes_since_start", "input_real_frac"]
TARGET_PREFIXES = ("y_", "hyper180_", "hyper140_", "hypo70_")


def is_target(col: str) -> bool:
    """True for regression targets and event labels."""
    return col.startswith(TARGET_PREFIXES)


def feature_sets(cols: list[str], cohort: str, cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Explicit, separate feature sets for a cohort's table columns."""
    cgm = [c for c in cols if c.startswith("cgm_")]
    tod = ["tod_sin", "tod_cos"]
    st_key = "cgmacros" if cohort.startswith("cgmacros") else cohort
    static_c = [f"static_{c}" for c in cfg["static"][st_key]]
    shared_static = [f"static_{c}" for c in cfg["static"]["shared"]]
    meal_type_cols = [f"meal_last_type_{m}" for m in cfg["meals"]["meal_types"]]
    no_info = set()
    if cohort == "shanghai_t2dm":
        # Shanghai meals carry no type or amount (all "unknown" / NA), so these are constant.
        no_info = set(meal_type_cols) | {"meal_last_amount_gt100"}
    dynamic_other = [
        c
        for c in cols
        if (c.startswith(("meal_", "hr_", "act_", "rest_proxy")) or c == "days_since_start")
        and c not in no_info
    ]
    full = cgm + tod + dynamic_other + static_c
    sets = {
        "cgm_only": cgm,
        "cgm_tod": cgm + tod,
        "full": full,
        "shared": cgm + tod + shared_static,
    }
    for name, fs in sets.items():
        bad = [c for c in fs if is_calendar_column(c)]
        if bad:
            raise ValueError(f"calendar features in set {name}: {bad}")
    return sets


def build_cohort(
    cohort: str,
    fcfg: dict[str, Any],
    grid_all: pd.DataFrame,
    meals: pd.DataFrame,
    static: pd.DataFrame,
) -> tuple[dict[int, pd.DataFrame], dict[str, Any]]:
    """Return {N: table} and window counts for one cohort."""
    cc = fcfg["cohorts"][cohort]
    grid = grid_all[grid_all.dataset == cc["dataset"]].sort_values(["period_id", "t"])
    step, stride = cc["step_min"], cc["anchor_stride_min"]
    anchors, counts = build_anchor_table(grid, step, stride, fcfg)
    st_key = "cgmacros" if cohort.startswith("cgmacros") else cohort
    st = static_features(static, fcfg["static"][st_key])
    tables: dict[int, pd.DataFrame] = {}
    for n_h in fcfg["windows"]["input_hours"]:
        feats = []
        for pid, g in grid.groupby("period_id", sort=True):
            g = g.reset_index(drop=True)
            m = meals[meals.period_id == pid]
            f = period_features(g, m, step, n_h, fcfg)
            f["period_id"], f["t"] = pid, g["t"].to_numpy()
            feats.append(f)
        fdf = pd.concat(feats, ignore_index=True)
        tab = anchors.merge(fdf, on=["period_id", "t"], how="left", validate="one_to_one")
        tab = tab.merge(st, left_on="patient_id", right_index=True, how="left")
        tables[n_h] = tab
    counts["kept_anchors"] = int(len(anchors))
    counts["patients"] = int(anchors.patient_id.nunique())
    counts["periods"] = int(anchors.period_id.nunique())
    counts["targets_available"] = {
        c: int(anchors[c].notna().sum()) for c in anchors.columns if is_target(c)
    }
    counts["targets_positive"] = {
        c: int((anchors[c] == 1).sum())
        for c in anchors.columns
        if is_target(c) and not c.startswith("y_")
    }
    counts["positive_patients"] = {
        c: int(anchors.loc[anchors[c] == 1, "patient_id"].nunique())
        for c in anchors.columns
        if is_target(c) and not c.startswith("y_")
    }
    return tables, counts


def main() -> None:
    """Build all feature tables and metadata."""
    fcfg = load_config("features")
    set_global_seed(fcfg["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    static = pd.read_parquet(PROC / "static.parquet")
    meals = pd.read_parquet(PROC / "meals.parquet")
    all_sets, all_counts = {}, {}
    for cohort, cc in fcfg["cohorts"].items():
        grid = pd.read_parquet(PROC / cc["grid_file"])
        tables, counts = build_cohort(cohort, fcfg, grid, meals, static)
        all_counts[cohort] = counts
        all_sets[cohort] = {}
        for n_h, tab in tables.items():
            tab.to_parquet(OUT / f"{cohort}_N{n_h}.parquet", index=False)
            all_sets[cohort][f"N{n_h}"] = feature_sets(list(tab.columns), cohort, fcfg)
        print(
            f"{cohort}: anchors={counts['kept_anchors']} patients={counts['patients']} "
            f"cols(N6)={tables[max(tables)].shape[1]}"
        )
    (OUT / "feature_sets.json").write_text(json.dumps(all_sets, indent=1), "utf-8")
    meta = {
        "git_hash": git_hash(),
        "config": fcfg,
        "data_hash": file_hash(sorted(PROC.glob("*.parquet"))),
        "counts": all_counts,
    }
    (OUT / "window_counts.json").write_text(json.dumps(meta, indent=1, default=str), "utf-8")


if __name__ == "__main__":
    main()
