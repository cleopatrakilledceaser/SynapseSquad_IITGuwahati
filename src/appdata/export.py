"""Export of fold models, residuals, stored cohort statistics and figures for the app (D-058)."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hybrid.twin_features import HORIZONS
from src.utils.config import PROJECT_ROOT

METRICS = PROJECT_ROOT / "reports" / "metrics"
FIGS = PROJECT_ROOT / "reports" / "figures"
FIGURES = {
    "fusion_ablation": "Ablation of input streams",
    "personalize_cold_start_lgbm": "Cold start of personalization",
    "unc_coverage": "Interval coverage",
    "unc_alarm_froc": "Alarm burden",
    "whatif_plausibility": "What-if plausibility tests",
    "whatif_matched_meals": "Matched real breakfasts",
}


def write_gz(path: Path, text: str) -> None:
    """Write a gzip text file."""
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)


def export_models(forms: dict[str, Any], cols: list[str], out: Path) -> dict[str, Any]:
    """Boosters (trimmed to the used trees) of the full model and the inner-fold models + residuals.

    ``forms``: name -> ``src.whatif.models.Form`` of the fold. The full model is the one that never
    saw the demo patients; the inner models are the CV+ ensemble of Part 6 / Part 7.
    """
    out.mkdir(parents=True, exist_ok=True)
    res: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"cols": cols, "horizons": HORIZONS, "forms": {}}
    for name, f in forms.items():
        meta["forms"][name] = {"base": {}, "inner": sorted(f.inner)}
        for h in HORIZONS:
            meta["forms"][name]["base"][str(h)] = f.pop[h].base
            write_gz(out / f"{name}_{h}_pop.txt.gz", f.pop[h].model_str())
            for j, m in f.inner.items():
                write_gz(out / f"{name}_{h}_in{j}.txt.gz", m[h].model_str())
            for j, r in f.res[h].items():
                res[f"{name}_{h}_{j}"] = np.asarray(r, float)
    np.savez_compressed(out / "residuals.npz", **res)
    (out / "models.json").write_text(json.dumps(meta, indent=1), "utf-8")
    return meta


def _metrics(name: str) -> dict[str, Any]:
    return json.loads((METRICS / f"{name}.json").read_text("utf-8"))["metrics"]


def cohort_stats() -> dict[str, Any]:
    """Numbers shown on the reference pages, copied from the stored Part 3/5/6/7 metric files."""
    fus, per, unc = (
        _metrics("fusion_cgmacros"),
        _metrics("personalize_cgmacros"),
        _metrics("uncertainty"),
    )
    pl, mt = _metrics("whatif_plausibility"), _metrics("whatif_matched")
    uc = unc["A"]["cgmacros"]
    keep = ("lgbm_ns_none", "lgbm_s_none", "lgbm_ns_finetune", "lgbm_s_finetune")
    tiers = [r for r in fus["tiers"] if r["subset"] == "overall"]
    tier_models = ("persistence", "lgbm_E_full_delta", "lgbm_T2_fs4_delta", "lgbm_T2_fs2_delta",
                   "lgbm_T3_fs4_delta", "lgbm_T3_fs2_delta")  # fmt: skip
    drift = [c for c in uc["robustness"]["cells"] if c["fault"] in ("clean", "drift", "bias")]
    alarms = {
        e: {s: uc["alarms"][e][s]["train_chosen"] for s in ("model", "naive", "trend")}
        | {"label": uc["alarms"][e]["label"]}
        for e in ("hyper180", "hypo70")
    }
    return {
        "n_patients_fusion": fus["n_patients"],
        "tiers": [
            {k: r[k] for k in ("model", "horizon_min", "mae", "mae_ci", "n_patients")}
            for r in tiers
            if r["model"] in tier_models
        ],
        "personalize": {
            "k_grid": per["k_grid"],
            "curves": {
                h: [r for r in per[f"curve_overall_{h}"] if r["model"] in keep]
                for h in (30, 60, 120)
            },
            "head_start": per["crossover_overall_60"]["lgbm"],
            "eligibility": {k: per["eligibility"][k] for k in ("n_patients_total", "n_eligible")},
        },
        "coverage": [c for c in uc["coverage"]["overall"] if c["method"] in ("cvp", "scp")
                     and c["model"] == "pop"],  # fmt: skip
        "drift_cells": drift,
        "alarms": alarms,
        "events_auroc": {
            k: v["auroc"] for k, v in uc["events"].items() if k.endswith("_60") and "auroc" in v
        },
        "plausibility": {
            "final": pl["final_test_patients"],
            "d056": pl["d056_posthoc"]["final"],
        },
        "matched": {k: mt["final_all_patients"][k] for k in mt["final_all_patients"]
                    if k.startswith("within|")},  # fmt: skip
        "mono_cost": mt["test_mae_mono_vs_unc"],
    }


def copy_figures(out: Path) -> list[str]:
    """Copy the evidence figures (already committed under reports/figures)."""
    out.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        shutil.copy2(FIGS / f"{name}.png", out / f"{name}.png")
    return list(FIGURES)


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Parquet with a stable index."""
    df.reset_index(drop=True).to_parquet(path, index=False)
