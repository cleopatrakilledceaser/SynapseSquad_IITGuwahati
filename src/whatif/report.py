"""Part 7 report: metrics JSON (+ figures) from the dev / final stage outputs (D-051, D-053).

reports/metrics/whatif_plausibility.json  pass rates (patient-bootstrap CIs), dev and final
reports/metrics/whatif_matched.json       matched-meal metrics, dev and final; mono vs unc MAE
reports/metrics/whatif_examples.json      example scenarios (training patient + India presets)
Usage: python -m src.whatif.report [--final]   (without --final: dev outputs only)
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pandas as pd

from src.hybrid.twin_features import HORIZONS
from src.uncertainty.common import boot_paired_diff, boot_ratio
from src.utils.config import load_config
from src.utils.provenance import file_hash, write_metrics
from src.whatif.engine import PROC
from src.whatif.matched import metrics as pair_metrics
from src.whatif.run import LOCK, OUT

ML_NOTE = {
    ("carb_monotonicity", "mono"): "true by construction (monotone constraint)",
    ("carb_monotonicity_2mgdl", "mono"): "true by construction (monotone constraint)",
    ("activity_direction", "mono"): "true by construction (monotone constraint)",
    ("activity_direction", "unc"): "what the data learned",
    ("activity_direction", "twin_assumed"): "ASSUMED walk effect, not learned from data",
}


def pass_rates(d: pd.DataFrame, bc: dict, seed: int) -> list[dict[str, Any]]:
    """Pass rate per (test, model) with a patient-bootstrap CI (per-patient sums)."""
    d = d[d["pass"].notna()].assign(p=lambda x: x["pass"].astype(float))
    rows = []
    for (test, model), g in d.groupby(["test", "model"]):
        per = g.groupby("patient_id").p.agg(["sum", "count"])
        r = boot_ratio(per["sum"], per["count"], bc["n_boot"], seed, bc["ci"])
        rows.append(
            {
                "test": test,
                "model": model,
                "pass_rate": r["value"],
                "ci": r["ci"],
                "n_checks": int(len(g)),
                "n_patients": r["n_patients"],
                "note": ML_NOTE.get((test, model), ""),
            }
        )
    return rows


def dose_summary(d: pd.DataFrame, pc: dict) -> list[dict[str, Any]]:
    """Median dose (mg/dL per g) vs the D-051 median range."""
    out = []
    lo, hi = pc["dose_median_range"]
    for model, g in d[d.test == "dose_window"].groupby("model"):
        med = float(g.value.median())
        out.append(
            {"model": model, "median_mgdl_per_g": med, "range": [lo, hi], "pass": lo <= med <= hi}
        )
    return out


def descriptive(d: pd.DataFrame) -> list[dict[str, Any]]:
    """Median / IQR of descriptive effects (protein / fat / fiber edits, walk AUC effect)."""
    sel = d[d.test.str.startswith(("edit_", "walk_effect"))]
    out = []
    for (test, model), g in sel.groupby(["test", "model"]):
        q = g.value.quantile([0.25, 0.5, 0.75]).to_numpy()
        out.append({"test": test, "model": model, "median": q[1], "iqr": [q[0], q[2]]})
    return out


def cross_spearman(d: pd.DataFrame) -> dict[str, float]:
    """Spearman correlation of ML vs twin 30 -> 60 g peak effects (cross-model rows)."""
    c = d[d.test == "cross_model"]
    return {
        m: float(g["ml_effect"].corr(g["twin_effect"], method="spearman"))
        for m, g in c.groupby("model")
    }


def plaus_block(path: Any, wc: dict) -> dict[str, Any]:
    """All plausibility summaries of one stage."""
    d = pd.read_parquet(path)
    bc = wc["bootstrap"]
    return {
        "n_patients": int(d.patient_id.nunique()),
        "pass_rates": pass_rates(d, bc, wc["seed"]),
        "dose_median": dose_summary(d, wc["plausibility"]),
        "descriptive": descriptive(d),
        "cross_model_spearman": cross_spearman(d),
    }


def matched_block(path: Any, wc: dict) -> dict[str, Any]:
    """Matched-meal metrics per design x model x target."""
    d = pd.read_parquet(path)
    bc = wc["bootstrap"]
    return {
        f"{de}|{mo}|{ta}": pair_metrics(g, bc["n_boot"], wc["seed"], bc["ci"])
        for (de, mo, ta), g in d.groupby(["design", "model", "target"])
    }


def mae_block(path: Any, wc: dict) -> dict[str, Any]:
    """Test MAE of unc and mono and the paired difference mono - unc (patient bootstrap)."""
    d = pd.read_parquet(path)
    bc = wc["bootstrap"]
    out = {}
    for h in HORIZONS:
        a, b = f"ae_mono|{h}", f"ae_unc|{h}"
        ok = d[a].notna() & d[b].notna()
        per = d[ok].groupby("patient_id").agg(mono=(a, "sum"), unc=(b, "sum"), n=(a, "size"))
        diff = boot_paired_diff(per.mono, per.unc, per.n, bc["n_boot"], wc["seed"], bc["ci"])
        out[str(h)] = {
            "mae_unc": float(per.unc.sum() / per.n.sum()),
            "mae_mono": float(per.mono.sum() / per.n.sum()),
            "mono_minus_unc": diff,
            "n_patients": int(len(per)),
        }
    return out


def main() -> None:
    """Write the metrics files (final blocks only with --final and an existing lock)."""
    wc = load_config("whatif")
    final = "--final" in sys.argv
    if final and not LOCK.exists():
        raise SystemExit("final stage has not run")
    dh = file_hash([PROC / f"{f}.parquet" for f in ("cgmacros_ts5", "meals", "static")])
    plaus = {"dev_fold0_training_patients": plaus_block(OUT / "dev_plaus.parquet", wc)}
    mat = {"dev_fold0_training_patients": matched_block(OUT / "dev_matched.parquet", wc)}
    if final:
        plaus["final_test_patients"] = plaus_block(OUT / "final_plaus.parquet", wc)
        mat["final_all_patients"] = matched_block(OUT / "final_matched.parquet", wc)
        mat["test_mae_mono_vs_unc"] = mae_block(OUT / "final_mae.parquet", wc)
        run = json.loads((OUT / "final.json").read_text())
        plaus["meta"] = {"final_run": run, "lock": json.loads(LOCK.read_text())}
    from src.whatif import d056

    if (OUT / "d056_dev_walk.parquet").exists():
        plaus["d056_posthoc"] = d056.summary(wc)  # post-hoc, not pre-registered
    write_metrics("whatif_plausibility", plaus, wc, wc["seed"], dh)
    write_metrics("whatif_matched", mat, wc, wc["seed"], dh)
    from src.whatif import examples, figures

    ex = examples.build(wc)
    write_metrics("whatif_examples", ex, wc, wc["seed"], dh)
    figures.all_figures(plaus, mat, ex, wc)
    print("wrote whatif_{plausibility,matched,examples}.json + figures", flush=True)


if __name__ == "__main__":
    main()
