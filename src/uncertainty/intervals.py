"""Interval metrics (D-045): coverage overall / per group / per patient, width, failures."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.uncertainty.common import HORIZONS, boot_ratio


def interval_columns(d: pd.DataFrame) -> list[tuple[str, str, str, int]]:
    """(method, model, level, h) for every interval present in a test frame."""
    out = []
    for c in d.columns:
        if "_lo" in c and "|" in c:
            meth, rest = c.split("_", 1)
            model, tail = rest.split("_lo")
            lvl, h = tail.split("|")
            out.append((meth, model, lvl, int(h)))
    return sorted(set(out))


def cov_frame(d: pd.DataFrame, meth: str, model: str, lvl: str, h: int) -> pd.DataFrame:
    """Per-row covered / width / infinite flags on rows with a real target and an interval."""
    lo = d[f"{meth}_{model}_lo{lvl}|{h}"].to_numpy(float)
    hi = d[f"{meth}_{model}_hi{lvl}|{h}"].to_numpy(float)
    y = d[f"y|{h}"].to_numpy(float)
    ok = ~np.isnan(y) & ~np.isnan(lo) & ~np.isnan(hi)
    w = hi - lo
    return pd.DataFrame(
        {
            "patient_id": d["patient_id"].to_numpy()[ok],
            "group": d["group"].to_numpy()[ok],
            "covered": ((y >= lo) & (y <= hi))[ok].astype(float),
            "width": np.where(np.isfinite(w), w, np.nan)[ok],
            "infinite": (~np.isfinite(w))[ok].astype(float),
        }
    )


def summarize(f: pd.DataFrame, target: float, bc: dict[str, Any], min_pat: int) -> dict:
    """Pooled coverage (patient bootstrap CI), median width, per-patient coverage spread."""
    g = f.groupby("patient_id")
    cov_p = g["covered"].mean()
    n_p = g.size()
    s = boot_ratio(g["covered"].sum(), n_p, bc["n_boot"], 42, bc["ci"])
    fin = f["width"].dropna()
    return {
        "coverage": s["value"],
        "coverage_ci": s["ci"],
        "target": target,
        "n_patients": int(len(cov_p)),
        "n_rows": int(len(f)),
        "median_width": float(fin.median()) if len(fin) else float("nan"),
        "mean_width": float(fin.mean()) if len(fin) else float("nan"),
        "infinite_frac": float(f["infinite"].mean()),
        "per_patient_min": float(cov_p.min()),
        "per_patient_q10": float(cov_p.quantile(0.1)),
        "per_patient_median": float(cov_p.median()),
        "frac_patients_below_target_minus_10pt": float((cov_p < target - 0.10).mean()),
        "ci_excludes_target": bool(s["ci"][1] < target or s["ci"][0] > target),
        "reliable": bool(len(cov_p) >= min_pat),
    }


def coverage_tables(d: pd.DataFrame, bc: dict, min_pat: int) -> dict[str, Any]:
    """All methods x models x levels x horizons: overall, per group, per patient."""
    out: dict[str, Any] = {"overall": [], "group": [], "per_patient": []}
    for meth, model, lvl, h in interval_columns(d):
        f = cov_frame(d, meth, model, lvl, h)
        tgt = int(lvl) / 100
        key = {"method": meth, "model": model, "level": int(lvl), "horizon_min": h}
        out["overall"].append(key | summarize(f, tgt, bc, min_pat))
        for grp, fg in f.groupby("group"):
            out["group"].append(key | {"group": grp} | summarize(fg, tgt, bc, min_pat))
        pp = f.groupby("patient_id").agg(
            coverage=("covered", "mean"), width=("width", "median"), n=("covered", "size")
        )
        for pid, r in pp.iterrows():
            out["per_patient"].append(key | {"patient_id": pid, **r.to_dict()})
    return out


def failures(tab: list[dict], tol: float = 0.0) -> list[str]:
    """Plain-language list of cells whose 95 % CI lies entirely below the target."""
    msgs = []
    for r in tab:
        if r["coverage_ci"][1] < r["target"] - tol:
            where = f" group {r['group']}" if "group" in r else ""
            flag = "" if r.get("reliable", True) else " (unreliable: < 5 patients)"
            msgs.append(
                f"{r['method']}/{r['model']} {r['level']}% +{r['horizon_min']} min{where}: "
                f"{100 * r['coverage']:.1f}% [{100 * r['coverage_ci'][0]:.1f}, "
                f"{100 * r['coverage_ci'][1]:.1f}]{flag}"
            )
    return msgs


__all__ = ["HORIZONS", "coverage_tables", "failures", "interval_columns", "cov_frame"]
