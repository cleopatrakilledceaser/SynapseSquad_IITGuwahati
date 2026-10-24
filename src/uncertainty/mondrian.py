"""Group-conditional ("Mondrian") conformal intervals for the served model (D-076).

Question: the served CV+ 80 % band covers 80.5 % of rows overall, but only 73.8 % for CGMacros'
type 2 patients, because the residual pool it is calibrated on is two-thirds non-diabetic.
Does calibrating each group on residuals from its own group restore coverage within the group?

Method (no refitting, reuses the stored Part 6 outputs of the served D-072a model):
  * point forecast = the served population model on test patients (``cvp_pop|h``);
  * scores = absolute out-of-fold residuals of the SAME fold's training patients (``oof_pop|h``,
    rows from day ``test_from_day``, exactly the rows the CV+ arm uses);
  * ``cvc``  pooled:   point +/- conformal quantile of all training residuals (control);
  * ``cvm``  Mondrian: point +/- conformal quantile of the residuals of training patients in
    the test row's group (CGMacros diabetes status, Shanghai HbA1c band), falling back to the
    pooled quantile when a group has fewer than ``MIN_RES`` residuals or ``MIN_PAT`` patients.
Groups are static record fields known before any forecast, so no test information is used.
The served CV+ intervals (``cvp``) are reported beside both for reference.

Run (needs the Part 6 outputs in data/processed/uncertainty/, i.e. after ``make uncertainty``):
    python -W ignore -u -m src.uncertainty.mondrian
Writes reports/metrics/mondrian.json and prints the D-076 primary result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.uncertainty import conformal as cf
from src.uncertainty.common import COHORTS, HORIZONS, UNC, load
from src.uncertainty.intervals import coverage_tables
from src.utils.config import load_config
from src.utils.provenance import file_hash, write_metrics

MIN_RES = 200  # residual rows a group needs for its own quantile (else pooled)
MIN_PAT = 2  # training patients a group needs for its own quantile (else pooled)


def lv(a: float) -> str:
    """'80' / '95' label of a miscoverage level."""
    return str(round(100 * (1 - a)))


def group_quantiles(
    res: np.ndarray, grp: np.ndarray, pid: np.ndarray, alpha: float
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Pooled conformal quantile and one per group (with fallback flags).

    Args:
        res: absolute residuals (NaN ignored).
        grp: group label per residual.
        pid: patient id per residual.
        alpha: miscoverage level.
    """
    ok = ~np.isnan(res)
    pooled = cf.conformal_quantile(res[ok], alpha)
    out = {}
    for g in sorted(set(grp[ok])):
        m = ok & (grp == g)
        n_res, n_pat = int(m.sum()), int(len(set(pid[m])))
        own = n_res >= MIN_RES and n_pat >= MIN_PAT
        q = cf.conformal_quantile(res[m], alpha) if own else pooled
        out[g] = {"q": q, "n_res": n_res, "n_pat": n_pat, "fallback": not own}
    return pooled, out


def add_intervals(te: pd.DataFrame, oof: pd.DataFrame, uc: dict) -> tuple[pd.DataFrame, list]:
    """Add cvc_ (pooled) and cvm_ (Mondrian) intervals to the test rows, fold by fold."""
    te = te.copy()
    log = []
    for f in sorted(te["fold"].unique()):
        tf = te["fold"] == f
        of = oof[(oof["fold"] == f) & (oof["days_since_start"] >= uc["test_from_day"])]
        for h in HORIZONS:
            res = np.abs(of[f"y|{h}"].to_numpy(float) - of[f"oof_pop|{h}"].to_numpy(float))
            pt = te.loc[tf, f"cvp_pop|{h}"].to_numpy(float)
            g_te = te.loc[tf, "group"].to_numpy()
            for a in uc["alphas"]:
                pooled, qs = group_quantiles(
                    res, of["group"].to_numpy(), of["patient_id"].to_numpy(), a
                )
                qg = np.array([qs[g]["q"] if g in qs else pooled for g in g_te])
                te.loc[tf, f"cvc_pop_lo{lv(a)}|{h}"] = pt - pooled
                te.loc[tf, f"cvc_pop_hi{lv(a)}|{h}"] = pt + pooled
                te.loc[tf, f"cvm_pop_lo{lv(a)}|{h}"] = pt - qg
                te.loc[tf, f"cvm_pop_hi{lv(a)}|{h}"] = pt + qg
                log.append({"fold": int(f), "horizon_min": h, "level": int(lv(a)),
                            "pooled_q": pooled, "groups": qs})  # fmt: skip
    return te, log


def keep_cols(te: pd.DataFrame) -> pd.DataFrame:
    """Only the served CV+ population intervals and the two new ones (plus keys and targets)."""
    base = ["patient_id", "period_id", "t", "group", "fold"] + [f"y|{h}" for h in HORIZONS]
    iv = [c for c in te.columns if c.startswith(("cvp_pop_lo", "cvp_pop_hi", "cvc_", "cvm_"))]
    return te[base + iv]


def run() -> dict[str, Any]:
    """Score designs A (headline) and B (confirmatory) for both cohorts."""
    uc = load_config("uncertainty")
    bsc = uc["bootstrap"]
    m: dict[str, Any] = {}
    for des in uc["designs"]:
        m[des] = {}
        for c in COHORTS:
            te = load(des, c, "test")
            oof = load(des, c, "oof")
            te2, log = add_intervals(te, oof, uc)
            cov = coverage_tables(keep_cols(te2), bsc, uc["min_patients_reliable"])
            cov.pop("per_patient")
            m[des][c] = {"coverage": cov, "quantiles": log}
            print(f"[mondrian] {des} {c}: {te.patient_id.nunique()} test patients", flush=True)
    return m


def pick(rows: list[dict], meth: str, h: int, lvl: int, group: str | None = None) -> dict:
    """One coverage row."""
    for r in rows:
        if (r["method"], r["model"], r["horizon_min"], r["level"]) == (meth, "pop", h, lvl) and (
            group is None or r.get("group") == group
        ):
            return r
    return {}


def report(m: dict[str, Any]) -> None:
    """Print the D-076 primary endpoint and a compact table."""
    names = {"cvp": "served CV+", "cvc": "pooled (control)", "cvm": "group-wise (Mondrian)"}
    for des in m:
        for c in m[des]:
            cov = m[des][c]["coverage"]
            print(f"\n== design {des} · {c} · 80 % level ==")
            groups = sorted({r["group"] for r in cov["group"]})
            for h in HORIZONS:
                for meth in ("cvp", "cvc", "cvm"):
                    o = pick(cov["overall"], meth, h, 80)
                    if not o:
                        continue
                    parts = [f"overall {100 * o['coverage']:.1f}% (w {o['median_width']:.1f})"]
                    for g in groups:
                        r = pick(cov["group"], meth, h, 80, g)
                        if r:
                            lo, hi = (100 * x for x in r["coverage_ci"])
                            parts.append(
                                f"{g} {100 * r['coverage']:.1f}% [{lo:.1f}, {hi:.1f}]"
                                f" w {r['median_width']:.1f}"
                            )
                    print(f"  +{h:>3} {names[meth]:<22} " + " | ".join(parts))
    a = m.get("A", {}).get("cgmacros", {}).get("coverage")
    if not a:
        return
    t2 = pick(a["group"], "cvm", 60, 80, "t2d")
    if not t2:
        print("\nD-076 primary: no t2d group found in design A CGMacros")
        return
    lo, hi = (100 * x for x in t2["coverage_ci"])
    ok_t2 = 0.78 <= t2["coverage"] <= 0.86 and hi >= 80.0  # D-076 rule, written before the run
    ok_all = all(
        pick(m["A"][c]["coverage"]["overall"], "cvm", h, 80).get("coverage", 0) >= 0.79
        for c in m["A"]
        for h in HORIZONS
    )
    verdict = "MET" if ok_t2 else "NOT MET"
    print(
        "\nD-076 primary (design A, CGMacros T2D, +60, 80 %): "
        f"{100 * t2['coverage']:.1f}% [{lo:.1f}, {hi:.1f}] -> {verdict}; "
        f"overall >= 79 % everywhere: {'yes' if ok_all else 'no'}"
    )


def main() -> None:
    """Run, write reports/metrics/mondrian.json, print the summary."""
    m = run()
    cfg = {"min_res": MIN_RES, "min_pat": MIN_PAT, "uncertainty": load_config("uncertainty")}
    dh = file_hash(sorted(UNC.glob("*_test.parquet")) + sorted(UNC.glob("*_oof.parquet")))
    path = write_metrics("mondrian", m, cfg, cfg["uncertainty"]["seed"], dh)
    report(m)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
