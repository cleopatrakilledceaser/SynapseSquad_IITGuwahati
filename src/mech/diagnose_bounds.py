"""SG / beta bound diagnosis (Part 4b, D-037). TRAINING patients of each fold, days < 3 ONLY.

No test window and no evaluated patient is read: for each outer fold (CV repeat 0) the analysis
uses that fold's training patients and only their first 3 days (adaptation days).
Stage-1 fits (config prior, the fold's own alpha_E):
  1. bound: does SG go further down when its lower bound is widened (0.001 -> 1e-5)?
  2. timescale: 1/SG at the bound is 1000 min, longer than the 6-h simulation segment; do 12-h
     segments move SG off the bound?
  3. profile (fold 0): data-only delta(-2 log L) for SG fixed on a grid, others refit.
  4. beta: do fits at beta's upper bound go further up when it is widened (10 -> 100)? kabs-beta
     trade-off and within-patient variation of the slowing ratio.
  5. split-half stability inside the adaptation days: days [0, 1.5) vs [1.5, 3).
Usage: python -m src.mech.diagnose_bounds  -> reports/metrics/mech_bounds_diagnosis.json
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from src.mech.fit import config_prior, fit_patient, log_bounds
from src.mech.identify import shifted
from src.mech.inputs import Record
from src.mech.model import FIT_PARAMS
from src.mech.protocol import Ctx
from src.mech.run_mech import OUT, PROC, make_ctx
from src.models.fusion_lgbm import folds
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.provenance import file_hash, write_metrics

ISG, IBETA = FIT_PARAMS.index("SG"), FIT_PARAMS.index("beta")
SG_GRID = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def variant(mc: dict[str, Any], **over: Any) -> dict[str, Any]:
    """Config copy with overrides: sg_lo, beta_hi, segment_hours."""
    m = deepcopy(mc)
    if "sg_lo" in over:
        m["params"]["fit"]["SG"]["lo"] = over["sg_lo"]
    if "beta_hi" in over:
        m["params"]["fit"]["beta"]["hi"] = over["beta_hi"]
    if "segment_hours" in over:
        m["fit"]["segment_hours"] = over["segment_hours"]
    return m


def m2ll(fr: Any, mc: dict[str, Any]) -> float:
    """Data-only -2 log L (Gaussian, sigma_obs) of a fit."""
    return float(fr.n_obs * fr.rmse**2 / mc["fit"]["sigma_obs"] ** 2)


def fit_all(recs: list[Record], mc: dict[str, Any], k: float, alpha: float) -> pd.DataFrame:
    """Stage-1 fits (config prior of ``mc``): log params + data-only -2 log L per patient."""
    pr = config_prior(mc["params"])
    rows = []
    for r in recs:
        fr = fit_patient(r, k, pr, mc, alpha)
        rows.append(
            {"patient_id": r.patient_id, "m2ll": m2ll(fr, mc)}
            | {p: fr.logp[j] for j, p in enumerate(FIT_PARAMS)}
        )
    return pd.DataFrame(rows)


def at(v: pd.Series, edge: float, tol: float) -> float:
    """Fraction of log values within tol of a log edge."""
    return float((np.abs(v - edge) < tol).mean())


def slowing_cv(rec: Record, k: float) -> float:
    """Within-patient CV of the slowing ratio at meals in the first k days."""
    n = int(np.searchsorted(rec.days, k))
    r = rec.slow[:n][rec.carbs[:n] > 0]
    return float(np.std(r) / np.mean(r)) if len(r) > 1 and np.mean(r) > 0 else float("nan")


def fold_diag(c: Ctx, mc: dict[str, Any], train: list[str], alpha: float) -> dict[str, Any]:
    """All diagnostics for one fold's training patients (days < 3)."""
    tol = mc["identifiability"]["at_bound_tol_log"]
    k = mc["fit"]["stage1_days"]
    recs = [c.by_pat[p][0] for p in train]
    lo, hi = log_bounds(mc["params"])
    base = fit_all(recs, mc, k, alpha)
    wide = fit_all(recs, variant(mc, sg_lo=1e-5, beta_hi=100.0), k, alpha)
    seg = fit_all(recs, variant(mc, segment_hours=12), k, alpha)
    h1 = fit_all(recs, mc, k / 2, alpha)
    h2 = fit_all([shifted(r, k / 2) for r in recs], mc, k / 2, alpha)
    hit = (base.SG - lo[ISG]).abs() < tol
    return {
        "n_train": len(train),
        "alpha_E": alpha,
        "orig_SG_at_lower": at(base.SG, lo[ISG], tol),
        "orig_beta_at_upper": at(base.beta, hi[IBETA], tol),
        "orig_SG_median": float(np.exp(base.SG.median())),
        "wide_SG_below_old_bound": float((lo[ISG] - tol > wide.SG).mean()),
        "wide_SG_at_new_bound": at(wide.SG, np.log(1e-5), tol),
        "wide_SG_median_of_orig_hitters": float(np.exp(wide.SG[hit].median())),
        "wide_m2ll_gain_median_orig_hitters": float((base.m2ll - wide.m2ll)[hit].median()),
        "wide_beta_above_old_bound": float((wide.beta > hi[IBETA] + tol).mean()),
        "wide_beta_at_new_bound": at(wide.beta, np.log(100.0), tol),
        "seg12_SG_at_lower": at(seg.SG, lo[ISG], tol),
        "corr_log_kabs_beta": float(np.corrcoef(base.kabs, base.beta)[0, 1]),
        "corr_log_SG_Gb": float(np.corrcoef(base.SG, base.Gb)[0, 1]),
        "slowing_ratio_cv_median": float(np.nanmedian([slowing_cv(r, k) for r in recs])),
        "split_half_corr": {p: float(np.corrcoef(h1[p], h2[p])[0, 1]) for p in FIT_PARAMS},
    }


def sg_profile(c: Ctx, mc: dict[str, Any], train: list[str], alpha: float) -> dict[str, Any]:
    """Data-only delta(-2 log L) vs fixed SG (others refit, config prior), training patients."""
    m = variant(mc, sg_lo=min(SG_GRID))
    pr = config_prior(m["params"])
    others = [p for p in FIT_PARAMS if p != "SG"]
    k = mc["fit"]["stage1_days"]
    rows = []
    for p in train:
        for v in SG_GRID:
            base = pr.mu.copy()
            base[ISG] = np.log(v)
            fr = fit_patient(c.by_pat[p][0], k, pr, m, alpha, free=others, base_logp=base)
            rows.append({"pid": p, "SG": v, "m2ll": m2ll(fr, mc)})
    d = pd.DataFrame(rows)
    d["delta"] = d.m2ll - d.groupby("pid").m2ll.transform("min")
    best = d.loc[d.groupby("pid").m2ll.idxmin(), "SG"]
    low = d[d.SG <= 1e-3].groupby("pid").delta
    return {
        "sg_grid": SG_GRID,
        "delta_m2ll_median": d.groupby("SG").delta.median().round(2).tolist(),
        "frac_best_at_or_below_1e-3": float((best <= 1e-3).mean()),
        "frac_flat_1e-5_to_1e-3": float(((low.max() - low.min()) < 1.92).mean()),
    }


def split_half(
    recs: list[Record], mc: dict[str, Any], alpha: float, fix: list[str]
) -> dict[str, float]:
    """Split-half corr (days [0,1.5) vs [1.5,3)) with ``fix`` held at the training mean."""
    k = mc["fit"]["stage1_days"]
    pr = config_prior(mc["params"])
    st = fit_all(recs, mc, k, alpha)
    base = pr.mu.copy()
    for p in fix:
        base[FIT_PARAMS.index(p)] = st[p].mean()
    free = [p for p in FIT_PARAMS if p not in fix]
    halves = []
    for rs in (recs, [shifted(r, k / 2) for r in recs]):
        halves.append(np.array([fit_patient(r, k / 2, pr, mc, alpha, free, base).logp for r in rs]))
    return {
        p: float(np.corrcoef(halves[0][:, j], halves[1][:, j])[0, 1])
        for j, p in enumerate(FIT_PARAMS)
        if p in free
    }


def candidates(c: Ctx, mc: dict[str, Any], train: list[str], alpha: float) -> dict[str, Any]:
    """Split-half stability of candidate specifications (training patients, days < 3)."""
    recs = [c.by_pat[p][0] for p in train]
    wide = variant(mc, sg_lo=1e-5)
    return {
        "original": split_half(recs, mc, alpha, []),
        "sg_lo_1e-5": split_half(recs, wide, alpha, []),
        "sg_lo_1e-5_beta_fixed": split_half(recs, wide, alpha, ["beta"]),
        "sg_fixed_beta_fixed": split_half(recs, mc, alpha, ["SG", "beta"]),
    }


def main() -> None:
    """Run the diagnosis on every fold's training patients; write the metrics JSON."""
    if len(sys.argv) > 1 and sys.argv[1] == "candidates":
        main_candidates()
        return
    mc, fc = load_config("mech"), load_config("fusion")
    c = make_ctx("cgmacros", mc, fc)
    out: dict[str, Any] = {"scope": "training patients of each fold, days < 3 only", "folds": {}}
    for _, f, train, _test in folds("cgmacros", [mc["cv_repeat"]]):
        train = [p for p in train if p in c.by_pat]
        alpha = json.loads((OUT / f"twin_cgmacros_fold{f}.json").read_text())["alpha_on"]
        out["folds"][str(f)] = fold_diag(c, mc, train, alpha)
        if f == 0:
            out["sg_profile_fold0"] = sg_profile(c, mc, train, alpha)
        print(f"fold {f} done", flush=True)
    dh = file_hash([PROC / "cgmacros_ts5.parquet", PROC / "meals.parquet"])
    write_metrics("mech_bounds_diagnosis", out, mc, mc["seed"], dh)


def main_candidates() -> None:
    """Candidate-spec split-half stability per fold; merged into the diagnosis JSON."""
    mc, fc = load_config("mech"), load_config("fusion")
    c = make_ctx("cgmacros", mc, fc)
    path = PROJECT_ROOT / "reports" / "metrics" / "mech_bounds_diagnosis.json"
    payload = json.loads(path.read_text())
    res = {}
    for _, f, train, _test in folds("cgmacros", [mc["cv_repeat"]]):
        train = [p for p in train if p in c.by_pat]
        alpha = json.loads((OUT / f"twin_cgmacros_fold{f}.json").read_text())["alpha_on"]
        res[str(f)] = candidates(c, mc, train, alpha)
        print(f"fold {f} done", flush=True)
    payload["metrics"]["candidates_split_half"] = res
    write_metrics("mech_bounds_diagnosis", payload["metrics"], mc, mc["seed"], payload["data_hash"])


if __name__ == "__main__":
    main()
