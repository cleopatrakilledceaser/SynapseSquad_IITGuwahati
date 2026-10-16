"""Build the per-fold what-if context (data, both ML forms with guards, twins) (D-050).

Development (``dev=True``): forms fitted on the fold-training patients; training patients are
scored by the inner-fold model that did not see them. Final: test patients scored by the fold
model. No function here reads a test patient's outcome.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.features.fusion_data import load_grid, load_table
from src.features.make_splits import strata_for
from src.hybrid.models import base_columns
from src.mech.inputs import build_records
from src.mech.run_mech import by_patient
from src.models.fusion_lgbm import folds
from src.personalize.data import first_period
from src.utils.config import load_config
from src.whatif import twin as tw
from src.whatif.engine import PROC, FoldCtx, Period, part4_choices, twin_specs
from src.whatif.models import FORMS, fit_form, oof_guard

COHORT = "cgmacros"


def load_data(wc: dict[str, Any]) -> dict[str, Any]:
    """Cohort data shared by all folds."""
    fc, hc, fcfg = load_config("fusion"), load_config("hybrid"), load_config("features")
    mc, bc, sc = load_config("mech"), load_config("baselines"), load_config("splits")
    tab = load_table(COHORT, fc["n_hours"], fc)
    grid = load_grid(COHORT)
    meals = pd.read_parquet(PROC / "meals.parquet")
    meals = meals[meals.dataset == COHORT]
    recs = build_records(grid, meals, mc["activity"], mc["params"]["slow_protein_weight"], True)
    return {
        "tab": tab,
        "fp": first_period(tab),
        "grid": grid,
        "meals": meals,
        "recs": by_patient(recs),
        "cols": base_columns(COHORT, hc, tab),
        "strata": strata_for(pd.read_parquet(PROC / "static.parquet"), COHORT, sc),
        "mc": mc,
        "bc": bc,
        "pc": load_config("personalize"),
        "ectx": {
            "fcfg": fcfg,
            "step": fcfg["cohorts"][COHORT]["step_min"],
            "n_h": fc["n_hours"],
            "slice_hours": wc["slice_hours"],
            "tol": wc["feature_guard_tol"],
        },
        "folds": {f: (tr, te) for _, f, tr, te in folds(COHORT, [wc["cv_repeat"]])},
    }


def period(data: dict[str, Any], period_id: str) -> Period:
    """Period object (grid, meal log, anchor rows, twin record)."""
    rows = data["tab"][data["tab"].period_id == period_id].sort_values("t")
    pid = str(rows["patient_id"].iloc[0])
    rec = next(r for r in data["recs"][pid] if r.period_id == period_id)
    g = data["grid"][data["grid"].period_id == period_id].sort_values("t").reset_index(drop=True)
    m = data["meals"][data["meals"].period_id == period_id].sort_values("t")
    return Period(pid, period_id, g, m, rows.reset_index(drop=True), rec)


def assumed_alphas(fold: int, mc: dict, wc: dict) -> dict[str, float]:
    """D-052 alpha_E per target reduction, for the fold's training-population mean parameters."""
    mu = np.array(part4_choices(fold)["pop_prior_off"]["mu"])
    ac = wc["twin"]["assumed"]
    targets = [ac["reduction"], *ac["sensitivity"]]
    return {f"{r:g}": tw.calibrate_alpha(mu, mc, wc["walk"], ac, r) for r in targets}


def fold_ctx(data: dict[str, Any], wc: dict[str, Any], fold: int, dev: bool) -> FoldCtx:
    """Fit both forms (+ guards) and set up twins for one fold."""
    t0 = time.time()
    train, test = data["folds"][fold]
    fp = data["fp"]
    trf = fp[fp.patient_id.isin(train)].reset_index(drop=True)
    ctx = {
        "cols": data["cols"],
        "strata": data["strata"],
        "bc": data["bc"],
        "wc": wc,
        "inner_folds": data["pc"]["inner_folds"],
        "seed": 42 + fold,
    }
    forms = {n: fit_form(n, data["tab"], trf, train, ctx) for n in FORMS}
    p6 = pd.read_parquet(PROC / "uncertainty" / f"unc_A_{COHORT}_fold{fold}_oof.parquet")
    guard = oof_guard(forms["unc"], trf, p6, wc["oof_guard_tol"])
    mc = data["mc"]
    pids, role = (train, "train") if dev else (test, "test")
    twins = twin_specs(fold, pids, role, mc, wc["twin"]["k"])
    p4 = part4_choices(fold)
    extra = {
        "oof_guard_max_abs_diff": guard,
        "fit_seconds": round(time.time() - t0, 1),
        "n_train": len(train),
        "n_test": len(test),
        "constraints": {
            n: f.pop[60].est.get_params().get("monotone_constraints") is not None
            for n, f in forms.items()
        },
    }
    return FoldCtx(
        fold=fold,
        forms=forms,
        twins=twins,
        alpha_on=float(p4["alpha_on"]),
        alpha_assumed=assumed_alphas(fold, mc, wc),
        ectx=data["ectx"],
        wc=wc,
        dev=dev,
        cols=data["cols"],
        extra=extra,
    )
