"""Part 4 runner (``make mech``): mechanistic twin vs comparators on identical test windows.

Stages (cached in data/processed/mech/, skipped if present):
  twin      CGMacros twin variants per fold (CV repeat 0)
  ar        per-patient AR refit on the first k days (k = 1, 3)
  lgbm      population LightGBM, CGM + time of day + meals, delta target (same folds)
  shanghai  meal-timing-only twin, k = 3 (pooled dose x per-patient fc) + its comparators
  identify  synthetic recovery + Laplace / profile likelihood
Usage: python -m src.mech.run_mech [twin|ar|lgbm|shanghai|identify|all]
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from src.features.fusion_data import config_columns, load_grid, load_table, resolve_groups
from src.features.make_splits import strata_for
from src.mech.baselines_k import ar_first_k
from src.mech.fit import config_prior
from src.mech.inputs import Record, build_records
from src.mech.model import FIT_PARAMS
from src.mech.protocol import Ctx, run_fold
from src.models.fusion_lgbm import folds, run_config
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seed import set_global_seed

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROC / "mech"
HORIZONS = [30, 60, 120]


def by_patient(recs: dict[str, Record]) -> dict[str, list[Record]]:
    """Patient -> records sorted by start time."""
    out: dict[str, list[Record]] = {}
    for r in sorted(recs.values(), key=lambda r: r.t[0]):
        out.setdefault(r.patient_id, []).append(r)
    return out


def make_ctx(cohort: str, mc: dict[str, Any], fc: dict[str, Any]) -> Ctx:
    """Build the protocol context for a cohort."""
    grid = load_grid(cohort)
    meals = pd.read_parquet(PROC / "meals.parquet")
    meals = meals[meals.dataset == cohort]
    sh = cohort != "cgmacros"
    recs = build_records(
        grid,
        meals,
        mc["activity"],
        mc["params"]["slow_protein_weight"],
        use_activity=not sh,
        pooled_dose=mc["shanghai"]["pooled_dose_g"] if sh else None,
    )
    tab = load_table(cohort, fc["n_hours"], fc)
    keep = ["patient_id", "period_id", "t", "days_since_start", "cgm_now"]
    keep += [f"y_{h}" for h in HORIZONS]
    return Ctx(
        cohort=cohort,
        mc=mc,
        by_pat=by_patient(recs),
        anchors=tab[keep],
        static=pd.read_parquet(PROC / "static.parquet"),
        nsub=mc["shanghai"]["n_sub"] if sh else mc["solver"]["n_sub"],
        free=mc["shanghai"]["free"] if sh else list(FIT_PARAMS),
        base_logp=config_prior(mc["params"]).mu.copy(),
    )


def _fold_job(cohort: str, fold: int, train: list[str], test: list[str], ks: list[int]) -> None:
    mc, fc = load_config("mech"), load_config("fusion")
    path = OUT / f"twin_{cohort}_fold{fold}.parquet"
    if path.exists():
        return
    t0 = time.time()
    c = make_ctx(cohort, mc, fc)
    train = [p for p in train if p in c.by_pat]
    test = [p for p in test if p in c.by_pat]
    preds, choices, fits = run_fold(c, fold, train, test, ks)
    preds.to_parquet(path, index=False)
    fits.to_parquet(OUT / f"twinfits_{cohort}_fold{fold}.parquet", index=False)
    choices["seconds"] = time.time() - t0
    (OUT / f"twin_{cohort}_fold{fold}.json").write_text(json.dumps(choices, indent=1, default=str))
    print(f"  {path.name}: {time.time() - t0:.0f}s", flush=True)


def stage_twin(cohort: str, ks: list[int], n_jobs: int = 5) -> None:
    """Twin variants for every fold of CV repeat 0 (folds in parallel processes)."""
    mc = load_config("mech")
    fl = folds(cohort, [mc["cv_repeat"]])
    Parallel(n_jobs=n_jobs)(delayed(_fold_job)(cohort, f, tr, te, ks) for _, f, tr, te in fl)


def stage_ar(cohort: str) -> None:
    """Per-patient AR refit on exactly the first k days (k >= 1), every fold."""
    path = OUT / f"ar_{cohort}.parquet"
    if path.exists():
        return
    mc, fc, bc, sc = (load_config(n) for n in ("mech", "fusion", "baselines", "splits"))
    tab = load_table(cohort, fc["n_hours"], fc)
    grid = load_grid(cohort)
    strata = strata_for(pd.read_parquet(PROC / "static.parquet"), cohort, sc)
    step = 5 if cohort == "cgmacros" else 15
    ks = [k for k in mc["adapt_days"] if k >= 1]
    if cohort != "cgmacros":
        ks = [mc["shanghai"]["k"]]
    par = mc["patient_ar"]
    parts = []
    for _, f, tr, te in folds(cohort, [mc["cv_repeat"]]):
        o = ar_first_k(
            tab,
            grid,
            tr,
            te,
            ks,
            strata,
            bc,
            par["ridge_lambda"],
            par["min_pairs"],
            step,
            mc["test_from_day"],
        )
        o["fold"] = f
        parts.append(o)
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    print(f"  {path.name}", flush=True)


def stage_lgbm(cohort: str) -> None:
    """Population LightGBM (CGM + tod + meals, delta), CV repeat 0, same folds as the twin."""
    path = OUT / f"lgbm_{cohort}.parquet"
    if path.exists():
        return
    mc, fc, bc, sc = (load_config(n) for n in ("mech", "fusion", "baselines", "splits"))
    tab = load_table(cohort, fc["n_hours"], fc)
    groups = resolve_groups(cohort, fc, list(tab.columns))
    use = [g for g in mc["lgbm"]["groups"] if g in groups]
    cols = config_columns(use, groups)
    strata = strata_for(pd.read_parquet(PROC / "static.parquet"), cohort, sc)
    clip = tuple(load_config("data")["cgm"]["glucose_range_mgdl"][cohort])
    o = run_config(
        tab, cols, cohort, [mc["cv_repeat"]], strata, bc, HORIZONS, "delta", "cgm_now", clip
    )
    o.to_parquet(path, index=False)
    print(f"  {path.name}: groups {use}, {len(cols)} cols", flush=True)


def main() -> None:
    """Run the requested stage(s)."""
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    set_global_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    mc = load_config("mech")
    todo = ["twin", "ar", "lgbm", "identify", "shanghai"] if stage == "all" else [stage]
    for s in todo:
        t0 = time.time()
        print(f"[{s}]", flush=True)
        if s == "twin":
            stage_twin("cgmacros", mc["adapt_days"])
        elif s == "ar":
            stage_ar("cgmacros")
        elif s == "lgbm":
            stage_lgbm("cgmacros")
        elif s == "shanghai" and mc["shanghai"]["enabled"]:
            stage_twin("shanghai_t2dm", [mc["shanghai"]["k"]])
            stage_ar("shanghai_t2dm")
            stage_lgbm("shanghai_t2dm")
        elif s == "identify":
            from src.mech.identify import run_identify

            run_identify()
        print(f"[{s}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
