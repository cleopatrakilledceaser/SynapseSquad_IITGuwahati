"""Part 2 runner (``make models``): window selection, repeated patient-grouped CV, final holdout.

Steps
1. Choose N (input hours) by patient-grouped K-fold on the HOLDOUT-TRAIN patients only.
2. Headline: repeated patient-grouped K-fold over all patients of each cohort with the chosen N
   (every fold tunes on an inner split of its own training patients). Out-of-fold predictions
   are saved to data/processed/predictions/ (gitignored).
3. Final run, once: fit on holdout-train patients, predict holdout-test patients.
4. Shuffle check: LightGBM trained on labels shuffled across all training rows.
Metrics are computed by ``src.eval.report``.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from src.features.make_splits import strata_for
from src.models.experiment import fit_predict
from src.models.lgbm import LGBMForecaster
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seed import set_global_seed

PROC = PROJECT_ROOT / "data" / "processed"
PRED = PROC / "predictions"
COHORTS = {
    "cgmacros": {"grid": "cgmacros_ts5.parquet", "step": 5, "clip": "cgmacros"},
    "shanghai_t2dm": {"grid": "common_ts15.parquet", "step": 15, "clip": "shanghai_t2dm"},
}


def load_cohort(cohort: str, n_h: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Feature table for N, the cohort's grid, and feature sets."""
    tab = pd.read_parquet(PROC / "features" / f"{cohort}_N{n_h}.parquet")
    grid = pd.read_parquet(PROC / COHORTS[cohort]["grid"])
    grid = grid[grid.dataset == cohort].sort_values(["period_id", "t"]).reset_index(drop=True)
    sets = json.loads((PROC / "features" / "feature_sets.json").read_text())[cohort][f"N{n_h}"]
    return tab, grid, sets


def select_window(cohort: str, cfgs: dict[str, Any], strata: dict[str, str]) -> dict[str, Any]:
    """Patient-grouped CV on holdout-train patients for each N; returns MAE table + choice."""
    bcfg, fcfg, scfg = cfgs["b"], cfgs["f"], cfgs["s"]
    ws = bcfg["window_selection"]
    hold = pd.read_csv(PROC / "splits" / f"{cohort}_holdout.csv")
    train_p = sorted(hold.loc[hold.split == "train", "patient_id"])
    from src.data.splits import repeated_patient_group_kfold

    folds = list(
        repeated_patient_group_kfold(
            train_p, ws["n_splits"], 1, scfg["seed"], {p: strata[p] for p in train_p}
        )
    )
    res: dict[str, Any] = {}
    for n_h in fcfg["windows"]["input_hours"]:
        tab, _, sets = load_cohort(cohort, n_h)
        abs_err = {h: [] for h in ws["horizons_min"]}
        for f in folds:
            from src.models.experiment import inner_split

            tr_p, va_p = inner_split(sorted(f.train), strata, bcfg, scfg["seed"])
            m = LGBMForecaster(sets["cgm_tod"], bcfg["lightgbm"], seed=scfg["seed"])
            m.fit(
                tab[tab.patient_id.isin(tr_p)],
                val=tab[tab.patient_id.isin(va_p)],
                horizons=ws["horizons_min"],
                events=False,
            )
            ev = tab[tab.patient_id.isin(f.val)]
            p = m.predict(ev, horizons=ws["horizons_min"])
            for h in ws["horizons_min"]:
                abs_err[h].append(np.abs(p[h] - ev[f"y_{h}"].to_numpy()))
        res[f"N{n_h}"] = {str(h): float(np.nanmean(np.concatenate(v))) for h, v in abs_err.items()}
    mean_mae = {k: float(np.mean(list(v.values()))) for k, v in res.items()}
    best = min(mean_mae, key=mean_mae.get)
    return {
        "mae_by_N": res,
        "mean_mae_by_N": mean_mae,
        "chosen": best,
        "n_train_patients": len(train_p),
    }


def run_cv(cohort: str, n_h: int, cfgs: dict[str, Any], strata: dict[str, str]) -> pd.DataFrame:
    """Repeated patient-grouped K-fold out-of-fold predictions (one row per anchor per repeat)."""
    bcfg, fcfg, scfg = cfgs["b"], cfgs["f"], cfgs["s"]
    tab, grid, sets = load_cohort(cohort, n_h)
    cv = pd.read_csv(PROC / "splits" / f"{cohort}_cv.csv")
    clip = tuple(load_config("data")["cgm"]["glucose_range_mgdl"][cohort])
    parts = []
    for (rep, fold), g in cv.groupby(["repeat", "fold"]):
        ev_p = sorted(g.patient_id)
        tr_p = sorted(set(cv.patient_id) - set(ev_p))
        t0 = time.time()
        out, ch = fit_predict(
            tab,
            grid,
            tr_p,
            ev_p,
            strata,
            sets,
            COHORTS[cohort]["step"],
            clip,
            bcfg,
            fcfg,
            scfg["seed"] + int(rep),
        )
        out["repeat"], out["fold"] = int(rep), int(fold)
        parts.append(out)
        print(
            f"  {cohort} rep {rep} fold {fold}: {len(out)} rows, {time.time() - t0:.0f}s",
            flush=True,
        )
    return pd.concat(parts, ignore_index=True)


def run_holdout(cohort: str, n_h: int, cfgs: dict[str, Any], strata: dict[str, str]):  # type: ignore[no-untyped-def]
    """Final single run: train on holdout-train patients, predict holdout-test patients."""
    bcfg, fcfg, scfg = cfgs["b"], cfgs["f"], cfgs["s"]
    tab, grid, sets = load_cohort(cohort, n_h)
    hold = pd.read_csv(PROC / "splits" / f"{cohort}_holdout.csv")
    tr_p = sorted(hold.loc[hold.split == "train", "patient_id"])
    te_p = sorted(hold.loc[hold.split == "test", "patient_id"])
    clip = tuple(load_config("data")["cgm"]["glucose_range_mgdl"][cohort])
    out, ch = fit_predict(
        tab, grid, tr_p, te_p, strata, sets, COHORTS[cohort]["step"], clip, bcfg, fcfg, scfg["seed"]
    )
    return out, ch


def shuffle_check(cfgs: dict[str, Any], strata: dict[str, str], n_h: int) -> dict[str, Any]:
    """Shuffle check: LightGBM trained on labels/targets shuffled across ALL training rows.

    Shuffling destroys any feature-label link, so test AUROC should be about 0.5 and regression
    MAE no better than predicting the training mean. Repeated ``n_shuffles`` times.
    """
    from sklearn.metrics import roc_auc_score

    from src.models.experiment import inner_split

    bcfg, scfg = cfgs["b"], cfgs["s"]
    sc = bcfg["shuffle_check"]
    h, ev = sc["horizon_min"], sc["event"]
    tab, _, sets = load_cohort(sc["cohort"], n_h)
    hold = pd.read_csv(PROC / "splits" / f"{sc['cohort']}_holdout.csv")
    tr_p = sorted(hold.loc[hold.split == "train", "patient_id"])
    te = tab[tab.patient_id.isin(hold.loc[hold.split == "calibration", "patient_id"])]
    in_tr, in_va = inner_split(tr_p, strata, bcfg, scfg["seed"])
    rng = np.random.default_rng(scfg["seed"])
    lcfg = {**bcfg["lightgbm"], "classifier_events": [ev.rsplit("_", 1)[0]]}
    ok = te[ev].notna().to_numpy()
    okr = te[f"y_{h}"].notna().to_numpy()
    res: dict[str, Any] = {"true": {}, "shuffled": []}
    for i in range(sc["n_shuffles"] + 1):
        tr = tab[tab.patient_id.isin(in_tr)].copy()
        va = tab[tab.patient_id.isin(in_va)].copy()
        if i > 0:
            for d in (tr, va):
                d[ev] = rng.permutation(d[ev].to_numpy())
                d[f"y_{h}"] = rng.permutation(d[f"y_{h}"].to_numpy())
        m = LGBMForecaster(sets["cgm_tod"], lcfg, seed=scfg["seed"])
        m.fit(tr, val=va, horizons=[h], events=True)
        auc = float(roc_auc_score(te[ev].to_numpy()[ok], m.predict_events(te)[ev][ok]))
        mae = float(np.mean(np.abs(m.predict(te, horizons=[h])[h][okr] - te[f"y_{h}"][okr])))
        r = {"auroc": auc, "mae": mae}
        if i == 0:
            res["true"] = r
        else:
            res["shuffled"].append(r)
    sh = pd.DataFrame(res["shuffled"])
    mean_pred_mae = float(
        np.mean(np.abs(te[f"y_{h}"][okr] - tab.loc[tab.patient_id.isin(in_tr), f"y_{h}"].mean()))
    )
    return {
        "event": ev,
        "horizon_min": h,
        "eval": "holdout calibration patients",
        "true": res["true"],
        "shuffled_auroc_mean": float(sh.auroc.mean()),
        "shuffled_auroc_range": [float(sh.auroc.min()), float(sh.auroc.max())],
        "shuffled_mae_mean": float(sh.mae.mean()),
        "mae_predict_train_mean": mean_pred_mae,
        "n_shuffles": int(sc["n_shuffles"]),
    }


def main() -> None:
    """Run window selection, repeated CV, final holdout and the shuffle check.

    ``--shuffle-only`` re-runs only the shuffle check (reusing run_summary.json).
    """
    import sys

    cfgs = {"b": load_config("baselines"), "f": load_config("features"), "s": load_config("splits")}
    set_global_seed(cfgs["s"]["seed"])
    PRED.mkdir(parents=True, exist_ok=True)
    static = pd.read_parquet(PROC / "static.parquet")
    if "--shuffle-only" in sys.argv:
        summary = json.loads((PRED / "run_summary.json").read_text())
        summary["shuffle_check"] = shuffle_check(
            cfgs, strata_for(static, "cgmacros", cfgs["s"]), summary["cgmacros"]["N"]
        )
        sc = summary["shuffle_check"]
        print("shuffle check:", sc["true"], "| shuffled AUROC", round(sc["shuffled_auroc_mean"], 3))
        (PRED / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str), "utf-8")
        return
    summary = {}
    for cohort in cfgs["b"]["cohorts"]:
        strata = strata_for(static, cohort, cfgs["s"])
        sel = select_window(cohort, cfgs, strata)
        n_h = int(sel["chosen"][1:])
        print(f"{cohort}: window selection {sel['mean_mae_by_N']} -> N={n_h}", flush=True)
        oof = run_cv(cohort, n_h, cfgs, strata)
        oof.to_parquet(PRED / f"{cohort}_cv_oof.parquet", index=False)
        hold, choices = run_holdout(cohort, n_h, cfgs, strata)
        hold.to_parquet(PRED / f"{cohort}_holdout_test.parquet", index=False)
        summary[cohort] = {"window_selection": sel, "N": n_h, "holdout_choices": choices}
    summary["shuffle_check"] = shuffle_check(
        cfgs, strata_for(static, "cgmacros", cfgs["s"]), summary["cgmacros"]["N"]
    )
    sc = summary["shuffle_check"]
    print(
        "shuffle check:", sc["true"], "| shuffled AUROC mean", round(sc["shuffled_auroc_mean"], 3)
    )
    (PRED / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str), "utf-8")


if __name__ == "__main__":
    main()
