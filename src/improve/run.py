"""Part 10 runner (D-065): tuned LightGBM vs the status-quo LightGBM.

``python -W ignore -u -m src.improve.run cv``       outer CV (repeat 0), paired with Part 3
``python -W ignore -u -m src.improve.run holdout``  single-shot confirmation on the holdout test
                                                    patients; refuses to run twice (lock = output).
Writes reports/metrics/improve.json and improve_holdout.json.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.features.fusion_data import config_columns
from src.models.experiment import inner_split
from src.models.fusion_lgbm import KEYS, folds
from src.models.lgbm import LGBMForecaster
from src.models.run_fusion import ctx
from src.models.tuned import TunedLGBM
from src.transfer import core as tc
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.provenance import file_hash, write_metrics

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROC / "improve"
METRICS = PROJECT_ROOT / "reports" / "metrics"


def _predict(model: Any, ev: pd.DataFrame, hz: list[int], clip: tuple[float, float]) -> pd.DataFrame:
    out = ev[KEYS].copy()
    for h, p in model.predict(ev, horizons=hz).items():
        out[f"pred_{h}"] = np.clip(p, *clip)
    return out


def fit_both(
    tab: pd.DataFrame, cols: list[str], tr_p: list[str], va_p: list[str], ev_p: list[str],
    c: dict[str, Any], icfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:  # fmt: skip
    """Fit the status-quo and the tuned model on ``tr_p`` (validate on ``va_p``); predict ``ev_p``."""
    hz, clip = icfg["horizons_min"], c["clip"]
    if tab["cgm_now"].isna().any():
        tab = tab.copy()
        tab["cgm_now"] = tab["cgm_now"].fillna(float(tab.loc[tab.patient_id.isin(tr_p), "cgm_now"].mean()))
    tr, va, ev = (tab[tab.patient_id.isin(p)] for p in (tr_p, va_p, ev_p))
    bl = c["bc"]["lightgbm"]
    sq = LGBMForecaster(
        cols, {**bl, "params": {**bl["params"], "n_jobs": icfg["lgbm_threads"]}, "classifier_events": []},
        seed=icfg["seed"], target="delta", base="cgm_now",
    )  # fmt: skip
    sq.fit(tr, val=va, horizons=hz, events=False)
    tu = TunedLGBM(
        cols, {}, seed=icfg["seed"], target="delta", base="cgm_now",
        tcfg=icfg["tuning"], n_jobs=icfg["lgbm_threads"],
    )  # fmt: skip
    tu.fit(tr, val=va, horizons=hz, events=False)
    return _predict(sq, ev, hz, clip), _predict(tu, ev, hz, clip), tu.chosen


def _cv_fold(tab, cols, fold, strata, c, icfg):  # type: ignore[no-untyped-def]
    rep, fno, tr_p, ev_p = fold
    in_tr, in_va = inner_split(tr_p, strata, c["bc"], icfg["seed"] + rep)
    sq, tu, chosen = fit_both(tab, cols, in_tr, in_va, ev_p, c, icfg)
    for d in (sq, tu):
        d["repeat"], d["fold"] = rep, fno
    return sq, tu, {"repeat": rep, "fold": fno, "chosen": chosen}


def evaluate(
    sq: pd.DataFrame, tu: pd.DataFrame, tab: pd.DataFrame, hz: list[int], boot: dict[str, Any]
) -> dict[str, Any]:  # fmt: skip
    """MAE for persistence / status quo / tuned and the paired tuned - status-quo difference."""
    y = tab[KEYS + ["cgm_now"] + [f"y_{h}" for h in hz]]
    df = sq.merge(tu, on=KEYS, suffixes=("_status_quo", "_tuned"), validate="1:1")
    df = df.merge(y, on=KEYS, validate="1:1")
    for h in hz:
        df[f"pred_persistence_{h}"] = df["cgm_now"]
    return {
        "n_patients": int(df["patient_id"].nunique()),
        "mae": tc.mae_rows(df, ["persistence"], hz, boot) + _named(df, hz, boot),
        "diff": tc.diff_rows(df, [("tuned", "status_quo")], hz, boot),
    }


def _named(df: pd.DataFrame, hz: list[int], boot: dict[str, Any]) -> list[dict[str, Any]]:
    for h in hz:
        for m in ("status_quo", "tuned"):
            df[f"pred_{m}_{h}"] = df[f"pred_{h}_{m}"]
    return tc.mae_rows(df, ["status_quo", "tuned"], hz, boot)


def stage_cv() -> None:
    icfg, bc = load_config("improve"), load_config("baselines")
    metrics: dict[str, Any] = {"design": "A: outer CV repeat 0, paired with Part 3", "runs": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    for cohort, names in icfg["configs"].items():
        c = ctx(cohort)
        for name in names:
            t0 = time.time()
            cols = config_columns(c["fc"]["ablation"][cohort][name], c["groups"])
            fl = folds(cohort, icfg["cv_repeats"])
            res = Parallel(n_jobs=icfg["n_parallel"])(
                delayed(_cv_fold)(c["tab"], cols, f, c["strata"], c, icfg) for f in fl
            )
            sq = pd.concat([r[0] for r in res], ignore_index=True)
            tu = pd.concat([r[1] for r in res], ignore_index=True)
            sq.to_parquet(OUT / f"cv_{cohort}_{name}_status_quo.parquet", index=False)
            tu.to_parquet(OUT / f"cv_{cohort}_{name}_tuned.parquet", index=False)
            run = evaluate(sq, tu, c["tab"], icfg["horizons_min"], bc["bootstrap"])
            run["selection"] = [r[2] for r in res]
            metrics["runs"][f"{cohort}|{name}"] = run
            print(f"  {cohort} {name}: {time.time() - t0:.0f}s", flush=True)
            _print(f"{cohort} {name}", run, icfg["horizons_min"])
    write_metrics("improve", metrics, icfg, icfg["seed"], _data_hash(icfg))


def stage_holdout() -> None:
    out = METRICS / "improve_holdout.json"
    if out.exists():
        sys.exit("improve_holdout.json exists: the single-shot holdout was already used (D-065).")
    icfg, bc = load_config("improve"), load_config("baselines")
    metrics: dict[str, Any] = {"design": "B: holdout test patients, single shot", "runs": {}}
    for cohort, names in icfg["configs"].items():
        c = ctx(cohort)
        hold = pd.read_csv(PROC / "splits" / f"{cohort}_holdout.csv")
        tr_p, va_p, ev_p = (sorted(hold.loc[hold.split == s, "patient_id"]) for s in ("train", "calibration", "test"))
        for name in names:
            cols = config_columns(c["fc"]["ablation"][cohort][name], c["groups"])
            sq, tu, chosen = fit_both(c["tab"], cols, tr_p, va_p, ev_p, c, icfg)
            run = evaluate(sq, tu, c["tab"], icfg["horizons_min"], bc["bootstrap"])
            run["selection"] = chosen
            metrics["runs"][f"{cohort}|{name}"] = run
            _print(f"holdout {cohort} {name}", run, icfg["horizons_min"])
    write_metrics("improve_holdout", metrics, icfg, icfg["seed"], _data_hash(icfg))


def _data_hash(icfg: dict[str, Any]):  # type: ignore[no-untyped-def]
    fc = load_config("fusion")
    files = [PROC / "static.parquet"] + [
        PROC / "features" / f"{co}_N{fc['n_hours']}.parquet" for co in icfg["configs"]
    ]
    return file_hash(files)


def _print(title: str, run: dict[str, Any], hz: list[int]) -> None:
    print(f"== {title} ({run['n_patients']} patients)")
    for h in hz:
        m = {r["model"]: r["mae"] for r in run["mae"] if r["horizon_min"] == h}
        d = next(r for r in run["diff"] if r["horizon_min"] == h)
        ci = d["diff_mae_ci"]
        print(f"  +{h}: persistence {m['persistence']:.2f} status quo {m['status_quo']:.2f} "
              f"tuned {m['tuned']:.2f} diff {d['diff_mae']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")  # fmt: skip


if __name__ == "__main__":
    {"cv": stage_cv, "holdout": stage_holdout}[sys.argv[1]]()
