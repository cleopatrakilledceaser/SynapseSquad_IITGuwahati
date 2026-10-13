"""Part 3 runner (``make fusion``).

LightGBM ablation / leave-one-group-out / tiers, neural fusion (GRU + FiLM, several seeds), SHAP.
Resumable: every run is cached as a parquet file in data/processed/fusion/ and skipped if
present. Part 2 predictions are reused, never rerun.

Usage: python -m src.models.run_fusion [lgbm|events|tiers|neural|shap|all]
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.features.fusion_data import (
    config_columns,
    drop_groups,
    fingerstick_features,
    load_grid,
    load_table,
    resolve_groups,
)
from src.features.make_splits import strata_for
from src.models.experiment import inner_split
from src.models.fusion_lgbm import KEYS, folds, run_config
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seed import set_global_seed

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROC / "fusion"
HORIZONS = [30, 60, 120]


def ctx(cohort: str) -> dict[str, Any]:
    """Everything a cohort's runs need (table, groups, strata, configs, clip range)."""
    fc, bc, sc = load_config("fusion"), load_config("baselines"), load_config("splits")
    tab = load_table(cohort, fc["n_hours"], fc)
    static = pd.read_parquet(PROC / "static.parquet")
    return {
        "cohort": cohort,
        "fc": fc,
        "bc": bc,
        "tab": tab,
        "groups": resolve_groups(cohort, fc, list(tab.columns)),
        "strata": strata_for(static, cohort, sc),
        "clip": tuple(load_config("data")["cgm"]["glucose_range_mgdl"][cohort]),
    }


def _run(
    c: dict[str, Any],
    name: str,
    cols: list[str],
    target: str,
    base: str = "cgm_now",
    repeats: list[int] | None = None,
    events: list[str] | None = None,
    tab: pd.DataFrame | None = None,
) -> None:
    path = OUT / f"lgbm_{c['cohort']}_{name}_{target}.parquet"
    if path.exists():
        return
    t0 = time.time()
    reps = repeats if repeats is not None else c["fc"]["lgbm"]["repeats"]
    o = run_config(
        tab if tab is not None else c["tab"],
        cols,
        c["cohort"],
        reps,
        c["strata"],
        c["bc"],
        HORIZONS,
        target,
        base,
        c["clip"],
        events,
    )
    o.to_parquet(path, index=False)
    print(f"  {path.name}: {len(cols)} cols, {time.time() - t0:.0f}s", flush=True)


def stage_lgbm(c: dict[str, Any]) -> None:
    """Ablation configs (abs + delta) and leave-one-group-out from E (abs + delta)."""
    coh, fc, g = c["cohort"], c["fc"], c["groups"]
    for name, grp in fc["ablation"][coh].items():
        for tgt in fc["targets"]:
            # A_static + delta: the delta base is the current CGM value, so "A delta" means
            # "current glucose + static" (no other CGM history). Reported with that label (D-027).
            _run(c, name, config_columns(grp, g), tgt)
    full = config_columns(fc["ablation"][coh]["E_full"], g)
    for name, drop in fc["logo"][coh].items():
        for tgt in fc["targets"]:
            _run(c, name, drop_groups(full, drop, g), tgt)


def stage_events(c: dict[str, Any]) -> None:
    """LightGBM event classifiers (repeat 0) for the configs in fusion.yaml event_classifier."""
    coh, fc, g = c["cohort"], c["fc"], c["groups"]
    for name in fc["event_classifier"]["configs"]:
        if name not in fc["ablation"][coh]:
            continue
        _run(
            c,
            f"{name}_events",
            config_columns(fc["ablation"][coh][name], g),
            "delta",
            repeats=[0],
            events=fc["event_classifier"]["events"],
        )


def tier_table(c: dict[str, Any]) -> pd.DataFrame:
    """Anchor table + causal fingerstick features for every schedule (prefixed fs4_/fs2_)."""
    grid = load_grid(c["cohort"])
    tab = c["tab"].copy()
    fcfg = c["fc"]["fingerstick"]
    for sname, times in fcfg["schedules"].items():
        fs = fingerstick_features(grid, tab, times, fcfg["lookback_h"])
        for col in fs.columns:
            tab[f"{sname}_{col}"] = fs[col].to_numpy()
    return tab


def stage_tiers(c: dict[str, Any]) -> None:
    """Input tiers: T1 = full fusion (already run as E_full); T2/T3 replace CGM by fingersticks.

    Delta target base for T2/T3 = the last fingerstick (missing before the first stick of a
    period -> training-fold mean, see fusion_lgbm.fit_fold).
    """
    coh, fc, g = c["cohort"], c["fc"], c["groups"]
    tab = tier_table(c)
    fs_cache = OUT / f"fingersticks_{coh}.parquet"
    if not fs_cache.exists():
        tab[KEYS + [x for x in tab.columns if x.startswith(("fs4_", "fs2_"))]].to_parquet(
            fs_cache, index=False
        )
    for name, spec in fc["tiers"][coh].items():
        if "fingerstick" not in spec:
            continue
        s = spec["fingerstick"]
        cols = config_columns(spec["groups"], g) + [x for x in tab.columns if x.startswith(f"{s}_")]
        for tgt in fc["targets"]:
            _run(c, name, cols, tgt, base=f"{s}_fs_last", tab=tab)


NEURAL_CHANNELS = {
    "B_cgm": ["cgm", "tod"],
    "E_full_cgmacros": ["cgm", "wearable", "meals", "tod"],
    "E_full_shanghai_t2dm": ["cgm", "tod"],
}


def _neural_fit(seq, stat, y, base, tr_m, va_m, ev_m, hidden, seed, ncfg, threads):  # type: ignore[no-untyped-def]  # noqa: E501
    from src.models.neural import NeuralForecaster

    cfg = {**ncfg, "threads": threads}
    m = NeuralForecaster(hidden, ncfg["grid"]["dropout"][0], cfg, seed, target="delta")
    m.fit(
        (seq[tr_m], stat[tr_m], y[tr_m], base[tr_m]), (seq[va_m], stat[va_m], y[va_m], base[va_m])
    )
    pred = m.predict(seq[ev_m], stat[ev_m], base[ev_m]) if ev_m is not None else None
    return pred, m.best_val, m.best_epoch


def _neural_inputs(c: dict[str, Any], name: str) -> tuple[np.ndarray, ...]:
    from src.models.neural import CHANNELS, build_sequences

    coh, fc, tab = c["cohort"], c["fc"], c["tab"]
    key = name if name == "B_cgm" else f"{name}_{coh}"
    chans = [ch for grp in NEURAL_CHANNELS[key] for ch in CHANNELS[grp]]
    step = 5 if coh == "cgmacros" else 15
    meals = pd.read_parquet(PROC / "meals.parquet")
    seq = build_sequences(load_grid(coh), meals, tab, fc["n_hours"], step, 15, chans)
    st_cols = [
        x for x in config_columns(fc["ablation"][coh][name], c["groups"]) if x.startswith("static_")
    ]
    stat = tab[st_cols].to_numpy(float) if st_cols else np.zeros((len(tab), 0))
    y = tab[[f"y_{h}" for h in HORIZONS]].to_numpy(float)
    return seq, stat, y, tab["cgm_now"].to_numpy(float)


def _select_hidden(c: dict[str, Any], name: str, arrays: tuple[np.ndarray, ...]) -> int:
    """Hidden size chosen ONCE on holdout-train patients (inner split, seed 0); cached."""
    coh, ncfg = c["cohort"], c["fc"]["neural"]
    sel_path = OUT / f"neural_{coh}_selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    if name not in sel:
        seq, stat, y, base = arrays
        pid = c["tab"]["patient_id"].to_numpy()
        hold = pd.read_csv(PROC / "splits" / f"{coh}_holdout.csv")
        hold_tr = sorted(hold.loc[hold.split == "train", "patient_id"])
        in_tr, in_va = inner_split(hold_tr, c["strata"], c["bc"], 42)
        grid = ncfg["grid"]["hidden"]
        res = Parallel(n_jobs=len(grid))(
            delayed(_neural_fit)(
                seq, stat, y, base, np.isin(pid, in_tr), np.isin(pid, in_va), None, hd, 0, ncfg, 4
            )
            for hd in grid
        )
        losses = {str(hd): float(r[1]) for hd, r in zip(grid, res, strict=True)}
        sel[name] = {"val_loss": losses, "hidden": int(min(grid, key=lambda hd: losses[str(hd)]))}
        sel_path.write_text(json.dumps(sel, indent=1))
    return int(sel[name]["hidden"])


def stage_neural(c: dict[str, Any]) -> None:
    """GRU + FiLM over repeat-0 CV x seeds; 4 fits in parallel x 2 threads (measured fastest)."""
    coh, ncfg, tab = c["cohort"], c["fc"]["neural"], c["tab"]
    pid = tab["patient_id"].to_numpy()
    for name in ncfg["configs"]:
        path = OUT / f"neural_{coh}_{name}_delta.parquet"
        if path.exists():
            continue
        t0 = time.time()
        seq, stat, y, base = arrays = _neural_inputs(c, name)
        hidden = _select_hidden(c, name, arrays)
        jobs = []
        for rep, fno, tr_p, ev_p in folds(coh, ncfg["repeats"]):
            in_tr, in_va = inner_split(tr_p, c["strata"], c["bc"], 42 + rep)
            for seed in ncfg["seeds"]:
                jobs.append(
                    (rep, fno, seed, np.isin(pid, in_tr), np.isin(pid, in_va), np.isin(pid, ev_p))
                )
        res = Parallel(n_jobs=4)(
            delayed(_neural_fit)(seq, stat, y, base, a, b, e, hidden, s, ncfg, 2)
            for (_, _, s, a, b, e) in jobs
        )
        parts = []
        for (rep, fno, seed, _, _, e), (pred, _, be) in zip(jobs, res, strict=True):
            o = tab.loc[e, KEYS].copy()
            o["repeat"], o["fold"], o["seed"], o["best_epoch"] = rep, fno, seed, be
            for j, h in enumerate(HORIZONS):
                o[f"pred_{h}"] = np.clip(pred[:, j], *c["clip"])
            parts.append(o)
        pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
        print(
            f"  {path.name}: hidden={hidden}, {len(jobs)} fits, {time.time() - t0:.0f}s", flush=True
        )


def stage_shap(c: dict[str, Any]) -> None:
    """LightGBM gain + SHAP for E_full (delta, +60 min) fitted on holdout-train patients."""
    import shap

    from src.models.lgbm import LGBMForecaster

    coh, fc, bc, tab = c["cohort"], c["fc"], c["bc"], c["tab"]
    path = OUT / f"shap_{coh}.parquet"
    if path.exists():
        return
    h = fc["shap"]["horizon_min"]
    cols = config_columns(fc["ablation"][coh][fc["shap"]["config"]], c["groups"])
    hold = pd.read_csv(PROC / "splits" / f"{coh}_holdout.csv")
    in_tr, in_va = inner_split(
        sorted(hold.loc[hold.split == "train", "patient_id"]), c["strata"], bc, 42
    )
    m = LGBMForecaster(cols, bc["lightgbm"], seed=42, target="delta")
    m.fit(
        tab[tab.patient_id.isin(in_tr)],
        val=tab[tab.patient_id.isin(in_va)],
        horizons=[h],
        events=False,
    )
    booster = m.models[h]
    cal = tab[tab.patient_id.isin(hold.loc[hold.split == "calibration", "patient_id"])]
    x = cal[cols].sample(min(fc["shap"]["n_rows"], len(cal)), random_state=42)
    sv = shap.TreeExplainer(booster).shap_values(x)
    gain = booster.booster_.feature_importance("gain")
    out = pd.DataFrame({"feature": cols, "gain": gain, "mean_abs_shap": np.abs(sv).mean(0)})
    out["group"] = out.feature.map({col: grp for grp, cs in c["groups"].items() for col in cs})
    out.to_parquet(path, index=False)
    np.save(OUT / f"shap_values_{coh}.npy", sv)
    x.to_parquet(OUT / f"shap_x_{coh}.parquet", index=False)
    print(f"  {path.name}: {len(x)} rows (holdout-calibration patients)", flush=True)


def main() -> None:
    """Run the requested stage(s) for both cohorts (cached runs are skipped)."""
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    set_global_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    stages = {
        "lgbm": stage_lgbm,
        "events": stage_events,
        "tiers": stage_tiers,
        "neural": stage_neural,
        "shap": stage_shap,
    }
    todo = list(stages) if stage == "all" else [stage]
    for coh in ("cgmacros", "shanghai_t2dm"):
        c = ctx(coh)
        for s in todo:
            print(f"[{coh}] {s}", flush=True)
            stages[s](c)


if __name__ == "__main__":
    main()
