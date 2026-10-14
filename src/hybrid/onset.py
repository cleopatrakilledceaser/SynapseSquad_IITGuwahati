"""Neural onset fix (Part 5, D-042): Part 3 GRU + FiLM retrained with an onset-weighted L1 loss.

Weight per (window, horizon) = w if the window starts below the threshold (cgm_now < 140 mg/dL)
and ends above it (y(t + h) > 140), else 1. The weight uses the TRAINING target only (loss
weighting, never an input). Same channels, hidden size (Part 3 selection), CV repeat 0 folds,
inner patient split and seeds as Part 3. w is chosen ONCE per (cohort, config) on holdout-train
patients (inner split, seed 0) by the mean of overall and onset validation MAE (mg/dL).
Output: data/processed/hybrid/onset_<cohort>_<config>.parquet (pred_<h> per seed).
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.hybrid.build import OUT, PROC
from src.models.experiment import inner_split
from src.models.fusion_lgbm import KEYS, folds
from src.models.run_fusion import _neural_inputs, ctx
from src.utils.config import load_config

HORIZONS = [30, 60, 120]


def onset_weights(y: np.ndarray, base: np.ndarray, thr: float, w: float) -> np.ndarray:
    """(n, n_h) loss weights: w where base < thr and y > thr (NaN y -> weight 1, masked later)."""
    with np.errstate(invalid="ignore"):
        on = (base[:, None] < thr) & (y > thr)
    return np.where(on, w, 1.0).astype(np.float32)


def _fit(arrays, tr_m, va_m, ev_m, hidden, seed, ncfg, threads, w, thr):  # type: ignore[no-untyped-def]  # noqa: E501
    from src.models.neural import NeuralForecaster

    seq, stat, y, base = arrays
    cfg = {**ncfg, "threads": threads}
    m = NeuralForecaster(hidden, ncfg["grid"]["dropout"][0], cfg, seed, target="delta")
    wt = onset_weights(y, base, thr, w)
    m.fit(
        (seq[tr_m], stat[tr_m], y[tr_m], base[tr_m]),
        (seq[va_m], stat[va_m], y[va_m], base[va_m]),
        wt[tr_m],
        wt[va_m],
    )
    pv = m.predict(seq[va_m], stat[va_m], base[va_m])
    yv, bv = y[va_m], base[va_m]
    ok = ~np.isnan(yv)
    with np.errstate(invalid="ignore"):
        on = ok & (bv[:, None] < thr) & (yv > thr)
    err = np.abs(pv - np.nan_to_num(yv))
    scores = {"val_mae": float(err[ok].mean()), "val_onset_mae": float(err[on].mean())}
    pred = m.predict(seq[ev_m], stat[ev_m], base[ev_m]) if ev_m is not None else None
    return pred, scores, m.best_epoch


def _select_w(c: dict[str, Any], name: str, arrays: tuple, hidden: int, oc: dict) -> dict:
    """W chosen once on holdout-train patients (inner split, seed 0); cached."""
    coh, ncfg = c["cohort"], c["fc"]["neural"]
    path = OUT / f"onset_{coh}_selection.json"
    sel = json.loads(path.read_text()) if path.exists() else {}
    if name not in sel:
        pid = c["tab"]["patient_id"].to_numpy()
        hold = pd.read_csv(PROC / "splits" / f"{coh}_holdout.csv")
        hold_tr = sorted(hold.loc[hold.split == "train", "patient_id"])
        in_tr, in_va = inner_split(hold_tr, c["strata"], c["bc"], 42)
        grid = oc["weight_grid"]
        res = Parallel(n_jobs=len(grid))(
            delayed(_fit)(
                arrays,
                np.isin(pid, in_tr),
                np.isin(pid, in_va),
                None,
                hidden,
                0,
                ncfg,
                4,
                w,
                oc["threshold_mgdl"],
            )
            for w in grid
        )
        sc = {str(w): r[1] for w, r in zip(grid, res, strict=True)}
        crit = {w: 0.5 * (v["val_mae"] + v["val_onset_mae"]) for w, v in sc.items()}
        sel[name] = {"scores": sc, "criterion": crit, "w": float(min(crit, key=crit.get))}
        path.write_text(json.dumps(sel, indent=1))
    return sel[name]


def stage_onset() -> None:
    """Onset-weighted retraining of nn_B_cgm and nn_E_full, both cohorts (4 fits x 2 threads)."""
    hc = load_config("hybrid")
    oc = hc["onset_fix"]
    for coh in ("cgmacros", "shanghai_t2dm"):
        c = ctx(coh)
        ncfg, tab = c["fc"]["neural"], c["tab"]
        hsel = json.loads((PROC / "fusion" / f"neural_{coh}_selection.json").read_text())
        pid = tab["patient_id"].to_numpy()
        for name in oc["configs"]:
            path = OUT / f"onset_{coh}_{name}.parquet"
            if path.exists():
                continue
            t0 = time.time()
            arrays = _neural_inputs(c, name)
            hidden = int(hsel[name]["hidden"])
            w = _select_w(c, name, arrays, hidden, oc)["w"]
            jobs = []
            for rep, fno, tr_p, ev_p in folds(coh, ncfg["repeats"]):
                in_tr, in_va = inner_split(tr_p, c["strata"], c["bc"], 42 + rep)
                for seed in ncfg["seeds"]:
                    m = (np.isin(pid, in_tr), np.isin(pid, in_va), np.isin(pid, ev_p))
                    jobs.append((rep, fno, seed, *m))
            res = Parallel(n_jobs=4)(
                delayed(_fit)(arrays, a, b, e, hidden, s, ncfg, 2, w, oc["threshold_mgdl"])
                for (_, _, s, a, b, e) in jobs
            )
            parts = []
            for (rep, fno, seed, _, _, e), (pred, _, be) in zip(jobs, res, strict=True):
                o = tab.loc[e, KEYS].copy()
                o["repeat"], o["fold"], o["seed"], o["best_epoch"], o["w"] = rep, fno, seed, be, w
                for j, h in enumerate(HORIZONS):
                    o[f"pred_{h}"] = np.clip(pred[:, j], *c["clip"])
                parts.append(o)
            pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
            print(f"  {path.name}: w={w:g}, {len(jobs)} fits, {time.time() - t0:.0f}s", flush=True)
