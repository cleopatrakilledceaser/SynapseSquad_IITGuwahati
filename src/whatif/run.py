"""Part 7 runner (``make whatif``): plausibility suite + matched meals (D-050..D-053).

Stages:
  dev    fold(s) in ``dev_folds``: TRAINING patients only, scored by the inner-fold model that
         did not see them. Never touches a test patient.
  final  every fold: TEST patients scored by the fold model (CV+ bands); matched meals for every
         patient once; test MAE of mono vs unc. Runs ONCE: writes data/processed/whatif/FINAL_DONE;
         a re-run needs ``--force`` and is recorded (``forced_reruns``) and must be disclosed.
Outputs: data/processed/whatif/{dev,final}_{plaus,matched}.parquet, final_mae.parquet, json.
Usage: python -m src.whatif.run dev | final [--force]
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.hybrid.twin_features import HORIZONS
from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.whatif import context as cx
from src.whatif import matched as mt
from src.whatif import plausibility as pl
from src.whatif.engine import PROC

OUT = PROC / "whatif"
LOCK = OUT / "FINAL_DONE"


def check_lock(force: bool) -> int:
    """Number of earlier final runs; raises if the lock exists and force is not given."""
    if not LOCK.exists():
        return 0
    n = int(json.loads(LOCK.read_text()).get("runs", 1))
    if not force:
        raise RuntimeError(
            "final stage already ran (data/processed/whatif/FINAL_DONE). A re-run needs --force "
            "and must be disclosed to the owner (D-053)."
        )
    return n


def write_lock(n_before: int, force: bool, note: str = "") -> None:
    """Record the final run (count, forced re-runs); earlier log entries are kept."""
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    lock = {**old, "runs": n_before + 1}
    if force:
        n = int(old.get("forced_reruns", 0)) + 1
        lock["forced_reruns"] = n
        lock["forced_rerun_log"] = old.get("forced_rerun_log", []) + [
            f"{n}: final stage re-run{': ' + note if note else ''}"
        ]
    else:
        lock["forced_reruns"] = 0
    LOCK.write_text(json.dumps(lock, indent=1))


def _periods(data: dict, pids: list[str]) -> list[str]:
    return [r.period_id for p in pids for r in data["recs"].get(p, [])]


def _plaus_job(fc: Any, data: dict, period_ids: list[str], seed: int) -> list[dict]:
    out = []
    for i, per_id in enumerate(period_ids):
        recs = pl.run_period(fc, cx.period(data, per_id), seed + i)
        for r in recs:
            r["period_id"], r["fold"] = per_id, fc.fold
        out += recs
    return out


def plausibility(fc: Any, data: dict, pids: list[str], wc: dict) -> pd.DataFrame:
    """Suite over the given patients' periods (chunks in parallel processes)."""
    ids = _periods(data, pids)
    chunks = [ids[i :: wc["n_jobs"]] for i in range(wc["n_jobs"])]
    res = Parallel(n_jobs=wc["n_jobs"])(
        delayed(_plaus_job)(fc, data, c, wc["seed"] + 1000 * k) for k, c in enumerate(chunks) if c
    )
    return pd.DataFrame([r for part in res for r in part])


def matched(fcs: dict, data: dict, allowed: set[str], wc: dict) -> pd.DataFrame:
    """Matched-meal records for pairs whose patients are all in ``allowed``."""
    mc = wc["matched"]
    m = mt.clean_meals(data["meals"], data["tab"], mc)
    static = pd.read_parquet(PROC / "static.parquet")
    pairs = mt.within_pairs(m, mc) + mt.cross_pairs(m, static, mc)
    pairs = [p for p in pairs if {p["A"].patient_id, p["B"].patient_id} <= allowed]
    # dev: training patients of the dev fold (inner-fold models); final: test patients of each fold
    fold_of = {p: f for f, fc in fcs.items() for p in data["folds"][f][0 if fc.dev else 1]}
    pers = {pid: cx.period(data, pid) for pid in {p[s].period_id for p in pairs for s in "AB"}}
    out = []
    for p in pairs:
        pred = mt.predicted(fcs, pers, p, fold_of, mc["twin_min_day"])
        out += mt.pair_record(p, pred, mt.observed(data["tab"], p))
    return pd.DataFrame(out)


def dev(wc: dict) -> None:
    """Development stage: training patients of dev_folds only."""
    t0 = time.time()
    data = cx.load_data(wc)
    plaus, fcs, info = [], {}, {}
    for f in wc["dev_folds"]:
        fc = cx.fold_ctx(data, wc, f, dev=True)
        fcs[f] = fc
        info[f] = fc.extra | {"alpha_on": fc.alpha_on, "alpha_assumed": fc.alpha_assumed}
        plaus.append(plausibility(fc, data, data["folds"][f][0], wc))
    allowed = {p for f in wc["dev_folds"] for p in data["folds"][f][0]}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(plaus, ignore_index=True).to_parquet(OUT / "dev_plaus.parquet", index=False)
    matched(fcs, data, allowed, wc).to_parquet(OUT / "dev_matched.parquet", index=False)
    info["seconds"] = round(time.time() - t0, 1)
    (OUT / "dev.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"dev stage done: {info['seconds']:.0f}s", flush=True)


def test_mae(fc: Any, data: dict, wc: dict) -> pd.DataFrame:
    """Per-patient absolute errors of unc / mono on the fold's test patients (Part 6 region)."""
    fp = data["fp"]
    te = fp[fp.patient_id.isin(data["folds"][fc.fold][1])]
    te = te[te.days_since_start >= wc["band_region_from_day"]].reset_index(drop=True)
    clip = tuple(load_config("data")["cgm"]["glucose_range_mgdl"][cx.COHORT])
    out = te[["patient_id"]].copy()
    for h in HORIZONS:
        y = te[f"y_{h}"].to_numpy(float)
        for n, f in fc.forms.items():
            out[f"ae_{n}|{h}"] = np.abs(np.clip(f.predict(te, h), *clip) - y)
    return out


def _final_fold(wc: dict, fold: int) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    data = cx.load_data(wc)
    fc = cx.fold_ctx(data, wc, fold, dev=False)
    wc1 = {**wc, "n_jobs": 1}
    pl_df = plausibility(fc, data, data["folds"][fold][1], wc1)
    return fc, pl_df, test_mae(fc, data, wc)


def final(wc: dict, force: bool, note: str = "") -> None:
    """Final stage (run once): every fold's test patients, matched meals for every patient."""
    n_before = check_lock(force)
    t0 = time.time()
    res = Parallel(n_jobs=wc["n_jobs"])(
        delayed(_final_fold)(wc, f) for f in sorted(cx.load_data(wc)["folds"])
    )
    data = cx.load_data(wc)
    fcs = {fc.fold: fc for fc, _, _ in res}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([r[1] for r in res], ignore_index=True).to_parquet(
        OUT / "final_plaus.parquet", index=False
    )
    pd.concat([r[2] for r in res], ignore_index=True).to_parquet(
        OUT / "final_mae.parquet", index=False
    )
    allp = {p for f in fcs for p in data["folds"][f][1]}
    matched(fcs, data, allp, wc).to_parquet(OUT / "final_matched.parquet", index=False)
    info = {
        f: fc.extra | {"alpha_on": fc.alpha_on, "alpha_assumed": fc.alpha_assumed}
        for f, fc in fcs.items()
    }
    info["seconds"] = round(time.time() - t0, 1)
    info["forced_reruns"] = n_before if force else 0
    (OUT / "final.json").write_text(json.dumps(info, indent=1, default=str))
    write_lock(n_before, force, note)
    print(f"final stage done: {info['seconds']:.0f}s", flush=True)


def main() -> None:
    """CLI entry point."""
    wc = load_config("whatif")
    set_global_seed(wc["seed"])
    stage = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if stage == "dev":
        dev(wc)
    elif stage == "final":
        note = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--note=")), "")
        final(wc, "--force" in sys.argv, note)
    else:
        raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
