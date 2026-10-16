"""D-056 (POST-HOC, not pre-registered): twin walk tests with the one-sided activity term.

Same windows as Part 7: the same periods, the same per-period seeds and the same sampler
(``plausibility.windows``); the meal windows are drawn first, exactly as in ``run_period``, so the
twin walk windows are identical. Only the twin walk tests are run (activity direction + walk AUC
effect) for ``twin_learned_d056`` (Part 4 alpha_on) and ``twin_assumed_d056`` (alpha re-calibrated
with the D-056 kernel to the D-052 target).
Usage: python -m src.whatif.d056 dev | final --force
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.utils.config import load_config
from src.whatif import context as cx
from src.whatif import edits as ed
from src.whatif import plausibility as pl
from src.whatif import run as wr
from src.whatif import twin as tw
from src.whatif.engine import part4_choices, twin_run

LABEL = "post-hoc, not pre-registered (D-056)"


def onesided_ctx(fc: object, mc: dict, wc: dict) -> dict[str, float]:
    """Switch the fold's twins to the D-056 kernel; return the D-056 alphas."""
    fc.twins = {p: dataclasses.replace(s, onesided=True) for p, s in fc.twins.items()}
    mu = np.array(part4_choices(fc.fold)["pop_prior_off"]["mu"])
    ac = wc["twin"]["assumed"]
    al = {"twin_learned_d056": fc.alpha_on}
    al["twin_assumed_d056"] = tw.calibrate_alpha(mu, mc, wc["walk"], ac, ac["reduction"], True)
    for r in ac["sensitivity"]:
        al[f"assumed_sens_{r:g}"] = tw.calibrate_alpha(mu, mc, wc["walk"], ac, r, True)
    return al


def walk_period(fc: object, data: dict, per_id: str, seed: int, alphas: dict) -> list[dict]:
    """Twin walk tests on the Part 7 meal windows of one period (same sampling order)."""
    per = cx.period(data, per_id)
    if per.patient_id not in fc.twins:
        return []
    rng = np.random.default_rng(seed)
    out = []
    for t, meal in pl.windows(per, fc.wc["plausibility"], rng, "meal"):
        for name in ("twin_learned_d056", "twin_assumed_d056"):
            al = alphas[name]
            r0, r1 = twin_run(fc, per, t, [], al), twin_run(fc, per, t, [ed.add_walk(meal)], al)
            (p0, a0), (p1, a1) = pl.tw_peak_auc(r0), pl.tw_peak_auc(r1)
            ok = bool(p1 <= p0 + 1e-9 and a1 <= a0 + 1e-9)
            base = {"model": name, "patient_id": per.patient_id, "period_id": per_id,
                    "fold": fc.fold, "by_construction": False, "note": LABEL}  # fmt: skip
            out.append(base | {"test": "activity_direction", "pass": ok, "value": p1 - p0})
            out.append(base | {"test": "walk_effect_auc", "pass": None, "value": a1 - a0})
    return out


def _fold(wc: dict, fold: int, dev: bool) -> tuple[pd.DataFrame, dict]:
    data = cx.load_data(wc)
    fc = cx.fold_ctx(data, wc, fold, dev=dev)
    alphas = onesided_ctx(fc, data["mc"], wc)
    pids = data["folds"][fold][0 if dev else 1]
    ids = wr._periods(data, pids)
    # Part 7 seeds: chunk k of n_jobs gets seed + 1000 k (dev); final used n_jobs = 1
    nj = wc["n_jobs"] if dev else 1
    chunks = [ids[i::nj] for i in range(nj)]
    recs = []
    for k, c in enumerate(chunks):
        for i, per_id in enumerate(c):
            recs += walk_period(fc, data, per_id, wc["seed"] + 1000 * k + i, alphas)
    return pd.DataFrame(recs), alphas


def main() -> None:
    """dev: fold-0 training windows; final: test windows (forced re-run of the locked stage)."""
    wc = load_config("whatif")
    stage = sys.argv[1] if len(sys.argv) > 1 else "dev"
    t0 = time.time()
    if stage == "dev":
        d, al = _fold(wc, wc["dev_folds"][0], True)
        d.to_parquet(wr.OUT / "d056_dev_walk.parquet", index=False)
        info = {"alphas": {wc["dev_folds"][0]: al}}
    elif stage == "final":
        n_before = wr.check_lock("--force" in sys.argv)
        res = Parallel(n_jobs=wc["n_jobs"])(delayed(_fold)(wc, f, False) for f in range(5))
        pd.concat([r[0] for r in res], ignore_index=True).to_parquet(
            wr.OUT / "d056_final_walk.parquet", index=False
        )
        info = {"alphas": dict(enumerate(r[1] for r in res)), "scope": "twin walk tests only"}
        lock = json.loads(wr.LOCK.read_text())
        lock["runs"] = n_before + 1
        lock["forced_reruns"] = int(lock.get("forced_reruns", 0)) + 1
        lock["forced_rerun_log"] = lock.get("forced_rerun_log", []) + [
            "1: D-056 post-hoc twin walk tests only (owner-approved)"
        ]
        wr.LOCK.write_text(json.dumps(lock, indent=1))
    else:
        raise SystemExit(f"unknown stage {stage}")
    info["seconds"] = round(time.time() - t0, 1)
    (wr.OUT / f"d056_{stage}.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"D-056 {stage}: {info['seconds']:.0f}s", flush=True)


if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] != "part4"):
    main()


def part4_fold(fold: int, train: list[str], test: list[str]) -> None:
    """Part 4 CGMacros twin protocol, unchanged except the D-056 kernel; separate output files."""
    from copy import deepcopy

    from src.mech.protocol import run_fold
    from src.mech.run_mech import make_ctx

    mc = deepcopy(load_config("mech"))
    mc["params"]["fixed"]["act_onesided"] = 1
    c = make_ctx("cgmacros", mc, load_config("fusion"))
    train = [p for p in train if p in c.by_pat]
    test = [p for p in test if p in c.by_pat]
    preds, choices, _ = run_fold(c, fold, train, test, mc["adapt_days"])
    preds.to_parquet(wr.OUT / f"d056_part4_twin_fold{fold}.parquet", index=False)
    (wr.OUT / f"d056_part4_twin_fold{fold}.json").write_text(json.dumps(choices, default=str))


def part4() -> None:
    """All 5 folds in parallel (Part 4 runtime about 7 min each)."""
    from src.models.fusion_lgbm import folds

    fl = folds("cgmacros", [load_config("mech")["cv_repeat"]])
    Parallel(n_jobs=5)(delayed(part4_fold)(f, tr, te) for _, f, tr, te in fl)
    print("D-056 part4 re-run done", flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "part4":
    part4()


def summary(wc: dict) -> dict:
    """D-056 metrics block (walk tests dev / final; Part 4 forecast comparison). Post-hoc."""
    from src.features.fusion_data import load_table
    from src.uncertainty.common import boot_ratio

    bc, out = wc["bootstrap"], {"label": LABEL}
    for st in ("dev", "final"):
        f = wr.OUT / f"d056_{st}_walk.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        blk = {}
        for m, g in d.groupby("model"):
            a = g[g.test == "activity_direction"].assign(p=lambda x: x["pass"].astype(float))
            per = a.groupby("patient_id").p.agg(["sum", "count"])
            r = boot_ratio(per["sum"], per["count"], bc["n_boot"], wc["seed"], bc["ci"])
            auc = g[g.test == "walk_effect_auc"].value
            blk[m] = r | {
                "n_checks": int(len(a)),
                "max_peak_rise": float(a.value.max()),
                "walk_auc_median": float(auc.median()),
                "walk_auc_max": float(auc.max()),
            }
        out[st] = blk
    if not (wr.OUT / "d056_part4_twin_fold0.parquet").exists():
        return out
    fc = load_config("fusion")
    tab = load_table("cgmacros", fc["n_hours"], fc)[["period_id", "t", "y_30", "y_60", "y_120"]]
    k, parts, alph = ["patient_id", "period_id", "t", "model"], [], {}
    for f in range(5):
        a = pd.read_parquet(wr.PROC / "mech" / f"twin_cgmacros_fold{f}.parquet")
        b = pd.read_parquet(wr.OUT / f"d056_part4_twin_fold{f}.parquet")
        parts.append(a.merge(b, on=k, suffixes=("", "_n"), validate="one_to_one"))
        j0 = json.loads((wr.PROC / "mech" / f"twin_cgmacros_fold{f}.json").read_text())
        j1 = json.loads((wr.OUT / f"d056_part4_twin_fold{f}.json").read_text())
        alph[f] = {"original": j0["alpha_on"], "d056": j1["alpha_on"]}
    m = pd.concat(parts).merge(tab, on=["period_id", "t"])
    p4: dict = {"alpha_on": alph, "variants": {}}
    for mod, g in m.groupby("model"):
        v = {}
        for h in (30, 60, 120):
            ok = g[f"y_{h}"].notna()
            dif = (g[f"mu_{h}_n"] - g[f"mu_{h}"]).abs()
            v[str(h)] = {
                "max_abs_diff": float(dif.max()),
                "median_abs_diff": float(dif.median()),
                "mae_original": float((g.loc[ok, f"mu_{h}"] - g.loc[ok, f"y_{h}"]).abs().mean()),
                "mae_d056": float((g.loc[ok, f"mu_{h}_n"] - g.loc[ok, f"y_{h}"]).abs().mean()),
            }
        p4["variants"][mod] = v
    out["part4_rerun"] = p4
    return out
