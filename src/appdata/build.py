"""Part 9 demo artifact builder (D-058): writes ``app/demo`` (the offline app bundle).

No test-patient outcome label (``y_*`` / ``*_any_*``) is ever scored, read for display, or stored:
every stored model output is an out-of-fold forecast of a Part 6 / Part 7 fold model that never saw
the demo patient, and calibration / alert thresholds are fitted on the fold's TRAINING patients
only. The one read of a test patient's rows projects to the ``patient_id`` column alone.

Written (all under ``app/demo``):
  models/           both forms' boosters + OOF residuals (export.export_models; read by appdata.rt)
  calib.npz/json    event calibration maps + alert thresholds (training OOF only)
  stats.json        stored Part 3/5/6/7 cohort numbers for the reference pages
  global_shap.json  mean |SHAP| per feature and horizon over the DISPLAYED windows
  patients.json     labels, status, profile, twin Gb, per-patient metadata
  manifest.json     build provenance, display config echo, guards
  figures/          the evidence figures
  patients/<pid>/   replay.parquet (anchors: forecasts, CV+ band, event probs, alerts, quality,
                    confidence, model inputs), explain.parquet (template sentences + reasons),
                    shap.npz, grid.parquet + meals.parquet (causal feature window, truncated at
                    the last replay anchor = the app's live what-if state), twin.npz (what-if
                    twin trajectories on the app's carb / slow-ratio grid) + whatif.json

Guards run on every build (recorded in manifest.json):
  rtform_max_abs_diff          shipped text models reproduce in-memory predictions and bands
  recompute_full_max_abs_diff  the truncated live window reproduces the stored feature rows
  interp_max_abs_diff          linear interpolation of the twin grid vs the engine
  labels_in_outputs            no outcome column ever appears in an output frame

Usage:  python -m src.appdata.build [--check]
  --check  guards and numbers only, nothing is written
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.appdata import events as ev
from src.appdata import export as ex
from src.appdata import replay as rp
from src.appdata import rt
from src.appdata import synthetic as sy
from src.appdata import whatif_grid as wg
from src.explain import shap_values as shap
from src.explain import templates as tp
from src.hybrid.twin_features import HORIZONS
from src.mech.model import FIT_PARAMS
from src.uncertainty.common import UNC, groups
from src.utils.config import PROJECT_ROOT, load_config
from src.whatif import context as cx
from src.whatif import edits as ed

APP = PROJECT_ROOT / "app" / "demo"
GEN = PROJECT_ROOT / "logs"
COHORT = "cgmacros"
ALPHA80 = 0.2
CLIP = (40.0, 405.0)
KEYS = ["patient_id", "period_id", "t"]
VAR_ORDER = ("twin_learned", "twin_assumed")  # rows of every precomputed twin grid
LABEL_MARKERS = ("_any_",)  # outcome-column name markers forbidden in every output frame
EVENTS_ALL = ("hyper180", "hyper140", "hypo70")  # event probabilities shown in the app


# --------------------------------------------------------------------------- io helpers


def sync_dir(dst: Path, src: Path) -> None:
    """Copy a directory tree, deleting files in dst that are not in src (deterministic)."""
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(dst.rglob("*")):
        if p.is_file():
            p.unlink()
    for p in sorted(src.rglob("*")):
        if p.is_file():
            (dst / p.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / p.relative_to(src))


def gzip_json(obj: dict[str, Any], path: Path) -> None:
    """Write a gzip-compressed JSON file."""
    buf = gzip.compress(json.dumps(clean_pandas(obj), ensure_ascii=False).encode("utf-8"), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf)


def clean_pandas(obj: Any) -> Any:
    """JSON-serialisable mirror of a small value (dict / list / numpy / pandas scalars)."""
    if isinstance(obj, dict):
        return {str(k): clean_pandas(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_pandas(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


# --------------------------------------------------------------------------- config


def load_cfgs() -> dict[str, dict[str, Any]]:
    """All configs the builder reads."""
    return {n: load_config(n) for n in ("features", "whatif", "uncertainty", "app", "explain")}


def is_label(col: str) -> bool:
    """Outcome columns: ``y_*`` / ``y|*`` and ``*_any_*`` (never shown or stored).

    ``cgm_slope_30`` is NOT an outcome: it is the trailing least-squares slope of the last 30 min
    of CGM (``src.features.build``), the input of the trend alert. Part 6 merely stores it with the
    labels because the out-of-fold frames do not carry it.
    """
    return col.startswith(("y_", "y|")) or any(m in col for m in LABEL_MARKERS)


def label_free(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` without any outcome column."""
    return df[[c for c in df.columns if not is_label(c)]].copy()


def assert_label_free(name: str, df: pd.DataFrame) -> int:
    """Raise if an outcome column is in an output frame; returns the number of columns checked."""
    bad = [c for c in df.columns if is_label(str(c))]
    if bad:
        raise RuntimeError(f"outcome columns in output {name}: {bad[:5]}")
    return len(df.columns)


def say(msg: str) -> None:
    """Progress line."""
    print(f"[app-data {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- calibration


def export_calibration(cal: rp.Calib, out: Path) -> None:
    """Platt maps, event thresholds, residuals, alert thresholds (fitted on training rows only)."""
    platt: dict[str, Any] = {}
    for (e, h), (_, pl) in cal.maps.items():
        platt[f"{e}|{h}"] = (
            None
            if pl is None
            else {"coef": float(pl.m.coef_[0, 0]), "intercept": float(pl.m.intercept_[0])}
        )
    meta = {
        "platt": platt,
        "events": {e: [thr, above] for e, (thr, above) in ev.EV.items()},
        "theta": {e: (None if v is None else float(v)) for e, v in cal.theta.items()},
        "fa_per_day": cal.rate,
        "alert_horizon_min": cal.ac["horizon_min"],
        "refractory_min": cal.ac["refractory_min"],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "calib.json").write_text(json.dumps(meta, indent=1), "utf-8")
    res = {f"res_{h}": cal.maps[("hyper180", h)][0] for h in HORIZONS}
    np.savez_compressed(out / "calib.npz", **res)


# --------------------------------------------------------------------------- one patient


def replay_tables(
    pid: str,
    label: str,
    rows_all: pd.DataFrame,
    g: pd.DataFrame,
    meals: pd.DataFrame,
    start_day: float,
    forms: dict[str, Any],
    cal: rp.Calib,
    cols: list[str],
    ref_width: float,
    cfgs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Replay, explanation and SHAP tables of one patient (all label-free)."""
    app, unc, xcfg = cfgs["app"], cfgs["uncertainty"], cfgs["explain"]
    f = forms["unc"]
    rows = rp.replay_rows(rows_all, app["demo"], start_day)
    fc = rp.forecast_block(f, rows, ALPHA80)
    now = rows["cgm_now"].to_numpy(float)
    probs = cal.probs(fc, now)
    q = rp.quality_at(g, rows["t"], unc["dq"], CLIP)
    al = rp.alert_block(
        cal, g, rows, fc, q["grid_pos"].to_numpy(int), list(app["alerts"]["events"])
    )
    width = (fc["hi80|60"] - fc["lo80|60"]).to_numpy(float)
    lab, ratio = ev.confidence(
        width, ref_width, q["dq"].to_numpy(float), q["stale_min"].to_numpy(float),
        app["display"]["confidence"],
    )  # fmt: skip
    conf = pd.DataFrame({"confidence": lab, "width_ratio": ratio}, index=rows.index)
    keep = rows[KEYS + ["days_since_start"]].reset_index(drop=True)
    inputs = rows[[c for c in cols if c in rows.columns]].reset_index(drop=True)
    replay = pd.concat(
        [keep, fc.reset_index(drop=True), probs.reset_index(drop=True),
         q.reset_index(drop=True), al.reset_index(drop=True), conf.reset_index(drop=True),
         inputs],
        axis=1,
    )  # fmt: skip
    replay = replay.loc[:, ~replay.columns.duplicated()]
    # SHAP of the population model's predicted change, and the template sentences
    phis: dict[int, pd.DataFrame] = {}
    bases: dict[int, np.ndarray] = {}
    used: dict[int, set[str]] = {}
    for h in HORIZONS:
        phis[h], bases[h] = shap.contributions(f.pop[h], rows)
        used[h] = set(shap.used_features(f.pop[h]))
    recs = []
    for i in range(len(rows)):
        row = rows.iloc[i].to_dict()
        for e in EVENTS_ALL:
            for h in HORIZONS:
                pr = probs[f"p_{e}|{h}"].iloc[i]
                x = tp.explain_event(
                    e, h, None if np.isnan(pr) else float(pr), row, phis[h].iloc[i].to_dict(),
                    used[h], xcfg,
                )  # fmt: skip
                recs.append(
                    {"t": rows["t"].iloc[i], "event": e, "horizon": h, "sentence": x["sentence"],
                     "risk": x["risk"], "reasons": json.dumps(x["reasons"])}
                )  # fmt: skip
    explain = pd.DataFrame(recs)
    return {
        "rows": rows, "replay": replay, "explain": explain, "phi": phis, "base": bases,
        "pos": q["grid_pos"].to_numpy(int),
    }  # fmt: skip


def write_patient(
    out: Path, pid: str, t: dict[str, Any], g: pd.DataFrame, meals: pd.DataFrame, cols: list[str]
) -> dict[str, int]:
    """Write one patient folder; returns the number of columns checked for outcome names."""
    d = out / "patients" / pid
    d.mkdir(parents=True, exist_ok=True)
    last = t["rows"]["t"].max()
    checked = {}
    for name, df in (
        ("replay", t["replay"]),
        ("explain", t["explain"]),
        ("grid", g[g["t"] <= last]),
        ("meals", meals[meals["t"] <= last]),
    ):
        checked[name] = assert_label_free(f"{pid}/{name}", df)
        ex.save_table(df, d / f"{name}.parquet")
    arrs: dict[str, np.ndarray] = {"cols": np.array(cols)}
    for h in HORIZONS:
        arrs[f"phi_{h}"] = t["phi"][h].to_numpy(float)
        arrs[f"base_{h}"] = np.asarray(t["base"][h], float)
    np.savez_compressed(d / "shap.npz", **arrs)
    return checked


# --------------------------------------------------------------------------- main


def main(check: bool = False) -> None:
    """Build ``app/demo`` (or only run the guards with ``--check``)."""
    t0 = time.time()
    cfgs = load_cfgs()
    app, wc, unc, fcfg = cfgs["app"], cfgs["whatif"], cfgs["uncertainty"], cfgs["features"]
    demo, disp = app["demo"], app["display"]
    fold = int(app["fold"])
    say("loading data and fitting the fold models (about the Part 7 development cost)")
    data = cx.load_data(wc)
    fcx = cx.fold_ctx(data, wc, fold, dev=False)  # fold models that never saw the test patients
    forms, cols = fcx.forms, data["cols"]
    train, test = data["folds"][fold]
    say(f"fold {fold}: {len(train)} training / {len(test)} test patients; oof guard "
        f"{fcx.extra['oof_guard_max_abs_diff']:.3g}")  # fmt: skip
    # ---- patients with Part 6 test rows (projection to the id column only)
    p6 = pd.read_parquet(UNC / f"unc_A_{COHORT}_fold{fold}_test.parquet", columns=["patient_id"])
    status = groups(COHORT)
    eligible = sorted(set(p6["patient_id"].astype(str)) & set(test))
    demo_ids = rp.pick_demo(eligible, status, demo)
    say(f"demo patients (rule in configs/app.yaml): {demo_ids}")
    # ---- label-free rows of every test patient's first period
    tab = data["tab"]
    fp = data["fp"]
    rows_lf = label_free(fp[fp["patient_id"].isin(test)]).sort_values(["patient_id", "t"])
    del fp
    # ---- calibration and alert thresholds from the fold's TRAINING patients only
    oof = pd.read_parquet(UNC / f"unc_A_{COHORT}_fold{fold}_oof.parquet")
    if not set(oof["patient_id"].astype(str)) <= set(train):
        raise RuntimeError("calibration rows include a non-training patient")
    slope = unc["alerts"][
        "trend_slope_feature"
    ]  # trailing CGM slope: a causal feature, not a label
    need = [f"{e}_any_{h}" for e in ev.EV for h in HORIZONS] + [slope]
    need = [c for c in need if c not in oof.columns]
    if need:  # training patients' rows only (the oof frame holds nothing else, checked above)
        lab = tab[tab["patient_id"].isin(train)][KEYS + need]
        oof = oof.merge(lab, on=KEYS, how="left", validate="one_to_one")
    cal = rp.Calib(oof, unc["alerts"], list(app["alerts"]["events"]),
                   int(app["alerts"]["default_fa_per_day"]))  # fmt: skip
    say(f"alert thresholds at {cal.rate} false alarms/day: {cal.theta}")
    # ---- reference width of the CV+ band at +60 (fold test rows, no labels)
    reg = rows_lf[rows_lf["days_since_start"] >= wc["band_region_from_day"]]
    lo, hi = forms["unc"].band(reg, 60, ALPHA80, None)
    ref_width = float(np.nanmedian(hi - lo))
    say(f"median CV+ 80% width at +60: {ref_width:.1f} mg/dL ({len(reg)} rows)")
    twin_ids = set(fcx.twins)
    guards: dict[str, float] = {}
    patients_meta: dict[str, Any] = {}
    tables: dict[str, dict[str, Any]] = {}
    twin_arrays: dict[str, dict[str, np.ndarray]] = {}
    windows: dict[str, list[dict[str, Any]]] = {}
    gbs: dict[str, float | None] = {}
    rng = np.random.default_rng(int(app["seed"]))
    for pid in demo_ids:
        say(f"patient {pid}")
        r_all = rows_lf[rows_lf["patient_id"] == pid].reset_index(drop=True)
        per_id = str(r_all["period_id"].iloc[0])
        per = cx.period(data, per_id)
        t = replay_tables(pid, pid, r_all, per.g, per.meals, demo["replay_start_day"],
                          forms, cal, cols, ref_width, cfgs)  # fmt: skip
        tables[pid] = t
        gbs[pid] = (
            float(np.exp(fcx.twins[pid].logp[FIT_PARAMS.index("Gb")])) if pid in twin_ids else None
        )
        # guard: the zero-edit recompute on the truncated live window reproduces stored rows
        last = t["rows"]["t"].max()
        g_tr, m_tr = per.g[per.g["t"] <= last], per.meals[per.meals["t"] <= last]
        a, b = (ed.recompute(t["rows"], g_tr, m_tr, cols, fcx.ectx),
                ed.recompute_full(t["rows"], g_tr, m_tr, cols, fcx.ectx))  # fmt: skip
        ecols = ed.edited_columns(cols)
        guards[f"recompute_full_max_abs_diff|{pid}"] = max(
            ed.max_diff(a, t["rows"], ecols), ed.max_diff(b, t["rows"], ecols)
        )
        # what-if windows and the twin grid (only if the patient has a fitted twin)
        if pid in twin_ids:
            per_lf = replace(per, rows=t["rows"])
            wins = wg.pick_windows(per_lf, t["rows"], demo)
            alphas = {
                "twin_learned": fcx.alpha_on,
                "twin_assumed": fcx.alpha_assumed[f"{wc['twin']['assumed']['reduction']:g}"],
            }
            arrs: dict[str, np.ndarray] = {}
            for k, w in enumerate(wins):
                arrs[f"w{k}"] = wg.twin_grid(fcx, per_lf, w, alphas, app["whatif"]["twin_grid"])
            if wins:
                guards[f"interp_max_abs_diff|{pid}"] = wg.interp_check(
                    fcx, per_lf, wins[0], alphas, app["whatif"]["twin_grid"], arrs["w0"], rng, 3
                )
            twin_arrays[pid] = arrs
            windows[pid] = [
                {"k": k, "t": str(w["t"]), "meal": int(w["meal"]), "offset_min": w["offset_min"],
                 "meal_row": {c: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.number)) else str(v)))
                              for c, v in per.meals.loc[w["meal"]].items()},
                 "slow_ratio_obs": wg.slow_ratio(per.meals.loc[w["meal"]].to_dict(),
                                                 data["mc"]["params"]["slow_protein_weight"])}
                for k, w in enumerate(wins)
            ]  # fmt: skip
        patients_meta[pid] = {
            "label": f"Patient {pid}", "synthetic": False, "status": status.get(pid, "unknown"),
            "hba1c_pct": float(r_all["static_hba1c_pct"].iloc[0]),
            "fasting_glucose_mgdl": float(r_all["static_fasting_glucose_mgdl"].iloc[0]),
            "twin_gb_mgdl": gbs[pid], "n_anchors": int(len(t["rows"])),
            "windows": windows.get(pid, []),
            "start": str(t["rows"]["t"].iloc[0]), "end": str(last),
        }  # fmt: skip
    # ---- the synthetic patient
    say("synthetic patient (Part 4 simulator, labelled synthetic)")
    syn = sy.build(demo, fcfg, CLIP, data["ectx"]["n_h"])
    miss = [c for c in cols if c not in syn["rows"].columns]
    for c in miss:
        syn["rows"][c] = np.nan
    ts = replay_tables(syn["patient_id"], "synthetic", syn["rows"], syn["grid"], syn["meals"],
                       demo["synthetic"]["replay_start_day"], forms, cal, cols, ref_width, cfgs)  # fmt: skip
    sid = f"syn_{syn['patient_id']}"
    tables[sid] = ts
    patients_meta[sid] = {
        "label": "Synthetic patient", "synthetic": True, "status": "synthetic",
        "hba1c_pct": demo["synthetic"]["static_hba1c_pct"],
        "fasting_glucose_mgdl": demo["synthetic"]["static_fasting_glucose_mgdl"],
        "twin_gb_mgdl": None, "gb_simulated_true_mgdl": syn["gb_true"],
        "n_anchors": int(len(ts["rows"])), "windows": [],
        "start": str(ts["rows"]["t"].iloc[0]), "end": str(ts["rows"]["t"].max()),
        "inputs_not_in_simulator": miss,
    }  # fmt: skip
    # ---- guard: shipped text models reproduce the in-memory models (population + CV+ band)
    say("exporting models and checking the shipped runtime")
    tmp = GEN / "app_models_check"
    shutil.rmtree(tmp, ignore_errors=True)
    meta = ex.export_models(forms, cols, tmp)
    rt_forms = rt.load_forms(tmp)
    probe = tables[demo_ids[0]]["rows"].iloc[:24]
    diff = 0.0
    for name, rf in rt_forms.items():
        for h in HORIZONS:
            diff = max(
                diff,
                float(
                    np.nanmax(np.abs(rf.predict(probe, h) - forms[name].predict(probe, h, None)))
                ),
            )
            lo_r, hi_r = rf.band(probe, h, ALPHA80)
            lo_m, hi_m = forms[name].band(probe, h, ALPHA80, None)
            diff = max(
                diff, float(np.nanmax(np.abs(lo_r - lo_m))), float(np.nanmax(np.abs(hi_r - hi_m)))
            )
    guards["rtform_max_abs_diff"] = diff
    n_cols = {k: assert_label_free(k, v["replay"]) + assert_label_free(k, v["explain"])
              for k, v in tables.items()}  # fmt: skip
    guards["labels_in_outputs"] = 0.0
    say(f"guards: {json.dumps({k: round(v, 12) for k, v in guards.items()})}")
    bad = {k: v for k, v in guards.items()
           if k != "labels_in_outputs" and not v <= (1e-6 if k.startswith(("rtform", "recompute")) else 5.0)}  # fmt: skip
    if bad:
        raise RuntimeError(
            f"guards failed (limits: runtime/recompute 1e-6, twin interpolation 5 mg/dL): {bad}"
        )
    if check:
        say(f"--check passed in {time.time() - t0:.0f} s; nothing written")
        return
    # ---- write
    say("writing app/demo")
    stage = GEN / "app_demo_stage"
    shutil.rmtree(stage, ignore_errors=True)
    ex.export_models(forms, cols, stage / "models")
    export_calibration(cal, stage)
    gzip_json(ex.cohort_stats(), stage / "stats.json.gz")
    figs = ex.copy_figures(stage / "figures")
    gs: dict[str, Any] = {}
    for h in HORIZONS:
        phi = pd.concat([v["phi"][h] for v in tables.values()], ignore_index=True)
        gs[str(h)] = shap.global_importance(phi, 15).to_dict("records")
    (stage / "global_shap.json").write_text(json.dumps(gs, indent=1), "utf-8")
    for pid, t in tables.items():
        real = pid in demo_ids
        if real:
            per = cx.period(data, str(t["rows"]["period_id"].iloc[0]))
            g, m = per.g, per.meals
        else:
            g, m = syn["grid"], syn["meals"]
        write_patient(stage, pid, t, g, m, cols)
        if pid in twin_arrays:
            np.savez_compressed(stage / "patients" / pid / "twin.npz", **twin_arrays[pid])
        (stage / "patients" / pid / "whatif.json").write_text(
            json.dumps(clean_pandas(patients_meta[pid]["windows"]), indent=1), "utf-8"
        )
    ctx_meta = {"ectx": clean_pandas(fcx.ectx), "kcal_per_g": wc["kcal_per_g"], "walk": wc["walk"],
                "twin_grid": app["whatif"]["twin_grid"], "variants": list(VAR_ORDER),
                "alphas": {"twin_learned": fcx.alpha_on}}  # fmt: skip
    (stage / "context.json").write_text(json.dumps(ctx_meta, indent=1), "utf-8")
    (stage / "patients.json").write_text(json.dumps(clean_pandas(patients_meta), indent=1), "utf-8")
    man = {
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_hash": _git_hash(), "fold": fold, "cohort": COHORT, "demo_patients": demo_ids,
        "synthetic_patient": sid, "n_train": len(train), "n_test": len(test),
        "ref_width60_mgdl": ref_width, "alert_theta": {k: v for k, v in cal.theta.items()},
        "figures": figs, "guards": guards, "columns_checked_for_outcomes": n_cols,
        "display": clean_pandas(disp), "build_seconds": round(time.time() - t0, 1),
        "models": {"horizons": meta["horizons"], "n_cols": len(meta["cols"])},
    }  # fmt: skip
    (stage / "manifest.json").write_text(json.dumps(man, indent=1), "utf-8")
    sync_dir(APP, stage)
    size = sum(p.stat().st_size for p in APP.rglob("*") if p.is_file()) / 1e6
    say(f"done in {time.time() - t0:.0f} s; app/demo is {size:.1f} MB (design target < 25 MB)")


def _git_hash() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    main(check="--check" in sys.argv)
