"""Part 8b: India stress test and synthetic-augmentation experiment (D-062).

EVERY number from this module that involves a synthetic cohort is a SIMULATED result. It says how
the model behaves if South Asian physiology differs in the assumed way. It does not measure
accuracy on Indian patients.

A. Stress test: LightGBM trained on all real CGMacros patients, applied to synthetic cohorts at
   strengths control / mild / moderate / strong (zero-shot, then with k days of the synthetic
   patient's own data). Reported against persistence and against the in-cohort CGMacros reference.
B. Augmentation: in CGMacros CV (repeat 0) compare training on real patients only with training on
   real + synthetic patients; the evaluation patients are always real.

Run: ``python -W ignore -u -m src.india.stress`` (about 15 to 30 minutes; needs ``make data``).
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import pandas as pd

from src.features.make_features import build_cohort, is_target
from src.features.make_splits import strata_for
from src.india import cohort as ic
from src.models.fusion_lgbm import folds, run_config
from src.transfer import core
from src.transfer.run import fit_model, load_tables, predict_into, strata_of
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.provenance import file_hash, write_metrics

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROC / "india"
SYN = "india"  # cohort key used for the synthetic tables (strata label, static spec)
WEAR_PREFIXES = ("hr_", "act_", "rest_proxy")
MEAL_PREFIX = "meal_"
TOD = ["tod_sin", "tod_cos"]


def feature_columns(real_cols: list[str], syn_cols: list[str], scfg: dict[str, Any]) -> list[str]:
    """Features present in BOTH tables for the configured groups (never targets or calendar)."""
    shared = [c for c in real_cols if c in set(syn_cols) and not is_target(c)]
    groups = set(scfg["feature_groups"])
    static = {f"static_{x}" for x in scfg["static"]}
    rules = [
        ("cgm", lambda c: c.startswith("cgm_")),
        ("tod", lambda c: c in TOD),
        ("wearable", lambda c: c.startswith(WEAR_PREFIXES)),
        ("meals", lambda c: c.startswith(MEAL_PREFIX)),
        ("static", lambda c: c in static),
    ]
    out = [c for c in shared if any(g in groups and ok(c) for g, ok in rules)]
    if "days_since_start" in out:
        out.remove("days_since_start")
    return out


def constant_columns(tab: pd.DataFrame, cols: list[str]) -> list[str]:
    """Columns with no variation in ``tab`` (a sign a channel carries no information)."""
    return [c for c in cols if tab[c].dropna().unique().size <= 1]


def synthetic_table(
    c: dict[str, pd.DataFrame], icfg: dict[str, Any], fcfg: dict[str, Any], scfg: dict[str, Any]
) -> pd.DataFrame:
    """Anchor table (N-hour windows) of one synthetic cohort on the 15-min grid."""
    f2 = copy.deepcopy(fcfg)
    f2["cohorts"][SYN] = {**fcfg["cohorts"]["cgmacros_15"], "dataset": icfg["name"]}
    f2["static"][SYN] = list(scfg["static"])
    grid = ic.to_grid(c, f2["cohorts"][SYN]["step_min"])
    tabs, _ = build_cohort(SYN, f2, grid, c["meals"], c["static"])
    return tabs[scfg["n_hours"]]


def bias_rows(df: pd.DataFrame, models: list[str], horizons: list[int]) -> list[dict[str, Any]]:
    """Mean signed error (prediction minus truth, mg/dL) per model and horizon."""
    rows = []
    for h in horizons:
        for m in models:
            d = df[[f"pred_{m}_{h}", f"y_{h}"]].dropna()
            rows.append({"model": m, "horizon_min": h, "bias": float((d.iloc[:, 0] - d.iloc[:, 1]).mean()),
                         "n_rows": int(len(d))})  # fmt: skip
    return rows


def stress_one(
    name: str, syn: pd.DataFrame, real: pd.DataFrame, cols: list[str], strata: dict[str, str],
    bc: dict[str, Any], scfg: dict[str, Any], fcfg_clip: tuple[float, float], seed: int,
    real_model: Any,
) -> dict[str, Any]:  # fmt: skip
    """Zero-shot and few-shot results of the real-trained model on one synthetic cohort."""
    hz = scfg["horizons_min"]
    ev = core.eval_table(syn, scfg["test_from_day"], scfg["min_test_anchors"])
    for h in hz:
        ev[f"pred_persistence_{h}"] = ev["cgm_now"]
        ev[f"pred_zeroshot_{h}"] = np.nan
    predict_into(ev, ev.index, real_model, "zeroshot", hz, fcfg_clip)
    fs_cfg = load_config("transfer")["fewshot"]
    for k in scfg["k_grid"]:
        for h in hz:
            ev[f"pred_zeroshot_offset{k:g}_{h}"] = np.clip(
                core.fewshot(ev, f"pred_zeroshot_{h}", h, k, "offset", fs_cfg), *fcfg_clip
            )
    models = ["persistence", "zeroshot"] + [f"zeroshot_offset{k:g}" for k in scfg["k_grid"]]
    kmax = max(scfg["k_grid"])
    pairs = [("zeroshot", "persistence"), (f"zeroshot_offset{kmax:g}", "persistence"),
             (f"zeroshot_offset{kmax:g}", "zeroshot")]  # fmt: skip
    boot = bc["bootstrap"]
    OUT.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(OUT / f"stress_{name}.parquet", index=False)
    return {"n_patients": int(ev.patient_id.nunique()), "mae": core.mae_rows(ev, models, hz, boot),
            "diff": core.diff_rows(ev, pairs, hz, boot), "bias": bias_rows(ev, models, hz)}  # fmt: skip


def augmentation(
    real: pd.DataFrame, extras: dict[str, pd.DataFrame], cols: list[str], strata: dict[str, str],
    bc: dict[str, Any], scfg: dict[str, Any], clip: tuple[float, float], seed: int,
) -> dict[str, Any]:  # fmt: skip
    """CGMacros CV repeat 0: real-only vs real + each synthetic cohort in the training set."""
    hz = scfg["horizons_min"]
    ev = core.eval_table(real, scfg["test_from_day"], scfg["min_test_anchors"])
    names = ["real_only"] + [f"real_plus_{k}" for k in extras]
    for n in names:
        for h in hz:
            ev[f"pred_{n}_{h}"] = np.nan
    for h in hz:
        ev[f"pred_persistence_{h}"] = ev["cgm_now"]
    for rep, _fold, tr_p, va_p in folds("cgmacros", [scfg["cv_repeat"]]):
        tr = sorted(set(tr_p) & set(real.patient_id))
        rows = ev.index[ev.patient_id.isin(set(va_p))]
        if not len(rows):
            continue
        for n, extra in [("real_only", None)] + [(f"real_plus_{k}", v) for k, v in extras.items()]:
            tab = (
                real[real.patient_id.isin(tr)]
                if extra is None
                else pd.concat([real[real.patient_id.isin(tr)], extra], ignore_index=True)
            )
            pids = tr if extra is None else tr + sorted(extra.patient_id.unique())
            m = fit_model(tab, pids, strata, cols, bc, hz, seed + rep)
            predict_into(ev, rows, m, n, hz, clip)
        print(f"    fold {_fold} done", flush=True)
    for k in scfg["k_grid"]:
        for n in names:
            for h in hz:
                ev[f"pred_{n}_offset{k:g}_{h}"] = np.clip(
                    core.fewshot(
                        ev, f"pred_{n}_{h}", h, k, "offset", load_config("transfer")["fewshot"]
                    ),
                    *clip,
                )
    models = ["persistence"] + names + [f"{n}_offset{k:g}" for k in scfg["k_grid"] for n in names]
    pairs = [(n, "real_only") for n in names[1:]]
    pairs += [
        (f"{n}_offset{k:g}", f"real_only_offset{k:g}") for k in scfg["k_grid"] for n in names[1:]
    ]
    boot = bc["bootstrap"]
    region = ev[ev["days_since_start"] >= scfg["test_from_day"]]
    ev.to_parquet(OUT / "augmentation_cgmacros.parquet", index=False)
    return {"n_patients": int(region.patient_id.nunique()), "mae": core.mae_rows(region, models, hz, boot),
            "diff": core.diff_rows(region, pairs, hz, boot)}  # fmt: skip


def main() -> None:
    """Generate the four synthetic cohorts, run A and B, write reports/metrics/india_stress.json."""
    icfg, mc = load_config("india"), load_config("mech")
    fcfg, bc, sc = load_config("features"), load_config("baselines"), load_config("splits")
    scfg, tcfg = icfg["stress"], load_config("transfer")
    static = pd.read_parquet(PROC / "static.parquet")
    real = load_tables(
        {**tcfg, "tables": {"cgmacros": tcfg["tables"]["cgmacros"]}, "n_hours": scfg["n_hours"]},
        static,
    )["cgmacros"]
    clip = tuple(load_config("data")["cgm"]["glucose_range_mgdl"]["cgmacros"])
    fits = ic.load_fits(icfg)
    noise, alpha = ic.load_noise_and_alpha(icfg)
    tabs, summ = {}, {}
    for tag, strength in icfg["strengths"].items():
        t0 = time.time()
        c = ic.generate(icfg, mc, strength, tag, scfg["seeds"][tag], scfg["n_patients"], scfg["days"],
                        fits, noise, alpha)  # fmt: skip
        tabs[tag] = synthetic_table(c, icfg, fcfg, scfg)
        tabs[tag] = tabs[tag].assign(
            _status=tabs[tag].patient_id.map(c["static"].set_index("patient_id")["diabetes_status"])
        )
        summ[tag] = ic.summary(c, icfg)
        print(f"  cohort {tag}: {len(tabs[tag])} anchors ({time.time() - t0:.0f}s)", flush=True)
    cols = feature_columns(list(real.columns), list(tabs["control"].columns), scfg)
    dead = constant_columns(real, cols)
    cols = [c for c in cols if c not in dead]
    print(f"{len(cols)} shared features (dropped constant in real: {len(dead)})", flush=True)
    strata = {p: f"cgmacros|{v}" for p, v in strata_for(static, "cgmacros", sc).items()}
    for t in tabs.values():
        strata |= {p: f"{SYN}|{s}" for p, s in t.groupby("patient_id")["_status"].first().items()}
    metrics: dict[str, Any] = {"n_features": len(cols), "dropped_constant_in_real": dead,
                               "cohort_summary": summ, "SYNTHETIC": True}  # fmt: skip
    real_p = sorted(real.patient_id.unique())
    print("A. stress test", flush=True)
    m_real = fit_model(real, real_p, strata, cols, bc, scfg["horizons_min"], icfg["seed"])
    metrics["stress"] = {tag: stress_one(tag, t, real, cols, strata, bc, scfg, clip, icfg["seed"], m_real)
                         for tag, t in tabs.items()}  # fmt: skip
    reference = strata_of(strata, "cgmacros", static)
    ref_ev = core.eval_table(real, scfg["test_from_day"], scfg["min_test_anchors"])
    for h in scfg["horizons_min"]:
        ref_ev[f"pred_persistence_{h}"] = ref_ev["cgm_now"]
    oof = run_config(real, cols, "cgmacros", [scfg["cv_repeat"]], reference, bc, scfg["horizons_min"],
                     "delta", "cgm_now", clip)  # fmt: skip
    oof = oof[core.KEYS + [f"pred_{h}" for h in scfg["horizons_min"]]]
    oof = oof.rename(columns={f"pred_{h}": f"pred_in_cohort_{h}" for h in scfg["horizons_min"]})
    ref_ev = ref_ev.merge(oof, on=core.KEYS, how="left")
    metrics["in_cohort_reference"] = core.mae_rows(
        ref_ev, ["persistence", "in_cohort"], scfg["horizons_min"], bc["bootstrap"]
    )
    print("B. augmentation", flush=True)
    extras = {k: tabs[v] for k, v in scfg["augment"].items()}
    metrics["augmentation"] = augmentation(real, extras, cols, strata, bc, scfg, clip, icfg["seed"])
    files = [
        PROC / "static.parquet",
        PROC / "features" / f"{tcfg['tables']['cgmacros']}_N{scfg['n_hours']}.parquet",
    ]
    write_metrics("india_stress", metrics, icfg, icfg["seed"], file_hash(files))
    print("wrote reports/metrics/india_stress.json")


if __name__ == "__main__":
    main()
