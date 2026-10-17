"""Part 8a runner (``make transfer``): cross-cohort transfer (D-061).

CGMacros (on the common 15-min grid) and Shanghai T2DM, both directions. For each direction and
feature set: persistence, in-cohort CV (target patients only), ZERO-SHOT transfer (source patients
only), zero-shot + few-shot correction with the target patient's first k days, and POOLED training
(all source patients + the target training patients of each CV fold). Every method is scored on the
same rows, with patient-bootstrap intervals. Usage: python -m src.transfer.run
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.features.build import static_features
from src.features.make_splits import strata_for
from src.models.experiment import inner_split
from src.models.fusion_lgbm import folds, run_config
from src.models.lgbm import LGBMForecaster
from src.transfer import core
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.provenance import file_hash, write_metrics

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROC / "transfer"


def load_tables(cfg: dict[str, Any], static: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Anchor tables (6 h windows) with the common static columns re-joined."""
    st = static_features(static, cfg["common_static"])
    out = {}
    for cohort, name in cfg["tables"].items():
        tab = pd.read_parquet(PROC / "features" / f"{name}_N{cfg['n_hours']}.parquet")
        tab = tab.drop(columns=[c for c in tab.columns if c.startswith("static_")])
        out[cohort] = tab.merge(st, left_on="patient_id", right_index=True, how="left")
    return out


def fit_model(
    tab: pd.DataFrame,
    train_pids: list[str],
    strata: dict[str, str],
    cols: list[str],
    bc: dict[str, Any],
    horizons: list[int],
    seed: int,
) -> LGBMForecaster:
    """LightGBM (delta target, Part 2 protocol): inner patient split for early stopping."""
    in_tr, in_va = inner_split(train_pids, strata, bc, seed)
    lcfg = {**bc["lightgbm"], "classifier_events": []}
    m = LGBMForecaster(cols, lcfg, seed=seed, target="delta", base="cgm_now")
    m.fit(
        tab[tab.patient_id.isin(in_tr)],
        val=tab[tab.patient_id.isin(in_va)],
        horizons=horizons,
        events=False,
    )
    return m


def predict_into(
    ev: pd.DataFrame, rows: pd.Index, m: LGBMForecaster, name: str, horizons: list[int],
    clip: tuple[float, float],
) -> None:  # fmt: skip
    """Write ``pred_<name>_<h>`` for the rows ``rows`` of ``ev``."""
    sub = ev.loc[rows]
    for h, p in m.predict(sub, horizons=horizons).items():
        ev.loc[rows, f"pred_{name}_{h}"] = np.clip(p, *clip)


def run_direction(
    src: str, tgt: str, fset: str, cols: list[str], tabs: dict[str, pd.DataFrame],
    strata: dict[str, str], static: pd.DataFrame, bc: dict[str, Any], cfg: dict[str, Any],
) -> dict[str, Any]:  # fmt: skip
    """All methods for one (source, target, feature set)."""
    hz, seed = cfg["horizons_min"], cfg["seed"]
    clip = tuple(load_config("data")["cgm"]["glucose_range_mgdl"][tgt])
    ts, tt = tabs[src], tabs[tgt]
    ev = core.eval_table(tt, cfg["test_from_day"], cfg["min_test_anchors"])
    src_p, tgt_p = sorted(ts.patient_id.unique()), sorted(ev.patient_id.unique())
    for h in hz:
        ev[f"pred_persistence_{h}"] = ev["cgm_now"]
    t0 = time.time()
    # in-cohort reference: repeated CV of the target cohort, same features, same split files
    oof = run_config(tt, cols, tgt, [cfg["cv_repeat"]], strata_of(strata, tgt, static), bc, hz,
                     "delta", "cgm_now", clip)  # fmt: skip
    oof = oof[core.KEYS + [f"pred_{h}" for h in hz]].rename(columns={f"pred_{h}": f"pred_in_cohort_{h}" for h in hz})  # noqa: E501
    ev = ev.merge(oof, on=core.KEYS, how="left")
    print(f"    in-cohort CV {time.time() - t0:.0f}s", flush=True)
    # zero-shot: source patients only
    m = fit_model(ts, src_p, strata, cols, bc, hz, seed)
    for h in hz:
        ev[f"pred_zeroshot_{h}"] = np.nan
    predict_into(ev, ev.index, m, "zeroshot", hz, clip)
    # few-shot corrections of the zero-shot forecast
    for meth in ("offset", "ridge"):
        for k in cfg["k_grid"]:
            for h in hz:
                ev[f"pred_zeroshot_{meth}{k:g}_{h}"] = np.clip(
                    core.fewshot(ev, f"pred_zeroshot_{h}", h, k, meth, cfg["fewshot"]), *clip
                )
    # pooled: all source patients + the target training patients of each CV fold
    for h in hz:
        ev[f"pred_pooled_{h}"] = np.nan
    tgt_all = set(tt.patient_id)
    for rep, _fold, tr_p, va_p in folds(tgt, [cfg["cv_repeat"]]):
        tr_t = sorted(set(tr_p) & tgt_all)
        both = pd.concat([ts, tt[tt.patient_id.isin(tr_t)]], ignore_index=True)
        mp = fit_model(both, src_p + tr_t, strata, cols, bc, hz, seed + rep)
        rows = ev.index[ev.patient_id.isin(set(va_p))]
        if len(rows):
            predict_into(ev, rows, mp, "pooled", hz, clip)
    print(f"    all methods {time.time() - t0:.0f}s", flush=True)
    models = ["persistence", "in_cohort", "zeroshot"]
    models += [f"zeroshot_{m}{k:g}" for m in ("offset", "ridge") for k in cfg["k_grid"]] + ["pooled"]
    pairs = [("zeroshot", "in_cohort"), ("pooled", "in_cohort"), ("in_cohort", "persistence"),
             ("zeroshot", "persistence"), ("pooled", "persistence")]  # fmt: skip
    kmax = max(cfg["k_grid"])
    pairs += [(f"zeroshot_{m}{kmax:g}", "zeroshot") for m in ("offset", "ridge")]
    boot = bc["bootstrap"]
    res: dict[str, Any] = {"n_source_patients": len(src_p), "n_target_patients": len(tgt_p)}
    region = ev[ev["days_since_start"] >= cfg["test_from_day"]]
    groups = {"all": region}
    sg = cfg["subgroups"].get(tgt)
    if sg:
        s = static.set_index("patient_id")[sg["column"]].astype(str)
        groups[sg["name"]] = region[region.patient_id.map(s) == sg["value"]]
    for gname, d in groups.items():
        if d.patient_id.nunique() < 2:
            continue
        res[gname] = {"mae": core.mae_rows(d, models, hz, boot), "diff": core.diff_rows(d, pairs, hz, boot)}
    OUT.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(OUT / f"{src}_to_{tgt}_{fset}.parquet", index=False)
    return res


def strata_of(strata: dict[str, str], cohort: str, static: pd.DataFrame) -> dict[str, str]:
    """Strata of one cohort's patients (the mixed dict has labels for both cohorts)."""
    pids = set(static.loc[static.dataset == cohort, "patient_id"])
    return {p: v for p, v in strata.items() if p in pids}


def main() -> None:
    """Run both directions and both feature sets; write reports/metrics/transfer.json."""
    cfg, bc, sc = load_config("transfer"), load_config("baselines"), load_config("splits")
    static = pd.read_parquet(PROC / "static.parquet")
    tabs = load_tables(cfg, static)
    # strata with the cohort name in the label so pooled inner splits never mix label meanings
    strata = {}
    for c in cfg["tables"]:
        strata |= {p: f"{c}|{v}" for p, v in strata_for(static, c, sc).items()}
    a, b = (list(t.columns) for t in tabs.values())
    sets, dropped = core.feature_sets(a, b, cfg)
    metrics: dict[str, Any] = {"dropped_cgm_columns": dropped,
                               "n_features": {k: len(v) for k, v in sets.items()}}  # fmt: skip
    for src, tgt in cfg["directions"]:
        for fset, cols in sets.items():
            print(f"{src} -> {tgt} [{fset}] ({len(cols)} features)", flush=True)
            metrics[f"{src}_to_{tgt}|{fset}"] = run_direction(src, tgt, fset, cols, tabs, strata, static, bc, cfg)
    files = [PROC / "static.parquet"] + [PROC / "features" / f"{n}_N{cfg['n_hours']}.parquet" for n in cfg["tables"].values()]
    write_metrics("transfer", metrics, cfg, cfg["seed"], file_hash(files))
    print("wrote reports/metrics/transfer.json")


if __name__ == "__main__":
    main()
