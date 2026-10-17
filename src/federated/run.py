"""Part 8c: federated learning simulation (D-063).

Run: ``python -W ignore -u -m src.federated.run`` (a few minutes; needs ``make features``).
Writes reports/metrics/federated.json and data/processed/federated/<layout>.parquet.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.federated import core as fc
from src.transfer import core as tc
from src.transfer.run import load_tables
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.provenance import file_hash, write_metrics

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROC / "federated"


def arrays(
    tab: pd.DataFrame, cols: list[str], horizons: list[int], scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features, scaled delta targets and the mask of usable targets for one table."""
    x = tab[cols].to_numpy(float)
    now = tab["cgm_now"].to_numpy(float)
    y = np.stack([(tab[f"y_{h}"].to_numpy(float) - now) / scale for h in horizons], axis=1)
    m = np.isfinite(y) & np.isfinite(now)[:, None]
    return x, np.where(m, y, 0.0), m.astype(float)


def predict(
    w: np.ndarray, shapes: list[tuple[int, ...]], x_std: np.ndarray, now: np.ndarray,
    scale: float, clip: tuple[float, float],
) -> np.ndarray:  # fmt: skip
    """Absolute glucose forecasts (rows x horizons) from a trained vector."""
    out, _ = fc.forward(fc.unflatten(w, shapes), x_std)
    return np.clip(now[:, None] + out * scale, *clip)


def one_split(
    layout: dict[str, Any], split: int, pool: pd.DataFrame, rows: pd.DataFrame, ev: pd.DataFrame,
    cols: list[str], cfg: dict[str, Any],
) -> pd.DataFrame:  # fmt: skip
    """All methods for one clinic layout and one random hold-out split; returns test predictions."""
    hz, scale = cfg["horizons_min"], cfg["model"]["target_scale_mgdl"]
    clip = tuple(load_config("features")["glucose_clip_mgdl"])
    rng = np.random.default_rng(cfg["seed"] + split)
    clinics = fc.partition(pool, layout["scheme"], layout["clinics"], rng)
    tr, te = fc.holdout(clinics, cfg["holdout_fraction"], rng)
    x_all, y_all, m_all = arrays(rows, cols, hz, scale)
    pid = rows["patient_id"].to_numpy()
    raw = {
        n: (x_all[np.isin(pid, p)], y_all[np.isin(pid, p)], m_all[np.isin(pid, p)])
        for n, p in tr.items()
    }
    names = list(tr)
    mean, sd = fc.federated_stats([raw[n][0] for n in names])
    std = {n: (fc.standardize(raw[n][0], mean, sd), raw[n][1], raw[n][2]) for n in names}
    clients = [std[n] for n in names]
    test_ids = {p: n for n, ps in te.items() for p in ps}
    te_ev = ev[ev["patient_id"].isin(test_ids)].copy()
    te_ev["clinic"] = te_ev["patient_id"].map(test_ids)
    now = te_ev["cgm_now"].to_numpy(float)
    x_te = te_ev[cols].to_numpy(float)
    x_te_std = fc.standardize(x_te, mean, sd)
    w_rng = np.random.default_rng(cfg["seed"] + 1000 + split)
    p0 = fc.init_params(len(cols), cfg["model"]["hidden"], len(hz), w_rng)
    shapes, w0 = fc.shapes_of(p0), fc.flatten(p0)
    tcfg = cfg["train"]

    def put(name: str, pred: np.ndarray, mask: np.ndarray | None = None) -> None:
        for j, h in enumerate(hz):
            col = f"pred_{name}_{h}"
            if col not in te_ev:
                te_ev[col] = np.nan
            if mask is None:
                te_ev[col] = pred[:, j]
            else:
                te_ev.loc[mask, col] = pred[:, j]

    for h in hz:
        te_ev[f"pred_persistence_{h}"] = te_ev["cgm_now"]
    # local-only: each clinic alone, its own standardization, tested on its own held-out patients
    for n in names:
        mk = (te_ev["clinic"] == n).to_numpy()
        if not mk.any():
            continue
        xl, yl, ml = raw[n]
        mn, s = fc.federated_stats([xl])
        wl = fc.local_train(w0, shapes, (fc.standardize(xl, mn, s), yl, ml),
                            tcfg["rounds"] * tcfg["local_epochs"], tcfg["batch"], tcfg["lr"],
                            np.random.default_rng(cfg["seed"] + 2000 + split))  # fmt: skip
        put(
            "local_only",
            predict(wl, shapes, fc.standardize(x_te[mk], mn, s), now[mk], scale, clip),
            mk,
        )
    wc = fc.central_train(
        clients, shapes, w0, tcfg, np.random.default_rng(cfg["seed"] + 3000 + split)
    )
    put("central", predict(wc, shapes, x_te_std, now, scale, clip))
    wf = fc.federated_train(
        clients, shapes, w0, tcfg, np.random.default_rng(cfg["seed"] + 4000 + split)
    )
    put("fedavg", predict(wf, shapes, x_te_std, now, scale, clip))
    wp = fc.federated_train(clients, shapes, w0, tcfg, np.random.default_rng(cfg["seed"] + 5000 + split),
                            mu=cfg["fedprox"]["mu"])  # fmt: skip
    put("fedprox", predict(wp, shapes, x_te_std, now, scale, clip))
    for z in cfg["dp"]["noise_multipliers"]:
        wd = fc.federated_train(clients, shapes, w0, tcfg, np.random.default_rng(cfg["seed"] + 6000 + split),
                                dp={"clip": cfg["dp"]["clip"], "z": z})  # fmt: skip
        put(f"fedavg_dp_z{z:g}", predict(wd, shapes, x_te_std, now, scale, clip))
    bad = [c for c in te_ev if c.startswith("pred_") and te_ev[c].isna().any()]
    if bad:  # MAE would silently drop these rows and score a subset of patients
        raise RuntimeError(f"non-finite predictions, layout {layout}, split {split}: {bad}")
    te_ev["split"] = split
    te_ev["patient_id"] = te_ev["patient_id"].astype(str) + f"|s{split}"
    return te_ev


def main() -> None:
    """Run all layouts and splits, write reports/metrics/federated.json."""
    cfg, tcfg = load_config("federated"), load_config("transfer")
    bc = load_config("baselines")
    static = pd.read_parquet(PROC / "static.parquet")
    tabs = load_tables(tcfg, static)
    a, b = (list(t.columns) for t in tabs.values())
    sets, _ = tc.feature_sets(a, b, tcfg)
    cols = sets[cfg["feature_set"]]
    for c, t in tabs.items():
        t["cohort"] = c
    rows = pd.concat(tabs.values(), ignore_index=True)
    ev = pd.concat(
        [tc.eval_table(t, cfg["test_from_day"], cfg["min_test_anchors"]) for t in tabs.values()],
        ignore_index=True,
    )
    ev = ev[ev["days_since_start"] >= cfg["test_from_day"]].reset_index(drop=True)  # D-061 region
    pool = (
        rows.groupby("patient_id")
        .agg(cohort=("cohort", "first"), hba1c=("static_hba1c_pct", "first"))
        .reset_index()
    )
    print(f"{len(cols)} features, {len(pool)} patients, {len(ev)} evaluation rows", flush=True)
    hz = cfg["horizons_min"]
    dps = [f"fedavg_dp_z{z:g}" for z in cfg["dp"]["noise_multipliers"]]
    models = ["persistence", "local_only", "central", "fedavg", "fedprox", *dps]
    pairs = [("fedavg", "central"), ("fedavg", "local_only"), ("fedprox", "fedavg"),
             ("fedavg", "persistence"), ("central", "persistence")] + [(d, "fedavg") for d in dps]  # fmt: skip
    metrics: dict[str, Any] = {
        "n_features": len(cols), "SIMULATION": "clinics are groups of open-cohort patients",
        "epsilon": {f"z{z:g}": fc.epsilon_gaussian(cfg["train"]["rounds"], z, cfg["dp"]["delta"])
                    for z in cfg["dp"]["noise_multipliers"]},
        "delta": cfg["dp"]["delta"], "clip": cfg["dp"]["clip"],
        "rounds": cfg["train"]["rounds"],
        "n_parameters": int(sum(np.prod(s) for s in fc.shapes_of(fc.init_params(len(cols), cfg["model"]["hidden"], len(hz), np.random.default_rng(0))))),
    }  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    for name, layout in cfg["layouts"].items():
        t0 = time.time()
        parts = [one_split(layout, i, pool, rows, ev, cols, cfg) for i in range(cfg["n_splits"])]
        res = pd.concat(parts, ignore_index=True)
        res.to_parquet(OUT / f"{name}.parquet", index=False)
        metrics[name] = {
            "clinics": layout["clinics"], "n_patient_splits": int(res["patient_id"].nunique()),
            "mae": tc.mae_rows(res, models, hz, bc["bootstrap"]),
            "diff": tc.diff_rows(res, pairs, hz, bc["bootstrap"]),
        }  # fmt: skip
        print(f"  layout {name}: {time.time() - t0:.0f}s", flush=True)
    files = [PROC / "static.parquet"] + [
        PROC / "features" / f"{n}_N{tcfg['n_hours']}.parquet" for n in tcfg["tables"].values()
    ]
    write_metrics("federated", metrics, cfg, cfg["seed"], file_hash(files))
    print("wrote reports/metrics/federated.json")


if __name__ == "__main__":
    main()
