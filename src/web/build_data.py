"""Data file for the standalone clinician console (web/). ALL PATIENTS ARE SYNTHETIC.

Run: ``python -W ignore -u -m src.web.build_data`` -> web/data.json (not committed).

Patients come from the D-067 synthetic Indian cohort. Each patient gets a "now" moment on the last
day, the previous 24 h of glucose, the true next 2 h (for the replay button), AGP / time-in-range
statistics over all days, simulated sleep / HRV / steps and the simulated EHR history.

FORECAST SOURCE. ``--forecast dev`` (default here) fits a small ridge model on the synthetic cohort
itself, cross-fitted by patient. It is a PLACEHOLDER so the page can be developed; the file marks it
``forecast_source: "dev"`` and the page shows a development banner. The submission build replaces it
with the real model (trained on the real cohorts, zero-shot on these patients; see D-069).
"""

from __future__ import annotations

import argparse
import json
from math import erf, sqrt
from typing import Any

import numpy as np
import pandas as pd

from src.india import cohort as ic
from src.india import export as ex
from src.utils.config import PROJECT_ROOT, load_config

OUT = PROJECT_ROOT / "web" / "data.json"
HZ = [30, 60, 120]
STEP = 5  # minutes per grid step


def cdf(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def feats(g: np.ndarray, carbs: np.ndarray, act: np.ndarray, i: int) -> np.ndarray:
    """Causal features at 5-min index i (needs i >= 12)."""
    now = g[i]
    minute = (i * STEP) % 1440
    last = np.nonzero(carbs[: i + 1] > 0)[0]
    since = min((i - last[-1]) * STEP, 360) if len(last) else 360
    c3 = carbs[max(i - 36, 0) : i + 1].sum()
    return np.array([
        now, now - g[i - 3], now - g[i - 6], now - g[i - 12],
        np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440),
        c3, since, act[max(i - 12, 0) : i + 1].mean(),
    ])  # fmt: skip


def ridge(x: np.ndarray, y: np.ndarray, lam: float = 5.0) -> np.ndarray:
    """Ridge regression weights with standardised inputs and an intercept."""
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = np.c_[(x - mu) / sd, np.ones(len(x))]
    w = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ y)
    return np.r_[mu, sd, w]


def apply(m: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply ridge weights from ``ridge`` to new rows."""
    k = (len(m) - 1) // 3
    mu, sd, w = m[:k], m[k : 2 * k], m[2 * k :]
    return np.c_[(x - mu) / sd, np.ones(len(x))] @ w


def agp(g: np.ndarray) -> dict[str, list[float]]:
    """AGP percentiles over all days, 15-minute bins of the day."""
    d = g[: (len(g) // 288) * 288].reshape(-1, 288)
    d = d.reshape(d.shape[0], 96, 3).mean(2)
    q = np.percentile(d, [5, 25, 50, 75, 95], axis=0)
    return {
        k: np.round(v, 1).tolist()
        for k, v in zip(("p5", "p25", "p50", "p75", "p95"), q, strict=True)
    }


def tir(g: np.ndarray) -> dict[str, float]:
    """Share of readings in each glucose zone, in percent."""
    n = len(g)
    return {
        "lt54": round(100 * (g < 54).sum() / n, 1),
        "b54_70": round(100 * ((g >= 54) & (g < 70)).sum() / n, 1),
        "b70_180": round(100 * ((g >= 70) & (g <= 180)).sum() / n, 1),
        "b180_250": round(100 * ((g > 180) & (g <= 250)).sum() / n, 1),
        "gt250": round(100 * (g > 250).sum() / n, 1),
    }


def simulate() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """The D-067 SIMULATED cohort (grid, meals, sleep), its static + EHR table and export config."""
    icfg, mc = load_config("india"), load_config("mech")
    e = icfg["export"]
    c = ic.generate(icfg, mc, icfg["strengths"][e["strength"]], "exp", e["seed"],
                    int(e["n_patients"] * 1.25), e["days"], sleep_mult=e["sleep_mult"])  # fmt: skip
    c = ex.plausible(c, e["n_patients"], icfg)
    return c, ex.with_history(c["static"], e["seed"], icfg["ehr"]), e


def web_id(pid: str) -> str:
    """Console id of a simulated patient ('S-007')."""
    return "S-" + pid.rsplit("_", 1)[1]


def build(n_show: int, seed: int) -> dict[str, Any]:
    """Build the console payload for ``n_show`` simulated T2D patients."""
    c, static, e = simulate()
    pids = list(static["patient_id"])
    rng = np.random.default_rng(seed)
    per: dict[str, dict[str, Any]] = {}
    for pid, gdf in c["grid"].groupby("patient_id", sort=False):
        gdf = gdf.reset_index(drop=True)
        n = len(gdf)
        carbs = np.zeros(n)
        mt = c["meals"][c["meals"].patient_id == pid]
        t0 = gdf["t"].iloc[0]
        idx = ((mt["t"] - t0).dt.total_seconds() / 60 / STEP).round().astype(int).to_numpy()
        np.add.at(
            carbs, idx[(idx >= 0) & (idx < n)], mt["carbs_g"].to_numpy()[(idx >= 0) & (idx < n)]
        )
        per[pid] = {"g": gdf["glucose"].to_numpy(float), "carbs": carbs,
                    "act": gdf["activity_intensity"].to_numpy(float), "hr": gdf["hr"].to_numpy(float),
                    "steps": gdf["steps"].to_numpy(float), "t0": t0, "meals": mt}  # fmt: skip
    # ---- cross-fitted placeholder forecasts (patient folds)
    folds = np.array_split(rng.permutation(len(pids)), 5)
    models: dict[int, dict[int, np.ndarray]] = {}
    oof_res: dict[int, list[float]] = {h: [] for h in HZ}
    for k, test in enumerate(folds):
        tr = [pids[i] for f in range(5) if f != k for i in folds[f]]
        xs, ys = [], {h: [] for h in HZ}
        for pid in tr:
            p = per[pid]
            for i in range(3 * 288, len(p["g"]) - 24, 6):
                xs.append(feats(p["g"], p["carbs"], p["act"], i))
                for h in HZ:
                    ys[h].append(p["g"][i + h // STEP] - p["g"][i])
        x = np.array(xs)
        models[k] = {h: ridge(x, np.array(ys[h])) for h in HZ}
        for i in test:  # residuals on held-out patients
            p = per[pids[i]]
            ii = np.arange(3 * 288, len(p["g"]) - 24, 12)
            xx = np.array([feats(p["g"], p["carbs"], p["act"], j) for j in ii])
            for h in HZ:
                pred = p["g"][ii] + apply(models[k][h], xx)
                oof_res[h].extend((p["g"][ii + h // STEP] - pred).tolist())
    res_q = {h: np.percentile(oof_res[h], [10, 90]) for h in HZ}
    res_sd = {h: float(np.std(oof_res[h])) for h in HZ}
    fold_of = {pids[i]: k for k, f in enumerate(folds) for i in f}
    # ---- patients shown
    t2d = set(static.loc[static["diabetes_status"] == "t2d", "patient_id"])
    show = [pids[i] for i in rng.permutation(len(pids)) if pids[i] in t2d][:n_show]
    out_p = []
    for pid in sorted(show):
        p, s = per[pid], static[static.patient_id == pid].iloc[0]
        g, n = p["g"], len(p["g"])
        day = n // 288 - 1
        i = day * 288 + int(rng.integers(7 * 12, 22 * 12))  # 07:00 to 22:00 on the last day
        x = feats(g, p["carbs"], p["act"], i)[None, :]
        fc, ev = {}, {}
        for h in HZ:
            mean = float(g[i] + apply(models[fold_of[pid]][h], x)[0])
            fc[str(h)] = {"p": round(mean, 1), "lo": round(mean + res_q[h][0], 1),
                          "hi": round(mean + res_q[h][1], 1)}  # fmt: skip
        sd = lambda h: res_sd[h]  # noqa: E731
        ev["hi180_60"] = round(1 - cdf((180 - fc["60"]["p"]) / sd(60)), 3)
        ev["hi180_120"] = round(1 - cdf((180 - fc["120"]["p"]) / sd(120)), 3)
        ev["hi250_120"] = round(1 - cdf((250 - fc["120"]["p"]) / sd(120)), 3)
        ev["lo70_60"] = round(cdf((70 - fc["60"]["p"]) / sd(60)), 3)
        t_now = p["t0"] + pd.Timedelta(minutes=i * STEP)
        ml = p["meals"][
            (p["meals"]["t"] > t_now - pd.Timedelta(hours=26))
            & (p["meals"]["t"] <= t_now + pd.Timedelta(hours=2))
        ]
        nights = c["sleep"][c["sleep"].patient_id == pid].reset_index(drop=True)
        daily_steps = [int(p["steps"][d * 288 : (d + 1) * 288].sum()) for d in range(n // 288)]
        rest_hr = [
            round(float(np.percentile(p["hr"][d * 288 : (d + 1) * 288], 5)), 1)
            for d in range(n // 288)
        ]
        rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in s.to_dict().items()
               if k not in ("patient_id", "dataset")}  # fmt: skip
        out_p.append({
            "id": web_id(pid), "record": rec,
            "now": t_now.isoformat(), "hist": np.round(g[i - 287 : i + 1], 1).tolist(),
            "future": np.round(g[i + 1 : i + 25], 1).tolist(), "fc": fc, "ev": ev,
            "meals": [{"m": int((r.t - (t_now - pd.Timedelta(hours=24))).total_seconds() / 60),
                       "carbs": round(float(r.carbs_g)), "name": str(r.preset)} for r in ml.itertuples()],
            "agp": agp(g), "tir": tir(g), "mean": round(float(g.mean()), 1),
            "gmi": round(3.31 + 0.02392 * float(g.mean()), 1),
            "cv": round(100 * float(g.std() / g.mean()), 1), "days": n // 288,
            "sleep": [{"h": round(float(r.sleep_h), 2), "eff": round(float(r.sleep_eff), 3),
                       "awake": round(float(r.sleep_awake_min)), "light": round(float(r.sleep_light_min)),
                       "deep": round(float(r.sleep_deep_min)), "rem": round(float(r.sleep_rem_min)),
                       "hrv": round(float(r.hrv_rmssd), 1)} for r in nights.itertuples()],
            "steps_daily": daily_steps, "rest_hr": rest_hr,
        })  # fmt: skip
    return {
        "SYNTHETIC": True,
        "forecast_source": "dev",
        "horizons": HZ,
        "residual_sd": {str(h): round(res_sd[h], 1) for h in HZ},
        "cohort": {"n_total": len(pids), "days": e["days"], "seed": e["seed"]},
        "patients": out_p,
    }


def main() -> None:
    """Write web/data.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    d = build(a.n, a.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(d['patients'])} patients)")


if __name__ == "__main__":
    main()
