"""Live-twin payload for the web console (web/twin.json). ALL PATIENTS ARE SIMULATED.

The console runs the Part 4 mechanistic twin (src/mech/model.py: gut + minimal model + insulin
surrogate, RK4 at 5 min, extended Kalman filter) in the browser. This module prepares what the
browser needs and the reference numbers the browser must reproduce:

* per patient: 48 h of CGM before the console's "now" plus the 2 h after it, the meals in that
  window (grid index, carbohydrate, slowing ratio), and the twin's parameters fitted exactly as in
  Part 4 at k = 3 (MAP on the first 3 days, population prior from the CGMacros folds, activity
  off for forecasting, D-031 to D-037);
* the patient's TRUE basal glucose from the simulator, so the console can show how close the
  twin's only identifiable parameter comes to the truth (possible only for simulated patients);
* a walk effect for the what-if: the D-056 one-sided activity kernel with the population alpha
  that gives the D-052 assumed 20 % reduction of the 3-h iAUC (ASSUMED, labelled as such);
* parity references (tests/fixtures/twin_parity.json): Python EKF forecasts at many anchors,
  which ``tests/twin_parity.mjs`` compares with the browser code;
* the twin's own accuracy on the simulated cohort (every patient, every 30 min from day 3).

Run: ``python -W ignore -u -m src.web.twin_live`` (target ``make web-twin``).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.mech.fit import Prior, fit_patient
from src.mech.inputs import Record, meal_arrays
from src.mech.model import (
    FIT_PARAMS,
    IG,
    STATES,
    bounds_array,
    clip_state,
    ekf_forecast,
    initial_state,
    step_jacobian,
    theta_from_log,
)
from src.utils.config import PROJECT_ROOT, load_config
from src.web.build_data import simulate, web_id
from src.whatif.twin import calibrate_alpha

OUT = PROJECT_ROOT / "web" / "twin.json"
EXAMPLE = PROJECT_ROOT / "reports" / "metrics" / "twin_example_s151.json"
PARITY = PROJECT_ROOT / "tests" / "fixtures" / "twin_parity.json"
DATA = PROJECT_ROOT / "web" / "data.json"
STEP = 5
BEFORE = 576  # 48 h of history before "now" in the browser window
AFTER = 24  # the 2 h after "now" (replay)
HSTEPS = np.array([6, 12, 24])  # +30 / +60 / +120 min
K_DAYS = 3.0
Q_SCALE, R_OBS = 16.0, 25.0  # Part 4 choice for k = 3, activity off, in all 5 CGMacros folds


def part4_prior() -> Prior:
    """Population prior (activity off) averaged over the five CGMacros Part 4 folds."""
    m = json.loads((PROJECT_ROOT / "reports/metrics/mech_cgmacros.json").read_text("utf-8"))
    fc = m["metrics"]["fold_choices"]
    for f in fc:  # the browser constants above must be the stored choice in every fold
        assert tuple(f["qr_off_k3"].values())[:2] == (Q_SCALE, R_OBS), f["fold"]
    mu = np.mean([f["pop_prior_off"]["mu"] for f in fc], axis=0)
    sd = np.mean([f["pop_prior_off"]["sd"] for f in fc], axis=0)
    return Prior(np.asarray(mu), np.asarray(sd))


def record(pid: str, g: pd.DataFrame, meals: pd.DataFrame, w_protein: float) -> Record:
    """Part 4 input arrays for one simulated patient (activity off)."""
    t = g["t"].to_numpy().astype("datetime64[ns]")
    carbs, slow = meal_arrays(t, meals, w_protein, None)
    return Record(pid, pid, t, g["days_since_start"].to_numpy(float),
                  g["glucose"].to_numpy(float), carbs, slow, np.zeros(len(g)), "none")  # fmt: skip


def browser_arrays(n: int, meals: list[list[float]], slow0: float) -> tuple[np.ndarray, np.ndarray]:
    """Carbohydrate and slowing-ratio arrays rebuilt from the payload, as the browser does."""
    carbs, slow = np.zeros(n), np.zeros(n)
    at = {int(m[0]): m for m in meals}
    curr = slow0
    for i in range(n):
        if i in at:
            carbs[i], curr = at[i][1], at[i][2]
        slow[i] = curr
    return carbs, slow


def filter_state(th, y, carbs, slow, act, i_end, qd, r, p0, dt, nsub, bnd):  # noqa: ANN001, ANN201
    """EKF state (x, P) after the update at ``i_end``: the loop of model.ekf_forecast, stopped there."""
    n = len(y)
    first = 0
    while first < n and np.isnan(y[first]):
        first += 1
    x = initial_state(y[first] if first < n else th[0], th)
    pm = np.diag(p0)
    for i in range(first, i_end + 1):
        if i > first:
            x, f = step_jacobian(x, th, slow[i - 1], act[i - 1], dt, nsub, bnd)
            pm = f @ pm @ f.T + qd
        x[0] += carbs[i]
        if not np.isnan(y[i]):
            s_ = pm[IG, IG] + r
            kg = pm[:, IG] / s_
            x = x + kg * (y[i] - x[IG])
            pm = pm - np.outer(kg, kg) * s_
            pm = 0.5 * (pm + pm.T)
            clip_state(x, bnd)
    return x, pm


def scenario_ref(
    x, pm, th, slow_held, sc, walk, steps, qd, r, dt, nsub, bnd
):  # noqa: ANN001, ANN201
    """Python reference for the browser's TWIN.scenario (mean glucose per 5-min step, 0..steps)."""
    x = x.copy()
    pm = pm.copy()
    if sc["carbs"] > 0:
        x[0] += sc["carbs"]
    thx = th
    act = np.zeros(steps)
    if sc["walk"]:
        thx = np.append(th.copy(), 1.0)
        thx[6] = walk["alpha"]
        a0 = int(round(walk["start_after_meal_min"] / dt))
        act[a0 : a0 + int(round(walk["duration_min"] / dt))] = walk["intensity"]
    sl = sc["slow"] if sc["carbs"] > 0 else slow_held
    mean = [x[IG]]
    for j in range(1, steps + 1):
        x, f = step_jacobian(x, thx, sl, act[j - 1], dt, nsub, bnd)
        pm = f @ pm @ f.T + qd
        mean.append(x[IG])
    return np.array(mean)


def slow_of(m: dict) -> float:
    """Slowing ratio of a preset's macros (same weight 0.5 on protein as Part 4)."""
    return (m["fat_g"] + 0.5 * m["protein_g"] + m["fiber_g"]) / (m["carbs_g"] + 1.0)


def r1(x: np.ndarray, nd: int = 1) -> list[float]:
    """Rounded list (JSON size)."""
    return [round(float(v), nd) for v in x]


def build() -> dict[str, Any]:
    """Fit every console patient's twin and write the browser + parity payloads."""
    mc = load_config("mech")
    wc = load_config("whatif")
    sc, fixed = mc["solver"], mc["params"]["fixed"]
    dt, nsub, bnd = float(sc["dt_min"]), int(sc["n_sub"]), bounds_array(sc)
    e = mc["ekf"]
    qd_diag = np.array([Q_SCALE * e["q_base"][s] for s in STATES])
    p0 = np.array([e["p0"][s] for s in STATES])
    prior = part4_prior()
    ac = wc["twin"]["assumed"]
    alpha_walk = calibrate_alpha(prior.mu, mc, wc["walk"], ac, ac["reduction"], True)

    c, static, _ = simulate()
    web = json.loads(DATA.read_text("utf-8"))
    truth = c["params"].set_index("patient_id")
    by_web = {web_id(p): p for p in static["patient_id"]}
    w_p = mc["params"]["slow_protein_weight"]

    pats, parity, ev = [], [], {h: {"twin": [], "persist": []} for h in (30, 60, 120)}
    for wp in web["patients"]:
        pid = by_web[wp["id"]]
        g = c["grid"][c["grid"].patient_id == pid].reset_index(drop=True)
        ml = c["meals"][c["meals"].patient_id == pid]
        rec = record(pid, g, ml, w_p)
        i_now = int((pd.Timestamp(wp["now"]) - g["t"].iloc[0]).total_seconds() / 60 / STEP)
        assert abs(rec.y[i_now] - wp["hist"][-1]) < 0.051, wp["id"]

        fr = fit_patient(rec, K_DAYS, prior, mc, 0.0)
        th = theta_from_log(fr.logp, fixed, 0.0)
        gb_fit = float(np.exp(fr.logp[0]))

        # browser window
        a, b = i_now - BEFORE, i_now + AFTER + 1
        y = rec.y[a:b]
        idx = np.nonzero(rec.carbs[a:b])[0]
        meals = [[int(j), round(float(rec.carbs[a + j]), 1), round(float(rec.slow[a + j]), 4)]
                 for j in idx]  # fmt: skip
        slow0 = float(rec.slow[a])

        # parity reference on exactly the (rounded) inputs the browser receives
        n = b - a
        th_js = np.array([float(f"{v:.6g}") for v in th])
        y_js = np.round(y, 1)
        carbs_js, slow_js = browser_arrays(n, meals, round(slow0, 4))
        anchors = np.arange(BEFORE - 287, BEFORE + 1, 6)
        m, v = ekf_forecast(th_js, y_js, carbs_js, slow_js, np.zeros(n), anchors, HSTEPS,
                            np.diag(qd_diag), R_OBS, p0, dt, nsub, bnd)  # fmt: skip
        parity.append({"id": wp["id"], "anchors": anchors.tolist(),
                       "mean": m.tolist(), "var": v.tolist()})  # fmt: skip

        # twin accuracy on the whole simulated record, every 30 min from day 3
        full_a = np.arange(int(K_DAYS * 288), len(rec.y) - 24, 6)
        fm, _ = ekf_forecast(th, rec.y, rec.carbs, rec.slow, np.zeros(len(rec.y)), full_a,
                             HSTEPS, np.diag(qd_diag), R_OBS, p0, dt, nsub, bnd)  # fmt: skip
        for j, h in enumerate((30, 60, 120)):
            obs = rec.y[full_a + HSTEPS[j]]
            ev[h]["twin"].append(float(np.nanmean(np.abs(fm[:, j] - obs))))
            ev[h]["persist"].append(float(np.nanmean(np.abs(rec.y[full_a] - obs))))

        pats.append({
            "id": wp["id"],
            "y": r1(y),
            "meals": meals,
            "slow0": round(slow0, 4),
            "theta": [float(f"{x:.6g}") for x in th],
            "gb_fit": round(gb_fit, 1),
            "gb_true": round(float(truth.loc[pid, "Gb"]), 1),
            "fit_rmse": round(fr.rmse, 1),
        })  # fmt: skip
        print(wp["id"], "Gb fit", round(gb_fit, 1), "true", round(float(truth.loc[pid, "Gb"]), 1))

    gb_err = [abs(p["gb_fit"] - p["gb_true"]) for p in pats]
    gb_r = float(np.corrcoef([p["gb_fit"] for p in pats], [p["gb_true"] for p in pats])[0, 1])
    acc = {
        str(h): {
            "twin_mae": round(float(np.mean(ev[h]["twin"])), 2),
            "persistence_mae": round(float(np.mean(ev[h]["persist"])), 2),
        }
        for h in (30, 60, 120)
    }
    out = {
        "SYNTHETIC": True,
        "meta": {
            "dt_min": dt,
            "n_sub": nsub,
            "bounds": bnd.tolist(),
            "qd": qd_diag.tolist(),
            "r_obs": R_OBS,
            "p0": p0.tolist(),
            "states": list(STATES),
            "theta_names": list(FIT_PARAMS) + ["aE", "p2", "tauI", "cR", "tauE", "soft_s"],
            "before": BEFORE,
            "after": AFTER,
            "walk": {**wc["walk"], "alpha": round(alpha_walk, 6), "reduction": ac["reduction"]},
            "k_days": K_DAYS,
            "gb_identifiability": {
                "median_abs_err_mgdl": round(float(np.median(gb_err)), 1),
                "corr_fit_true": round(gb_r, 3),
                "n": len(pats),
            },
            "accuracy_simulated": acc,
            "real_patient_accuracy": {"30": 12.7, "60": 18.1, "120": 23.2,
                                      "source": "docs/RESULTS_SUMMARY.md Part 4 (CGMacros, k = 3)"},
        },  # fmt: skip
        "patients": pats,
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    PARITY.parent.mkdir(parents=True, exist_ok=True)
    # what-if scenarios: the browser's meal / walk path against a Python reference
    presets = web["whatif_meta"]["presets"]
    walk = out["meta"]["walk"]
    qd = np.diag(qd_diag)

    def run_scenarios(pt: dict, i: int, scs: list[dict]) -> list[dict]:
        n = len(pt["y"])
        carbs_js, slow_js = browser_arrays(n, pt["meals"], pt["slow0"])
        y_js = np.array([np.nan if v is None else v for v in pt["y"]], dtype=float)
        th_js = np.array(pt["theta"])
        x, pm = filter_state(th_js, y_js, carbs_js, slow_js, np.zeros(n), i, qd, R_OBS, p0,
                             dt, nsub, bnd)  # fmt: skip
        res = []
        for sc in scs:
            m = scenario_ref(x, pm, th_js, slow_js[i], sc, walk, 24, qd, R_OBS, dt, nsub, bnd)
            res.append({**sc, "mean": m.tolist(), "peak": float(m.max())})
        return res

    def sc_of(k: int | None, walk_on: bool) -> dict:
        if k is None:
            return {"label": "no change" + (" + walk" if walk_on else ""), "carbs": 0.0,
                    "slow": 0.5, "walk": walk_on}  # fmt: skip
        m = presets[k]["macros"]
        return {"label": presets[k]["label"] + (" + walk" if walk_on else ""),
                "carbs": float(round(m["carbs_g"])), "slow": slow_of(m), "walk": walk_on}  # fmt: skip

    cases = []
    for pr, pt in zip(parity[:12], pats[:12], strict=True):
        i = BEFORE - 7 * (len(cases) % 5)  # a few different moments
        scs = [sc_of(3, False), sc_of(6, True), sc_of(None, True), sc_of(len(cases) % 16, False)]
        cases.append({**pr, **{k: pt[k] for k in ("y", "meals", "slow0", "theta")},
                      "scenario_i": int(i), "scenarios": run_scenarios(pt, i, scs)})  # fmt: skip
    # gaps: the same patient with every 7th reading removed (sensor dropouts)
    g = dict(pats[0])
    g["y"] = [None if j % 7 == 3 else v for j, v in enumerate(g["y"])]
    y_g = np.array([np.nan if v is None else v for v in g["y"]], dtype=float)
    cg, sg = browser_arrays(len(y_g), g["meals"], g["slow0"])
    anchors = np.arange(BEFORE - 287, BEFORE + 1, 6)
    mg, vg = ekf_forecast(np.array(g["theta"]), y_g, cg, sg, np.zeros(len(y_g)), anchors, HSTEPS,
                          qd, R_OBS, p0, dt, nsub, bnd)  # fmt: skip
    cases.append({"id": g["id"] + "-gaps", "anchors": anchors.tolist(), "mean": mg.tolist(),
                  "var": vg.tolist(), **{k: g[k] for k in ("y", "meals", "slow0", "theta")},
                  "scenario_i": BEFORE, "scenarios": run_scenarios(g, BEFORE, [sc_of(5, True)])})  # fmt: skip

    # the README example: S-151, 20 minutes before "now"
    names = {q["label"]: k for k, q in enumerate(presets)}
    s151 = next(pt for pt in pats if pt["id"] == "S-151")
    ex_sc = [sc_of(names["Vegetarian thali"], False), sc_of(names["Vegetarian thali"], True),
             sc_of(names["Upma"], False), sc_of(names["Upma"], True)]  # fmt: skip
    ex = run_scenarios(s151, BEFORE - 4, ex_sc)
    cases.append({"id": "S-151", "scenario_i": BEFORE - 4, "scenarios": ex,
                  **{k: s151[k] for k in ("y", "meals", "slow0", "theta")}})  # fmt: skip
    EXAMPLE.write_text(json.dumps({
        "SYNTHETIC": True,
        "patient": "S-151", "minutes_before_now": 20,
        "glucose_now_mgdl": round(float(s151["y"][BEFORE - 4]), 1),
        "note": "Simulated patient. Twin rollout from the filtered state; walk effect ASSUMED (D-052).",
        "scenarios": [{"plan": r["label"], "carbs_g": r["carbs"],
                       "peak_mgdl": round(r["peak"], 1)} for r in ex],
    }, indent=1), encoding="utf-8")  # fmt: skip
    PARITY.write_text(json.dumps({"meta": out["meta"], "cases": cases}), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB); Gb median |err| "
          f"{out['meta']['gb_identifiability']}; accuracy {acc}")  # fmt: skip
    return out


if __name__ == "__main__":
    build()
