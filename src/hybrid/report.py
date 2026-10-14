"""Part 5.1 evaluation: hybrid comparison on the Part 4 windows (day >= 3), paired patient CIs.

Models per k (identical rows within each k): persistence, per-patient AR (k >= 1), population
LightGBM (Part 4 cache), twin alone (twin_pop_actoff_k<k>), Hybrid A0 / A / A_noact / B, and
Hybrid C (k = 3; seed-averaged, plus the lambda = 0 ablation). Overall / post-meal / onset.
Neural onset fix: Part 3 nn_B_cgm / nn_E_full vs onset-weighted retrains (repeat 0, same rows).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.eval.fusion_report import paired_rows, regression_rows
from src.hybrid.build import OUT
from src.mech.report import build_wide, common_rows
from src.utils.config import load_config

HORIZONS = [30, 60, 120]
KEY = ["period_id", "t"]


def hybrid_wide() -> pd.DataFrame:
    """Part 4 wide table (+ hybrids per k as '<model>_k<k>|h', Hybrid C as 'C_k3|h')."""
    mc, fc = load_config("mech"), load_config("fusion")
    w = build_wide("cgmacros", mc, fc)
    for f in sorted(OUT.glob("hyb_cgmacros_fold*_k*.parquet")):
        d = pd.read_parquet(f)
        k = int(d["k"].iloc[0])
        cols = [c for c in d.columns if "|" in c and not c.startswith("twin|")]
        ren = {c: f"{c.split('|')[0]}_k{k}|{c.split('|')[1]}" for c in cols}
        part = d[KEY + cols].rename(columns=ren)
        w = w.merge(part, on=KEY, how="left", suffixes=("", "_dup"))
        for c in ren.values():  # folds fill disjoint rows
            if f"{c}_dup" in w:
                w[c] = w[c].fillna(w.pop(f"{c}_dup"))
    pf = OUT / "pinn_cgmacros_k3.parquet"
    if pf.exists():
        p = pd.read_parquet(pf)
        # folds where lambda = 0 was chosen: C IS the no-physics model, so C_lam0 = C there
        dup = p[(p.variant == "C") & (p.lam == 0.0)].assign(variant="C_lam0")
        p = pd.concat([p, dup], ignore_index=True)
        for var, g in p.groupby("variant"):
            avg = g.groupby(KEY, as_index=False)[[f"pred_{h}" for h in HORIZONS]].mean()
            name = "C_k3" if var == "C" else "C_lam0_k3"
            w = w.merge(
                avg.rename(columns={f"pred_{h}": f"{name}|{h}" for h in HORIZONS}),
                on=KEY,
                how="left",
            )
            for s, gs in g.groupby("seed"):
                w = w.merge(
                    gs[KEY + [f"pred_{h}" for h in HORIZONS]].rename(
                        columns={f"pred_{h}": f"{name}_s{s}|{h}" for h in HORIZONS}
                    ),
                    on=KEY,
                    how="left",
                )
    return w


def model_set(w: pd.DataFrame, k: int) -> list[str]:
    """Models compared at level k (present columns only)."""
    ms = ["persistence", "lgbm", f"twin_pop_actoff_k{k}"] + ([f"ar_k{k}"] if k >= 1 else [])
    ms += [f"{v}_k{k}" for v in ("A0", "A", "A_noact", "B")]
    if k == 3:
        ms += ["C_k3", "C_lam0_k3"]
    return [m for m in ms if f"{m}|60" in w]


def pairs_for(w: pd.DataFrame, k: int) -> list[tuple[str, str]]:
    """Paired comparisons at level k."""
    p = [
        (f"A_k{k}", f"A0_k{k}"),
        (f"A_noact_k{k}", f"A0_k{k}"),
        (f"A_k{k}", f"A_noact_k{k}"),
        (f"B_k{k}", f"A0_k{k}"),
        (f"B_k{k}", f"twin_pop_actoff_k{k}"),
        (f"A_k{k}", "lgbm"),
        (f"A0_k{k}", "lgbm"),
        (f"A_k{k}", f"twin_pop_actoff_k{k}"),
    ]
    if k >= 1:
        p.append((f"A_k{k}", f"ar_k{k}"))
    if k == 3:
        p += [("C_k3", "A0_k3"), ("C_k3", "C_lam0_k3"), ("C_k3", "A_k3")]
    return [x for x in p if f"{x[0]}|60" in w and f"{x[1]}|60" in w]


def evaluate_hybrid() -> dict[str, Any]:
    """Hybrid tables per k (all patients and without 007) + Hybrid C seed variance."""
    fc, bc, hc = load_config("fusion"), load_config("baselines"), load_config("hybrid")
    w = hybrid_wide()
    res: dict[str, Any] = {}
    for k in hc["comparison_k"]:
        ms = model_set(w, k)
        wk = common_rows(w, ms)
        for tag, ex in (("", None), ("_no007", hc["sensitivity_exclude"])):
            res[f"regression_k{k}{tag}"] = regression_rows(wk, ms, "cgmacros", fc, bc, ex)
            res[f"paired_k{k}{tag}"] = paired_rows(wk, pairs_for(w, k), "cgmacros", fc, bc, ex)
        res[f"n_patients_k{k}"] = int(wk.loc[wk[f"{ms[0]}|60"].notna(), "patient_id"].nunique())
        res[f"n_rows_k{k}"] = int(wk[f"{ms[0]}|60"].notna().sum())
    res["seed_variance"] = seed_rows(w)
    return res


def onset_wide(cohort: str) -> pd.DataFrame:
    """Part 3 wide table restricted to repeat 0 + onset-weighted retrains (seed-averaged)."""
    from src.eval.fusion_report import _add
    from src.eval.fusion_report import build_wide as fw

    w = fw(cohort, load_config("fusion"))
    w = w[w["repeat"] == 0].copy()
    for name in load_config("hybrid")["onset_fix"]["configs"]:
        f = OUT / f"onset_{cohort}_{name}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        avg = d.groupby(["period_id", "t", "repeat"], as_index=False)[
            [f"pred_{h}" for h in HORIZONS]
        ].mean()
        w = _add(w, avg, f"nnw_{name}", ["period_id", "t", "repeat"])
        for s, g in d.groupby("seed"):
            w = _add(w, g, f"nnw_{name}_s{s}", ["period_id", "t", "repeat"])
    return w


def evaluate_onset(cohort: str) -> dict[str, Any]:
    """Unweighted vs onset-weighted neural models (+ LightGBM E, per-patient AR) on same rows."""
    fc, bc, hc = load_config("fusion"), load_config("baselines"), load_config("hybrid")
    w = onset_wide(cohort)
    ms = ["nn_B_cgm", "nnw_B_cgm", "nn_E_full", "nnw_E_full", "lgbm_E_full_delta",
          "ar_per_patient", "persistence"]  # fmt: skip
    ms = [m for m in ms if f"{m}|60" in w]
    wk = common_rows(w, ms)
    pairs = [("nnw_B_cgm", "nn_B_cgm"), ("nnw_E_full", "nn_E_full"),
             ("nnw_E_full", "lgbm_E_full_delta"), ("nn_E_full", "lgbm_E_full_delta"),
             ("nnw_B_cgm", "ar_per_patient")]  # fmt: skip
    pairs = [p for p in pairs if p[0] in ms and p[1] in ms]
    res: dict[str, Any] = {}
    ex = hc["sensitivity_exclude"] if cohort == "cgmacros" else None
    for tag, e in (("", None), ("_no007", ex)):
        if tag and e is None:
            continue
        res[f"regression{tag}"] = regression_rows(wk, ms, cohort, fc, bc, e)
        res[f"paired{tag}"] = paired_rows(wk, pairs, cohort, fc, bc, e)
    sel = OUT / f"onset_{cohort}_selection.json"
    if sel.exists():
        res["weight_selection"] = json.loads(sel.read_text())
    return res


def leakage_check(w: pd.DataFrame) -> dict[str, Any]:
    """Structural leakage checks recorded with the metrics.

    Training rows of every hybrid start at day >= k (twin parameters from days < k only), and
    test windows of every model are identical to the Part 4 windows (row counts per k).
    """
    out: dict[str, Any] = {"part4_rows_k": {}}
    for k in load_config("hybrid")["comparison_k"]:
        ms = model_set(w, k)
        out["part4_rows_k"][str(k)] = {m: int(w[f"{m}|60"].notna().sum()) for m in ms}
    out["train_min_day"] = {}
    for f in sorted(OUT.glob("hyb_cgmacros_fold*_k*.json")):
        j = json.loads(f.read_text())
        out["train_min_day"][f.stem] = {"k": j["k"], "min_day": j["train_min_day"]}
    out["train_rows_respect_k"] = all(v["min_day"] >= v["k"] for v in out["train_min_day"].values())
    return out


def main() -> None:
    """hybrid_cgmacros.json, hybrid_neural.json and figures."""
    from src.hybrid import figures
    from src.utils.provenance import file_hash, write_metrics

    hc = load_config("hybrid")
    res = evaluate_hybrid()
    res["checks"] = leakage_check(hybrid_wide())
    pj = OUT / "pinn_cgmacros_k3.json"
    if pj.exists():
        res["pinn_choices"] = json.loads(pj.read_text())
    res["fold_choices"] = [
        json.loads(f.read_text()) for f in sorted(OUT.glob("hyb_cgmacros_fold*_k*.json"))
    ]
    res["twinfeat_choices"] = [
        json.loads(f.read_text()) for f in sorted(OUT.glob("twinfeat_*.json"))
    ]
    write_metrics("hybrid_cgmacros", res, {"hybrid": hc}, hc["seed"],
                  file_hash(sorted(OUT.glob("hyb_cgmacros_*.parquet"))))  # fmt: skip
    figures.comparison(res, hc["comparison_k"], "hybrid_comparison")
    if res["seed_variance"]:
        figures.seed_variance(res["seed_variance"], "hybrid_seed_variance")
    on = {c: evaluate_onset(c) for c in ("cgmacros", "shanghai_t2dm")
          if (OUT / f"onset_{c}_E_full.parquet").exists()}  # fmt: skip
    if on:
        write_metrics("hybrid_neural", on, {"hybrid": hc}, hc["seed"],
                      file_hash(sorted(OUT.glob("onset_*.parquet"))))  # fmt: skip
        figures.onset_fix(on, "hybrid_onset_fix")
    print("hybrid report written", flush=True)


def seed_rows(w: pd.DataFrame) -> list[dict[str, Any]]:
    """Hybrid C seed variance (pooled MAE per seed on the k = 3 common rows)."""
    rows = []
    wk = common_rows(w, model_set(w, 3))
    for name in ("C_k3", "C_lam0_k3"):
        if f"{name}|60" not in wk:
            continue
        seeds = sorted({c.split("|")[0] for c in wk.columns if c.startswith(f"{name}_s")})
        for h in HORIZONS:
            ok = wk[f"{name}|{h}"].notna() & wk[f"y_{h}"].notna()
            maes = [
                float((wk.loc[ok, f"{s}|{h}"] - wk.loc[ok, f"y_{h}"]).abs().mean()) for s in seeds
            ]
            if maes:
                rows.append(
                    {
                        "model": name,
                        "horizon_min": h,
                        "seed_mae": maes,
                        "mean": float(pd.Series(maes).mean()),
                        "sd": float(pd.Series(maes).std(ddof=1)),
                        "range": [min(maes), max(maes)],
                    }
                )
    return rows


if __name__ == "__main__":
    main()
