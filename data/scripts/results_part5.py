"""Part 5 section of docs/RESULTS.md (hybrid twin, neural onset fix, personalization).

Every number comes from reports/metrics/{hybrid_*,personalize_*}.json.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "reports" / "metrics"
COHORTS = {"cgmacros": "CGMacros", "shanghai_t2dm": "Shanghai T2DM"}
SUBS = ("overall", "post_meal_3h", "onset_hyper140")


def curve_table(rows: list[dict], models: list[str], ci: Callable) -> pd.DataFrame:
    """Cold-start curve: model x k MAE [95% CI]."""
    t = pd.DataFrame(rows)
    t = t[t.model.isin(models)].copy()
    t["v"] = [ci(a, b, 2) for a, b in zip(t.mae, t.mae_ci, strict=True)]
    p = t.pivot_table(index="model", columns="k", values="v", aggfunc="first")
    p.columns = [f"k={c:g}" for c in p.columns]
    return p.loc[[m for m in models if m in p.index]]


def hybrid(md: Callable, reg: Callable, paired: Callable) -> list[str]:
    """5.1 hybrid comparison tables."""
    f = M / "hybrid_cgmacros.json"
    if not f.exists():
        return []
    r = json.loads(f.read_text())["metrics"]
    s = [
        "## Part 5 — Hybrid twin and fast personalization (tag `part-5-done`)",
        "",
        "### 5.1 Hybrid twin (CGMacros, Part 4 windows: day >= 3, CV repeat 0)",
        "",
        "Source: `reports/metrics/hybrid_cgmacros.json`. A0 = LightGBM (CGM + time of day +",
        "meals + glycemic static, delta target) trained on the SAME rows as A; A = A0 + twin",
        "features; A_noact = A without activity features; B = LightGBM on the twin residual;",
        "C = physics-informed GRU (k = 3, 5-seed mean); C_lam0 = same network, no physics.",
        "`twin_pop_actoff_k<k>` = the Part 4 twin. Paired: model_1 − model_2 (< 0 = model_1",
        "better), patient bootstrap. Identical rows per k (D-038 to D-041).",
        "",
    ]
    for k in (0, 1, 3):
        if f"regression_k{k}" not in r:
            continue
        for sub, lab in zip(SUBS, ("overall", "post-meal 3 h", "onset"), strict=True):
            s += [f"**MAE, k = {k}, {lab}** ({r[f'n_patients_k{k}']} patients)", ""]
            s += [md(reg(r[f"regression_k{k}"], sub)), ""]
        s += [f"**Paired, k = {k}**", "", md(paired(r[f"paired_k{k}"], SUBS)), ""]
        s += [f"**Paired, k = {k}, without cgmacros_007**", ""]
        s += [md(paired(r[f"paired_k{k}_no007"], ("overall",))), ""]
    sv = r.get("seed_variance") or []
    if sv:
        t = pd.DataFrame(sv)
        t["seeds"] = [", ".join(f"{x:.2f}" for x in v) for v in t.seed_mae]
        t = t.set_index(["model", "horizon_min"])[["mean", "sd", "seeds"]].round(2)
        s += ["**Hybrid C seed variance (MAE per seed, mg/dL)**", "", md(t), ""]
    return s


def neural(md: Callable, reg: Callable, paired: Callable) -> list[str]:
    """5.1b onset-weighted neural loss."""
    f = M / "hybrid_neural.json"
    if not f.exists():
        return []
    r = json.loads(f.read_text())["metrics"]
    s = [
        "### 5.1b Neural onset-weighted loss (Part 3 GRU + FiLM, repeat 0, same rows)",
        "",
        "`nn_*` = Part 3 (unweighted), `nnw_*` = onset-weighted retrain (D-042). Source:",
        "`reports/metrics/hybrid_neural.json`.",
        "",
    ]
    for coh, name in COHORTS.items():
        if coh not in r:
            continue
        for sub in ("overall", "onset_hyper140"):
            s += [f"**{name}, {sub}**", "", md(reg(r[coh]["regression"], sub)), ""]
        s += [f"**{name}, paired**", ""]
        s += [md(paired(r[coh]["paired"], ("overall", "onset_hyper140"))), ""]
    return s


def personalize(md: Callable, ci: Callable) -> list[str]:
    """5.2 personalization tables (both cohorts)."""
    s: list[str] = []
    for coh, name in COHORTS.items():
        f = M / f"personalize_{coh}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())["metrics"]
        el = r["eligibility"]
        if not s:
            s += [
                "### 5.2 Personalization (cold start)",
                "",
                "Common test region: first period, day >= 7, for EVERY k (paired across the",
                "curve). `<base>_<s|ns>_<method>`: base lgbm (A0) or hybrid (best of A / B on",
                "training patients); s / ns = with / without static features; twin_s / twin_ns =",
                "Part 4 twin with static / population prior. All adaptation hyperparameters were",
                "chosen on training patients' out-of-fold forecasts (D-043).",
                "",
            ]
        s += [
            f"#### {name}",
            "",
            f"Eligible {el['n_eligible']} of {el['n_patients_total']} patients "
            f"({el['n_excluded']} excluded: < 96 anchors at day >= 7); later periods excluded: "
            f"{el['n_later_periods_excluded']}; test anchors {el['test_anchors']}.",
            "",
        ]
        ms = [f"{b}_{st}_{m}" for b in ("lgbm", "hybrid") for st in ("s", "ns")
              for m in ("none", "shrink", "finetune", "rls", "rls_cont")]  # fmt: skip
        ms += ["twin_s_twin", "twin_ns_twin"]
        for sub in ("overall", "onset"):
            key = f"curve_{sub}_60"
            if key in r:
                s += [f"**Cold-start curve, {sub}, +60 min, MAE (mg/dL)** "
                      f"({r[f'n_patients_{sub}_60']} patients)", ""]  # fmt: skip
                s += [md(curve_table(r[key], ms, ci)), ""]
        rows = []
        for base, co in r["crossover_overall_60"].items():
            rows.append({
                "base": base,
                "head start (mg/dL)": ci(co["head_start_mgdl"], co["head_start_ci"], 2),
                "best no-static method": co["best_no_static_method"],
                "crossover k* (days)": ci(co["crossover_k"], co["crossover_ci"], 2),
                "no crossover by k=7 (% resamples)": f"{100 * co['frac_no_crossover']:.0f}",
            })  # fmt: skip
        s += ["**Static head start and crossover (+60 min)**", ""]
        s += [md(pd.DataFrame(rows).set_index("base")), ""]
        t = pd.DataFrame(r["methods_overall_60"])
        t = t[t["diff_vs_none"].notna()].copy()
        t["d"] = [ci(a, b, 2) for a, b in zip(t.diff_vs_none, t.diff_ci, strict=True)]
        p = t.pivot_table(index="model", columns="k", values="d", aggfunc="first")
        p.columns = [f"k={c:g}" for c in p.columns]
        s += ["**Method − no adaptation, +60 min (mg/dL, < 0 = better)**", "", md(p), ""]
        g = pd.DataFrame(r["gains_overall_60"])
        g = g[g.k == 3.0].set_index("model")[["n_improve", "n_patients", "median_gain"]]
        s += ["**Per-patient gain at k = 3 (+60 min)**", "", md(g.round(2)), ""]
    return s


def part5(md: Callable, reg: Callable, paired: Callable, ci: Callable) -> list[str]:
    """Whole Part 5 section."""
    return hybrid(md, reg, paired) + neural(md, reg, paired) + personalize(md, ci)
