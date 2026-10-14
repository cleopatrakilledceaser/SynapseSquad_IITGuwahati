"""Part 5.2 evaluation: cold-start curves, crossover, method table, per-patient gains (D-043).

All numbers on the common test region (first period, day >= 7) of eligible patients; within a
cohort every (base, static, method, k) is evaluated on IDENTICAL rows (row kept only if every
model at every k has a forecast and the target is real). CIs: patient bootstrap (1,000).
Crossover k*: smallest k (linear interpolation between grid levels) at which the best no-static
adapted method's MAE <= the with-static population model at k = 0 (the static head start);
per bootstrap resample over patients; resamples with no crossover by k = 7 are counted.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.features.fusion_data import load_table
from src.utils.config import PROJECT_ROOT, load_config

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROC / "personalize"
HORIZONS = [30, 60, 120]
KEY = ["patient_id", "period_id", "t"]
METHODS = ["none", "scratch", "resid", "shrink", "finetune", "rls", "rls_cont"]


def load_long(cohort: str) -> tuple[pd.DataFrame, list[float]]:
    """Wide frame per k merged into one: columns '<model>@k<k>|h' + targets + onset flags."""
    pc, fc = load_config("personalize"), load_config("fusion")
    ks = pc["k_grid"]
    w = None
    for k in ks:
        parts = [
            pd.read_parquet(f) for f in sorted(OUT.glob(f"pers_{cohort}_fold*_k{k:g}.parquet"))
        ]
        if not parts:
            raise FileNotFoundError(f"personalize outputs missing for {cohort} k={k}")
        d = pd.concat(parts, ignore_index=True).drop(columns=["fold", "k"])
        d = d.rename(columns={c: c.replace("|", f"@k{k:g}|") for c in d.columns if "|" in c})
        w = d if w is None else w.merge(d, on=KEY, how="outer")
    tab = load_table(cohort, fc["n_hours"], fc)
    keep = (
        KEY + ["cgm_now"] + [f"y_{h}" for h in HORIZONS] + [f"hyper140_any_{h}" for h in HORIZONS]
    )
    w = w.merge(tab[keep], on=KEY, how="left")
    return w, ks


def model_names(w: pd.DataFrame) -> list[str]:
    """Model labels without k / horizon ('lgbm_s_none', 'twin_ns_twin', ...)."""
    return sorted({c.split("@")[0] for c in w.columns if "@k" in c})


def common_mask(w: pd.DataFrame, h: int) -> np.ndarray:
    """Rows where every model at every k and the target exist (paired across the curve)."""
    cols = [c for c in w.columns if c.endswith(f"|{h}") and "@k" in c]
    ok = w[f"y_{h}"].notna().to_numpy()
    for c in cols:
        ok &= w[c].notna().to_numpy()
    return ok


def per_patient_abs(w: pd.DataFrame, h: int, subset: str) -> tuple[pd.DataFrame, pd.Series]:
    """(patient x column) mean absolute error and patient row counts on the common rows."""
    ok = common_mask(w, h)
    if subset == "onset":
        ok &= ((w["cgm_now"] <= 140) & (w[f"hyper140_any_{h}"] == 1)).to_numpy()
    d = w[ok]
    cols = [c for c in d.columns if c.endswith(f"|{h}") and "@k" in c]
    err = d[cols].sub(d[f"y_{h}"], axis=0).abs()
    err["patient_id"] = d["patient_id"].to_numpy()
    g = err.groupby("patient_id")
    return g.sum(), g.size()


def pooled(sums: pd.DataFrame, n: pd.Series, idx: np.ndarray | None = None) -> pd.Series:
    """Pooled MAE per column (rows of resampled patients ``idx``)."""
    if idx is None:
        return sums.sum() / n.sum()
    return sums.iloc[idx].sum() / n.iloc[idx].sum()


def boot_matrix(sums: pd.DataFrame, n: pd.Series, nb: int, seed: int = 42) -> pd.DataFrame:
    """(nb x columns) pooled MAE over patient-bootstrap resamples.

    The same resamples are used for all columns, so differences between columns are paired.
    """
    rng = np.random.default_rng(seed)
    cnt = np.zeros((nb, len(n)))
    for b in range(nb):
        cnt[b] = np.bincount(rng.integers(0, len(n), len(n)), minlength=len(n))
    num = cnt @ sums.to_numpy(float)
    den = cnt @ n.to_numpy(float)
    return pd.DataFrame(num / den[:, None], columns=sums.columns)


def ref_column(c: str) -> str | None:
    """'<base>_<s|ns>_none' column matching a method column (None for none / twin)."""
    m, rest = c.split("@k")
    if m.startswith("twin_") or m.endswith("_none"):
        return None
    base, st = m.split("_")[:2]
    return f"{base}_{st}_none@k{rest}"


def curves(sums: pd.DataFrame, n: pd.Series, ks: list[float], nb: int, ci: float) -> list:
    """Cold-start curve rows: model x k -> MAE + patient-bootstrap CI."""
    bt = boot_matrix(sums, n, nb)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    est = pooled(sums, n)
    rows = []
    for c in sums.columns:
        m, rest = c.split("@k")
        rows.append(
            {
                "model": m,
                "k": float(rest.split("|")[0]),
                "mae": float(est[c]),
                "mae_ci": [float(np.percentile(bt[c], lo)), float(np.percentile(bt[c], hi))],
            }
        )
    return rows


def best_by_k(curve_rows: list, prefix: str, methods: list[str]) -> str:
    """Name of the method (with prefix) with the lowest mean MAE over k >= 0.5."""
    t = pd.DataFrame(curve_rows)
    t = t[t.model.isin([f"{prefix}_{m}" for m in methods]) & (t.k > 0)]
    return str(t.groupby("model").mae.mean().idxmin())


def crossover_rows(
    sums: pd.DataFrame, n: pd.Series, ks: list[float], base: str, nb: int, ci: float, curve_rows
) -> dict[str, Any]:
    """Head start and crossover k* for one base (lgbm | hybrid), patient bootstrap.

    Head start = MAE(no-static population, k = 0) - MAE(static population, k = 0).
    k* = first k where the best no-static ADAPTED method (chosen by mean MAE over k > 0 on the
    full sample; method choice is a reporting choice, not a model choice) reaches the static
    population model's k = 0 MAE.
    """
    if base == "twin":  # mechanistic twin: static vs population prior, MAP update on k days
        best, s0, ns0 = "twin_ns_twin", "twin_s_twin@k0|", "twin_ns_twin@k0|"
    else:
        # rls_cont uses data after day k (continuous updating), so it is not a k-day method
        meth = [m for m in METHODS if m not in ("none", "rls_cont")]
        best = best_by_k(curve_rows, f"{base}_ns", meth)
        s0, ns0 = f"{base}_s_none@k0|", f"{base}_ns_none@k0|"
    s0 = [c for c in sums.columns if c.startswith(s0)][0]
    ns0 = [c for c in sums.columns if c.startswith(ns0)][0]
    cur = [[c for c in sums.columns if c.startswith(f"{best}@k{k:g}|")][0] for k in ks]

    def stat(p: pd.Series) -> tuple[float, float]:
        return float(p[ns0] - p[s0]), crossover(p[cur].to_numpy(), ks, float(p[s0]))

    hs, kx = stat(pooled(sums, n))
    bt = boot_matrix(sums[list(dict.fromkeys([s0, ns0, *cur]))], n, nb)
    bs = [stat(bt.iloc[b]) for b in range(nb)]
    hsb, kxb = np.array([b[0] for b in bs]), np.array([b[1] for b in bs])
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    fin = kxb[np.isfinite(kxb)]
    return {
        "best_no_static_method": best,
        "head_start_positive": bool(hs > 0),
        "note": (
            "head start <= 0: the static population model is not better than the no-static one "
            "at k = 0, so the crossover is trivially k* = 0 and carries no information"
            if hs <= 0
            else ""
        ),
        "head_start_mgdl": hs,
        "head_start_ci": [float(np.percentile(hsb, lo)), float(np.percentile(hsb, hi))],
        "crossover_k": kx,
        "crossover_ci": (
            [float(np.percentile(kxb, lo)), float(np.percentile(kxb, hi))] if len(fin) else None
        ),
        "crossover_median": float(np.median(kxb)),
        "frac_no_crossover": float(np.mean(~np.isfinite(kxb))),
        "boot_k": [float(x) if np.isfinite(x) else None for x in kxb[:1000]],
    }


def method_table(sums: pd.DataFrame, n: pd.Series, ks_tab: list[float], nb: int, ci: float):
    """Method x base x static at k = 1/3/5: MAE and paired difference vs 'none' (same base)."""
    bt = boot_matrix(sums, n, nb)
    est_all = pooled(sums, n)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    rows = []
    for c in sums.columns:
        m, rest = c.split("@k")
        k = float(rest.split("|")[0])
        if k not in ks_tab or m.startswith("twin_"):
            continue
        est = float(est_all[c])
        r = {"model": m, "k": k, "mae": est}
        ref = ref_column(c)
        if ref and ref in sums.columns:
            dd = (bt[c] - bt[ref]).to_numpy()
            r |= {
                "diff_vs_none": float(est - est_all[ref]),
                "diff_ci": [float(np.percentile(dd, lo)), float(np.percentile(dd, hi))],
            }
        rows.append(r)
    return rows


def paired_k_rows(
    sums: pd.DataFrame,
    n: pd.Series,
    ks: list[float],
    pairs: list[tuple[str, str]],
    nb: int,
    ci: float,
) -> list[dict[str, Any]]:
    """Paired MAE difference (model_1 - model_2) at every k, patient bootstrap (< 0: 1 better)."""
    bt = boot_matrix(sums, n, nb)
    est = pooled(sums, n)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    rows = []
    for m1, m2 in pairs:
        for k in ks:
            c1 = [c for c in sums.columns if c.startswith(f"{m1}@k{k:g}|")]
            c2 = [c for c in sums.columns if c.startswith(f"{m2}@k{k:g}|")]
            if not c1 or not c2:
                continue
            d = (bt[c1[0]] - bt[c2[0]]).to_numpy()
            ci_ = [float(np.percentile(d, lo)), float(np.percentile(d, hi))]
            rows.append({"model_1": m1, "model_2": m2, "k": k,
                         "diff": float(est[c1[0]] - est[c2[0]]), "diff_ci": ci_,
                         "distinguishable_from_zero": bool(ci_[0] > 0 or ci_[1] < 0)})  # fmt: skip
    return rows


def key_pairs() -> list[tuple[str, str]]:
    """Static effect within each base and method; twin effect (hybrid vs lgbm) within each."""
    p = []
    for m in ("none", "shrink", "finetune", "rls", "rls_cont"):
        for b in ("lgbm", "hybrid"):
            p.append((f"{b}_s_{m}", f"{b}_ns_{m}"))  # static effect
        for st in ("s", "ns"):
            p.append((f"hybrid_{st}_{m}", f"lgbm_{st}_{m}"))  # twin effect
    p.append(("twin_s_twin", "twin_ns_twin"))  # static prior effect inside the twin
    return p


def gain_rows(sums: pd.DataFrame, n: pd.Series, ks_tab: list[float]) -> list[dict[str, Any]]:
    """Per-patient gain (MAE none - MAE method, same base/static/k): >0 = the patient improves."""
    pp = sums.div(n, axis=0)
    rows = []
    for c in pp.columns:
        m, rest = c.split("@k")
        k = float(rest.split("|")[0])
        ref = ref_column(c)
        if k not in ks_tab or ref is None:
            continue
        if ref not in pp.columns:
            continue
        g = (pp[ref] - pp[c]).to_numpy()
        rows.append({"model": m, "k": k, "gains": [float(x) for x in g],
                     "n_improve": int((g > 0).sum()), "n_patients": int(len(g)),
                     "median_gain": float(np.median(g))})  # fmt: skip
    return rows


def evaluate(cohort: str, exclude: list[str] | None = None) -> dict[str, Any]:
    """All personalization tables for one cohort (optionally excluding patients)."""
    pc = load_config("personalize")
    b = pc["bootstrap"]
    w, ks = load_long(cohort)
    if exclude:
        w = w[~w.patient_id.isin(exclude)]
    res: dict[str, Any] = {"k_grid": ks}
    for h in HORIZONS:
        for subset in ("overall", "onset"):
            sums, n = per_patient_abs(w, h, subset)
            if len(n) < 5:
                continue
            tag = f"{subset}_{h}"
            cr = curves(sums, n, ks, b["n_boot"], b["ci"])
            res[f"curve_{tag}"] = cr
            res[f"n_patients_{tag}"] = int(len(n))
            res[f"n_rows_{tag}"] = int(n.sum())
            if subset == "overall":
                res[f"crossover_{tag}"] = {
                    base: crossover_rows(sums, n, ks, base, b["n_boot"], b["ci"], cr)
                    for base in [*pc["bases"], "twin"]
                }
                res[f"methods_{tag}"] = method_table(sums, n, pc["table_k"], b["n_boot"], b["ci"])
                res[f"gains_{tag}"] = gain_rows(sums, n, pc["table_k"])
            res[f"paired_{tag}"] = paired_k_rows(sums, n, ks, key_pairs(), b["n_boot"], b["ci"])
    return res


def patients_per_k(cohort: str) -> dict[str, Any]:
    """Eligibility summary: patients / anchors in the common test region (same for every k)."""
    from src.personalize.data import eligible, first_period

    pc, fc = load_config("personalize"), load_config("fusion")
    tab = load_table(cohort, fc["n_hours"], fc)
    fp = first_period(tab)
    el = eligible(fp, pc["test_from_day"], pc["min_test_anchors"])
    ev = fp[fp.patient_id.isin(el) & (fp.days_since_start >= pc["test_from_day"])]
    adapt = {}
    for k in pc["k_grid"]:
        a = fp[fp.patient_id.isin(el) & (fp.days_since_start + 60 / 1440 < k) & fp.y_60.notna()]
        n = a.groupby("patient_id").size().reindex(el, fill_value=0)
        adapt[f"{k:g}"] = {
            "median_pairs_60": float(n.median()),
            "n_below_min_pairs": int((n < pc["ridge"]["min_pairs"]).sum()),
        }
    return {
        "n_patients_total": int(tab.patient_id.nunique()),
        "n_eligible": len(el),
        "n_excluded": int(tab.patient_id.nunique() - len(el)),
        "n_later_periods_excluded": int(tab.period_id.nunique() - fp.period_id.nunique()),
        "test_anchors": int(len(ev)),
        "adaptation_pairs": adapt,
    }


def main() -> None:
    """personalize_<cohort>.json (+ _no007 sensitivity for CGMacros) and figures."""
    from src.personalize.figures import all_figures
    from src.utils.provenance import file_hash, write_metrics

    pc = load_config("personalize")
    res = {}
    for coh in ("cgmacros", "shanghai_t2dm"):
        r = evaluate(coh)
        r["eligibility"] = patients_per_k(coh)
        if coh == "cgmacros":
            r["no007"] = evaluate(coh, pc["sensitivity_exclude"])
        r["choices"] = [json.loads(f.read_text()) for f in sorted(OUT.glob(f"pers_{coh}_*.json"))]
        for c in r["choices"]:  # keep the json small: drop per-patient alarm dicts / scores
            for cell in c.values():
                if isinstance(cell, dict):
                    for kk in [x for x in cell if x.startswith(("rls", "finetune"))]:
                        cell[kk] = {a: v for a, v in cell[kk].items() if a not in ("scores",)}
        write_metrics(
            f"personalize_{coh}", r, {"personalize": pc}, pc["seed"],
            file_hash(sorted(OUT.glob(f"pers_{coh}_*.parquet"))),
        )  # fmt: skip
        res[coh] = r
        print(f"written personalize_{coh}.json", flush=True)
    all_figures(res)
    print("personalize figures written", flush=True)


def crossover(curve: np.ndarray, ks: list[float], target: float) -> float:
    """Smallest k (linear interpolation) with curve(k) <= target; inf if none."""
    for i, (k, v) in enumerate(zip(ks, curve, strict=True)):
        if v <= target:
            if i == 0:
                return float(k)
            k0, v0 = ks[i - 1], curve[i - 1]
            return float(k0 + (k - k0) * (v0 - target) / (v0 - v)) if v0 != v else float(k)
    return float("inf")


if __name__ == "__main__":
    main()
