"""Part 5.2 runner (``make personalize``): cold-start personalization, both cohorts (D-043).

Per cohort x outer fold (CV repeat 0) x base (lgbm | hybrid) x static (s | ns) x k:
  population model on TRAINING patients (hybrid: rows day >= k, twin features from each training
  patient's own first k days; best of A / B by inner-val MAE) -> test patients' forecasts;
  training patients' out-of-fold forecasts (inner 3-fold refits, fixed hyperparameters) ->
  every adaptation hyperparameter is chosen on TRAINING patients' common test region only.
Methods: none, scratch, resid, shrink, finetune, rls (frozen at k), rls_cont (continuous),
twin (Part 4 twin, population / static prior; independent of base).
Output: data/processed/personalize/pers_<cohort>_fold<f>.parquet (wide: <base>_<s>_<m>|h).
Usage: python -m src.personalize.run [cgmacros|shanghai_t2dm|all]
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.features.fusion_data import load_table
from src.features.make_splits import strata_for
from src.hybrid.models import base_columns, hybrid_rows
from src.hybrid.twin_features import HORIZONS
from src.models.experiment import inner_split
from src.models.fusion_lgbm import KEYS, folds
from src.personalize import methods as pm
from src.personalize.data import (
    PopH,
    eligible,
    first_period,
    fit_pop,
    inner_folds,
    oof_forecasts,
)
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seed import set_global_seed

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROC / "personalize"
HYB = PROC / "hybrid"


def ridge_matrix(df: pd.DataFrame, cols: list[str], fill: pd.Series, cap: float) -> np.ndarray:
    """Ridge feature matrix; minutes-since-meal capped; NaN -> training-row means."""
    d = pd.DataFrame(index=df.index)
    for c in cols:
        if c == "meal_minutes_since_last_capped":
            d[c] = df["meal_minutes_since_last"].clip(upper=cap).fillna(cap)
        else:
            d[c] = df[c]
    return d.fillna(fill).to_numpy(float)


def mae_rows(y: np.ndarray, p: np.ndarray, pid: np.ndarray) -> float:
    """Patient-weighted MAE (mean over patients of per-patient MAE) on rows with real y."""
    ok = ~np.isnan(y) & ~np.isnan(p)
    s = pd.Series(np.abs(y[ok] - p[ok])).groupby(pid[ok]).mean()
    return float(s.mean())


class Adapter:
    """Selects adaptation hyperparameters on training patients and applies them to test ones.

    ``tr`` / ``te``: frames of FIRST-period rows of training (out-of-fold ``pop_h``) or test
    patients (outer ``pop_h``) with targets, ridge inputs and ``days_since_start``.
    """

    def __init__(self, pc: dict[str, Any], tr: pd.DataFrame, k: float) -> None:  # noqa: D107
        self.pc, self.k = pc, k
        rc = pc["ridge"]
        ok = set(tr.columns) | {"meal_minutes_since_last_capped"}
        self.cols = [c for c in rc["features"] if c in ok]
        if "meal_minutes_since_last" not in tr.columns:
            self.cols = [c for c in self.cols if c != "meal_minutes_since_last_capped"]
        self.cap = rc["meal_cap_min"]
        probe = ridge_matrix(tr, self.cols, pd.Series(dtype=float), self.cap)
        self.fill = pd.Series(np.nanmean(probe, 0), index=self.cols)
        self.test_from = pc["test_from_day"]

    def xmat(self, df: pd.DataFrame) -> np.ndarray:
        """Ridge inputs (NaN filled with training-row means)."""
        return ridge_matrix(df, self.cols, self.fill, self.cap)

    def per_patient(self, df: pd.DataFrame, h: int, lam: float, n0: float) -> dict[str, Any]:
        """Scratch / resid / shrink forecasts on the test region for every patient in ``df``."""
        out = {m: np.full(len(df), np.nan) for m in ("scratch", "resid", "shrink")}
        y = df[f"y_{h}"].to_numpy(float)
        pop = df[f"pop_{h}"].to_numpy(float)
        base = df["cgm_now"].to_numpy(float)
        days = df["days_since_start"].to_numpy(float)
        x = self.xmat(df)
        ev = days >= self.test_from
        for _, rows in df.groupby("patient_id").indices.items():
            a = rows[pm.adapt_mask(days[rows], h, self.k, y[rows])]
            e = rows[ev[rows]]
            n = len(a)
            if n < self.pc["ridge"]["min_pairs"]:
                for m in out:
                    out[m][e] = pop[e]
                continue
            sm = pm.ridge_fit(x[a], y[a] - base[a], lam)
            rm = pm.ridge_fit(x[a], y[a] - pop[a], lam)
            scr = base[e] + pm.ridge_predict(sm, x[e])
            out["scratch"][e] = scr
            out["resid"][e] = pop[e] + pm.ridge_predict(rm, x[e])
            w = pm.shrink_weight(n, n0)
            out["shrink"][e] = w * scr + (1 - w) * pop[e]
        return out

    def select_linear(self, tr: pd.DataFrame, h: int) -> tuple[float, float, dict]:
        """(lambda, n0) on training patients: lambda by scratch+resid MAE, n0 by shrink MAE."""
        y, pid = tr[f"y_{h}"].to_numpy(float), tr["patient_id"].to_numpy()
        ev = tr["days_since_start"].to_numpy(float) >= self.test_from
        rc, sc = self.pc["ridge"], self.pc["shrink"]
        lam_s = {}
        for lam in rc["lambda_grid"]:
            o = self.per_patient(tr, h, lam, sc["n0_grid"][0])
            lam_s[lam] = np.mean([mae_rows(y[ev], o[m][ev], pid[ev]) for m in ("scratch", "resid")])
        lam = float(min(lam_s, key=lam_s.get))
        n0_s = {}
        for n0 in sc["n0_grid"]:
            n0_s[n0] = mae_rows(y[ev], self.per_patient(tr, h, lam, n0)["shrink"][ev], pid[ev])
        n0 = float(min(n0_s, key=n0_s.get))
        return lam, n0, {"lambda": lam_s, "n0": n0_s}

    def rls_x(self, df: pd.DataFrame, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
        """Standardized RLS features + intercept."""
        x = df[self.pc["rls"]["features"]].to_numpy(float)
        x = np.nan_to_num((x - mu) / sd)
        return np.column_stack([np.ones(len(x)), x])

    def rls(
        self,
        df: pd.DataFrame,
        h: int,
        hp: tuple[float, float, float],
        limit: float,
        xs: tuple,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """RLS-corrected forecasts (all rows) and alarm counts; limit = k (frozen) or inf.

        ``hp`` = (forgetting factor, Page-Hinkley lambda, prior variance p0).
        """
        lam, phl, p0 = hp
        rc = self.pc["rls"]
        out, alarms = df[f"pop_{h}"].to_numpy(float).copy(), {}
        x = self.rls_x(df, *xs)
        res = df[f"y_{h}"].to_numpy(float) - df[f"pop_{h}"].to_numpy(float)
        tmin = df["days_since_start"].to_numpy(float) * pm.MIN_PER_DAY
        for p, rows in df.groupby("patient_id").indices.items():
            o = rows[np.argsort(tmin[rows], kind="stable")]
            corr, na = pm.rls_run(
                x[o],
                res[o],
                tmin[o],
                float(h),
                limit * pm.MIN_PER_DAY,
                lam,
                p0,
                rc["ph_delta"],
                phl,
                rc["boost"],
            )
            out[o] = out[o] + corr
            alarms[p] = int(na)
        return out, alarms

    def select_rls(self, tr: pd.DataFrame, h: int, xs: tuple, cont: bool) -> tuple:
        """(forgetting, PH lambda, p0) with the lowest MAE on training patients' test region."""
        y, pid = tr[f"y_{h}"].to_numpy(float), tr["patient_id"].to_numpy()
        ev = tr["days_since_start"].to_numpy(float) >= self.test_from
        rc, sc = self.pc["rls"], {}
        lim = np.inf if cont else self.k
        for lam in rc["forgetting_grid"]:
            for phl in rc["ph_lambda_grid"]:
                for p0 in rc["p0_grid"]:
                    p, _ = self.rls(tr, h, (lam, phl, p0), lim, xs)
                    sc[f"{lam}|{phl}|{p0}"] = mae_rows(y[ev], p[ev], pid[ev])
        hp = tuple(float(v) for v in min(sc, key=sc.get).split("|"))
        return hp, sc


def finetune_preds(
    df: pd.DataFrame,
    h: int,
    k: float,
    strings: dict[str, str],
    n_init: dict[str, int],
    pop: PopH,
    pc: dict[str, Any],
    bc: dict[str, Any],
) -> dict[tuple[float, int], np.ndarray]:
    """Fine-tuned forecasts on the test region for every (lr, trees) grid point.

    ``strings[p]`` = trimmed population booster that did NOT see patient p (outer model for test
    patients, the inner out-of-fold model for training patients). Rows with fewer than
    min_pairs adaptation pairs keep the population forecast.
    """
    ft = pc["finetune"]
    tmax = max(ft["trees_grid"])
    y = df[f"y_{h}"].to_numpy(float)
    base = df[pop.base].to_numpy(float)
    days = df["days_since_start"].to_numpy(float)
    popf = df[f"pop_{h}"].to_numpy(float)
    ev = days >= pc["test_from_day"]
    out = {(lr, t): popf.copy() for lr in ft["lr_grid"] for t in ft["trees_grid"]}
    for p, rows in df.groupby("patient_id").indices.items():
        a = rows[pm.adapt_mask(days[rows], h, k, y[rows])]
        e = rows[ev[rows]]
        if len(a) < pc["ridge"]["min_pairs"] or len(e) == 0:
            continue
        xa = df.iloc[a][pop.cols]
        xe = df.iloc[e][pop.cols]
        for lr in ft["lr_grid"]:
            prm = pm.finetune_params(bc["lightgbm"]["params"], ft, lr)
            b = pm.finetune(strings[p], n_init[p], xa, y[a] - base[a], prm, tmax)
            for t in ft["trees_grid"]:
                extra = b.predict(xe, start_iteration=n_init[p], num_iteration=t)
                out[(lr, t)][e] = popf[e] + extra
    return out


def select_finetune(tr: pd.DataFrame, h: int, preds: dict, test_from: float) -> tuple:
    """(lr, trees) with the lowest patient-weighted MAE on training patients' test region."""
    y, pid = tr[f"y_{h}"].to_numpy(float), tr["patient_id"].to_numpy()
    ev = tr["days_since_start"].to_numpy(float) >= test_from
    sc = {f"{lr}|{t}": mae_rows(y[ev], p[ev], pid[ev]) for (lr, t), p in preds.items()}
    lr, t = min(sc, key=sc.get).split("|")
    return float(lr), int(t), sc


def base_frames(
    tab: pd.DataFrame,
    cohort: str,
    fold: int,
    k: float,
    train: list[str],
    test: list[str],
    twin: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(population-fit rows, training-patient first-period rows, test first-period rows).

    Plain LightGBM: fit rows = all anchors of training patients (as Part 3). Hybrid: fit rows =
    training patients' anchors with day >= k carrying twin features from their own first k days
    (D-038); training / test frames carry twin features at the same k (test: population prior).
    """
    fp = first_period(tab)
    if not twin:
        return (
            tab[tab.patient_id.isin(train)],
            fp[fp.patient_id.isin(train)],
            fp[fp.patient_id.isin(test)],
        )
    fit = hybrid_rows(tab, cohort, fold, k, train, "train", k)
    trf = hybrid_rows(fp, cohort, fold, k, train, "train", 0.0)
    tef = hybrid_rows(fp, cohort, fold, k, test, "test", 0.0)
    return fit, trf, tef


def pop_variants(cohort: str, hc: dict, tab: pd.DataFrame, static: bool, twin: bool) -> dict:
    """Candidate population models: name -> (columns, residual)."""
    a0 = base_columns(cohort, hc, tab)
    if not static:
        a0 = [c for c in a0 if not c.startswith("static_")]
    if not twin:
        return {"A0": (a0, False)}
    tw = list(hc["lgbm"]["twin_features"])
    return {"A": (a0 + tw, False), "B": (a0 + tw, True)}


def run_cell(
    tab: pd.DataFrame,
    cohort: str,
    fold: int,
    k: float,
    train: list[str],
    test: list[str],
    twin: bool,
    static: bool,
    strata: dict,
    cfg: dict,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every method for one (fold, base, static, k) -> test predictions + choices."""
    pc, hc, bc = cfg["pc"], cfg["hc"], cfg["bc"]
    fit, trf, tef = base_frames(tab, cohort, fold, k, train, test, twin)
    in_tr, in_va = inner_split(train, strata, bc, 42 + pc["cv_repeat"])
    best, ch = None, {}
    for name, (cols, resid) in pop_variants(cohort, hc, tab, static, twin).items():
        pop, chosen = fit_pop(
            fit[fit.patient_id.isin(in_tr)], fit[fit.patient_id.isin(in_va)], cols, resid, bc, 2
        )
        score = float(np.mean([chosen[f"inner_val_mae_{h}"] for h in HORIZONS]))
        ch[f"pop_{name}"] = {"inner_val_mae": score, **chosen}
        if best is None or score < best[2]:
            best = (name, pop, score)
    name, pop, _ = best
    ch["pop_chosen"] = name
    fmap = inner_folds(train, pc["inner_folds"], 42 + fold)
    trf, strings = oof_forecasts(fit, trf, pop, fmap, bc, 2)
    tef = tef.copy()
    for h in HORIZONS:
        tef[f"pop_{h}"] = pop[h].predict(tef)
    ad = Adapter(pc, trf, k)
    ev_te = tef["days_since_start"].to_numpy(float) >= pc["test_from_day"]
    out = tef.loc[ev_te, KEYS].copy()
    rc = pc["rls"]["features"]
    xs = (np.nanmean(trf[rc].to_numpy(float), 0), np.nanstd(trf[rc].to_numpy(float), 0) + 1e-6)
    for h in HORIZONS:
        out[f"none|{h}"] = tef.loc[ev_te, f"pop_{h}"].to_numpy()
        lam, n0, sc = ad.select_linear(trf, h)
        ch[f"linear_{h}"] = {"lambda": lam, "n0": n0}
        o = ad.per_patient(tef, h, lam, n0)
        for m in ("scratch", "resid", "shrink"):
            out[f"{m}|{h}"] = o[m][ev_te]
        tr_str = {p: strings[fmap[p]][h] for p in train}
        te_str = {p: pop[h].model_str() for p in test}
        n_tr = {p: pop[h].n_iter for p in train}
        n_te = {p: pop[h].n_iter for p in test}
        ptr = finetune_preds(trf, h, k, tr_str, n_tr, pop[h], pc, bc)
        lr, nt, fsc = select_finetune(trf, h, ptr, pc["test_from_day"])
        ch[f"finetune_{h}"] = {"lr": lr, "trees": nt, "scores": fsc}
        pte = finetune_preds(tef, h, k, te_str, n_te, pop[h], pc, bc)
        out[f"finetune|{h}"] = pte[(lr, nt)][ev_te]
        for mode, cont in (("rls", False), ("rls_cont", True)):
            hp, rsc = ad.select_rls(trf, h, xs, cont)
            p, alarms = ad.rls(tef, h, hp, np.inf if cont else k, xs)
            out[f"{mode}|{h}"] = p[ev_te]
            ch[f"{mode}_{h}"] = {
                "forgetting": hp[0],
                "ph_lambda": hp[1],
                "p0": hp[2],
                "scores": rsc,
                "alarms": alarms,
            }
    return out, ch


def twin_preds(tab: pd.DataFrame, cohort: str, fold: int, k: float, test: list[str], tf: float):
    """Method 6: Part 4 twin forecasts (first k days, population / static prior) on the region."""
    fp = first_period(tab)
    ev = fp[fp.patient_id.isin(test) & (fp.days_since_start >= tf)][KEYS + ["cgm_now"]]
    clip = tuple(load_config("data")["cgm"]["glucose_range_mgdl"][cohort])
    f = pd.read_parquet(HYB / f"twinfeat_{cohort}_fold{fold}.parquet")
    f = f[(f.k == k) & (f.role == "test")]
    for prior, lab in (("pop", "ns"), ("static", "s")):
        g = f[f.prior == prior][["period_id", "t"] + [f"tw_mu_{h}" for h in HORIZONS]]
        ren = {f"tw_mu_{h}": f"twin_{lab}_twin|{h}" for h in HORIZONS}
        ev = ev.merge(g.rename(columns=ren), on=["period_id", "t"], how="left")
        for h in HORIZONS:
            ev[f"twin_{lab}_twin|{h}"] = ev[f"twin_{lab}_twin|{h}"].clip(*clip)
    return ev.drop(columns="cgm_now")


def fold_job(cohort: str, fold: int, train: list[str], test: list[str], k: float) -> None:
    """All bases x static x methods for one (cohort, fold, k) -> parquet + json."""
    path = OUT / f"pers_{cohort}_fold{fold}_k{k:g}.parquet"
    if path.exists():
        return
    t0 = time.time()
    cfg = {n: load_config(m) for n, m in (("pc", "personalize"), ("hc", "hybrid"))}
    cfg["bc"] = load_config("baselines")
    fc, sc = load_config("fusion"), load_config("splits")
    pc = cfg["pc"]
    tab = load_table(cohort, fc["n_hours"], fc)
    strata = strata_for(pd.read_parquet(PROC / "static.parquet"), cohort, sc)
    elig = eligible(first_period(tab), pc["test_from_day"], pc["min_test_anchors"])
    test = [p for p in test if p in elig]
    out = twin_preds(tab, cohort, fold, k, test, pc["test_from_day"])
    choices: dict[str, Any] = {"fold": fold, "k": k, "n_test": len(test)}
    for bname, bcfg in pc["bases"].items():
        for static in (True, False):
            lab = f"{bname}_{'s' if static else 'ns'}"
            o, ch = run_cell(tab, cohort, fold, k, train, test, bcfg["twin"], static, strata, cfg)
            o = o.rename(columns={c: f"{lab}_{c}" for c in o.columns if "|" in c})
            out = out.merge(o, on=KEYS, how="left")
            choices[lab] = ch
    out["fold"], out["k"] = fold, k
    choices["seconds"] = round(time.time() - t0, 1)
    (OUT / f"pers_{cohort}_fold{fold}_k{k:g}.json").write_text(
        json.dumps(choices, indent=1, default=str)
    )
    out.to_parquet(path, index=False)
    print(f"  {path.name}: {len(out)} rows, {choices['seconds']:.0f}s", flush=True)


def run_cohort(cohort: str, n_jobs: int = 4) -> None:
    """Every (fold, k) of one cohort, in parallel processes."""
    pc = load_config("personalize")
    jobs = [
        (cohort, f, tr, te, k)
        for _, f, tr, te in folds(cohort, [pc["cv_repeat"]])
        for k in pc["k_grid"]
    ]
    Parallel(n_jobs=n_jobs)(delayed(fold_job)(*j) for j in jobs)


def main() -> None:
    """Run the requested cohort(s)."""
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    set_global_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    for c in ["cgmacros", "shanghai_t2dm"] if which == "all" else [which]:
        t0 = time.time()
        print(f"[{c}]", flush=True)
        run_cohort(c)
        print(f"[{c}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
