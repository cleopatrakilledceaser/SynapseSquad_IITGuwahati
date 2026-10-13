"""Part 3 evaluation (second stage of ``make fusion``).

Ablation, leave-one-group-out, paired comparisons, neural vs LightGBM vs ensemble, input tiers,
events (onset-only + all windows), groups.

All CIs: patient bootstrap (1,000 resamples). Per-patient statistics are averaged over CV
repeats so each patient counts once. Paired differences use identical rows for both models.
Part 2 predictions (persistence, per-patient AR) are reused from data/processed/predictions/.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.eval.metrics import (
    bootstrap_event,
    bootstrap_point,
    paired_bootstrap_diff,
    point_stats,
    reliability,
)
from src.features.fusion_data import load_table
from src.utils.config import PROJECT_ROOT, load_config

PROC = PROJECT_ROOT / "data" / "processed"
FUS = PROC / "fusion"
J = ["period_id", "t", "repeat"]
HORIZONS = [30, 60, 120]
EVENTS = {
    "hyper180_any": ("above", 180),
    "hyper140_any": ("above", 140),
    "hypo70_any": ("below", 70),
}


def _add(w: pd.DataFrame, df: pd.DataFrame, name: str, on: list[str]) -> pd.DataFrame:
    """Left-merge prediction columns pred_<h> of ``df`` into ``w`` as '<name>|<h>'."""
    ren = {f"pred_{h}": f"{name}|{h}" for h in HORIZONS}
    return w.merge(df[on + list(ren)].rename(columns=ren), on=on, how="left")


def build_wide(cohort: str, fc: dict[str, Any]) -> pd.DataFrame:
    """One row per (anchor, CV repeat) with targets, subsets and every model's prediction."""
    p2_cols = J + ["patient_id"]
    p2_cols += [f"pred_{m}_{h}" for m in ("persistence", "ar_per_patient") for h in HORIZONS]
    p2_cols += [f"score_persistence_{e}_{h}" for e in EVENTS for h in HORIZONS]
    w = pd.read_parquet(PROC / "predictions" / f"{cohort}_cv_oof.parquet", columns=p2_cols)
    w = w.rename(
        columns={
            f"pred_{m}_{h}": f"{m}|{h}" for m in ("persistence", "ar_per_patient") for h in HORIZONS
        }
    )
    tab = load_table(cohort, fc["n_hours"], fc)
    keep = ["period_id", "t", "cgm_now", "meal_minutes_since_last"] + [f"y_{h}" for h in HORIZONS]
    keep += [f"{e}_{h}" for e in EVENTS for h in HORIZONS]
    w = w.merge(tab[keep], on=["period_id", "t"], how="left", validate="many_to_one")
    for f in sorted(FUS.glob(f"lgbm_{cohort}_*.parquet")):
        name = f.stem.replace(f"lgbm_{cohort}_", "lgbm_")
        df = pd.read_parquet(f)
        if name.endswith("_events_delta"):
            pc = [c for c in df.columns if c.startswith("prob_")]
            w = w.merge(
                df[J + pc].rename(columns={c: f"{name}|{c[5:]}" for c in pc}), on=J, how="left"
            )
        else:
            w = _add(w, df, name, J)
    for f in sorted(FUS.glob(f"neural_{cohort}_*_delta.parquet")):
        df = pd.read_parquet(f)
        name = f.stem.replace(f"neural_{cohort}_", "nn_").replace("_delta", "")
        avg = df.groupby(J, as_index=False)[[f"pred_{h}" for h in HORIZONS]].mean()
        w = _add(w, avg, name, J)  # seed-averaged prediction
        for s, g in df.groupby("seed"):
            w = _add(w, g, f"{name}_s{s}", J)
    if "nn_E_full|60" in w and "lgbm_E_full_delta|60" in w:
        for h in HORIZONS:  # simple ensemble: mean of LightGBM E (delta) and neural E (seed avg)
            w[f"ens_E_full|{h}"] = (w[f"lgbm_E_full_delta|{h}"] + w[f"nn_E_full|{h}"]) / 2
    fs = FUS / f"fingersticks_{cohort}.parquet"
    if fs.exists():
        fdf = pd.read_parquet(fs)
        for sch in fc["fingerstick"]["schedules"]:
            fdf[[f"fs_persist_{sch}|{h}" for h in HORIZONS]] = np.repeat(
                fdf[[f"{sch}_fs_last"]].to_numpy(), len(HORIZONS), axis=1
            )
        cols = [c for c in fdf.columns if c.startswith("fs_persist_")]
        w = w.merge(fdf[["period_id", "t"] + cols], on=["period_id", "t"], how="left")
    return w


def stats(d: pd.DataFrame, models: list[str], h: int) -> list[pd.DataFrame]:
    """Per-patient sufficient statistics per model, on rows where ALL models and y exist.

    Statistics are averaged over the CV repeats present, so each patient counts once.
    """
    ok = d[f"y_{h}"].notna()
    for m in models:
        ok &= d[f"{m}|{h}"].notna()
    d = d[ok]
    if d.empty:
        return [pd.DataFrame() for _ in models]
    n_rep = d["repeat"].nunique()
    out = []
    for m in models:
        per = [point_stats(g, f"y_{h}", f"{m}|{h}") for _, g in d.groupby("repeat")]
        out.append(pd.concat(per).groupby(level=0).sum() / n_rep)
    return out


def subsets(w: pd.DataFrame, cohort: str, fc: dict[str, Any], h: int) -> dict[str, pd.Series]:
    """Row masks: overall, post-meal (CGMacros), rising onset (start <= 140, crosses within h)."""
    thr = EVENTS["hyper140_any"][1]
    m = {"overall": pd.Series(True, index=w.index)}
    if cohort == "cgmacros":
        m["post_meal_3h"] = w["meal_minutes_since_last"] <= fc["post_meal_window_min"]
    m["onset_hyper140"] = (w["cgm_now"] <= thr) & (w[f"hyper140_any_{h}"] == 1)
    return m


def regression_rows(
    w: pd.DataFrame,
    models: list[str],
    cohort: str,
    fc: dict[str, Any],
    bc: dict[str, Any],
    exclude: list[str] | None = None,
    repeats: list[int] | None = None,
) -> list[dict[str, Any]]:
    """MAE/RMSE/MARD/Clarke with CIs per model x horizon x subset (models present only)."""
    b = bc["bootstrap"]
    d = w[~w.patient_id.isin(exclude or [])]
    if repeats is not None:
        d = d[d["repeat"].isin(repeats)]
    rows = []
    for h in HORIZONS:
        for sname, mask in subsets(d, cohort, fc, h).items():
            dd = d[mask]
            for m in models:
                if f"{m}|{h}" not in dd or dd[f"{m}|{h}"].notna().sum() == 0:
                    continue
                (st,) = stats(dd, [m], h)
                r = bootstrap_point(st, b["n_boot"], 42, b["ci"])
                rows.append({"model": m, "horizon_min": h, "subset": sname, **r})
    return rows


def paired_rows(
    w: pd.DataFrame,
    pairs: list[tuple[str, str]],
    cohort: str,
    fc: dict[str, Any],
    bc: dict[str, Any],
    exclude: list[str] | None = None,
    repeats: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Paired patient-bootstrap MAE differences (m1 - m2) on identical rows (< 0: m1 better)."""
    b = bc["bootstrap"]
    d = w[~w.patient_id.isin(exclude or [])]
    if repeats is not None:
        d = d[d["repeat"].isin(repeats)]
    rows = []
    for h in HORIZONS:
        for sname, mask in subsets(d, cohort, fc, h).items():
            for m1, m2 in pairs:
                if f"{m1}|{h}" not in d or f"{m2}|{h}" not in d:
                    continue
                s1, s2 = stats(d[mask], [m1, m2], h)
                if s1.empty:
                    continue
                r = paired_bootstrap_diff(s1, s2, b["n_boot"], 42, b["ci"])
                r["mae_1"] = float(s1["abs"].sum() / s1["n"].sum())
                r["mae_2"] = float(s2["abs"].sum() / s2["n"].sum())
                rows.append({"model_1": m1, "model_2": m2, "horizon_min": h, "subset": sname, **r})
    return rows


def event_rows(
    w: pd.DataFrame,
    cohort: str,
    fc: dict[str, Any],
    bc: dict[str, Any],
    exclude: list[str] | None = None,
) -> list[dict[str, Any]]:
    """AUROC/AUPRC/Brier for event classifiers (repeat 0): all windows AND onset-only windows."""
    b, e = bc["bootstrap"], bc["events"]
    d = w[(w["repeat"] == 0) & ~w.patient_id.isin(exclude or [])]
    limited = set(fc["limited_evidence"].get(cohort, []))
    rows = []
    for ev, (direction, thr) in EVENTS.items():
        for h in HORIZONS:
            lab = f"{ev}_{h}"
            onset = d["cgm_now"] <= thr if direction == "above" else d["cgm_now"] >= thr
            scorers = {"persistence_value": f"score_persistence_{lab}"}
            for cfg in fc["event_classifier"]["configs"]:
                scorers[f"lgbm_{cfg}"] = f"lgbm_{cfg}_events_delta|{lab}"
            for subset, dd in (("all_windows", d), ("onset_only", d[onset])):
                for sname, col in scorers.items():
                    if col not in dd or dd[col].notna().sum() == 0:
                        continue
                    r = bootstrap_event(dd, lab, col, b["n_boot"], 42, b["ci"], b["auc_bins"])
                    if sname == "persistence_value":
                        r["brier"], r["brier_ci"] = None, None
                    rel = reliability(r, e["min_pos_windows"], e["min_pos_patients"])
                    if ev.split("_")[0] in limited:
                        rel = "limited evidence"
                    rows.append(
                        {"event": lab, "subset": subset, "scorer": sname, "reliability": rel, **r}
                    )
    return rows


def group_rows(
    w: pd.DataFrame, models: list[str], cohort: str, bc: dict[str, Any], h: int = 60
) -> list[dict[str, Any]]:
    """MAE by diabetes status (CGMacros) / HbA1c group, patient-bootstrap CIs."""
    s = pd.read_parquet(PROC / "static.parquet").set_index("patient_id")
    gc = bc["groups"]
    grp = {
        "hba1c_group": pd.cut(
            s["hba1c_pct"], gc["hba1c_bins_pct"], labels=gc["hba1c_labels"], right=False
        ).astype(str)
    }
    if cohort == "cgmacros":
        grp["diabetes_status"] = s["diabetes_status"].astype(str)
    rows = []
    for gname, gmap in grp.items():
        for gval in sorted(gmap.unique()):
            d = w[w.patient_id.isin(gmap[gmap == gval].index)]
            if d.empty:
                continue
            sts = stats(d, models, h)
            for m, st in zip(models, sts, strict=True):
                r = bootstrap_point(st, bc["bootstrap"]["n_boot"], 42, bc["bootstrap"]["ci"])
                rows.append(
                    {
                        "grouping": gname,
                        "group": gval,
                        "model": m,
                        "horizon_min": h,
                        "n_patients": r["n_patients"],
                        "mae": r["mae"],
                        "mae_ci": r["mae_ci"],
                    }
                )
    return rows


def seed_rows(w: pd.DataFrame, cohort: str) -> list[dict[str, Any]]:
    """Neural seed variance: pooled MAE per seed (repeat 0) and across-seed SD."""
    d = w[w["repeat"] == 0]
    rows = []
    for name in ("nn_B_cgm", "nn_E_full"):
        seeds = sorted({c.split("|")[0] for c in d.columns if c.startswith(f"{name}_s")})
        for h in HORIZONS:
            maes = []
            for s in seeds:
                ok = d[f"y_{h}"].notna() & d[f"{s}|{h}"].notna()
                maes.append(float((d.loc[ok, f"{s}|{h}"] - d.loc[ok, f"y_{h}"]).abs().mean()))
            if maes:
                rows.append(
                    {
                        "model": name,
                        "horizon_min": h,
                        "seed_mae": maes,
                        "mean": float(np.mean(maes)),
                        "sd": float(np.std(maes, ddof=1)),
                        "range": [min(maes), max(maes)],
                    }
                )
    return rows


def model_lists(w: pd.DataFrame, cohort: str, fc: dict[str, Any]) -> dict[str, list[str]]:
    """Model names grouped for the tables (only those present in the wide frame)."""
    have = {c.split("|")[0] for c in w.columns if "|" in c}
    base = ["persistence", "ar_per_patient"]
    abl = [f"lgbm_{n}_{t}" for n in fc["ablation"][cohort] for t in fc["targets"]]
    logo = [f"lgbm_{n}_{t}" for n in fc["logo"][cohort] for t in fc["targets"]]
    tiers = [
        f"lgbm_{n}_{t}"
        for n, s in fc["tiers"][cohort].items()
        if "fingerstick" in s
        for t in fc["targets"]
    ] + [f"fs_persist_{s}" for s in fc["fingerstick"]["schedules"]]
    fam = ["lgbm_B_cgm_delta", "lgbm_E_full_delta", "nn_B_cgm", "nn_E_full", "ens_E_full"]
    keep = lambda xs: [x for x in xs if x in have]  # noqa: E731
    return {
        "ablation": keep(base + abl),
        "logo": keep(["lgbm_E_full_abs", "lgbm_E_full_delta"] + logo),
        "tiers": keep(base + ["lgbm_E_full_delta", "lgbm_E_full_abs"] + tiers),
        "families": keep(base + fam),
    }


def logo_pairs(cohort: str, fc: dict[str, Any]) -> list[tuple[str, str]]:
    """(E minus group, E) pairs for both targets: positive difference = the group helped."""
    return [
        (f"lgbm_{n}_{t}", f"lgbm_E_full_{t}") for n in fc["logo"][cohort] for t in fc["targets"]
    ]


def key_pairs(cohort: str, fc: dict[str, Any]) -> list[tuple[str, str]]:
    """Owner's key comparisons for both targets (+ neural/ensemble vs LightGBM E)."""
    out = []
    for a, b in fc["key_comparisons"][cohort]:
        for t in fc["targets"]:
            na = a if a in ("ar_per_patient", "persistence") else f"lgbm_{a}_{t}"
            nb = b if b in ("ar_per_patient", "persistence") else f"lgbm_{b}_{t}"
            out.append((na, nb))
    out += [
        ("nn_E_full", "lgbm_E_full_delta"),
        ("ens_E_full", "lgbm_E_full_delta"),
        ("nn_E_full", "nn_B_cgm"),
        ("ens_E_full", "ar_per_patient"),
        ("nn_B_cgm", "lgbm_B_cgm_delta"),
        ("nn_B_cgm", "ar_per_patient"),
    ]
    return list(dict.fromkeys(out))


def evaluate_cohort(cohort: str, fc: dict[str, Any], bc: dict[str, Any]) -> dict[str, Any]:
    """All Part 3 tables for one cohort."""
    w = build_wide(cohort, fc)
    ml = model_lists(w, cohort, fc)
    excl = bc["sensitivity_exclude"].get(cohort, [])
    tag = "_".join(x.split("_")[1] for x in excl)
    nn_reps = fc["neural"]["repeats"]
    res: dict[str, Any] = {
        "n_rows_per_repeat": int((w["repeat"] == 0).sum()),
        "n_patients": int(w.patient_id.nunique()),
        "models": ml,
        "ablation": regression_rows(w, ml["ablation"], cohort, fc, bc),
        "logo": regression_rows(w, ml["logo"], cohort, fc, bc),
        "logo_paired": paired_rows(w, logo_pairs(cohort, fc), cohort, fc, bc),
        "key_paired": paired_rows(
            w,
            [p for p in key_pairs(cohort, fc) if not p[0].startswith(("nn_", "ens_"))],
            cohort,
            fc,
            bc,
        ),
        # neural is repeat 0 only: families compared on repeat 0 for every model (paired)
        "families_r0": regression_rows(w, ml["families"], cohort, fc, bc, repeats=nn_reps),
        "families_paired_r0": paired_rows(
            w,
            [p for p in key_pairs(cohort, fc) if p[0].startswith(("nn_", "ens_"))]
            + [("lgbm_E_full_delta", "lgbm_B_cgm_delta")],
            cohort,
            fc,
            bc,
            repeats=nn_reps,
        ),
        "seed_variance": seed_rows(w, cohort),
        "tiers": regression_rows(w, ml["tiers"], cohort, fc, bc),
        "events": event_rows(w, cohort, fc, bc),
        "groups_60min": group_rows(
            w,
            [
                m
                for m in ("persistence", "ar_per_patient", "lgbm_B_cgm_delta", "lgbm_E_full_delta")
                if m in ml["ablation"]
            ],
            cohort,
            bc,
        ),
    }
    if excl:
        res[f"ablation_without_{tag}"] = regression_rows(w, ml["ablation"], cohort, fc, bc, excl)
        res[f"key_paired_without_{tag}"] = paired_rows(
            w,
            [p for p in key_pairs(cohort, fc) if not p[0].startswith(("nn_", "ens_"))],
            cohort,
            fc,
            bc,
            excl,
        )
        res[f"events_without_{tag}"] = event_rows(w, cohort, fc, bc, excl)
    sel = FUS / f"neural_{cohort}_selection.json"
    if sel.exists():
        import json

        res["neural_selection"] = json.loads(sel.read_text())
    return res


def main() -> None:
    """Compute all Part 3 metrics and figures (``--figures-only`` reuses the saved JSON)."""
    import json
    import sys

    from src.eval.fusion_figures import all_figures
    from src.utils.provenance import file_hash, write_metrics

    fc, bc = load_config("fusion"), load_config("baselines")
    metrics = PROJECT_ROOT / "reports" / "metrics"
    results = {}
    for coh in ("cgmacros", "shanghai_t2dm"):
        path = metrics / f"fusion_{coh}.json"
        if "--figures-only" in sys.argv and path.exists():
            results[coh] = json.loads(path.read_text())["metrics"]
            continue
        results[coh] = evaluate_cohort(coh, fc, bc)
        write_metrics(
            f"fusion_{coh}",
            results[coh],
            {"fusion": fc, "baselines": bc},
            42,
            file_hash(sorted(FUS.glob(f"*_{coh}_*.parquet"))),
        )
        print(f"written fusion_{coh}.json", flush=True)
    all_figures(results, fc, load_config("base")["figures"]["dpi"])
    print("figures written", flush=True)


if __name__ == "__main__":
    main()
