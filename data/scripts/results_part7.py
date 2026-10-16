"""Part 7 section of docs/RESULTS.md (what-if engine).

Every number comes from reports/metrics/whatif_{plausibility,matched}.json.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "reports" / "metrics"
MODELS = ["unc", "mono", "twin_learned", "twin_assumed"]


def _load(name: str) -> dict:
    return json.loads((M / f"{name}.json").read_text(encoding="utf-8"))["metrics"]


def _pc(r: dict) -> str:
    s = f"{100 * r['pass_rate']:.1f} [{100 * r['ci'][0]:.1f}, {100 * r['ci'][1]:.1f}]"
    if "construction" in r["note"]:
        s += " (by construction)"
    elif "ASSUMED" in r["note"]:
        s += " (ASSUMED)"
    elif "learned" in r["note"]:
        s += " (learned)"
    return s


def plaus_table(b: dict) -> pd.DataFrame:
    """Test x model pass rate [95 % patient-bootstrap CI]."""
    t = pd.DataFrame(b["pass_rates"])
    t["cell"] = [_pc(r) for r in b["pass_rates"]]
    p = t.pivot_table(index="test", columns="model", values="cell", aggfunc="first")
    return p[[m for m in MODELS if m in p.columns]].fillna("n/a")


def _v(d: dict, k: str) -> str:
    return f"{d[k]['value']:.2f} [{d[k]['ci'][0]:.2f}, {d[k]['ci'][1]:.2f}]"


def matched_table(b: dict, target: str) -> pd.DataFrame:
    """Design x model matched-meal metrics at one target (+60 or peak)."""
    rows = []
    for key, d in b.items():
        design, model, tgt = key.split("|")
        if tgt != target:
            continue
        rows.append(
            {
                "design / model": f"{design} / {model}",
                "pairs (clusters)": f"{d['n_pairs']} ({d['n_clusters']})",
                "Pearson r": _v(d, "pearson"),
                "Spearman": _v(d, "spearman"),
                "sign agreement": _v(d, "sign_agreement"),
                "slope": _v(d, "slope"),
                "mean predicted": _v(d, "mean_pred"),
                "mean observed": _v(d, "mean_obs"),
            }
        )
    return pd.DataFrame(rows).set_index("design / model")


def part7(md: Callable) -> list[str]:
    """Markdown lines of the Part 7 section."""
    if not (M / "whatif_matched.json").exists():
        return []
    p, m = _load("whatif_plausibility"), _load("whatif_matched")
    s = [
        "## Part 7 - Counterfactual what-if engine (CGMacros)",
        "",
        "Model-based what-ifs, not causal guarantees. Bands are observational (no counterfactual",
        "coverage guarantee). unc / mono = population A0 LightGBM without / with monotone",
        "constraints (carbs, kcal +1; post-meal activity -1); twin_learned = Part 4 twin with the",
        "Part 4 activity effect; twin_assumed = twin with the ASSUMED walk effect of D-052 (20 %",
        "3-h iAUC reduction), not learned from data. Shanghai is excluded (no meal macros).",
        "CIs are 95 % patient-bootstrap. Decisions D-050 to D-055.",
        "",
    ]
    for key, title in (
        ("final_test_patients", "Plausibility suite, test patients (final run, once)"),
        ("dev_fold0_training_patients", "Plausibility suite, fold-0 training patients (dev)"),
    ):
        if key not in p:
            continue
        b = p[key]
        s += [f"### {title}: pass rate % (n patients = {b['n_patients']})", ""]
        s += [md(plaus_table(b)), ""]
        dm = "; ".join(
            f"{r['model']} {r['median_mgdl_per_g']:.2f} ({'pass' if r['pass'] else 'FAIL'})"
            for r in b["dose_median"]
        )
        s += [
            f"Median dose response, 30 -> 60 g peak (mg/dL per g; D-051: 0.3-1.3): {dm}.",
            "",
        ]
        sp = ", ".join(f"{k} {v:.2f}" for k, v in b["cross_model_spearman"].items())
        s += [f"Spearman of ML vs twin carb effects: {sp}.", ""]
        de = [
            f"{r['test']} / {r['model']}: {r['median']:.2f} [{r['iqr'][0]:.2f}, {r['iqr'][1]:.2f}]"
            for r in b["descriptive"]
        ]
        s += ["Descriptive effects (median [IQR]; peak mg/dL, walk AUC mg/dL x min):", ""]
        s += [f"- {x}" for x in de] + [""]
    for key, title in (
        ("final_all_patients", "Matched breakfasts, every patient scored once (final)"),
        ("dev_fold0_training_patients", "Matched breakfasts, fold-0 training patients (dev)"),
    ):
        if key not in m:
            continue
        for tgt in ("60", "peak"):
            lab = "+60 min (primary)" if tgt == "60" else "peak of +30/+60/+120 (secondary)"
            s += [
                f"### {title}: carb effect at {lab}, mg/dL",
                "",
                md(matched_table(m[key], tgt)),
                "",
            ]
    if "test_mae_mono_vs_unc" in m:
        rows = []
        for h, d in m["test_mae_mono_vs_unc"].items():
            dd = d["mono_minus_unc"]
            rows.append(
                {
                    "horizon": f"+{h} min",
                    "MAE unc": f"{d['mae_unc']:.2f}",
                    "MAE mono": f"{d['mae_mono']:.2f}",
                    "mono - unc [CI]": f"{dd['diff']:.2f} [{dd['ci'][0]:.2f}, {dd['ci'][1]:.2f}]",
                    "patients": d["n_patients"],
                }
            )
        s += ["### Cost of the monotone constraint (test MAE, day >= 7)", ""]
        s += [md(pd.DataFrame(rows).set_index("horizon")), ""]
    if "meta" in p:
        lk = p["meta"]["lock"]
        s += [f"Final stage runs: {lk['runs']}, forced re-runs: {lk['forced_reruns']}", ""]
        s += [f"({'; '.join(lk.get('forced_rerun_log', [])) or 'none'}).", ""]
    return s + part7b(md)


def part7b(md: Callable) -> list[str]:
    """D-056 post-hoc block (one-sided twin activity term); original Part 7 tables untouched."""
    p = _load("whatif_plausibility").get("d056_posthoc")
    if not p:
        return []
    s = [
        "### D-056 (POST-HOC, not pre-registered): one-sided twin activity term",
        "",
        "Activity may only add glucose uptake above basal. Same windows as Part 7. The test-window",
        "run is forced re-run 1 of the locked final stage (twin walk tests only).",
        "",
    ]
    rows = []
    for st, lab in (("dev", "fold-0 training"), ("final", "test")):
        for m, r in p.get(st, {}).items():
            rows.append(
                {
                    "windows / model": f"{lab} / {m}",
                    "walk direction pass % [CI]": f"{100 * r['value']:.1f} "
                    f"[{100 * r['ci'][0]:.1f}, {100 * r['ci'][1]:.1f}] (post-hoc)",
                    "checks": r["n_checks"],
                    "max peak rise": f"{r['max_peak_rise']:.3f}",
                    "walk AUC median": f"{r['walk_auc_median']:.1f}",
                }
            )
    s += [md(pd.DataFrame(rows).set_index("windows / model")), ""]
    p4 = p.get("part4_rerun")
    if p4:
        rows = []
        for mod, v in p4["variants"].items():
            d = v["60"]
            rows.append(
                {
                    "Part 4 variant": mod,
                    "max abs diff +60": f"{d['max_abs_diff']:.3f}",
                    "MAE +60 original": f"{d['mae_original']:.2f}",
                    "MAE +60 D-056": f"{d['mae_d056']:.2f}",
                }
            )
        al = ", ".join(
            f"fold {f}: {a['original']:g} -> {a['d056']:g}" for f, a in p4["alpha_on"].items()
        )
        s += ["Part 4 twin re-run with the D-056 kernel (separate files; originals unchanged)."]
        s += [
            f"Re-selected alpha_E (acton): {al}.",
            "",
            md(pd.DataFrame(rows).set_index("Part 4 variant")),
            "",
        ]
    return s
