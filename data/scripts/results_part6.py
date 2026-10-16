"""Part 6 section of docs/RESULTS.md (uncertainty, calibration, alarms, robustness, fusion).

Every number comes from reports/metrics/{uncertainty,patient_fusion}.json.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "reports" / "metrics"
COHORTS = {"cgmacros": "CGMacros", "shanghai_t2dm": "Shanghai T2DM"}
METH = {"scp": "split conformal", "cqr": "CQR", "aci": "ACI", "cvp": "CV+"}


def pct(v: float, c: list[float] | None = None) -> str:
    """'x.x [lo, hi]' in percent."""
    if c is None:
        return f"{100 * v:.1f}"
    return f"{100 * v:.1f} [{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def _row(meth: str, model: str) -> str:
    return f"{METH[meth]} / {'personalized k=3' if model == 'ft' else 'population'}"


def cov_table(rows: list[dict], lvl: int) -> pd.DataFrame:
    """Method/model x horizon coverage [CI] (median width) at one level."""
    t = pd.DataFrame([r for r in rows if r["level"] == lvl])
    t["cell"] = [
        f"{pct(r.coverage, r.coverage_ci)} (w {r.median_width:.0f}"
        + (f"; INFINITE {100 * r.infinite_frac:.0f} %)" if r.infinite_frac > 0 else ")")
        for r in t.itertuples()
    ]
    t["row"] = [_row(m, x) for m, x in zip(t.method, t.model, strict=True)]
    p = t.pivot_table(index="row", columns="horizon_min", values="cell", aggfunc="first")
    p.columns = [f"+{c} min" for c in p.columns]
    return p


def group_table(rows: list[dict], lvl: int, h: int) -> pd.DataFrame:
    """Per-group coverage at one level / horizon (unreliable cells flagged)."""
    t = pd.DataFrame([r for r in rows if r["level"] == lvl and r["horizon_min"] == h])
    t["cell"] = [
        f"{pct(r.coverage, r.coverage_ci)} n={r.n_patients}"
        + ("" if r.reliable else " (unreliable)")
        for r in t.itertuples()
    ]
    t["row"] = [_row(m, x) for m, x in zip(t.method, t.model, strict=True)]
    return t.pivot_table(index="row", columns="group", values="cell", aggfunc="first")


def coverage_part(m: dict, md: Callable) -> list[str]:
    """Coverage tables, failures, per-group tables."""
    s: list[str] = []
    for c, name in COHORTS.items():
        for des in ("A", "B"):
            r = m[des][c]
            s += [f"### {name}, design {des} ({r['n_test_patients']} test patients)", ""]
            for lvl in (80, 95):
                s += [f"{lvl} % target:", "", md(cov_table(r["coverage"]["overall"], lvl)), ""]
            fl = r["coverage_failures_overall"]
            s += ["Failures (95 % CI entirely below target): " + ("; ".join(fl) or "none."), ""]
        for des in ("A", "B"):
            s += [
                f"Per group, design {des}, 80 %, +60 min ({name}):",
                "",
                md(group_table(m[des][c]["coverage"]["group"], 80, 60)),
                "",
            ]
    return s


def events_part(m: dict, md: Callable) -> list[str]:
    """Event-probability calibration table (design A, +60 min)."""
    rows = []
    for c, name in COHORTS.items():
        for ev in ("hyper180", "hyper140", "hypo70"):
            e = m["A"][c]["events"].get(f"{ev}_60", {})
            if "platt" not in e:
                continue
            a = e["auroc"]
            rows.append(
                {
                    "cohort": name,
                    "event": ev + (" (limited evidence)" if e["label"] else ""),
                    "pos / rows": f"{e['n_pos']} / {e['n_rows']}",
                    "patients w/ events": e["n_patients_with_events"],
                    "ECE raw": f"{e['raw']['ece']:.3f}",
                    "ECE Platt": f"{e['platt']['ece']:.3f}",
                    "ECE isotonic": f"{e['isotonic']['ece']:.3f}",
                    "Brier Platt": f"{e['platt']['brier']:.3f}",
                    "AUROC": f"{a['value']:.3f} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]",
                }
            )
    return [
        "### Event probabilities (design A, onset-only windows, +60 min)",
        "",
        md(pd.DataFrame(rows).set_index("cohort")),
        "",
    ]


def alarms_part(m: dict, md: Callable) -> list[str]:
    """Alarm burden with training-chosen thresholds (design A)."""
    rows = []
    for c, name in COHORTS.items():
        for ev, r in m["A"][c]["alarms"].items():
            if "model" not in r:
                continue
            for sysn in ("model", "naive", "trend"):
                for fa in ("1", "2", "3"):
                    t = r[sysn]["train_chosen"][fa]
                    f = r[sysn]["sens_at_fa_curve"][fa]
                    rows.append(
                        {
                            "cohort": name,
                            "event": ev + (" (LE)" if r["label"] else ""),
                            "system": sysn,
                            "FA/day budget": fa,
                            "sens % (train thr.)": f"{100 * t['sensitivity']:.1f}",
                            "realized FA/day": f"{t['fa_per_day']:.2f}",
                            "PPV %": f"{100 * t['ppv']:.1f}",
                            "lead median [IQR] min": f"{t['lead_median_min']:.0f} "
                            f"[{t['lead_q25_min']:.0f}, {t['lead_q75_min']:.0f}]",
                            "sens % at FA (test curve)": f"{100 * f['sensitivity']:.1f}",
                            "episodes": t["episodes_eligible"],
                        }
                    )
    return [
        "### Alarm burden (design A, 60-min alerts, LE = limited evidence)",
        "",
        "Thresholds chosen on training patients' out-of-fold rows for each FA/day budget, then",
        "applied to test; the last column is the test FROC curve point (no tuning; optimistic).",
        "",
        md(pd.DataFrame(rows).set_index("cohort")),
        "",
    ]


def selective_part(m: dict, md: Callable) -> list[str]:
    """Risk-coverage: AURC per abstention score (lower = better), both designs."""
    rows = []
    for des in ("A", "B"):
        for c, name in COHORTS.items():
            for h, s in m[des][c]["selective"].items():
                if not isinstance(s, dict):
                    continue  # e.g. the shared coverage grid
                row = {"design / cohort": f"{des} / {name}", "horizon": f"+{h} min"}
                row |= {k: f"{v['aurc']:.2f}" for k, v in s.items() if isinstance(v, dict)}
                rows.append(row)
    return [
        "### Selective prediction (area under the MAE-vs-coverage curve, mg/dL; lower = better)",
        "",
        "Abstention ranks rows by CQR interval width, a data-quality score, both, or at random;",
        "the oracle ranks by the true error (a lower bound, not achievable).",
        "",
        md(pd.DataFrame(rows).set_index("design / cohort")),
        "",
    ]


def robustness_part(m: dict, md: Callable) -> list[str]:
    """Fault x severity table and DQ-vs-error correlation (design A, +60 min)."""
    s = ["### Robustness (design A, +60 min, population model, split-arm intervals)", ""]
    for c, name in COHORTS.items():
        rows = []
        for x in m["A"][c]["robustness"]["cells"]:
            ci = x["mae_ci_60"]
            rows.append(
                {
                    "fault": x["fault"],
                    "level": x["level"],
                    "available %": f"{100 * x['availability']:.1f}",
                    "MAE": f"{x['mae_60']:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]",
                    "cov 80 % split": pct(x["cov_scp_pop_80_60"]),
                    "cov 80 % CQR": pct(x["cov_cqr_pop_80_60"]),
                    "cov 95 % split": pct(x["cov_scp_pop_95_60"]),
                    "DQ mean": f"{x['dq_mean']:.2f}",
                }
            )
        dq = m["A"][c]["robustness"]["dq"]["60"]
        b = dq["spearman_ci_200boot"]
        s += [
            f"{name}:",
            "",
            md(pd.DataFrame(rows).set_index("fault")),
            "",
            f"DQ vs absolute error, Spearman {dq['spearman']:.3f} [{b[0]:.3f}, {b[1]:.3f}] "
            f"(pooled clean + faulted rows); clean rows only {dq['clean_only_spearman']:.3f}.",
            "",
        ]
    return s


def fusion_part(md: Callable) -> list[str]:
    """Item 2 table."""
    pf = M / "patient_fusion.json"
    if not pf.exists():
        return []
    p = json.loads(pf.read_text(encoding="utf-8"))["metrics"]
    rows = []
    for c, name in COHORTS.items():
        for tgt, lab in (("y_tar180", "next-day TAR>180 %"), ("y_tir", "next-day TIR %")):
            r = p[c][tgt]
            row = {"cohort": f"{name} (n={p[c]['n_patients']})", "target": lab}
            row |= {k: f"{v:.2f}" for k, v in r["mae"].items()}
            for k in ("both-static", "both-dynamic"):
                d = r[k]
                row[k] = f"{d['diff']:.2f} [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]"
            rows.append(row)
    return [
        "### Item 2: patient-level static + dynamic fusion (MAE in percentage points of the day)",
        "",
        md(pd.DataFrame(rows).set_index("cohort")),
        "",
    ]


def part6(md: Callable) -> list[str]:
    """Markdown lines of the Part 6 section."""
    f = M / "uncertainty.json"
    if not f.exists():
        return []
    m = json.loads(f.read_text(encoding="utf-8"))["metrics"]
    head = [
        "## Part 6 - Uncertainty, calibration, alarms, robustness",
        "",
        "Design A (headline) = CV repeat 0, 5 folds, every eligible patient tested once. Design B",
        "(confirmatory, never used for choices) = holdout train / cal / test. Coverage CIs are",
        "95 % patient-bootstrap. Rows within a patient are autocorrelated, so the conformal",
        "guarantees are nominal (D-045). Cells with < 5 patients are 'unreliable'. w = median",
        "interval width (mg/dL) over finite intervals; INFINITE = share of rows whose interval is",
        "infinite (ACI only; counted as covered, so ACI coverage is not informative, D-049).",
        "",
    ]
    return (
        head
        + coverage_part(m, md)
        + events_part(m, md)
        + alarms_part(m, md)
        + selective_part(m, md)
        + robustness_part(m, md)
        + fusion_part(md)
    )
