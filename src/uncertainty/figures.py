"""Part 6 figures (300 dpi, labeled axes with units, Okabe-Ito colorblind-safe palette)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.uncertainty.common import COHORTS, HORIZONS  # noqa: E402
from src.utils.config import PROJECT_ROOT  # noqa: E402

FIG = PROJECT_ROOT / "reports" / "figures"
OK = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
METH = {"scp": "split conformal", "cqr": "CQR", "aci": "ACI", "cvp": "CV+"}


def _save(fig: plt.Figure, name: str, dpi: int) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=dpi)
    plt.close(fig)


def coverage(m: dict[str, Any], dpi: int) -> None:
    """Coverage (95 % patient-bootstrap CI) per method x model, design A vs B, 80 / 95 %."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey="row")
    for i, lvl in enumerate((80, 95)):
        for j, c in enumerate(COHORTS):
            ax, k, labels = axes[i, j], 0, []
            for meth in METH:
                for model in ("pop", "ft"):
                    for d_i, des in enumerate(("A", "B")):
                        for r in m[des][c]["coverage"]["overall"]:
                            if (r["method"], r["model"], r["level"]) != (meth, model, lvl):
                                continue
                            hi_ = HORIZONS.index(r["horizon_min"])
                            x = k + 0.25 * (hi_ - 1) + 0.08 * d_i
                            y, lo, hi = (100 * v for v in (r["coverage"], *r["coverage_ci"]))
                            ax.errorbar(
                                x,
                                y,
                                [[y - lo], [hi - y]],
                                fmt="o" if des == "A" else "s",
                                color=OK[hi_],
                                ms=3,
                                alpha=1 if des == "A" else 0.5,
                            )
                    name = METH[meth].replace(" ", "\n")
                    labels.append(f"{name}\n{'pers.' if model == 'ft' else 'pop.'}")
                    k += 1
            ax.axhline(lvl, color="black", lw=1, ls="--")
            ax.set_xticks(range(k), labels, fontsize=7)
            ax.set_ylabel("empirical coverage (%)")
            ax.set_title(f"{COHORTS[c]}, {lvl}% target (circles A, squares B)", fontsize=9)
    for h_i, h in enumerate(HORIZONS):
        axes[0, 0].plot([], [], "o", color=OK[h_i], label=f"+{h} min")
    axes[0, 0].legend(fontsize=7)
    _save(fig, "unc_coverage", dpi)


def reliability(m: dict[str, Any], dpi: int) -> None:
    """Reliability curves (design A, +60 min) per event and cohort."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i, c in enumerate(COHORTS):
        for j, ev in enumerate(("hyper180", "hyper140", "hypo70")):
            ax = axes[i, j]
            r = m["A"][c]["events"].get(f"{ev}_60", {})
            ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--")
            for k, cal in enumerate(("raw", "platt", "isotonic")):
                if cal in r:
                    rc = r[cal]["reliability"]
                    ax.plot(
                        rc["pred"],
                        rc["obs"],
                        "o-",
                        color=OK[k],
                        ms=3,
                        label=f"{cal} (ECE {r[cal]['ece']:.3f})",
                    )
            lab = " (limited evidence)" if r.get("label") else ""
            ax.set_title(f"{COHORTS[c]} {ev} onset, +60 min{lab}", fontsize=9)
            ax.set_xlabel("predicted probability")
            ax.set_ylabel("observed frequency")
            if "raw" in r:
                ax.legend(fontsize=7)
    _save(fig, "unc_reliability", dpi)


def risk_coverage(m: dict[str, Any], dpi: int) -> None:
    """Risk-coverage (MAE of retained windows, mg/dL) at +60 min, design A."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, c in zip(axes, COHORTS, strict=True):
        s = m["A"][c]["selective"]
        g = np.asarray(s["coverage_grid"]) * 100
        for k, (sig, v) in enumerate(x for x in s["60"].items() if isinstance(x[1], dict)):
            ax.plot(g, v["mae"], "o-", ms=3, color=OK[k], label=f"{sig} (AURC {v['aurc']:.1f})")
        ax.set_xlabel("windows retained (%)")
        ax.set_ylabel("MAE of retained windows (mg/dL)")
        ax.set_title(f"{COHORTS[c]}, +60 min")
        ax.invert_xaxis()
        ax.legend(fontsize=7)
    _save(fig, "unc_risk_coverage", dpi)


def alarms(m: dict[str, Any], dpi: int) -> None:
    """Sensitivity vs false alarms per day (test FROC curves), design A."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i, c in enumerate(COHORTS):
        for j, ev in enumerate(("hyper180", "hyper140", "hypo70")):
            ax = axes[i, j]
            r = m["A"][c]["alarms"].get(ev, {})
            for k, s in enumerate(("model", "naive", "trend")):
                if s in r:
                    f = r[s]["froc"]
                    ax.plot(
                        [p["fa_per_day"] for p in f],
                        [100 * p["sensitivity"] for p in f],
                        "-",
                        color=OK[k],
                        label=s,
                    )
            ax.set_xlim(0, 6)
            ax.set_xlabel("false alarms per day")
            ax.set_ylabel("episodes alerted within 60 min (%)")
            lab = " (limited evidence)" if r.get("label") else ""
            ax.set_title(f"{COHORTS[c]} {ev}{lab}", fontsize=9)
            if "model" in r:
                ax.legend(fontsize=7)
    _save(fig, "unc_alarm_froc", dpi)


def dca(m: dict[str, Any], dpi: int) -> None:
    """Decision curves (net benefit) for 60-min hyper-180 alerts, design A."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, c in zip(axes, COHORTS, strict=True):
        r = m["A"][c]["alarms"].get("hyper180", {}).get("dca")
        if not r:
            continue
        pt = np.asarray(r["pt"]) * 100
        ax.plot(pt, r["treat_all"], color="#999999", label="alert always")
        ax.axhline(0, color="black", lw=1, label="never alert")
        for k, s in enumerate(("model", "naive", "trend")):
            ax.plot(pt, r[s], color=OK[k], label=s)
        ax.set_ylim(-0.01, max(0.02, 1.2 * r["prevalence"]))
        ax.set_xlabel("threshold probability (%)")
        ax.set_ylabel("net benefit (per at-risk window)")
        ax.set_title(f"{COHORTS[c]}: hyper-180 within 60 min")
        ax.legend(fontsize=7)
    _save(fig, "unc_decision_curve", dpi)


def robustness(m: dict[str, Any], dpi: int) -> None:
    """MAE (+60 min) and 80 % split-conformal coverage by fault severity, design A."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for j, c in enumerate(COHORTS):
        cells = m["A"][c]["robustness"]["cells"]
        clean = next(x for x in cells if x["fault"] == "clean")
        for k, f in enumerate(sorted({x["fault"] for x in cells} - {"clean"})):
            xs = sorted((x for x in cells if x["fault"] == f), key=lambda x: x["level"])
            sev = range(len(xs) + 1)
            col = OK[k % len(OK)]
            axes[0, j].plot(
                sev, [clean["mae_60"]] + [x["mae_60"] for x in xs], "o-", ms=3, color=col, label=f
            )
            cov = [clean["cov_scp_pop_80_60"]] + [x["cov_scp_pop_80_60"] for x in xs]
            axes[1, j].plot(sev, [100 * v for v in cov], "o-", ms=3, color=col, label=f)
        axes[1, j].axhline(80, color="black", ls="--", lw=1)
        axes[0, j].set_ylabel("MAE at +60 min (mg/dL)")
        axes[1, j].set_ylabel("80% interval coverage (%)")
        for i in range(2):
            axes[i, j].set_xlabel("severity level (0 = clean; values in configs/uncertainty.yaml)")
            axes[i, j].set_title(COHORTS[c])
        axes[0, j].legend(fontsize=7)
    _save(fig, "unc_robustness", dpi)


def all_figures(m: dict[str, Any], dpi: int) -> None:
    """Every Part 6 uncertainty figure."""
    for f in (coverage, reliability, risk_coverage, alarms, dca, robustness):
        f(m, dpi)


def fusion(pf: dict[str, Any], dpi: int) -> None:
    """Item 2: next-day TAR180 / TIR MAE (percentage points) by input set."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sets = ["cohort_mean_reference", "static", "dynamic", "both"]
    for ax, tgt in zip(axes, ("y_tar180", "y_tir"), strict=True):
        for i, c in enumerate(COHORTS):
            v = [pf[c][tgt]["mae"][s] for s in sets]
            ax.bar(
                np.arange(len(sets)) + 0.38 * i,
                v,
                0.38,
                color=OK[i],
                label=f"{COHORTS[c]} (n = {pf[c]['n_patients']})",
            )
        ax.set_xticks(np.arange(len(sets)) + 0.19, ["cohort mean", "static", "day-1 CGM", "both"])
        ax.set_ylabel("MAE (percentage points of the day)")
        ax.set_title(
            "next-day " + ("time > 180 mg/dL" if tgt == "y_tar180" else "time 70-180 mg/dL")
        )
        ax.legend(fontsize=7)
    _save(fig, "patient_fusion", dpi)
