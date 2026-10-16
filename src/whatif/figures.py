"""Part 7 figures (300 dpi, Okabe-Ito palette). Assumed walk effects get separate panels."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.uncertainty.figures import OK, _save  # noqa: E402

ORDER = ["unc", "mono", "twin_learned", "twin_assumed"]
TESTS = ["carb_monotonicity", "carb_monotonicity_2mgdl", "activity_direction", "identity",
         "causality", "bounds", "dose_window", "cross_model"]  # fmt: skip


def examples(ex: dict[str, Any], dpi: int) -> None:
    """Training-patient window: 30 vs 60 g, walk; ML + learned twin | ASSUMED twin | India."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    hz = np.array(ex["horizons_min"])
    sc = ex["scenarios"]
    for i, (key, o) in enumerate(sc.items()):
        ls = "--" if "walk" in key and "no walk" not in key else "-"
        col = OK[i // 2]
        # a walk after t is unknown to the ML models (identical forecasts): plot ML once
        for m, mk in (("unc", "o"), ("mono", "s")) if ls == "-" else ():
            ax[0].plot([0, *hz], [o["cgm_now"], *o[m]["mean"]], ls, marker=mk, color=col, ms=4,
                       label=f"{m}: {key}")  # fmt: skip
        b = np.array(o["mono"]["band80"])
        ax[0].fill_between(hz, b[:, 0], b[:, 1], color=col, alpha=0.08)
        if "twin_learned" in o:
            tr = np.array(o["twin_learned"]["traj"])
            ax[0].plot(np.arange(len(tr)) * 5, tr, ls, color=col, lw=1, alpha=0.6)
        if "twin_assumed" in o:
            tr = np.array(o["twin_assumed"]["traj"])
            ax[1].plot(np.arange(len(tr)) * 5, tr, ls, color=col, lw=1.5, label=key)
    ax[0].set_title(
        "Learned models (lines = twin, markers = ML)\nwalk unknown to ML at t", fontsize=9
    )
    ax[1].set_title(
        "Twin with ASSUMED walk effect (D-052)\nassumed, NOT learned from data", fontsize=9
    )
    for k, o in enumerate(ex["india"]["scenarios"].values()):
        ax[2].plot([0, *hz], [o["cgm_now"], *o["mono"]["mean"]], "-o", ms=3, color=OK[k % 8],
                   label=o["preset"]["label"])  # fmt: skip
    ax[2].set_title("India presets, mono ML (OUT OF DISTRIBUTION)", fontsize=9)
    for a in ax:
        a.set_xlabel("minutes after forecast time t")
        a.set_ylabel("glucose (mg/dL)")
        a.legend(fontsize=6)
    fig.suptitle("Model-based / SIMULATED what-ifs, fold-0 training patient; bands 80 % "
                 "observational (no counterfactual guarantee)", fontsize=10)  # fmt: skip
    _save(fig, "whatif_examples", dpi)


def plausibility(block: dict[str, Any], name: str, title: str, dpi: int) -> None:
    """Pass rate (95 % patient-bootstrap CI) per test x model; hatched = by construction."""
    rows = {(r["test"], r["model"]): r for r in block["pass_rates"]}
    fig, ax = plt.subplots(figsize=(12, 4.5))
    w = 0.2
    for j, m in enumerate(ORDER):
        for i, t in enumerate(TESTS):
            r = rows.get((t, m))
            if r is None:
                continue
            x, y = i + (j - 1.5) * w, 100 * r["pass_rate"]
            lo, hi = (100 * v for v in r["ci"])
            hatch = "//" if "construction" in r["note"] else None
            ax.bar(x, y, w, color=OK[j], hatch=hatch, edgecolor="white" if not hatch else "k")
            ax.errorbar(x, y, [[y - lo], [hi - y]], color="k", lw=0.8)
    from matplotlib.patches import Patch

    handles = [
        Patch(color=OK[j], label=m + (" (ASSUMED)" if m == "twin_assumed" else ""))
        for j, m in enumerate(ORDER)
    ]
    ax.set_xticks(range(len(TESTS)), [t.replace("_", "\n") for t in TESTS], fontsize=7)
    ax.set_ylabel("pass rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title + " (hatched = true by construction)", fontsize=9, pad=22)
    ax.legend(handles=handles, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False)  # fmt: skip
    _save(fig, name, dpi)


def matched(rows: list[dict], name: str, title: str, dpi: int) -> None:
    """Predicted vs observed +60-min carb effect, within-patient pairs, y = x and OLS line."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for k, m in enumerate(("unc", "mono", "twin_learned")):
        d = [r for r in rows if r["model"] == m]
        if not d:
            continue
        p = np.array([r["predicted"] for r in d])
        o = np.array([r["observed"] for r in d])
        ok = ~np.isnan(p) & ~np.isnan(o)
        ax[k].scatter(p[ok], o[ok], s=12, color=OK[k], alpha=0.7)
        lim = np.nanmax(np.abs(np.r_[p[ok], o[ok], 1.0]))
        ax[k].plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
        if ok.sum() >= 3:
            s, c = np.polyfit(p[ok], o[ok], 1)
            ax[k].plot([-lim, lim], [c - s * lim, c + s * lim], color=OK[k], label=f"slope {s:.2f}")
        ax[k].set_title(f"{m} (n = {int(ok.sum())} pairs)", fontsize=9)
        ax[k].set_xlabel("predicted effect, +60 min (mg/dL)")
        ax[k].legend(fontsize=7)
    ax[0].set_ylabel("observed effect, +60 min (mg/dL)")
    fig.suptitle(title, fontsize=10)
    _save(fig, name, dpi)


def all_figures(plaus: dict, mat: dict, ex: dict, wc: dict) -> None:
    """Every Part 7 figure from the metrics dicts + raw matched rows."""
    import pandas as pd

    from src.whatif.run import OUT

    dpi = wc["dpi"]
    examples(ex, dpi)
    stage = "final_test_patients" if "final_test_patients" in plaus else None
    key = stage or "dev_fold0_training_patients"
    plausibility(plaus[key], "whatif_plausibility", f"Plausibility suite: {key}", dpi)
    f = OUT / ("final_matched.parquet" if stage else "dev_matched.parquet")
    d = pd.read_parquet(f)
    d = d[(d.design == "within") & (d.target == "60")]
    lab = "all patients, final" if stage else "fold-0 training patients, dev"
    matched(d.to_dict("records"), "whatif_matched_meals", f"Matched breakfasts ({lab})", dpi)
