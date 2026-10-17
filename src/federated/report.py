"""Part 8c report (``python -m src.federated.report``): figure and plain-text table.

Reads reports/metrics/federated.json and writes reports/figures/federated_mae.png (300 dpi,
Okabe-Ito palette): +60 min MAE of every method for each clinic layout with 95% patient-bootstrap
intervals, and the last-value-carried-forward reference as a dashed line. Prints the privacy budget.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils.config import PROJECT_ROOT  # noqa: E402

OI = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#999999", "#000000"]
LABEL = {
    "local_only": "Each clinic alone",
    "central": "Pooled (records shared)",
    "fedavg": "Federated (FedAvg)",
    "fedprox": "Federated (FedProx)",
}
TITLE = {
    "iid": "5 similar clinics",
    "hba1c": "5 clinics split by HbA1c",
    "dataset": "2 clinics = 2 cohorts",
}


def main() -> None:
    """Draw the federated figure and print the results table."""
    m = json.loads((PROJECT_ROOT / "reports" / "metrics" / "federated.json").read_text("utf-8"))[
        "metrics"
    ]
    layouts = [k for k in TITLE if k in m]
    dps = sorted(k for k in m["epsilon"])
    label = dict(LABEL)
    for z in dps:
        label[f"fedavg_dp_{z}"] = (
            f"FedAvg + privacy noise (z={z[1:]}, epsilon {m['epsilon'][z]:.1f})"
        )
    fig, axes = plt.subplots(1, len(layouts), figsize=(5.6 * len(layouts), 4.6))
    axes = [axes] if len(layouts) == 1 else list(axes)
    for ax, lay in zip(axes, layouts, strict=True):
        rows = {r["model"]: r for r in m[lay]["mae"] if r["horizon_min"] == 60}
        names = [n for n in label if n in rows]
        for i, n in enumerate(names):
            r = rows[n]
            ax.barh(i, r["mae"], xerr=[[r["mae"] - r["mae_ci"][0]], [r["mae_ci"][1] - r["mae"]]],
                    color=OI[i % len(OI)], capsize=3)  # fmt: skip
            print(
                f"{lay:8s} {label[n]:55s} MAE +60 {r['mae']:.2f} ({r['mae_ci'][0]:.2f} to {r['mae_ci'][1]:.2f})"
            )
        ax.axvline(
            rows["persistence"]["mae"], color="k", ls="--", lw=1, label="Last value carried forward"
        )
        ax.set_yticks(range(len(names)), [label[n] for n in names], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean absolute error at +60 min (mg/dL)")
        ax.set_title(f"{TITLE[lay]} (SIMULATION)", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = PROJECT_ROOT / "reports" / "figures" / "federated_mae.png"
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
    print("privacy budget (delta {:g}, {} rounds):".format(m["delta"], m["rounds"]), m["epsilon"])


def privacy_figure() -> None:
    """Privacy-utility curve (EXPLORATORY, D-063a): +60 min MAE against the noise multiplier."""
    root = PROJECT_ROOT / "reports" / "metrics"
    main_p, sweep_p = root / "federated.json", root / "federated_privacy.json"
    if not (main_p.exists() and sweep_p.exists()):
        print("privacy sweep not run yet (python -m src.federated.privacy_sweep)")
        return
    a = json.loads(main_p.read_text("utf-8"))["metrics"]
    b = json.loads(sweep_p.read_text("utf-8"))["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=False)
    for ax, lay in zip(axes, TITLE, strict=True):
        pts = {}  # z -> (mae, lo, hi, epsilon)
        for src, eps in ((a, a["epsilon"]), (b, b["epsilon"])):
            for r in src[lay]["mae"]:
                if r["horizon_min"] == 60 and "_dp_z" in r["model"]:
                    z = r["model"].split("_dp_z")[1]
                    pts[float(z)] = (r["mae"], r["mae_ci"][0], r["mae_ci"][1], eps[f"z{z}"])
        zs = sorted(pts)
        ax.errorbar(zs, [pts[z][0] for z in zs],
                    yerr=[[pts[z][0] - pts[z][1] for z in zs], [pts[z][2] - pts[z][0] for z in zs]],
                    color=OI[3], marker="o", capsize=3, label="Federated + privacy noise")  # fmt: skip
        ref = {r["model"]: r for r in b[lay]["mae"] if r["horizon_min"] == 60}
        ax.axhline(ref["fedavg"]["mae"], color=OI[2], lw=1.5, label="Federated, no noise")
        ax.axhline(ref["persistence"]["mae"], color="k", ls="--", lw=1, label="Last value carried forward")
        for z in zs:
            ax.annotate(f"e={pts[z][3]:.0f}" if pts[z][3] < 1e4 else f"e={pts[z][3]:.0e}", (z, pts[z][0]),
                        textcoords="offset points", xytext=(4, 8), fontsize=7)  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlabel("Noise multiplier z (more noise to the right)")
        ax.set_ylabel("MAE at +60 min (mg/dL)")
        ax.set_title(f"{TITLE[lay]} (SIMULATION, EXPLORATORY)", fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = PROJECT_ROOT / "reports" / "figures" / "federated_privacy.png"
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
    privacy_figure()
