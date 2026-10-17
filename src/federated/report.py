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
TITLE = {"iid": "5 similar clinics", "hba1c": "5 clinics split by HbA1c", "dataset": "2 clinics = 2 cohorts"}


def main() -> None:
    """Draw the federated figure and print the results table."""
    m = json.loads((PROJECT_ROOT / "reports" / "metrics" / "federated.json").read_text("utf-8"))["metrics"]
    layouts = [k for k in TITLE if k in m]
    dps = sorted(k for k in m["epsilon"])
    label = dict(LABEL)
    for z in dps:
        label[f"fedavg_dp_{z}"] = f"FedAvg + privacy noise (z={z[1:]}, epsilon {m['epsilon'][z]:.1f})"
    fig, axes = plt.subplots(1, len(layouts), figsize=(5.6 * len(layouts), 4.6))
    axes = [axes] if len(layouts) == 1 else list(axes)
    for ax, lay in zip(axes, layouts, strict=True):
        rows = {r["model"]: r for r in m[lay]["mae"] if r["horizon_min"] == 60}
        names = [n for n in label if n in rows]
        for i, n in enumerate(names):
            r = rows[n]
            ax.barh(i, r["mae"], xerr=[[r["mae"] - r["mae_ci"][0]], [r["mae_ci"][1] - r["mae"]]],
                    color=OI[i % len(OI)], capsize=3)  # fmt: skip
            print(f"{lay:8s} {label[n]:55s} MAE +60 {r['mae']:.2f} ({r['mae_ci'][0]:.2f} to {r['mae_ci'][1]:.2f})")
        ax.axvline(rows["persistence"]["mae"], color="k", ls="--", lw=1, label="Last value carried forward")
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


if __name__ == "__main__":
    main()
