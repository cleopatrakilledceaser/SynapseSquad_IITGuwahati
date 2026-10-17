"""Part 8a report (``python -m src.transfer.report``): figure and plain-text summary.

Reads reports/metrics/transfer.json and writes reports/figures/transfer_matrix.png (300 dpi,
Okabe-Ito palette): +60 min MAE of each method for each direction, with 95% patient-bootstrap
intervals, and the persistence reference as a dashed line.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils.config import PROJECT_ROOT  # noqa: E402

OI = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#999999", "#000000"]
LABEL = {
    "in_cohort": "Trained on target patients (CV)",
    "zeroshot": "Zero-shot from other cohort",
    "zeroshot_offset1": "Zero-shot + offset, 1 day",
    "zeroshot_offset3": "Zero-shot + offset, 3 days",
    "zeroshot_ridge1": "Zero-shot + ridge, 1 day",
    "zeroshot_ridge3": "Zero-shot + ridge, 3 days",
    "pooled": "Pooled (other cohort + target train)",
}
NAME = {"cgmacros": "CGMacros", "shanghai_t2dm": "Shanghai T2DM"}


def main() -> None:
    """Draw the transfer figure for the CGM + time-of-day feature set (+60 min)."""
    path = PROJECT_ROOT / "reports" / "metrics" / "transfer.json"
    m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    keys = [k for k in m if "|cgm_tod" in k and not k.endswith("static")]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.4 * len(keys), 4.6), sharex=False)
    axes = [axes] if len(keys) == 1 else list(axes)
    for ax, key in zip(axes, keys, strict=True):
        src, tgt = key.split("|")[0].split("_to_")
        rows = {r["model"]: r for r in m[key]["all"]["mae"] if r["horizon_min"] == 60}
        names = [n for n in LABEL if n in rows]
        for i, n in enumerate(names):
            r = rows[n]
            ax.barh(i, r["mae"], xerr=[[r["mae"] - r["mae_ci"][0]], [r["mae_ci"][1] - r["mae"]]],
                    color=OI[i % len(OI)], capsize=3)  # fmt: skip
        ax.axvline(rows["persistence"]["mae"], color="k", ls="--", lw=1, label="Last value carried forward")
        ax.set_yticks(range(len(names)), [LABEL[n] for n in names], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean absolute error at +60 min (mg/dL)")
        ax.set_title(f"{NAME[src]} to {NAME[tgt]} (n = {m[key]['n_target_patients']} patients)", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = PROJECT_ROOT / "reports" / "figures" / "transfer_matrix.png"
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
