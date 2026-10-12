"""Print compact Part 2 tables from reports/metrics/baselines_*.json (for the report / docs)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

M = Path(__file__).resolve().parents[2] / "reports" / "metrics"


def fmt(v: float, ci: list[float], d: int = 1) -> str:
    """'x [lo, hi]'."""
    return f"{v:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def main() -> None:
    """Print regression, event, group and holdout tables."""
    for coh in ("cgmacros", "shanghai_t2dm"):
        r = json.loads((M / f"baselines_{coh}.json").read_text())["metrics"]
        sel = r["window_selection"]["mean_mae_by_N"]
        print(f"\n=== {coh} (N={r['N_hours']} h) window selection: {sel}")
        keys = [k for k in r if k.startswith("cv_regression")] + ["holdout_test_regression"]
        for key in keys:
            t = pd.DataFrame(r[key])
            t["MAE"] = [fmt(a, b) for a, b in zip(t.mae, t.mae_ci, strict=True)]
            t["RMSE"] = t.rmse.round(1)
            t["MARD%"] = t.mard_pct.round(1)
            t["ClarkeA%"] = t.clarke_A_pct.round(1)
            t["A+B%"] = t.clarke_AB_pct.round(1)
            w = t.pivot_table(index="model", columns="horizon_min", values="MAE", aggfunc="first")
            print(
                f"-- {key}: MAE mg/dL [95% patient-bootstrap CI]; n_patients={t.n_patients.iloc[0]}"
            )
            print(w.to_string())
            if key == "cv_regression":
                x = t[t.model.isin(["persistence", "ar_per_patient", "lgbm_cgm_tod"])]
                print(
                    x[["model", "horizon_min", "RMSE", "MARD%", "ClarkeA%", "A+B%"]].to_string(
                        index=False
                    )
                )
        for key in [k for k in r if k.startswith("cv_events")]:
            e = pd.DataFrame(r[key])
            e = e[e.scorer == "lgbm_classifier"]
            e["AUROC"] = [
                fmt(a, b, 3) if b else "nan"
                for a, b in zip(e.auroc, e.get("auroc_ci"), strict=True)
            ]
            e["AUPRC"] = [
                fmt(a, b, 3) if b else "nan"
                for a, b in zip(e.auprc, e.get("auprc_ci"), strict=True)
            ]
            e["Brier"] = e.brier.round(3)
            print(f"-- {key} (LightGBM classifier)")
            print(
                e[
                    [
                        "event",
                        "definition",
                        "horizon_min",
                        "n_pos",
                        "n_patients_with_events",
                        "AUROC",
                        "AUPRC",
                        "Brier",
                        "reliability",
                    ]
                ].to_string(index=False)
            )
        g = pd.DataFrame(r["cv_groups_60min"])
        g["MAE"] = [fmt(a, b) for a, b in zip(g.mae, g.mae_ci, strict=True)]
        print("-- MAE at 60 min by group")
        print(
            g.pivot_table(
                index=["grouping", "group", "n_patients"],
                columns="model",
                values="MAE",
                aggfunc="first",
            ).to_string()
        )
    c = json.loads((M / "baselines_checks.json").read_text())["metrics"]["shuffle_check"]
    print("\nshuffle check:", json.dumps(c))


if __name__ == "__main__":
    main()
