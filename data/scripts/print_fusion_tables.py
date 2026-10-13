"""Print compact Part 3 tables from reports/metrics/fusion_*.json (for the report / docs)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

M = Path(__file__).resolve().parents[2] / "reports" / "metrics"
pd.set_option("display.width", 220)


def ci(v: float, c: list[float], d: int = 1) -> str:
    """'x [lo, hi]'."""
    return f"{v:.{d}f} [{c[0]:.{d}f}, {c[1]:.{d}f}]"


def reg(rows: list[dict], subset: str = "overall") -> pd.DataFrame:
    """Model x horizon MAE table with CIs."""
    t = pd.DataFrame(rows)
    t = t[t.subset == subset].copy()
    t["v"] = [ci(a, b) for a, b in zip(t.mae, t.mae_ci, strict=True)]
    return t.pivot_table(index="model", columns="horizon_min", values="v", aggfunc="first")


def paired(rows: list[dict], subsets: tuple[str, ...] = ("overall",)) -> pd.DataFrame:
    """Paired MAE differences with CIs and a distinguishability flag."""
    p = pd.DataFrame(rows)
    p = p[p.subset.isin(subsets)].copy()
    p["diff"] = [
        ci(a, b, 2) + ("" if s else "  ns")
        for a, b, s in zip(p.diff_mae, p.diff_mae_ci, p.distinguishable_from_zero, strict=True)
    ]
    p["pair"] = p.model_1 + " - " + p.model_2
    return p.pivot_table(
        index=["subset", "pair"], columns="horizon_min", values="diff", aggfunc="first"
    )


def main() -> None:
    """Print every table needed for the Part 3 report."""
    for coh in ("cgmacros", "shanghai_t2dm"):
        r = json.loads((M / f"fusion_{coh}.json").read_text())["metrics"]
        print(f"\n######## {coh}: {r['n_patients']} patients, {r['n_rows_per_repeat']} windows")
        print("-- ablation (overall)\n", reg(r["ablation"]).to_string())
        for k in [k for k in r if k.startswith("ablation_without")]:
            print(f"-- {k} (overall)\n", reg(r[k]).to_string())
        print("-- key paired diffs (m1 - m2, <0 = m1 better; ns = CI includes 0)")
        subs = ("overall", "post_meal_3h", "onset_hyper140")
        print(paired(r["key_paired"], subs).to_string())
        for k in [k for k in r if k.startswith("key_paired_without")]:
            print(f"-- {k}\n", paired(r[k]).to_string())
        print("-- LOGO paired (E-minus-group - E; >0 = group helps)")
        print(paired(r["logo_paired"]).to_string())
        print("-- families (CV repeat 0)\n", reg(r["families_r0"]).to_string())
        print(paired(r["families_paired_r0"], subs).to_string())
        sv = pd.DataFrame(r["seed_variance"])
        print(
            "-- seed variance\n",
            sv[["model", "horizon_min", "mean", "sd", "range"]].round(2).to_string(index=False),
        )
        print("-- tiers\n", reg(r["tiers"]).to_string())
        e = pd.DataFrame(r["events"])
        e = e[e.event.str.endswith("_60")]
        e["AUROC"] = [
            ci(a, b, 3) if isinstance(b, list) else "nan"
            for a, b in zip(e.auroc, e.get("auroc_ci"), strict=True)
        ]
        e["AUPRC"] = [
            ci(a, b, 3) if isinstance(b, list) else "nan"
            for a, b in zip(e.auprc, e.get("auprc_ci"), strict=True)
        ]
        print(
            "-- events at 60 min\n",
            e[
                [
                    "event",
                    "subset",
                    "scorer",
                    "n_pos",
                    "n_patients_with_events",
                    "AUROC",
                    "AUPRC",
                    "reliability",
                ]
            ].to_string(index=False),
        )
        g = pd.DataFrame(r["groups_60min"])
        g["v"] = [ci(a, b) for a, b in zip(g.mae, g.mae_ci, strict=True)]
        print(
            "-- groups 60 min\n",
            g.pivot_table(
                index=["grouping", "group", "n_patients"],
                columns="model",
                values="v",
                aggfunc="first",
            ).to_string(),
        )
        print("-- neural selection", r.get("neural_selection"))


if __name__ == "__main__":
    main()
