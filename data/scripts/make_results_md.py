"""Generate docs/RESULTS.md from reports/metrics/*.json (no hand-copied numbers).

Usage: python data/scripts/make_results_md.py
Sections are appended per part; every number comes from the metrics JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "reports" / "metrics"
OUT = ROOT / "docs" / "RESULTS.md"
COHORTS = {"cgmacros": "CGMacros", "shanghai_t2dm": "Shanghai T2DM"}


def ci(v: float, c: list[float] | None, d: int = 1) -> str:
    """Format 'x [lo, hi]'."""
    if c is None or not isinstance(c, list) or pd.isna(v):
        return f"{v:.{d}f}" if not pd.isna(v) else "n/a"
    return f"{v:.{d}f} [{c[0]:.{d}f}, {c[1]:.{d}f}]"


def md(df: pd.DataFrame) -> str:
    """Render a DataFrame (index included) as a GitHub markdown table."""
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(x) else str(x) for x in row) + " |")
    return "\n".join(lines)


def reg(rows: list[dict], subset: str = "overall", models: list[str] | None = None) -> pd.DataFrame:
    """Model x horizon MAE [95% CI] table."""
    t = pd.DataFrame(rows)
    t = t[t.subset == subset].copy()
    if models is not None:
        t = t[t.model.isin(models)]
    t["v"] = [ci(a, b) for a, b in zip(t.mae, t.mae_ci, strict=True)]
    p = t.pivot_table(index="model", columns="horizon_min", values="v", aggfunc="first")
    p.columns = [f"+{c} min" for c in p.columns]
    order = [m for m in (models or list(dict.fromkeys(t.model))) if m in p.index]
    return p.loc[order]


def paired(rows: list[dict], subsets: tuple[str, ...] = ("overall",)) -> pd.DataFrame:
    """Paired MAE differences (model_1 - model_2) with CI; 'ns' when the CI includes 0."""
    p = pd.DataFrame(rows)
    p = p[p.subset.isin(subsets)].copy()
    p["d"] = [
        ci(a, b, 2) + ("" if s else " ns")
        for a, b, s in zip(p.diff_mae, p.diff_mae_ci, p.distinguishable_from_zero, strict=True)
    ]
    p["pair"] = p.model_1 + " − " + p.model_2
    t = p.pivot_table(index=["subset", "pair"], columns="horizon_min", values="d", aggfunc="first")
    t.columns = [f"+{c} min" for c in t.columns]
    rank = {s: i for i, s in enumerate(subsets)}
    return t.iloc[sorted(range(len(t)), key=lambda i: (rank[t.index[i][0]], t.index[i][1]))]


def events(rows: list[dict]) -> pd.DataFrame:
    """Onset-only event table at 60 min."""
    e = pd.DataFrame(rows)
    e = e[e.event.str.endswith("_60") & (e.subset == "onset_only")].copy()
    e["AUROC"] = [ci(a, b, 3) for a, b in zip(e.auroc, e.auroc_ci, strict=True)]
    e["AUPRC"] = [ci(a, b, 3) for a, b in zip(e.auprc, e.auprc_ci, strict=True)]
    e["reliability"] = e.reliability.astype(str)
    return e.set_index(["event", "scorer"])[
        ["n_pos", "n_patients_with_events", "AUROC", "AUPRC", "reliability"]
    ]


def part3() -> list[str]:
    """Part 3 section (fusion ablation)."""
    s = [
        "## Part 3 — Fusion ablation (tag `part-3-done`)",
        "",
        "Source: `reports/metrics/fusion_{cgmacros,shanghai_t2dm}.json`. MAE in mg/dL with 95%",
        "patient-bootstrap CIs (1,000 resamples over patients). Patient-grouped 5-fold CV × 5 repeats",
        "(neural: repeat 0 × 5 seeds). Paired differences: model_1 − model_2,",
        "negative = model_1",
        "better, `ns` = CI includes 0. Decisions: D-027 to D-030.",
        "",
    ]
    for coh, name in COHORTS.items():
        r = json.loads((M / f"fusion_{coh}.json").read_text())["metrics"]
        subs = ("overall", "post_meal_3h", "onset_hyper140")
        s += [
            f"### {name} ({r['n_patients']} patients, {r['n_rows_per_repeat']} windows)",
            "",
            "**Ablation (overall MAE)**",
            "",
            md(reg(r["ablation"])),
            "",
            "**Key paired comparisons** (overall / post-meal 3 h / hyper-140 onset)",
            "",
            md(paired(r["key_paired"], subs)),
            "",
            "**Leave-one-group-out** (E-minus-group − E; positive = the group helps)",
            "",
            md(paired(r["logo_paired"])),
            "",
            "**Model families, CV repeat 0** (overall MAE)",
            "",
            md(reg(r["families_r0"])),
            "",
            md(paired(r["families_paired_r0"], subs)),
            "",
            "**Input tiers** (fingersticks are simulated from real CGM without meter error, so",
            "these numbers are optimistic)",
            "",
            md(reg(r["tiers"])),
            "",
            "**Events at +60 min, onset windows only** (glucose at t below the threshold).",
            "`limited_evidence` rows are counts only, not results.",
            "",
            md(events(r["events"])),
            "",
        ]
    return s


def main() -> None:
    """Write docs/RESULTS.md."""
    head = [
        "# GlucoTwin — Results",
        "",
        "Generated by `data/scripts/make_results_md.py` from `reports/metrics/*.json`; do not edit",
        "by hand. Research prototype, decision support only, not a medical device.",
        "",
    ]
    OUT.write_text("\n".join(head + part3()) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
