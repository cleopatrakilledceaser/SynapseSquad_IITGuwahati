"""Onset-only event metrics: are high AUROCs just 'already above threshold now'?

Compares classifier vs persistence on all windows and on ONSET windows only (glucose at t not yet
beyond the threshold).

Writes reports/metrics/baselines_event_onset.json (repeat 0 of the CV, patient-bootstrap CIs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import bootstrap_event, reliability  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.provenance import file_hash, write_metrics  # noqa: E402

PRED = ROOT / "data" / "processed" / "predictions"


def main() -> None:
    """Compute onset-only AUROC/AUPRC for hyper180/hyper140 'any' events at 30/60/120 min."""
    bcfg, fcfg = load_config("baselines"), load_config("features")
    b, e = bcfg["bootstrap"], bcfg["events"]
    rows = []
    for coh in bcfg["cohorts"]:
        d = pd.read_parquet(PRED / f"{coh}_cv_oof.parquet")
        d = d[d["repeat"] == 0]
        for name in ("hyper180", "hyper140"):
            thr = fcfg["events"][name]["threshold"]
            for h in fcfg["windows"]["horizons_min"]:
                lab = f"{name}_any_{h}"
                for subset, dd in (("all", d), ("onset_only", d[d.cgm_now <= thr])):
                    for scorer, col in (
                        ("persistence_value", f"score_persistence_{lab}"),
                        ("lgbm_classifier", f"prob_lgbm_cgm_tod_{lab}"),
                    ):
                        r = bootstrap_event(dd, lab, col, b["n_boot"], 42, b["ci"], b["auc_bins"])
                        rows.append(
                            {
                                "cohort": coh,
                                "event": lab,
                                "subset": subset,
                                "scorer": scorer,
                                "reliability": reliability(
                                    r, e["min_pos_windows"], e["min_pos_patients"]
                                ),
                                **{
                                    k: r.get(k)
                                    for k in (
                                        "n_rows",
                                        "n_pos",
                                        "auroc",
                                        "auroc_ci",
                                        "auprc",
                                        "auprc_ci",
                                        "prevalence",
                                    )
                                },
                            }
                        )
    t = pd.DataFrame(rows)
    show = t.assign(AUROC=t.auroc.round(3), AUPRC=t.auprc.round(3), prev=t.prevalence.round(3))
    print(
        show.pivot_table(
            index=["cohort", "event", "subset"],
            columns="scorer",
            values=["AUROC", "AUPRC"],
            aggfunc="first",
        ).to_string()
    )
    write_metrics(
        "baselines_event_onset",
        {"rows": rows},
        {"baselines": bcfg},
        42,
        file_hash(sorted(PRED.glob("*_cv_oof.parquet"))),
    )


if __name__ == "__main__":
    main()
