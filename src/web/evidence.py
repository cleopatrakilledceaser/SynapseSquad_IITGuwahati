"""Cohort-level evidence for the web console: ``web/evidence.json`` and the figure copies.

The tables come from ``app/demo/stats.json.gz`` through the same helpers as the Streamlit
reference pages (``src.dash.ref``), so the two apps show the same numbers. Missing keys give
"not available", never an invented value. The India and sleep tables are copied from README
sections 9.4 and 9.5 (generated from ``reports/metrics``); a test checks that each row still
appears there.

Run:  python -m src.web.evidence
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.dash import ref
from src.dash.bundle import DEMO, read_json
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "web"
FIG_SRC = PROJECT_ROOT / "reports" / "figures"

LIMITS_WHATIF = [
    "It cannot tell you what the patient will actually do or eat, only what the models predict for the change you enter.",
    "It does not know about insulin or other medication doses, illness, stress or sleep.",
    "Carbohydrate effects are understated by the learned models (about half of the observed rise in matched real "
    "breakfasts); treat the size of any change as a lower bound.",
    "A walk can only be added 15 min after a meal for 30 min, the one setting that was tested.",
]

# (file stem, title, caption). Captions follow the Streamlit Evidence page.
FIGURES = [
    (
        "fusion_ablation",
        "Which inputs help",
        "Removing the CGM history hurts most. Adding wearable data or the clinical profile did not improve the forecasts.",
    ),
    (
        "unc_coverage",
        "Do 80% intervals contain the truth?",
        "Pooled coverage is close to 80 %, but it is not equal for all groups, and one method undercovered on the second cohort.",
    ),
    (
        "unc_alarm_froc",
        "Alert burden",
        "How many episodes each alert catches for a given number of false alarms a day.",
    ),
    (
        "whatif_plausibility",
        "What-if plausibility checks",
        "Pass rates of physiological checks. The monotone model passes the carbohydrate and walk checks by construction.",
    ),
    (
        "whatif_matched_meals",
        "Matched real breakfasts",
        "Predicted against observed peak difference in 87 matched pairs. The models understate the rise.",
    ),
    (
        "personalize_cold_start_lgbm",
        "Learning from a patient's first days",
        "Fine-tuning on the patient's own data lowers the error by about 1 mg/dL after 3 to 5 days.",
    ),
    (
        "transfer_matrix",
        "Does a model transfer to another population?",
        "A model trained on one cohort is no better than repeating the last value on the other, so it needs local training or calibration.",
    ),
    (
        "federated_mae",
        "Federated learning (simulated clinics)",
        "Sharing model updates instead of records matches pooled training and beats a clinic alone.",
    ),
    (
        "federated_privacy",
        "Privacy sweep",
        "Adding enough noise for a meaningful guarantee makes the model as bad as or worse than repeating the last value.",
    ),
]

# README 9.4 and 9.5 (SIMULATED where marked). Every row must appear in README.md (see tests).
INDIA = {
    "transfer": {
        "title": "Does a model trained on one population work on another? Not for free.",
        "sub": "Cross-cohort transfer on a shared 15-minute grid, mean absolute error at +60 min (mg/dL).",
        "cols": ["Direction", "Persistence", "Zero-shot", "Trained in-cohort", "Pooled"],
        "rows": [
            ["CGMacros → Shanghai T2DM", "19.50", "19.37", "15.95", "15.98"],
            ["Shanghai T2DM → CGMacros", "19.31", "20.08", "17.49", "17.85"],
        ],
        "note": "A model trained elsewhere is 3.4 mg/dL worse than one trained on the target population (CI 2.9 to 4.0) "
        "and no better than repeating the last value. A few days of the new patient's data close part of the gap "
        "(calibration with 3 days: −0.83, CI −1.19 to −0.50). A foreign model should not be deployed unchanged.",
    },
    "stress": {
        "title": "Synthetic Indian cohort (simulated)",
        "sub": "100 virtual patients per setting, 10 days, 16 Indian meal presets (carbohydrate from IFCT 2017), "
        "insulin sensitivity lowered by 0.46 log units. Models trained on the real cohorts; error at +60 min (mg/dL).",
        "cols": [
            "Insulin-resistance strength",
            "Persistence",
            "Zero-shot",
            "With 3-day calibration",
        ],
        "rows": [
            ["none (control)", "15.40", "15.21", "14.35"],
            ["mild", "16.57", "17.04", "15.87"],
            ["moderate", "16.81", "16.86", "15.68"],
            ["strong", "18.58", "17.05", "16.17"],
        ],
        "note": "Without calibration the model is not reliably better than repeating the last value on a shifted population; "
        "with 3 days of calibration it is. Adding synthetic patients to training never helped, so the synthetic "
        "cohort is used to stress-test and demonstrate, not to train.",
    },
    "federated": {
        "title": "Federated learning (simulated clinics)",
        "sub": "Open-cohort patients grouped into clinics; raw records never leave a clinic, only model updates are shared. Error at +60 min (mg/dL).",
        "cols": ["Layout", "Persistence", "Local only", "Pooled (central)", "FedAvg"],
        "rows": [
            ["5 similar clinics", "19.97", "19.54", "17.10", "17.19"],
            ["5 clinics split by HbA1c", "19.14", "19.19", "16.90", "17.03"],
            ["2 cohorts as 2 sites", "19.77", "17.81", "17.83", "17.64"],
        ],
        "note": "Federated averaging matches pooled training (difference 0.09 to 0.12, intervals include 0) and beats a small "
        "clinic alone by about 2 mg/dL. Formal differential privacy is not claimed: with a handful of clinics, enough "
        "noise for a meaningful guarantee makes the model as bad as or far worse than repeating the last value. "
        "The honest claim is data minimisation.",
    },
}

SLEEP = {
    "title": "Could a model detect an effect of sleep on glucose? (simulated)",
    "sub": "100 simulated patients, 67,000 scored windows per condition. Sleep shortens each night and lowers insulin "
    "sensitivity from 06:00 the next day. m scales the effect (m = 0 is a false-positive control, m = 1 is literature "
    "scale). Change in error from adding the sleep channel, mg/dL, 95 % interval.",
    "cols": ["m", "Mean change in log insulin sensitivity", "+30 min", "+60 min", "+120 min"],
    "rows": [
        [
            "0 (control)",
            "0.000",
            "−0.03 [−0.09, +0.01]",
            "+0.09 [+0.02, +0.17]",
            "+0.18 [+0.06, +0.30]",
        ],
        ["1", "−0.016", "+0.02 [−0.00, +0.04]", "−0.04 [−0.12, +0.04]", "+0.12 [−0.01, +0.25]"],
        ["3", "−0.048", "+0.01 [−0.01, +0.02]", "+0.03 [−0.00, +0.07]", "+0.15 [+0.04, +0.26]"],
        ["5", "−0.080", "+0.02 [−0.00, +0.04]", "+0.07 [−0.01, +0.15]", "+0.11 [−0.01, +0.24]"],
    ],
    "note": "No detectable gain, even at five times the literature effect. The control behaves correctly. Heart rate and "
    "activity add nothing either. So the console shows sleep, heart rate variability and steps as context for the "
    "clinician, not as drivers of the forecast. This says the effect is small relative to meal-driven variation "
    "over 10 days, not that sleep does not matter.",
    "scale": {
        "title": "Larger run: 300 simulated patients, 30 days (about 9 times more patient-days)",
        "sub": "777,000 scored windows per condition. Same design; change in error from adding the sleep channel, mg/dL, 95 % interval.",
        "cols": ["m", "+30 min", "+60 min", "+120 min"],
        "rows": [
            [
                "0 (control)",
                "+0.064 [+0.040, +0.100]",
                "+0.067 [+0.047, +0.087]",
                "+0.058 [+0.016, +0.113]",
            ],
            ["1", "−0.061 [−0.097, −0.035]", "−0.031 [−0.073, +0.011]", "+0.005 [−0.045, +0.051]"],
        ],
        "note": "No leak: with no sleep effect the extra features make forecasts slightly worse. At literature scale the effect is "
        "detectable only at +30 minutes and is tiny (0.06 mg/dL, about 0.5 % of the forecast error); at +60 and +120 minutes "
        "it is still not detectable. A simulation result that says nothing about real patients.",
    },
}


def frame(df: pd.DataFrame) -> dict[str, Any]:
    """DataFrame as {cols, rows} of strings."""
    return {
        "cols": [str(c) for c in df.columns],
        "rows": [[str(v) for v in r] for r in df.itertuples(index=False)],
    }


def personalization(stats: dict[str, Any]) -> dict[str, Any]:
    """Cold-start curve per horizon, gains and the head-start result."""
    out: dict[str, Any] = {"horizons": {}}
    for h in ref.H:
        cur = ref.personalization_curve(stats, h)
        series: dict[str, list[dict[str, float | None]]] = {}
        for name, g in (cur.groupby("model", sort=False) if len(cur) else []):
            series[str(name)] = [
                {
                    "k": float(r.k),
                    "mae": float(r.mae),
                    "lo": None if pd.isna(r.lo) else float(r.lo),
                    "hi": None if pd.isna(r.hi) else float(r.hi),
                }
                for r in g.itertuples()
            ]
        out["horizons"][h] = {
            "series": series,
            "gain": {str(int(k)): v for k, v in ref.personalization_gain(stats, h).items()},
        }
    hs = ref.head_start(stats)
    out["head_start"] = {
        "change_mgdl": None if hs["mgdl"] is None else -float(hs["mgdl"]),
        "ci": None if not hs["ci"] else [-float(hs["ci"][1]), -float(hs["ci"][0])],
    }
    return out


def build(stats: dict[str, Any]) -> dict[str, Any]:
    """The whole evidence payload."""
    return {
        "tiers": frame(ref.tier_table(stats)),
        "robustness": frame(ref.robustness_table(stats)),
        "plausibility": frame(ref.plausibility_table(stats)),
        "dose": frame(ref.dose_table(stats)),
        "matched": frame(ref.matched_summary(stats)),
        "alarms": frame(ref.alarm_table(stats, "2")),
        "auroc": ref.auroc_text(stats),
        "coverage": ref.coverage_text(stats),
        "personalization": personalization(stats),
        "limitations": ref.LIMITATIONS,
        "limits_whatif": LIMITS_WHATIF,
        "figures": [
            {"file": f"figures/{n}.png", "title": t, "caption": c}
            for n, t, c in FIGURES
            if (FIG_SRC / f"{n}.png").exists()
        ],
        "india": INDIA,
        "sleep": SLEEP,
    }


def main() -> None:
    """Write web/evidence.json and copy the figures to web/figures."""
    path = DEMO / "stats.json.gz"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `make app-data` first.")
    ev = build(read_json(path))
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for f in ev["figures"]:
        shutil.copy(FIG_SRC / Path(f["file"]).name, OUT / f["file"])
    (OUT / "evidence.json").write_text(
        json.dumps(ev, ensure_ascii=False, separators=(",", ":")), "utf-8"
    )
    print(f"wrote {OUT / 'evidence.json'} and {len(ev['figures'])} figures")


if __name__ == "__main__":
    main()
