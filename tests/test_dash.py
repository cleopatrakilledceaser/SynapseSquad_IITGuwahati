"""Dashboard logic (Part 9): text rules, reference tables, twin what-if, PDF. No Streamlit needed."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.dash import pdfexport, ref
from src.dash import text as T
from src.dash import whatif as W
from src.dash.bundle import PatientData

CFG = {
    "risk_words": {"bounds": [0.10, 0.30, 0.60], "words": ["low", "moderate", "high", "very high"]},
    "confidence": {"wide_ratio": 1.5, "dq_uncertain_below": 0.70, "dq_high_from": 0.95, "stale_min": 60},
    "history_hours": 6, "forecast_min": 120, "y_range": [40, 300],
}  # fmt: skip


def row(**kw: float) -> dict:
    base = {
        "t": pd.Timestamp("2026-01-05 08:00"), "cgm_now": 150.0, "cgm_slope_30": 0.4,
        "p_hyper180|60": 0.62, "p_hyper140|60": 0.81, "p_hypo70|60": 0.03,
        "pred|30": 160.0, "pred|60": 168.0, "pred|120": 160.0,
        "lo80|30": 145.0, "hi80|30": 175.0, "lo80|60": 150.0, "hi80|60": 186.0, "lo80|120": 130.0, "hi80|120": 190.0,
        "dq": 0.99, "stale_min": 0.0, "confidence": "moderate", "width_ratio": 1.1,
    }  # fmt: skip
    return base | kw


def test_decision_names_the_active_alert_and_the_probability() -> None:
    d = T.decision(row(active_hyper180=True), CFG)
    assert (
        d["event"] == "hyper180" and "62%" in d["sentence"] and "above 180 mg/dL" in d["sentence"]
    )
    assert "rising" in d["sentence"]


def test_decision_says_already_above_when_glucose_is_beyond_the_threshold() -> None:
    d = T.decision(row(cgm_now=205.0, **{"p_hyper180|60": float("nan")}), CFG)
    assert d["event"] == "hyper180" and "already above 180" in d["sentence"]


def test_missing_numbers_are_written_not_available() -> None:
    r = row(
        **{
            "p_hypo70|60": float("nan"),
            "p_hyper180|60": float("nan"),
            "p_hyper140|60": float("nan"),
        }
    )
    assert "not available" in T.decision(r, CFG)["sentence"]
    assert T.mgdl(None) == "not available" and T.pct(float("nan")) == "not available"


def test_uncertain_forecast_gets_a_banner_with_reasons() -> None:
    conf = T.confidence(row(confidence="uncertain", width_ratio=2.0, dq=0.5, stale_min=90.0), CFG)
    b = T.uncertain_banner(conf)
    assert b is not None and "Do not rely on this forecast" in b
    assert len(conf["reasons"]) == 3 and conf["bars"] == 1
    assert T.uncertain_banner(T.confidence(row(), CFG)) is None


def test_language_is_decision_support() -> None:
    texts = [
        T.DISCLAIMER,
        T.WHATIF_NOTE,
        T.quality_banner(row()),
        T.decision(row(), CFG)["sentence"],
    ]
    for s in texts:
        assert (
            "diagnos" not in s.lower().replace("not for diagnosis", "")
            and "prescri" not in s.lower()
        )
    assert "not a medical device" in T.DISCLAIMER.lower()
    assert T.WHATIF_NOTE.startswith("Model-based estimate, not a causal guarantee.")


def test_quality_banner_states_what_the_score_cannot_see() -> None:
    assert "not wrong data" in T.quality_banner(row())
    assert "not available" in T.quality_banner({})


def test_synthetic_profile_is_marked_assumed() -> None:
    lines = dict(
        T.profile_lines(
            {
                "synthetic": True,
                "hba1c_pct": 6.9,
                "fasting_glucose_mgdl": 128.0,
                "status": "synthetic",
            }
        )
    )
    assert "assumed" in lines["HbA1c"] and "assumed" in lines["Fasting glucose"]


STATS_PATH = "reports/metrics"


@pytest.fixture(scope="module")
def stats() -> dict:
    from src.appdata import export as ex

    try:
        return json.loads(
            json.dumps(ex.cohort_stats())
        )  # JSON round trip like the bundle (int keys become str)
    except FileNotFoundError:
        pytest.skip("stored metric files are not available")


def test_reference_tables_have_every_horizon_and_no_invented_numbers(stats: dict) -> None:
    t = ref.tier_table(stats)
    assert list(t.columns[-3:]) == ["+30 min", "+60 min", "+120 min"] and len(t) == 6
    assert (t[["+30 min", "+60 min", "+120 min"]] != "not available").all().all()
    assert ref.tier_table({})["+60 min"].eq("not available").all()


def test_fine_tuning_gain_matches_the_stored_curve(stats: dict) -> None:
    g = ref.personalization_gain(stats, "60")
    assert g[3.0] < 0 and g[5.0] < g[3.0] < g[1.0] <= 0.0
    assert -3.0 < g[5.0] < 0.0  # about one mg/dL, not a large effect


def test_robustness_table_shows_coverage_falling_under_drift(stats: dict) -> None:
    t = ref.robustness_table(stats)
    assert t.iloc[0]["Fault"] == "No fault" and len(t) == 7
    assert "46" in t[t["Fault"].str.contains("20 mg/dL per day")]["80% interval coverage"].iloc[0]


def test_alert_table_flags_limited_evidence(stats: dict) -> None:
    t = ref.alarm_table(stats)
    low = t[t["Event"].str.contains("Below 70")]
    assert low["Episodes caught"].str.contains("limited evidence").all()
    assert (
        not t[t["Event"].str.contains("Above 180")]["Episodes caught"].str.contains("limited").any()
    )


def test_plausibility_marks_what_passes_by_construction(stats: dict) -> None:
    t = ref.plausibility_table(stats)
    mono = t[(t["Model"].str.startswith("Monotone")) & (t["Test"].str.contains("never"))]
    assert (mono["By construction"] == "yes").all()
    assert ref.dose_table(stats)["Within range"].tolist().count("no") == 2


def make_patient() -> tuple[PatientData, dict, dict]:
    cs, rs = [0, 15, 30, 45, 60, 90, 120, 150], [0.0, 0.5, 1.0, 2.0, 4.0]
    arr = np.zeros((2, 2, 8, 5, 25))
    t = np.arange(25) * 5.0
    for iv in range(2):
        for iw in range(2):
            for ic, c in enumerate(cs):
                for ir, r in enumerate(rs):
                    arr[iv, iw, ic, ir] = 100 + c * 0.5 / (1 + 0.2 * r) * (1 - 0.3 * iw) * np.exp(
                        -(((t - 60) / 40) ** 2)
                    )
    meals = pd.DataFrame(
        {
            "t": [pd.Timestamp("2026-01-05 08:10")],
            "carbs_g": [45.0],
            "protein_g": [10.0],
            "fat_g": [8.0],
            "fiber_g": [4.0],
        }
    )
    p = PatientData(
        "p",
        {"label": "x"},
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        meals,
        {},
        {"w0": arr},
        [],
    )
    win = {"k": 0, "meal_idx": 0, "t": pd.Timestamp("2026-01-05 08:55"), "offset_min": 45.0}
    ctx = {
        "twin_grid": {"carbs_g": cs, "slow_ratio": rs},
        "variants": ["twin_learned", "twin_assumed"],
    }
    return p, win, ctx


def test_twin_whatif_reads_the_grid_and_labels_interpolation() -> None:
    p, win, ctx = make_patient()
    base = W.logged_scenario(p, win)
    on_grid = W.twin_run(
        p,
        win,
        "twin_learned",
        W.Scenario(45, 0, 0, 0, False),
        W.Scenario(45, 0, 0, 0),
        ctx,
        (40, 400),
    )
    assert on_grid is not None and not on_grid["interpolated"]
    off = W.twin_run(p, win, "twin_learned", W.Scenario(37, 5, 4, 2, True), base, ctx, (40, 400))
    assert off is not None and off["interpolated"]
    assert (
        off["new"][60]["mean"] < off["base"][60]["mean"]
    )  # less carbohydrate and a walk lower the peak
    assert off["new"][60]["lo"] is None  # the twin has no precomputed interval
    far = W.twin_run(p, win, "twin_assumed", W.Scenario(300, 0, 0, 0, False), base, ctx, (40, 400))
    assert far is not None and far["clamped_inputs"]


def test_comparison_table_reports_changes_in_mgdl() -> None:
    p, win, ctx = make_patient()
    r = W.twin_run(
        p,
        win,
        "twin_learned",
        W.Scenario(0, 0, 0, 0, False),
        W.logged_scenario(p, win),
        ctx,
        (40, 400),
    )
    t = W.comparison_table(r)
    assert (
        list(t["Horizon"]) == ["+30 min", "+60 min", "+120 min"]
        and t["Change"].str.endswith("mg/dL").all()
    )


def test_visit_summary_pdf_is_a_valid_one_page_pdf(stats: dict) -> None:
    t = pd.date_range("2026-01-05 02:00", "2026-01-05 08:00", freq="5min")
    hist = pd.DataFrame({"t": t, "glucose": 110 + 25 * np.sin(np.arange(len(t)) / 9)})
    meals = pd.DataFrame({"t": [pd.Timestamp("2026-01-05 07:30")], "carbs_g": [45.0]})
    b = pdfexport.visit_summary_pdf(
        {"label": "Synthetic patient", "synthetic": True}, row(), hist, meals, [], stats, CFG, 2
    )
    assert b.startswith(b"%PDF") and b.count(b"/Type /Page\n") + b.count(b"/Type /Page ") >= 1
