"""Simulated EHR history for the synthetic Indian cohort (D-068). ALL ASSUMED, ALL SYNTHETIC.

Adds past diagnoses, medications, kidney function and genetic-marker dosages to the static
profile. Nothing here feeds back into the simulated glucose (display only), and every probability
in configs/india.yaml is an assumption chosen to look plausible, not an estimate. It uses its own
random stream so no earlier simulated quantity changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _p(*terms: float) -> float:
    return float(np.clip(sum(terms), 0.0, 0.95))


def history(rng: np.random.Generator, row: dict[str, Any], ecfg: dict[str, Any]) -> dict[str, Any]:
    """EHR history for one patient from the static profile ``row`` (age, bmi, hba1c, status)."""
    age, bmi, a1c = float(row["age_years"]), float(row["bmi"]), float(row["hba1c_pct"])
    t2d = row["diabetes_status"] == "t2d"
    d = ecfg["duration_years"]
    dur = float(np.clip(rng.normal(d["mean"], d["sd"]), d["lo"], d["hi"])) if t2d else 0.0
    dur = min(dur, max(age - 25.0, 0.0))  # diagnosed in adulthood
    over = lambda x, lim: max(x - lim, 0.0)  # noqa: E731
    h, dl, ck, rt, nu, cd = (ecfg[k] for k in (
        "hypertension", "dyslipidaemia", "ckd_stage3plus", "retinopathy", "neuropathy",
        "coronary_disease"))  # fmt: skip
    flags = {
        "hypertension": rng.random() < _p(h["base"], h["per_year_age_over_40"] * over(age, 40),
                                          h["per_bmi_over_25"] * over(bmi, 25), h["t2d_add"] * t2d),
        "dyslipidaemia": rng.random() < _p(dl["base"], dl["t2d_add"] * t2d),
        "ckd_stage3plus": rng.random() < _p(ck["base"], ck["per_year_duration"] * dur,
                                            ck["per_hba1c_over_7"] * over(a1c, 7)),
        "retinopathy": rng.random() < _p(rt["per_year_duration"] * dur,
                                         rt["per_hba1c_over_7"] * over(a1c, 7)),
        "neuropathy": rng.random() < _p(nu["per_year_duration"] * dur,
                                        nu["per_hba1c_over_7"] * over(a1c, 7)),
        "coronary_disease": rng.random() < _p(cd["base"], cd["per_year_age_over_50"] * over(age, 50),
                                              cd["t2d_add"] * t2d),
    }  # fmt: skip
    ins = ecfg["meds"]["insulin"]
    meds = {f"med_{k}": bool(t2d and rng.random() < v) for k, v in ecfg["meds"].items()
            if k != "insulin"}  # fmt: skip
    meds["med_insulin"] = bool(t2d and rng.random() < _p(ins["base"], ins["per_year_duration"] * dur,
                                                         ins["per_hba1c_over_8"] * over(a1c, 8)))  # fmt: skip
    egfr = float(np.clip(rng.normal(95 - 0.7 * over(age, 30), 12), 15, 130))
    if flags["ckd_stage3plus"]:
        egfr = float(np.clip(rng.normal(48, 8), 15, 59))
    gen = {k: int(rng.binomial(2, f)) for k, f in ecfg["genetics"].items()}
    return {
        "diabetes_duration_years": round(dur, 1), **{k: bool(v) for k, v in flags.items()}, **meds,
        "egfr_ml_min": round(egfr, 1), "family_history_t2d": bool(rng.random() < ecfg["family_history_t2d"]),
        **{f"gt_{k}": v for k, v in gen.items()}, "genetic_risk_alleles": int(sum(gen.values())),
    }  # fmt: skip
