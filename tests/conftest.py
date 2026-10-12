"""Shared pytest fixtures: tiny synthetic toy data (labeled synthetic, never real)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TOY_N_PATIENTS = 30
TOY_ROWS_PER_PATIENT = 12
TOY_STATUSES = ("healthy", "prediabetes", "t2d")
TOY_SEED = 42


@pytest.fixture()
def toy_static() -> pd.DataFrame:
    """Synthetic static table: one row per patient with a diabetes status label."""
    pids = [f"toy_{i:03d}" for i in range(TOY_N_PATIENTS)]
    status = [TOY_STATUSES[i % len(TOY_STATUSES)] for i in range(TOY_N_PATIENTS)]
    return pd.DataFrame({"patient_id": pids, "diabetes_status": status, "synthetic": True})


@pytest.fixture()
def toy_timeseries(toy_static: pd.DataFrame) -> pd.DataFrame:
    """Synthetic row-level CGM-like frame (several rows per patient)."""
    rng = np.random.default_rng(TOY_SEED)
    pids = np.repeat(toy_static["patient_id"].to_numpy(), TOY_ROWS_PER_PATIENT)
    ts = pd.date_range("2026-01-01", periods=TOY_ROWS_PER_PATIENT, freq="15min")
    return pd.DataFrame(
        {
            "patient_id": pids,
            "timestamp": np.tile(ts, TOY_N_PATIENTS),
            "glucose": rng.normal(120, 25, size=len(pids)).round(1),
            "synthetic": True,
        }
    )
