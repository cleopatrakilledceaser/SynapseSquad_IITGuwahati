"""Loader of the offline app bundle ``app/demo`` written by ``make app-data`` (Part 9, D-058).

Pure pandas / numpy / json: no Streamlit and no plotting here, so it is unit-testable. Nothing is
computed from raw data: every number comes from files in the bundle.

Layout (see ``src/appdata/build.py``): patients.json, manifest.json, context.json, calib.json /
calib.npz, stats.json.gz, global_shap.json, figures/*.png, models/ and patients/<id>/ with
replay.parquet, explain.parquet, grid.parquet, meals.parquet, shap.npz, twin.npz, whatif.json.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT

DEMO = PROJECT_ROOT / "app" / "demo"
HORIZONS = (30, 60, 120)
EVENTS = ("hyper180", "hyper140", "hypo70")


class BundleMissing(FileNotFoundError):
    """Raised with a plain instruction when ``app/demo`` has not been built."""


def require(root: Path = DEMO) -> Path:
    """Return the bundle folder or raise with the exact command that builds it."""
    if not (root / "manifest.json").exists():
        raise BundleMissing(
            f"The app data is not built yet ({root}). Run: python -m src.appdata.build "
            "(about 7 minutes), then reload this page."
        )
    return root


def read_json(path: Path) -> Any:
    """JSON file (plain or .gz)."""
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    return json.loads(path.read_text("utf-8"))


@dataclass
class PatientData:
    """Everything the pages need for one demo patient (all label-free)."""

    pid: str
    meta: dict[str, Any]
    replay: pd.DataFrame
    explain: pd.DataFrame
    grid: pd.DataFrame
    meals: pd.DataFrame
    shap: dict[str, np.ndarray]
    twin: dict[str, np.ndarray] = field(default_factory=dict)
    windows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Display name (the synthetic patient is always called synthetic)."""
        return str(self.meta["label"])

    @property
    def synthetic(self) -> bool:
        """True for the simulator patient."""
        return bool(self.meta.get("synthetic", False))


@dataclass
class Bundle:
    """The whole app bundle."""

    root: Path
    patients: dict[str, dict[str, Any]]
    manifest: dict[str, Any]
    context: dict[str, Any]
    stats: dict[str, Any]
    global_shap: dict[str, list[dict[str, Any]]]
    calib: dict[str, Any]
    _cache: dict[str, PatientData] = field(default_factory=dict)

    def ids(self) -> list[str]:
        """Patient ids: real patients first (sorted), then the synthetic one."""
        real = sorted(p for p, m in self.patients.items() if not m.get("synthetic"))
        syn = sorted(p for p, m in self.patients.items() if m.get("synthetic"))
        return real + syn

    def patient(self, pid: str) -> PatientData:
        """Load (once) the tables of one patient."""
        if pid not in self._cache:
            self._cache[pid] = load_patient(self.root, pid, self.patients[pid])
        return self._cache[pid]

    def figure_path(self, name: str) -> Path:
        """Path of an evidence figure."""
        return self.root / "figures" / f"{name}.png"


def load_patient(root: Path, pid: str, meta: dict[str, Any]) -> PatientData:
    """Read one patient folder. Meal ids of the what-if windows are mapped by meal time."""
    d = root / "patients" / pid
    replay = pd.read_parquet(d / "replay.parquet")
    explain = pd.read_parquet(d / "explain.parquet")
    grid = pd.read_parquet(d / "grid.parquet")
    meals = pd.read_parquet(d / "meals.parquet").reset_index(drop=True)
    with np.load(d / "shap.npz", allow_pickle=False) as z:
        shap = {k: z[k] for k in z.files}
    twin: dict[str, np.ndarray] = {}
    if (d / "twin.npz").exists():
        with np.load(d / "twin.npz", allow_pickle=False) as z:
            twin = {k: z[k] for k in z.files}
    windows = []
    for w in read_json(d / "whatif.json"):
        t_meal = pd.Timestamp(w["meal_row"]["t"])
        hit = meals.index[meals["t"] == t_meal]
        if len(hit) != 1:
            continue  # a window whose meal cannot be identified is not offered
        windows.append({**w, "t": pd.Timestamp(w["t"]), "meal_idx": int(hit[0])})
    return PatientData(pid, meta, replay, explain, grid, meals, shap, twin, windows)


def load_bundle(root: Path = DEMO) -> Bundle:
    """Read the small files of the bundle (patient tables are loaded on demand)."""
    root = require(root)
    return Bundle(
        root=root,
        patients=read_json(root / "patients.json"),
        manifest=read_json(root / "manifest.json"),
        context=read_json(root / "context.json"),
        stats=read_json(root / "stats.json.gz"),
        global_shap=read_json(root / "global_shap.json"),
        calib=read_json(root / "calib.json"),
    )
