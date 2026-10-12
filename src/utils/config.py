"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"


def load_config(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` as a dictionary.

    Args:
        name: Config file stem (e.g. ``"base"``) or a path to a YAML file.
        config_dir: Optional override of the config directory.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(name)
    if path.suffix not in {".yaml", ".yml"}:
        path = (config_dir or CONFIG_DIR) / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping, got {type(data).__name__}")
    return data
