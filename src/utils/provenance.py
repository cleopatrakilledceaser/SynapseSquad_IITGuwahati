"""Provenance logging: every metrics file records git hash, config, seed and data hash."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT

_CHUNK_BYTES = 1 << 20


def git_hash(repo: Path = PROJECT_ROOT) -> str:
    """Return the current git commit hash (suffixed ``-dirty`` if uncommitted changes)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def file_hash(paths: list[Path]) -> str:
    """Return a SHA-256 over the contents of the given files (sorted by path)."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(str(path.name).encode())
        with path.open("rb") as fh:
            while chunk := fh.read(_CHUNK_BYTES):
                digest.update(chunk)
    return digest.hexdigest()


def write_metrics(
    name: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    data_hash: str,
    out_dir: Path | None = None,
) -> Path:
    """Write ``reports/metrics/<name>.json`` with provenance fields.

    Args:
        name: Experiment name (file stem).
        metrics: Metric values.
        config: Full configuration used.
        seed: Global seed used.
        data_hash: Hash of the input data (see :func:`file_hash`).
        out_dir: Override output directory.

    Returns:
        Path of the written file.
    """
    out = out_dir or PROJECT_ROOT / "reports" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_hash": git_hash(),
        "seed": seed,
        "data_hash": data_hash,
        "config": config,
        "metrics": metrics,
    }
    path = out / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
