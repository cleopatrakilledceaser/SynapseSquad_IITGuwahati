"""Download open-access raw datasets into data/raw/ and verify checksums.

Raw data is never committed (see .gitignore). Every source here is open access and needs no login:

- cgmacros : PhysioNet CGMacros v1.0.0 (CC BY-NC-SA 4.0). Fetched from PhysioNet's open-data S3
             mirror (same files, ~40x faster here) and verified against PhysioNet's SHA256SUMS.txt.
- shanghai : figshare article 21600933 (CC BY 4.0), verified by the figshare-supplied MD5.
- bigideas : PhysioNet BIG IDEAs v1.1.3 (ODC-By 1.0). Only small metadata files by default; the
             full 4.7 GB zip requires ``--full``.

Usage (repo root): ``python data/scripts/download.py cgmacros shanghai bigideas``
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CHUNK = 1 << 20

CGMACROS_BASE = "https://physionet.org/files/cgmacros/1.0.0/"
CGMACROS_S3 = "https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0/"
CGMACROS_FILES = [
    "SHA256SUMS.txt",
    "DataDictionary.pdf",
    "DataDictionary_Bio.csv",
    "DataDictionary_CGMacros-00X.csv",
    "DataDictionary_Gut_Health_Test.csv",
    "DataDictionary_Microbes.csv",
    "LICENSE.txt",
    "CGMacros_dateshifted365.zip",
]
SHANGHAI_URL = "https://ndownloader.figshare.com/files/42966622"
SHANGHAI_MD5 = "4bfb61cfa506b48155fd9e841ee48e21"
BIGIDEAS_BASE = "https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3/"
BIGIDEAS_SMALL = ["Demographics.csv", "SHA256SUMS.txt", "LICENSE.txt"]


def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``; resumes with curl when available."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    if curl:
        cmd = [curl, "-sS", "-L", "--retry", "5", "-C", "-", "-o", str(dest), url]
        subprocess.run(cmd, check=True)
    else:
        urllib.request.urlretrieve(url, dest)  # noqa: S310 (fixed https URLs)


def cgmacros() -> None:
    """Download CGMacros and verify every file against SHA256SUMS.txt."""
    out = RAW / "cgmacros"
    for name in CGMACROS_FILES:
        base = CGMACROS_S3 if name.endswith(".zip") else CGMACROS_BASE
        if not (out / name).exists() or name.endswith(".zip"):
            fetch(base + name, out / name)
    sums = dict(
        reversed(line.split()) for line in (out / "SHA256SUMS.txt").read_text().splitlines() if line
    )
    for name in CGMACROS_FILES[1:]:
        ok = _hash(out / name, "sha256") == sums[name]
        print(f"cgmacros {name}: sha256 {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"Checksum mismatch for {name}; delete it and re-run.")


def shanghai() -> None:
    """Download the Shanghai T1DM/T2DM zip and verify its MD5."""
    dest = RAW / "shanghai" / "diabetes_datasets.zip"
    if not dest.exists():
        fetch(SHANGHAI_URL, dest)
    ok = _hash(dest, "md5") == SHANGHAI_MD5
    print(f"shanghai diabetes_datasets.zip: md5 {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("Checksum mismatch; delete the file and re-run.")


def bigideas(full: bool) -> None:
    """Download BIG IDEAs metadata (default) or the full 4.7 GB release (``full``)."""
    out = RAW / "bigideas"
    for name in BIGIDEAS_SMALL:
        fetch(BIGIDEAS_BASE + name, out / name)
    if full:
        fetch(
            "https://physionet.org/content/big-ideas-glycemic-wearable/get-zip/1.1.3/",
            out / "big-ideas-glycemic-wearable-1.1.3.zip",
        )
    print("bigideas: metadata downloaded" + (" + full zip" if full else ""))


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("datasets", nargs="+", choices=["cgmacros", "shanghai", "bigideas"])
    p.add_argument("--full", action="store_true", help="BIG IDEAs: download the full 4.7 GB zip")
    args = p.parse_args()
    for d in args.datasets:
        if d == "cgmacros":
            cgmacros()
        elif d == "shanghai":
            shanghai()
        else:
            bigideas(args.full)


if __name__ == "__main__":
    main()
