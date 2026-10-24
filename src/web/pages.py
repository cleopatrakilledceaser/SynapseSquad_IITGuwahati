"""Wrap the console fragment into a standalone page for static hosting (GitHub Pages).

``web/index.html`` is written for the artifact host, which supplies the document skeleton
(doctype, charset, viewport, a small reset). A static host does not, so a browser guesses the
encoding and UTF-8 punctuation renders as mojibake. This adds the skeleton and the reset that the
artifact host provides, and copies the data files beside it.

Run:  python -m src.web.pages        (target ``make web-pages``)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.utils.config import PROJECT_ROOT

WEB = PROJECT_ROOT / "web"
OUT = PROJECT_ROOT / "web" / "_site"

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>GlucoTwin Console</title>
<style>
  :root { color-scheme: light; padding-top: env(safe-area-inset-top, 0px);
          padding-bottom: env(safe-area-inset-bottom, 0px); }
  html, body { margin: 0; }
  body { font: 14px system-ui, sans-serif; }
  img { max-width: 100%; }
  [hidden] { display: none !important; }
</style>
</head>
<body>
"""

TAIL = "\n</body>\n</html>\n"


def build() -> Path:
    """Write ``web/_site`` and return its path."""
    fragment = (WEB / "index.html").read_text("utf-8")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(HEAD + fragment + TAIL, "utf-8")
    (OUT / ".nojekyll").write_text("", "utf-8")
    for name in ("data.json", "evidence.json", "twin.json"):
        src = WEB / name
        if src.exists():
            shutil.copy(src, OUT / name)
    figs = WEB / "figures"
    if figs.exists():
        for f in sorted(figs.glob("*.png")):
            shutil.copy(f, OUT / "figures" / f.name)
    return OUT


def main() -> None:
    """Build the static site and report what was written."""
    out = build()
    n = len(list((out / "figures").glob("*.png")))
    print(f"wrote {out} (index.html, data.json, evidence.json, {n} figures)")


if __name__ == "__main__":
    main()
