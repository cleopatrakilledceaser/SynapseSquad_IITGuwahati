"""Start the dashboard, visit every page and save a PNG of each to reports/screens/.

Usage: python scripts/app_screens.py   (needs streamlit, playwright and an installed Chrome or Edge)
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PAGES = ["overview", "live", "report", "whatif", "twin", "personalization", "tiers", "quality", "evidence", "export"]
PORT = 8599
OUT = Path("reports/screens")


def main() -> int:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/main.py", "--server.headless", "true",
                             "--server.port", str(PORT), "--browser.gatherUsageStats", "false"])  # fmt: skip
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=1)
                break
            except Exception:
                time.sleep(1)
        with sync_playwright() as p:
            try:
                br = p.chromium.launch(channel="chrome")
            except Exception:
                try:
                    br = p.chromium.launch(channel="msedge")
                except Exception:
                    br = p.chromium.launch()
            pg = br.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            for name in PAGES:
                pg.goto(f"http://localhost:{PORT}/" + ("" if name == "overview" else name))
                pg.wait_for_timeout(6000)
                exc = pg.locator("[data-testid=stException]").count()
                pg.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
                print(f"{name}: saved, exceptions on page: {exc}")
            br.close()
        print("browser errors:", errors or "none")
    finally:
        proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
