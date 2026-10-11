"""
render_pipeline_diagram.py — render docs/pipeline_architecture.html to PNG (2x) and SVG (the PNG wrapped
in an SVG, as before).

    python tools/render_pipeline_diagram.py

Needs Playwright (pip install playwright) and a Chrome or Edge installation (no browser download).
Edit the HTML to change labels; this script only renders.
"""
from __future__ import annotations

import base64
import os
import shutil
import sys

from playwright.sync_api import sync_playwright

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
HTML = os.path.join(REPO, "docs", "pipeline_architecture.html")
PNG = os.path.join(REPO, "docs", "pipeline_architecture.png")
SVG = os.path.join(REPO, "docs", "pipeline_architecture.svg")
WIDTH = 940

CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
]


def find_browser() -> str:
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    found = shutil.which("chrome") or shutil.which("msedge") or shutil.which("chromium")
    if not found:
        sys.exit("No Chrome/Edge/Chromium found; install one or add its path to CANDIDATES.")
    return found


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=find_browser())
        page = browser.new_page(viewport={"width": WIDTH, "height": 800}, device_scale_factor=2)
        page.goto("file:///" + HTML.replace("\\", "/"))
        page.wait_for_timeout(300)
        height = page.evaluate("document.documentElement.scrollHeight")
        page.set_viewport_size({"width": WIDTH, "height": height})
        page.screenshot(path=PNG, full_page=True)
        browser.close()
    with open(PNG, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    with open(SVG, "w", encoding="utf-8") as fh:
        fh.write(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"\n'
                 f'     width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">\n'
                 f'  <title>Alqueva PSP + PV + BESS — 24-Hour Trading Pipeline</title>\n'
                 f'  <image width="{WIDTH}" height="{height}" href="data:image/png;base64,{b64}"/>\n</svg>\n')
    print(f"rendered {PNG} ({WIDTH}x{height} css px, 2x) and {SVG}")


if __name__ == "__main__":
    main()
