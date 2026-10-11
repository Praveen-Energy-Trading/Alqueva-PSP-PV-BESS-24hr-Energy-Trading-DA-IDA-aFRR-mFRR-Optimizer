"""
demo.py — an offline stand-in for the market sources, built from real files saved in tests/fixtures/raw.

Serves one complete real delivery day (2025-10-26, the 25-hour daylight-saving day, 100 quarter-hours)
for all eight sources, so the whole pipeline can be run and inspected with no network.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, Tuple

from common_layer.data_layer import fetchers

DEMO_DAY = dt.date(2025, 10, 26)
_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "tests", "fixtures", "raw")

_FILES = {
    "omie_da":       ("da_20251026.txt",            fetchers.url_omie_da(DEMO_DAY)),
    "omie_ida1":     ("ida1_20251026.txt",          fetchers.url_omie_ida(DEMO_DAY, 1)),
    "omie_ida2":     ("ida2_20251026.txt",          fetchers.url_omie_ida(DEMO_DAY, 2)),
    "omie_ida3":     ("ida3_20251026.txt",          fetchers.url_omie_ida(DEMO_DAY, 3)),
    "omie_xbid":     ("xbid_20251026.txt",          fetchers.url_omie_xbid(DEMO_DAY)),
    "ren_afrr":      ("ren_afrr_20251026.json",     fetchers.url_ren("BaFRRPreco/GetBaFRRPreco", DEMO_DAY)),
    "ren_mfrr":      ("ren_mfrr_20251026.json",     fetchers.url_ren("MFRRPreco/GetMFRRPreco", DEMO_DAY)),
    "ren_imbalance": ("ren_imbalance_20251026.json", fetchers.url_ren("DesvioPreco/GetDesvioPreco", DEMO_DAY)),
}


class DemoHttp:
    """Drop-in for fetchers.default_http_get: serves the saved files, 404 for everything else."""

    def __init__(self) -> None:
        self.routes: Dict[str, bytes] = {}
        for _, (name, url) in _FILES.items():
            with open(os.path.join(_FIX, name), "rb") as fh:
                self.routes[url] = fh.read()
        self.calls = []

    def __call__(self, url: str, headers: Dict[str, str], timeout: float) -> Tuple[int, bytes]:
        self.calls.append(url)
        return (200, self.routes[url]) if url in self.routes else (404, b"")
