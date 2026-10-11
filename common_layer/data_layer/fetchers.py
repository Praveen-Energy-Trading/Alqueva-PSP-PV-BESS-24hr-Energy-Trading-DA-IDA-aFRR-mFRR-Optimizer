"""
fetchers.py — download one delivery day from one source, politely and with retries.

Only HTTP lives here. Nothing is parsed or stored; the caller gets bytes, a status and the URL.

Policy
    * identify ourselves with a User-Agent, pause `min_interval_s` between requests (default 1 s);
    * 404 (or an empty body) means "not published (yet)": no retry, reported as NotPublished;
    * timeouts, connection errors, HTTP 429/5xx: retried with exponential backoff, then FetchError;
    * the transport (`http_get`) is injectable, so tests never touch the network.

The REN header key is the public key the REN market website itself sends (already used by the
existing loaders); it can be overridden with the REN_API_KEY environment variable.
"""
from __future__ import annotations

import datetime as dt
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

REN_KEY_B64 = os.environ.get(
    "REN_API_KEY", "bWVyY2Fkb19tTDI3M0J0aUxlUmNxZnFCcUltV0JmNXV2UFRtZFc0Vkh4YjRFZUQ2")
USER_AGENT = "alqueva-optimizer-research/1.0 (academic; contact: repository owner)"

HttpGet = Callable[[str, Dict[str, str], float], Tuple[int, bytes]]


class NotPublished(Exception):
    """The source has no file for this day (yet)."""


class FetchError(Exception):
    """Network or server failure after all retries."""


def default_http_get(url: str, headers: Dict[str, str], timeout: float) -> Tuple[int, bytes]:
    import requests
    r = requests.get(url, headers=headers, timeout=timeout)
    return r.status_code, r.content


# ----------------------------------------------------------------------- URLs
def url_omie_da(day: dt.date) -> str:
    y, m, d = day.strftime("%Y"), day.strftime("%m"), day.strftime("%d")
    return (f"https://www.omie.es/sites/default/files/dados/AGNO_{y}/MES_{m}/TXT/"
            f"INT_PBC_EV_H_1_{d}_{m}_{y}_{d}_{m}_{y}.TXT")


def url_omie_ida(day: dt.date, session: int) -> str:
    return ("https://www.omie.es/es/file-download?parents=marginalpibcpt&filename="
            f"marginalpibcpt_{day.strftime('%Y%m%d')}{session:02d}.1")


def url_omie_xbid(day: dt.date) -> str:
    return ("https://www.omie.es/es/file-download?parents=precios_pibcic&filename="
            f"precios_pibcic_{day.strftime('%Y%m%d')}.1")


def url_ren(service: str, day: dt.date) -> str:
    """service: BaFRRPreco/GetBaFRRPreco, MFRRPreco/GetMFRRPreco, DesvioPreco/GetDesvioPreco."""
    return (f"https://mercadoservices.ren.pt/api/{service}"
            f"?language=PT&dayQuery={day.day}&monthQuery={day.month}&yearQuery={day.year}&sWhere=")


def headers_for(url: str) -> Dict[str, str]:
    h = {"User-Agent": USER_AGENT}
    if "mercadoservices.ren.pt" in url:
        h.update({"X-ApiKey": REN_KEY_B64, "Accept": "application/json"})
    return h


# ------------------------------------------------------------------- download
@dataclass
class Fetched:
    url: str
    status: int
    content: bytes
    attempts: int


class Fetcher:
    def __init__(self, http_get: Optional[HttpGet] = None, *, retries: int = 4, backoff_s: float = 2.0,
                 min_interval_s: float = 1.0, timeout_s: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.http_get = http_get or default_http_get
        self.retries, self.backoff_s = retries, backoff_s
        self.min_interval_s, self.timeout_s = min_interval_s, timeout_s
        self._sleep = sleep
        self._last = 0.0

    def _pace(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            self._sleep(wait)
        self._last = time.monotonic()

    def get(self, url: str) -> Fetched:
        last_err = "unknown"
        for attempt in range(1, self.retries + 2):
            self._pace()
            try:
                status, body = self.http_get(url, headers_for(url), self.timeout_s)
            except Exception as exc:                              # timeout, connection reset, ...
                last_err = f"{type(exc).__name__}: {exc}"
            else:
                if status == 404 or (status == 200 and not body.strip()):
                    raise NotPublished(f"{url} -> HTTP {status}, empty or missing")
                if status == 200:
                    return Fetched(url, status, body, attempt)
                last_err = f"HTTP {status}"
                if status not in (429, 500, 502, 503, 504):
                    raise FetchError(f"{url}: {last_err} (not retried)")
            if attempt <= self.retries:
                self._sleep(self.backoff_s * (2 ** (attempt - 1)))
        raise FetchError(f"{url}: {last_err} after {self.retries + 1} attempts")
