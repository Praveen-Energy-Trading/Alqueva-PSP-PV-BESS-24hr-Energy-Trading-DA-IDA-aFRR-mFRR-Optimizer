"""
ingest.py — raw zone -> clean zone, one delivery day at a time.

    ingest_day(source, day)       download -> check it parses -> store raw -> validate -> write clean
    ingest_range(sources, a, b)   the same for many days; skips days already in the clean zone
    rebuild_clean(source, days)   clean zone from the raw zone only (no network): the exit test

Outcomes per (source, day): loaded | unchanged | skipped | not_published | rejected | failed |
out_of_regime. Every outcome is appended to the load log (JSON lines) with the ingest id.

Order of safety: a response is stored in the raw zone only if it parses as the expected file (an
HTML error page returned with HTTP 200 is not data); the clean zone only receives days that pass
the contract's error rules.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from common_layer.data_layer import fetchers, parsers, regimes
from common_layer.data_layer.clean_store import CleanStore
from common_layer.data_layer.contracts import validate_day
from common_layer.data_layer.raw_store import RawRecord, RawStore

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_LOG = os.path.join(_REPO, "runtime", "data", "load_log.jsonl")


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    ext: str
    url: Callable[[dt.date], str]
    parse: Callable[[bytes, dt.date], pd.DataFrame]
    legacy_before: Optional[str] = None     # regime id; days before it are a different rule set


def _ren(service: str, dataset: str) -> SourceSpec:
    return SourceSpec(dataset, ".json", lambda d, s=service: fetchers.url_ren(s, d),
                      lambda raw, d, ds=dataset: parsers.parse_ren(raw, d, ds))


SOURCES: Dict[str, SourceSpec] = {
    "omie_da": SourceSpec("da_price", ".txt", fetchers.url_omie_da, parsers.parse_omie_da),
    "omie_xbid": SourceSpec("xbid_price", ".txt", fetchers.url_omie_xbid, parsers.parse_omie_xbid),
    "ren_afrr": _ren("BaFRRPreco/GetBaFRRPreco", "afrr_price"),
    "ren_mfrr": _ren("MFRRPreco/GetMFRRPreco", "mfrr_price"),
    "ren_imbalance": _ren("DesvioPreco/GetDesvioPreco", "imbalance_price"),
}
for _s in (1, 2, 3):
    SOURCES[f"omie_ida{_s}"] = SourceSpec(
        "ida_price", ".txt", lambda d, s=_s: fetchers.url_omie_ida(d, s),
        lambda raw, d, s=_s: parsers.parse_omie_ida(raw, d, s), legacy_before="ida_three_sessions")


class LoadLog:
    def __init__(self, path: Optional[str] = None):
        self.path = os.path.abspath(path or DEFAULT_LOG)

    def write(self, **entry) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        entry["at_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")

    def read(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


def new_ingest_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def _build_clean(source: str, day: dt.date, rec: RawRecord, raw: RawStore, clean: CleanStore,
                 ingest_id: str) -> dict:
    spec = SOURCES[source]
    df = spec.parse(raw.read(rec), day)
    result = validate_day(spec.dataset, df, day)
    if not result.ok:
        return {"status": "rejected", "reason": "; ".join(f"{i.rule}: {i.message}" for i in result.issues
                                                         if i.severity == "error")}
    n = clean.replace_day(spec.dataset, df, raw_sha256=rec.sha256, raw_version=rec.version,
                          parser_version=parsers.PARSER_VERSION, ingest_id=ingest_id,
                          quality_flags=result.flags())
    return {"status": "loaded", "rows": n, "warnings": [i.message for i in result.warnings]}


def ingest_day(source: str, day: dt.date, *, raw: RawStore, clean: CleanStore, fetcher: fetchers.Fetcher,
               log: LoadLog, ingest_id: Optional[str] = None, refresh: bool = False) -> dict:
    ingest_id = ingest_id or new_ingest_id()
    spec = SOURCES[source]
    base = {"ingest_id": ingest_id, "source": source, "delivery_date": day.isoformat()}

    def done(out: dict) -> dict:
        out = {**base, **out}
        log.write(**out)
        return out

    if spec.legacy_before:
        cut = next(e["date"] for e in regimes.load_regimes() if e["id"] == spec.legacy_before)
        if day < cut:
            return done({"status": "out_of_regime",
                         "reason": f"before {spec.legacy_before} ({cut}); sessions mean something else"})
    if not refresh and raw.latest(source, day.isoformat()) is not None:
        return done({"status": "skipped", "reason": "already in raw zone (use refresh=True to re-check the source)"})

    url = spec.url(day)
    try:
        got = fetcher.get(url)
    except fetchers.NotPublished as exc:
        return done({"status": "not_published", "reason": str(exc)})
    except fetchers.FetchError as exc:
        return done({"status": "failed", "reason": str(exc)})

    try:                                   # must parse as the expected file before it is kept
        spec.parse(got.content, day)
    except parsers.EmptyPublication as exc:    # source says "no data"; not stored, so a later run re-checks
        return done({"status": "empty_publication", "reason": str(exc), "bytes": len(got.content)})
    except Exception as exc:
        import hashlib
        return done({"status": "rejected", "reason": f"unparseable response: {exc}",
                     "sha256": hashlib.sha256(got.content).hexdigest(), "bytes": len(got.content),
                     "head": got.content[:120].decode("latin-1", "replace")})

    rec = raw.save(source, day.isoformat(), got.content, url=got.url, http_status=got.status, ext=spec.ext)
    out = _build_clean(source, day, rec, raw, clean, ingest_id)
    out.update({"raw_status": rec.status, "raw_version": rec.version, "sha256": rec.sha256,
                "attempts": got.attempts})
    if out["status"] == "loaded" and rec.status == "unchanged":
        out["status"] = "unchanged"
    return done(out)


def ingest_range(sources: Iterable[str], start: dt.date, end: dt.date, *, raw: RawStore,
                 clean: CleanStore, fetcher: fetchers.Fetcher, log: LoadLog,
                 refresh: bool = False) -> Dict[str, int]:
    """Load every day in [start, end] for each source. Returns outcome counts."""
    iid = new_ingest_id()
    counts: Dict[str, int] = {}
    day = start
    while day <= end:
        for src in sources:
            out = ingest_day(src, day, raw=raw, clean=clean, fetcher=fetcher, log=log,
                             ingest_id=iid, refresh=refresh)
            counts[out["status"]] = counts.get(out["status"], 0) + 1
        day += dt.timedelta(days=1)
    return counts


def rebuild_clean(source: str, days: Iterable[dt.date], *, raw: RawStore, clean: CleanStore,
                  log: Optional[LoadLog] = None, as_of_utc: Optional[dt.datetime] = None) -> Dict[str, int]:
    """Recreate clean-zone rows from the raw zone only. `as_of_utc` rebuilds the state known then."""
    iid = new_ingest_id()
    counts: Dict[str, int] = {}
    for day in days:
        rec = raw.latest(source, day.isoformat(), as_of_utc)
        if rec is None:
            out = {"status": "no_raw"}
        else:
            out = _build_clean(source, day, rec, raw, clean, iid)
        counts[out["status"]] = counts.get(out["status"], 0) + 1
        if log:
            log.write(ingest_id=iid, source=source, delivery_date=day.isoformat(), mode="rebuild", **out)
    return counts
