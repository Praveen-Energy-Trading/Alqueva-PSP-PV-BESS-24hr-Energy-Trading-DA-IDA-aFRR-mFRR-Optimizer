"""
raw_store.py — the raw zone: what was downloaded, byte for byte, never edited.

Rules
    * A file is written once and never overwritten or deleted by this code.
    * Every capture is listed in a manifest (JSON lines) with SHA-256, size, time of retrieval
      (UTC), URL and HTTP status.
    * Downloading the same delivery date again: identical bytes add only an "unchanged" line to
      the manifest; different bytes (a source revision) are stored as a NEW version, so earlier
      versions stay available and "as of" reads are possible.
    * Reading verifies the checksum, so silent corruption is detected.

Layout:  <root>/<source>/<YYYY>/<key>__v<NNN>__<sha10><ext>     key = delivery date (YYYY-MM-DD)
Default root: <repo>/runtime/data/raw
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_ROOT = os.path.join(_REPO, "runtime", "data", "raw")


class RawIntegrityError(RuntimeError):
    """A raw file is missing or its checksum no longer matches the manifest."""


@dataclass(frozen=True)
class RawRecord:
    source: str
    key: str
    version: int
    path: str                 # relative to the store root
    sha256: str
    size: int
    retrieved_at_utc: str
    url: str = ""
    http_status: int = 200
    meta: Dict[str, object] = field(default_factory=dict)
    status: str = "new"       # "new" | "unchanged" (set on the object returned by save())


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RawStore:
    def __init__(self, root: Optional[str] = None):
        self.root = os.path.abspath(root or DEFAULT_ROOT)
        self.manifest_path = os.path.join(self.root, "manifest.jsonl")

    # ------------------------------------------------------------ manifest
    def _read_manifest(self) -> List[dict]:
        if not os.path.exists(self.manifest_path):
            return []
        with open(self.manifest_path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def _append(self, entry: dict) -> None:
        os.makedirs(self.root, exist_ok=True)
        with open(self.manifest_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    @staticmethod
    def _to_record(e: dict, status: Optional[str] = None) -> RawRecord:
        return RawRecord(source=e["source"], key=e["key"], version=e["version"], path=e["path"],
                         sha256=e["sha256"], size=e["size"], retrieved_at_utc=e["retrieved_at_utc"],
                         url=e.get("url", ""), http_status=e.get("http_status", 200),
                         meta=e.get("meta", {}), status=status or "new")

    # ------------------------------------------------------------- queries
    def versions(self, source: str, key: str) -> List[RawRecord]:
        """All stored versions of (source, key), oldest first."""
        seen: Dict[int, dict] = {}
        for e in self._read_manifest():
            if e.get("event") == "stored" and e["source"] == source and e["key"] == key:
                seen[e["version"]] = e
        return [self._to_record(seen[v]) for v in sorted(seen)]

    def latest(self, source: str, key: str, as_of_utc: Optional[dt.datetime] = None) -> Optional[RawRecord]:
        """Newest version, or the newest retrieved at or before `as_of_utc`."""
        recs = self.versions(source, key)
        if as_of_utc is not None:
            limit = _iso(as_of_utc)
            recs = [r for r in recs if r.retrieved_at_utc <= limit]
        return recs[-1] if recs else None

    def keys(self, source: str) -> List[str]:
        return sorted({e["key"] for e in self._read_manifest()
                       if e.get("event") == "stored" and e["source"] == source})

    # --------------------------------------------------------------- write
    def save(self, source: str, key: str, content: bytes, *, url: str = "", http_status: int = 200,
             ext: str = ".txt", meta: Optional[dict] = None,
             retrieved_at: Optional[dt.datetime] = None) -> RawRecord:
        sha = hashlib.sha256(content).hexdigest()
        when = _iso(retrieved_at or _now_utc())
        existing = self.versions(source, key)
        if existing and existing[-1].sha256 == sha:
            self._append({"event": "unchanged", "source": source, "key": key,
                          "version": existing[-1].version, "sha256": sha, "retrieved_at_utc": when})
            r = existing[-1]
            return RawRecord(**{**r.__dict__, "status": "unchanged"})
        version = (existing[-1].version + 1) if existing else 1
        year = key[:4]
        rel = os.path.join(source, year, f"{key}__v{version:03d}__{sha[:10]}{ext}")
        full = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "xb") as fh:                       # "x": refuses to overwrite an existing file
            fh.write(content)
        entry = {"event": "stored", "source": source, "key": key, "version": version, "path": rel,
                 "sha256": sha, "size": len(content), "retrieved_at_utc": when, "url": url,
                 "http_status": http_status, "meta": meta or {}}
        self._append(entry)
        return self._to_record(entry, "new")

    # ---------------------------------------------------------------- read
    def read(self, rec: RawRecord) -> bytes:
        full = os.path.join(self.root, rec.path)
        if not os.path.exists(full):
            raise RawIntegrityError(f"raw file missing: {rec.path}")
        with open(full, "rb") as fh:
            data = fh.read()
        if hashlib.sha256(data).hexdigest() != rec.sha256:
            raise RawIntegrityError(f"checksum mismatch: {rec.path}")
        return data

    def verify_all(self) -> List[str]:
        """Check every stored file against the manifest; returns a list of problems."""
        problems: List[str] = []
        for e in self._read_manifest():
            if e.get("event") != "stored":
                continue
            rec = self._to_record(e)
            try:
                self.read(rec)
            except RawIntegrityError as exc:
                problems.append(str(exc))
        return problems
