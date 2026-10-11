"""
clean_store.py — the clean zone: validated, typed, de-duplicated tables in SQLite.

    runtime/data/clean/market_clean.db        one table per dataset (see contracts.DATASETS)

Every row carries its lineage: raw_sha256 (the raw file it was parsed from), raw_version,
parser_version, ingest_id and quality_flags. Writes are per delivery day inside one transaction
(replace the day's rows), so repeating a load gives the same table: idempotent.

The clean zone can always be deleted and rebuilt from the raw zone; `content_digest` gives a hash
of the data (lineage columns excluded) to prove two builds are identical.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import sqlite3
from typing import Dict, List, Optional

import pandas as pd

from common_layer.data_layer.contracts import DATASETS, Dataset

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "runtime", "data", "clean", "market_clean.db")

LINEAGE = ("raw_sha256", "raw_version", "parser_version", "ingest_id", "quality_flags")


def _sql_type(spec: Dataset, col: str) -> str:
    if col in spec.text_cols or col in ("delivery_date", "start_utc") or col in LINEAGE[:1] + LINEAGE[2:]:
        return "TEXT"
    if col in spec.int_cols or col in ("period_index", "resolution_min", "raw_version"):
        return "INTEGER"
    return "REAL"


def _columns(spec: Dataset) -> List[str]:
    base = ["delivery_date", "period_index", "resolution_min", "start_utc"]
    if "session" in spec.keys:
        base.insert(1, "session")
    return base + list(spec.value_cols) + list(spec.text_cols) + \
        [c for c in spec.int_cols if c not in base]


class CleanStore:
    def __init__(self, path: Optional[str] = None):
        self.path = os.path.abspath(path or DEFAULT_PATH)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init(self) -> None:
        with self._conn() as c:
            for spec in DATASETS.values():
                cols = _columns(spec) + list(LINEAGE)
                decl = ", ".join(f"{col} {_sql_type(spec, col)}" for col in cols)
                pk = ", ".join(spec.keys)
                c.execute(f"CREATE TABLE IF NOT EXISTS {spec.name} ({decl}, PRIMARY KEY ({pk}))")

    # --------------------------------------------------------------- write
    def replace_day(self, dataset: str, df: pd.DataFrame, *, raw_sha256: str, raw_version: int,
                    parser_version: str, ingest_id: str, quality_flags: str = "") -> int:
        """Replace all rows of the delivery day (and session) with `df`. One transaction."""
        spec = DATASETS[dataset]
        cols = _columns(spec)
        out = df[cols].copy()
        out["raw_sha256"] = raw_sha256
        out["raw_version"] = raw_version
        out["parser_version"] = parser_version
        out["ingest_id"] = ingest_id
        out["quality_flags"] = quality_flags
        days = sorted(out["delivery_date"].unique())
        sessions = sorted(out["session"].unique()) if "session" in out.columns else [None]
        all_cols = cols + list(LINEAGE)
        marks = ",".join("?" * len(all_cols))
        rows = [tuple(None if (isinstance(v, float) and v != v) else v for v in r)
                for r in out[all_cols].itertuples(index=False, name=None)]
        with self._conn() as c:                       # context manager = one transaction
            for d in days:
                for s in sessions:
                    if s is None:
                        c.execute(f"DELETE FROM {dataset} WHERE delivery_date=?", (d,))
                    else:
                        c.execute(f"DELETE FROM {dataset} WHERE delivery_date=? AND session=?", (d, int(s)))
            c.executemany(f"INSERT INTO {dataset} ({','.join(all_cols)}) VALUES ({marks})", rows)
        return len(rows)

    # ---------------------------------------------------------------- read
    def read(self, dataset: str, start: Optional[dt.date] = None, end: Optional[dt.date] = None,
             lineage: bool = False) -> pd.DataFrame:
        spec = DATASETS[dataset]
        cols = _columns(spec) + (list(LINEAGE) if lineage else [])
        sql = f"SELECT {','.join(cols)} FROM {dataset}"
        where, args = [], []
        if start:
            where.append("delivery_date>=?"); args.append(start.isoformat())
        if end:
            where.append("delivery_date<=?"); args.append(end.isoformat())
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY " + ",".join(spec.keys)
        with self._conn() as c:
            return pd.read_sql_query(sql, c, params=args)

    def days(self, dataset: str) -> List[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute(f"SELECT DISTINCT delivery_date FROM {dataset} ORDER BY 1")]

    def latest_day(self, dataset: str) -> Optional[dt.date]:
        d = self.days(dataset)
        return dt.date.fromisoformat(d[-1]) if d else None

    def content_digest(self, dataset: str, start: Optional[dt.date] = None,
                       end: Optional[dt.date] = None) -> str:
        """SHA-256 of the data columns (lineage excluded), for proving two builds are identical."""
        df = self.read(dataset, start, end)
        return hashlib.sha256(df.to_csv(index=False, float_format="%.10g").encode("utf-8")).hexdigest()
