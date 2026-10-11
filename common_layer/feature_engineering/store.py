"""
store.py — the feature store: features computed as of each delivery day's forecast time, saved for reuse.

One SQLite table per (target, feature set); a row per hour of a delivery day, one column per feature, keyed by the
UTC start of the hour. The table's metadata records the feature set's version hash (a hash of every member's
definition) and the catalogue version, so a changed definition is detected and the table is rebuilt rather than
silently mixed with old values. Values are exactly what `FeatureBuilder.build` returns on the as-of view, so
reading from the store and building on the fly give the same numbers (tested).

    store = FeatureStore()
    store.materialize(view, builder, target_id, set_name, days)       # idempotent: a day is replaced as a whole
    X = store.read(target_id, set_name, first, last)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from typing import Iterable, List, Optional

import pandas as pd

from common_layer.feature_engineering.builder import FeatureBuilder
from common_layer.feature_engineering.catalogue import feature_set_version
from common_layer.forecast_evaluation.asof import AsOfView

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "runtime", "data", "features", "features.db")


class FeatureStore:
    def __init__(self, path: Optional[str] = None):
        self.path = os.path.abspath(path or DEFAULT_PATH)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS feature_sets_meta (table_name TEXT PRIMARY KEY, target_id TEXT, "
                      "set_name TEXT, set_version TEXT, features TEXT)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=60)

    @staticmethod
    def table_name(target_id: str, set_name: str) -> str:
        return f"fs_{target_id}__{set_name}"

    def _meta(self, table: str) -> Optional[dict]:
        with self._conn() as c:
            r = c.execute("SELECT set_version, features FROM feature_sets_meta WHERE table_name=?", (table,)).fetchone()
        return None if r is None else {"set_version": r[0], "features": json.loads(r[1])}

    def materialize(self, view: AsOfView, builder: FeatureBuilder, catalogue: dict, set_name: str,
                    days: Iterable[dt.date], history_days: int = 400) -> int:
        """Build and store the features of each day (as-of view of that day). Returns rows written. A table built
        with a different version of the feature set is dropped and rebuilt."""
        target_id, ids = view.target_id, builder.ids
        version = feature_set_version(ids, catalogue)
        table = self.table_name(target_id, set_name)
        meta = self._meta(table)
        rows = 0
        with self._conn() as c:
            if meta is not None and meta["set_version"] != version:
                c.execute(f"DROP TABLE IF EXISTS {table}")
                meta = None
            if meta is None:
                cols = ", ".join(f'"{i}" REAL' for i in ids)
                c.execute(f"CREATE TABLE IF NOT EXISTS {table} (start_utc TEXT PRIMARY KEY, delivery_date TEXT, {cols})")
                c.execute("INSERT OR REPLACE INTO feature_sets_meta VALUES (?,?,?,?,?)",
                          (table, target_id, set_name, version, json.dumps(ids)))
            for D in days:
                X = builder.build(view.at(D, history_days), D)
                X.insert(0, "delivery_date", D.isoformat())
                X.insert(0, "start_utc", X.index.strftime("%Y-%m-%dT%H:%M:%SZ"))
                c.execute(f"DELETE FROM {table} WHERE delivery_date=?", (D.isoformat(),))
                marks = ",".join("?" * (len(ids) + 2))
                c.executemany(f"INSERT INTO {table} VALUES ({marks})",
                              [tuple(None if (isinstance(v, float) and v != v) else v for v in r)
                               for r in X.itertuples(index=False, name=None)])
                rows += len(X)
        return rows

    def read(self, target_id: str, set_name: str, first: Optional[dt.date] = None, last: Optional[dt.date] = None) -> pd.DataFrame:
        table = self.table_name(target_id, set_name)
        if self._meta(table) is None:
            raise KeyError(f"no stored features for {target_id} / {set_name}")
        sql, args = f"SELECT * FROM {table}", []
        cond = []
        if first:
            cond.append("delivery_date>=?"); args.append(first.isoformat())
        if last:
            cond.append("delivery_date<=?"); args.append(last.isoformat())
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        with self._conn() as c:
            df = pd.read_sql_query(sql + " ORDER BY start_utc", c, params=args)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("start_utc"), utc=True))
        return df

    def version(self, target_id: str, set_name: str) -> Optional[str]:
        m = self._meta(self.table_name(target_id, set_name))
        return None if m is None else m["set_version"]
