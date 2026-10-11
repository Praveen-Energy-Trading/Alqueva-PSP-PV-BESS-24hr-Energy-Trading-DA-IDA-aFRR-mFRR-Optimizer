"""
provenance.py — every row is labelled live or synthetic; synthetic rows never enter training or testing.

Market data in the clean zone is live by construction (each row traces to a downloaded file via its
raw checksum). External series from static files (PV weather, inflow) are labelled by date from
config `provenance.external` (Phase 1 finding D1: real data end on 2026-06-30).
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd

from common_layer.data_quality.config import load_quality_config

LIVE, SYNTHETIC = "live", "synthetic"


def label(series: str, day: dt.date, cfg: Optional[dict] = None) -> str:
    cfg = cfg or load_quality_config()
    ext = cfg["provenance"]["external"].get(series)
    if ext is None:
        return cfg["provenance"]["market_default"]
    return LIVE if day <= dt.date.fromisoformat(ext["live_until"]) else ext["after"]


def label_frame(df: pd.DataFrame, series: str, date_col: str = "delivery_date",
                cfg: Optional[dict] = None) -> pd.DataFrame:
    """Add a `provenance` column. `date_col` holds ISO dates or dates."""
    cfg = cfg or load_quality_config()
    out = df.copy()
    out["provenance"] = [label(series, d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d)[:10]), cfg)
                         for d in out[date_col]]
    return out


def trainable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows allowed in training and testing: live only."""
    if "provenance" not in df.columns:
        raise KeyError("rows have no provenance label; call label_frame first")
    return df[df["provenance"] == LIVE]
