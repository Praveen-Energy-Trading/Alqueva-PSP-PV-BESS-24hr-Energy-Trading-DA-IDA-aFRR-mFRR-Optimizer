"""
imputation.py — the written fill rule of each variable (config `imputation`), applied backwards in
time only: a value at time t may use observations from before t, never from after.

    none      leave missing (default; for auction prices a gap is an error, not something to fill)
    ffill     carry the last observed value forward, at most `max_periods` periods per gap
    prev_day  the observed value exactly 24 h earlier (UTC)

Every filled value is marked in a companion boolean column `<col>__imputed`, so the model can see
which numbers are real. Used when features are built (Phase 5); the clean zone is never filled.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from common_layer.data_quality.config import load_quality_config


def rule_for(cfg: dict, dataset: str, col: str) -> dict:
    return cfg["imputation"].get(f"{dataset}.{col}", cfg["imputation"]["default"])


def impute(df: pd.DataFrame, dataset: str, col: str, cfg: Optional[dict] = None) -> pd.DataFrame:
    """Return a copy of one series (single session, any order) with `col` filled by its rule."""
    cfg = cfg or load_quality_config()
    rule = rule_for(cfg, dataset, col)
    out = df.sort_values("start_utc").reset_index(drop=True).copy()
    observed = out[col].astype(float)
    filled = observed.copy()
    method = rule["method"]
    if method == "ffill":
        filled = observed.ffill(limit=int(rule["max_periods"]))
    elif method == "prev_day":
        t = pd.to_datetime(out["start_utc"], utc=True)
        earlier = pd.Series(observed.to_numpy(), index=t)
        lookup = earlier.reindex(t - pd.Timedelta(days=1)).to_numpy()
        filled = observed.where(observed.notna(), pd.Series(lookup, index=observed.index))
    out[col] = filled
    out[f"{col}__imputed"] = observed.isna() & filled.notna()
    return out
