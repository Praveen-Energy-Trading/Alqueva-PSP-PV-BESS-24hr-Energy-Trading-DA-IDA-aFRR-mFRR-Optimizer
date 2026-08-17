"""
ida2_price_forecaster.py — IDA2 intraday price forecast (gate closes D-1 22:00 CET).

Spread model trained on historical SIDC IDA2 clearing prices (H3-H24, 2020-2025).
Each gate has its own dedicated model and training data.

Training data: ida2_training_data_2024_2025.xlsx
    12,298 rows · H3-H24 · 2024-06-13 to 2025-12-31
    Source: OMIE/ENTSO-E SIDC IDA2 clearing prices.

Floor: -600 EUR/MWh (OMIE regulatory minimum, effective delivery 29 May 2026).
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

_PHASE1_FCST = os.path.join(_REPO, "phase_1_da_day_ahead_bidding",
                             "da_price_pv_inflow_forecasting")
if _PHASE1_FCST not in sys.path:
    sys.path.insert(0, _PHASE1_FCST)

from ml_train_val_test_common import fit_selected, mae as _mae, walk_forward_cv, MODEL_NAMES

_EXCEL_PATH  = os.path.join(_HERE, "ida2_training_data_2024_2025.xlsx")
_SHEET       = "IDA2_2024_2025"
_JSON_PATH   = os.path.join(_HERE, "ida2_selected_model.json")
_WARMUP_ROWS = 100
_N_CV_FOLDS  = 4
_PRICE_FLOOR = -600.0
_cache: dict = {}


def forecast_ida2_prices(hours: List[int], delivery_date: str,
                         da_prices: Dict[int, float]) -> Dict[int, float]:
    """Return {hour: EUR/MWh} IDA2 price forecast for delivery_date."""
    try:
        return _model_forecast(hours, delivery_date, da_prices)
    except Exception as exc:
        import warnings
        warnings.warn(
            f"[IDA2 Forecaster] Model failed ({exc}); using DA prices as fallback.",
            RuntimeWarning, stacklevel=2
        )
        return {h: da_prices.get(h, 55.0) for h in hours}


def _model_forecast(hours: List[int], delivery_date: str,
                    da_prices: Dict[int, float]) -> Dict[int, float]:
    df        = _load_history()
    target_dt = pd.Timestamp(delivery_date)
    train_raw = df[df["Date"] <= target_dt - pd.Timedelta(days=1)].copy()
    if len(train_raw) < _WARMUP_ROWS:
        raise ValueError(f"Insufficient IDA2 history ({len(train_raw)} rows)")

    feat_tr   = _build_features(train_raw).dropna()
    cache_key = f"ida2_{delivery_date}"

    if cache_key not in _cache:
        selected = _auto_select_model(feat_tr)
        fcols    = _feature_cols()
        X, y     = feat_tr[fcols], feat_tr["spread_EUR_MWh"].values
        model    = fit_selected(selected, X, y, fcols)
        _cache[cache_key]                   = model
        _cache[f"sel_ida2_{delivery_date}"] = selected

    model    = _cache[cache_key]
    selected = _cache[f"sel_ida2_{delivery_date}"]

    # Sequential (autoregressive) prediction — see ida1_price_forecaster.py for
    # the full explanation. spread_lag_h1 must be the model's own predicted
    # spread for hour h-1, not a static placeholder.
    fcols = _feature_cols()
    prices: Dict[int, float] = {}
    prev_spread = 0.0
    for h in sorted(hours):
        row = _build_pred_row(h, delivery_date, da_prices, feat_tr, prev_spread)
        spread = float(model.predict(row[fcols])[0])
        prices[h] = round(float(np.clip(da_prices.get(h, 55.0) + spread,
                                        _PRICE_FLOOR, 3_000.0)), 2)
        prev_spread = spread

    return prices


def _auto_select_model(feat_tr: pd.DataFrame) -> str:
    excel_last_date = feat_tr["Date"].max().date()
    if os.path.exists(_JSON_PATH):
        with open(_JSON_PATH) as f:
            info = json.load(f)
        if pd.Timestamp(info.get("data_end_date", "2000-01-01")).date() >= excel_last_date:
            return info["selected"]

    fcols    = _feature_cols()
    y        = feat_tr["spread_EUR_MWh"].values
    cv_mae   = walk_forward_cv(feat_tr[fcols], y, np.zeros_like(y), fcols, _N_CV_FOLDS)
    selected = min(cv_mae, key=cv_mae.get)

    with open(_JSON_PATH, "w") as f:
        json.dump({"selected": selected,
                   "cv_mae": {k: round(v, 4) for k, v in cv_mae.items()},
                   "data_end_date": str(excel_last_date),
                   "updated_on": str(datetime.date.today())}, f, indent=2)
    print(f"\n[IDA2 Forecaster] Selected => {selected}")
    for name in MODEL_NAMES:
        mark = " <--" if name == selected else ""
        print(f"  {name:<12} MAE {cv_mae.get(name, float('inf')):.4f} EUR/MWh{mark}")
    return selected


def _load_history() -> pd.DataFrame:
    mtime = os.path.getmtime(_EXCEL_PATH)
    if "ida2_hist" not in _cache or _cache.get("ida2_hist_mtime") != mtime:
        df = pd.read_excel(_EXCEL_PATH, sheet_name=_SHEET)
        df.columns = [c.strip() for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Date", "Hour"]).reset_index(drop=True)
        _cache["ida2_hist"]       = df
        _cache["ida2_hist_mtime"] = mtime
    return _cache["ida2_hist"]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["Date", "Hour", "price_DA_PT_EUR_MWh", "spread_EUR_MWh"]].copy()
    out = out.rename(columns={"Hour": "hour", "price_DA_PT_EUR_MWh": "da_price"})

    out["hour_sin"]   = np.sin(2 * np.pi * (out["hour"] - 1) / 24)
    out["hour_cos"]   = np.cos(2 * np.pi * (out["hour"] - 1) / 24)
    dow               = out["Date"].dt.dayofweek
    month             = out["Date"].dt.month
    out["dow_sin"]    = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"]    = np.cos(2 * np.pi * dow / 7)
    out["month_sin"]  = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"]  = np.cos(2 * np.pi * (month - 1) / 12)
    out["is_weekend"] = (dow >= 5).astype(int)

    out = out.sort_values(["Date", "hour"]).reset_index(drop=True)
    grp = out.groupby("Date", sort=False)
    out["da_roll_mean_24h"] = grp["da_price"].transform(
        lambda x: x.rolling(min(24, len(x)), min_periods=1).mean())
    out["da_roll_std_24h"]  = grp["da_price"].transform(
        lambda x: x.rolling(min(24, len(x)), min_periods=2).std().fillna(0.0))

    lag_df = out[["Date", "hour", "spread_EUR_MWh"]].copy()
    lag_df["Date"] = lag_df["Date"] + pd.Timedelta(days=7)
    lag_df = lag_df.rename(columns={"spread_EUR_MWh": "da_lag_168h_spread"})
    out = out.merge(lag_df, on=["Date", "hour"], how="left")
    out["spread_lag_h1"] = grp["spread_EUR_MWh"].transform(lambda x: x.shift(1))
    return out


def _build_pred_row(h: int, delivery_date: str, da_prices: Dict[int, float],
                    feat_tr: pd.DataFrame, prev_spread: float) -> pd.DataFrame:
    """Build ONE hour's feature row. spread_lag_h1 = prev_spread, the caller's
    own just-predicted spread for hour h-1 (0.0 only for the first hour)."""
    target_dt  = pd.Timestamp(delivery_date)
    dow, month = target_dt.dayofweek, target_dt.month
    week_ago   = (target_dt - pd.Timedelta(days=7)).date()
    recent     = (feat_tr[feat_tr["Date"].dt.date == week_ago]
                  .groupby("hour")["spread_EUR_MWh"].mean())
    da_p = float(da_prices.get(h, 55.0))
    row = {
        "hour"              : h,
        "da_price"          : da_p,
        "hour_sin"          : np.sin(2 * np.pi * (h - 1) / 24),
        "hour_cos"          : np.cos(2 * np.pi * (h - 1) / 24),
        "dow_sin"           : np.sin(2 * np.pi * dow / 7),
        "dow_cos"           : np.cos(2 * np.pi * dow / 7),
        "month_sin"         : np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos"         : np.cos(2 * np.pi * (month - 1) / 12),
        "is_weekend"        : int(dow >= 5),
        "da_roll_mean_24h"  : float(np.mean(list(da_prices.values()))),
        "da_roll_std_24h"   : float(np.std(list(da_prices.values()))),
        "da_lag_168h_spread": float(recent.get(h, 0.0)),
        "spread_lag_h1"     : prev_spread,
    }
    return pd.DataFrame([row])


def _feature_cols() -> List[str]:
    return [
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        "is_weekend",
        "da_price",
        "da_roll_mean_24h", "da_roll_std_24h",
        "da_lag_168h_spread",
        "spread_lag_h1",
    ]
