"""
price_forecast_validator.py — forecast accuracy metrics (MAE / RMSE).

Generic error metrics between a forecast series and the realised series, used for
DA price validation and (reused) PV validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict


@dataclass
class ErrorMetrics:
    mae: float
    rmse: float
    n: int


def error_metrics(forecast: Dict[int, float], actual: Dict[int, float]) -> ErrorMetrics:
    keys = sorted(set(forecast) & set(actual))
    if not keys:
        return ErrorMetrics(0.0, 0.0, 0)
    abs_err = [abs(forecast[k] - actual[k]) for k in keys]
    sq_err = [(forecast[k] - actual[k]) ** 2 for k in keys]
    mae = sum(abs_err) / len(keys)
    rmse = math.sqrt(sum(sq_err) / len(keys))
    return ErrorMetrics(mae=mae, rmse=rmse, n=len(keys))
