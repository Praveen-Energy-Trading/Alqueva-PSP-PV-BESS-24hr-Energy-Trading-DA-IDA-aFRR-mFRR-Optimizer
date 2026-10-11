"""
targets.py — build each target as the formulation defines it, and turn a forecast of it back into a price.

    frame = build_target(panel, spec)           columns: y_raw (level or spread), base, y_model (transformed)
    price = to_level(spec, y_model_forecast, base_values)

`spec` is the `modelled_quantity` block of a target in config/forecast_formulation.yaml, so this module
carries no decisions of its own. A spread target is `series - base`, both in the same unit; the forecast of
the level is `base + inverse_transform(forecast)`. The base price used when predicting must be the one the
as-of time allows (the day-ahead result for the intraday gates), which the information-set check enforces.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from common_layer.forecast_formulation.transforms import make_transform


def _transform(spec: dict, challenger: bool = False):
    tr = spec["challenger"] if challenger else spec["transform"]
    return make_transform(tr["name"], tr.get("scale"))


def build_target(panel: pd.DataFrame, spec: dict, challenger: bool = False) -> pd.DataFrame:
    """Target table on the panel's hourly UTC index. Rows where the series (or the base) is missing are dropped."""
    series = spec.get("series")
    if series is None:
        raise ValueError("this target has no price series in the panel (weather and inflow are built elsewhere)")
    kind = spec["kind"]
    cols = [series] + ([spec["base"]] if kind == "spread" else [])
    d = panel[cols].dropna()
    out = pd.DataFrame(index=d.index)
    if kind == "spread":
        out["base"] = d[spec["base"]]
        out["y_raw"] = d[series] - d[spec["base"]]
    else:
        out["base"] = 0.0
        out["y_raw"] = d[series]
    out["y_model"] = _transform(spec, challenger).forward(out["y_raw"].to_numpy())
    return out


def to_level(spec: dict, y_model_forecast, base_values, challenger: bool = False) -> np.ndarray:
    """Price forecast from a forecast in modelled (transformed) space. For a level target `base_values` is ignored."""
    raw = _transform(spec, challenger).inverse(np.asarray(y_model_forecast, dtype=float))
    if spec["kind"] == "spread":
        return np.asarray(base_values, dtype=float) + raw
    return raw
