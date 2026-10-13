"""
live.py — what the live pipeline calls to get a forecast from the registry champion (stage 10 of
docs/Model_Development_Plan.pdf).

`champion_prices(target, hours, delivery_date, base_prices)` returns {hour: price} for the requested hours, in the shape the
legacy forecasters return, or None when the legacy forecaster should be used. Which one is used is set per target in
config/live_models.yaml (legacy, auto, registry); the default is legacy. The champion is loaded through serving.load_champion,
so a file that does not match the registry, an artifact of another version or changed feature definitions are all refused.

The forecast is made from the as-of view of the delivery day: the data known at the gate, and the base price (for the spread
targets) that the caller passes in, because for a future day the panel has no rows yet. Nothing from the delivery day or later
is read.
"""
from __future__ import annotations

import datetime as dt
import os
import time
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.model_development import champions, serving

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "live_models.yaml")
MODES = ("legacy", "auto", "registry")
SWITCHABLE = ("da_price", "ida1_price", "ida2_price", "ida3_price", "xbid_price", "afrr_cap_price", "afrr_dn_price")
PRICE_FLOOR = -600.0
PANEL_TTL_SECONDS = 600
_panel_cache: dict = {}


def load_modes(path: Optional[str] = None) -> Dict[str, str]:
    """The mode of every switchable target (the LIVE_MODELS_MODE environment variable overrides all of them)."""
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        t = yaml.safe_load(fh)["targets"]
    bad = [k for k, v in t.items() if v["mode"] not in MODES]
    if bad or set(t) != set(SWITCHABLE):
        raise ValueError(f"live_models.yaml: modes must be one of {MODES} and targets exactly {SWITCHABLE} (bad: {bad})")
    env = os.environ.get("LIVE_MODELS_MODE")
    if env:
        if env not in MODES:
            raise ValueError(f"LIVE_MODELS_MODE must be one of {MODES}")
        return {k: env for k in t}
    return {k: v["mode"] for k, v in t.items()}


def default_panel() -> pd.DataFrame:
    """The hourly panel of the clean zone, rebuilt at most every PANEL_TTL_SECONDS (it takes a few seconds)."""
    now = time.time()
    if not _panel_cache or now - _panel_cache["at"] > PANEL_TTL_SECONDS:
        from common_layer.data_layer.clean_store import CleanStore
        from common_layer.market_analysis import panel as pnl
        _panel_cache.update(panel=pnl.build_panel(CleanStore(), curate=True), at=now)
    return _panel_cache["panel"]


def with_delivery_day(panel: pd.DataFrame, D: dt.date, base_col: Optional[str], base_prices: Optional[Dict[int, float]]) -> pd.DataFrame:
    """The panel plus empty rows for delivery day D (a future day has none), and the base price for D when it is given."""
    start = pd.Timestamp(D).tz_localize("Europe/Madrid")
    end = pd.Timestamp(D + dt.timedelta(days=1)).tz_localize("Europe/Madrid")
    idx = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    missing = idx.difference(panel.index)
    out = pd.concat([panel, pd.DataFrame(np.nan, index=missing, columns=panel.columns)]).sort_index() if len(missing) else panel.copy()
    if base_col and base_prices:
        for i, ts in enumerate(idx):
            h = i + 1
            if h in base_prices:
                out.loc[ts, base_col] = float(base_prices[h])
    return out


def champion_prices(target: str, hours: List[int], delivery_date: str, base_prices: Optional[Dict[int, float]] = None,
                    db_path: Optional[str] = None, mode: Optional[str] = None,
                    panel_provider: Callable[[], pd.DataFrame] = default_panel, form: Optional[dict] = None,
                    history_days: int = 400) -> Optional[Dict[int, float]]:
    """{hour: price} from the registry champion, or None if the legacy forecaster should be used (see the module header).

    `base_prices` ({hour: price}) is the day-ahead price of the delivery day: required for a spread target (the spread is added to it),
    and for a level target whose information set includes the day-ahead result of the day itself (the aFRR prices) it fills that input
    when the clean store does not have it yet. Without it such an input is missing and is filled with its training mean."""
    try:
        mode = mode or load_modes()[target]
    except Exception as exc:                                  # a broken config must never stop the pipeline
        warnings.warn(f"[champion {target}] live_models.yaml unusable ({exc}); using the legacy model", RuntimeWarning, stacklevel=2)
        return None
    if mode == "legacy":
        return None
    try:
        if champions.active(target, db_path) is None:
            raise serving.ChampionError(f"{target} has no active champion")
        if form is None:
            from common_layer.configuration.forecast_formulation import load_forecast_formulation
            form = load_forecast_formulation()
        D = dt.date.fromisoformat(delivery_date)
        base_col = None
        probe = AsOfView(panel_provider(), form, target)
        if probe.spec["kind"] == "spread":
            base_col = probe.spec["base"]
            if not base_prices:
                raise ValueError(f"{target} is a spread target: the base price of the delivery day is required")
        elif base_prices and probe.offsets.get("da_pt") == 0:        # a level target that uses the day-ahead price of the delivery day (aFRR)
            base_col = "da_pt"
        view = AsOfView(with_delivery_day(panel_provider(), D, base_col, base_prices), form, target)
        fc = serving.load_champion(target, view.interface(), db_path)
        v = view.at(D, history_days)
        prices = fc.forecast_prices(v, D)
        out = {i + 1: round(max(PRICE_FLOOR, float(p)), 2) for i, p in enumerate(prices.to_numpy())}
        if not out or not np.isfinite(list(out.values())).all():
            raise ValueError("the champion returned no usable forecast")
        return {h: out[h] for h in hours if h in out}
    except Exception as exc:
        if mode == "registry":
            raise
        warnings.warn(f"[champion {target}] not used ({exc}); using the legacy model", RuntimeWarning, stacklevel=3)
        return None
