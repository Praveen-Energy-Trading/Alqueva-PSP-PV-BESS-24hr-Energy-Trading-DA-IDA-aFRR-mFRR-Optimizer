"""
synthetic.py — artificial outcomes for locked-window days that have none (config/synthetic_fill.yaml).

A demonstration-project choice: where a target has no real outcome on a locked-window day, a made-up value is inserted so the
locked test can run. The filled days are returned alongside the panel so every report can say which days are not real prices.
Only the locked window is touched, only days with no value at all are filled, and a real value is never overwritten.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from common_layer.forecast_evaluation.asof import AsOfView

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "synthetic_fill.yaml")
METHODS = ("base_price",)


def load_fill_config(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    bad = {t: v for t, v in (cfg.get("targets") or {}).items() if v.get("method") not in METHODS}
    if bad:
        raise ValueError(f"synthetic_fill.yaml: method must be one of {METHODS} (bad: {list(bad)})")
    return cfg


def fill_locked(panel: pd.DataFrame, form: dict, target: str, eval_cfg: dict, fill_cfg: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]:
    """The panel with artificial outcomes on the locked-window days that have none, and a description of what was filled.

    The description always has `target`, `method` (None if the target is not filled), `days` (ISO dates) and `window`."""
    cfg = fill_cfg if fill_cfg is not None else load_fill_config()
    lk = eval_cfg["locked_test"]
    info: Dict = {"target": target, "method": None, "days": [], "window": [str(lk["start"]), str(lk["end"])]}
    spec = (cfg.get("targets") or {}).get(target)
    if spec is None:
        return panel, info
    view = AsOfView(panel, form, target)
    series, base = view.spec["series"], view.spec.get("base")
    if spec["method"] == "base_price" and (view.spec["kind"] != "spread" or not base):
        raise ValueError(f"{target}: base_price fill needs a spread target with a base price")
    info["method"] = spec["method"]
    local = panel.index.tz_convert("Europe/Madrid")
    day = np.array([d.toordinal() for d in local.date])
    lo, hi = lk["start"].toordinal(), lk["end"].toordinal()
    out = panel.copy()
    filled: List[int] = []
    for d in range(lo, hi + 1):
        rows = day == d
        if not rows.any() or out.loc[rows, series].notna().any():                 # no rows at all, or a real value exists: leave it
            continue
        b = out.loc[rows, base]
        if b.isna().all():                                                        # nothing to base a value on
            continue
        out.loc[rows, series] = b.to_numpy()
        filled.append(d)
    info["days"] = [str(dt.date.fromordinal(d)) for d in filled]
    return out, info


def real_only(pred: pd.DataFrame, info: Optional[dict]) -> pd.DataFrame:
    """A prediction table without the filled days."""
    if not info or not info["days"]:
        return pred
    ords = {dt.date.fromisoformat(d).toordinal() for d in info["days"]}
    return pred[~pred["local_date"].isin(ords)]
