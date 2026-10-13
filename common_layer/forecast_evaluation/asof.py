"""
asof.py — the as-of view: what a forecaster is allowed to see when it forecasts delivery day D.

The information set of every target is written in config/forecast_formulation.yaml: for each input series,
the latest delivery day it may use (`latest_day_offset`, e.g. -1 = up to D-1). `AsOfView.at(D)` returns the panel
with every value beyond that boundary replaced by NaN and every series that is not in the information set
removed entirely. Forecasters receive only this view (and labels of days already known at fit time), never the
raw panel, so a forecaster cannot use information its gate would not have. The leakage tests in guards.py
confirm that changing data beyond the boundary changes no prediction.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TZ = "Europe/Madrid"

# panel column -> information-set series in forecast_formulation.yaml
PANEL_TO_INPUT: Dict[str, str] = {
    "da_pt": "da_result", "ida1_pt": "ida1_result", "ida2_pt": "ida2_result",
    "ida3_pt": "ida3_result", "xbid_pt": "xbid_summary", "afrr_up": "afrr_band_result",
    "afrr_dn": "afrr_band_result", "mfrr_ap": "mfrr_series", "imb_short": "imbalance_price",
    "imb_long": "imbalance_price",
    "da_vol_pt_buy": "da_result", "da_vol_pt_sell": "da_result",
}


def target_spec(formulation: dict, target_id: str) -> dict:
    return next(t for t in formulation["targets"] if t["id"] == target_id)


def allowed_offsets(formulation: dict, target_id: str) -> Dict[str, int]:
    """panel column -> latest allowed delivery-day offset (<= 0) for this target. Inputs that only become known at
    a later XBID window (`from_window` after W1) are left out, so the view is valid at the earliest window; when an
    input is listed more than once the most restrictive entry applies."""
    t = target_spec(formulation, target_id)
    by_series: Dict[str, int] = {}
    for inp in t["inputs"]:
        if inp.get("from_window") not in (None, "W1"):
            continue
        if inp["series"] in by_series:
            by_series[inp["series"]] = min(by_series[inp["series"]], inp["latest_day_offset"])
        else:
            by_series[inp["series"]] = inp["latest_day_offset"]
    return {col: by_series[s] for col, s in PANEL_TO_INPUT.items() if s in by_series}


class AsOfView:
    def __init__(self, panel: pd.DataFrame, formulation: dict, target_id: str):
        self.target_id = target_id
        self.spec = target_spec(formulation, target_id)["modelled_quantity"]
        self.offsets = allowed_offsets(formulation, target_id)
        cols = [c for c in panel.columns if c in PANEL_TO_INPUT]
        self._truth = panel[cols].copy()
        local = panel.index.tz_convert(TZ)
        self._ord = np.array([d.toordinal() for d in local.date], dtype=np.int64)
        self._hour = local.hour.to_numpy()
        for need in (self.spec.get("series"), self.spec.get("base")):
            if need and need not in self.offsets:
                raise ValueError(f"{target_id}: series {need} is not in the information set, so it cannot be a target or base")
        base = self.spec.get("base")
        if base and self.offsets[base] != 0:
            raise ValueError(f"{target_id}: the base price {base} must be known for day D itself (offset 0), "
                             f"found offset {self.offsets[base]}")

    # -------------------------------------------------------------------- slices
    def _slice(self, lo_ord: int, hi_ord: int) -> slice:
        return slice(int(np.searchsorted(self._ord, lo_ord, side="left")), int(np.searchsorted(self._ord, hi_ord, side="right")))

    def truth_day(self, D: dt.date) -> pd.DataFrame:
        """Unmasked rows of delivery day D (labels and evaluation only; never given to a forecaster at predict time)."""
        s = self._slice(D.toordinal(), D.toordinal())
        out = self._truth.iloc[s].copy()
        out["local_date"] = self._ord[s]
        out["local_hour"] = self._hour[s]
        return out

    def at(self, D: dt.date, history_days: int = 400, masked: bool = True) -> pd.DataFrame:
        """The as-of view for day D: rows from D-history_days to D, values beyond each series' boundary masked.
        `masked=False` returns the same rows unmasked; it exists only for the point-in-time tests of the features."""
        d0 = D.toordinal()
        s = self._slice(d0 - history_days, d0)
        out = self._truth.iloc[s].copy()
        od = self._ord[s]
        for col in (out.columns if masked else []):
            off = self.offsets.get(col)
            if off is None:
                out[col] = np.nan
            else:
                out.loc[od > d0 + off, col] = np.nan
        out["local_date"] = od
        out["local_hour"] = self._hour[s]
        return out

    # ------------------------------------------------------------ modelled quantity
    def modelled(self, frame: pd.DataFrame) -> pd.Series:
        """The modelled quantity in price units (level, or series minus base) from any frame of this view."""
        if self.spec["kind"] == "spread":
            return frame[self.spec["series"]] - frame[self.spec["base"]]
        return frame[self.spec["series"]]

    def truth_matrix(self, first: dt.date, last: dt.date) -> pd.DataFrame:
        """Unmasked day x hour matrix of the modelled quantity for [first, last] (training-window statistics only)."""
        s = self._slice(first.toordinal(), last.toordinal())
        f = self._truth.iloc[s].copy()
        f["local_date"] = self._ord[s]
        f["local_hour"] = self._hour[s]
        return matrix(f, self.modelled(f))

    def first_day(self) -> dt.date:
        return dt.date.fromordinal(int(self._ord[0]))

    def last_day(self) -> dt.date:
        return dt.date.fromordinal(int(self._ord[-1]))

    def interface(self) -> "ModelInterface":
        return ModelInterface(self)

    def known_through(self, D: dt.date, col: str) -> Optional[dt.date]:
        off = self.offsets.get(col)
        return None if off is None else D + dt.timedelta(days=off)


class ModelInterface:
    """What a forecaster may know about its target: the definition of the modelled quantity, never any data.
    (Data reaches a forecaster only as masked as-of views through FitContext and predict().)"""

    def __init__(self, view: "AsOfView"):
        self.target_id = view.target_id
        self.spec = view.spec
        self.offsets = dict(view.offsets)
        self._modelled = view.modelled

    def modelled(self, frame: pd.DataFrame) -> pd.Series:
        return self._modelled(frame)


def matrix(frame: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """days (ordinal) x local hour (0..23) matrix of `values`; the first value wins on the repeated autumn hour."""
    df = pd.DataFrame({"d": frame["local_date"].to_numpy(), "h": frame["local_hour"].to_numpy(), "v": values.to_numpy()})
    m = df.pivot_table(index="d", columns="h", values="v", aggfunc="first", dropna=False)
    return m.reindex(columns=range(24))
