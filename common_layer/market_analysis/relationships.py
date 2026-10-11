"""
relationships.py — how the markets relate: day-ahead, IDA1-3, continuous market, reserves, imbalance.

All series are on the hourly UTC grid of panel.py, so a comparison is always between the same hour.
A spread is `other - base` in EUR/MWh; the question for forecasting is how large and how predictable it is
once the day-ahead price is known.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from common_layer.market_analysis.panel import period_mask


def aligned_correlation(panel: pd.DataFrame, cols: List[str], min_overlap: int, differences: bool = False
                        ) -> Dict[str, pd.DataFrame]:
    """Pairwise correlations on hours where both series exist; pairs with too little overlap stay NaN.

    differences=True correlates hour-to-hour changes (removes shared slow level)."""
    x = panel[cols].diff() if differences else panel[cols]
    corr = pd.DataFrame(np.nan, index=cols, columns=cols)
    n = pd.DataFrame(0, index=cols, columns=cols, dtype=int)
    for a in cols:
        for b in cols:
            if a == b:
                n.loc[a, b] = int(x[a].notna().sum())
                corr.loc[a, b] = 1.0 if n.loc[a, b] >= min_overlap else np.nan
                continue
            both = x[[a, b]].dropna()
            n.loc[a, b] = len(both)
            if len(both) >= min_overlap and both[a].std() > 0 and both[b].std() > 0:
                corr.loc[a, b] = both[a].corr(both[b])
    return {"corr": corr, "n": n}


def spread_stats(panel: pd.DataFrame, base: str, other: str, periods: List[dict]) -> pd.DataFrame:
    """Spread `other - base` per analysis period: size, sign, link to the level, persistence."""
    rows = []
    for p in periods:
        m = period_mask(panel.index, p["start"], p["end"])
        d = panel.loc[m, [base, other]].dropna()
        if len(d) < 200:
            continue
        sp = d[other] - d[base]
        reg = stats.linregress(d[base], d[other])
        acf1 = sp.autocorr(1) if len(sp) > 3 else np.nan
        rows.append({"period": p["id"], "n_hours": len(d), "mean_spread": float(sp.mean()),
                     "std_spread": float(sp.std()), "mae": float(sp.abs().mean()),
                     "share_positive": float((sp > 0).mean()), "slope_on_base": float(reg.slope),
                     "r2_on_base": float(reg.rvalue ** 2), "spread_acf1": float(acf1),
                     "std_ratio_spread_to_price": float(sp.std() / d[other].std()) if d[other].std() else np.nan})
    return pd.DataFrame(rows)


def spread_by_hour(panel_cal: pd.DataFrame, base: str, other: str) -> pd.DataFrame:
    """Mean and mean-absolute spread by local hour."""
    d = panel_cal[[base, other, "local_hour"]].dropna()
    sp = (d[other] - d[base]).rename("spread")
    g = sp.groupby(d["local_hour"])
    return pd.DataFrame({"mean_spread": g.mean(), "mae": sp.abs().groupby(d["local_hour"]).mean(), "n": g.count()})


def reserve_links(panel: pd.DataFrame, base: str, others: List[str], min_overlap: int) -> pd.DataFrame:
    """Level and change correlation of each reserve / imbalance series with the day-ahead price."""
    rows = []
    for o in others:
        both = panel[[base, o]].dropna()
        if len(both) < min_overlap:
            continue
        d = panel[[base, o]].diff().dropna()
        rows.append({"series": o, "n_hours": len(both), "corr_level": float(both[base].corr(both[o])),
                     "corr_change": float(d[base].corr(d[o])) if len(d) > 10 else np.nan,
                     "mean": float(both[o].mean()), "std": float(both[o].std())})
    return pd.DataFrame(rows)
