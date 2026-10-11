"""
breaks.py — structural breaks, two ways.

1. Event tests: for each documented market event, compare the `window_days` before and after on the daily
   mean price: shift in level (OLS on a step dummy, Newey-West standard error because daily prices are
   autocorrelated) and change in variance (Levene test). A significant shift next to an event is evidence the
   event mattered; it does not prove the event caused it (other things move at the same time).

   Prices also shift around the same calendar dates every year (season, hydrology), so each event shift is
   compared with the shifts at the same date in the other years (placebo years, `z_vs_placebo`).

2. Data-driven changepoints: binary segmentation of the daily mean for shifts in level. It finds the dates the
   data itself would split at, in the order they are found, with the share of the total variance each split
   removes at that step ("gain share"; later splits can remove more than earlier ones when the series has
   several steps), and each is matched to the nearest documented event. Splits are limited to `max_data_breaks` and a minimum
   segment length; the gain share, not a p-value, says how much a split matters (serial correlation makes
   naive p-values far too small).
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def event_test(daily: pd.Series, date: dt.date, window: int, min_side: int) -> Dict[str, float]:
    """Level and variance change around one date. NaN fields when either side is too short."""
    t = pd.Timestamp(date)
    before = daily[(daily.index >= t - pd.Timedelta(days=window)) & (daily.index < t)].dropna()
    after = daily[(daily.index >= t) & (daily.index < t + pd.Timedelta(days=window))].dropna()
    out = {"n_before": len(before), "n_after": len(after), "mean_before": np.nan, "mean_after": np.nan,
           "shift": np.nan, "shift_pct": np.nan, "t_nw": np.nan, "p_level": np.nan,
           "std_before": np.nan, "std_after": np.nan, "p_variance": np.nan}
    if len(before) < min_side or len(after) < min_side:
        return out
    y = np.concatenate([before.to_numpy(), after.to_numpy()])
    step = np.concatenate([np.zeros(len(before)), np.ones(len(after))])
    fit = sm.OLS(y, sm.add_constant(step)).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    out.update(mean_before=float(before.mean()), mean_after=float(after.mean()), shift=float(fit.params[1]),
               shift_pct=float(fit.params[1] / abs(before.mean()) * 100.0) if before.mean() else np.nan,
               t_nw=float(fit.tvalues[1]), p_level=float(fit.pvalues[1]),
               std_before=float(before.std()), std_after=float(after.std()),
               p_variance=float(stats.levene(before, after, center="median").pvalue))
    return out


def placebo_shifts(daily: pd.Series, date: dt.date, window: int, min_side: int) -> np.ndarray:
    """Level shifts at the same calendar date in every other year of the data. Seasonality and weather
    move prices around the same dates every year, so these show what a 'normal' shift looks like there.
    (Windows may contain other events; with only a few years this is a rough yardstick, not a null model.)"""
    out = []
    for y in sorted(set(daily.index.year)):
        if y == date.year:
            continue
        try:
            d = date.replace(year=y)
        except ValueError:                    # 29 February
            d = date.replace(year=y, day=28)
        s = event_test(daily, d, window, min_side)["shift"]
        if not np.isnan(s):
            out.append(s)
    return np.asarray(out)


def event_tests(daily: pd.Series, events: List[dict], window: int, min_side: int) -> pd.DataFrame:
    """One row per event: the level and variance change, plus where that shift sits among the shifts at the
    same calendar date in other years (z_vs_placebo; placebo_rank = share of placebo |shifts| below the event's)."""
    rows = []
    for e in events:
        r = {"event": e["id"], "date": e["date"], "status": e.get("status", "")}
        r.update(event_test(daily, e["date"], window, min_side))
        pl = placebo_shifts(daily, e["date"], window, min_side)
        r["n_placebo"] = len(pl)
        r["placebo_median_shift"] = float(np.median(pl)) if len(pl) else np.nan
        r["placebo_std_shift"] = float(np.std(pl, ddof=1)) if len(pl) >= 3 else np.nan
        r["z_vs_placebo"] = ((r["shift"] - r["placebo_median_shift"]) / r["placebo_std_shift"]
                             if len(pl) >= 3 and r["placebo_std_shift"] > 0 and not np.isnan(r["shift"]) else np.nan)
        r["placebo_rank"] = float(np.mean(np.abs(pl) < abs(r["shift"]))) if len(pl) and not np.isnan(r["shift"]) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _best_split(cs: np.ndarray, cs2: np.ndarray, lo: int, hi: int, min_size: int) -> Tuple[int, float]:
    """Best split index in [lo, hi) of the SSE-minimising two-segment fit; returns (index, SSE reduction)."""
    n = hi - lo
    if n < 2 * min_size:
        return -1, 0.0
    idx = np.arange(lo + min_size, hi - min_size + 1)
    s_all = cs[hi] - cs[lo]
    sse_all = (cs2[hi] - cs2[lo]) - s_all ** 2 / n
    n1 = idx - lo
    s1 = cs[idx] - cs[lo]
    s2 = s_all - s1
    n2 = hi - idx
    sse_split = (cs2[idx] - cs2[lo]) - s1 ** 2 / n1 + (cs2[hi] - cs2[idx]) - s2 ** 2 / n2
    k = int(np.argmin(sse_split))
    return int(idx[k]), float(sse_all - sse_split[k])


def binary_segmentation(x: np.ndarray, max_breaks: int, min_size: int) -> List[Tuple[int, float]]:
    """Up to `max_breaks` mean-shift split points of x as (index, gain share), strongest first.

    gain share = SSE removed by that split (given earlier splits) / total SSE around the overall mean."""
    x = np.asarray(x, dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x ** 2)])
    total = float(cs2[-1] - cs[-1] ** 2 / len(x))
    segs = [(0, len(x))]
    found: List[Tuple[int, float]] = []
    for _ in range(max_breaks):
        best = None
        for lo, hi in segs:
            i, gain = _best_split(cs, cs2, lo, hi, min_size)
            if i > 0 and (best is None or gain > best[2]):
                best = (lo, hi, gain, i)
        if best is None or total <= 0:
            break
        lo, hi, gain, i = best
        found.append((i, gain / total))
        segs.remove((lo, hi))
        segs += [(lo, i), (i, hi)]
    return found


def data_driven_breaks(daily: pd.Series, events: List[dict], max_breaks: int, min_size: int,
                       tolerance_days: int) -> pd.DataFrame:
    """Binary-segmentation breaks of the daily mean, each matched to the nearest documented event."""
    d = daily.dropna()
    found = binary_segmentation(d.to_numpy(), max_breaks, min_size)
    rows = []
    for rank, (i, share) in enumerate(found, 1):
        day = d.index[i]
        nearest = min(events, key=lambda e: abs((pd.Timestamp(e["date"]) - day).days))
        gap = (day - pd.Timestamp(nearest["date"])).days
        left = d.iloc[max(0, i - 60):i].mean()
        right = d.iloc[i:i + 60].mean()
        rows.append({"rank": rank, "break_date": day.date(), "gain_share": share, "mean_60d_before": float(left),
                     "mean_60d_after": float(right), "nearest_event": nearest["id"], "days_from_event": int(gap),
                     "matches_event": abs(gap) <= tolerance_days})
    return pd.DataFrame(rows)
