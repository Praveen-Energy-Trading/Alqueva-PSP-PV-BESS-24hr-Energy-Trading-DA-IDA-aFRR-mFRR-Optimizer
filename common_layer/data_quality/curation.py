"""
curation.py — clean and fill the hourly market data before it is used for forecasting (config/data_curation.yaml).

The clean zone keeps what the sources published. `curate` turns the hourly panel into a complete one:

    1. data errors are removed: a value outside its legal range; for auction prices also a day with one value for
       all 24 hours, a day that copies the day before, and a day whose level jumps by a factor of 20 or more
       (the thresholds of config/data_quality.yaml). Genuine spikes and negative prices are kept.
    2. every missing hour (a publication gap, an empty auction, a removed error) is filled by the rule of its series:
         auction / reserve / volume   a gap of up to 3 hours: a straight line between its neighbours;
                                      otherwise the same hour a week earlier, else yesterday, else two weeks earlier
         band                         the last value carried forward (up to 48 hours), then as above
         spread                       the day-ahead price plus the median spread of that hour over the last 28 days
                                      (an empty auction is then the day-ahead price)
       Nothing is filled before a series starts (those hours are not missing: the series did not exist yet), nor for hours an
       intraday auction does not trade by design, nor beyond the last hour of the panel. Every other hour is filled, including the
       days after a feed's last published value (config gaps.max_tail_hours can limit that; by default it does not).

Every filled value is flagged (`Curated.imputed`), every removed error too (`Curated.errors`), and `Curated.log` lists
each filled gap with the method used. Except for the straight line over a short gap, a value is only ever filled from
earlier values (tests/test_data_curation.py::test_only_the_straight_line_looks_at_a_value_after_the_gap).

Look-ahead: the straight line uses the first real value after a gap of 1-3 hours. A live run never has a later value, so it cannot matter there. In a
replay it matters only for a gap that ends at the cut between two days, where that value would lie beyond the cut: measured on the whole history
(2026-10-16), 6 hours of the mFRR series, none in any price series. The evaluation of the price models uses the panel without filling.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from common_layer.data_quality.config import load_quality_config

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "data_curation.yaml")
TZ = "Europe/Madrid"
RULES = ("auction", "spread", "band", "reserve", "volume")

STRAIGHT = "straight line between the neighbours"
CARRIED = "last value carried forward"
SPREAD = "day-ahead price plus the typical spread"
SEASONAL = {168: "same hour a week earlier", 24: "same hour yesterday", 336: "same hour two weeks earlier"}


def load_curation_config(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    problems = validate_curation_config(cfg)
    if problems:
        raise ValueError("; ".join(problems))
    return cfg


def validate_curation_config(cfg: dict) -> List[str]:
    out = []
    for k in ("gaps", "series"):
        if k not in cfg:
            out.append(f"section {k} missing")
    if out:
        return out
    g = cfg["gaps"]
    if int(g.get("interpolate_max_hours", -1)) < 0:
        out.append("gaps.interpolate_max_hours must be >= 0")
    if not g.get("seasonal_lags_hours") or any(int(x) < 1 for x in g["seasonal_lags_hours"]):
        out.append("gaps.seasonal_lags_hours must be a list of positive hours")
    for name, spec in cfg["series"].items():
        if spec.get("rule") not in RULES:
            out.append(f"{name}: rule must be one of {RULES}")
        lo, hi = (spec.get("range") or [None, None])
        if lo is None or hi is None or lo >= hi:
            out.append(f"{name}: range must be [low, high] with low < high")
        if spec.get("rule") == "spread" and spec.get("base") not in cfg["series"]:
            out.append(f"{name}: a spread series needs a base series that is configured")
    return out


@dataclass
class Curated:
    panel: pd.DataFrame          # the complete hourly panel
    imputed: pd.DataFrame        # True where a value was filled
    errors: pd.DataFrame         # True where a published value was removed as a data error
    log: pd.DataFrame            # one row per filled gap
    summary: pd.DataFrame        # one row per series


def _local_day(index: pd.DatetimeIndex) -> np.ndarray:
    return np.array([d.toordinal() for d in index.tz_convert(TZ).date], dtype=np.int64)


def remove_errors(s: pd.Series, spec: dict, qcfg: dict) -> pd.Series:
    """True for each published value that is a data error (see the module header)."""
    lo, hi = spec["range"]
    err = ((s < lo) | (s > hi)).fillna(False)
    if not spec.get("stuck"):
        return err
    day = pd.Series(_local_day(s.index), index=s.index)
    n = s.notna().groupby(day).sum()
    bad_days = set()
    flat = n[(n >= int(qcfg["stuck"]["flat_run_hours"])) & (s.groupby(day).nunique() == 1)].index
    bad_days |= set(flat)
    same = ((s == s.shift(24)) & s.notna() & s.shift(24).notna()).groupby(day).sum()
    dup = (same / n.where(n > 0)).dropna()
    bad_days |= set(dup[(dup >= float(qcfg["stuck"]["duplicate_day_share"])) & (n.reindex(dup.index) >= 20)].index)
    jump = qcfg["level_jump"]
    med = s.groupby(day).median()
    trail = med.shift(1).rolling(int(jump["trailing_days"]), min_periods=10).median()
    ok = trail.abs() >= float(jump["min_abs_median_eur"])
    ratio = (med.abs() / trail.abs()).where(ok)
    bad_days |= set(ratio[ratio >= float(jump["ratio"])].index)          # only a jump UP: a day near zero is a real windy-holiday price, not an error
    return err | day.isin(bad_days)


def _gap_runs(mask: np.ndarray) -> List[tuple]:
    """(start, end) positions (inclusive) of each run of True."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def covered_hours(s: pd.Series, min_share: float = 0.5) -> np.ndarray:
    """Boolean per position: is this local hour of the day one that the series normally covers? An intraday auction that only
    trades the afternoon hours has no value for the morning hours by design; that is not a gap and is never filled."""
    first = s.first_valid_index()
    if first is None:
        return np.zeros(len(s), dtype=bool)
    hour = s.index.tz_convert(TZ).hour
    share = s[s.index >= first].notna().groupby(hour[s.index >= first]).mean()
    return np.isin(hour, share[share >= min_share].index.to_numpy())


def fill_series(s: pd.Series, spec: dict, gaps: dict, base: Optional[pd.Series] = None) -> tuple:
    """(filled series, how: array of the method used per position or '') for one hourly series on a full grid."""
    v = s.to_numpy(dtype=float).copy()
    how = np.full(len(v), "", dtype=object)
    first = int(np.argmax(~np.isnan(v))) if (~np.isnan(v)).any() else len(v)
    allowed = covered_hours(s)
    last = int(len(v) - 1 - np.argmax(~np.isnan(v)[::-1])) if (~np.isnan(v)).any() else -1
    tail = gaps.get("max_tail_hours")
    if tail is not None:                                                            # an optional limit on how far past the last published value to fill
        allowed[last + 1 + int(tail):] = False
    todo = np.isnan(v) & allowed
    todo[:first] = False                                          # nothing before the series starts, and no hours the series does not cover
    rule = spec["rule"]

    if rule == "spread" and base is not None:
        b = base.to_numpy(dtype=float)
        spread = v - b
        days = int(gaps["spread_history_days"])
        for i in np.flatnonzero(todo & ~np.isnan(b)):
            hist = spread[[i - 24 * k for k in range(1, days + 1) if i - 24 * k >= 0]]
            hist = hist[~np.isnan(hist)]
            v[i] = b[i] + (float(np.median(hist)) if len(hist) else 0.0)
            how[i] = SPREAD
            spread[i] = v[i] - b[i]
        todo = np.isnan(v) & allowed
        todo[:first] = False

    if rule == "band":
        limit = int(gaps["band_ffill_max_hours"])
        carried = pd.Series(v).ffill(limit=limit).to_numpy()
        new = todo & ~np.isnan(carried)
        v[new], how[new] = carried[new], CARRIED
        todo = np.isnan(v) & allowed
        todo[:first] = False

    n_max = int(gaps["interpolate_max_hours"])
    for a, b_ in _gap_runs(todo):
        if b_ - a + 1 <= n_max and a > 0 and b_ < len(v) - 1 and not np.isnan(v[a - 1]) and not np.isnan(v[b_ + 1]):
            v[a:b_ + 1] = np.interp(np.arange(a, b_ + 1), [a - 1, b_ + 1], [v[a - 1], v[b_ + 1]])
            how[a:b_ + 1] = STRAIGHT
    todo = np.isnan(v) & allowed
    todo[:first] = False

    lags = [int(x) for x in gaps["seasonal_lags_hours"]]
    for i in np.flatnonzero(todo):                                # earlier values only, so a long gap repeats the weekly pattern
        for lag in lags:
            j = i - lag
            if j >= 0 and not np.isnan(v[j]):
                v[i], how[i] = v[j], SEASONAL.get(lag, f"{lag} hours earlier")
                break
    left = np.isnan(v) & allowed
    left[:first] = False
    if left.any():                                                # nothing a week earlier (the first weeks of a series): the last value so far
        carried = pd.Series(v).ffill().to_numpy()
        new = left & ~np.isnan(carried)
        v[new], how[new] = carried[new], CARRIED
    return pd.Series(v, index=s.index, name=s.name), how


def curate(panel: pd.DataFrame, cfg: Optional[dict] = None, qcfg: Optional[dict] = None) -> Curated:
    """Clean and fill an hourly UTC panel (columns as in market_analysis.panel.SERIES). Columns without a rule are left as they are."""
    cfg = cfg or load_curation_config()
    qcfg = qcfg or load_quality_config()
    if panel.empty:
        e = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=bool)
        return Curated(panel.copy(), e, e.copy(), pd.DataFrame(), pd.DataFrame())
    grid = pd.date_range(panel.index.min(), panel.index.max(), freq="h", tz="UTC")
    full = panel.reindex(grid)
    out, imputed, errors, logs, rows = full.copy(), pd.DataFrame(False, index=grid, columns=full.columns), pd.DataFrame(False, index=grid, columns=full.columns), [], []
    names = [c for c in full.columns if c in cfg["series"]]
    names.sort(key=lambda c: cfg["series"][c]["rule"] == "spread")           # the base series are filled before the spreads
    for name in names:
        spec = cfg["series"][name]
        s = full[name]
        err = remove_errors(s, spec, qcfg)
        clean = s.where(~err)
        base = out[spec["base"]] if spec["rule"] == "spread" else None
        filled, how = fill_series(clean, spec, cfg["gaps"], base)
        flag = how != ""
        out[name] = filled
        imputed[name] = flag
        errors[name] = err.to_numpy() & s.notna().to_numpy()
        for a, b_ in _gap_runs(flag):
            for method in dict.fromkeys(how[a:b_ + 1]):
                pos = np.flatnonzero(how[a:b_ + 1] == method) + a
                logs.append({"series": name, "start_utc": grid[pos[0]], "end_utc": grid[pos[-1]], "hours": int(len(pos)), "method": method,
                             "reason": "data error removed" if errors[name].to_numpy()[pos].any() else "missing"})
        first = s.first_valid_index()
        n_obs = int(s.notna().sum())
        n_hours = int((grid >= first).sum()) if first is not None else 0
        counts = pd.Series(how[flag]).value_counts()
        rows.append({"series": name, "hours": n_hours, "published": n_obs, "errors_removed": int(errors[name].sum()), "filled": int(flag.sum()),
                     "filled_share": float(flag.sum() / n_hours) if n_hours else 0.0,
                     "straight_line": int(counts.get(STRAIGHT, 0)), "carried_forward": int(counts.get(CARRIED, 0)),
                     "day_ahead_plus_spread": int(counts.get(SPREAD, 0)),
                     "same_hour_earlier": int(sum(c for m, c in counts.items() if m in SEASONAL.values() or str(m).endswith("hours earlier")))})
    log = pd.DataFrame(logs, columns=["series", "start_utc", "end_utc", "hours", "method", "reason"])
    return Curated(out, imputed, errors, log, pd.DataFrame(rows))


def write_outputs(cur: Curated, out_dir: str) -> Dict[str, str]:
    """summary.csv and gaps.csv under out_dir (what was cleaned and filled, for the report and the dashboard)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {"summary": os.path.join(out_dir, "curation_summary.csv"), "gaps": os.path.join(out_dir, "curation_gaps.csv")}
    cur.summary.to_csv(paths["summary"], index=False, float_format="%.4f")
    cur.log.to_csv(paths["gaps"], index=False)
    return paths
