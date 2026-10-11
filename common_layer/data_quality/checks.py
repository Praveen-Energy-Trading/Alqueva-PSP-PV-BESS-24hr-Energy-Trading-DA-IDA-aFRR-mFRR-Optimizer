"""
checks.py — automated checks on the clean zone. Each check returns tidy "issue" rows:

    dataset, delivery_date, session (0 when the dataset has none), check, severity, value, detail

severity: "warning" (reduces the daily score) or "info" (counted, never penalised: spikes, negative
prices and prices at the market limit are information, not errors).
Nothing here changes or removes data.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml

from common_layer.data_layer import calendar as cal
from common_layer.data_layer import regimes
from common_layer.data_layer.contracts import DATASETS

COLUMNS = ["dataset", "delivery_date", "session", "check", "severity", "value", "detail"]
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _issue(ds, day, sess, check, severity, value=np.nan, detail="") -> dict:
    return {"dataset": ds, "delivery_date": day, "session": int(sess), "check": check,
            "severity": severity, "value": value, "detail": detail}


def _frame(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def _one_per_day(rows: List[dict]) -> pd.DataFrame:
    """Several price columns can raise the same check on the same day; the score needs one issue."""
    df = _frame(rows)
    return df.drop_duplicates(["dataset", "delivery_date", "session", "check"], keep="first").reset_index(drop=True)


def _with_session(df: pd.DataFrame) -> pd.DataFrame:
    if "session" not in df.columns:
        df = df.assign(session=0)
    return df


def _series_groups(df: pd.DataFrame):
    """Yield (session, frame sorted by start_utc with a UTC DatetimeIndex)."""
    df = _with_session(df)
    for sess, g in df.groupby("session"):
        g = g.sort_values("start_utc").copy()
        g.index = pd.to_datetime(g["start_utc"], utc=True)
        yield int(sess), g


# ----------------------------------------------------------------------- gaps
def check_gaps(ds: str, df: pd.DataFrame, session_sources: List[tuple], start: dt.date, end: dt.date,
               last_status: Dict[tuple, str]) -> pd.DataFrame:
    """Delivery days without rows between start and end (per session for the intraday auctions).

    `session_sources` is a list of (session, source) pairs (session 0 when the dataset has none).
    The cause comes from the load log: what the source answered the last time it was asked."""
    rows = []
    df = _with_session(df)
    have = {(int(s), d) for s, d in zip(df["session"], df["delivery_date"])}
    day = start
    while day <= end:
        for sess, source in session_sources:
            if (sess, day.isoformat()) not in have:
                cause = last_status.get((source, day.isoformat()), "not_attempted")
                rows.append(_issue(ds, day.isoformat(), sess, "missing_day", "warning", 1.0, cause))
        day += dt.timedelta(days=1)
    return _frame(rows)


# ---------------------------------------------------------- contract warnings
def check_contract_flags(ds: str, df: pd.DataFrame) -> pd.DataFrame:
    """Warnings the contract attached to the rows when the day was loaded."""
    df = _with_session(df)
    rows = []
    flagged = df[df["quality_flags"].fillna("") != ""].drop_duplicates(["delivery_date", "session"])
    for _, r in flagged.iterrows():
        for rule in str(r["quality_flags"]).split(";"):
            rows.append(_issue(ds, r["delivery_date"], r["session"], rule, "warning", 1.0, "from contract at load"))
    return _frame(rows)


# ------------------------------------------------------------------ flat runs
def check_flat_runs(ds: str, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """A whole day at one value on an auction price = a stuck feed. (Shorter runs are real: the
    Portuguese day-ahead price sat at 1.95 EUR/MWh for up to 9 hours on several days in 2020-2021.)"""
    hours = cfg["stuck"]["flat_run_hours"]
    if not cfg["datasets"][ds].get("stuck_checks"):
        return _frame([])
    rows = []
    for sess, g in _series_groups(df):
        for col in cfg["datasets"][ds]["price_cols"]:
            v = g[col]
            run_id = (v != v.shift()).cumsum()
            ok = v.notna()
            span_h = g["resolution_min"].where(ok, 0).groupby(run_id).transform("sum") / 60.0
            bad = ok & (span_h >= hours)
            for day in sorted(g.loc[bad, "delivery_date"].unique()):
                longest = float(span_h[bad & (g["delivery_date"] == day)].max())
                rows.append(_issue(ds, day, sess, "flat_run", "warning", longest, f"{col}: {longest:.0f} h identical"))
    return _one_per_day(rows)


# ---------------------------------------------------------- duplicate day
def check_duplicate_day(ds: str, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """A day whose prices equal the previous day's in almost every period = the same (stale) file served
    again. Only for auction prices; skipped when the day itself is constant (reported as flat_run)."""
    if not cfg["datasets"][ds].get("stuck_checks"):
        return _frame([])
    share_min = float(cfg["stuck"]["duplicate_day_share"])
    rows = []
    for sess, g in _series_groups(df):
        for col in cfg["datasets"][ds]["price_cols"]:
            days = {d: x.sort_values("period_index")[col].to_numpy(dtype=float) for d, x in g.groupby("delivery_date")}
            for d in sorted(days):
                prev = (dt.date.fromisoformat(d) - dt.timedelta(days=1)).isoformat()
                a, b = days[d], days.get(prev)
                if b is None or len(a) != len(b) or np.isnan(a).all() or np.nanstd(a) == 0:
                    continue
                share = float(np.mean(a == b))
                if share >= share_min:
                    rows.append(_issue(ds, d, sess, "duplicate_day", "warning", share,
                                       f"{col}: {share:.0%} of periods equal to {prev}"))
    return _one_per_day(rows)


# ------------------------------------------------------------------------ DST
def check_dst_copy(ds: str, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """25-hour quarter-hour day: the repeated hour must not be a plain copy of the first one (a typical
    artefact of tools that cannot represent two hour-3 slots). Four equal quarters in a row that are
    themselves not constant is real evidence; for hourly data one equal value is not, so it is skipped."""
    if not cfg.get("dst_copy", {}).get("enabled", True) or not cfg["datasets"][ds].get("stuck_checks"):
        return _frame([])
    rows = []
    cols = cfg["datasets"][ds]["price_cols"]
    for sess, g in _series_groups(df):
        for day, d in g.groupby("delivery_date"):
            if cal.hours_in_day(dt.date.fromisoformat(day)) != 25 or int(d["resolution_min"].iloc[0]) != 15:
                continue
            d = d.sort_values("period_index")
            first = d[(d["period_index"] >= 9) & (d["period_index"] <= 12)]
            second = d[(d["period_index"] >= 13) & (d["period_index"] <= 16)]
            if len(first) != 4 or len(second) != 4:
                continue                                  # partial session (IDA3) does not contain the hour
            for c in cols:
                a, b = first[c].to_numpy(dtype=float), second[c].to_numpy(dtype=float)
                if not (np.isnan(a).any() or np.isnan(b).any()) and np.ptp(a) > 0 and np.array_equal(a, b):
                    rows.append(_issue(ds, day, sess, "dst_copy_suspect", "warning", 1.0,
                                       f"{c}: repeated hour identical to the first one"))
                    break
    return _frame(rows)


# ------------------------------------------------- spikes, negatives, limits
def _bid_cap() -> float:
    with open(os.path.join(_REPO, "config", "market.yaml"), encoding="utf-8") as fh:
        return float(yaml.safe_load(fh)["bid_limits"]["price_max_eur_mwh"])


def check_value_flags(ds: str, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per day counts of spikes (trailing robust z-score), negative prices and prices at the limit.

    The spike reference uses ONLY earlier values, so the flag could be used as a model input."""
    sp = cfg["spikes"]
    rows = []
    cap = _bid_cap()
    for sess, g in _series_groups(df):
        for col in cfg["datasets"][ds]["price_cols"]:
            v = g[col].astype(float)
            roll = v.rolling(f"{int(sp['window_days'])}D", closed="left", min_periods=int(sp["min_history_periods"]))
            med = roll.median()
            sigma = (roll.quantile(0.75) - roll.quantile(0.25)) / 1.349
            dev = v - med
            z = dev.abs() / sigma.where(sigma > 1e-6)
            is_spike = (z > sp["robust_z"]) & (dev.abs() > sp["min_abs_deviation_eur"])
            flags = {"spike_up": is_spike & (dev > 0), "spike_down": is_spike & (dev < 0), "negative_price": v < 0}
            if ds == "da_price":
                tol = float(cfg["cap_tolerance_eur"])
                events = regimes.load_regimes()
                floor_of = {d: regimes.regime_at(dt.date.fromisoformat(d), events)["price_floor_eur_mwh"]
                            for d in g["delivery_date"].unique()}
                floor = g["delivery_date"].map(floor_of)
                flags["at_limit"] = (v >= cap - tol) | (v <= floor + tol)
            for name, mask in flags.items():
                counts = mask.groupby(g["delivery_date"]).sum()
                for day, n in counts[counts > 0].items():
                    rows.append(_issue(ds, day, sess, name, "info", float(n), col))
    return _frame(rows)


# ----------------------------------------------------------- cross-source
def check_cross_source(frames: Dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """The REN files carry the day-ahead price. It must equal the OMIE price in the same period;
    a systematic difference means a unit error, a shifted hour or a wrong day."""
    da = frames.get("da_price")
    if da is None or da.empty:
        return _frame([])
    tol = float(cfg["cross_source"]["da_reference_tolerance_eur"])
    share_max = float(cfg["cross_source"]["max_mismatch_share"])
    da_q = da.set_index("start_utc")["price_pt_eur_mwh"]
    da_h = da[da["resolution_min"] == 60].set_index("start_utc")["price_pt_eur_mwh"]
    rows = []
    for ds in ("afrr_price", "mfrr_price", "imbalance_price"):
        df = frames.get(ds)
        if df is None or df.empty:
            continue
        t = pd.to_datetime(df["start_utc"], utc=True)
        hour_key = t.dt.floor("h").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ref_da = df["start_utc"].map(da_q)
        ref_da = ref_da.where(ref_da.notna(), hour_key.map(da_h))
        cmp = pd.DataFrame({"day": df["delivery_date"], "diff": (df["price_da_ref_eur_mwh"] - ref_da).abs(),
                            "have": ref_da.notna() & df["price_da_ref_eur_mwh"].notna()})
        for day, g in cmp.groupby("day"):
            n = int(g["have"].sum())
            if n < 0.5 * len(g):
                rows.append(_issue(ds, day, 0, "cross_unmatched", "info", float(n), "less than half of the periods comparable"))
                continue
            share = float((g.loc[g["have"], "diff"] > tol).mean())
            if share > share_max:
                rows.append(_issue(ds, day, 0, "cross_mismatch", "warning", share,
                                   f"{share:.0%} of periods: REN day-ahead reference differs from OMIE"))
    return _frame(rows)


# --------------------------------------------------------------- level jump
def check_level_jump(ds: str, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Day median far from the trailing median: catches unit or scaling errors (EUR vs cEUR, MWh vs kWh)."""
    lj = cfg["level_jump"]
    rows = []
    for sess, g in _series_groups(df):
        for col in cfg["datasets"][ds]["price_cols"][:1]:       # one representative column is enough
            daily = g.groupby("delivery_date")[col].median()
            trail = daily.rolling(int(lj["trailing_days"]), min_periods=7).median().shift(1)
            ratio = (daily / trail).abs()
            big = (daily.abs() >= lj["min_abs_median_eur"]) & (trail.abs() >= lj["min_abs_median_eur"]) & (
                (ratio > lj["ratio"]) | (ratio < 1.0 / lj["ratio"]))
            for day in daily.index[big.fillna(False)]:
                rows.append(_issue(ds, day, sess, "level_jump", "warning", float(ratio[day]),
                                   f"{col}: day median {daily[day]:.1f} vs trailing {trail[day]:.1f}"))
    return _frame(rows)
