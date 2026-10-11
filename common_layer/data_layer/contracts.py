"""
contracts.py — what a valid table of each dataset must look like (data contracts).

A contract names, per dataset: key columns, value columns with ranges, and the rules checked on
every delivery day before the data is accepted into the clean zone:

    structure   all columns present, no duplicate keys                           -> error
    calendar    period count equals the true count for the day (23/25 h, 92/100)  -> error
    time        start_utc matches the calendar and is strictly increasing         -> error
    range       values inside plausibility bands                                  -> warning
    missing     NaN values (allowed only where the dataset says so)               -> warning / error

Errors reject the day (it is not written to the clean zone, the load log says why). Warnings are
kept as quality flags on the day's rows and in the load log. Ranges are plausibility bands, not
market limits: they catch unit mistakes and corrupted values, not unusual but real prices.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from common_layer.data_layer import calendar as cal
from common_layer.data_layer import regimes

Range = Tuple[float, float]


@dataclass(frozen=True)
class Dataset:
    name: str
    keys: Tuple[str, ...]                       # primary key in the clean store
    value_cols: Dict[str, Range]                # column -> plausibility band
    optional_cols: Tuple[str, ...] = ()         # NaN is expected/allowed
    text_cols: Tuple[str, ...] = ()             # carried as text (no range)
    int_cols: Tuple[str, ...] = ()
    max_lag_days: int = 2                       # freshness: latest delivery day vs today (+1 for D+1 DA)


_PRICE: Range = (-1000.0, 5000.0)
_ANY_PRICE: Range = (-10000.0, 10000.0)
_MW: Range = (0.0, 60000.0)

DATASETS: Dict[str, Dataset] = {
    "da_price": Dataset(
        "da_price", ("delivery_date", "period_index"),
        {"price_es_eur_mwh": _PRICE, "price_pt_eur_mwh": _PRICE,
         "buy_es_mw": _MW, "sell_es_mw": _MW, "buy_pt_mw": _MW, "sell_pt_mw": _MW,
         "total_iberian_mw": _MW, "total_with_bilaterals_mw": _MW,
         "flow_pt_to_es_mw": (0.0, 5000.0), "flow_es_to_pt_mw": (0.0, 5000.0)},
        optional_cols=("buy_es_mw", "sell_es_mw", "buy_pt_mw", "sell_pt_mw", "total_iberian_mw",
                       "total_with_bilaterals_mw", "flow_pt_to_es_mw", "flow_es_to_pt_mw"),
        text_cols=("issued_utc", "flags"), max_lag_days=1),
    "ida_price": Dataset(
        "ida_price", ("delivery_date", "session", "period_index"),
        {"price_es_eur_mwh": _ANY_PRICE, "price_pt_eur_mwh": _ANY_PRICE}, int_cols=("session",),
        max_lag_days=2),
    "xbid_price": Dataset(
        "xbid_price", ("delivery_date", "period_index"),
        {"price_pt_max_eur_mwh": _ANY_PRICE, "price_pt_min_eur_mwh": _ANY_PRICE,
         "price_pt_mean_eur_mwh": _ANY_PRICE, "price_es_mean_eur_mwh": _ANY_PRICE},
        optional_cols=("price_pt_max_eur_mwh", "price_pt_min_eur_mwh", "price_pt_mean_eur_mwh",
                       "price_es_mean_eur_mwh"),
        text_cols=("issued_utc",), int_cols=("pt_traded",), max_lag_days=2),
    "afrr_price": Dataset(
        "afrr_price", ("delivery_date", "period_index"),
        {"price_initial_up": (-1000.0, 3000.0), "price_initial_dn": (-1000.0, 3000.0),
         "price_adjusted_up": (-1000.0, 3000.0), "price_adjusted_dn": (-1000.0, 3000.0),
         "price_da_ref_eur_mwh": _PRICE},
        optional_cols=("price_initial_up", "price_initial_dn", "price_adjusted_up", "price_adjusted_dn"),
        max_lag_days=2),
    "mfrr_price": Dataset(
        "mfrr_price", ("delivery_date", "period_index"),
        {"price_programmed_activation": _ANY_PRICE, "price_direct_q0_up": _ANY_PRICE,
         "price_direct_q0_dn": _ANY_PRICE, "price_direct_q1_up": _ANY_PRICE,
         "price_direct_q1_dn": _ANY_PRICE, "price_da_ref_eur_mwh": _PRICE},
        optional_cols=("price_programmed_activation", "price_direct_q0_up", "price_direct_q0_dn",
                       "price_direct_q1_up", "price_direct_q1_dn"),
        max_lag_days=2),
    "imbalance_price": Dataset(
        "imbalance_price", ("delivery_date", "period_index"),
        {"price_short_eur_mwh": _ANY_PRICE, "price_long_eur_mwh": _ANY_PRICE,
         "price_da_ref_eur_mwh": _PRICE},
        optional_cols=(), max_lag_days=2),
}

_COMMON = ("delivery_date", "period_index", "resolution_min", "start_utc")

# Intraday auctions that deliver only part of the day: session -> hours covered, ending at midnight.
# Checked on real files (2025-10-25, 2025-10-26): IDA1 and IDA2 cover the full day, IDA3 the last 12 h.
SESSION_WINDOW_HOURS = {3: 12}


@dataclass
class Issue:
    severity: str          # "error" | "warning"
    rule: str
    message: str


@dataclass
class ContractResult:
    dataset: str
    delivery_date: str
    issues: List[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    def flags(self) -> str:
        return ";".join(sorted({i.rule for i in self.warnings}))


def validate_day(dataset: str, df: pd.DataFrame, day: dt.date) -> ContractResult:
    spec = DATASETS[dataset]
    res = ContractResult(dataset, day.isoformat())
    err = lambda rule, msg: res.issues.append(Issue("error", rule, msg))
    warn = lambda rule, msg: res.issues.append(Issue("warning", rule, msg))

    needed = list(_COMMON) + list(spec.value_cols) + list(spec.text_cols) + list(spec.int_cols)
    if dataset == "ida_price" and "session" not in df.columns:
        needed.append("session")
    missing = [c for c in needed if c not in df.columns]
    if missing:
        err("structure", f"missing columns {missing}")
        return res
    if df.empty:
        err("structure", "no rows")
        return res

    dup = df.duplicated(subset=list(spec.keys)).sum()
    if dup:
        err("structure", f"{int(dup)} duplicate key rows")

    # calendar and time, checked per session for the intraday auctions
    groups = [(None, df)] if "session" not in spec.keys else [(s, g) for s, g in df.groupby("session")]
    for sess, g in groups:
        tag = f" (session {sess})" if sess is not None else ""
        resolution = int(g["resolution_min"].iloc[0])
        full = cal.expected_periods(day, resolution)
        # IDA3 covers only the last 12 hours of the delivery day (periods 49-96, or 53-100 on the 25 h day)
        window_h = SESSION_WINDOW_HOURS.get(int(sess)) if sess is not None else None
        want = window_h * 60 // resolution if window_h else full
        first = full - want + 1
        wanted_idx = list(range(first, full + 1))
        if len(g) != want:
            err("calendar", f"{len(g)} periods{tag}, expected {want} at {resolution} min on {day}")
            continue
        if list(g["period_index"]) != wanted_idx:
            err("calendar", f"period_index{tag} is not {first}..{full}")
            continue
        expect = [cal.period_start_utc(day, i, resolution).strftime("%Y-%m-%dT%H:%M:%SZ")
                  for i in wanted_idx]
        if list(g["start_utc"]) != expect:
            err("time", f"start_utc{tag} does not follow the calendar")

    # resolution against the market regime (information, not an error: files can be revised)
    rg = regimes.regime_at(day)
    if dataset in ("da_price",) and int(df["resolution_min"].iloc[0]) != rg["da_resolution_min"]:
        warn("regime_resolution",
             f"DA resolution {int(df['resolution_min'].iloc[0])} min, regime says {rg['da_resolution_min']}")

    for col, (lo, hi) in spec.value_cols.items():
        s = pd.to_numeric(df[col], errors="coerce")
        n_nan = int(s.isna().sum())
        if n_nan:
            if col in spec.optional_cols:
                if n_nan == len(s):
                    warn("missing_all", f"{col} entirely missing")
            else:
                err("missing", f"{col}: {n_nan} missing values")
        bad = int(((s < lo) | (s > hi)).sum())
        if bad:
            warn("range", f"{col}: {bad} values outside [{lo}, {hi}]")
    if dataset == "da_price":
        floor = rg["price_floor_eur_mwh"]
        low = int((pd.to_numeric(df["price_pt_eur_mwh"], errors="coerce") < floor - 1e-9).sum())
        if low:
            warn("below_floor", f"{low} DA prices below the price floor {floor} in force")
    return res


def freshness_issues(latest_by_dataset: Dict[str, Optional[dt.date]], today: dt.date) -> List[Issue]:
    """Each dataset's newest delivery day against its allowed lag (day-ahead is published for D+1)."""
    out: List[Issue] = []
    for name, spec in DATASETS.items():
        latest = latest_by_dataset.get(name)
        if latest is None:
            out.append(Issue("error", "freshness", f"{name}: no data"))
            continue
        lag = (today - latest).days
        allowed = spec.max_lag_days - 1 if name == "da_price" else spec.max_lag_days
        if lag > allowed:
            out.append(Issue("warning", "freshness",
                             f"{name}: newest delivery day {latest} is {lag} days old (allowed {allowed})"))
    return out
