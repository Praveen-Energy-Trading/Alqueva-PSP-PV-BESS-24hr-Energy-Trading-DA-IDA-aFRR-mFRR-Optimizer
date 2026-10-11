"""
run.py — run every Phase 4 analysis on the clean zone and write the tables (CSV) and figures (PNG).

    tables = run_analysis(CleanStore())            # compute only
    write_outputs(tables, panel, out_dir, fig_dir)  # CSV + PNG

The main series is the Portuguese day-ahead price (`da_pt`, hourly; the mean of four quarters from
2025-10-01). Everything that needs other markets uses whatever hours overlap; a table that cannot be built
because a market has too little data is simply absent.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from common_layer.data_layer.clean_store import CleanStore
from common_layer.data_quality.config import load_quality_config
from common_layer.market_analysis import (breaks, dependence, extremes, panel as pnl, relationships, sample_size,
                                          seasonality)
from common_layer.market_analysis.config import events, load_analysis_config

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_OUT = os.path.join(_REPO, "runtime", "analysis", "phase4")
DEFAULT_FIG = os.path.join(_REPO, "docs", "forecasting", "figures")

PRIMARY = "da_pt"
OTHERS = ["ida1_pt", "ida2_pt", "ida3_pt", "xbid_pt", "imb_short", "imb_long", "afrr_up", "afrr_dn", "mfrr_ap"]


def coverage(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in pnl.SERIES:
        s = p[c].dropna() if c in p else pd.Series(dtype=float)
        if s.empty:
            rows.append({"series": c, "first": None, "last": None, "hours": 0, "share_of_span": np.nan})
            continue
        span = int((s.index.max() - s.index.min()).total_seconds() // 3600) + 1
        rows.append({"series": c, "first": s.index.min().date(), "last": s.index.max().date(), "hours": len(s),
                     "share_of_span": len(s) / span})
    return pd.DataFrame(rows)


def run_analysis(clean: CleanStore, cfg: Optional[dict] = None) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Returns (tables, calendar-enriched hourly panel, scalar results)."""
    cfg = cfg or load_analysis_config()
    spikes = load_quality_config()["spikes"]
    raw = pnl.build_panel(clean)
    p = pnl.add_calendar(raw)
    da = raw[PRIMARY].dropna()
    daily = pnl.daily_mean(raw[PRIMARY])
    ev = events(cfg)
    periods = cfg["periods"]
    T: Dict[str, pd.DataFrame] = {}
    scalars: dict = {}

    T["coverage"] = coverage(raw)

    # --- seasonality
    T["variance_shares_all"] = seasonality.variance_shares(p, PRIMARY).assign(period="all")
    parts = []
    for per in periods:
        m = pnl.period_mask(p.index, per["start"], per["end"])
        if m.sum() > 24 * 60:
            parts.append(seasonality.variance_shares(p[m], PRIMARY).assign(period=per["id"]))
    T["variance_shares_by_period"] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    T["profile_hour"] = seasonality.profile(p, PRIMARY, "local_hour")
    T["profile_dow"] = seasonality.profile(p, PRIMARY, "dow")
    T["profile_month"] = seasonality.profile(p, PRIMARY, "month")
    T["weekend_effect"] = seasonality.weekend_effect(p, PRIMARY)
    T["annual_shape"] = seasonality.annual_shape(p, PRIMARY)
    scalars["annual_share_within_year"] = seasonality.annual_share(p, PRIMARY)

    # --- dependence and stationarity
    resid = dependence.deseasonalise(p, PRIMARY)
    scalars["resid_std"] = float(resid.std())
    for name, tab in dependence.dependence_summary(da, daily, resid, cfg).items():
        T[name] = tab
    scalars["arch_lm_residual"] = dependence.arch_lm(resid.iloc[-24 * 365 * 3:], 24)
    rows = [dependence.stationarity(daily, "daily mean, all years"),
            dependence.stationarity(daily.diff().dropna(), "daily mean, first difference, all years")]
    for per in periods:
        m = pnl.period_mask(daily.index, per["start"], per["end"])
        rows.append(dependence.stationarity(daily[m], f"daily mean, {per['id']}"))
    T["stationarity"] = pd.DataFrame(rows)

    # --- structural breaks
    b = cfg["break_tests"]
    T["event_tests"] = breaks.event_tests(daily, ev, b["window_days"], b["min_days_each_side"])
    T["data_driven_breaks"] = breaks.data_driven_breaks(daily, ev, b["max_data_breaks"], b["min_segment_days"],
                                                        b["match_tolerance_days"])

    # --- extremes
    ex = cfg["extremes"]
    T["distribution_by_year"] = extremes.distribution_summary(p, PRIMARY, "year", ex["tail_quantiles"])
    T["negative_by_hour_year"] = extremes.negative_by_hour(p, PRIMARY, ex["negative_threshold"])
    T["negative_by_month_year"] = extremes.negative_by_month(p, PRIMARY, ex["negative_threshold"])
    T["cannibalisation"] = extremes.cannibalisation(p, PRIMARY, ex["midday_local_hours"])
    T["spike_counts_by_year"] = extremes.spike_counts(p, PRIMARY, spikes)
    T["spike_by_hour"] = extremes.spike_by_hour(p, PRIMARY, spikes)
    tails = []
    for per in periods:
        m = pnl.period_mask(da.index, per["start"], per["end"])
        h = extremes.hill_tail(da[m].to_numpy(), ex["hill_tail_fraction"])
        tails.append({"period": per["id"], "n_hours": int(m.sum()), "hill_alpha": h["alpha"]})
    T["tail_index"] = pd.DataFrame(tails)
    scalars["heteroscedasticity"] = extremes.heteroscedasticity(daily)

    # --- relationships
    rel = cfg["relationships"]
    have = [c for c in [PRIMARY] + OTHERS if raw[c].notna().sum() >= rel["min_overlap_hours"]]
    if len(have) > 1:
        lv = relationships.aligned_correlation(raw, have, rel["min_overlap_hours"], False)
        df_ = relationships.aligned_correlation(raw, have, rel["min_overlap_hours"], True)
        T["corr_level"], T["corr_change"], T["corr_overlap_hours"] = lv["corr"], df_["corr"], lv["n"]
        sp = [relationships.spread_stats(raw, PRIMARY, o, periods).assign(series=o) for o in have if o != PRIMARY]
        sp = [s for s in sp if len(s)]
        T["spreads"] = pd.concat(sp, ignore_index=True) if sp else pd.DataFrame()
        T["reserve_links"] = relationships.reserve_links(raw, PRIMARY, [o for o in have if o != PRIMARY],
                                                         rel["min_overlap_hours"])
        for o in ("ida1_pt", "ida2_pt", "ida3_pt", "xbid_pt", "imb_short"):
            if o in have:
                T[f"spread_by_hour_{o}"] = relationships.spread_by_hour(p, PRIMARY, o)

    # --- sample size
    ss = cfg["sample_size"]
    T["ess_by_period"] = sample_size.ess_by_period(daily, periods)
    scalars["ess_all"] = sample_size.effective_sample_size(daily)
    rep = sample_size.representativeness(da, periods, ss["reference_months"], ss["ks_threshold"])
    T["representativeness"] = rep
    scalars["reference"] = {k: rep.attrs.get(k) for k in ("reference_start", "reference_end", "reference_mean", "reference_std")}
    scalars["n_hours_primary"] = int(len(da))
    scalars["n_days_primary"] = int(len(daily))
    scalars["first_day"], scalars["last_day"] = str(daily.index.min().date()), str(daily.index.max().date())
    return T, p, scalars


def write_outputs(tables: Dict[str, pd.DataFrame], p: pd.DataFrame, scalars: dict, cfg: dict,
                  out_dir: Optional[str] = None, fig_dir: Optional[str] = None, figures: bool = True) -> Dict[str, str]:
    out_dir = out_dir or DEFAULT_OUT
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    for name, t in tables.items():
        if isinstance(t, pd.DataFrame) and not t.empty:
            t.to_csv(os.path.join(out_dir, "tables", f"{name}.csv"), index=t.index.name is not None or not isinstance(t.index, pd.RangeIndex))
    with open(os.path.join(out_dir, "scalars.json"), "w", encoding="utf-8") as fh:
        json.dump(scalars, fh, indent=2, default=str)
    paths = {"tables": os.path.join(out_dir, "tables"), "scalars": os.path.join(out_dir, "scalars.json")}
    if figures:
        from common_layer.market_analysis import figures as fg
        paths["figures"] = fg.make_all(tables, p, cfg, fig_dir or DEFAULT_FIG)
    return paths
