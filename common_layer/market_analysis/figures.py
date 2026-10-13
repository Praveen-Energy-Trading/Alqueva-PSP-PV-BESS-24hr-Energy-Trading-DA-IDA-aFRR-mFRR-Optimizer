"""
figures.py — the figures of the Phase 4 findings note. Each function draws one figure from the tables that
run.run_analysis produces and returns the file name; a figure whose data is missing is skipped.
"""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common_layer.market_analysis import dependence, panel as pnl  # noqa: E402
from common_layer.market_analysis.config import events  # noqa: E402

PRIMARY = "da_pt"
NAMES = {"da_pt": "Day-ahead", "ida1_pt": "IDA1", "ida2_pt": "IDA2", "ida3_pt": "IDA3", "xbid_pt": "XBID", "afrr_up": "aFRR up", "afrr_dn": "aFRR down",
         "mfrr_ap": "mFRR activation", "imb_short": "Imbalance (short)", "imb_long": "Imbalance (long)"}      # readable names for the axes
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "svg.fonttype": "none"})      # svg.fonttype none: the text stays text in the SVG (sharp at any zoom)


def _save(fig, fig_dir: str, name: str) -> str:
    os.makedirs(fig_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, name))
    fig.savefig(os.path.join(fig_dir, os.path.splitext(name)[0] + ".svg"), bbox_inches="tight", pad_inches=0.05)    # a vector copy for the dashboard, cropped to the plot
    plt.close(fig)
    return name


def price_history(p: pd.DataFrame, cfg: dict, fig_dir: str) -> str:
    daily = pnl.daily_mean(p[PRIMARY])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily.values, lw=0.5, color="#9ab", label="daily mean")
    ax.plot(daily.index, daily.rolling(30, min_periods=10).mean(), lw=1.6, color="#1f5fbf", label="30-day mean")
    ymax = daily.max()
    for e in events(cfg):
        if daily.index.min() <= pd.Timestamp(e["date"]) <= daily.index.max():
            ax.axvline(pd.Timestamp(e["date"]), color="#c0392b", lw=0.8, ls="--", alpha=0.7)
            ax.text(pd.Timestamp(e["date"]), ymax * 0.97, e["id"], rotation=90, va="top", ha="right", fontsize=6.5, color="#c0392b")
    ax.set(title="Portuguese day-ahead price, daily mean, with documented market events", ylabel="EUR/MWh")
    ax.legend(loc="upper left")
    return _save(fig, fig_dir, "phase4_price_history.png")


def seasonal_shape(p: pd.DataFrame, fig_dir: str) -> str:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.1]})
    by = p.dropna(subset=[PRIMARY]).groupby(["year", "local_hour"])[PRIMARY].mean().unstack(0)
    by.plot(ax=axs[0], lw=1.4)
    axs[0].set(title="Mean price by hour of day (market time), by year", xlabel="hour", ylabel="EUR/MWh")
    last = int(p["year"].max()) - (0 if (p["year"] == p["year"].max()).sum() > 24 * 270 else 1)
    hm = p[p["year"] == last].pivot_table(index="local_hour", columns="month", values=PRIMARY, aggfunc="median")
    im = axs[1].imshow(hm.values, aspect="auto", origin="lower", cmap="RdYlBu_r",
                       extent=[hm.columns.min() - 0.5, hm.columns.max() + 0.5, -0.5, 23.5])
    axs[1].set(title=f"Median price by hour and month, {last}", xlabel="month", ylabel="hour")
    axs[1].grid(False)
    fig.colorbar(im, ax=axs[1], label="EUR/MWh")
    return _save(fig, fig_dir, "phase4_seasonal_shape.png")


def autocorrelation(p: pd.DataFrame, cfg: dict, fig_dir: str) -> str:
    da = p[PRIMARY].dropna()
    resid = dependence.deseasonalise(p, PRIMARY)
    n = cfg["dependence"]["hourly_lags"]
    a1, a2 = dependence.acf_series(da, n), dependence.acf_series(resid, n)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    axs[0].bar(a1.index[1:], a1.values[1:], width=1.0, color="#1f5fbf")
    axs[0].set(title="Autocorrelation of the hourly price", xlabel="lag (hours)", ylabel="ACF")
    axs[1].bar(a2.index[1:], a2.values[1:], width=1.0, color="#c0732b")
    axs[1].set(title="After removing year x month x hour x weekend means", xlabel="lag (hours)")
    for ax in axs:
        ax.axvline(24, color="k", lw=0.5, ls=":")
        ax.axvline(168, color="k", lw=0.5, ls=":")
    return _save(fig, fig_dir, "phase4_autocorrelation.png")


def structural_breaks(p: pd.DataFrame, T: Dict[str, pd.DataFrame], cfg: dict, fig_dir: str) -> str:
    daily = pnl.daily_mean(p[PRIMARY])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily.values, lw=0.5, color="#9ab")
    db = T.get("data_driven_breaks", pd.DataFrame())
    cuts = [daily.index.min()] + [pd.Timestamp(d) for d in sorted(db["break_date"])] + [daily.index.max()] if len(db) else []
    for a, b in zip(cuts, cuts[1:]):
        seg = daily[(daily.index >= a) & (daily.index <= b)]
        ax.hlines(seg.mean(), a, b, color="#c0392b", lw=2)
    for e in events(cfg):
        if daily.index.min() <= pd.Timestamp(e["date"]) <= daily.index.max():
            ax.axvline(pd.Timestamp(e["date"]), color="#1f5fbf", lw=0.7, ls=":", alpha=0.8)
    ax.set(title="Data-driven level breaks (red: segment means) and documented events (blue dotted)", ylabel="EUR/MWh")
    return _save(fig, fig_dir, "phase4_structural_breaks.png")


def negative_prices(T: Dict[str, pd.DataFrame], fig_dir: str) -> str:
    hy, my = T["negative_by_hour_year"], T["negative_by_month_year"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    im = axs[0].imshow(hy.values * 100, aspect="auto", origin="lower", cmap="Blues",
                       extent=[hy.columns.min() - 0.5, hy.columns.max() + 0.5, -0.5, 23.5])
    axs[0].set(title="Share of hours with price <= 0 (%), by hour and year", xlabel="year", ylabel="hour")
    axs[0].grid(False)
    fig.colorbar(im, ax=axs[0])
    im2 = axs[1].imshow(my.values * 100, aspect="auto", origin="lower", cmap="Blues",
                        extent=[my.columns.min() - 0.5, my.columns.max() + 0.5, 0.5, 12.5])
    axs[1].set(title="... by month and year", xlabel="year", ylabel="month")
    axs[1].grid(False)
    fig.colorbar(im2, ax=axs[1])
    return _save(fig, fig_dir, "phase4_negative_prices.png")


def price_distribution(p: pd.DataFrame, fig_dir: str) -> str:
    d = p.dropna(subset=[PRIMARY])
    years = sorted(d["year"].unique())
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot([d.loc[d["year"] == y, PRIMARY].clip(-100, 600).values for y in years], labels=[str(y) for y in years],
               showfliers=False, whis=(1, 99))
    ax.set(title="Hourly day-ahead price by year (box: quartiles, whiskers: 1st and 99th percentile)", ylabel="EUR/MWh")
    return _save(fig, fig_dir, "phase4_price_distribution.png")


def spreads(p: pd.DataFrame, cfg: dict, fig_dir: str) -> str:
    cols = [c for c in ("ida1_pt", "ida2_pt", "ida3_pt", "xbid_pt", "imb_short", "imb_long") if p[c].notna().sum() > 500]
    if not cols:
        return ""
    fig, ax = plt.subplots(figsize=(10, 4))
    data = [(p[c] - p[PRIMARY]).dropna().clip(-150, 150).values for c in cols]
    ax.boxplot(data, labels=[c.replace("_pt", "") for c in cols], showfliers=False, whis=(1, 99))
    ax.axhline(0, color="k", lw=0.6)
    ax.set(title="Spread to the day-ahead price in the same hour (EUR/MWh)", ylabel="other minus day-ahead")
    return _save(fig, fig_dir, "phase4_spreads_vs_da.png")


def correlations(T: Dict[str, pd.DataFrame], fig_dir: str) -> str:
    c = T.get("corr_level")
    if c is None or len(c) < 3:
        return ""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(c.values.astype(float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(c))), ax.set_xticklabels([NAMES.get(x, x) for x in c.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(c))), ax.set_yticklabels([NAMES.get(x, x) for x in c.index])
    for i in range(len(c)):
        for j in range(len(c)):
            v = c.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation of hourly prices (levels)")
    return _save(fig, fig_dir, "phase4_market_correlations.png")


def sample_size_fig(T: Dict[str, pd.DataFrame], fig_dir: str) -> str:
    ess, rep = T["ess_by_period"], T["representativeness"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax = axs[0]
    x = np.arange(len(ess))
    ax.bar(x - 0.2, ess["days"], 0.4, label="days", color="#9ab")
    ax.bar(x + 0.2, ess["effective_days"], 0.4, label="effective days", color="#1f5fbf")
    ax.set_xticks(x), ax.set_xticklabels(ess["period"], rotation=30, ha="right")
    ax.set(title="Days and effective independent days per period"), ax.legend()
    yr = rep[rep["kind"] == "year"]
    colors = ["#2e8b57" if s else "#c0392b" for s in yr["similar"]]
    axs[1].bar(yr["group"], yr["ks_distance"], color=colors)
    axs[1].axhline(0.2, color="k", lw=0.8, ls="--")
    axs[1].set(title="Distance of each year to the latest 12 months (KS; green = similar)", ylabel="KS distance")
    return _save(fig, fig_dir, "phase4_sample_size.png")


def make_all(T: Dict[str, pd.DataFrame], p: pd.DataFrame, cfg: dict, fig_dir: str) -> List[str]:
    made = [price_history(p, cfg, fig_dir), seasonal_shape(p, fig_dir), autocorrelation(p, cfg, fig_dir),
            structural_breaks(p, T, cfg, fig_dir), negative_prices(T, fig_dir), price_distribution(p, fig_dir),
            spreads(p, cfg, fig_dir), correlations(T, fig_dir), sample_size_fig(T, fig_dir)]
    return [m for m in made if m]
