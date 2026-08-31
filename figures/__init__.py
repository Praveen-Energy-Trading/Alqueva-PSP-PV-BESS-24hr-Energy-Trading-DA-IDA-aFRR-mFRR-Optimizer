"""
figures — Alqueva trading system figure package.

Rebuilt from scratch. Every figure is generated automatically at the end of
every pipeline run and overwrites figures/output/, so the folder always
reflects the latest run. All numbers are read straight from the run's own
outputs (PositionStore/ReserveStore/DeliveryStore + the daily Excel report)
-- nothing here is illustrative or recomputed independently of the solver.

Public API (called by run_production.py):
    figures.generate(date)

Outputs saved to figures/output/:
    fig01_dispatch_and_price.png     Plant net dispatch (all sources) vs. DA price
    fig02_reservoir_trajectory.png   Upper/lower reservoir levels + terminal constraint
    fig03_bess_state_of_charge.png   BESS SoC trajectory vs. operating bounds
    fig04_pv_bess_energy_flow.png    PV disposition (grid/BESS/curtailed) + BESS power
    fig05_reserve_capacity.png       aFRR + mFRR capacity offered (MW up/down)
    fig06_revenue_breakdown.png      Revenue by market, cumulative waterfall
    fig07_gate_position_evolution.png  DA position vs. final committed + intraday delta
    fig08_water_balance.png          Mass-balance integrity check (actual vs. predicted dV)

Design system:
    - Pure white background, no panel shading -- reads like a report figure,
      not a dashboard widget.
    - Muted, restrained palette (steel blue / slate / amber / forest / brick)
      instead of saturated primary colours -- built to sit in a PDF or slide
      deck next to text, not to shout for attention on a screen.
    - One idea per figure. Where the old figure set showed the same
      quantity (net dispatch, gate position) from two or three angles across
      separate files, this set merges them into one figure that carries the
      full story instead of asking the reader to cross-reference.
    - 220 DPI, not 600 -- indistinguishable in a report or on screen, a
      fraction of the file size, and several times faster to render (the
      previous 600 DPI setting made every dashboard page load noticeably
      slower for no visible benefit).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT   = Path(__file__).resolve().parent.parent
_DB_POS = _ROOT / "runtime" / "db" / "positions.db"
_DB_RT  = _ROOT / "runtime" / "db" / "realtime.db"
_DB_RES = _ROOT / "runtime" / "db" / "reserve.db"
_RPTS   = _ROOT / "runtime" / "reports"
_OUT    = Path(__file__).resolve().parent / "output"
_OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Design system — muted, restrained palette. No saturated primaries.
# ---------------------------------------------------------------------------

_BG      = "#ffffff"
_INK     = "#1a1a1a"        # body text / axis labels
_INK_MUT = "#6b6b6b"        # secondary text
_GRID    = "#e6e6e6"        # gridlines
_SPINE   = "#333333"        # axis spines

_NAVY    = "#2c4a6e"        # primary series (generation / DA / positive flow)
_SLATE   = "#7a8a99"        # secondary series (pump / comparison)
_AMBER   = "#b8862e"        # price overlays
_FOREST  = "#3f7057"        # gains / pass / good
_BRICK   = "#a24a3f"        # losses / fail / warning
_TEAL    = "#3f7a86"        # tertiary accent (lower reservoir, BESS)
_LILAC   = "#7d6a94"        # quaternary accent (IDA/XBID series)

_DPI = 220

# ---------------------------------------------------------------------------
# Font / line scaling — same idea as before (fonts grow with figure width),
# tuned down slightly since the new layouts are less panel-dense.
# ---------------------------------------------------------------------------

_REF_WIDTH = 10.0

def _scale(fig_width: float) -> float:
    return max(1.0, fig_width / _REF_WIDTH)

def _rcparams(fig_width: float) -> dict:
    s = _scale(fig_width)
    return {
        "figure.facecolor" : _BG,
        "axes.facecolor"   : _BG,
        "axes.edgecolor"   : _SPINE,
        "axes.labelcolor"  : _INK,
        "axes.titlecolor"  : _INK,
        "axes.labelweight" : "medium",
        "axes.linewidth"   : round(0.9 * s, 2),
        "xtick.color"      : _INK,
        "ytick.color"      : _INK,
        "text.color"       : _INK,
        "grid.color"       : _GRID,
        "grid.linestyle"   : "-",
        "grid.linewidth"   : round(0.6 * s, 2),
        "grid.alpha"       : 1.0,
        "legend.facecolor" : _BG,
        "legend.edgecolor" : _GRID,
        "legend.framealpha": 1.0,
        "font.family"      : "sans-serif",
        "font.size"        : round(11.0 * s, 1),
        "axes.labelsize"   : round(11.0 * s, 1),
        "axes.titlesize"   : round(11.5 * s, 1),
        "xtick.labelsize"  : round(10.0 * s, 1),
        "ytick.labelsize"  : round(10.0 * s, 1),
        "legend.fontsize"  : round(10.0 * s, 1),
        "lines.linewidth"  : round(1.9 * s, 2),
    }

def _polish(ax: plt.Axes, fig_width: float = _REF_WIDTH, y_only: bool = True) -> None:
    """Top/right spines off, left gridlines only, ticks in the ink colour --
    the 'report figure' look, not the boxed 'dashboard widget' look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y" if y_only else "both")

def _panel(ax: plt.Axes, letter: str, title: str, fig_width: float = _REF_WIDTH) -> None:
    s = _scale(fig_width)
    ax.set_title(f"({letter})  {title}", loc="left",
                 fontsize=round(10 * s, 1), fontweight="bold", color=_INK, pad=10)

def _eur_fmt(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"€{x/1e3:,.0f}k" if abs(x) >= 1000 else f"€{x:.0f}")
    )

def _legend(ax: plt.Axes, **kwargs) -> None:
    defaults = dict(frameon=True, framealpha=1.0, edgecolor=_GRID, fancybox=False)
    defaults.update(kwargs)
    ax.legend(**defaults)

# ---------------------------------------------------------------------------
# Data loaders — unchanged from the previous version, verified correct
# against the real stores; only the visual layer was rebuilt.
# ---------------------------------------------------------------------------

def _pos(date: str) -> pd.DataFrame:
    con = sqlite3.connect(_DB_POS)
    df = pd.read_sql("SELECT * FROM positions WHERE delivery_date=?", con, params=(date,))
    con.close()
    return df

def _res(date: str) -> pd.DataFrame:
    con = sqlite3.connect(_DB_RES)
    df = pd.read_sql("SELECT * FROM reserve WHERE delivery_date=?", con, params=(date,))
    con.close()
    return df

def _rt(date: str) -> pd.DataFrame:
    con = sqlite3.connect(_DB_RT)
    df = pd.read_sql("SELECT * FROM delivery WHERE delivery_date=?", con, params=(date,))
    con.close()
    return df

def _dispatch_df(date: str) -> pd.DataFrame:
    """Read Dispatch_Hourly sheet from the Excel report.

    Sheet layout: row1=title, row2=group band, row3=column names, rows4+=data.
    skiprows=2 skips title+group-band so row3 becomes the header automatically.
    """
    path = _RPTS / f"daily_report_{date}.xlsx"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name="Dispatch_Hourly", skiprows=2)
    except Exception:
        return pd.DataFrame()

def _summary(date: str) -> dict:
    """Read Summary_KPIs sheet. Returns {metric_name: float_value}."""
    path = _RPTS / f"daily_report_{date}.xlsx"
    if not path.exists():
        return {}
    try:
        raw = pd.read_excel(path, sheet_name="Summary_KPIs", header=None, skiprows=2)
        out = {}
        for _, row in raw.iterrows():
            k = row.iloc[1] if len(row) > 1 else None
            v = row.iloc[2] if len(row) > 2 else None
            if pd.notna(k) and pd.notna(v):
                try:
                    out[str(k).strip()] = float(v)
                except (ValueError, TypeError):
                    pass
        return out
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(_OUT / name, dpi=_DPI, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"    {name}")

# ---------------------------------------------------------------------------
# fig01 — Plant Dispatch & Price
# ---------------------------------------------------------------------------

def _fig01(date: str) -> None:
    """Net plant dispatch (PSP + PV + BESS combined, the real physical
    output) against the DA clearing price -- the single figure that answers
    'what did the plant actually do, and against what price'. The previous
    set split this across a DA-only chart and a PSP-only chart; the plant's
    real net position is the number that matters, and combining removes an
    unnecessary cross-reference."""
    W = 12.0
    plt.rcParams.update(_rcparams(W))
    df = _dispatch_df(date)
    if df.empty or "Plant_net_final_MW" not in df.columns:
        return
    hours = df["Hour"].astype(int).values if "Hour" in df.columns else np.arange(1, len(df) + 1)
    net   = df["Plant_net_final_MW"].values
    price = df["DA_price_EUR_MWh"].values if "DA_price_EUR_MWh" in df.columns else None
    lw = round(1.8 * _scale(W), 2)

    fig, ax1 = plt.subplots(figsize=(W, 5))
    ax1.bar(hours, net, color=[_FOREST if v >= 0 else _BRICK for v in net],
            width=0.72, label="Net dispatch (MW)")
    ax1.axhline(0, color=_SPINE, linewidth=1.0)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Net dispatch (MW)")
    ax1.set_xticks(hours[::2])
    _polish(ax1, W)

    if price is not None:
        ax2 = ax1.twinx()
        ax2.plot(hours, price, color=_AMBER, linewidth=lw, label="DA price (EUR/MWh)")
        ax2.set_ylabel("DA price (EUR/MWh)", color=_AMBER)
        ax2.tick_params(axis="y", colors=_AMBER)
        ax2.spines["top"].set_visible(False)
        h2, l2 = ax2.get_legend_handles_labels()
    else:
        h2, l2 = [], []

    h1, l1 = ax1.get_legend_handles_labels()
    _legend(ax1, handles=h1 + h2, labels=l1 + l2, loc="upper center",
            bbox_to_anchor=(0.5, -0.16), ncol=len(h1 + h2))
    _panel(ax1, "a", f"Plant dispatch vs. DA price — {date}", W)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    _save(fig, "fig01_dispatch_and_price.png")

# ---------------------------------------------------------------------------
# fig02 — Reservoir Trajectory
# ---------------------------------------------------------------------------

def _fig02(date: str) -> None:
    """Two-reservoir hydro storage trajectory -- the physical asset the
    whole trading strategy sits on top of. Alqueva (~2,500 hm3) and
    Pedrogao (~30 hm3) are ~100x apart in scale, so each gets its own
    panel and its own zoomed range (plotting from a 0 baseline flattens
    Alqueva's real ~1% daily swing into an apparently dead line). The
    dashed reference marks the no-free-lunch constraint: the upper
    reservoir must end the day at or above where it started."""
    W = 12.0
    plt.rcParams.update(_rcparams(W))
    df = _dispatch_df(date)
    if df.empty or "Reservoir_upper_hm3" not in df.columns:
        return
    hours = df["Hour"].astype(int).values if "Hour" in df.columns else np.arange(1, len(df) + 1)
    upper = df["Reservoir_upper_hm3"].values
    lower = df["Reservoir_lower_hm3"].values
    d0 = df["dReservoir_upper_hm3"].values[0] if "dReservoir_upper_hm3" in df.columns else 0.0
    start_upper = upper[0] - d0
    lw = round(1.8 * _scale(W), 2)

    fig, (ax_u, ax_l) = plt.subplots(2, 1, figsize=(W, 6.5), sharex=True)

    u_vals = list(upper) + [start_upper]
    u_lo, u_hi = min(u_vals), max(u_vals)
    u_pad = max((u_hi - u_lo) * 0.15, 0.5)
    ax_u.set_ylim(u_lo - u_pad, u_hi + u_pad)
    ax_u.fill_between(hours, upper, u_lo - u_pad, color=_NAVY, alpha=0.12)
    ax_u.plot(hours, upper, color=_NAVY, linewidth=lw, label="Alqueva level (hm3)")
    ax_u.axhline(start_upper, color=_AMBER, linestyle="--", linewidth=lw * 0.8,
                 label=f"Must end ≥ start ({start_upper:,.0f} hm3)")
    ax_u.set_ylabel("Alqueva (hm3)")
    _polish(ax_u, W)
    _legend(ax_u, loc="upper right")
    _panel(ax_u, "a", "Alqueva — upper reservoir", W)

    l_vals = list(lower)
    l_lo, l_hi = min(l_vals), max(l_vals)
    l_pad = max((l_hi - l_lo) * 0.15, 0.5)
    ax_l.set_ylim(l_lo - l_pad, l_hi + l_pad)
    ax_l.fill_between(hours, lower, l_lo - l_pad, color=_TEAL, alpha=0.12)
    ax_l.plot(hours, lower, color=_TEAL, linewidth=lw, label="Pedrogão level (hm3)")
    ax_l.set_xlabel("Hour")
    ax_l.set_ylabel("Pedrogão (hm3)")
    ax_l.set_xticks(hours[::2])
    _polish(ax_l, W)
    _legend(ax_l, loc="upper right")
    _panel(ax_l, "b", "Pedrogão — lower reservoir", W)

    fig.suptitle(f"Reservoir Trajectory — {date}", fontweight="bold", fontsize=13, color=_INK)
    fig.tight_layout()
    _save(fig, "fig02_reservoir_trajectory.png")

# ---------------------------------------------------------------------------
# fig03 — BESS State of Charge
# ---------------------------------------------------------------------------

_BESS_CAP_MWH = 2.0
_SOC_MIN_PCT  = 10.0
_SOC_MAX_PCT  = 95.0

def _fig03(date: str) -> None:
    W = 12.0
    plt.rcParams.update(_rcparams(W))
    df = _dispatch_df(date)
    if df.empty or "BESS_SOC_MWh" not in df.columns:
        return
    hours = df["Hour"].astype(int).values if "Hour" in df.columns else np.arange(1, len(df) + 1)
    soc_pct = df["BESS_SOC_MWh"].values / _BESS_CAP_MWH * 100.0
    sh = list(hours) + [hours[-1] + 1]
    soc_step = list(soc_pct) + [soc_pct[-1]]
    lw = round(1.8 * _scale(W), 2)

    fig, ax = plt.subplots(figsize=(W, 4.2))
    ax.fill_between(sh, soc_step, step="post", color=_NAVY, alpha=0.15)
    ax.step(sh, soc_step, where="post", color=_NAVY, linewidth=lw, label="BESS SoC (%)")
    ax.axhline(_SOC_MIN_PCT, color=_BRICK, linestyle=":", linewidth=lw * 0.7,
               label=f"Min {_SOC_MIN_PCT:.0f}%")
    ax.axhline(_SOC_MAX_PCT, color=_FOREST, linestyle=":", linewidth=lw * 0.7,
               label=f"Max {_SOC_MAX_PCT:.0f}%")
    ax.set_xlabel("Hour")
    ax.set_ylabel("State of charge (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(hours[::2])
    _polish(ax, W)
    _legend(ax, loc="lower right")
    _panel(ax, "a", f"BESS state of charge — {date}", W)
    fig.tight_layout()
    _save(fig, "fig03_bess_state_of_charge.png")

# ---------------------------------------------------------------------------
# fig04 — PV & BESS Energy Flow
# ---------------------------------------------------------------------------

def _fig04(date: str) -> None:
    W = 12.0
    plt.rcParams.update(_rcparams(W))
    df = _dispatch_df(date)
    if df.empty or "PV_used_MW" not in df.columns:
        return
    hours   = df["Hour"].astype(int).values if "Hour" in df.columns else np.arange(1, len(df) + 1)
    pv_used = df["PV_used_MW"].values
    pv_bess = df["PV_to_BESS_MW"].values if "PV_to_BESS_MW" in df.columns else np.zeros(len(hours))
    pv_curt = df["PV_curtailed_MW"].values if "PV_curtailed_MW" in df.columns else np.zeros(len(hours))
    bess_dis = df["BESS_discharge_MW"].values if "BESS_discharge_MW" in df.columns else np.zeros(len(hours))
    bess_chg = df["BESS_total_charge_MW"].values if "BESS_total_charge_MW" in df.columns else np.zeros(len(hours))
    lw = round(1.6 * _scale(W), 2)

    fig, (ax_pv, ax_bs) = plt.subplots(2, 1, figsize=(W, 6.5), sharex=True)

    ax_pv.bar(hours, pv_used, width=0.65, color=_NAVY, label="PV → grid")
    ax_pv.bar(hours, pv_bess, width=0.65, bottom=pv_used, color=_TEAL, label="PV → BESS")
    ax_pv.bar(hours, pv_curt, width=0.65, bottom=pv_used + pv_bess, color=_SLATE,
              alpha=0.7, label="PV curtailed")
    ax_pv.set_ylabel("PV power (MW)")
    _polish(ax_pv, W)
    _legend(ax_pv, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    _panel(ax_pv, "a", "PV generation disposition", W)

    bw = 0.32
    ax_bs.bar([h - bw / 2 for h in hours],  bess_dis, width=bw, color=_FOREST, label="Discharge")
    ax_bs.bar([h + bw / 2 for h in hours], -bess_chg, width=bw, color=_BRICK, label="Charge (−ve)")
    ax_bs.axhline(0, color=_SPINE, linewidth=1.0)
    ax_bs.set_xlabel("Hour")
    ax_bs.set_ylabel("BESS power (MW)")
    ax_bs.set_xticks(hours[::2])
    _polish(ax_bs, W)
    _legend(ax_bs, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _panel(ax_bs, "b", "BESS charge / discharge power", W)

    fig.suptitle(f"PV + BESS Energy Flow — {date}", fontweight="bold", fontsize=13, color=_INK)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    _save(fig, "fig04_pv_bess_energy_flow.png")

# ---------------------------------------------------------------------------
# fig05 — Reserve Capacity
# ---------------------------------------------------------------------------

def _fig05(date: str) -> None:
    W = 12.0
    plt.rcParams.update(_rcparams(W))
    res = _res(date)
    if res.empty:
        return
    hours = list(range(1, 25))

    fig, axes = plt.subplots(2, 1, figsize=(W, 6.5), sharex=True)
    for ax, prod, letter, cu, cd in [
        (axes[0], "aFRR", "a", _FOREST, _BRICK),
        (axes[1], "mFRR", "b", _FOREST, _BRICK),
    ]:
        sub = res[res["product"] == prod].sort_values("hour")
        up = sub.set_index("hour")["up_mw"].reindex(hours, fill_value=0).values
        dn = sub.set_index("hour")["dn_mw"].reindex(hours, fill_value=0).values
        ax.bar(hours, up,  color=cu, width=0.4, align="edge", label="Up")
        ax.bar(hours, -dn, color=cd, width=-0.4, align="edge", label="Down")
        ax.axhline(0, color=_SPINE, linewidth=1.0)
        ax.set_ylabel("Capacity (MW)")
        _polish(ax, W)
        _legend(ax, loc="upper right")
        _panel(ax, letter, f"{prod} reserve capacity offered", W)
    axes[1].set_xlabel("Hour")
    axes[1].set_xticks(hours[::2])
    fig.suptitle(f"Reserve Capacity — {date}", fontweight="bold", fontsize=13, color=_INK)
    fig.tight_layout()
    _save(fig, "fig05_reserve_capacity.png")

# ---------------------------------------------------------------------------
# fig06 — Revenue Breakdown (waterfall)
# ---------------------------------------------------------------------------

def _fig06(date: str) -> None:
    W = 11.0
    plt.rcParams.update(_rcparams(W))
    sm = _summary(date)
    da_rev   = sm.get("DA energy revenue", 0.0)
    ida_rev  = sm.get("IDA incremental revenue", 0.0)
    afrr_rev = sm.get("aFRR capacity revenue", 0.0) + sm.get("aFRR activation revenue", 0.0)
    mfrr_rev = sm.get("mFRR capacity revenue", 0.0) + sm.get("mFRR activation revenue", 0.0)
    imb_rev  = sm.get("Imbalance settlement", 0.0)
    markets = ["DA", "IDA+XBID", "aFRR", "mFRR", "Imbalance"]
    values  = [da_rev, ida_rev, afrr_rev, mfrr_rev, imb_rev]
    total   = sum(values)

    fig, ax = plt.subplots(figsize=(W, 5))
    running = 0.0
    bottoms = []
    for v in values:
        bottoms.append(running if v >= 0 else running + v)
        running += v
    bars = ax.bar(markets, [abs(v) for v in values], bottom=bottoms,
                   color=[_FOREST if v >= 0 else _BRICK for v in values], width=0.55)
    ax.bar(["TOTAL"], [total], color=_NAVY, width=0.55)
    ax.axhline(0, color=_SPINE, linewidth=1.0)
    fs = round(8.5 * _scale(W), 1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                f"€{v:,.0f}", ha="center", va="center", fontsize=fs, color="white", fontweight="bold")
    ax.text(len(markets), total / 2, f"€{total:,.0f}",
            ha="center", va="center", fontsize=fs, color="white", fontweight="bold")
    ax.set_ylabel("Revenue (EUR)")
    _polish(ax, W)
    _eur_fmt(ax)
    _panel(ax, "a", f"Revenue by market — {date}", W)
    fig.tight_layout()
    _save(fig, "fig06_revenue_breakdown.png")

# ---------------------------------------------------------------------------
# fig07 — Gate Position Evolution
# ---------------------------------------------------------------------------

def _fig07(date: str) -> None:
    """DA position vs. the final committed position after every intraday
    gate, with the delta highlighted -- replaces the previous two separate
    figures (one plotting all 5 gates' raw positions overlaid, one plotting
    DA-vs-final) with a single figure that shows both the reference and
    what changed, since that comparison is the entire point of either."""
    W = 12.0
    plt.rcParams.update(_rcparams(W))
    pos = _pos(date)
    if pos.empty:
        return
    hours = list(range(1, 25))
    lw = round(1.8 * _scale(W), 2)
    da = pos[pos["gate"] == "DA"].set_index("hour")["volume_mwh"].reindex(hours, fill_value=0)
    intraday = (pos[pos["gate"].isin(["IDA1", "IDA2", "IDA3", "XBID"])]
                .groupby("hour")["volume_mwh"].sum().reindex(hours, fill_value=0))
    final = (da + intraday).values
    delta = final - da.values

    fig, ax = plt.subplots(figsize=(W, 5))
    ax.bar(hours, delta, color=[_FOREST if d >= 0 else _BRICK for d in delta],
           alpha=0.5, width=0.6, label="Intraday adjustment (MWh)")
    ax.step(hours, da.values, where="mid", color=_SLATE, linewidth=lw,
            linestyle="--", label="DA position (MWh)")
    ax.step(hours, final, where="mid", color=_NAVY, linewidth=lw,
            label="Final committed (MWh)")
    ax.axhline(0, color=_SPINE, linewidth=1.0)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Net position (MWh)")
    ax.set_xticks(hours[::2])
    _polish(ax, W)
    _legend(ax, loc="upper right")
    _panel(ax, "a", f"Gate position evolution — {date}", W)
    fig.tight_layout()
    _save(fig, "fig07_gate_position_evolution.png")

# ---------------------------------------------------------------------------
# fig08 — Water Balance Verification
# ---------------------------------------------------------------------------

def _fig08(date: str) -> None:
    """Mass-balance integrity check: actual hourly reservoir change vs. the
    change predicted from physical inputs (inflow + pump - turbine - spill).
    The two are computed from independent columns and should overlap
    exactly if the solved MILP respects mass conservation -- the figure a
    reviewer would ask for to confirm the model isn't creating or losing
    water."""
    W = 12.0
    plt.rcParams.update(_rcparams(W))
    df = _dispatch_df(date)
    needed = {"dReservoir_upper_hm3", "dReservoir_theoretical_hm3", "Mass_balance_error_hm3", "Spill_m3h"}
    if df.empty or not needed.issubset(df.columns):
        return
    hours = df["Hour"].astype(int).values if "Hour" in df.columns else np.arange(1, len(df) + 1)
    actual    = df["dReservoir_upper_hm3"].values
    predicted = df["dReservoir_theoretical_hm3"].values
    error     = df["Mass_balance_error_hm3"].values
    lw = round(1.8 * _scale(W), 2)

    fig, (ax_v, ax_e) = plt.subplots(2, 1, figsize=(W, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    ax_v.plot(hours, actual, color=_NAVY, linewidth=lw, label="Actual ΔV (hm3/h)")
    ax_v.plot(hours, predicted, color=_AMBER, linewidth=lw * 0.7, linestyle="--",
              label="Predicted ΔV = inflow + pump − turbine − spill")
    ax_v.axhline(0, color=_SPINE, linewidth=1.0)
    ax_v.set_ylabel("Reservoir ΔV (hm3/h)")
    _polish(ax_v, W)
    _legend(ax_v, loc="upper right")
    _panel(ax_v, "a", "Actual vs. predicted reservoir change", W)

    max_err = np.abs(error).max()
    passed = max_err < 1e-4
    ax_e.bar(hours, error, color=_FOREST if passed else _BRICK, width=0.6)
    ax_e.axhline(0, color=_SPINE, linewidth=1.0)
    ax_e.set_xlabel("Hour")
    ax_e.set_ylabel("Balance error (hm3)")
    ax_e.set_xticks(hours[::2])
    _polish(ax_e, W)
    _panel(ax_e, "b", f"Mass balance error — max |error| = {max_err:.2e} hm3  "
           + ("[PASS]" if passed else "[FAIL]"), W)

    fig.suptitle(f"Water Balance Verification — {date}", fontweight="bold", fontsize=13, color=_INK)
    fig.tight_layout()
    _save(fig, "fig08_water_balance.png")

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate(date: str) -> None:
    """Generate all 8 figures for the given delivery date.
    Overwrites previous output."""
    print("  Generating figures...")
    _fig01(date)
    _fig02(date)
    _fig03(date)
    _fig04(date)
    _fig05(date)
    _fig06(date)
    _fig07(date)
    _fig08(date)
    print(f"  Figures saved -> figures/output/  ({len(list(_OUT.glob('*.png')))} files)\n")
