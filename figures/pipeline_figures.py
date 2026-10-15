"""pipeline_figures.py - the two figures about the pipeline itself: when each market decides, and how good the price forecasts are."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from . import _data as D
from . import _style as S


# ------------------------------------------------------------------------------------------------------------------------------------ 1
def trading_day():
    """A swim-lane timeline: for each market the moment it closes and the hours of the delivery day it covers."""
    t0, t1 = 6.0, 48.0                      # hours since 00:00 of the day before delivery: 06:00 D-1 .. 24:00 D
    lanes = [
        ("Day-ahead", "sell the energy plan, hour by hour", [12.0], (24, 48), S.BLUE, "12:00", False),
        ("aFRR band", "hold capacity back for the grid", [12.5], (24, 48), S.TEAL, "after day-ahead", True),
        ("mFRR band", "hold more capacity back", [13.5], (24, 48), "#8b5cf6", "after aFRR", True),
        ("Intraday auction 1", "re-optimise with fresh prices", [15.0], (24, 48), S.AMBER, "15:00", False),
        ("Intraday auction 2", "re-optimise again", [22.0], (24, 48), S.AMBER, "22:00", False),
        ("Intraday auction 3", "covers the afternoon only", [34.0], (36, 48), S.AMBER, "10:00", False),
        ("Continuous intraday", "checked at six moments", [18.5, 22.5, 27.0, 30.0, 33.5, 36.0], (24, 48), S.ROSE, "", False),
        ("Delivery", "dispatch, reserve calls, settlement", [24.0], (24, 48), S.GREEN, "00:00", False),
    ]
    fig = S.canvas("The trading day",
                   "Seven decisions, one after the other, then the delivery: each starts from what the previous one locked in and only changes it when that pays",
                   "Source: market configuration. Reserve gate hours are estimates (their order is confirmed); the delivery day is 96 quarter-hours", accent=S.AMBER)
    ax = fig.add_axes([0.255, 0.115, 0.715, 0.70])
    ax.set_xlim(t0, t1)
    n = len(lanes)
    ax.set_ylim(n - 0.3, -1.15)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([])
    ax.tick_params(length=0)
    # the delivery day
    ax.add_patch(Rectangle((24, -1.0), 24, n + 0.7, facecolor=S.BLUE, alpha=0.06, linewidth=0, zorder=0))
    ax.text(36, -0.80, "Delivery day  ·  96 quarter-hours", ha="center", va="center", fontsize=10, fontweight="bold", color=S.BLUE)
    ax.text(15, -0.80, "Day before delivery", ha="center", va="center", fontsize=10, fontweight="bold", color=S.MUTED)
    ax.axvline(24, color=S.BLUE, linewidth=1.2, alpha=0.6, zorder=1)
    ticks = list(range(6, 49, 6))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{(t % 24):02d}:00" if t != 48 else "24:00" for t in ticks])
    ax.grid(axis="x", color=S.GRID, linewidth=0.8, zorder=0)
    for i, (name, what, times, cover, color, tlabel, estimate) in enumerate(lanes):
        y = i
        ax.add_patch(FancyBboxPatch((cover[0], y - 0.17), cover[1] - cover[0], 0.34, boxstyle="round,pad=0,rounding_size=0.1", facecolor=color, alpha=0.20,
                                    edgecolor="none", mutation_aspect=0.02, zorder=2))
        ax.hlines(y, times[0], cover[0], color=color, linewidth=1.4, linestyle=(0, (3, 3)), zorder=2, alpha=0.8)
        for k, tm in enumerate(times):
            ax.plot([tm], [y], "o", markersize=9 if len(times) == 1 else 7, markerfacecolor="white" if estimate else color, markeredgecolor=color, markeredgewidth=2.2, zorder=4)
        if tlabel:
            ax.text(times[0], y - 0.30, ("≈ " if estimate else "") + tlabel, ha="center", va="bottom", fontsize=8.5, color=color, fontweight="bold")
        # the lane name on the left
        fig.text(0.032, 0.115 + 0.70 * (1 - (y + 1.15) / (n + 0.85)), name, fontsize=11, fontweight="bold", color=S.INK, va="center")
        fig.text(0.032, 0.115 + 0.70 * (1 - (y + 1.15) / (n + 0.85)) - 0.027, what, fontsize=8.8, color=S.MUTED, va="center")
    return S.save(fig, "01_trading_day")


# ------------------------------------------------------------------------------------------------------------------------------------ 6
def forecast_skill(rows: list):
    rows = list(rows)
    n = len(rows)
    fig = S.canvas("How good are the price forecasts?",
                   "Error of the chosen model on a locked test period it never saw during tuning, against the plain reference model",
                   "Source: locked-test results of the model development (error per hour over a test period of about a year; EUR/MWh, EUR/MW/h for aFRR)", accent=S.VIOLET)
    ax_mae = fig.add_axes([0.105, 0.12, 0.275, 0.68])
    ax_rmse = fig.add_axes([0.425, 0.12, 0.275, 0.68])
    ax_gain = fig.add_axes([0.765, 0.12, 0.205, 0.68])
    y = np.arange(n)[::-1]
    h = 0.34

    def pair(ax, ref_key, champ_key, title, label_ref=False):
        S.clean(ax, grid="x", ticks=False)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.7, n - 0.3)
        ref = [r[ref_key] for r in rows]
        champ = [r[champ_key] for r in rows]
        ax.barh(y + h / 2, ref, height=h, color="#cbd5e1", label="Reference model (a single decision tree)")
        ax.barh(y - h / 2, champ, height=h, color=S.BLUE, label="Chosen model")
        top = max(ref)
        for yi, a, b in zip(y, ref, champ):
            ax.text(a + top * 0.015, yi + h / 2, f"{a:.2f}", va="center", fontsize=8.2, color=S.MUTED)
            ax.text(b + top * 0.015, yi - h / 2, f"{b:.2f}", va="center", fontsize=8.8, color=S.BLUE, fontweight="bold")
        ax.set_xlim(0, top * 1.22)
        ax.set_yticks(y)
        ax.set_yticklabels([r["name"] for r in rows] if label_ref else [], fontsize=10.5, color=S.INK, fontweight="bold")
        S.label(ax, title, y=1.02)

    pair(ax_mae, "reference", "champion", "MAE: the typical miss", label_ref=True)
    pair(ax_rmse, "reference_rmse", "champion_rmse", "RMSE: big misses weigh more")
    handles, labels = ax_mae.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.40, 0.865), ncol=2, fontsize=9, handlelength=1.2, columnspacing=2.0)
    ax_mae.set_xlabel("Mean absolute error", fontsize=9)
    ax_rmse.set_xlabel("Root mean squared error", fontsize=9)
    # the gain of the chosen model over the reference, on both measures
    S.clean(ax_gain, grid="x", ticks=False)
    ax_gain.spines["left"].set_visible(False)
    ax_gain.set_ylim(-0.7, n - 0.3)
    g_mae = [100 * (r["reference"] - r["champion"]) / r["reference"] for r in rows]
    g_rmse = [100 * (r["reference_rmse"] - r["champion_rmse"]) / r["reference_rmse"] for r in rows]
    ax_gain.barh(y + h / 2, g_mae, height=h, color=S.GREEN, label="MAE")
    ax_gain.barh(y - h / 2, g_rmse, height=h, color="#86efac", label="RMSE")
    top = max(g_mae + g_rmse)
    for yi, a, b in zip(y, g_mae, g_rmse):
        ax_gain.text(a + top * 0.02, yi + h / 2, f"{a:.1f}%", va="center", fontsize=8.8, color=S.GREEN, fontweight="bold")
        ax_gain.text(b + top * 0.02, yi - h / 2, f"{b:.1f}%", va="center", fontsize=8.5, color="#15803d")
    ax_gain.set_xlim(0, top * 1.3)
    ax_gain.set_yticks([])
    ax_gain.set_xlabel("Lower error than the reference (%)", fontsize=9)
    ax_gain.legend(loc="lower right", fontsize=8.5, handlelength=1.2)
    S.label(ax_gain, "Gain of the chosen model", y=1.02)
    return S.save(fig, "06_forecast_skill")
