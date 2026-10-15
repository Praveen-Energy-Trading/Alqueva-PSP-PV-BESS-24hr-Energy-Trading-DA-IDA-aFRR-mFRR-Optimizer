"""day_figures.py - the four figures about one solved day: the plan, the capacity stack, the water and battery, and the machine's efficiency."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch

from . import _data as D
from . import _style as S


def _step(ax, y, **kw):
    """A series drawn as one flat step per quarter-hour (ISP i covers [i-1, i])."""
    return ax.stairs(y, np.arange(len(y) + 1), **kw)


MFRR = "#a78bfa"      # mFRR, in both directions


def _kpi(fig, x, y, w, h, title, value, color):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
                                  facecolor=S.PANEL, edgecolor=S.GRID, linewidth=0.8, mutation_aspect=S.W / S.H))
    fig.text(x + 0.012, y + h * 0.68, title, fontsize=8.5, color=S.MUTED, va="center", ha="left")
    fig.text(x + 0.012, y + h * 0.30, value, fontsize=14, fontweight="bold", color=color, va="center", ha="left")


# ------------------------------------------------------------------------------------------------------------------------------------ 2
def day_plan(day: D.Day, plant: dict):
    n = D.N_ISP
    price = day.price if day.price is not None else np.zeros(n)
    sold = float(np.clip(day.net, 0, None).sum() * 0.25)
    bought = float(-np.clip(day.net, None, 0).sum() * 0.25)
    rev = day.solver.get("energy_revenue_eur")
    fig = S.canvas("One day, one plan",
                   f"What the optimiser decided for {day.date[8:10]}-{day.date[5:7]}-{day.date[:4]}: sell when the price is high, pump when it is low",
                   "Source: solved day (component store) and the day-ahead prices of the run", accent=S.BLUE)
    for i, (t, v, c) in enumerate([("Energy revenue (day-ahead)", f"€{rev:,.0f}" if rev else "–", S.BLUE), ("Energy sold", f"{sold:,.0f} MWh", S.INK),
                                   ("Energy bought to pump", f"{bought:,.0f} MWh", S.ORANGE),
                                   ("Price range", f"€{price.min():,.0f} – €{price.max():,.0f} /MWh", S.VIOLET)]):
        _kpi(fig, 0.032 + i * 0.2415, 0.775, 0.2325, 0.085, t, v, c)
    gs = fig.add_gridspec(2, 1, left=0.075, right=0.968, top=0.725, bottom=0.15, hspace=0.20, height_ratios=[1, 2.1])
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None)
    # price
    S.clean(a1)
    _step(a1, price, fill=True, color=S.VIOLET, alpha=0.12, linewidth=0)
    _step(a1, price, color=S.VIOLET, linewidth=2.0)
    i_lo, i_hi = int(price.argmin()), int(price.argmax())
    for i, txt, dy in ((i_lo, "lowest", 24), (i_hi, "highest", 12)):
        a1.annotate(f"€{price[i]:,.0f}  {txt}", (i + 0.5, price[i]), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=8.5, color=S.VIOLET, fontweight="bold")
        a1.plot([i + 0.5], [price[i]], "o", color="white", markeredgecolor=S.VIOLET, markeredgewidth=2, markersize=6, zorder=5)
    a1.set_ylabel("EUR/MWh", fontsize=9)
    a1.set_ylim(max(0, price.min() * 0.75), price.max() * 1.16)
    S.clock_axis(a1, n)
    a1.tick_params(labelbottom=False)
    S.label(a1, "Day-ahead price", y=1.05)
    # plan
    S.clean(a2)
    up = np.zeros(n)
    for series, color, name in ((day.turbine, S.BLUE, "Turbines"), (day.pv_used, S.TEAL, "PV"), (day.bess_discharge, S.AMBER, "Battery discharge")):
        if series.max() < 0.05:
            continue
        a2.stairs(up + series, np.arange(n + 1), baseline=up, fill=True, color=color, alpha=0.9, linewidth=0, label=name)
        up = up + series
    dn = np.zeros(n)
    for series, color, name in ((day.pump, S.ORANGE, "Pumps"), (day.bess_charge, "#fbbf24", "Battery charge")):
        if series.max() < 0.05:
            continue
        a2.stairs(-(dn + series), np.arange(n + 1), baseline=-dn, fill=True, color=color, alpha=0.9, linewidth=0, label=name)
        dn = dn + series
    _step(a2, day.net, color=S.INK, linewidth=1.3, label="Net to the grid")
    a2.axhline(0, color=S.TEXT, linewidth=0.9)
    lim = max(up.max(), dn.max()) * 1.12
    a2.set_ylim(-lim, lim)
    a2.set_ylabel("MW   (generation +, pumping −)", fontsize=9)
    S.clock_axis(a2, n)
    S.label(a2, "Plant dispatch", y=1.02)
    a2.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=9, handlelength=1.2, columnspacing=1.6)
    a1.set_xlim(0, n)
    return S.save(fig, "02_day_plan")


# ------------------------------------------------------------------------------------------------------------------------------------ 3
def capacity_stack(day: D.Day, plant: dict):
    h = day.hourly
    hours = np.arange(24)
    net = np.array(h["Plant_net_final_MW"], dtype=float) if "Plant_net_final_MW" in h else np.array(h["Plant_net_DA_MW"], dtype=float)
    afr_up, mfr_up = np.array(h["aFRR_up_MW"], dtype=float), np.array(h["mFRR_up_MW"], dtype=float)
    afr_dn, mfr_dn = np.array(h["aFRR_dn_MW"], dtype=float), np.array(h["mFRR_dn_MW"], dtype=float)
    gen_cap, pump_cap, fcr = plant["derived"]["gen_cap_mw"], plant["derived"]["pump_cap_mw"], plant["derived"]["fcr_mw"]
    fig = S.canvas("The capacity stack",
                   "Every MW of the plant is promised once: energy first, then the reserve held back for the grid, inside the plant's limits",
                   "Source: daily report of the run (hourly dispatch and reserve offers); limits from the plant configuration", accent=S.TEAL)
    gs = fig.add_gridspec(2, 1, left=0.075, right=0.968, top=0.80, bottom=0.095, hspace=0.34)
    up_ax, dn_ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    x = np.arange(25)
    # upward: energy sold, then aFRR up, then mFRR up, up to the generation limit
    S.clean(up_ax)
    sell = np.clip(net, 0, None)
    base = np.zeros(24)
    for series, color, name in ((sell, S.BLUE, "Energy sold"), (afr_up, S.TEAL, "aFRR up (held back)"), (mfr_up, MFRR, "mFRR up (held back)")):
        up_ax.stairs(base + series, x, baseline=base, fill=True, color=color, alpha=0.92, linewidth=0, label=name)
        base = base + series
    up_ax.hlines(gen_cap - fcr, 0, 24, color=S.ROSE, linewidth=1.4, linestyle=(0, (5, 3)), label=f"Limit {gen_cap - fcr:,.0f} MW")
    up_ax.set_ylim(0, gen_cap * 1.06)
    S.label(up_ax, "Upward: producing or ready to produce more", y=1.04)
    up_ax.legend(ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8.8, handlelength=1.2, columnspacing=1.4)
    up_ax.set_ylabel("MW", fontsize=9)
    # downward: energy bought for pumping, then aFRR down, mFRR down, down to the pumping limit
    S.clean(dn_ax)
    buy = np.clip(-net, 0, None)
    base = np.zeros(24)
    for series, color, name in ((buy, S.ORANGE, "Energy bought to pump"), (afr_dn, S.TEAL, "aFRR down"), (mfr_dn, MFRR, "mFRR down")):
        dn_ax.stairs(-(base + series), x, baseline=-base, fill=True, color=color, alpha=0.92 if name != "mFRR down" else 0.6, linewidth=0, label=name)
        base = base + series
    dn_ax.hlines(-(pump_cap - fcr), 0, 24, color=S.ROSE, linewidth=1.4, linestyle=(0, (5, 3)), label=f"Limit {pump_cap - fcr:,.0f} MW")
    dn_ax.set_ylim(-pump_cap * 1.06, 0)
    S.label(dn_ax, "Downward: pumping or ready to pump more", y=1.04)
    dn_ax.legend(ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8.8, handlelength=1.2, columnspacing=1.4)
    dn_ax.set_ylabel("MW", fontsize=9)
    for ax in (up_ax, dn_ax):
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        ax.set_xticklabels([f"{t:02d}:00" for t in range(0, 25, 4)])
    return S.save(fig, "03_capacity_stack")


# ------------------------------------------------------------------------------------------------------------------------------------ 4
def water_and_battery(day: D.Day, plant: dict):
    n = D.N_ISP
    d = plant["derived"]
    fig = S.canvas("Water and battery",
                   "The plan only works if the reservoirs and the battery stay inside their limits and the water ends the day where it started",
                   "Source: solved day (component store); limits from the plant configuration", accent=S.SKY)
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.968, top=0.80, bottom=0.10, hspace=0.42, wspace=0.18)
    x = np.arange(n + 1)
    # upper reservoir
    a = fig.add_subplot(gs[0, 0]); S.clean(a)
    up0 = day.initial.get("upper_reservoir_hm3", float(day.upper[0]))
    a.fill_between(np.arange(n) + 0.5, day.upper, d["upper_min_hm3"], color=S.BLUE, alpha=0.12, linewidth=0)
    a.plot(np.arange(n) + 0.5, day.upper, color=S.BLUE, linewidth=2.2)
    a.axhline(up0, color=S.AMBER, linewidth=1.3, linestyle=(0, (4, 3)))
    a.set_ylabel("hm³", fontsize=9)
    pad = max(day.upper.max() - day.upper.min(), 5) * 0.6
    a.set_ylim(min(day.upper.min(), up0) - pad, max(day.upper.max(), up0) + pad)
    S.label(a, "Alqueva (upper reservoir)", y=1.05); S.clock_axis(a, n)
    a.text(1.0, 1.05, "dashed: level at the start of the day; the day must end at or above it", transform=a.transAxes, fontsize=8, color=S.AMBER, ha="right", va="bottom", fontweight="bold")
    # lower reservoir
    b = fig.add_subplot(gs[0, 1]); S.clean(b)
    b.fill_between(np.arange(n) + 0.5, day.lower, 0, color=S.SKY, alpha=0.18, linewidth=0)
    b.plot(np.arange(n) + 0.5, day.lower, color=S.SKY, linewidth=2.2)
    cap_lo, cap_hi = plant["reservoir"]["lower_min_hm3"], plant["reservoir"]["lower_capacity_hm3"]
    b.axhline(cap_hi, color=S.ROSE, linewidth=1.1, linestyle=(0, (4, 3))); b.axhline(cap_lo, color=S.ROSE, linewidth=1.1, linestyle=(0, (4, 3)))
    b.text(n * 0.01, cap_hi, "full", fontsize=8, color=S.ROSE, va="bottom", fontweight="bold"); b.text(n * 0.01, cap_lo, "minimum", fontsize=8, color=S.ROSE, va="bottom", fontweight="bold")
    b.set_ylim(0, cap_hi * 1.12); b.set_ylabel("hm³", fontsize=9)
    S.label(b, "Pedrógão (lower reservoir)", y=1.05); S.clock_axis(b, n)
    # battery
    c = fig.add_subplot(gs[1, 0]); S.clean(c)
    c.fill_between(np.arange(n) + 0.5, day.bess_soc, 0, color=S.AMBER, alpha=0.16, linewidth=0)
    c.plot(np.arange(n) + 0.5, day.bess_soc, color=S.AMBER, linewidth=2.2)
    c.axhline(d["bess_max_mwh"], color=S.ROSE, linewidth=1.1, linestyle=(0, (4, 3))); c.axhline(d["bess_min_mwh"], color=S.ROSE, linewidth=1.1, linestyle=(0, (4, 3)))
    c.set_ylim(0, plant["bess"]["capacity_mwh"] * 1.1); c.set_ylabel("MWh", fontsize=9)
    S.label(c, f"Battery state of charge  (limits {d['bess_min_mwh']:.1f} – {d['bess_max_mwh']:.1f} MWh)", y=1.05); S.clock_axis(c, n)
    # head
    e = fig.add_subplot(gs[1, 1]); S.clean(e)
    e.plot(np.arange(n) + 0.5, day.head, color=S.VIOLET, linewidth=2.2)
    e.set_ylabel("m", fontsize=9)
    pad = max(day.head.max() - day.head.min(), 0.5) * 0.8
    e.set_ylim(day.head.min() - pad, day.head.max() + pad)
    S.label(e, "Net head the turbines see", y=1.05); S.clock_axis(e, n)
    return S.save(fig, "04_water_and_battery")


# ------------------------------------------------------------------------------------------------------------------------------------ 5
def efficiency_surface(day: D.Day, plant: dict):
    from common_layer.optimisation_model import core_milp_builder as M
    psp = plant["psp"]
    flows = np.linspace(psp["q_turbine_min_m3h"], psp["q_turbine_max_m3h"], 60) / 3600.0      # m3/s per unit
    heads = np.linspace(M.H_MIN_OP, M.H_MAX_OP, 60)
    fn = (np.linspace(psp["q_turbine_min_m3h"], psp["q_turbine_max_m3h"], 60) - psp["q_turbine_min_m3h"]) / (psp["q_turbine_max_m3h"] - psp["q_turbine_min_m3h"])
    hn = (heads - M.H_MIN_OP) / (M.H_MAX_OP - M.H_MIN_OP)
    a0, a1, a2, a3, a4, a5 = M.COEFFS_TRB
    F, Hh = np.meshgrid(fn, hn)
    eta = np.clip(a0 + a1 * F + a2 * Hh + a3 * F * Hh + a4 * F ** 2 + a5 * Hh ** 2, M.ETA_LO, M.ETA_HI)
    fig = S.canvas("The turbine's efficiency surface",
                   "How much power a cubic metre of water makes depends on the flow and on the head; the optimiser works on this surface",
                   "Source: efficiency polynomial of the optimisation model; operating points from the solved day", accent=S.VIOLET)
    ax = fig.add_axes([0.075, 0.115, 0.60, 0.715])
    S.clean(ax, grid="")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("eff", ["#f0f9ff", "#bae6fd", "#38bdf8", "#0284c7", "#0c4a6e"])
    lo, hi = float(eta.min() * 100), float(eta.max() * 100)
    im = ax.imshow(eta * 100, extent=(flows.min(), flows.max(), heads.min(), heads.max()), origin="lower", aspect="auto", cmap=cmap, vmin=lo, vmax=hi,
                   interpolation="bicubic")   # one embedded image, not thousands of vector cells
    cs = ax.contour(flows, heads, eta * 100, levels=np.arange(np.ceil(lo * 4) / 4, hi, 0.25), colors="white", linewidths=0.8, alpha=0.85)
    ax.clabel(cs, fmt="%.2f%%", fontsize=7.5, inline=True)
    ax.axhspan(day.head.min(), day.head.max(), color=S.INK, alpha=0.05, linewidth=0)
    # operating points of the day: one dot per quarter-hour and running unit
    q = day.units_q_turb / 3600.0
    hh = np.repeat(day.head[:, None], q.shape[1], axis=1)
    on = q > 1
    ax.scatter(q[on], hh[on], s=26, facecolor="white", edgecolor=S.INK, linewidth=1.1, zorder=5, label="Where the plant ran on this day")
    ax.set_xlabel("Flow through one turbine (m³/s)", fontsize=9.5)
    ax.set_ylabel("Net head (m)", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=8.8, frameon=True, facecolor="white", edgecolor=S.GRID)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([0.70, 0.115, 0.014, 0.715])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=8.5)
    fig.text(0.70, 0.845, "Efficiency (%)", fontsize=9, color=S.MUTED, ha="left")
    # reading guide
    x0 = 0.755
    fig.text(x0, 0.80, "How to read it", fontsize=11, fontweight="bold", color=S.INK)
    eta_day = float((day.eta_turbine * day.turbine).sum() / day.turbine.sum() * 100) if day.turbine.sum() > 0 else float("nan")
    lines = [["Darker = more of the water's energy", "becomes electricity."],
             ["A fuller reservoir gives more head,", "and more power from the same water."],
             ["The optimiser reads the surface at a", "grid of points and blends them with", "weights (the omega variables), which", "keeps the model linear."],
             ["The dots are where the plan ran each", "unit, every quarter-hour of the day."]]
    y = 0.745
    for t in lines:
        fig.text(x0, y, "\n".join(t), fontsize=9.3, color=S.TEXT, va="top", linespacing=1.4)
        y -= 0.0285 * len(t) + 0.045
    fig.add_artist(FancyBboxPatch((x0 - 0.004, 0.115), 0.215, 0.1, boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
                                  facecolor=S.PANEL, edgecolor=S.GRID, linewidth=0.8, mutation_aspect=S.W / S.H))
    fig.text(x0 + 0.01, 0.185, "Turbine efficiency achieved today", fontsize=8.5, color=S.MUTED, va="center")
    fig.text(x0 + 0.01, 0.145, f"{eta_day:.1f} %", fontsize=15, fontweight="bold", color=S.BLUE, va="center")
    return S.save(fig, "05_efficiency_surface")
