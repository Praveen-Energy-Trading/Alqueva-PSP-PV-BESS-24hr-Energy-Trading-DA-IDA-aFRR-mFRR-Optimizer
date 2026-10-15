"""backtest_figures.py - the two figures about a year of results: where the money comes from, and how bad a bad day can be."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch

from . import _style as S

DA, AFRR, MFRR, TOTAL = "realised_revenue_eur", "realised_afrr_capacity_eur", "realised_mfrr_capacity_eur", "realised_total_pnl_eur"


def _card(fig, x, y, w, h, title, value, color):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
                                  facecolor=S.PANEL, edgecolor=S.GRID, linewidth=0.8, mutation_aspect=S.W / S.H))
    fig.text(x + 0.012, y + h * 0.70, title, fontsize=8.5, color=S.MUTED, va="center", ha="left")
    fig.text(x + 0.012, y + h * 0.31, value, fontsize=14, fontweight="bold", color=color, va="center", ha="left")


def _eur(v: float) -> str:
    return f"−€{abs(v):,.0f}" if v < 0 else f"€{v:,.0f}"


# ------------------------------------------------------------------------------------------------------------------------------------ 7
def revenue_sources(bt: dict):
    d = bt["daily"].sort_values("date")
    parts = [("Day-ahead energy", DA, S.BLUE), ("aFRR capacity", AFRR, S.TEAL), ("mFRR capacity", MFRR, "#a78bfa")]
    total = float(d[TOTAL].sum())
    fig = S.canvas("Where the money comes from",
                   f"A year of daily results, {d['date'].min():%d-%m-%Y} to {d['date'].max():%d-%m-%Y}, valued at the prices that really cleared",
                   "Source: rolling backtest of the pipeline (day-ahead energy at the OMIE price; aFRR and mFRR capacity at the REN price)", accent=S.GREEN)
    cards = [("Total over the year", _eur(total), S.INK), ("Average per day", _eur(float(d[TOTAL].mean())), S.BLUE),
             ("Best day", _eur(float(d[TOTAL].max())), S.GREEN), ("Worst day", _eur(float(d[TOTAL].min())), S.ROSE)]
    for i, (t, v, c) in enumerate(cards):
        _card(fig, 0.032 + i * 0.2415, 0.775, 0.2325, 0.085, t, v, c)
    # cumulative, stacked
    ax = fig.add_axes([0.075, 0.115, 0.585, 0.585])
    S.clean(ax)
    x = d["date"].to_numpy()
    base = np.zeros(len(d))
    for name, col, colour in parts:
        cum = d[col].fillna(0).cumsum().to_numpy() / 1e6
        ax.fill_between(x, base, base + cum, color=colour, alpha=0.88, linewidth=0, label=name)
        ax.plot(x, base + cum, color="white", linewidth=0.8)
        base = base + cum
    ax.set_ylabel("Cumulative revenue (million EUR)", fontsize=9)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, base.max() * 1.06)
    ax.legend(loc="upper left", fontsize=9, handlelength=1.2)
    S.label(ax, "Revenue added up over the year", y=1.03)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
    for lab in ax.get_xticklabels():
        lab.set_fontsize(8.5)
    # shares
    bx = fig.add_axes([0.725, 0.115, 0.24, 0.585])
    S.clean(bx, grid="", ticks=False)
    for sp in bx.spines.values():
        sp.set_visible(False)
    shares = [(name, float(d[col].sum()) / total * 100, colour) for name, col, colour in parts]
    ypos = np.arange(len(shares))[::-1]
    bx.barh(ypos, [s for _, s, _ in shares], height=0.5, color=[c for _, _, c in shares])
    for yi, (name, share, colour) in zip(ypos, shares):
        bx.text(0, yi + 0.42, name, fontsize=10, fontweight="bold", color=S.INK, va="bottom")
        bx.text(share + 2, yi, f"{share:.1f}%", fontsize=12, fontweight="bold", color=colour, va="center")
    bx.set_xlim(0, 112)
    bx.set_ylim(-0.6, len(shares) - 0.1)
    bx.set_xticks([])
    bx.set_yticks([])
    S.label(bx, "Share of the year's revenue", y=1.03)
    return S.save(fig, "07_revenue_sources")


# ------------------------------------------------------------------------------------------------------------------------------------ 8
def _pick(risk: dict, section_prefix: str, label_prefix: str, contains: str = ""):
    for (sec, lab), val in risk.items():
        if sec and sec.startswith(section_prefix) and lab.startswith(label_prefix) and contains in lab:
            return float(val)
    return None


def risk(bt: dict):
    d = bt["daily"].sort_values("date")
    pnl = d[TOTAL].dropna().to_numpy(dtype=float)
    r = bt["risk"]
    var95, cvar95 = _pick(r, "Historical", "VaR(95%)"), _pick(r, "Historical", "CVaR(95%)")
    var99, cvar99 = _pick(r, "Historical", "VaR(99%)"), _pick(r, "Historical", "CVaR(99%)")
    mc_var95 = _pick(r, "Monte Carlo Bootstrap (VaR 95%)", "VaR(95%)", "mean")
    mc_cvar95 = _pick(r, "Monte Carlo Bootstrap (VaR 95%)", "CVaR(95%)", "mean")
    mc_var99 = _pick(r, "Monte Carlo Bootstrap (VaR 99%)", "VaR(99%)", "mean")
    mc_cvar99 = _pick(r, "Monte Carlo Bootstrap (VaR 99%)", "CVaR(99%)", "mean")
    fig = S.canvas("How bad can a bad day be?",
                   f"The {len(pnl)} daily results of the backtest: the weak tail of the distribution is what the risk numbers measure",
                   "Source: rolling backtest (historical simulation and a 10,000-draw Monte Carlo bootstrap of the daily results)", accent=S.ROSE)
    ax = fig.add_axes([0.075, 0.115, 0.575, 0.715])
    S.clean(ax)
    bins = np.linspace(pnl.min(), pnl.max(), 36)
    counts, edges = np.histogram(pnl, bins=bins)
    width = edges[1] - edges[0]
    centres = edges[:-1] + width / 2
    colours = [S.ROSE if (var95 is not None and c < var95) else S.BLUE for c in centres]
    ax.bar(centres, counts, width=width * 0.92, color=colours, alpha=0.9)
    ymax = counts.max() * 1.28
    for value, name, colour in ((var95, "5th-percentile day", S.ROSE), (var99, "1st-percentile day", "#9f1239"), (float(np.mean(pnl)), "Average day", S.INK)):
        if value is None:
            continue
        ax.axvline(value, color=colour, linewidth=1.6, linestyle=(0, (4, 3)), label=f"{name}: {_eur(value)}")
    ax.legend(loc="upper right", fontsize=9, handlelength=2.2, labelspacing=0.7)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Number of days", fontsize=9)
    ax.set_xlabel("Daily result (EUR)", fontsize=9)
    ax.xaxis.set_major_formatter(lambda v, _: "0" if abs(v) < 1 else f"{v / 1000:,.0f}k")
    S.label(ax, "Distribution of the daily result", y=1.03)
    # table of the risk numbers
    x0, y0 = 0.70, 0.80
    fig.text(x0, y0, "Risk numbers", fontsize=10.5, fontweight="bold", color=S.INK, va="bottom")
    heads = ["", "History", "Monte Carlo"]
    cx = [x0, x0 + 0.115, x0 + 0.195]
    for xx, hd in zip(cx, heads):
        fig.text(xx + (0.073 if hd else 0), y0 - 0.045, hd, fontsize=8.5, color=S.MUTED, ha="right" if hd else "left", va="center")
    rows = [("5th-percentile day", var95, mc_var95, "VaR 95%"), ("Average of the worst 5%", cvar95, mc_cvar95, "CVaR 95%"),
            ("1st-percentile day", var99, mc_var99, "VaR 99%"), ("Average of the worst 1%", cvar99, mc_cvar99, "CVaR 99%")]
    yy = y0 - 0.115
    for name, h, m, tag in rows:
        fig.text(x0, yy + 0.016, name, fontsize=9.3, color=S.TEXT, va="center", fontweight="bold")
        fig.text(x0, yy - 0.012, tag, fontsize=8, color=S.FAINT, va="center")
        for xx, v in ((cx[1] + 0.073, h), (cx[2] + 0.073, m)):
            fig.text(xx, yy + 0.002, _eur(v) if v is not None else "–", fontsize=10.5, fontweight="bold", color=S.ROSE if (v is not None and v < 0) else S.INK, ha="right", va="center")
        yy -= 0.095
    fig.text(x0, yy + 0.02, "\n".join(["The weakest day of the year: " + _eur(float(pnl.min())), "A negative risk number means the average", "of those bad days was a loss."]),
             fontsize=8.8, color=S.MUTED, va="top", linespacing=1.45)
    return S.save(fig, "08_risk")
