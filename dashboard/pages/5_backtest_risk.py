"""Backtest & Portfolio Risk - one page, one section per market, then the comparison and the portfolio risk.

Order of the page (scroll top to bottom):
    Headline numbers  ->  Day-ahead  ->  Intraday (IDA1-3, XBID)  ->  Reserves (aFRR, mFRR capacity)  ->  Comparison of the markets  ->  Portfolio risk
Everything comes from runtime/reports/backtest_*.xlsx (sheets Backtest, Summary, Risk)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data
import ml_ui
import theme

st.title("📈 Backtest & Portfolio Risk")

DA_COL = "realised_revenue_eur"
AFRR_COL = "realised_afrr_capacity_eur"
MFRR_COL = "realised_mfrr_capacity_eur"
TOTAL_COL = "realised_total_pnl_eur"
GATES = [("Day-ahead", DA_COL), ("IDA1", "realised_ida1_revenue_eur"), ("IDA2", "realised_ida2_revenue_eur"),
         ("IDA3", "realised_ida3_revenue_eur"), ("XBID", "realised_xbid_revenue_eur")]


def _eur(v) -> str:
    return "–" if v is None or pd.isna(v) else (f"−€{abs(v):,.0f}" if v < 0 else f"€{v:,.0f}")


def _thousands(t: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Whole-euro columns as text with thousands separators (the table formatter has none); a missing value stays missing."""
    t = t.copy()
    for c in cols:
        t[c] = t[c].map(lambda v: None if pd.isna(v) else f"{v:,.0f}")
    return t


def _section(title: str, definition: str) -> None:
    st.markdown("---")
    st.subheader(title)
    st.markdown(definition)


def _daily(bt: pd.DataFrame) -> pd.DataFrame:
    d = bt.copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _headline(bt: pd.DataFrame, report: dict) -> None:
    by_label = {l: v for l, v, is_hdr in report.get("Summary", []) if not is_hdr}
    days, feasible, passed = by_label.get("Days"), by_label.get("Feasible"), by_label.get("Checker passed")
    if days and feasible is not None and passed is not None:
        n_days, n_feas, n_pass = int(days), int(feasible), int(passed)
        if n_feas == n_days and n_pass == n_days:
            st.success(f"🟢 {n_feas}/{n_days} days feasible, {n_pass}/{n_days} passed the physical checker")
        else:
            st.warning(f"🟡 {n_feas}/{n_days} days feasible, {n_pass}/{n_days} passed the physical checker"
                       f" -  {n_days - n_feas} infeasible, {n_days - n_pass} checker failure(s)")
    tot = bt[TOTAL_COL].dropna()
    ml_ui.cards([("Days backtested", f"{len(bt):,}"), ("Profitable days", f"{(tot > 0).sum():,} of {len(tot):,}")], ncols=4, pre_formatted=True)


def _bars_with_cumulative(d: pd.DataFrame, col: str, name: str, colour: str) -> None:
    s = d[["date", col]].dropna()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=s["date"], y=s[col], name=f"{name} per day", marker_color=colour), secondary_y=False)
    fig.add_trace(go.Scatter(x=s["date"], y=s[col].cumsum(), name="Cumulative", mode="lines", line=dict(color=theme.COLOR_PRICE, width=2)), secondary_y=True)
    theme.style_fig(fig, height=340)
    fig.update_yaxes(title_text="EUR per day", secondary_y=False, gridcolor=theme.GRIDLINE)
    fig.update_yaxes(title_text="EUR cumulative", secondary_y=True, showgrid=False)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, width="stretch")


def _latest_run(gate: str, label: str):
    """(date, value) of one gate of the day selected in the sidebar, from its gate ticket (the run of run_production.py): `label` is the
    revenue line to read ("Energy revenue", "Capacity revenue", "Improvement vs committed"). It is valued at forecast prices, not realised
    ones, so it is shown beside the backtest and never inside its bars or lines. None without a run."""
    day = st.session_state.get("selected_date")
    if not day:
        return None
    ticket = next((t for t in data.all_gate_tickets(day) if t["gate"] == gate), None)
    value = next((v for lab, v in (ticket or {}).get("revenue_items", []) if lab == label), None)
    return (day, value) if value is not None else None


def _better_than(series: pd.Series, value: float) -> str:
    return f"better than {100.0 * (series.dropna() < value).mean():.0f}% of the past days"


def _day_ahead(d: pd.DataFrame) -> None:
    _section("Day-ahead market",
             "Energy sold the day before delivery, hour by hour, at the OMIE day-ahead price. It carries most of the plant's revenue. "
             "Revenue is the plan from the optimiser valued at the price that really cleared.")
    s = d[DA_COL].dropna()
    best, worst = d.loc[s.idxmax()], d.loc[s.idxmin()]
    latest = _latest_run("DA", "Energy revenue")
    first = [("Average day-ahead revenue per day", _eur(s.mean())), ("Median day", _eur(s.median()))]
    if latest:
        day, value = latest
        first.insert(0, (f"Latest run, {day} (valued at forecast prices)", _eur(value)))
        first.append(("Latest run against the past days", _better_than(s, value)))
    ml_ui.cards(first, ncols=4, pre_formatted=True)
    ml_ui.cards([
                 ("Best day", f"{_eur(best[DA_COL])} · {best['date']:%d %b %Y}"), ("Worst day", f"{_eur(worst[DA_COL])} · {worst['date']:%d %b %Y}"),
                 ("Days with the real OMIE price", f"{(d['realised_price_source'] == 'OMIE_LIVE').sum():,} of {len(d):,}"),
                 ("Price forecast error (MAE)", f"{d['price_mae'].mean():.1f} EUR/MWh"),
                 ("Price forecast error (RMSE)", f"{d['price_rmse'].mean():.1f} EUR/MWh"),
                 ("Days earning below zero", f"{(s < 0).sum():,}")], ncols=4, pre_formatted=True)
    _bars_with_cumulative(d, DA_COL, "Day-ahead revenue", theme.COLOR_GEN)


def _intraday(d: pd.DataFrame) -> None:
    _section("Intraday markets (IDA1, IDA2, IDA3 and XBID)",
             "After the day-ahead gate the plan is re-optimised at three intraday auctions (IDA1, IDA2, IDA3) and in the continuous XBID market, "
             "each time with fresher forecasts. Each step re-values the whole day's position, so the figures are not added together; "
             "the useful number is the **change against the step before**, on the days both steps exist.")
    rows, prev_col, prev_name = [], None, None
    run_key = {"IDA1": "IDA1", "IDA2": "IDA2", "IDA3": "IDA3", "XBID": "XBID"}
    for name, col in GATES:
        s = d[col]
        row = {"Market": name, "Days with a price": int(s.notna().sum()), "Average revenue per day (EUR)": s.mean()}
        if prev_col:
            both = d[[prev_col, col]].dropna()
            delta = both[col] - both[prev_col]
            row.update({"Compared with": prev_name, "Days compared": len(both), "Average change per day (EUR)": delta.mean(),
                        "Days improved": int((delta > 0).sum()), "Share of days improved (%)": 100.0 * (delta > 0).mean()})
            latest = _latest_run(run_key[name], "Improvement vs committed")
            if latest:
                row.update({"Latest run, forecast prices (EUR)": latest[1], "Latest run against the past days": _better_than(delta, latest[1])})
        rows.append(row)
        prev_col, prev_name = col, name
    t = pd.DataFrame(rows)
    if "Latest run, forecast prices (EUR)" in t.columns:
        t["Latest run, forecast prices (EUR)"] = t["Latest run, forecast prices (EUR)"].map(lambda v: None if pd.isna(v) else f"{v:,.0f}")
    ml_ui.table(_thousands(t.fillna({"Compared with": "–"}), ["Average revenue per day (EUR)", "Average change per day (EUR)"]),
                formats={"Share of days improved (%)": "%.0f"})
    chg = t.dropna(subset=["Average change per day (EUR)"])
    fig = go.Figure(go.Bar(x=chg["Market"], y=chg["Average change per day (EUR)"], marker_color=[theme.COLOR_GEN if v >= 0 else theme.COLOR_PUMP for v in chg["Average change per day (EUR)"]],
                           text=[_eur(v) for v in chg["Average change per day (EUR)"]], textposition="outside", name="Average change per day"))
    theme.style_fig(fig, height=320, yaxis_title="EUR per day, against the step before", legend=False)
    fig.update_yaxes(zerolinecolor=theme.INK_MUTED)
    st.plotly_chart(fig, width="stretch")
    st.caption("XBID is backtested for its first window only")
    fig2 = go.Figure()
    for (name, col), (pname, pcol) in zip(GATES[1:], GATES[:-1]):
        both = d[["date", pcol, col]].dropna()
        fig2.add_trace(go.Scatter(x=both["date"], y=(both[col] - both[pcol]).cumsum(), mode="lines", name=f"{name} vs {pname}"))
    theme.style_fig(fig2, height=320, yaxis_title="EUR, cumulative change")
    fig2.update_layout(hovermode="x unified")
    st.plotly_chart(fig2, width="stretch")


def _reserves(d: pd.DataFrame) -> None:
    _section("Reserve markets (aFRR and mFRR capacity)",
             "Balancing reserve is paid for being available, whether or not the grid operator calls on it. This page counts the capacity payment at the real REN "
             "price; the payment for energy actually activated is not included.")
    a, m = d[AFRR_COL].dropna(), d[MFRR_COL].dropna()
    latest_cards = []
    for name, gate, series in (("aFRR", "AFRR", a), ("mFRR", "MFRR", m)):
        latest = _latest_run(gate, "Capacity revenue")
        if latest:
            latest_cards += [(f"{name} capacity, latest run {latest[0]} (forecast prices)", _eur(latest[1])), (f"{name}: latest run against the past days", _better_than(series, latest[1]))]
    if latest_cards:
        ml_ui.cards(latest_cards, ncols=4, pre_formatted=True)
    ml_ui.cards([("aFRR capacity revenue per day (average)", _eur(a.mean())), ("mFRR capacity revenue per day (average)", _eur(m.mean())),
                 ("Days with the real REN price", f"{(d['realised_reserve_price_source'] == 'REN_LIVE').sum():,} of {len(d):,}")], ncols=3, pre_formatted=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["date"], y=d[AFRR_COL], name="aFRR capacity", marker_color=theme.COLOR_UP))
    fig.add_trace(go.Bar(x=d["date"], y=d[MFRR_COL], name="mFRR capacity", marker_color=theme.COLOR_PUMP))
    theme.style_fig(fig, height=340, yaxis_title="EUR per day", barmode="stack")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, width="stretch")


def _comparison(d: pd.DataFrame) -> None:
    _section("Comparing the markets",
             "What each market contributed to the realised P&L (day-ahead energy at the real OMIE price plus aFRR and mFRR capacity at the real REN price). The intraday steps are left out because they re-value the same energy rather than add to it.")
    parts = [("Day-ahead energy", DA_COL, theme.COLOR_GEN), ("aFRR capacity", AFRR_COL, theme.COLOR_UP), ("mFRR capacity", MFRR_COL, theme.COLOR_PUMP)]
    total = d[TOTAL_COL].sum()
    t = pd.DataFrame({"Market": [p[0] for p in parts]})
    t["Share of P&L (%)"] = [100.0 * d[p[1]].sum() / total for p in parts]
    t["Average per day (EUR)"] = [d[p[1]].mean() for p in parts]
    t["Best day (EUR)"] = [d[p[1]].max() for p in parts]
    t["Worst day (EUR)"] = [d[p[1]].min() for p in parts]
    ml_ui.table(_thousands(t, ["Average per day (EUR)", "Best day (EUR)", "Worst day (EUR)"]), formats={"Share of P&L (%)": "%.1f"})
    fig = go.Figure()
    for name, col, colour in parts:
        fig.add_trace(go.Scatter(x=d["date"], y=d[col].fillna(0).cumsum(), mode="lines", stackgroup="one", name=name, line=dict(color=colour, width=1)))
    theme.style_fig(fig, height=340, yaxis_title="EUR, cumulative")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, width="stretch")


def _risk(report: dict) -> None:
    st.markdown("---")
    st.subheader("Portfolio risk")
    st.markdown("How bad a day can get, from the realised daily P&L of this backtest.")
    if "Risk" not in report:
        st.info("This backtest report has no Risk sheet - risk metrics weren't computed for this run.")
        return
    section = None
    section_rows: dict[str, list[tuple[str, object]]] = {}
    for label, value, is_hdr in report["Risk"]:
        if is_hdr and label.startswith("---"):
            section = label.strip("- ").strip()
            section_rows[section] = []
        elif not is_hdr and section:
            section_rows[section].append((label, value))

    hist = dict(section_rows.get("Historical Simulation", []))
    mc_95 = dict(section_rows.get("Monte Carlo Bootstrap (VaR 95%)", []))
    mc_99 = dict(section_rows.get("Monte Carlo Bootstrap (VaR 99%)", []))

    def pick(src, prefix, need_mean=False):
        return next((v for k, v in src.items() if k.startswith(prefix) and (not need_mean or "mean" in k)), None)

    hist_y = [pick(hist, "VaR(95%)"), pick(hist, "CVaR(95%)"), pick(hist, "VaR(99%)"), pick(hist, "CVaR(99%)")]
    if any(v is not None for v in hist_y):
        st.markdown("**VaR / CVaR - how bad could a day get**")
        metrics_x = ["VaR 95%", "CVaR 95%", "VaR 99%", "CVaR 99%"]
        mc_x, mc_y = [], []
        for label, val in [("VaR 95%", pick(mc_95, "VaR(95%)", True)), ("CVaR 95%", pick(mc_95, "CVaR(95%)", True)),
                           ("VaR 99%", pick(mc_99, "VaR(99%)", True)), ("CVaR 99%", pick(mc_99, "CVaR(99%)", True))]:
            if val is not None:
                mc_x.append(label)
                mc_y.append(val)
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(x=metrics_x, y=hist_y, name="Historical", marker_color=theme.COLOR_GEN))
        if mc_x:
            fig_var.add_trace(go.Bar(x=mc_x, y=mc_y, name="Monte Carlo", marker_color=theme.COLOR_PUMP))
        theme.style_fig(fig_var, height=320, yaxis_title="EUR (downside from mean)", barmode="group")
        st.plotly_chart(fig_var, width="stretch")
        st.caption("Historical = worst days actually observed in the backtest. Monte Carlo = same estimate "
                   "from 10,000 bootstrap resamples, for a confidence check on the historical number")

    # Sharpe ratio and max drawdown stay out of the page (unstable with a small day count); they are in the Excel report.
    for section, rows in section_rows.items():
        if section == "Risk-Adjusted":
            continue
        st.markdown(f"**{section}**")
        ml_ui.cards(rows, ncols=4)


@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    files = data.list_backtest_reports()
    if not files:
        st.info("No backtest reports found in `runtime/reports/backtest_*.xlsx`. "
                "Run `phase_7_backtesting_and_validation/run_backtest.py` to generate one.")
        return

    chosen = st.selectbox("Backtest report", files, format_func=lambda p: p.name)
    report = data.load_backtest_report(chosen)
    if "Backtest" not in report:
        st.info("This report has no daily sheet to break down by market.")
        return
    d = _daily(report["Backtest"])
    for col in (DA_COL, AFRR_COL, MFRR_COL, TOTAL_COL, "price_mae", "price_rmse"):
        if col not in d.columns:
            d[col] = pd.NA
    _headline(d, report)
    _day_ahead(d)
    _intraday(d)
    _reserves(d)
    _comparison(d)
    _risk(report)


_render()
