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


def _scorecard(groups: list) -> None:
    """One tidy table of label / value rows under group headings: [(heading, [(label, value), ...]), ...]. Styled like the other tables of the dashboard."""
    import html
    import html_table as ht
    body = ""
    for heading, rows in groups:
        body += f'<tr><th colspan="2" class="sc-head">{html.escape(heading)}</th></tr>'
        body += "".join(f'<tr><td class="sc-label">{html.escape(label)}</td><td class="sc-value">{html.escape(str(value))}</td></tr>' for label, value in rows)
    css = (f"<style>table.sc td.sc-label {{ text-align:left; color:{ht.INK_MUTED}; white-space:normal; padding-right:16px; }}"
           f"table.sc td.sc-value {{ text-align:right; font-weight:600; color:{ht.INK}; white-space:normal; font-variant-numeric:tabular-nums; }}"
           f"table.sc {{ width:100% !important; }}"
           f"table.sc th.sc-head {{ position:static; text-align:left; background:{ht.HEAD_BG}; color:{ht.INK}; font-weight:600; letter-spacing:0.02em; "
           f"border-right:none; padding:8px 12px; }}</style>")
    st.markdown(ht.CSS + css + f'<div class="htab-wrap" style="width:100%"><table class="htab sc"><tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


def _section(title: str, definition: str) -> None:
    st.markdown("---")
    st.subheader(title)
    if definition:
        st.markdown(definition)


def _daily(bt: pd.DataFrame) -> pd.DataFrame:
    d = bt.copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _headline(bt: pd.DataFrame, report: dict) -> None:
    """One line: days feasible and checked, and how many days were profitable (the days backtested are the denominator)."""
    by_label = {l: v for l, v, is_hdr in report.get("Summary", []) if not is_hdr}
    days, feasible, passed = by_label.get("Days"), by_label.get("Feasible"), by_label.get("Checker passed")
    tot = bt[TOTAL_COL].dropna()
    profit = f" · {(tot > 0).sum():,} of {len(tot):,} days profitable"
    if days and feasible is not None and passed is not None:
        n_days, n_feas, n_pass = int(days), int(feasible), int(passed)
        text = f"{n_feas}/{n_days} days feasible · {n_pass}/{n_days} passed the physical checker{profit}"
        if n_feas == n_days and n_pass == n_days:
            st.success(f"🟢 {text}")
        else:
            st.warning(f"🟡 {text} · {n_days - n_feas} infeasible, {n_days - n_pass} checker failure(s)")
    else:
        st.info(f"{len(bt):,} days{profit}")


def _bars_with_cumulative(d: pd.DataFrame, col: str, name: str, colour: str, height: int = 340) -> None:
    s = d[["date", col]].dropna()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=s["date"], y=s[col], name=f"{name} per day", marker_color=colour), secondary_y=False)
    fig.add_trace(go.Scatter(x=s["date"], y=s[col].cumsum(), name="Cumulative", mode="lines", line=dict(color=theme.COLOR_PRICE, width=2)), secondary_y=True)
    theme.style_fig(fig, height=height)
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
    groups = [(f"Past days, valued at real prices ({len(s):,} days)",
               [("Average revenue per day", f"€{s.mean():,.0f}"), ("Median day", f"€{s.median():,.0f}"),
                ("Best day", f"€{best[DA_COL]:,.0f} · {best['date']:%d-%m-%Y}"), ("Worst day", f"−€{abs(worst[DA_COL]):,.0f} · {worst['date']:%d-%m-%Y}" if worst[DA_COL] < 0 else f"€{worst[DA_COL]:,.0f} · {worst['date']:%d-%m-%Y}"),
                ("Days earning below zero", f"{(s < 0).sum():,}"),
                ("Days with the real OMIE price", f"{(d['realised_price_source'] == 'OMIE_LIVE').sum():,} of {len(d):,}")])]
    latest = _latest_run("DA", "Energy revenue")
    if latest:
        groups.append((f"Latest run, {latest[0]} (valued at forecast prices)", [("Day-ahead revenue", f"€{latest[1]:,.0f}"), ("Against the past days", _better_than(s, latest[1]))]))
    table_col, chart_col = st.columns([5, 7])
    with table_col:
        _scorecard(groups)
    with chart_col:
        _bars_with_cumulative(d, DA_COL, "Day-ahead revenue", theme.COLOR_GEN, height=430)


def _intraday(d: pd.DataFrame) -> None:
    _section("Intraday markets (IDA1, IDA2, IDA3 and XBID)", "")
    rows, gains = [], {}
    for (name, col), (_, prev_col) in zip(GATES[1:], GATES[:-1]):
        both = d[["date", prev_col, col]].dropna()
        gain = both[col] - both[prev_col]
        gains[name] = both["date"], gain
        row = {"Market": name, "Days with a price": len(both), "Average gain per day (EUR)": gain.mean(), "Days improved": int((gain > 0).sum()),
               "Share of days improved (%)": 100.0 * (gain > 0).mean()}
        latest = _latest_run(name, "Improvement vs committed")
        if latest:
            row.update({"Latest run, forecast prices (EUR)": latest[1], "Latest run against the past days": _better_than(gain, latest[1])})
        rows.append(row)
    t = pd.DataFrame(rows)
    money = ["Average gain per day (EUR)"] + (["Latest run, forecast prices (EUR)"] if "Latest run, forecast prices (EUR)" in t.columns else [])
    ml_ui.table(_thousands(t, money), formats={"Share of days improved (%)": "%.0f"})
    fig = go.Figure(go.Bar(x=t["Market"], y=t["Average gain per day (EUR)"], marker_color=[theme.COLOR_GEN if v >= 0 else theme.COLOR_PUMP for v in t["Average gain per day (EUR)"]],
                           text=[_eur(v) for v in t["Average gain per day (EUR)"]], textposition="outside", name="Average gain per day"))
    theme.style_fig(fig, height=320, yaxis_title="EUR per day", legend=False)
    fig.update_yaxes(zerolinecolor=theme.INK_MUTED)
    st.plotly_chart(fig, width="stretch")
    st.caption("XBID is backtested for its first window only")
    fig2 = go.Figure()
    for name, (dates, gain) in gains.items():
        fig2.add_trace(go.Scatter(x=dates, y=gain.cumsum(), mode="lines", name=name))
    theme.style_fig(fig2, height=320, yaxis_title="EUR, cumulative gain")
    fig2.update_layout(hovermode="x unified")
    st.plotly_chart(fig2, width="stretch")


def _reserves(d: pd.DataFrame) -> None:
    _section("Reserve markets (aFRR and mFRR capacity)",
             "Balancing reserve is paid for being available, whether or not the grid operator calls on it. This page counts the capacity payment at the real REN "
             "price; the payment for energy actually activated is not included.")
    a, m = d[AFRR_COL].dropna(), d[MFRR_COL].dropna()
    groups = [("Past days, valued at the real REN price",
               [("aFRR capacity, average per day", _eur(a.mean())), ("mFRR capacity, average per day", _eur(m.mean())),
                ("Days with the real REN price", f"{(d['realised_reserve_price_source'] == 'REN_LIVE').sum():,} of {len(d):,}")])]
    for name, gate, series in (("aFRR", "AFRR", a), ("mFRR", "MFRR", m)):
        latest = _latest_run(gate, "Capacity revenue")
        if latest:
            groups.append((f"{name}, latest run {latest[0]} (valued at forecast prices)", [("Capacity revenue", _eur(latest[1])), ("Against the past days", _better_than(series, latest[1]))]))
    table_col, chart_col = st.columns([5, 7])
    with table_col:
        _scorecard(groups)
    with chart_col:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=d["date"], y=d[AFRR_COL], name="aFRR capacity", marker_color=theme.COLOR_UP))
        fig.add_trace(go.Bar(x=d["date"], y=d[MFRR_COL], name="mFRR capacity", marker_color=theme.COLOR_PUMP))
        theme.style_fig(fig, height=430, yaxis_title="EUR per day", barmode="stack")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, width="stretch")


def _comparison(d: pd.DataFrame) -> None:
    _section("Comparing the markets",
             "What each market contributed to the realised P&L (day-ahead energy at the real OMIE price plus aFRR and mFRR capacity at the real REN price). The intraday steps are left out because they re-value the same energy rather than add to it.")
    parts = [("Day-ahead energy", DA_COL, theme.COLOR_GEN, ("DA", "Energy revenue")), ("aFRR capacity", AFRR_COL, theme.COLOR_UP, ("AFRR", "Capacity revenue")),
             ("mFRR capacity", MFRR_COL, theme.COLOR_PUMP, ("MFRR", "Capacity revenue"))]
    total = d[TOTAL_COL].sum()
    groups = []
    for name, col, _, (gate, label) in parts:
        series = d[col].dropna()
        rows = [("Share of the P&L", f"{100.0 * series.sum() / total:.1f}%"), ("Average per day", _eur(series.mean())),
                ("Best day", _eur(series.max())), ("Worst day", _eur(series.min()))]
        latest = _latest_run(gate, label)
        if latest:
            rows.append((f"Latest run {latest[0]} (forecast prices)", _eur(latest[1])))
        groups.append((name, rows))
    table_col, chart_col = st.columns([5, 7])
    with table_col:
        _scorecard(groups)
    with chart_col:
        fig = go.Figure()
        for name, col, colour, _ in parts:
            fig.add_trace(go.Scatter(x=d["date"], y=d[col].fillna(0).cumsum(), mode="lines", stackgroup="one", name=name, line=dict(color=colour, width=1)))
        theme.style_fig(fig, height=330, yaxis_title="EUR, cumulative")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        fig2 = go.Figure(go.Bar(x=[p[0] for p in parts], y=[d[p[1]].mean() for p in parts], marker_color=[p[2] for p in parts],
                                text=[_eur(d[p[1]].mean()) for p in parts], textposition="outside"))
        theme.style_fig(fig2, height=300, yaxis_title="Average EUR per day", legend=False)
        st.plotly_chart(fig2, width="stretch")


def _today_energy_split(day: str):
    """(energy sold, energy bought for pumping) of the day-ahead position of a run, in EUR: volume x price, split by the sign of the volume.
    None when the run has no stored day-ahead position."""
    try:
        from common_layer.database.position_store import PositionStore
        pos = PositionStore().load_position(day, "DA")
    except Exception:
        return None
    if not pos:
        return None
    sold = sum(max(v["volume_mwh"], 0.0) * v["price_eur_mwh"] for v in pos.values())
    bought = sum(min(v["volume_mwh"], 0.0) * v["price_eur_mwh"] for v in pos.values())
    return sold, -bought


def _today_pnl():
    """(date, P&L) of the latest run on the same definition as the backtest's realised P&L: day-ahead energy + aFRR and mFRR capacity, valued at forecast prices."""
    da, af, mf = _latest_run("DA", "Energy revenue"), _latest_run("AFRR", "Capacity revenue"), _latest_run("MFRR", "Capacity revenue")
    if not da:
        return None
    return da[0], da[1] + (af[1] if af else 0.0) + (mf[1] if mf else 0.0)


def _risk(report: dict, d: pd.DataFrame) -> None:
    _section("Portfolio risk (P&L distribution, VaR, CVaR and Monte Carlo)",
             "How the daily P&L is spread over the backtest, how bad a bad day is, and where the latest run falls. P&L is day-ahead energy plus aFRR and mFRR capacity.")
    pnl = d[TOTAL_COL].dropna().astype(float)
    section_rows: dict = {}
    section = None
    for label, value, is_hdr in report.get("Risk", []):
        if is_hdr and label.startswith("---"):
            section = label.strip("- ").strip()
            section_rows[section] = {}
        elif not is_hdr and section:
            section_rows[section][label] = value

    def pick(sec, prefix, need_mean=False):
        return next((v for k, v in section_rows.get(sec, {}).items() if k.startswith(prefix) and (not need_mean or "mean" in k)), None)

    hist = {k: pick("Historical Simulation", k) for k in ("VaR(95%)", "CVaR(95%)", "VaR(99%)", "CVaR(99%)")}
    mc = {"VaR(95%)": pick("Monte Carlo Bootstrap (VaR 95%)", "VaR(95%)", True), "CVaR(95%)": pick("Monte Carlo Bootstrap (VaR 95%)", "CVaR(95%)", True),
          "VaR(99%)": pick("Monte Carlo Bootstrap (VaR 99%)", "VaR(99%)", True), "CVaR(99%)": pick("Monte Carlo Bootstrap (VaR 99%)", "CVaR(99%)", True)}
    if not section_rows:
        st.info("This backtest report has no Risk sheet - risk metrics weren't computed for this run.")

    def e(v):
        return "–" if v is None or pd.isna(v) else _eur(v)

    groups = [(f"Daily P&L, past days ({len(pnl):,})",
               [("Mean", e(pnl.mean())), ("Median", e(pnl.median())), ("Standard deviation", e(pnl.std())), ("Worst day", e(pnl.min())), ("Best day", e(pnl.max()))])]
    if section_rows:
        groups.append(("Value at risk and expected shortfall",
                       [(f"{k.replace('(', ' ').replace(')', '')} (history | Monte Carlo)", f"{e(hist[k])} | {e(mc[k])}") for k in hist]))
    today = _today_pnl()
    if today:
        day, value = today
        rows = [("P&L, total", e(value)), ("Against the past days", _better_than(pnl, value))]
        if hist["VaR(95%)"] is not None:
            rows.append(("Against the 5th-percentile day (VaR 95%)", "above it" if value >= hist["VaR(95%)"] else "below it"))
        split = _today_energy_split(day)
        if split:
            rows += [("Energy sold, day-ahead", e(split[0])), ("Energy bought for pumping", e(split[1])), ("Net energy value", e(split[0] - split[1]))]
        groups.append((f"Latest run, {day} (valued at forecast prices)", rows))

    table_col, chart_col = st.columns([5, 7])
    with table_col:
        _scorecard(groups)
    with chart_col:
        fig = go.Figure(go.Histogram(x=pnl, nbinsx=30, marker_color=theme.COLOR_GEN, name="Days"))
        for label, value, colour, where in (("VaR 95%", hist["VaR(95%)"], theme.COLOR_PUMP, "top right"), ("VaR 99%", hist["VaR(99%)"], theme.COLOR_DOWN, "top left"),
                                            ("Latest run", today[1] if today else None, theme.COLOR_PRICE, "top")):
            if value is not None:
                fig.add_vline(x=value, line_dash="dash", line_color=colour, annotation_text=label, annotation_position=where)
        theme.style_fig(fig, height=330, yaxis_title="Number of days", xaxis_title="Daily P&L (EUR)", legend=False)
        st.plotly_chart(fig, width="stretch")
        if any(v is not None for v in hist.values()):
            names = list(hist)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=[n.replace("(", " ").replace(")", "") for n in names], y=[hist[n] for n in names], name="Historical", marker_color=theme.COLOR_GEN))
            fig2.add_trace(go.Bar(x=[n.replace("(", " ").replace(")", "") for n in names], y=[mc[n] for n in names], name="Monte Carlo", marker_color=theme.COLOR_PUMP))
            theme.style_fig(fig2, height=300, yaxis_title="P&L of the day (EUR)", barmode="group")
            st.plotly_chart(fig2, width="stretch")


@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    files = data.list_backtest_reports()
    if not files:
        st.info("No backtest reports found in `runtime/reports/backtest_*.xlsx`. "
                "Run `phase_7_backtesting_and_validation/run_backtest.py` to generate one.")
        return

    pick_col, status_col = st.columns([2, 3])
    with pick_col:
        chosen = st.selectbox("Backtest report", files, format_func=lambda p: p.name, label_visibility="collapsed")
    report = data.load_backtest_report(chosen)
    if "Backtest" not in report:
        st.info("This report has no daily sheet to break down by market.")
        return
    d = _daily(report["Backtest"])
    for col in (DA_COL, AFRR_COL, MFRR_COL, TOTAL_COL, "price_mae", "price_rmse"):
        if col not in d.columns:
            d[col] = pd.NA
    with status_col:
        _headline(d, report)
    _day_ahead(d)
    _intraday(d)
    _reserves(d)
    _comparison(d)
    _risk(report, d)


_render()
