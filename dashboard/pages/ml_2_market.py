"""ML Lifecycle 2 - Market analysis: what the data looks like before any model is fitted. Told as a story in numbered steps."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ml_data
import ml_ui
import theme

SERIES = {"da_pt": "Day-ahead (PT)", "ida1_pt": "IDA1", "ida2_pt": "IDA2", "ida3_pt": "IDA3", "xbid_pt": "XBID (continuous)",
          "afrr_up": "aFRR capacity up", "afrr_dn": "aFRR capacity down", "mfrr_ap": "mFRR activation price", "imb_short": "Imbalance (short)",
          "imb_long": "Imbalance (long)"}
# Each series keeps one colour on every picture of this page, whatever else is chosen.
SHORT = {"pre_crisis": "Before the crisis", "crisis_pre_cap": "Crisis", "iberian_exception": "Iberian cap", "post_exception": "After the cap", "post_blackout": "After the blackout", "da_15min": "15-minute DA"}
COLOR = {"da_pt": theme.BLUE, "ida1_pt": theme.ORANGE, "ida2_pt": theme.AQUA, "ida3_pt": theme.YELLOW, "xbid_pt": theme.MAGENTA,
         "afrr_up": theme.GREEN, "afrr_dn": theme.VIOLET, "mfrr_ap": theme.RED, "imb_short": "#8791ab", "imb_long": "#6b6a64"}
MUTED, INK, NEUTRAL, CHAMP = theme.INK_MUTED, theme.INK_PRIMARY, "#6b6a64", "#2b2a28"
CARD_LINE = "#e3e1d9"
_cap = lambda h_: h_[:1].upper() + h_[1:] if h_[:1].islower() else h_                                                  # noqa: E731

ml_ui.header("Market analysis", "", step=1)
ml_ui.tour_bar(2)


def _section(title: str, hint: str = "", num: str = "") -> None:
    """A section heading: a number (the steps of the story), a bold title and, under it, one muted line on how to read the section."""
    _n = f"<span style='font-weight:400; color:{NEUTRAL};'>{num}</span>&nbsp; " if num else ""
    st.markdown(f"<div style='margin:1.7rem 0 0.45rem;'><div style='font-size:1.55rem; font-weight:700; color:{INK}; line-height:1.25;'>{_n}{title}</div>"
                + (f"<div style='font-size:0.88rem; color:{MUTED}; margin-top:0.2rem;'>{_cap(hint)}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _mini(title: str, hint: str = "") -> None:
    """The title of a chart or table inside a section."""
    st.markdown(f"<div style='margin:0.6rem 0 0.3rem;'><div style='font-size:1.02rem; font-weight:700; color:{INK};'>{title}</div>"
                + (f"<div style='font-size:0.82rem; color:{MUTED}; margin-top:0.1rem; line-height:1.35;'>{_cap(hint)}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _line(html_: str) -> None:
    """One short note under a table or chart, the key numbers in bold."""
    st.markdown(f"<div style='font-size:0.86rem; color:{theme.INK_SECONDARY}; margin:0.25rem 0 0.9rem;'>{html_}</div>", unsafe_allow_html=True)


def _table(headers: list, rows: list, widths: str | None = None, left: tuple = (0,)) -> None:
    """A table as on the Tuning lab: muted header, thin row lines, the first column left, the others centred."""
    grid = f"display:grid; grid-template-columns:{widths or f'minmax(9rem, 1.5fr) repeat({len(headers) - 1}, minmax(0, 1fr))'}; align-items:center; column-gap:0.6rem;"
    head = (f"<div style='{grid} padding:0.6rem 0 0.4rem; border-bottom:1px solid {CARD_LINE}; font-size:0.72rem; color:{MUTED}; font-weight:600; line-height:1.2;'>"
            + "".join(f"<span style='{'' if j in left else 'text-align:center;'}'>{h}</span>" for j, h in enumerate(headers)) + "</div>")
    body = "".join(f"<div style='{grid} min-height:2.5rem; font-size:0.86rem; border-bottom:1px solid #eeede8;'>"
                   + "".join(f"<span style='{'' if j in left else 'text-align:center;'} color:{INK if str(c) != '–' else MUTED};'>{c}</span>" for j, c in enumerate(row)) + "</div>" for row in rows)
    st.markdown(f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0 1.1rem 0.1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>{head}{body}</div>", unsafe_allow_html=True)


def _pill(text: str, color: str, bg: str) -> str:
    return f"<span style='background:{bg}; color:{color}; border-radius:999px; padding:0.1rem 0.6rem; font-size:0.78rem; font-weight:600; white-space:nowrap;'>{text}</span>"


def _legend_top(fig, y=1.08) -> None:
    fig.update_layout(legend=dict(orientation="h", x=0, y=y, xanchor="left", yanchor="bottom", font=dict(size=12, color=theme.INK_SECONDARY)))


prices = ml_ui.cached(lambda: ml_data.pack("prices_daily"))()
periods = ml_ui.cached(lambda: ml_data.config("market_analysis").get("periods", []))()

if prices.empty:
    ml_ui.not_yet_run("the data pack")
else:
    prices["date"] = pd.to_datetime(prices["date"])
    avail = [s for s in SERIES if s in prices.columns]
    # ======================================================================================== 1. price history
    _section("Price history", "Every market on one time axis, with the market regimes shaded", num="①")
    c1, c2, c3 = st.columns([3, 2, 2], gap="medium")
    pick = c1.multiselect("Series", avail, default=["da_pt"], format_func=SERIES.get, key="ml2_series")
    smooth = c2.selectbox("Smoothing", [1, 7, 30, 90], index=2, format_func=lambda d: "Daily" if d == 1 else f"{d}-day mean", key="ml2_smooth")
    lo, hi = prices["date"].min().date(), prices["date"].max().date()
    rng = c3.slider("Period", lo, hi, (lo, hi), key="ml2_range")
    sub = prices[(prices["date"].dt.date >= rng[0]) & (prices["date"].dt.date <= rng[1])]
    fig = go.Figure()
    for s in pick:
        y = sub[s].rolling(smooth, min_periods=1).mean() if smooth > 1 else sub[s]
        fig.add_trace(go.Scatter(x=sub["date"], y=y, mode="lines", name=SERIES[s], line=dict(width=1.6, color=COLOR[s])))
    shown = 0
    for j, p in enumerate(periods):
        a, b = pd.Timestamp(p["start"]), pd.Timestamp(p["end"])
        if b >= pd.Timestamp(rng[0]) and a <= pd.Timestamp(rng[1]) and a.year < 2099:
            fig.add_vrect(x0=max(a, pd.Timestamp(rng[0])), x1=min(b, pd.Timestamp(rng[1])), fillcolor="#8791ab" if j % 2 == 0 else "#b9b7ae", opacity=0.13,
                          line_width=0, annotation_text=SHORT.get(p["id"], p["id"].replace("_", " ").capitalize()), annotation_position="top left", annotation_font_size=9,
                          annotation_yshift=-13 * (shown % 3))                     # neighbouring labels on two levels, so short bands do not overprint each other
            shown += 1
    theme.style_fig(fig, height=380, yaxis_title="EUR/MWh (EUR/MW/h for aFRR)")
    _legend_top(fig)
    fig.update_layout(margin=dict(t=34))
    starts = [sub.loc[sub[s].notna(), "date"].min() for s in pick if sub[s].notna().any()]
    if starts:                                                                      # the axis starts at the first value of the chosen series, not at the first day of any series
        fig.update_xaxes(range=[min(starts), sub["date"].max()])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    last_day = prices["date"].max()
    rrows = []
    for p in periods:
        a, b = pd.Timestamp(p["start"]), pd.Timestamp(p["end"])
        open_end = b.year >= 2099
        b_ = last_day if open_end else b
        rrows.append([p.get("label", p["id"]), f"{a:%d-%m-%Y}", "Still running" if open_end else f"{b:%d-%m-%Y}", f"{(b_ - a).days + 1:,}"])
    if rrows:
        _mini("The market regimes", "Used to test for structural breaks and to describe results by regime")
        _table(["Regime", "From", "To", "Days"], rrows, widths="minmax(16rem, 3fr) repeat(2, minmax(6rem, 1fr)) minmax(4rem, 0.7fr)")

    # ======================================================================================== 2. the intraday spread
    _section("The intraday spread against day-ahead", "Intraday price minus the day-ahead price of the same day, over the period chosen above", num="②")
    ids = [s for s in ("ida1_pt", "ida2_pt", "ida3_pt", "xbid_pt") if s in sub and "da_pt" in sub]
    sp = pd.DataFrame({"date": sub["date"]})
    for s in ids:
        sp[s] = (sub[s] - sub["da_pt"]).rolling(30, min_periods=10).mean()
    fig = go.Figure()
    for s in ids:
        fig.add_trace(go.Scatter(x=sp["date"], y=sp[s], mode="lines", name=SERIES[s], line=dict(width=1.6, color=COLOR[s])))
    fig.add_hline(y=0, line_color=MUTED, line_dash="dot", annotation_text="0 = day-ahead price", annotation_position="top left")
    theme.style_fig(fig, height=300, yaxis_title="EUR/MWh against day-ahead (30-day mean)")
    _legend_top(fig)
    fig.update_layout(margin=dict(t=34))
    started = sp.dropna(how="all", subset=ids)["date"].min() if ids else pd.NaT                           # the intraday markets start in mid-2024: no empty years on the left
    if pd.notna(started):
        fig.update_xaxes(range=[started, sp["date"].max()])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if ids:
        _mini("The spread in numbers", "Daily values over the period chosen above; the level is the average day-ahead price on the same days")
        rows, shares = [], []
        for s in ids:
            both = sub[["da_pt", s]].dropna()
            if both.empty:
                continue
            d_ = both[s] - both["da_pt"]
            mean_sp, abs_sp, lvl = round(float(d_.mean()), 2), round(float(d_.abs().mean()), 2), round(float(both["da_pt"].mean()), 2)
            shares.append(100 * abs_sp / lvl)
            rows.append([f"<span style='color:{COLOR[s]}'>■</span>&nbsp;{SERIES[s]}", f"{len(both):,}", f"{mean_sp:+.2f}", f"{abs_sp:.2f}", f"{lvl:.2f}", f"{100 * abs_sp / lvl:.1f}%"])
        _table(["Market", "Days", "Mean spread (EUR/MWh)", "Mean absolute spread (EUR/MWh)", "Average day-ahead price (EUR/MWh)", "Absolute spread ÷ day-ahead price"], rows,
               widths="minmax(9rem, 1.4fr) minmax(4rem, 0.6fr) repeat(4, minmax(6rem, 1fr))")
        _line("Absolute spread ÷ day-ahead price = the mean absolute spread divided by the average day-ahead price, from the numbers shown. "
              "The intraday markets start on 13-06-2024 in their current form. "
              + (f"The mean absolute spread is <b>{min(shares):.0f}% to {max(shares):.0f}%</b> of the day-ahead price, small compared with its level, " if shares else "")
              + "which is why the intraday targets are modelled as a spread to the day-ahead price.")

    # ======================================================================================== 3. how the day-ahead price level changed
    _section("How the day-ahead price level changed", "Periods found from the data, with the known market events marked", num="③")
    da = prices.set_index("date")["da_pt"].dropna() if "da_pt" in prices.columns else pd.Series(dtype=float)
    if len(da) > 200:
        from common_layer.market_analysis import breaks as _breaks
        cfg_ma = ml_ui.cached(lambda: ml_data.config("market_analysis"))()
        reg = ml_ui.cached(lambda: ml_data.config("market_regimes"))()
        bt = cfg_ma.get("break_tests", {})
        cuts = sorted(da.index[i] for i, _ in _breaks.binary_segmentation(da.to_numpy(), int(bt.get("max_data_breaks", 6)), int(bt.get("min_segment_days", 45))))
        edges = [da.index.min()] + cuts + [da.index.max()]
        label_of = {"gas_crisis_start": "Gas crisis begins", "iberian_exception": "Iberian gas-price cap starts", "ida_three_sessions": "Three intraday auctions replace six",
                    "mari_live": "mFRR platform MARI goes live", "pt_isp_15min": "15-minute imbalance settlement", "iberian_blackout": "Iberian blackout",
                    "sdac_15min": "15-minute day-ahead products", "afrr_national_rules": "New national aFRR rules", "price_floor_600": "Price floor lowered to -600 EUR/MWh"}
        events = [{"id": e["id"], "date": pd.Timestamp(e["date"]), "text": label_of.get(e["id"], str(e.get("description", e["id"]))[:50])}
                  for e in list(reg.get("events", [])) + list(cfg_ma.get("extra_events", [])) if e.get("kind") != "deadline"]
        events = sorted([e for e in events if da.index.min() <= e["date"] <= da.index.max()], key=lambda e: e["date"])
        monthly = da.resample("MS").mean()
        top = float(max(monthly.max(), 100)) * 1.18
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly.index + pd.Timedelta(days=14), y=monthly.values, mode="lines", name="Monthly average day-ahead price",
                                 line=dict(color=COLOR["da_pt"], width=1.6), hovertemplate="%{y:,.0f} EUR/MWh"))
        levels = []
        for k, (a, b) in enumerate(zip(edges, edges[1:])):
            seg = da[(da.index >= a) & ((da.index <= b) if k == len(edges) - 2 else (da.index < b))]                          # a cut day belongs to the period that starts on it
            level = float(seg.mean())
            levels.append((a, b, len(seg), level))
            fig.add_trace(go.Scatter(x=[a, b], y=[level, level], mode="lines", line=dict(color=CHAMP, width=3), name="Average price level of a period (found from the data)",
                                     legendgroup="level", showlegend=(k == 0), hovertemplate=f"Average {level:,.0f} EUR/MWh<extra></extra>"))
            fig.add_annotation(x=a + (b - a) / 2, y=level, text=f"<b>{level:,.0f}</b>", showarrow=False, yshift=12, font=dict(color=CHAMP, size=12))
        for n, e in enumerate(events, 1):
            fig.add_vline(x=e["date"], line=dict(color=theme.ORANGE, width=1, dash="dot"))
        fig.add_trace(go.Scatter(x=[e["date"] for e in events], y=[top * (0.93 if n % 2 else 0.84) for n in range(1, len(events) + 1)], mode="markers+text",
                                 text=[str(n) for n in range(1, len(events) + 1)], textfont=dict(color="white", size=11),
                                 marker=dict(color=theme.ORANGE, size=20), name="Known market event (see the key below)",
                                 customdata=[[e["text"], f"{e['date']:%d-%m-%Y}"] for e in events], hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
        theme.style_fig(fig, height=380, yaxis_title="EUR/MWh")
        _legend_top(fig)
        fig.update_yaxes(range=[0, top])
        fig.update_layout(margin=dict(t=44))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        _mini("The periods found from the data", "Break points are found by binary segmentation of the daily day-ahead price; each period has its own average price")
        prow, prev = [], None
        for i, (a, b, n, level) in enumerate(levels, 1):
            lv = round(level, 1)
            change = "–" if prev is None else f"{100 * (lv - prev) / prev:+.0f}%"
            prow.append([f"Period {i}", f"{a:%d-%m-%Y}", f"{b:%d-%m-%Y}", f"{n:,}", f"{lv:,.1f}", change])
            prev = lv
        _table(["Period", "From", "To", "Days", "Average price (EUR/MWh)", "Change from the period before"], prow, widths="minmax(7rem, 1fr) repeat(2, minmax(6rem, 1fr)) minmax(4rem, 0.7fr) minmax(7rem, 1.2fr) minmax(7rem, 1.2fr)")
        _line("The change is worked out from the averages as shown: (this period − the period before) ÷ the period before.")
        _mini("The known market events", "The numbers on the chart")
        _table(["No.", "Event", "Date"], [[f"<b>{i}</b>", e["text"], f"{e['date']:%d-%m-%Y}"] for i, e in enumerate(events, 1)], widths="minmax(3.5rem, 0.4fr) minmax(16rem, 3fr) minmax(6rem, 1fr)", left=(0, 1))
