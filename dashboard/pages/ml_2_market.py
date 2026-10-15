"""ML Lifecycle 2 - Market analysis: what the data looks like before any model is fitted."""
from __future__ import annotations

import datetime as dt
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

ml_ui.header("Market analysis", "", step=1)
ml_ui.tour_bar(2)

prices = ml_ui.cached(lambda: ml_data.pack("prices_daily"))()
profile = ml_ui.cached(lambda: ml_data.pack("price_profile"))()
periods = ml_ui.cached(lambda: ml_data.config("market_analysis").get("periods", []))()

if prices.empty:
    ml_ui.not_yet_run("the data pack")
else:
    prices["date"] = pd.to_datetime(prices["date"])
    avail = [s for s in SERIES if s in prices.columns]
    st.subheader("Price history")
    c1, c2, c3 = st.columns([3, 2, 2])
    pick = c1.multiselect("Series", avail, default=["da_pt"], format_func=SERIES.get, key="ml2_series")
    smooth = c2.selectbox("Smoothing", [1, 7, 30, 90], index=2, format_func=lambda d: "daily" if d == 1 else f"{d}-day mean", key="ml2_smooth")
    lo, hi = prices["date"].min().date(), prices["date"].max().date()
    rng = c3.slider("Period", lo, hi, (lo, hi), key="ml2_range")
    sub = prices[(prices["date"].dt.date >= rng[0]) & (prices["date"].dt.date <= rng[1])]
    fig = go.Figure()
    for i, s in enumerate(pick):
        y = sub[s].rolling(smooth, min_periods=1).mean() if smooth > 1 else sub[s]
        fig.add_trace(go.Scatter(x=sub["date"], y=y, mode="lines", name=SERIES[s], line=dict(width=1.5, color=theme.CATEGORICAL[i % 8])))
    for j, p in enumerate(periods):
        a, b = pd.Timestamp(p["start"]), pd.Timestamp(p["end"])
        if b >= pd.Timestamp(rng[0]) and a <= pd.Timestamp(rng[1]) and a.year < 2099:
            fig.add_vrect(x0=max(a, pd.Timestamp(rng[0])), x1=min(b, pd.Timestamp(rng[1])), fillcolor=theme.CATEGORICAL[j % 8], opacity=0.06,
                          line_width=0, annotation_text=p["id"].replace("_", " "), annotation_position="top left", annotation_font_size=9)
    st.plotly_chart(theme.style_fig(fig, height=380, yaxis_title="EUR/MWh (EUR/MW/h for aFRR)"), width="stretch")
    st.caption("Shaded bands are the market regimes (the gas crisis, the Iberian gas-price cap, the April 2025 "
               "blackout, the move to 15-minute day-ahead products). They are used to test for structural breaks and to describe results by regime.")

    st.subheader("The intraday spread against day-ahead")
    sp = pd.DataFrame({"date": sub["date"]})
    for s in ("ida1_pt", "ida2_pt", "ida3_pt", "xbid_pt"):
        if s in sub and "da_pt" in sub:
            sp[SERIES[s]] = (sub[s] - sub["da_pt"]).rolling(30, min_periods=10).mean()
    fig = go.Figure()
    for i, c in enumerate([c for c in sp.columns if c != "date"]):
        fig.add_trace(go.Scatter(x=sp["date"], y=sp[c], mode="lines", name=c, line=dict(width=1.5, color=theme.CATEGORICAL[i % 8])))
    fig.add_hline(y=0, line_color=theme.INK_MUTED, line_dash="dot")
    st.plotly_chart(theme.style_fig(fig, height=300, yaxis_title="30-day mean of (price - day-ahead), EUR/MWh"), width="stretch")
    st.caption("The intraday markets start on 2024-06-13 in their current form. The spread is small and mean-reverting compared with the "
               "level of the price, which is why the intraday targets are modelled as a spread to the day-ahead price.")

if profile.empty:
    ml_ui.not_yet_run("the price profile")
else:
    st.subheader("The shape of the day, by month")
    s = st.selectbox("Series", [x for x in SERIES if x in set(profile["series"])], format_func=SERIES.get, key="ml2_prof")
    pv = profile[profile["series"] == s].pivot_table(index="month", columns="hour", values="mean")
    fig = go.Figure(go.Heatmap(z=pv.values, x=[f"{h:02d}" for h in pv.columns], y=[dt.date(2000, int(m), 1).strftime("%b") for m in pv.index],
                               colorscale="Viridis", colorbar=dict(title="mean")))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(theme.style_fig(fig, height=360, xaxis_title="hour of the local day", legend=False), width="stretch")
    st.caption("Average over all years. The midday trough of the day-ahead price in spring and summer is the solar effect; it is why hour and "
               "month are features and why a model must see the recent level.")

st.subheader("From the full market analysis report")
for name, cap in [("phase4_structural_breaks.png", "Structural breaks: the documented rule changes moved the level less than fuel and hydrology."),
                  ("phase4_negative_prices.png", "Negative and non-positive hours by year (solar cannibalisation)."),
                  ("phase4_autocorrelation.png", "Autocorrelation of the price and of its daily change."),
                  ("phase4_market_correlations.png", "How the markets relate (hours where both exist)."),
                  ("phase4_sample_size.png", "Effective sample size per target: how much history is still representative.")]:
    p = ml_data.REPO_ROOT / "docs" / "forecasting" / "figures" / name
    if p.exists():
        with st.expander(cap):
            ml_ui.figure(p)

