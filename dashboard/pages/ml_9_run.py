"""ML Lifecycle 9 - Run it: after the last day of data, how long the champion is trusted. One picture."""
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

ml_ui.header("Run it: monitoring, live", "After 11-10-2026 the champion is not chosen again every day: it keeps running, is refitted on new data and must keep earning its place.", step=7)
ml_ui.tour_bar(8)

MUTED, INK, CHAMP = theme.INK_MUTED, theme.INK_PRIMARY, "#2b2a28"
GREY, REAL, RED, AMBER = "#b9b7ae", theme.GREEN, theme.STATUS_CRITICAL, "#eda100"
CARD_LINE = "#e3e1d9"

mon = ml_ui.cached(lambda: ml_data.config("monitoring"))()
pol = ml_ui.cached(lambda: ml_data.config("champion_policy").get("promotion", {}))()
sch, trk, rel = mon.get("schedule", {}), mon.get("tracking", {}), mon.get("relative", {})
refit_d, tune_d = int(sch.get("refit_every_days", 30)), int(sch.get("retune_every_days", 180))
window = int(trk.get("window_days", 28))
shadow_d = int(pol.get("window_days", 90))

# ---- the dates: the data starts 31-12-2020, the backtest ends 11-10-2025, the locked year ends 11-10-2026, the champions go live the next day
d0, d_dev, d_live = pd.Timestamp("2020-12-31"), pd.Timestamp("2025-10-12"), pd.Timestamp("2026-10-12")
HOR = 400                                                                                                               # days of the future that are drawn
PAST_W, FUT_W = 300.0, 700.0                                                                                           # the past is drawn compressed, so that the future is readable


def X(day) -> float:
    day = pd.Timestamp(day)
    return PAST_W * (day - d0).days / (d_live - d0).days if day <= d_live else PAST_W + FUT_W * (day - d_live).days / HOR


def fut(n: int) -> pd.Timestamp:
    return d_live + pd.Timedelta(days=n)



def _box(num, title, lines, col):
    return (f"<div style='flex:1 1 0; min-width:0; background:{col}1a; border-left:4px solid {col}; border-radius:8px; padding:0.55rem 0.8rem;'>"
            f"<div style='font-weight:700; color:{INK}; font-size:0.95rem;'><span style='color:{col};'>{num}</span>&nbsp; {title}</div>"
            + "".join(f"<div style='font-size:0.82rem; color:{theme.INK_SECONDARY}; line-height:1.4;'>{ln}</div>" for ln in lines) + "</div>")


def _row(tag, tag_color, boxes, back=False):
    arrow_ = f"<div style='align-self:center; color:{MUTED}; font-size:1.2rem; padding:0 0.05rem;'>→</div>"
    inner = arrow_.join(boxes)
    st.markdown(f"<div style='display:flex; gap:0.7rem; align-items:stretch; margin:0.35rem 0;'>"
                f"<div style='flex:0 0 7.2rem; display:flex; align-items:center;'><span style='background:{tag_color}; color:#fff; border-radius:999px; padding:0.2rem 0.8rem; font-size:0.8rem; font-weight:700;'>{tag}</span></div>"
                f"<div style='flex:1 1 0; display:flex; gap:0.4rem; align-items:stretch; min-width:0;'>{inner}</div>"
                + (f"<div style='align-self:center; color:{MUTED}; font-size:0.9rem; white-space:nowrap;'>↺ next day</div>" if back else "") + "</div>", unsafe_allow_html=True)


_row("Every day", theme.BLUE, [
    _box("①", "New day", ["Today's prices and forecasts are added"], theme.BLUE),
    _box("②", "Data complete?", ["Yes: carry on", "No: the original forecaster runs"], theme.AQUA),
    _box("③", "Champion forecasts", ["Tomorrow's prices, saved so they cannot be edited"], CHAMP),
    _box("④", "Score", ["Real prices arrive: how wrong was it?"], theme.VIOLET),
    _box("⑤", "Still good?", [f"Last {window} days: error, bias", "<b style='color:#006100;'>OK</b>: keep &nbsp; <b style='color:#b00020;'>Flag</b>: a person reviews"], RED)], back=True)
_row(f"Every {refit_d} days", theme.BLUE, [
    _box("⑥", "Refit", [f"Retrain the same champion on the newest {730} days", "Same algorithm, same settings"], theme.BLUE)])
_row(f"Every {tune_d} days", theme.ORANGE, [
    _box("⑦", "Re-tune", ["Search every algorithm again on the newest data"], theme.ORANGE),
    _box("⑧", "Challenger", [f"Runs beside the champion for {shadow_d} days"], theme.VIOLET),
    _box("⑨", "Replace?", [f"Yes only if: {pol.get('min_improvement', 0.02):.0%} lower loss, p below {pol.get('alpha', 0.05):.0%},", f"{pol.get('min_days_between_promotions', 60)} days since the last change"], CHAMP)])
st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

SH, AN = [], []


def bar(x0, x1, y0, y1, color, opacity=1.0):
    SH.append(dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=color, opacity=opacity, line_width=0))


def note(x, y, text, size=12, color=INK, anchor="center", **kw):
    AN.append(dict(x=x, y=y, xanchor=anchor, yanchor="middle", showarrow=False, text=text, font=dict(size=size, color=color), **kw))


def arrow(x0, y0, x1, y1, color=MUTED):
    AN.append(dict(x=x1, y=y1, ax=x0, ay=y0, axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.6, arrowcolor=color, text=""))


fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1000], y=[0, 0], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))

# ---- lane 1: the data
Y1 = 1.0
note(0, Y1 - 0.75, "<b>Data</b>", 13, INK, "left")
bar(0, X(d_dev) - 2, Y1 - 0.4, Y1 + 0.4, GREY, 0.45)
note(X(d_dev) / 2, Y1, "Backtest: champion chosen", 12)
bar(X(d_dev), X(d_live) - 2, Y1 - 0.4, Y1 + 0.4, RED, 0.22)
note((X(d_dev) + X(d_live)) / 2, Y1, "Locked year", 10)
bar(X(d_live), 1000, Y1 - 0.4, Y1 + 0.4, REAL, 0.22)
note(X(d_live) + 8, Y1, "A new day arrives, every day  →", 12, anchor="left")
for dd_, xx_ in ((d0, 0), (d_dev, X(d_dev)), (d_live, X(d_live))):
    note(xx_, Y1 + 0.68, f"{dd_:%d-%m-%Y}", 10, MUTED, "left" if xx_ == 0 else "center")
note(1000, Y1 + 0.68, f"{fut(HOR):%d-%m-%Y}", 10, MUTED, "right")

SH.append(dict(type="line", x0=X(d_live), x1=X(d_live), y0=Y1 + 0.95, y1=8.7, line=dict(color=MUTED, width=1.2, dash="dot")))
note(X(d_live) - 6, Y1 + 1.15, "Data ends 11-10-2026 · live from 12-10-2026", 11, INK, "right")

# ---- lane 2: the champion
Y2 = 3.2
note(0, Y2, "<b style='font-size:15px'>Champion</b><br><span style='font-size:11px;color:#6b6a64'>③ ⑥ every day, every 30 days</span>", 13, INK, "left")
bar(X(d_live), 1000, Y2 - 0.38, Y2 + 0.38, CHAMP, 0.92)
note(X(d_live) + 8, Y2, "Same champion, every day", 12, "white", "left")
rf = [fut(refit_d * i) for i in range(1, HOR // refit_d + 1)]
fig.add_trace(go.Scatter(x=[X(d) for d in rf], y=[Y2 + 0.78] * len(rf), mode="markers", showlegend=False, marker=dict(size=10, color=theme.BLUE, line=dict(color="white", width=1)),
                         text=[f"Refit on {d:%d-%m-%Y}" for d in rf], hoverinfo="text"))
note(X(d_live) - 8, Y2 + 0.78, f"<b>Refit</b> every {refit_d} days", 11, theme.BLUE, "right")

# ---- lane 3: the challenger
Y3 = 5.9
note(0, Y3, "<b style='font-size:15px'>Challenger</b><br><span style='font-size:11px;color:#6b6a64'>⑦ ⑧ ⑨ every 180 days</span>", 13, INK, "left")
t1, t2 = fut(tune_d), fut(2 * tune_d)
dec = t1 + pd.Timedelta(days=shadow_d)
fig.add_trace(go.Scatter(x=[X(t1), X(t2)], y=[Y3, Y3], mode="markers", showlegend=False, marker=dict(size=18, color=theme.ORANGE, symbol="diamond", line=dict(color="white", width=1.2)),
                         text=[f"Re-tune on {t1:%d-%m-%Y}", f"Re-tune on {t2:%d-%m-%Y}"], hoverinfo="text"))
note(X(t1), Y3 + 0.72, f"<b>Re-tune</b><br>{t1:%d-%m-%Y}", 10, theme.ORANGE)
note(X(t2), Y3 + 0.72, f"<b>Re-tune</b><br>{t2:%d-%m-%Y}", 10, theme.ORANGE)
bar(X(t1) + 12, X(dec), Y3 - 0.38, Y3 + 0.38, theme.VIOLET, 0.25)
note((X(t1) + 12 + X(dec)) / 2, Y3, f"Alongside, {shadow_d} days", 11)
bar(X(dec), X(dec) + 104, Y3 - 0.52, Y3 + 0.52, "#ffffff")
SH.append(dict(type="rect", x0=X(dec), x1=X(dec) + 104, y0=Y3 - 0.52, y1=Y3 + 0.52, fillcolor="rgba(0,0,0,0)", line=dict(color=CHAMP, width=1.6)))
note(X(dec) + 52, Y3 - 0.22, "<b>Replace?</b>", 12)
note(X(dec) + 52, Y3 + 0.2, f"{dec:%d-%m-%Y}", 10, MUTED)
arrow(X(dec) - 2, Y3, X(dec) + 2, Y3)
note(X(dec) + 52, Y3 - 0.92, "Yes: it becomes champion", 10, REAL)
note(X(dec) + 52, Y3 - 1.25, "No: champion stays", 10, MUTED)

# ---- lane 4: the daily watch
Y4 = 8.0
note(0, Y4, "<b style='font-size:15px'>Watch</b><br><span style='font-size:11px;color:#6b6a64'>④ ⑤ every day</span>", 13, INK, "left")
bar(X(d_live), 1000, Y4 - 0.22, Y4 + 0.22, AMBER, 0.28)
note(X(d_live) + 8, Y4, f"Every day: error, bias, data ({window} days). A flag asks for a review, it never changes the champion", 11, anchor="left")

end = Y4 + 0.9
theme.style_fig(fig, height=int(end * 62), xaxis_title=None, legend=False)
fig.update_yaxes(autorange=False, range=[end, -0.4], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False)
fig.update_xaxes(range=[-8, 1010], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False)
fig.update_layout(margin=dict(t=4, b=4, l=6, r=6), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
fig.layout.shapes = SH
fig.layout.annotations = AN
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
