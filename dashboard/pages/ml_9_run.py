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



def _head(title, hint=""):
    """The title of one part of the page: bold, with a thin line above and one muted line under it."""
    st.markdown(f"<div style='margin:1.3rem 0 0.35rem; padding-top:0.7rem; border-top:1px solid {CARD_LINE};'><div style='font-size:1.15rem; font-weight:700; color:{INK};'>{title}</div>"
                + (f"<div style='font-size:0.82rem; color:{MUTED}; margin-top:0.1rem;'>{hint}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _box(num, title, lines, col, tag=None):
    """A box of the flowchart: a numbered badge and the title (which may wrap), the cadence as a small chip under it, then the lines; nothing can leave the box."""
    badge = f"<span style='display:inline-flex; align-items:center; justify-content:center; width:1.15rem; height:1.15rem; border-radius:50%; background:{col}; color:#fff; font-size:0.66rem; font-weight:700; margin-right:0.4rem; flex:none;'>{num}</span>"
    chip = f"<div style='margin:0.15rem 0 0.2rem 0; text-align:left; font-size:0.66rem; font-weight:700; color:{col};'>{tag}</div>" if tag else ""
    return (f"<div style='height:100%; box-sizing:border-box; overflow:hidden; overflow-wrap:anywhere; background:{col}14; border:1px solid {col}40; border-left:3px solid {col}; border-radius:8px; padding:0.45rem 0.6rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>"
            f"<div style='font-weight:700; color:{INK}; font-size:0.84rem; line-height:1.25; display:flex; align-items:flex-start;'>{badge}<span style='min-width:0;'>{title}</span></div>{chip}"
            + "".join(f"<div style='font-size:0.74rem; color:{theme.INK_SECONDARY}; line-height:1.3; margin-top:0.1rem;'>{ln}</div>" for ln in lines) + "</div>")


def _row(items, caption=None, back=False):
    """One row of the flowchart on a grid of five boxes: items are (column 0-4, box, linked to the previous box); a linked neighbour is joined by an arrow."""
    cells = ""
    for col_, box_, link_ in items:
        if link_:
            cells += f"<div style='grid-column:{2 * col_}; grid-row:1; align-self:center; color:{MUTED}; font-size:1.05rem; text-align:center;'>→</div>"
        cells += f"<div style='grid-column:{2 * col_ + 1}; grid-row:1;'>{box_}</div>"
    if back:
        cells += f"<div style='grid-column:10; grid-row:1; align-self:center; color:{MUTED}; font-size:0.72rem; white-space:nowrap; padding-left:0.4rem;'>↺ next day</div>"
    if caption:
        st.markdown(f"<div style='font-size:0.72rem; font-weight:700; color:{MUTED}; text-transform:uppercase; letter-spacing:0.04em; margin:0.5rem 0 0.15rem;'>{caption}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='display:grid; grid-template-columns:repeat(4, minmax(0, 1fr) 0.9rem) minmax(0, 1fr) auto; margin:0.15rem 0 0.35rem; align-items:stretch;'>{cells}</div>", unsafe_allow_html=True)


_head("The routine", "Nine steps: one every day, the others now and then")
_row([(0, _box("1", "New day", ["Today's prices and forecasts are added"], theme.BLUE), False),
      (1, _box("2", "Data complete?", ["Yes: carry on", "No: the original forecaster runs"], theme.AQUA), True),
      (2, _box("3", "Champion forecasts", ["Tomorrow's prices, saved so they cannot be edited"], CHAMP), True),
      (3, _box("4", "Score", ["Real prices arrive: how wrong was it?"], theme.VIOLET), True),
      (4, _box("5", "Still good?", [f"Last {window} days: error, bias", "<b style='color:#006100;'>OK</b>: keep &nbsp; <b style='color:#b00020;'>Flag</b>: a person reviews"], RED), True)], caption="Every day", back=True)
_row([(0, _box("6", "Refit", ["Retrain the champion on the newest 730 days", "Same algorithm, same settings"], theme.BLUE, f"every {refit_d} days"), False),
      (1, _box("7", "Re-tune", ["Search every algorithm again on the newest data"], theme.ORANGE, f"every {tune_d} days"), False),
      (2, _box("8", "Challenger", [f"Runs beside the champion for {shadow_d} days"], theme.VIOLET, f"every {tune_d} days"), True),
      (3, _box("9", "Replace?", [f"Yes only if: {pol.get('min_improvement', 0.02):.0%} lower loss, p below {pol.get('alpha', 0.05):.0%}, {pol.get('min_days_between_promotions', 60)} days since the last change"], CHAMP, f"every {tune_d} days"), True)],
     caption="Now and then")
_head("When it happens", "From 12-10-2026, the first live day")

SH, AN = [], []
PXU, PXY = 1.08, 62.0                                                                                                   # pixels per unit of the figure, across and down


def _path(x0, x1, y0, y1, r_px):
    rx, ry = min(r_px / PXU, (x1 - x0) / 2), min(r_px / PXY, (y1 - y0) / 2)
    return (f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
            f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z")


def bar(x0, x1, y0, y1, color, opacity=1.0, r_px=7.0, line=None, layer="above"):
    SH.append(dict(type="path", path=_path(x0, x1, y0, y1, r_px), fillcolor=color, opacity=opacity, layer=layer, line=line or dict(width=0)))


def note(x, y, text, size=12, color=INK, anchor="center", **kw):
    AN.append(dict(x=x, y=y, xanchor=anchor, yanchor="middle", showarrow=False, text=text, font=dict(size=size, color=color), **kw))


def arrow(x0, y0, x1, y1, color=MUTED):
    AN.append(dict(x=x1, y=y1, ax=x0, ay=y0, axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.6, arrowcolor=color, text=""))


fig = go.Figure()


def lane_label(y, title, steps, caption):
    """The name of a lane, and under it the numbers of the flowchart boxes it comes from, as circles in the colour of the box."""
    note(0, y - 0.28, f"<b style='font-size:15px'>{title}</b>", 13, INK, "left")
    fig.add_trace(go.Scatter(x=[10 + 26 * k_ for k_ in range(len(steps))], y=[y + 0.28] * len(steps), mode="markers+text", showlegend=False, hoverinfo="skip",
                             marker=dict(size=19, color=[c_ for _, c_ in steps]), text=[f"<b>{n_}</b>" for n_, _ in steps], textfont=dict(size=11, color="white")))
    note(10 + 26 * len(steps) - 6, y + 0.28, caption, 11, theme.INK_SECONDARY, "left")

fig.add_trace(go.Scatter(x=[0, 1000], y=[0, 0], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))

for k_, (ya_, yb_) in enumerate(((-0.2, 2.6), (2.6, 4.55), (4.55, 7.15), (7.15, 8.9))):
    bar(-8, 1010, ya_, yb_, "#f8f7f3" if k_ % 2 == 0 else "#ffffff", 1.0, r_px=10, line=dict(color=CARD_LINE, width=1), layer="below")

# ---- lane 1: the data
Y1 = 1.0
note(0, Y1 - 0.75, "<b>Data</b>", 13, INK, "left")
bar(0, X(d_dev) - 2, Y1 - 0.4, Y1 + 0.4, GREY, 0.45)
note(X(d_dev) / 2, Y1, "Backtest: champion chosen", 12)
bar(X(d_dev), X(d_live) - 2, Y1 - 0.4, Y1 + 0.4, RED, 0.22)
note((X(d_dev) + X(d_live)) / 2, Y1, "Locked<br>year", 10)
bar(X(d_live), 1000, Y1 - 0.4, Y1 + 0.4, REAL, 0.22)
note(X(d_live) + 8, Y1, "A new day arrives, every day  →", 12, anchor="left")
for dd_, xx_, an_ in ((d0, 0, "left"), (d_dev, X(d_dev), "right"), (d_live, X(d_live), "left")):                         # the two dates a year apart sit on either side of their lines
    note(xx_, Y1 + 0.68, f"{dd_:%d-%m-%Y}", 11, MUTED, an_)
note(1000, Y1 + 0.68, f"{fut(HOR):%d-%m-%Y}", 11, MUTED, "right")

SH.append(dict(type="line", x0=X(d_live), x1=X(d_live), y0=Y1 + 1.0, y1=8.7, line=dict(color=MUTED, width=1.2, dash="dot")))
note(X(d_live) - 6, Y1 + 1.3, "Data ends 11-10-2026 · live from 12-10-2026", 11, INK, "right")

# ---- lane 2: the champion
Y2 = 3.2
lane_label(Y2, "Champion", [(3, CHAMP), (6, theme.BLUE)], "daily · every 30 days")
bar(X(d_live), 1000, Y2 - 0.38, Y2 + 0.38, CHAMP, 0.92)
note(X(d_live) + 8, Y2, "Same champion, every day", 12, "white", "left")
rf = [fut(refit_d * i) for i in range(1, HOR // refit_d + 1)]
fig.add_trace(go.Scatter(x=[X(d) for d in rf], y=[Y2 + 0.78] * len(rf), mode="markers", showlegend=False, marker=dict(size=10, color=theme.BLUE, line=dict(color="white", width=1)),
                         text=[f"Refit on {d:%d-%m-%Y}" for d in rf], hoverinfo="text"))
note(X(d_live) - 8, Y2 + 0.78, f"<b>Refit</b> every {refit_d} days", 11, theme.BLUE, "right")

# ---- lane 3: the challenger
Y3 = 5.9
lane_label(Y3, "Challenger", [(7, theme.ORANGE), (8, theme.VIOLET), (9, CHAMP)], "every 180 days")
t1, t2 = fut(tune_d), fut(2 * tune_d)
dec = t1 + pd.Timedelta(days=shadow_d)
fig.add_trace(go.Scatter(x=[X(t1), X(t2)], y=[Y3, Y3], mode="markers", showlegend=False, marker=dict(size=18, color=theme.ORANGE, symbol="diamond", line=dict(color="white", width=1.2)),
                         text=[f"Re-tune on {t1:%d-%m-%Y}", f"Re-tune on {t2:%d-%m-%Y}"], hoverinfo="text"))
note(X(t1), Y3 + 0.72, f"<b>Re-tune</b><br>{t1:%d-%m-%Y}", 11, theme.ORANGE)
note(X(t2), Y3 + 0.72, f"<b>Re-tune</b><br>{t2:%d-%m-%Y}", 11, theme.ORANGE)
bar(X(t1), X(dec), Y3 - 0.38, Y3 + 0.38, theme.VIOLET, 0.25)                                                           # exactly the days it runs, from the re-tune to the decision
note((X(t1) + X(dec)) / 2 + 10, Y3, f"Alongside, {shadow_d} days", 11)
SH.append(dict(type="line", x0=X(dec), x1=X(dec), y0=Y3 - 0.55, y1=Y3 + 0.55, line=dict(color=CHAMP, width=2.2)))        # the decision day
note(X(dec), Y3 + 0.72, f"<b>Decision</b><br>{dec:%d-%m-%Y}", 11, INK)
bar(X(dec) + 10, X(dec) + 114, Y3 - 0.4, Y3 + 0.4, "#ffffff", 1.0, r_px=9, line=dict(color=CHAMP, width=1.6))
note(X(dec) + 62, Y3, "<b>Replace?</b>", 12)
note(X(dec) + 62, Y3 - 0.92, "Yes: it becomes champion", 11, REAL)
note(X(dec) + 62, Y3 - 1.25, "No: champion stays", 11, theme.INK_SECONDARY)

# ---- lane 4: the daily watch
Y4 = 8.0
lane_label(Y4, "Watch", [(4, theme.VIOLET), (5, RED)], "every day")
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

# ============================================================================================ where we are today: the only part that moves with the data
snap = ml_ui.cached(lambda: ml_data.pack_json("registry_snapshot"))() or {}
locked = ml_ui.cached(lambda: ml_data.pack("locked_forecasts"))()
_live = [pd.Timestamp(v["promoted_utc"]).tz_localize(None).normalize() for vs in snap.get("champions", {}).values() for v in vs if v.get("status") == "active"]
if _live and not locked.empty:
    since = min(_live)
    last_data = pd.to_datetime(locked["ts_utc"], utc=True).dt.tz_convert("Europe/Madrid").dt.tz_localize(None).dt.normalize().max()
    _cards = [("Last day of data", f"{last_data:%d-%m-%Y}", theme.BLUE), ("Champions live since", f"{since:%d-%m-%Y}", CHAMP),
              ("Next refit", f"{since + pd.Timedelta(days=refit_d):%d-%m-%Y}", theme.BLUE), ("Next re-tune", f"{since + pd.Timedelta(days=tune_d):%d-%m-%Y}", theme.ORANGE)]
    _head("Where we are today", "Read from the data and the registry: this part moves when the data pack is rebuilt")
    st.markdown("<div style='display:flex; gap:0.6rem; margin-bottom:0.7rem;'>" + "".join(
        f"<div style='flex:1 1 0; background:#fff; border:1px solid {CARD_LINE}; border-top:3px solid {c_}; border-radius:10px; padding:0.5rem 0.8rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>"
        f"<div style='font-size:0.75rem; color:{MUTED};'>{n_}</div><div style='font-size:1.25rem; font-weight:700; color:{INK};'>{v_}</div></div>" for n_, v_, c_ in _cards) + "</div>", unsafe_allow_html=True)
