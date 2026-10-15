"""ML Lifecycle 3 - Problem formulation: what is forecast and what is known when."""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ml_data
import ml_ui
import theme

ml_ui.header("Problem formulation", "", step=2)
ml_ui.tour_bar(3)

form = ml_ui.cached(lambda: ml_data.config("forecast_formulation"))()
catalogue = ml_ui.cached(lambda: ml_data.config("forecast_catalogue"))()
if not form:
    ml_ui.not_yet_run("the formulation file")
    st.stop()

target = ml_ui.target_picker(targets=list(ml_data.TARGETS) + ["pv_weather", "reservoir_inflow"])
tdef = next((t for t in form["targets"] if t["id"] == target), None)
cdef = next((t for t in catalogue.get("targets", []) if t["id"] == target), {})
if tdef is None:
    st.warning("This target is not in the formulation file.")
    st.stop()
q = tdef["modelled_quantity"]


def hours_after_midnight(rule: str, data_day_offset: int = 0) -> float | None:
    """'d-1 13:30' / 'D 09:10' (day offset and clock time) -> hours after the start of delivery day D, for data of day D+data_day_offset."""
    m = re.match(r"\s*[dD]\s*([+-]\d+)?\s+(\d{1,2}):(\d{2})", str(rule))
    if not m:
        return None
    return (int(m.group(1) or 0) + data_day_offset) * 24 + int(m.group(2)) + int(m.group(3)) / 60.0


_res = tdef.get("resolution", "-")
_short = lambda t: str(t).replace("15-minute", "15-min").replace("reconciled so the four quarters average to the hourly forecast", "the four quarters average back to the hourly value")
if isinstance(_res, dict):
    _res_pairs = [(str(k).capitalize(), _short(v)) for k, v in _res.items()]
else:
    _res_pairs = [("Resolution", _short(", ".join(map(str, _res)) if isinstance(_res, list) else _res))]


def _esc(v) -> str:
    return html.escape(str(v))


_plant_input = q.get("series") is None                                    # the plant inputs (PV, inflow) are not prices: no series name
_forecast = ({"pv_weather": "irradiance and temperature, hourly", "reservoir_inflow": "inflow, one value per day"}.get(target, "level") if _plant_input
             else f"{'spread to ' + ml_data.series_label(q['base']) if q['kind'] == 'spread' else 'price level'}")
_top = [("Forecast", _forecast, theme.BLUE), ("Series", ml_data.label(target) if _plant_input else ml_data.series_label(q["series"]), theme.AQUA)]
_rows = [("Used for", cdef.get("decision", "-")), ("Needed by", cdef.get("as_of", {}).get("rule", "-"))] + _res_pairs
_tiles = "".join(
    f'<div style="flex:1 1 0; min-width:0; background:{c}12; border-left:4px solid {c}; border-radius:10px; padding:0.55rem 0.9rem;">'
    f'<div style="font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase; color:{theme.INK_MUTED}; font-weight:600;">{_esc(k)}</div>'
    f'<div style="font-size:1.02rem; font-weight:700; color:{theme.INK_PRIMARY}; margin-top:0.1rem; overflow-wrap:anywhere;">{_esc(v)}</div></div>'
    for k, v, c in _top)
_lines = "".join(
    f'<div style="display:grid; grid-template-columns:7.2rem 1fr; gap:0.8rem; padding:0.42rem 0; border-top:1px solid {theme.GRIDLINE};">'
    f'<div style="font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase; color:{theme.INK_MUTED}; font-weight:600; padding-top:0.15rem;">{_esc(k)}</div>'
    f'<div style="font-size:0.93rem; color:{theme.INK_PRIMARY}; line-height:1.45; overflow-wrap:anywhere;">{_esc(v)}</div></div>'
    for k, v in _rows)
st.markdown(
    f'<div style="border:1px solid {theme.GRIDLINE}; border-radius:14px; background:{theme.SURFACE}; padding:0.85rem 1rem 0.5rem; box-shadow:0 1px 3px rgba(0,0,0,0.05);">'
    f'<div style="display:flex; gap:0.7rem; margin-bottom:0.7rem;">{_tiles}</div>{_lines}</div>', unsafe_allow_html=True)

st.subheader("What is known when")
gate = cdef.get("as_of", {}).get("gate", "")
gate_rule = form["as_of_times"].get(gate)
window = None
if isinstance(gate_rule, dict):                                      # XBID has six forecasting windows; the framework uses the earliest
    window = st.selectbox("Forecasting window", list(gate_rule), index=0, key="ml3_window",
                          help="XBID is forecast at six windows. The evaluation framework uses W1, the earliest, so that its view is valid at every window.")
    gate_rule = gate_rule[window]
order = {w: i for i, w in enumerate(form["as_of_times"].get("XBID", {}))}
gate_h = hours_after_midnight(gate_rule) if gate_rule else None
rows = []
for inp in tdef["inputs"]:
    avail = form["series_availability"].get(inp["series"], {})
    pub = avail.get("published")
    later = window is not None and order.get(inp.get("from_window", "W1"), 0) > order.get(window, 0)
    ok = inp["status"] == "available" and not later
    h = hours_after_midnight(pub, inp["latest_day_offset"]) if (pub and ok) else None
    rows.append({"series": ml_data.series_label(inp["series"]), "latest_day_offset": inp["latest_day_offset"],
                 "status": "from window " + inp["from_window"] if later else ml_data.pretty(inp["status"]).lower(),
                 "published": pub or "-", "hours": h, "evidence": avail.get("evidence", "-")})
tl = pd.DataFrame(rows)
plot = tl.dropna(subset=["hours"])
if len(plot) and gate_h is not None:
    left_edge = -62.0                                                # anything published earlier than this is drawn at the edge as "long before"

    def clock(h: float) -> str:
        """Hours after the start of day D -> 'D-2 13:30' (day D is the delivery day)."""
        total = int(round(h * 60))
        day, minute = divmod(total, 1440)
        return ("D" if day == 0 else f"D{day:+d}".replace("-", "−")) + f" {minute // 60:02d}:{minute % 60:02d}"

    early = plot["hours"] < left_edge
    ends = [gate_h if e else h for h, e in zip(plot["hours"], early)]                # a green bar ends where the series' latest data became available
    labels = ["known in advance" if e else clock(h) for h, e in zip(plot["hours"], early)]
    series = list(plot["series"])
    fig = go.Figure()
    fig.add_vrect(x0=gate_h, x1=10, fillcolor=theme.STATUS_CRITICAL, opacity=0.07, line_width=0, layer="below")
    fig.add_trace(go.Bar(y=series, base=[left_edge] * len(series), x=[e - left_edge for e in ends], orientation="h", width=0.55, name="Real data, used by the model",
                         marker=dict(color=theme.STATUS_GOOD), text=labels, textposition="inside", insidetextanchor="end", textfont=dict(color="white", size=11),
                         customdata=[[s_, f"D{o:+d}".replace("-", "−").replace("D+0", "D"), lab] for s_, o, lab in zip(series, plot["latest_day_offset"], labels)],
                         hovertemplate="%{customdata[0]}<br>latest data: day %{customdata[1]}, available %{customdata[2]}<extra></extra>"))
    waiting = [max(gate_h - e, 0) for e in ends]
    fig.add_trace(go.Bar(y=series, base=ends, x=waiting, orientation="h", width=0.55, name="Nothing newer yet: not used, the model relies on patterns",
                         marker=dict(color="rgba(0,0,0,0)", pattern=dict(shape="/", fgcolor=theme.INK_MUTED, bgcolor="rgba(0,0,0,0)", size=6), line=dict(color=theme.INK_MUTED, width=1)),
                         hoverinfo="skip"))
    fig.add_vline(x=gate_h, line_dash="dash", line_color=theme.INK_PRIMARY, annotation_text=f"forecast made: {clock(gate_h)}", annotation_position="top")
    fig.add_vline(x=0, line_color=theme.INK_MUTED, annotation_text="delivery day D starts", annotation_position="bottom right")
    fig.add_annotation(x=(gate_h + 8) / 2, y=1.0, yref="paper", yanchor="bottom", text="not yet known", showarrow=False, font=dict(color=theme.STATUS_CRITICAL, size=11))
    theme.style_fig(fig, height=360, xaxis_title="time before the delivery day (D = the day being forecast, D−1 = the day before)")
    fig.update_layout(margin=dict(t=44, b=10, l=10, r=10), barmode="overlay", legend=dict(orientation="h", x=0, y=1.16, xanchor="left"))
    ticks = list(range(-60, 1, 12))
    fig.update_xaxes(range=[left_edge, 8], tickvals=ticks, ticktext=[clock(t) for t in ticks])
    fig.update_yaxes(range=[-0.7, len(plot["series"].unique()) - 0.3])
    st.plotly_chart(fig, width="stretch")

if tdef.get("forbidden"):
    st.markdown("**Forbidden for this target:**\n" + "\n".join(f"- {x[:1].upper() + x[1:]}" if isinstance(x, str) else f"- {ml_data.series_label(x.get('series', x))}" for x in tdef["forbidden"]))
