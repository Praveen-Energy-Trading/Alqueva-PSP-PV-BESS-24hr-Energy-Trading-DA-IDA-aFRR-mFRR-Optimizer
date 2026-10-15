"""ML Lifecycle 3 - Problem formulation: what is forecast, what is known when, and how the target is transformed."""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
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


_plant_input = q.get("series") is None                                    # the plant inputs (PV, inflow) are not prices: no series name, no transform
_forecast = ({"pv_weather": "irradiance and temperature, hourly", "reservoir_inflow": "inflow, one value per day"}.get(target, "level") if _plant_input
             else f"{'spread to ' + ml_data.series_label(q['base']) if q['kind'] == 'spread' else 'price level'}")
_top = [("Forecast", _forecast, theme.BLUE), ("Series", ml_data.label(target) if _plant_input else ml_data.series_label(q["series"]), theme.AQUA),
        ("Transform", "none" if q["transform"]["name"] == "none" else f"{q['transform']['name']}, scale {q['transform'].get('scale', '-')}", theme.ORANGE)]
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot["hours"], y=plot["series"], mode="markers", marker=dict(size=13, color=[theme.STATUS_GOOD if h <= gate_h else theme.STATUS_CRITICAL
                                                                                                        for h in plot["hours"]]),
                             text=[f"{s}: data of day D{o:+d}, published {p}" for s, o, p in zip(plot["series"], plot["latest_day_offset"], plot["published"])],
                             hoverinfo="text", showlegend=False))
    fig.add_vline(x=gate_h, line_dash="dash", line_color=theme.INK_PRIMARY, annotation_text=f"forecast made: {gate_rule}", annotation_position="top")
    fig.add_vline(x=0, line_color=theme.INK_MUTED, annotation_text="delivery day D starts", annotation_position="bottom right")
    theme.style_fig(fig, height=330, xaxis_title="hours relative to the start of delivery day D (negative = before)", legend=False)
    fig.update_layout(margin=dict(t=44, b=10, l=10, r=10))      # room above the plot for the "forecast made" label
    fig.update_yaxes(range=[-0.7, len(plot["series"].unique()) - 0.3])
    st.plotly_chart(fig, width="stretch")
    st.caption("Each dot is the latest data of an input series that exists when the forecast is made. A dot after the dashed line would be a "
               "leak, and the framework refuses it: the as-of view gives a model only what the information set allows.")
def _input_rows(frame) -> str:
    """One card per input series: name and when it is published on the left, the day of its latest data and its status on the right."""
    out = ""
    for _, r in frame.iterrows():
        status = str(r["status"])
        ok = status == "available"
        bg, fg = ("#e6f4ea", "#1e6b33") if ok else ("#fdf1e3", "#9a5a00")
        day = f"D{r['latest_day_offset']:+d}" if r["latest_day_offset"] else "D"
        out += (f'<div style="display:flex; align-items:center; gap:0.7rem; padding:0.5rem 0.2rem; border-top:1px solid {theme.GRIDLINE};">'
                f'<div style="width:0.55rem; height:0.55rem; border-radius:50%; background:{fg}; flex:none;"></div>'
                f'<div style="flex:1 1 auto; min-width:0;"><div style="font-weight:600; color:{theme.INK_PRIMARY}; font-size:0.93rem;">{_esc(r["series"])}</div>'
                f'<div style="color:{theme.INK_MUTED}; font-size:0.78rem;">published {_esc(r["published"])} &middot; {_esc(r["evidence"])}</div></div>'
                f'<span style="flex:none; background:{theme.GRIDLINE}; color:{theme.INK_SECONDARY}; border-radius:999px; padding:0.1rem 0.55rem; font-size:0.76rem; font-weight:600;" '
                f'title="day of the latest data">{day}</span>'
                f'<span style="flex:none; background:{bg}; color:{fg}; border-radius:999px; padding:0.1rem 0.6rem; font-size:0.76rem; font-weight:600; white-space:nowrap;">{_esc(status)}</span></div>')
    return out


_half = (len(tl) + 1) // 2
_l, _r = st.columns(2)
for _col, _part in ((_l, tl.iloc[:_half]), (_r, tl.iloc[_half:])):
    with _col:
        st.markdown(f'<div style="border:1px solid {theme.GRIDLINE}; border-radius:14px; background:{theme.SURFACE}; padding:0.2rem 0.9rem 0.3rem; '
                    f'box-shadow:0 1px 3px rgba(0,0,0,0.05); margin-bottom:0.9rem;">{_input_rows(_part)}</div>', unsafe_allow_html=True)
if tdef.get("forbidden"):
    st.markdown("**Forbidden for this target:**\n" + "\n".join(f"- {x[:1].upper() + x[1:]}" if isinstance(x, str) else f"- {ml_data.series_label(x.get('series', x))}" for x in tdef["forbidden"]))

if q["transform"]["name"] == "asinh":                                       # only the price targets use the transform
    st.subheader("Why a spread, and why the asinh transform")
    l, r = st.columns(2)
    with l:
        st.markdown("""
    - **Level targets** (day-ahead, aFRR, mFRR): the price itself is forecast.
    - **Spread targets** (IDA1-3, XBID): the forecast is *price minus the day-ahead price*, which is known at the gate. The day-ahead price explains 90-98% of the intraday variance, so the spread is the part that is left to forecast; the forecast price is the day-ahead price plus the forecast spread.
    - **The transform** compresses spikes and keeps negative prices (a logarithm cannot): `z = asinh(price / scale)`. The scale was chosen from the data so that the day-over-day change no longer depends on the price level (the elasticity of the change falls from 0.58 to 0.00 at scale 20 for the day-ahead price).
    """)
    with r:
        scale = st.slider("Scale", 1.0, 50.0, float(q["transform"].get("scale", 20.0)), 1.0, key="ml3_scale")
        x = np.linspace(-200, 400, 400)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=np.arcsinh(x / scale), mode="lines", name=f"asinh(x / {scale:g})", line=dict(color=theme.BLUE, width=2)))
        fig.add_trace(go.Scatter(x=x, y=x / max(abs(x).max(), 1), mode="lines", name="linear (rescaled)", line=dict(color=theme.INK_MUTED, dash="dot")))
        st.plotly_chart(theme.style_fig(fig, height=300, yaxis_title="transformed value", xaxis_title="price"), width="stretch")

