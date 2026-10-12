"""ML Lifecycle 3 - Problem formulation: what is forecast, what is known when, and how the target is transformed."""
from __future__ import annotations

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

ml_ui.header("Problem formulation", "A forecast is only useful if it uses what is known at the time the decision is made. Each target is "
             "defined by what it forecasts, when it is needed and which data exists by then.", step=3)
ml_ui.tour_bar(3)

form = ml_ui.cached(lambda: ml_data.config("forecast_formulation"))()
catalogue = ml_ui.cached(lambda: ml_data.config("forecast_catalogue"))()
if not form:
    ml_ui.not_yet_run("the formulation file")
    st.stop()

target = ml_ui.target_picker(targets=[t for t in ml_data.TARGETS])
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


c1, c2, c3, c4 = st.columns(4)
theme.metric_cards([("Forecast", f"{'spread to ' + ml_data.series_label(q['base']) if q['kind'] == 'spread' else 'price level'}"), ("Series", ml_data.series_label(q["series"])),
                    ("Transform", f"{q['transform']['name']}, scale {q['transform'].get('scale', '-')}"),
                    ("Resolution", ", ".join(map(str, tdef.get("resolution", []))) if isinstance(tdef.get("resolution"), list) else str(tdef.get("resolution", "-")))],
                   ncols=4, pre_formatted=True)
st.markdown(f"**Used for:** {cdef.get('decision', '-')}  \n**Needed by:** {cdef.get('as_of', {}).get('rule', '-')}")

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
    st.plotly_chart(theme.style_fig(fig, height=300, xaxis_title="hours relative to the start of delivery day D (negative = before)", legend=False),
                    width="stretch")
    st.caption("Each dot is the latest data of an input series that exists when the forecast is made. A dot after the dashed line would be a "
               "leak, and the framework refuses it: the as-of view gives a model only what the information set allows.")
st.dataframe(tl[["series", "latest_day_offset", "status", "published", "evidence"]].rename(columns={
    "series": "Input series", "latest_day_offset": "Latest day (D+k)", "status": "Status", "published": "Published", "evidence": "How known"}),
    hide_index=True, width="stretch")
if tdef.get("forbidden"):
    st.markdown("**Forbidden for this target:** " + ", ".join(f"`{x}`" if isinstance(x, str) else f"`{x.get('series', x)}`" for x in tdef["forbidden"]))

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

