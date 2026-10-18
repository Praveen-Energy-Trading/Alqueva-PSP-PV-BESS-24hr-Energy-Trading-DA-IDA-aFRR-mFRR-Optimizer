"""ML Lifecycle 3 - Problem formulation: what is forecast and what is known when. Told as a story in numbered steps."""
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
from maths import tex as M

ml_ui.header("Problem formulation", "", step=2)
ml_ui.tour_bar(3)

# The colours mean one thing each: green is real data the model may use, hatched grey is a wait for something newer, red is what is not yet known or never allowed.
MUTED, INK, NEUTRAL = theme.INK_MUTED, theme.INK_PRIMARY, "#6b6a64"
REAL, BAD = theme.STATUS_GOOD, theme.STATUS_CRITICAL
CARD_LINE = "#e3e1d9"
_cap = lambda h_: h_[:1].upper() + h_[1:] if h_[:1].islower() else h_                                                  # noqa: E731
_esc = lambda v: html.escape(str(v))                                                                                   # noqa: E731


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


def _pill(text: str, color: str, bg: str, title: str = "") -> str:
    return f"<span title='{_esc(title)}' style='background:{bg}; color:{color}; border-radius:999px; padding:0.1rem 0.6rem; font-size:0.78rem; font-weight:600; white-space:nowrap;'>{text}</span>"


_ok_pill = lambda t_, title="": _pill(t_, "#006100", "#e3f1e3", title)                                                # noqa: E731
_off_pill = lambda t_, title="": _pill(t_, "#5a5953", "#ecebe6", title)                                                # noqa: E731
_warn_pill = lambda t_, title="": _pill(t_, "#7a5200", "#fdf0d0", title)                                               # noqa: E731
_bad_pill = lambda t_, title="": _pill(t_, "#8a1f1f", "#f7e3e3", title)                                                # noqa: E731

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


def clock(h: float) -> str:
    """Hours after the start of day D -> 'D−2 13:30' (day D is the delivery day)."""
    total = int(round(h * 60))
    day, minute = divmod(total, 1440)
    return ("D" if day == 0 else f"D{day:+d}".replace("-", "−")) + f" {minute // 60:02d}:{minute % 60:02d}"


# ============================================================================================ 1. what is forecast
_res = tdef.get("resolution", "–")
_short = lambda t: str(t).replace("15-minute", "15-min").replace("reconciled so the four quarters average to the hourly forecast", "the four quarters average back to the hourly value")
if isinstance(_res, dict):
    _res_pairs = [(str(k).capitalize(), _cap(_short(v))) for k, v in _res.items()]
else:
    _res_pairs = [("Resolution", _cap(_short(", ".join(map(str, _res)) if isinstance(_res, list) else _res)))]
_plant_input = q.get("series") is None                                    # the plant inputs (PV, inflow) are not prices: no series name
_forecast = ({"pv_weather": "Irradiance and temperature, hourly", "reservoir_inflow": "Inflow, one value per day"}.get(target, "Level") if _plant_input
             else f"{'Spread to ' + ml_data.series_label(q['base']) if q['kind'] == 'spread' else 'Price level'}")
_top = [("Forecast", _forecast, theme.BLUE), ("Series", ml_data.label(target) if _plant_input else ml_data.series_label(q["series"]), theme.AQUA)]
_rows = [("Used for", cdef.get("decision", "–")), ("Needed by", cdef.get("as_of", {}).get("rule", "–"))] + _res_pairs + [("Horizon", _cap(cdef.get("horizon", "–")))]
_section("What is forecast", f"{ml_data.label(target)}: the quantity, and when it is needed", num="①")
_tiles = "".join(
    f'<div style="flex:1 1 0; min-width:0; background:{c}12; border-left:4px solid {c}; border-radius:10px; padding:0.55rem 0.9rem;">'
    f'<div style="font-size:0.74rem; color:{MUTED}; font-weight:600;">{_esc(k)}</div>'
    f'<div style="font-size:1.02rem; font-weight:700; color:{INK}; margin-top:0.1rem; overflow-wrap:anywhere;">{_esc(v)}</div></div>' for k, v, c in _top)
_lines = "".join(
    f'<div style="display:grid; grid-template-columns:9rem 1fr; gap:0.8rem; padding:0.5rem 0; border-top:1px solid {CARD_LINE};">'
    f'<div style="font-size:0.74rem; color:{MUTED}; font-weight:600; padding-top:0.1rem;">{_esc(k)}</div>'
    f'<div style="font-size:0.9rem; color:{INK}; line-height:1.45; overflow-wrap:anywhere;">{_esc(v)}</div></div>' for k, v in _rows)
st.markdown(f'<div style="border:1px solid {CARD_LINE}; border-radius:12px; background:#fff; padding:0.85rem 1rem 0.4rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);">'
            f'<div style="display:flex; gap:0.7rem; margin-bottom:0.7rem;">{_tiles}</div>{_lines}</div>', unsafe_allow_html=True)

# ============================================================================================ 2. what is known when
_section("What is known when", "The forecast is made at one moment; only data published before it may be used", num="②")
gate = cdef.get("as_of", {}).get("gate", "")
gate_rule = form["as_of_times"].get(gate)
window = None
if isinstance(gate_rule, dict):                                      # XBID has six forecasting windows; the framework uses the earliest
    _wc, _ = st.columns([1, 3], gap="medium")
    with _wc:
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
    rows.append({"series": ml_data.series_label(inp["series"]), "latest_day_offset": inp["latest_day_offset"], "status_raw": inp["status"], "later": later, "from_window": inp.get("from_window", ""),
                 "published": pub or "–", "hours": h, "evidence": avail.get("evidence", "–"), "note": avail.get("note", "")})
tl = pd.DataFrame(rows)
plot = tl.dropna(subset=["hours"])
if len(plot) and gate_h is not None:
    left_edge = -62.0                                                # anything published earlier than this is drawn at the edge as "long before"
    early = plot["hours"] < left_edge
    ends = [gate_h if e else h for h, e in zip(plot["hours"], early)]                # a green bar ends where the series' latest data became available
    labels = ["Known in advance" if e else clock(h) for h, e in zip(plot["hours"], early)]
    series = list(plot["series"])
    fig = go.Figure()
    fig.add_vrect(x0=gate_h, x1=10, fillcolor=BAD, opacity=0.07, line_width=0, layer="below")
    fig.add_trace(go.Bar(y=series, base=[left_edge] * len(series), x=[e - left_edge for e in ends], orientation="h", width=0.55, name="Real data, used by the model",
                         marker=dict(color=REAL), text=labels, textposition="inside", insidetextanchor="end", textfont=dict(color="white", size=11),
                         customdata=[[s_, f"D{o:+d}".replace("-", "−").replace("D+0", "D"), lab] for s_, o, lab in zip(series, plot["latest_day_offset"], labels)],
                         hovertemplate="%{customdata[0]}<br>Latest data: day %{customdata[1]}, available %{customdata[2]}<extra></extra>"))
    waiting = [max(gate_h - e, 0) for e in ends]
    fig.add_trace(go.Bar(y=series, base=ends, x=waiting, orientation="h", width=0.55, name="Nothing newer yet: not used, the model relies on patterns",
                         marker=dict(color="rgba(0,0,0,0)", pattern=dict(shape="/", fgcolor=MUTED, bgcolor="rgba(0,0,0,0)", size=6), line=dict(color=MUTED, width=1)),
                         hoverinfo="skip"))
    fig.add_vline(x=gate_h, line_dash="dash", line_color=INK, annotation_text=f"Forecast made: {clock(gate_h)}", annotation_position="top")
    fig.add_vline(x=0, line_color=MUTED, annotation_text="Delivery day D starts", annotation_position="bottom left")
    fig.add_annotation(x=(gate_h + 8) / 2, y=0.55, yref="paper", yanchor="middle", text="Not yet known", showarrow=False, font=dict(color=BAD, size=11))
    theme.style_fig(fig, height=360, xaxis_title="Time before the delivery day (D = the day being forecast, D−1 = the day before)")
    fig.update_layout(margin=dict(t=64, b=10, l=10, r=10), barmode="overlay", legend=dict(orientation="h", x=0, y=1.22, xanchor="left", font=dict(size=12, color=theme.INK_SECONDARY)))
    ticks = list(range(-60, 1, 12))
    fig.update_xaxes(range=[left_edge, 8], tickvals=ticks, ticktext=[clock(t) for t in ticks])
    fig.update_yaxes(range=[-0.7, len(plot["series"].unique()) - 0.3])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
if len(tl):
    _mini("The inputs, one by one", "How long the newest value of each series has been known when the forecast is made" if gate_h is not None else "")
    trows = []
    for _, r_ in tl.iterrows():
        h = None if pd.isna(r_["hours"]) else float(r_["hours"])
        off = f"D{int(r_['latest_day_offset']):+d}".replace("-", "−").replace("D+0", "D")
        if r_["later"]:
            state = _warn_pill(f"From window {r_['from_window']}")
        elif r_["status_raw"] != "available":
            state = _off_pill(_cap(ml_data.pretty(r_["status_raw"]).lower()))
        elif h is None:
            state = _off_pill("No publication time")
        elif gate_h is not None and h > gate_h:
            state = _bad_pill("After the forecast")
        else:
            state = _ok_pill("Used")
        gap_h = None if (h is None or gate_h is None) else gate_h - h
        known = "–" if gap_h is None else (f"{gap_h:,.1f} hours" if abs(gap_h) <= 72 else f"{gap_h / 24:,.1f} days")
        ev = r_["evidence"]
        evid = _ok_pill("Measured from files", r_["note"]) if ev == "empirical" else _off_pill("From the market rules", r_["note"]) if ev == "rulebook" else _off_pill(_cap(str(ev)), r_["note"])
        trows.append([r_["series"], off, "–" if h is None else clock(h), known, state, evid])
    _table(["Series", "Latest day used", "Available at", "Known for", "Status", "Evidence"], trows,
           widths="minmax(9rem, 1.6fr) minmax(4.8rem, 0.8fr) minmax(5.2rem, 0.9fr) minmax(5.2rem, 0.9fr) minmax(7rem, 1.1fr) minmax(10rem, 1.5fr)")
    _line(M(r"\text{Known for}=t_{\text{forecast made}}-t_{\text{newest value available}}") + ", from the clock times shown. "
          "Hover over a pill in the evidence column for the note behind it.")

# ============================================================================================ 3. what may and may not be used
_aso = cdef.get("as_of", {})
if _aso.get("allowed") or _aso.get("forbidden"):
    _section("What may and may not be used", "The rule that keeps tomorrow's answer out of the model", num="③")
    arows = [[_ok_pill("Allowed"), _esc(a_)] for a_ in _aso.get("allowed", [])] + [[_bad_pill("Never used"), _esc(f_)] for f_ in _aso.get("forbidden", [])]
    _table(["Rule", "What"], arows, widths="minmax(7rem, 0.7fr) minmax(20rem, 4fr)", left=(1,))
    _line("Anything of the delivery day itself, or published after the moment the forecast is made, is never used: it would make the evaluation look better than the live run can be.")
