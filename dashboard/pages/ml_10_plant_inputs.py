"""ML Lifecycle 10 - Plant inputs: the forecasts of solar irradiance, air temperature and natural inflow that feed the optimiser. Told as a story in numbered steps."""
from __future__ import annotations

import html
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

ml_ui.header("Plant inputs: PV and inflow", "", step=9)
ml_ui.tour_bar(10)

# The colours mean one thing each, on every picture of this page (and of the other ML pages):
#   what happened is light grey, the serving forecast is near-black, a simple forecast is dark grey and dotted, an algorithm keeps its own colour.
MUTED, INK, CHAMP = theme.INK_MUTED, theme.INK_PRIMARY, "#2b2a28"
NEUTRAL, GREY, REAL, LUCK = "#6b6a64", "#b9b7ae", theme.GREEN, "#9a9890"
CARD_LINE = "#e3e1d9"
_cap = lambda h_: h_[:1].upper() + h_[1:] if h_[:1].islower() else h_                                                  # noqa: E731
_esc = lambda v: html.escape(str(v))                                                                                   # noqa: E731
_dmy = lambda x: f"{str(x)[8:10]}-{str(x)[5:7]}-{str(x)[:4]}"                                                          # noqa: E731
ALG = {"DecisionTree": "decision_tree", "RandomForest": "random_forest", "GradientBoosting": "gradient_boosting", "XGBoost": "xgboost", "LightGBM": "lightgbm"}
_key = lambda m_: ALG.get(m_, m_)                                                                                      # noqa: E731
_label = lambda m_: ml_data.MODEL_LABELS.get(_key(m_), m_)                                                             # noqa: E731
_col = lambda m_: theme.CATEGORICAL[ml_data.model_index(_key(m_)) % 8] if _key(m_) in ml_data.MODEL_ORDER else LUCK    # noqa: E731


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


def _panel(inner: str) -> str:
    """A white rounded card for a calculation, as on the Tuning lab."""
    return (f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0.55rem 1.1rem 0.8rem; box-shadow:0 1px 2px rgba(0,0,0,0.04); "
            f"font-size:0.9rem; color:{theme.INK_SECONDARY};'>{inner}</div>")


def _table(headers: list, rows: list, highlight: int | None = None, widths: str | None = None, left: tuple = (0,)) -> None:
    """A table as on the Tuning lab: muted header, thin row lines, the first column left, the others centred; one row can be shaded."""
    grid = f"display:grid; grid-template-columns:{widths or f'minmax(9rem, 1.5fr) repeat({len(headers) - 1}, minmax(0, 1fr))'}; align-items:center; column-gap:0.6rem;"
    head = (f"<div style='{grid} padding:0.6rem 0 0.4rem; border-bottom:1px solid {CARD_LINE}; font-size:0.72rem; color:{MUTED}; font-weight:600; line-height:1.2;'>"
            + "".join(f"<span style='{'' if j in left else 'text-align:center;'}'>{h}</span>" for j, h in enumerate(headers)) + "</div>")
    body = ""
    for i, row in enumerate(rows):
        hl = i == highlight
        body += (f"<div style='{grid} min-height:2.5rem; font-size:0.86rem; "
                 + ("border-top:1px solid #e3e1d9; background:#f8f7f3; font-weight:700;" if hl else "border-bottom:1px solid #eeede8;") + "'>"
                 + "".join(f"<span style='{'' if j in left else 'text-align:center;'} color:{INK if str(c) != '–' else MUTED};'>{c}</span>" for j, c in enumerate(row)) + "</div>")
    st.markdown(f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0 1.1rem 0.1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>{head}{body}</div>", unsafe_allow_html=True)


def _pill(text: str, color: str, bg: str) -> str:
    return f"<span style='background:{bg}; color:{color}; border-radius:999px; padding:0.1rem 0.6rem; font-size:0.78rem; font-weight:600; white-space:nowrap;'>{text}</span>"


def _legend_top(fig, y=1.08) -> None:
    fig.update_layout(legend=dict(orientation="h", x=0, y=y, xanchor="left", yanchor="bottom", font=dict(size=12, color=theme.INK_SECONDARY)))


labels = {key: name for key, name, _ in ml_data.PLANT_INPUTS}
choice = st.selectbox("Forecast", [key for key, _, _ in ml_data.PLANT_INPUTS], format_func=lambda k: labels[k], key="ml10_choice")
catalogue = ml_ui.cached(lambda: ml_data.config("forecast_catalogue"))()
entry_id = next(cid for key, _, cid in ml_data.PLANT_INPUTS if key == choice)
entry = next((t for t in catalogue.get("targets", []) if t.get("id") == entry_id), {})
w = ml_ui.cached(ml_data.weather_eval)(choice) if choice in ("ghi", "temperature") else None
weather = bool(w and "daily" in w)

# ============================================================================================ what the page needs, from the two kinds of forecast
if weather:                                                    # irradiance and temperature: the weather forecast against the algorithms, as of the decision time
    unit, nd = w["unit"], (1 if w["unit"] == "W/m²" else 2)
    d_ = w["data"]
    tiles = [("Feeds", entry.get("decision", "–"), theme.BLUE), ("Needed by", entry.get("as_of", {}).get("rule", "–"), theme.AQUA), ("Serving forecast", "Weather forecast (Open-Meteo), two days old", CHAMP)]
    p_ = ml_data.plant_input(choice)
    rows = [("Series", f"{w['target']} ({unit}), hourly"),
            ("Inputs", "The numerical weather forecast of the site; without it, a model on past observations and clear-sky physics"),
            ("Without the forecast", (f"The model uses {p_['features'].replace('_', ' ')}. Its lags and rolling means end at the same hour yesterday, which is later than what is observed at 11:00 on the day before." if p_ else "–")),
            ("Truth", f"{d_['source']}, {_dmy(d_['first'])} to {_dmy(d_['last'])}. A reanalysis, not an on-site measurement; the weather forecasts are archived from {_dmy(d_['forecast_first'])}."),
            ("Test", f"The last twelve months, {_dmy(w['test_range'][0])} to {_dmy(w['test_range'][1])} ({w['n_test']:,} hours), judged as of 11:00 on the day before: "
                     "no observation of the last hours before the delivery day is used by anything below")]
    serve_name = "Weather forecast (serving)"
    serve = w["baselines"]["Weather forecast alone"]
    perf = [(serve_name, "Weather forecast", serve, True, None)]
    perf += [(k, "Simple forecast", v, False, None) for k, v in w["baselines"].items() if k != "Weather forecast alone"]
    perf += [(_label(m), "Algorithm, past observations only", v, False, m) for m, v in w["without_forecast_full_history"].items()]
    perf += [(_label(m) + " + weather forecast", "Algorithm with the forecast as an input", v, False, m) for m, v in w["with_forecast"].items()]
    plain_m, plain = min(w["without_forecast_full_history"].items(), key=lambda kv: kv[1]["mae"])
    simple_n, simple = min((kv for kv in w["baselines"].items() if kv[0] != "Weather forecast alone"), key=lambda kv: kv[1]["mae"])
    with_m, with_v = min(w["with_forecast"].items(), key=lambda kv: kv[1]["mae"])
    daily, week = w["daily"], w["week"]
    names = ("Reanalysis (what happened)", "Weather forecast", "Newest same-hour observation")
    week_x = "The last seven days of the test year (hourly, UTC)"
else:
    r = ml_ui.cached(ml_data.plant_input)(choice)              # the choice is an argument, so each forecast has its own cache entry
    if not r:
        ml_ui.not_yet_run(f"the {labels[choice].lower()} evaluation (tools/evaluate_plant_inputs.py)")
        st.stop()
    unit = r["unit"]
    nd = 0 if unit == "m³/h" else 2
    selected = r["selected"]
    chosen = r["models"][selected]
    simple_n, simple = min(r["baselines"].items(), key=lambda kv: kv[1]["mae"])
    d_ = r.get("data") or {}
    days = (lambda n: round(n / 24)) if r["resolution"] == "hourly" else (lambda n: n)
    tiles = [("Feeds", entry.get("decision", "–"), theme.BLUE), ("Needed by", entry.get("as_of", {}).get("rule", "–"), theme.AQUA), ("Serving model", f"{_label(selected)}, retrained daily", CHAMP)]
    rows = [("Series", f"{r['name']} ({unit}), {r['resolution']}"), ("Inputs of the model", _cap(r["features"].replace("_", " "))),
            ("History", f"Development {_dmy(r['dev_range'][0])} to {_dmy(r['dev_range'][1])} ({days(r['n_dev']):,} days); test {_dmy(r['test_range'][0])} to {_dmy(r['test_range'][1])} ({days(r['n_test']):,} days)")]
    if d_:
        src = "Plant sensor" if choice in ("ghi", "temperature") else "Guadiana inflow research dataset"
        kind = "synthetic" if choice in ("ghi", "temperature") else "monthly means (climatology)"
        rows.append(("Data", f"{src}, measured {_dmy(d_['first'])} to {_dmy(d_['last_measured'])} ({d_['measured']:,} values). The {d_['filled']:,} later rows are {kind} and are never scored."))
    serve_name = _label(selected) + " (serving)"
    serve = chosen
    perf = [(_label(m) + (" (serving)" if m == selected else ""), "Algorithm", s_, m == selected, m) for m, s_ in r["models"].items()]
    perf += [(n, "Simple forecast", s_, False, None) for n, s_ in r["baselines"].items()]
    daily, week = r["daily"], r.get("week")
    names = ("Measured", "Serving model", "Same time yesterday")
    week_x = "The last seven days of the test year (hourly)"
_n = lambda v_: f"{(round(v_, nd) or 0.0):,.{nd}f}".replace(",", "{,}")                                              # a number for a formula
fmt = lambda v_: f"{(round(v_, nd) or 0.0):,.{nd}f}"                                                                 # noqa: E731                                       (no "-0.0")

# ============================================================================================ 1. what it is
_section("What it is", f"{labels[choice]}: the forecast that feeds the optimiser", num="①")
tile_html = "".join(
    f'<div style="flex:1 1 0; min-width:0; background:{c}12; border-left:4px solid {c}; border-radius:10px; padding:0.55rem 0.9rem;">'
    f'<div style="font-size:0.74rem; color:{MUTED}; font-weight:600;">{_esc(k)}</div>'
    f'<div style="font-size:0.93rem; font-weight:600; color:{INK}; margin-top:0.1rem; overflow-wrap:anywhere;">{_esc(v)}</div></div>' for k, v, c in tiles)
row_html = "".join(
    f'<div style="display:grid; grid-template-columns:9rem 1fr; gap:0.8rem; padding:0.5rem 0; border-top:1px solid {CARD_LINE};">'
    f'<div style="font-size:0.74rem; color:{MUTED}; font-weight:600; padding-top:0.1rem;">{_esc(k)}</div>'
    f'<div style="font-size:0.9rem; color:{INK}; line-height:1.45; overflow-wrap:anywhere;">{_esc(v)}</div></div>' for k, v in rows)
st.markdown(f'<div style="border:1px solid {CARD_LINE}; border-radius:12px; background:#fff; padding:0.85rem 1rem 0.4rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);">'
            f'<div style="display:flex; gap:0.7rem; margin-bottom:0.7rem;">{tile_html}</div>{row_html}</div>', unsafe_allow_html=True)

# ============================================================================================ 2. how good is it
_section("How good is it?", f"Every forecast is scored on the same test values, mean absolute error in {unit}", num="②")
_mini("The forecasts on the held-out test", "Sorted by MAE; the serving forecast is shaded")
perf = sorted(perf, key=lambda p_: p_[2]["mae"])
hl = next(i for i, p_ in enumerate(perf) if p_[3])
_rows = []
for nm, kind, sc, srv, m in perf:
    dot_col = CHAMP if srv else (_col(m) if m else (NEUTRAL if kind == "Simple forecast" else LUCK))
    dot = f"<span style='color:{dot_col}'>■</span>&nbsp;"
    _rows.append([f"{dot}{_esc(nm)}", kind, fmt(sc["mae"]), fmt(sc["rmse"]), fmt(sc["bias"])])
_table(["Forecast", "Kind", f"MAE ({unit})", f"RMSE ({unit})", f"Bias ({unit})"], _rows, highlight=hl, widths="minmax(12rem, 2fr) minmax(15rem, 2.2fr) repeat(3, minmax(4.5rem, 0.7fr))", left=(0, 1))
_line("MAE is the mean absolute error. RMSE is the same with squared errors, so large misses weigh more. Bias is the mean of forecast minus actual: below zero the forecast is too low.")

_mini("Calculation")
if weather:
    s_m, p_m = round(serve["mae"], nd), round(plain["mae"], nd)
    sm_m = round(simple["mae"], nd)
    ch1, ch2 = 100 * (s_m / p_m - 1), 100 * (s_m / sm_m - 1)
    st.markdown(_panel(
        f"<div style='display:grid; grid-template-columns:15rem 1fr; row-gap:0.45rem; align-items:baseline;'>"
        f"<span style='font-weight:600; color:{INK};'>① The weather forecast</span><span>MAE <b style='font-size:1.1rem; color:{INK};'>{fmt(serve['mae'])}</b> {unit}</span>"
        f"<span style='font-weight:600; color:{INK};'>② The best algorithm without it</span><span>{_label(plain_m)}, MAE <b style='font-size:1.1rem; color:{INK};'>{fmt(plain['mae'])}</b> {unit}</span>"
        f"<span style='font-weight:600; color:{INK};'>③ The change</span><span>{M(rf"\frac{{{_n(serve['mae'])}}}{{{_n(plain['mae'])}}}-1=")} <b style='font-size:1.25rem; color:{REAL if ch1 < 0 else LUCK};'>{ch1:+.0f}%</b></span>"
        f"<span style='font-weight:600; color:{INK};'>④ Against the best simple forecast</span><span>{simple_n}: {M(rf"\frac{{{_n(serve['mae'])}}}{{{_n(simple['mae'])}}}-1=")} <b style='font-size:1.1rem; color:{INK};'>{ch2:+.0f}%</b></span></div>"), unsafe_allow_html=True)
    rel_w = 100 * (round(with_v["mae"], nd) / round(serve["mae"], nd) - 1)
    verdict_w = (f"better than the forecast itself by {abs(rel_w):.1f}%." if rel_w < -2 else
                 f"about as good as the forecast itself ({rel_w:+.1f}%), so the forecast is used directly." if rel_w <= 2 else
                 f"worse than the forecast itself ({rel_w:+.1f}%), so the forecast is used directly.")
    _line(f"The weather forecast cuts the error by <b>{abs(ch1):.0f}%</b> against the best algorithm that sees only past observations, and by <b>{abs(ch2):.0f}%</b> against the best simple forecast. "
          f"Algorithms that also take the forecast as an input ({_label(with_m)}: {fmt(with_v['mae'])} {unit}) are " + verdict_w)
    _mini("Two limits on these numbers")
    _table(["Limit", "Why it matters"],
           [["1. The truth is a reanalysis", "It is made with a weather model similar to the one that makes the forecast, so the forecast probably looks somewhat better here than it would against a sensor at the plant"],
            ["2. The forecast is two days old", "It is the one issued two days earlier, which is certainly available at the decision time; a real run uses the newest forecast, which is usually a little better"]],
           widths="minmax(12rem, 1fr) minmax(20rem, 3.2fr)", left=(0, 1))
else:
    skill = r.get("skill_vs_yesterday")
    yk = next((k_ for k_ in r["baselines"] if "yesterday" in k_.lower()), simple_n)
    yv = r["baselines"][yk]
    c_m, y_m = round(chosen["mae"], nd), round(yv["mae"], nd)
    sk = 100 * (1 - c_m / y_m) if y_m else None
    st.markdown(_panel(
        f"<div style='display:grid; grid-template-columns:15rem 1fr; row-gap:0.45rem; align-items:baseline;'>"
        f"<span style='font-weight:600; color:{INK};'>① The serving model</span><span>{_label(selected)}, MAE <b style='font-size:1.1rem; color:{INK};'>{fmt(chosen['mae'])}</b> {unit}</span>"
        f"<span style='font-weight:600; color:{INK};'>② '{_esc(yk)}'</span><span>MAE <b style='font-size:1.1rem; color:{INK};'>{fmt(yv['mae'])}</b> {unit}</span>"
        f"<span style='font-weight:600; color:{INK};'>③ The skill against it</span><span>{M(rf"1-\frac{{{_n(chosen['mae'])}}}{{{_n(yv['mae'])}}}=")} <b style='font-size:1.25rem; color:{REAL if (sk or 0) > 0 else LUCK};'>{('–' if sk is None else f'{sk:+.1f}%')}</b></span></div>"), unsafe_allow_html=True)
    if simple["mae"] < chosen["mae"]:
        _line(f"On the held-out year the serving model does not beat the simple forecast '{_esc(simple_n.lower())}': its MAE is {fmt(chosen['mae'])} {unit} against {fmt(simple['mae'])} {unit}. "
              "All forecasts are scored on the same test values; only measured values are scored.")
        _mini("Two limits on these numbers")
        _table(["Limit", "Why it matters"],
               [["1. Yesterday's full day is an input", "Both use it, but at 11:00 on the day before only observations up to about 10:00 exist (the rule of the forecast catalogue), so both scores are optimistic"],
                ["2. The origin of the history", "The origin of the measured history is not independently verified"]],
               widths="minmax(12rem, 1fr) minmax(20rem, 3.2fr)", left=(0, 1))

# ============================================================================================ 3. forecast against what happened
_section("Forecast against what happened", "The daily mean over the test year, then the last seven days hour by hour", num="③")


def _lines(fig, x, ys, widths):
    fig.add_trace(go.Scatter(x=x, y=ys[0], name=names[0], mode="lines", line=dict(color=GREY, width=widths[0])))
    fig.add_trace(go.Scatter(x=x, y=ys[1], name=names[1], mode="lines", line=dict(color=CHAMP, width=widths[1])))
    fig.add_trace(go.Scatter(x=x, y=ys[2], name=names[2], mode="lines", line=dict(color=NEUTRAL, width=widths[2], dash="dot")))


_mini("The daily mean")
fig = go.Figure()
_lines(fig, daily["day"], (daily["actual"], daily["model"], daily["yesterday"]), (3, 1.6, 1.4))
theme.style_fig(fig, height=320, yaxis_title=f"{unit}, daily mean")
fig.update_layout(hovermode="x unified")
_legend_top(fig)
fig.update_layout(margin=dict(t=34))
fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="solid", spikecolor=MUTED)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
if week:
    _mini("The last seven days")
    pos = list(range(len(week["t"])))
    fig = go.Figure()
    _lines(fig, pos, (week["actual"], week["model"], week["yesterday"]), (3.4, 1.8, 1.5))
    theme.style_fig(fig, height=300, yaxis_title=f"{unit}, hourly", xaxis_title=week_x)
    ticks = [i for i, t_ in enumerate(week["t"]) if t_[11:13] == "00"]
    fig.update_xaxes(tickvals=ticks, ticktext=[_dmy(week["t"][i]) for i in ticks], showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="solid", spikecolor=MUTED)
    fig.update_traces(customdata=week["t"], hovertemplate="%{y:,.1f}  (%{customdata})")
    fig.update_layout(hovermode="x unified")
    _legend_top(fig)
    fig.update_layout(margin=dict(t=34))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
