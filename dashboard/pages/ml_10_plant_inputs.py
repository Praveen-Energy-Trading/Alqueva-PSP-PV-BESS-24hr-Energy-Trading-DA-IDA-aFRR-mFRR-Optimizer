"""ML Lifecycle 10 - Plant inputs: the forecasts of solar irradiance, air temperature and natural inflow that feed the optimiser."""
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

ml_ui.header("Plant inputs: PV and inflow", "", step=9)
ml_ui.tour_bar(10)

labels = {key: name for key, name, _ in ml_data.PLANT_INPUTS}
choice = st.selectbox("Forecast", [key for key, _, _ in ml_data.PLANT_INPUTS], format_func=lambda k: labels[k], key="ml10_choice")
w = ml_ui.cached(ml_data.weather_eval)(choice) if choice in ("ghi", "temperature") else None
if w and "daily" in w:                                         # irradiance and temperature: the weather forecast against the algorithms, as of the decision time
    _dmy = lambda x: f"{str(x)[8:10]}-{str(x)[5:7]}-{str(x)[:4]}"
    _catalogue = ml_ui.cached(lambda: ml_data.config("forecast_catalogue"))()
    _entry = next((t for t in _catalogue.get("targets", []) if t.get("id") == "pv_weather"), {})
    _unit, _nd = w["unit"], (0 if w["unit"] == "W/m²" else 2)
    _esc = lambda v: __import__("html").escape(str(v))
    _tiles = [("Feeds", _entry.get("decision", "-"), theme.BLUE), ("Needed by", _entry.get("as_of", {}).get("rule", "-"), theme.AQUA),
              ("Serving forecast", "Weather forecast (Open-Meteo), two days old", theme.ORANGE)]
    _tile_html = "".join(
        f'<div style="flex:1 1 0; min-width:0; background:{c}12; border-left:4px solid {c}; border-radius:10px; padding:0.55rem 0.9rem;">'
        f'<div style="font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase; color:{theme.INK_MUTED}; font-weight:600;">{_esc(k)}</div>'
        f'<div style="font-size:0.93rem; font-weight:600; color:{theme.INK_PRIMARY}; margin-top:0.1rem; overflow-wrap:anywhere;">{_esc(v)}</div></div>' for k, v, c in _tiles)
    _d = w["data"]
    _rows = [("Series", f"{w['target']} ({_unit}), hourly"),
             ("Inputs", "the numerical weather forecast of the site; without it, a model on past observations and clear-sky physics"),
             ("Truth", f"{_d['source']}, {_dmy(_d['first'])} to {_dmy(_d['last'])}. A reanalysis, not an on-site measurement; the weather forecasts are archived from {_dmy(_d['forecast_first'])}."),
             ("Test", f"the last twelve months, {_dmy(w['test_range'][0])} to {_dmy(w['test_range'][1])} ({w['n_test']:,} hours), judged as of 11:00 on the day before: "
                      "no observation of the last hours before the delivery day is used by anything below")]
    _row_html = "".join(
        f'<div style="display:grid; grid-template-columns:8rem 1fr; gap:0.8rem; padding:0.42rem 0; border-top:1px solid {theme.GRIDLINE};">'
        f'<div style="font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase; color:{theme.INK_MUTED}; font-weight:600; padding-top:0.15rem;">{_esc(k)}</div>'
        f'<div style="font-size:0.9rem; color:{theme.INK_PRIMARY}; line-height:1.45; overflow-wrap:anywhere;">{_esc(v)}</div></div>' for k, v in _rows)
    st.markdown(f'<div style="border:1px solid {theme.GRIDLINE}; border-radius:14px; background:{theme.SURFACE}; padding:0.85rem 1rem 0.5rem; box-shadow:0 1px 3px rgba(0,0,0,0.05);">'
                f'<div style="display:flex; gap:0.7rem; margin-bottom:0.7rem;">{_tile_html}</div>{_row_html}</div>', unsafe_allow_html=True)

    st.subheader("How good is it?")
    _f = lambda v: f"{v:,.{_nd}f}"
    _serve = w["baselines"]["Weather forecast alone"]
    _plain = min(w["without_forecast_full_history"].items(), key=lambda kv: kv[1]["mae"])
    _simple = min((kv for kv in w["baselines"].items() if kv[0] != "Weather forecast alone"), key=lambda kv: kv[1]["mae"])
    ml_ui.cards([("Error of the weather forecast (MAE)", f"{_f(_serve['mae'])} {_unit}"),
                 (f"Best algorithm without it ({ml_data.MODEL_LABELS.get(_plain[0], _plain[0])})", f"{_f(_plain[1]['mae'])} {_unit}"),
                 ("Weather forecast against that algorithm", theme.pct(100 * (_serve["mae"] / _plain[1]["mae"] - 1), 0, signed=True))], ncols=3, pre_formatted=True)
    _table = [{"Forecast": "Weather forecast (serving)", "Kind": "Weather forecast", "MAE": _serve["mae"], "RMSE": _serve["rmse"], "Bias": _serve["bias"]}]
    _table += [{"Forecast": k, "Kind": "Simple forecast", "MAE": v["mae"], "RMSE": v["rmse"], "Bias": v["bias"]} for k, v in w["baselines"].items() if k != "Weather forecast alone"]
    _table += [{"Forecast": ml_data.MODEL_LABELS.get(m, m), "Kind": "Algorithm, past observations only", "MAE": v["mae"], "RMSE": v["rmse"], "Bias": v["bias"]}
               for m, v in w["without_forecast_full_history"].items()]
    _table += [{"Forecast": ml_data.MODEL_LABELS.get(m, m) + " + weather forecast", "Kind": "Algorithm with the forecast as an input", "MAE": v["mae"], "RMSE": v["rmse"], "Bias": v["bias"]}
               for m, v in w["with_forecast"].items()]
    ml_ui.table(pd.DataFrame(_table).sort_values("MAE").reset_index(drop=True).rename(columns={"MAE": f"MAE ({_unit})", "RMSE": f"RMSE ({_unit})", "Bias": f"Bias ({_unit})"}),
                formats={f"MAE ({_unit})": f"%.{_nd}f", f"RMSE ({_unit})": f"%.{_nd}f", f"Bias ({_unit})": f"%.{_nd}f"}, stretch=True)
    _best_with = min(w["with_forecast"].items(), key=lambda kv: kv[1]["mae"])
    st.info(f"The weather forecast cuts the error by {abs(100 * (_serve['mae'] / _plain[1]['mae'] - 1)):.0f}% against the best algorithm that sees only past observations, "
            f"and by {abs(100 * (_serve['mae'] / _simple[1]['mae'] - 1)):.0f}% against the best simple forecast. Algorithms that also take the forecast as an input "
            f"({ml_data.MODEL_LABELS.get(_best_with[0], _best_with[0])}: {_f(_best_with[1]['mae'])} {_unit}) are no better than the forecast itself, so the forecast is used directly.\n\n"
            "Two limits. **1.** The truth is a reanalysis made with a weather model similar to the one that makes the forecast, so the forecast probably looks somewhat "
            "better here than it would against a sensor at the plant. **2.** The forecast is the one issued two days earlier, which is certainly available at the decision "
            "time; a real run uses the newest forecast, which is usually a little better.")

    st.subheader("Forecast against what happened")
    _d = w["daily"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_d["day"], y=_d["actual"], name="Reanalysis (what happened)", mode="lines", line=dict(color=theme.INK_PRIMARY, width=1.6)))
    fig.add_trace(go.Scatter(x=_d["day"], y=_d["model"], name="Weather forecast", mode="lines", line=dict(color=theme.BLUE, width=1.6)))
    fig.add_trace(go.Scatter(x=_d["day"], y=_d["yesterday"], name="Newest same-hour observation", mode="lines", line=dict(color=theme.ORANGE, width=1.2, dash="dot")))
    theme.style_fig(fig, height=330, yaxis_title=f"{w['target']} ({_unit}), daily mean")
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="solid", spikecolor=theme.INK_MUTED)
    st.plotly_chart(fig, width="stretch")
    _w = w["week"]
    pos = list(range(len(_w["t"])))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos, y=_w["actual"], name="Reanalysis (what happened)", mode="lines", line=dict(color=theme.INK_PRIMARY, width=1.8)))
    fig.add_trace(go.Scatter(x=pos, y=_w["model"], name="Weather forecast", mode="lines", line=dict(color=theme.BLUE, width=1.8)))
    fig.add_trace(go.Scatter(x=pos, y=_w["yesterday"], name="Newest same-hour observation", mode="lines", line=dict(color=theme.ORANGE, width=1.4, dash="dot")))
    theme.style_fig(fig, height=300, yaxis_title=f"{w['target']} ({_unit}), hourly", xaxis_title="the last seven days of the test year (hourly, UTC)")
    ticks = [i for i, t in enumerate(_w["t"]) if t[11:13] == "00"]
    fig.update_xaxes(tickvals=ticks, ticktext=[f"{_w['t'][i][8:10]}-{_w['t'][i][5:7]}-{_w['t'][i][:4]}" for i in ticks],
                     showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="solid", spikecolor=theme.INK_MUTED)
    fig.update_traces(customdata=_w["t"], hovertemplate="%{y:,.1f}  (%{customdata})")
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
    st.plotly_chart(fig, width="stretch")
    st.stop()

r = ml_ui.cached(ml_data.plant_input)(choice)                  # the choice is an argument, so each forecast has its own cache entry
if not r:
    ml_ui.not_yet_run(f"the {labels[choice].lower()} evaluation (tools/evaluate_plant_inputs.py)")
    st.stop()

catalogue = ml_ui.cached(lambda: ml_data.config("forecast_catalogue"))()
entry_id = next(cid for key, _, cid in ml_data.PLANT_INPUTS if key == choice)
entry = next((t for t in catalogue.get("targets", []) if t.get("id") == entry_id), {})
unit = r["unit"]
selected = r["selected"]
chosen = r["models"][selected]
simple = r["baselines"]
best_simple_name, best_simple = min(simple.items(), key=lambda kv: kv[1]["mae"])
fmt = (lambda v: f"{v:,.0f}") if unit == "m³/h" else (lambda v: f"{v:,.2f}")


def _esc(v) -> str:
    return html.escape(str(v))


# ---------------------------------------------------------------------------------------------------------------- what it is
tiles = [("Feeds", entry.get("decision", "-"), theme.BLUE), ("Needed by", entry.get("as_of", {}).get("rule", "-"), theme.AQUA),
         ("Serving model", f"{ml_data.MODEL_LABELS.get(selected, selected)}, retrained daily", theme.ORANGE)]
tile_html = "".join(
    f'<div style="flex:1 1 0; min-width:0; background:{c}12; border-left:4px solid {c}; border-radius:10px; padding:0.55rem 0.9rem;">'
    f'<div style="font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase; color:{theme.INK_MUTED}; font-weight:600;">{_esc(k)}</div>'
    f'<div style="font-size:0.93rem; font-weight:600; color:{theme.INK_PRIMARY}; margin-top:0.1rem; overflow-wrap:anywhere;">{_esc(v)}</div></div>'
    for k, v, c in tiles)
_days = (lambda n: round(n / 24)) if r["resolution"] == "hourly" else (lambda n: n)
rows = [("Series", f"{r['name']} ({unit}), {r['resolution']}"), ("Inputs of the model", r["features"].replace("_", " ")),
        ("History", f"development {r['dev_range'][0]} to {r['dev_range'][1]} ({_days(r['n_dev']):,} days); test {r['test_range'][0]} to {r['test_range'][1]} ({_days(r['n_test']):,} days)")]
_d = r.get("data") or {}
_dmy = lambda x: f"{str(x)[8:10]}-{str(x)[5:7]}-{str(x)[:4]}"
if _d:
    _src = "Plant sensor" if choice in ("ghi", "temperature") else "Guadiana inflow research dataset"
    _kind = "synthetic" if choice in ("ghi", "temperature") else "monthly means (climatology)"
    rows.append(("Data", f"{_src}, measured {_dmy(_d['first'])} to {_dmy(_d['last_measured'])} ({_d['measured']:,} values). "
                         f"The {_d['filled']:,} later rows are {_kind} and are never scored."))
row_html = "".join(
    f'<div style="display:grid; grid-template-columns:8rem 1fr; gap:0.8rem; padding:0.42rem 0; border-top:1px solid {theme.GRIDLINE};">'
    f'<div style="font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase; color:{theme.INK_MUTED}; font-weight:600; padding-top:0.15rem;">{_esc(k)}</div>'
    f'<div style="font-size:0.9rem; color:{theme.INK_PRIMARY}; line-height:1.45; overflow-wrap:anywhere;">{_esc(v)}</div></div>'
    for k, v in rows)
st.markdown(f'<div style="border:1px solid {theme.GRIDLINE}; border-radius:14px; background:{theme.SURFACE}; padding:0.85rem 1rem 0.5rem; box-shadow:0 1px 3px rgba(0,0,0,0.05);">'
            f'<div style="display:flex; gap:0.7rem; margin-bottom:0.7rem;">{tile_html}</div>{row_html}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------------------------------- how good
st.subheader("How good is it?")
skill = r.get("skill_vs_yesterday")
ml_ui.cards([("Error of the serving model (MAE)", f"{fmt(chosen['mae'])} {unit}"), (f"Best simple forecast ({best_simple_name.lower()})", f"{fmt(best_simple['mae'])} {unit}"),
             ("Serving model against 'same time yesterday'", theme.pct(100 * skill, 1, signed=True) if skill is not None else "-")], ncols=3, pre_formatted=True)

table = [{"Forecast": ml_data.MODEL_LABELS.get(m, m) + ("  (serving)" if m == selected else ""), "Kind": "Algorithm", "MAE": s_["mae"], "RMSE": s_["rmse"], "Bias": s_["bias"]}
         for m, s_ in r["models"].items()]
table += [{"Forecast": n, "Kind": "Simple forecast", "MAE": s_["mae"], "RMSE": s_["rmse"], "Bias": s_["bias"]} for n, s_ in simple.items()]
df = pd.DataFrame(table).sort_values("MAE").reset_index(drop=True)
nd = 0 if unit == "m³/h" else 2
ml_ui.table(df.rename(columns={"MAE": f"MAE ({unit})", "RMSE": f"RMSE ({unit})", "Bias": f"Bias ({unit})"}),
            formats={f"MAE ({unit})": f"%.{nd}f", f"RMSE ({unit})": f"%.{nd}f", f"Bias ({unit})": f"%.{nd}f"}, stretch=True)
if best_simple["mae"] < chosen["mae"]:
    st.info(f"On the held-out year the serving model does not beat the simple forecast '{best_simple_name.lower()}': its MAE is {fmt(chosen['mae'])} {unit} "
            f"against {fmt(best_simple['mae'])} {unit}. All forecasts are scored on the same test values; only measured values are scored.\n\n"
            "Two limits on these numbers. **1.** Both use yesterday's full day as an input, but at 11:00 on the day before only observations up to about 10:00 exist "
            "(the rule of the forecast catalogue), so both scores are optimistic. **2.** The origin of the measured history is not independently verified.")

# ---------------------------------------------------------------------------------------------------------------- charts
st.subheader("Forecast against measured")
d = r["daily"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=d["day"], y=d["actual"], name="Measured", mode="lines", line=dict(color=theme.INK_PRIMARY, width=1.6)))
fig.add_trace(go.Scatter(x=d["day"], y=d["model"], name="Serving model", mode="lines", line=dict(color=theme.BLUE, width=1.6)))
fig.add_trace(go.Scatter(x=d["day"], y=d["yesterday"], name="Same time yesterday", mode="lines", line=dict(color=theme.ORANGE, width=1.2, dash="dot")))
theme.style_fig(fig, height=330, yaxis_title=f"{r['name']} ({unit}), daily mean")
fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="solid", spikecolor=theme.INK_MUTED)
st.plotly_chart(fig, width="stretch")
if "week" in r:
    w = r["week"]
    pos = list(range(len(w["t"])))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos, y=w["actual"], name="Measured", mode="lines", line=dict(color=theme.INK_PRIMARY, width=1.8)))
    fig.add_trace(go.Scatter(x=pos, y=w["model"], name="Serving model", mode="lines", line=dict(color=theme.BLUE, width=1.8)))
    fig.add_trace(go.Scatter(x=pos, y=w["yesterday"], name="Same time yesterday", mode="lines", line=dict(color=theme.ORANGE, width=1.4, dash="dot")))
    theme.style_fig(fig, height=300, yaxis_title=f"{r['name']} ({unit}), hourly", xaxis_title="the last seven days of the test year (hourly)")
    ticks = [i for i, t in enumerate(w["t"]) if t[11:13] == "00"]
    fig.update_xaxes(tickvals=ticks, ticktext=[f"{w['t'][i][8:10]}-{w['t'][i][5:7]}-{w['t'][i][:4]}" for i in ticks],
                     showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="solid", spikecolor=theme.INK_MUTED)
    fig.update_traces(customdata=w["t"], hovertemplate="%{y:,.1f}  (%{customdata})")
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
    st.plotly_chart(fig, width="stretch")
