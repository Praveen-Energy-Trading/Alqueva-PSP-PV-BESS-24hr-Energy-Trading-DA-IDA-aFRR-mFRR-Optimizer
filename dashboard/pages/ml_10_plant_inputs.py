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
r = ml_ui.cached(lambda c=choice: ml_data.plant_input(c))()
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
rows = [("Series", f"{r['name']} ({unit}), {r['resolution']}"), ("Inputs of the model", r["features"].replace("_", " ")),
        ("History", f"development {r['dev_range'][0]} to {r['dev_range'][1]} ({r['n_dev']:,} {r['resolution']} values); test {r['test_range'][0]} to {r['test_range'][1]} ({r['n_test']:,})")]
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
             ("Serving model against 'same time yesterday'", f"{100 * skill:+.1f}%" if skill is not None else "-")], ncols=3, pre_formatted=True)

table = [{"Forecast": ml_data.MODEL_LABELS.get(m, m) + ("  (serving)" if m == selected else ""), "Kind": "Algorithm", "MAE": s_["mae"], "RMSE": s_["rmse"], "Bias": s_["bias"]}
         for m, s_ in r["models"].items()]
table += [{"Forecast": n, "Kind": "Simple forecast", "MAE": s_["mae"], "RMSE": s_["rmse"], "Bias": s_["bias"]} for n, s_ in simple.items()]
df = pd.DataFrame(table).sort_values("MAE").reset_index(drop=True)
nd = 0 if unit == "m³/h" else 2
ml_ui.table(df.rename(columns={"MAE": f"MAE ({unit})", "RMSE": f"RMSE ({unit})", "Bias": f"Bias ({unit})"}),
            formats={f"MAE ({unit})": f"%.{nd}f", f"RMSE ({unit})": f"%.{nd}f", f"Bias ({unit})": f"%.{nd}f"}, stretch=True)
if best_simple["mae"] < chosen["mae"]:
    st.info(f"On the held-out year the simple forecast '{best_simple_name.lower()}' has a lower MAE ({fmt(best_simple['mae'])} {unit}) than the serving model "
            f"({fmt(chosen['mae'])} {unit}). All forecasts are scored on the same test values; only measured values are scored.")

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
