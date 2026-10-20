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

ml_ui.header("Plant inputs: PV and inflow", "", step=8)
ml_ui.tour_bar(9)

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
            ("Real value", f"The real value y of every hour is taken from {d_['source']}, {_dmy(d_['first'])} to {_dmy(d_['last'])}. It is a reanalysis (a weather model fitted to observations), not a sensor at the plant; the weather forecasts are archived from {_dmy(d_['forecast_first'])}."),
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
st.markdown(f'<div style="display:flex; gap:0.7rem; margin-bottom:0.6rem;">{tile_html}</div>', unsafe_allow_html=True)
with st.expander("The series, its inputs, the real value and the test"):                                                       # the detail, opened only when asked
    st.markdown(f'<div style="border:1px solid {CARD_LINE}; border-radius:12px; background:#fff; padding:0.2rem 1rem 0.4rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);">{row_html}</div>', unsafe_allow_html=True)

# ============================================================================================ 2. how good is it
_section("How good is it?", f"Every forecast is scored on the same test values, mean absolute error in {unit}", num="②")
_mini("The forecasts on the held-out test", "Sorted by MAE; the serving forecast is shaded")
perf = sorted(perf, key=lambda p_: p_[2]["mae"])
hl = next(i for i, p_ in enumerate(perf) if p_[3])
_rows = []
for nm, kind, sc, srv, m in perf:
    dot_col = CHAMP if srv else (_col(m) if m else (NEUTRAL if kind == "Simple forecast" else LUCK))
    dot = f"<span style='color:{dot_col}'>■</span>&nbsp;"
    _rows.append([f"{dot}{_esc(nm)}", kind, fmt(sc["mae"])])
_table(["Forecast", "Kind", f"MAE ({unit})"], _rows, highlight=hl, widths="minmax(12rem, 2fr) minmax(15rem, 2.2fr) minmax(6rem, 0.8fr)", left=(0, 1))
if weather:                                                                                                              # three ways to a forecast, built from the start, step by step, side by side
    _tg = "irradiance" if choice == "ghi" else "temperature"
    _obs = "irradiance" if choice == "ghi" else "temperature"
    _cs = "clear-sky irradiance, the irradiance with no cloud at that hour"
    _in = (f"hour and month (as sine and cosine), day of the year, {_cs}; the newest {_obs} observed at the same hour, the one 48 and 168 hours earlier and the mean of the 24 hours ending 48 hours earlier"
           + ("; the newest observation divided by the clear-sky irradiance" if choice == "ghi" else ""))
    _fc_in = "the weather forecast of that hour" + (" and the forecast divided by the clear-sky irradiance" if choice == "ghi" else "")
    _best = lambda d_: min(d_.items(), key=lambda kv: kv[1]["mae"])
    _bf, _bw, _bs = _best(w["with_forecast"]), _best(w["without_forecast_full_history"]), _best(w["without_forecast_same_period"])
    _all5 = lambda d_: "<br>".join(f"{'<b>' if k_ == _best(d_)[0] else ''}{_label(k_)} {fmt(v_['mae'])}{'</b>' if k_ == _best(d_)[0] else ''}" for k_, v_ in sorted(d_.items(), key=lambda kv: kv[1]["mae"]))
    _dmy2 = lambda s_: f"{s_[8:10]}-{s_[5:7]}-{s_[:4]}"
    _t0, _t1 = _dmy2(w["test_range"][0]), _dmy2(w["test_range"][1])
    _d0f, _d1f = _dmy2(w["dev_forecast_range"][0]), _dmy2(w["dev_forecast_range"][1])
    _first = _dmy2(w["data"]["first"])
    _ntest, _nfull, _nfc = f"{w['n_test']:,}", f"{w['n_dev_full']:,}", f"{w['n_dev_forecast']:,}"
    _cols = [("Weather forecast alone", CHAMP), ("Algorithm + weather forecast", theme.VIOLET), ("Algorithm alone, past observations only", theme.BLUE)]
    _steps = [
        ("1", "The real value", [f"The real {_tg} y of every hour: the ERA5 reanalysis of the site"] * 3),
        ("2", "What it uses for one hour",
         [f"Only the numerical weather forecast of that hour, issued two days earlier",
          f"{_in}; plus {_fc_in}",
          f"{_in}. Nothing from the weather forecast"]),
        ("3", "What it learns from",
         ["Nothing: no model is fitted, nothing is tuned",
          f"The {_nfc} hours from {_d0f} to {_d1f}: the only hours with an archived forecast",
          f"The {_nfull} hours from {_first} to {_d1f}, all the history before the test"]),
        ("4", "How it is built",
         ["The forecast is taken as it is",
          "Each of five tree algorithms learns, from those hours, how its inputs link to the real value",
          "The same five algorithms learn the same link, without the forecast"]),
        ("5", "How an hour is forecast",
         [f"For each of the {_ntest} hours from {_t0} to {_t1}: ŷ = the weather forecast of that hour",
          f"For each of the {_ntest} test hours: the inputs go into the fitted algorithm and it returns ŷ. Only what is known at 11:00 on the day before is used: the newest observation is yesterday's if the hour is up to 09:00 UTC, else the day before",
          "The same, with the same rule on the newest observation"]),
        ("6", "The score",
         [f"e = ŷ − y for every hour, MAE = the mean of |e|: <b>{fmt(w['baselines']['Weather forecast alone']['mae'])}</b> {unit}",
          f"The same MAE for each algorithm, in {unit}:<br>{_all5(w['with_forecast'])}",
          f"The same MAE for each algorithm, in {unit}:<br>{_all5(w['without_forecast_full_history'])}"]),
        ("7", "A fair check",
         ["Nothing to compare",
          f"The algorithms alone, trained on the same {_nfc} hours, score {fmt(_bs[1]['mae'])} at best ({_label(_bs[0])}): so a gain is not only a shorter history",
          f"Trained on the whole history: {fmt(_bw[1]['mae'])} at best ({_label(_bw[0])})"]),
    ]
    _g = "grid-template-columns:7.5rem repeat(3, minmax(0, 1fr)); column-gap:0.7rem;"
    _head = "".join(f"<div style='border-top:3px solid {c_}; background:{c_}14; border-radius:8px 8px 0 0; padding:0.4rem 0.7rem; font-weight:700; font-size:0.86rem; color:{INK};'>{n_}</div>" for n_, c_ in _cols)
    _body = ""
    for num_, ttl_, cells_ in _steps:
        _body += (f"<div style='display:grid; {_g} border-top:1px solid {CARD_LINE}; padding:0.5rem 0;'>"
                  f"<div style='font-size:0.8rem; font-weight:700; color:{INK}; line-height:1.3;'><span style='display:inline-flex; align-items:center; justify-content:center; width:1.15rem; height:1.15rem; border-radius:50%; background:{NEUTRAL}; color:#fff; font-size:0.66rem; margin-right:0.35rem;'>{num_}</span>{ttl_}</div>"
                  + "".join(f"<div style='font-size:0.78rem; color:{theme.INK_SECONDARY}; line-height:1.4; overflow-wrap:anywhere;'>{c_}</div>" for c_ in cells_) + "</div>")
    with st.expander("How each forecast is calculated, from the start"):
        st.markdown(f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0.6rem 1rem 0.3rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>"
                    f"<div style='display:grid; {_g} padding-bottom:0.4rem;'><div></div>{_head}</div>{_body}</div>", unsafe_allow_html=True)
else:
    with st.expander("How each MAE is calculated, step by step"):                                                          # the blackboard version, opened only when asked
        _n_all = w["n_test"] if weather else r["n_test"]
        _per = "hour" if weather else "day"
        _ser = week if weather else daily                                                                                    # the series stored for the picture: the real values, the serving forecast, the simple one
        _tk = list(_ser["t"] if weather else _ser["day"])
        _feat = (p_["features"] if (weather and p_) else r["features"] if not weather else "").replace("_", " ")
        _opt = [row_[0] for row_ in perf]
        _pick_f = st.selectbox("Forecast", _opt, key="ml10_calc_forecast")
        _row = next(row_ for row_ in perf if row_[0] == _pick_f)
        _kind, _sc = _row[1], _row[2]
        _has = {"model": _pick_f == serve_name, "yesterday": (_pick_f == "Newest same-hour observation") if weather else (_pick_f == "Same time yesterday")}
        _sk = "model" if _has["model"] else "yesterday" if _has["yesterday"] else None

        def _step(num, title):
            st.markdown(f"<div style='font-weight:700; color:{INK}; margin:0.8rem 0 0.2rem;'><span style='color:{MUTED};'>{num}</span>&nbsp; {title}</div>", unsafe_allow_html=True)

        _step("①", "How the forecast of one " + _per + " is made")
        if _kind == "Weather forecast":
            _how = "No model of ours: the numerical weather forecast of the site, issued two days earlier, taken for that hour."
        elif _kind.startswith("Algorithm, past"):
            _how = f"{_pick_f} is fitted on the past observations and predicts the hour from: {_feat}. It does not see the weather forecast."
        elif _kind.startswith("Algorithm with"):
            _how = f"{_pick_f.replace(' + weather forecast', '')} is fitted on the past observations and predicts the hour from: {_feat}, plus the weather forecast of that hour."
        elif _kind == "Algorithm":
            _how = f"{_pick_f.replace(' (serving)', '')} is fitted on the history and predicts the {_per} from: {_feat}. It is retrained daily."
        elif _pick_f in ("Newest same-hour observation", "Same time yesterday"):
            _how = "No model: the value observed at the same " + ("hour" if weather else "time") + " on the newest day with an observation, which is yesterday."
        elif "two days" in _pick_f:
            _how = "No model: the value observed at the same hour two days earlier."
        elif "week" in _pick_f:
            _how = "No model: the value observed at the same time one week earlier."
        elif "Average of the month" in _pick_f:
            _how = "No model: the average of that month's observations."
        else:
            _how = _row[1] + "."
        st.markdown(f"<div style='font-size:0.9rem; color:{theme.INK_SECONDARY};'>{_how}</div>", unsafe_allow_html=True)
        _m_all = _sc["mae"]
        if _sk is not None:
            _dys = sorted({str(k_)[:10] for k_ in _tk})
            _dy = st.selectbox("Example day", _dys[::-1], format_func=lambda d_: f"{d_[8:10]}-{d_[5:7]}-{d_[:4]}", key="ml10_calc_day")
            _idx = [i_ for i_, k_ in enumerate(_tk) if str(k_)[:10] == _dy]
            if weather:
                _idx = [i_ for i_ in _idx if str(_tk[i_])[11:13] in ("02", "06", "10", "14", "18", "22")]
            _idx = _idx[:6]
            _y = [_ser["actual"][i_] for i_ in _idx]
            _f = [_ser[_sk][i_] for i_ in _idx]
            _e = [f_ - y_ for f_, y_ in zip(_f, _y)]
            _step("②", f"The error of one {_per}: forecast minus real value")
            st.markdown(M(r"e=\hat{y}-y", display=False, size=1.05) + f" &nbsp;in {unit}", unsafe_allow_html=True)
            _rows = [[(str(_tk[i_])[11:16] if weather else f"{_dy[8:10]}-{_dy[5:7]}-{_dy[:4]}"), fmt(y_), fmt(f_), f"{e_:+.{nd}f}", fmt(abs(e_))] for i_, y_, f_, e_ in zip(_idx, _y, _f, _e)]
            _table([_per.capitalize(), "Real y", "Forecast ŷ", "Error e", "|e|"], _rows, widths="minmax(5rem, 1fr) repeat(4, minmax(4.5rem, 1fr))", left=(0,))
            if _e:
                _step("③", "MAE: the average of the absolute errors")
                _num = " + ".join(fmt(abs(e_)) for e_ in _e)
                st.markdown(M(r"\mathrm{MAE}=\frac{1}{n}\sum|e|", display=False, size=1.05) + " &nbsp;&nbsp; " + M(rf"\frac{{{_num}}}{{{len(_e)}}}={sum(abs(e_) for e_ in _e) / len(_e):.{nd}f}", display=False, size=1.0)
                            + f" &nbsp;{unit} on these {len(_e)} {_per}s", unsafe_allow_html=True)
            _step("④", f"The same average over every {_per} of the test")
        else:
            _step("②", "MAE: the average of the absolute errors")
            st.markdown(M(r"e=\hat{y}-y", display=False, size=1.05) + " &nbsp;&nbsp; " + M(r"\mathrm{MAE}=\frac{1}{n}\sum|e|", display=False, size=1.05), unsafe_allow_html=True)
            _line(f"The forecasts of the single {_per}s of this forecast are not kept on this page; the average over the whole test is.")
            _step("③", f"The average over every {_per} of the test")
        st.markdown(M(rf"\mathrm{{MAE}}=\frac{{1}}{{n}}\sum|e|\quad n={_n_all:,}".replace(",", "{,}") + rf"\quad\Rightarrow\quad {fmt(_m_all)}", display=False, size=1.05) + f" &nbsp;{unit}, the figure in the table.", unsafe_allow_html=True)

if not weather and simple["mae"] < chosen["mae"]:
    _line(f"On the held-out year the serving model does not beat the simple forecast '{_esc(simple_n.lower())}': its MAE is {fmt(chosen['mae'])} {unit} against {fmt(simple['mae'])} {unit}. "
          "All forecasts are scored on the same test values; only measured values are scored.")

# ============================================================================================ 3. forecast against what happened: one picture, one day
def _key_cards(items):
    """What each line of the graph is, one row each, in a drop-down: the line, what it is, and one worked example from the chosen day."""
    with st.expander("What each line of the graph is"):
        _table(["Line", "What it is", "Example"],
               [[f"<span style='color:{c_};'>■</span>&nbsp;<b>{n_}</b>", w_, e_] for c_, n_, w_, e_ in items],
               widths="minmax(11rem, 1.1fr) minmax(18rem, 2.4fr) minmax(14rem, 1.7fr)", left=(0, 1, 2))


_AMB = "rgba(237,161,0,0.32)"                                                                                            # the miss: the gap between the real value and the forecast
_section("The table, on one day", "Two forecasts of the table, whose hours are kept, against the real value y. The shaded gap is the error e = ŷ − y; the MAE of the table is its average over every test hour", num="③")
fig = go.Figure()
if weather and week:
    _wt = pd.to_datetime(pd.Series(week["t"]))
    _wd = pd.DataFrame({"t": _wt, "d": _wt.dt.strftime("%Y-%m-%d"), "h": _wt.dt.hour, "y": week["actual"], "f": week["model"], "n": week["yesterday"]})
    _full = [d_ for d_, g_ in _wd.groupby("d") if len(g_) == 24]                                                         # days with all 24 hours
    _day = st.select_slider("Day", options=_full, value=_full[-1], format_func=lambda d_: f"{d_[8:10]}-{d_[5:7]}-{d_[:4]}", key="ml10_day")
    g_ = _wd[_wd["d"] == _day].sort_values("h")
    _mae = lambda s_: float((g_["y"] - g_[s_]).abs().mean())
    x_ = g_["h"]
    _eh = 14 if 14 in set(g_["h"]) else int(g_["h"].iloc[len(g_) // 2])                                                    # the example hour: 14:00 when it exists
    _er = g_[g_["h"] == _eh].iloc[0]
    _dd = pd.Timestamp(_day)
    _sd = _dd - pd.Timedelta(days=1 if _eh <= 9 else 2)                                                                  # the newest day observed at 11:00 on the day before, for that hour
    _dmy3 = lambda d_: f"{d_:%d-%m-%Y}"
    _where = f"{_eh:02d}:00 on {_dmy3(_dd)}"
    _key_items = ([
        (GREY, "Real value y", f"The {_tg} that really happened in that hour (the ERA5 reanalysis of the site).", f"{_where}: <b>{fmt(_er['y'])}</b> {unit}."),
        (CHAMP, "Weather forecast ŷ", "The weather model's forecast for that hour, issued two days earlier. Nothing is fitted: it is taken as it is.", f"For {_where} it said <b>{fmt(_er['f'])}</b> {unit}."),
        (theme.BLUE, "Newest same-hour observation ŷ", "No model: the real value of the same hour on the newest day observed at 11:00 on the day before (yesterday up to 09:00 UTC, otherwise the day before yesterday).",
         f"For {_where} it is the real value of {_eh:02d}:00 on {_dmy3(_sd)}: <b>{fmt(_er['n'])}</b> {unit}."),
        (_AMB.replace("0.32", "1"), "Error e = ŷ − y", "The forecast minus the real value: the shaded gap. The MAE is the average of its size.", f"{_where}: {fmt(_er['f'])} − {fmt(_er['y'])} = <b>{_er['f'] - _er['y']:+.{nd}f}</b> {unit} for the weather forecast.")]
    )
    fig.add_trace(go.Scatter(x=x_, y=g_["y"], name="Real value y", mode="lines", line=dict(color=GREY, width=0), fill="tozeroy", fillcolor="rgba(185,183,174,0.45)", hovertemplate="%{y:,.1f}"))
    fig.add_trace(go.Scatter(x=x_, y=g_["f"], name=f"Weather forecast ŷ (MAE of the day {fmt(_mae('f'))})", mode="lines", line=dict(color=CHAMP, width=2.4), fill="tonexty", fillcolor=_AMB, hovertemplate="%{y:,.1f}"))
    fig.add_trace(go.Scatter(x=x_, y=g_["n"], name=f"Newest same-hour observation ŷ (MAE of the day {fmt(_mae('n'))})", mode="lines", line=dict(color=theme.BLUE, width=1.8, dash="dot"), hovertemplate="%{y:,.1f}"))
    fig.add_trace(go.Scatter(x=[None], y=[None], name="Error e (the shaded gap)", mode="markers", marker=dict(symbol="square", size=11, color=_AMB)))
    theme.style_fig(fig, height=360, yaxis_title=unit, xaxis_title="Hour of the day (UTC)")
    fig.update_xaxes(tickvals=list(range(0, 24, 3)), ticktext=[f"{h_:02d}:00" for h_ in range(0, 24, 3)], range=[0, 23])
elif not weather:
    _lastd = len(daily["day"]) - 1
    _key_items = ([
        (GREY, "Measured y", "The inflow that was really measured on that day.", f"{daily['day'][_lastd]}: <b>{fmt(daily['actual'][_lastd])}</b> {unit}."),
        (CHAMP, f"{_label(selected)} ŷ", "The serving model's forecast for that day, retrained daily on the history.", f"For {daily['day'][_lastd]} it said <b>{fmt(daily['model'][_lastd])}</b> {unit}."),
        (theme.BLUE, "Same time yesterday ŷ", "No model: the value measured the day before.", f"For {daily['day'][_lastd]} it is yesterday's value: <b>{fmt(daily['yesterday'][_lastd])}</b> {unit}."),
        (_AMB.replace("0.32", "1"), "Error e = ŷ − y", "The forecast minus the measured value: the shaded gap.", f"{fmt(daily['model'][_lastd])} − {fmt(daily['actual'][_lastd])} = <b>{daily['model'][_lastd] - daily['actual'][_lastd]:+.{nd}f}</b> {unit} for the model.")]
    )
    _n_ = min(60, len(daily["day"]))
    _x = list(daily["day"])[-_n_:]
    _y = list(daily["actual"])[-_n_:]
    _f = list(daily["model"])[-_n_:]
    _nv = list(daily["yesterday"])[-_n_:]
    fig.add_trace(go.Scatter(x=_x, y=_y, name="Measured y", mode="lines", line=dict(color=GREY, width=0), fill="tozeroy", fillcolor="rgba(185,183,174,0.45)"))
    fig.add_trace(go.Scatter(x=_x, y=_f, name=f"{_label(selected)} ŷ", mode="lines", line=dict(color=CHAMP, width=2.4), fill="tonexty", fillcolor=_AMB))
    fig.add_trace(go.Scatter(x=_x, y=_nv, name="Same time yesterday ŷ", mode="lines", line=dict(color=theme.BLUE, width=1.8, dash="dot")))
    fig.add_trace(go.Scatter(x=[None], y=[None], name="Error e (the shaded gap)", mode="markers", marker=dict(symbol="square", size=11, color=_AMB)))
    theme.style_fig(fig, height=360, yaxis_title=f"{unit}, last {_n_} days")
if len(fig.data):
    fig.update_layout(hovermode="x unified", margin=dict(t=40))
    _legend_top(fig)
    fig.update_layout(legend_traceorder="normal")                                                                            # the legend reads in the order of the steps: the real value, the forecasts, the error
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    _key_cards(_key_items)                                                                                        # what each line is, with an example, under the graph
