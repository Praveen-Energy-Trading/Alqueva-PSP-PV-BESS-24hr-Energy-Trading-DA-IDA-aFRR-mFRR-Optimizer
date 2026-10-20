"""ML Lifecycle 8 - Decision and locked test: how the scores of the Model arena become one champion, then one look at the locked year. Only what is new: nothing of the arena is repeated."""
from __future__ import annotations

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
from maths import tex as M

ml_ui.header("Decision and locked test", "", step=6)
ml_ui.tour_bar(7)

# The colours mean one thing each, on every picture of this page (and of the Model arena):
#   an algorithm keeps its own colour; the champion is near-black; the naive forecasts are grey; a real difference is green, a gap that could be luck is grey.
MUTED, INK, CHAMP = theme.INK_MUTED, theme.INK_PRIMARY, "#2b2a28"
NEUTRAL, GREY, REAL, LUCK, T_DONE, RED = "#6b6a64", "#b9b7ae", theme.GREEN, "#9a9890", "#8791ab", theme.STATUS_CRITICAL
CARD_LINE = "#e3e1d9"
_cap = lambda h_: h_[:1].upper() + h_[1:] if h_[:1].islower() else h_                                                  # noqa: E731
_alg = lambda m_: CHAMP if m_ == "champion" else (theme.CATEGORICAL[ml_data.model_index(m_) % 8] if m_ in ml_data.MODEL_ORDER else LUCK)             # noqa: E731
_dot = lambda m_: f"<span style='color:{_alg(m_)}'>■</span>&nbsp;"                                                      # noqa: E731


def _section(title: str, hint: str = "", num: str = "") -> None:
    """A section heading: a number (the steps of the story), a bold title and, under it, one muted line on how to read the section."""
    _n = f"<span style='font-weight:400; color:{NEUTRAL};'>{num}</span>&nbsp; " if num else ""
    st.markdown(f"<div style='margin:1.7rem 0 0.45rem;'><div style='font-size:1.55rem; font-weight:700; color:{INK}; line-height:1.25;'>{_n}{title}</div>"
                + (f"<div style='font-size:0.88rem; color:{MUTED}; margin-top:0.2rem;'>{_cap(hint)}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _mini(title: str, hint: str = "") -> None:
    """The title of a chart or table inside a section."""
    st.markdown(f"<div style='margin:0.6rem 0 0.3rem;'><div style='font-size:1.02rem; font-weight:700; color:{INK};'>{title}</div>"
                + (f"<div style='font-size:0.82rem; color:{MUTED}; margin-top:0.1rem; line-height:1.35;'>{_cap(hint)}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def _line(html: str) -> None:
    """One short note under a table or chart, the key numbers in bold."""
    st.markdown(f"<div style='font-size:0.86rem; color:{theme.INK_SECONDARY}; margin:0.25rem 0 0.9rem;'>{html}</div>", unsafe_allow_html=True)


def _panel(inner: str, min_height: str = "") -> str:
    """A white rounded card for a calculation, as on the Tuning lab."""
    return (f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0.55rem 1.1rem 0.8rem; box-shadow:0 1px 2px rgba(0,0,0,0.04); "
            f"font-size:0.9rem; color:{theme.INK_SECONDARY}; {min_height}'>{inner}</div>")


def _table(headers: list, rows: list, highlight: int | None = None, first: str = "minmax(9rem, 1.5fr)", widths: str | None = None, left: tuple = (0,)) -> None:
    """A table as on the Tuning lab: muted header, thin row lines, the first column left, the others centred; one row can be shaded."""
    grid = f"display:grid; grid-template-columns:{widths or f'{first} repeat({len(headers) - 1}, minmax(0, 1fr))'}; align-items:center; column-gap:0.6rem;"
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


_legend_top = lambda fig, y=1.08: fig.update_layout(legend=dict(orientation="h", x=0, y=y, xanchor="left", yanchor="bottom", font=dict(size=12, color=theme.INK_SECONDARY)))     # noqa: E731


def _gain(ref_mae: float, mae: float) -> float:
    """The gain over a reference, from the two MAEs as shown (3 decimals), in per cent."""
    r_, m_ = round(ref_mae, 3), round(mae, 3)
    return 100 * (r_ - m_) / r_


policy = ml_ui.cached(lambda: ml_data.config("model_selection").get("policy", {}))()
decisions = ml_ui.cached(ml_data.decisions)()
locked = ml_ui.cached(ml_data.locked_results)()
lf = ml_ui.cached(lambda: ml_data.pack("locked_forecasts"))()
bench = ml_ui.cached(ml_data.benchmarks)()

targets = list(ml_data.TARGETS)
target = ml_ui.target_picker(targets=targets)
d, lk = decisions.get(target), locked.get(target)
if d is None or not policy:
    ml_ui.not_yet_run(f"the champion decision for {ml_data.label(target)}")
    st.stop()
units = ml_data.UNITS[target]
pol = {**policy, **(d.get("policy") or {})}
ref = pol["reference"]
par = d.get("parsimony") or {}
tied = d.get("tied") or []
comb = d.get("combination")
champ_txt = ml_data.champion_text(d["champion"])
chl = ml_data.champion_label(target)                                                                                         # "Champion (LightGBM)"
tn = (d.get("treatment") or {}).get("name", "standard")

refl = ml_data.model_label(ref)
if tn in ("limited_evidence", "very_limited_evidence"):                                                                     # a caution that belongs to the whole page
    st.markdown(f"<div style='margin:0.2rem 0 0.4rem;'>{_pill('Very limited evidence' if tn == 'very_limited_evidence' else 'Limited evidence', '#7a5200', '#fdf0d0')}"
                f"<span style='font-size:0.86rem; color:{theme.INK_SECONDARY}; margin-left:0.6rem;'>With {d['n_days']:,} development days only larger differences can be established, so ties are common.</span></div>", unsafe_allow_html=True)

# ============================================================================================ 2. the locked year
_section("The locked year, one look", "Scored once, on a year that nothing was tuned or chosen on", num="①")
if lk is None:
    ml_ui.not_yet_run(f"the locked test of {ml_data.label(target)}")
else:
    s, rs = lk["locked_scores"], lk.get("reference_scores")
    real = ml_ui.cached(ml_data.locked_real_only)(target)                   # a target with days the auction published no price: the numbers use the real days only
    if real:
        ch_mae, rf_mae = real["mae"], real["reference_mae"]
    else:
        ch_mae, rf_mae = s["mae"], (rs["mae"] if rs else None)
    gain_lk = _gain(rf_mae, ch_mae) if rf_mae else None
    lock_days = (pd.Timestamp(lk["window"][1]) - pd.Timestamp(lk["window"][0])).days + 1
    _chips = [f"scored once, {pd.Timestamp(lk['window'][0]):%d-%m-%Y} to {pd.Timestamp(lk['window'][1]):%d-%m-%Y}"]
    st.markdown("<div style='display:flex; flex-wrap:wrap; gap:0.4rem 0.6rem; margin:0.1rem 0 0.6rem;'>"
                + "".join(f"<span style='background:#f1f0eb; border-radius:999px; padding:0.15rem 0.75rem; font-size:0.82rem; color:{INK};'>{c_}</span>" for c_ in _chips) + "</div>", unsafe_allow_html=True)
    xl = pd.DataFrame()
    if not lf.empty and target in set(lf["target"]):
        xl = lf[lf["target"] == target].copy()
        xl["ts"] = pd.to_datetime(xl["ts_utc"], utc=True).dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    if len(xl):
        x = xl
        days = sorted(x["ts"].dt.date.unique())
        _mini("A week of the locked year", "The real price, the champion's forecast and the reference's forecast")
        pick = st.select_slider("Week to inspect", options=days[::7], value=days[::7][len(days[::7]) // 2], key="ml8_week")
        w = x[(x["ts"].dt.date >= pick) & (x["ts"].dt.date < pick + pd.Timedelta(days=7))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w["ts"], y=w["y_true"], mode="lines", name="Real price", line=dict(color=GREY, width=3)))
        fig.add_trace(go.Scatter(x=w["ts"], y=w["champion"], mode="lines", name=chl, line=dict(color=CHAMP, width=1.8)))
        fig.add_trace(go.Scatter(x=w["ts"], y=w["reference"], mode="lines", name=f"{refl} (reference)", line=dict(color=theme.BLUE, width=1.4, dash="dot")))
        theme.style_fig(fig, height=270, yaxis_title=units)
        _legend_top(fig)
        fig.update_layout(margin=dict(t=34))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if lk["loss_ratio_locked_over_dev"] < 0.8 or lk["loss_ratio_locked_over_dev"] > 1.25:
        _line(f"The locked-year loss is {lk['loss_ratio_locked_over_dev']:.2f} times the development loss. That compares two different price regimes, so it is not a measure of skill: the gain over the reference on the same hours is.")
    if lk.get("synthetic_fill"):
        sf = lk["synthetic_fill"]
        _line(f"On {len(sf['days'])} days of the locked year the auction published no price (an empty result or a failed download). They are left out of every number above; "
              "the frozen test data holds the day-ahead price on those days, and the scores that include them are in the locked-test report.")
    if lk.get("members"):
        with st.expander("The other algorithms scored in the same locked run (reported, not used for any choice)"):
            mm = [(k_, v_) for k_, v_ in lk["members"].items()]
            cols_ = [c_ for c_ in ("mae", "rmse", "coverage") if all(c_ in v_ for _, v_ in mm)]
            _table(["Forecaster"] + [c_.upper() if c_ != "coverage" else "Coverage" for c_ in cols_],
                   [[f"{_dot(str(k_).replace('_tuned', ''))}{ml_data.model_label(str(k_).replace('_tuned', ''))}"] + [f"{v_[c_]:.3f}" for c_ in cols_] for k_, v_ in mm])

# ============================================================================================ 3. is machine learning worth having
bm = bench.get(target)
_section("Is machine learning worth having?", "The champion against forecasts with no model at all, on the locked year", num="②")
NAMES = {"champion": chl, "reference": f"{refl} (reference)", "base_price": "Day-ahead price, no spread", "naive_seasonal": "Seasonal naive", "naive_day": "Latest day available", "naive_week": "Last week"}
WHAT = {"champion": "The chosen forecast", "reference": "The simplest algorithm",
        "base_price": "The intraday price equals the day-ahead price", "naive_seasonal": "Tue to Fri: the day before. Sat to Mon: the same weekday last week",
        "naive_day": "The latest day's price known at forecast time", "naive_week": "The price of the same day last week"}
if not bm or not bm.get("locked"):
    ml_ui.not_yet_run(f"the naive benchmark of {ml_data.label(target)}")
else:
    b = bm["locked"]
    sc = b["scores"]
    scd = (bm.get("development") or {}).get("scores", {})
    order = ["champion", "reference"] + sorted([n_ for n_ in sc if n_ not in ("champion", "reference")], key=lambda n_: sc[n_]["mae"])
    order = [n_ for n_ in order if n_ in sc]
    chm = round(sc["champion"]["mae"], 3)
    gains = {n_: _gain(sc[n_]["mae"], sc["champion"]["mae"]) for n_ in order if n_ != "champion"}
    txt = [f"<b>{chm:.3f}</b>"] + [f"{round(sc[n_]['mae'], 3):.3f}   champion {abs(gains[n_]):.1f}% " + ("lower" if gains[n_] > 0 else "higher") for n_ in order if n_ != "champion"]
    hov = []
    for n_ in order:
        h_ = f"<b>{NAMES.get(n_, n_)}</b><br>{WHAT.get(n_, '')}<br>Locked year MAE {round(sc[n_]['mae'], 3):.3f}"
        if n_ in scd and "champion" in scd:
            h_ += f"<br>Development window MAE {round(scd[n_]['mae'], 3):.3f}" + ("" if n_ == "champion" else f", champion {abs(_gain(scd[n_]['mae'], scd['champion']['mae'])):.1f}% " + ("lower" if _gain(scd[n_]["mae"], scd["champion"]["mae"]) > 0 else "higher"))
        hov.append(h_)
    fd = go.Figure(go.Bar(y=[NAMES.get(n_, n_) for n_ in order], x=[sc[n_]["mae"] for n_ in order], orientation="h",
                          marker_color=[CHAMP if n_ == "champion" else theme.BLUE if n_ == "reference" else LUCK for n_ in order], text=txt, textposition="outside", cliponaxis=False,
                          textfont=dict(size=11, color=theme.INK_SECONDARY), hovertext=hov, hoverinfo="text",
                          hoverlabel=dict(align="left", bgcolor="white", bordercolor="#e3e1d9", font=dict(size=12, color=INK))))
    theme.style_fig(fd, height=int(46 * len(order)) + 60, xaxis_title=f"MAE ({units}), locked year, {round(b['hours'] / 24):,} days", legend=False)
    fd.update_yaxes(autorange="reversed", showgrid=False, zeroline=False, fixedrange=True)
    fd.update_xaxes(range=[0, max(sc[n_]["mae"] for n_ in order) * 2.0], showgrid=False, zeroline=False)
    fd.update_layout(margin=dict(t=6, b=44, l=190, r=10), bargap=0.4)
    st.plotly_chart(fd, width="stretch", config={"displayModeBar": False})
    naive_ = [n_ for n_ in order if n_ not in ("champion", "reference")]
    LEVEL = target in ("da_price", "afrr_cap_price", "afrr_dn_price")                                                       # a price level forecast: the naive curve can be drawn from the real prices
    RULE = {"naive_day": "Copies the most recent day whose prices are known, hour by hour", "naive_week": "Every day: copies the same weekday of last week (7 days back)", "base_price": "The day-ahead price, no spread",
            "naive_seasonal": "Tuesday to Friday: copies yesterday. Saturday, Sunday, Monday: copies last week. A mix of the two forecasts above"}
    _mini("How each naive forecast is made", "No model, nothing to fit or tune: only prices known when the forecast is made")
    D_ = pd.Timestamp(lk["window"][1]) if lk is not None else pd.Timestamp(b["window"][1])                                    # the examples: the last Wednesday and the last Sunday of the locked year
    minlag = {"xbid_price": 2, "mfrr_price": 3}.get(target, 1)                                                              # the days the target's own prices take to become known
    wed_, sun_ = (D_ - pd.Timedelta(days=(D_.weekday() - 2) % 7)), (D_ - pd.Timedelta(days=(D_.weekday() - 6) % 7))         # two weekdays on which the seasonal naive forecast acts differently

    def _lag(n_, day_):
        return max({"naive_day": 1, "naive_week": 7, "naive_seasonal": 7 if day_.weekday() in (5, 6, 0) else 1}[n_], minlag)

    def _ex(n_, day_):                                                                                                      # one plain sentence: which day is forecast, which day is copied
        if n_ == "base_price":
            return f"{day_:%A} {day_:%d-%m-%Y} uses the day-ahead price of {day_:%d-%m-%Y}"
        s_ = day_ - pd.Timedelta(days=_lag(n_, day_))
        return f"{day_:%A} {day_:%d-%m-%Y} uses {s_:%A} {s_:%d-%m-%Y}"
    _table(["Forecast", "Rule", f"Example: a {wed_:%A}", f"Example: a {sun_:%A}"], [[NAMES.get(n_, n_), RULE.get(n_, WHAT.get(n_, "")), _ex(n_, wed_), _ex(n_, sun_)] for n_ in naive_],
           widths="minmax(9rem, 1fr) minmax(14rem, 2.2fr) minmax(11rem, 1.3fr) minmax(11rem, 1.3fr)", left=(0, 1, 2, 3))
    lag_of = {n_: _lag(n_, D_) for n_ in ("naive_day", "naive_week", "naive_seasonal")}                                      # used by the drop-down below
    _stages = [(k_, t_) for k_, t_ in (("development", "Development window"), ("locked", "Locked year")) if bm.get(k_)]                  # the two windows, in time order
    _mini("The numbers behind the chart", "Mean absolute error and root mean squared error of every forecast on each window, and the champion's gain over it")
    _heads = ["Forecast"] + [h_ for _, t_ in _stages for h_ in (f"{t_.split()[0]}: MAE", "RMSE", "Champion's gain")]
    _trs = []
    for n_ in order:
        row_ = [f"{_dot(n_) if n_ in ('champion', 'reference') else ''}{NAMES.get(n_, n_)}"]
        for k_, _t in _stages:
            sk_ = bm[k_]["scores"]
            row_ += [f"{round(sk_[n_]['mae'], 3):.3f}", f"{sk_[n_]['rmse']:.3f}", "–" if n_ == "champion" else f"{_gain(sk_[n_]['mae'], sk_['champion']['mae']):.1f}%"] if n_ in sk_ else ["–", "–", "–"]
        _trs.append(row_)
    _table(_heads, _trs, highlight=0, widths="minmax(10rem, 1.4fr) " + "minmax(3.6rem, 0.8fr) minmax(3.6rem, 0.8fr) minmax(4.6rem, 0.9fr) " * len(_stages))
    if LEVEL and not lf.empty and target in set(lf["target"]):                                                             # how a naive forecast gets its value: a real price, copied
        with st.expander("How a naive forecast predicts one hour"):
            _lfx = lf[lf["target"] == target].copy()
            _lfx["ts"] = pd.to_datetime(_lfx["ts_utc"], utc=True).dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
            _lfx = _lfx.dropna(subset=["y_true"]).assign(d=lambda g_: g_["ts"].dt.normalize(), h=lambda g_: g_["ts"].dt.hour)
            _ysh = _lfx.groupby(["d", "h"])["y_true"].mean()
            _cA, _cB = st.columns(2)
            _cmp = _cA.selectbox("Forecast", [n_ for n_ in naive_ if n_ in lag_of], format_func=lambda n_: NAMES.get(n_, n_), key="ml8_calc_cmp")
            _dys = sorted(_lfx["d"].unique())
            _day = pd.Timestamp(_cB.selectbox("Delivery day", _dys[::-1], format_func=lambda d_: pd.Timestamp(d_).strftime("%d-%m-%Y"), key="ml8_calc_day"))
            _lg = max(7 if _cmp == "naive_week" or (_cmp == "naive_seasonal" and _day.weekday() in (5, 6, 0)) else 1, minlag)
            _src = _day - pd.Timedelta(days=_lg)
            st.markdown(f"<div style='margin:0.4rem 0;'>" + M(r"\hat{y}_{d,h}=y_{d-" + str(_lg) + r",\,h}", display=False, size=1.1)
                        + f" &nbsp; the forecast for {_day:%d-%m-%Y} ({_day:%A}) is the real price of <b>{_src:%d-%m-%Y}</b> ({_src:%A}), hour by hour</div>", unsafe_allow_html=True)
            _rows = []
            for h_ in (2, 6, 10, 14, 18, 22):
                yv_, ys_ = _ysh.get((_day, h_), np.nan), _ysh.get((_src, h_), np.nan)
                if not (np.isnan(yv_) or np.isnan(ys_)):
                    _rows.append([f"{h_:02d}:00", f"{_src:%d-%m-%Y} {h_:02d}:00", f"{ys_:.2f}", f"{ys_:.2f}", f"{yv_:.2f}"])
            _table(["Hour", "Copied from", "Real price there", "Forecast ŷ", f"Real price on {_day:%d-%m-%Y}"], _rows, widths="minmax(4rem, 0.8fr) minmax(10rem, 1.6fr) " + "minmax(6rem, 1fr) " * 3, left=(0, 1))
    if b.get("synthetic_days"):
        _line(f"The locked year has {b['synthetic_days']} days on which the auction published no price. They are left out for every forecast, because a forecast built on the day-ahead price would be flattered by them.")

    if lk is not None and not lf.empty and target in set(lf["target"]):                                                    # month by month: the champion, the decision tree and the strongest naive forecast
        xa = lf[lf["target"] == target].copy()
        xa["ts"] = pd.to_datetime(xa["ts_utc"], utc=True).dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
        xa = xa.dropna(subset=["y_true", "champion", "reference"]).assign(d=lambda g_: g_["ts"].dt.normalize(), h=lambda g_: g_["ts"].dt.hour)
        xa["month"] = xa["ts"].dt.to_period("M")
        strong = min(naive_, key=lambda n_: sc[n_]["mae"]) if naive_ else None
        series_ = {chl: (xa["champion"] - xa["y_true"]).abs(), refl: (xa["reference"] - xa["y_true"]).abs()}
        nm_ = None
        if LEVEL and strong in ("naive_day", "naive_week", "naive_seasonal"):                                              # a price level: the naive curve is the real price of an earlier day, same hour
            ysh = xa.groupby(["d", "h"])["y_true"].mean()
            lag_s = xa["d"].map(lambda d_: 7 if strong == "naive_week" or (strong == "naive_seasonal" and d_.weekday() in (5, 6, 0)) else 1)
            src = pd.MultiIndex.from_arrays([xa["d"] - pd.to_timedelta(lag_s, unit="D"), xa["h"]])
            ysrc = pd.Series(ysh.reindex(src).to_numpy(), index=xa.index)
            series_[NAMES.get(strong, strong)] = (ysrc - xa["y_true"]).abs()
            nm_ = NAMES.get(strong, strong)
        mon_ = pd.DataFrame({k_: v_.groupby(xa["month"]).mean() for k_, v_ in series_.items()})
        _mini("Month by month", "Mean absolute error of the champion, the " + refl.lower() + (f" and the {nm_.lower()} on every month of the locked year" if nm_ else " on every month of the locked year"))
        cols_ = {chl: CHAMP, refl: theme.BLUE, nm_: LUCK}
        fmn = go.Figure()
        for k_ in mon_.columns:
            fmn.add_trace(go.Bar(x=[m_.strftime("%b %Y") for m_ in mon_.index], y=mon_[k_], name=k_, marker_color=cols_[k_], text=[f"{v_:.1f}" for v_ in mon_[k_]], textposition="outside",
                                 textfont=dict(size=9, color=theme.INK_SECONDARY), cliponaxis=False, hovertemplate="%{x}<br>" + k_ + ": %{y:.3f}<extra></extra>"))
        theme.style_fig(fmn, height=300, yaxis_title=f"MAE ({units})", barmode="group")
        _legend_top(fmn)
        fmn.update_layout(margin=dict(t=34), bargap=0.25, bargroupgap=0.05)
        st.plotly_chart(fmn, width="stretch", config={"displayModeBar": False})
