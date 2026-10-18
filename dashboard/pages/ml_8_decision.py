"""ML Lifecycle 8 - Decision and locked test: a written selection rule, applied by code, and one honest test on locked data. Told as a story in numbered steps."""
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

ml_ui.header("Decision and locked test", "The champion is chosen by a rule written down before the results existed, applied by code, and then "
             "scored once on a year that nothing was tuned or chosen on.", step=7)
ml_ui.tour_bar(8)

# The colours mean one thing each, on every picture of this page (and of the Model arena):
#   an algorithm keeps its own colour; the champion is near-black; the naive forecasts are grey; a real difference is green, a gap that could be luck is grey.
MUTED, INK, CHAMP = theme.INK_MUTED, theme.INK_PRIMARY, "#2b2a28"
NEUTRAL, GREY, REAL, LUCK, T_DONE, RED = "#6b6a64", "#b9b7ae", theme.GREEN, "#9a9890", "#8791ab", theme.STATUS_CRITICAL
CARD_LINE = "#e3e1d9"
_cap = lambda h_: h_[:1].upper() + h_[1:] if h_[:1].islower() else h_                                                  # noqa: E731
_alg = lambda m_: CHAMP if m_ == "champion" else (theme.CATEGORICAL[ml_data.model_index(m_) % 8] if m_ in ml_data.MODEL_ORDER else LUCK)             # noqa: E731
_dot = lambda m_: f"<span style='color:{_alg(m_)}'>■</span>&nbsp;"                                                      # noqa: E731
_p = lambda v_: "–" if v_ is None or (isinstance(v_, float) and np.isnan(v_)) else f"{v_:.2g}"                          # noqa: E731


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


def _table(headers: list, rows: list, highlight: int | None = None, first: str = "minmax(9rem, 1.5fr)") -> None:
    """A table as on the Tuning lab: muted header, thin row lines, the first column left, the others centred; one row can be shaded."""
    grid = f"display:grid; grid-template-columns:{first} repeat({len(headers) - 1}, minmax(0, 1fr)); align-items:center; column-gap:0.6rem;"
    head = (f"<div style='{grid} padding:0.6rem 0 0.4rem; border-bottom:1px solid {CARD_LINE}; font-size:0.72rem; color:{MUTED}; font-weight:600; line-height:1.2;'>"
            + "".join(f"<span style='{'' if j == 0 else 'text-align:center;'}'>{h}</span>" for j, h in enumerate(headers)) + "</div>")
    body = ""
    for i, row in enumerate(rows):
        hl = i == highlight
        body += (f"<div style='{grid} min-height:2.5rem; font-size:0.86rem; "
                 + ("border-top:1px solid #e3e1d9; background:#f8f7f3; font-weight:700;" if hl else "border-bottom:1px solid #eeede8;") + "'>"
                 + "".join(f"<span style='{'' if j == 0 else 'text-align:center;'} color:{INK if str(c) != '–' else MUTED};'>{c}</span>" for j, c in enumerate(row)) + "</div>")
    st.markdown(f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0 1.1rem 0.1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>{head}{body}</div>", unsafe_allow_html=True)


def _pill(text: str, color: str, bg: str) -> str:
    return f"<span style='background:{bg}; color:{color}; border-radius:999px; padding:0.1rem 0.6rem; font-size:0.78rem; font-weight:600; white-space:nowrap;'>{text}</span>"


_real_pill = lambda: _pill("Real", "#006100", "#e3f1e3")                                                              # noqa: E731
_luck_pill = lambda: _pill("Could be luck", "#5a5953", "#ecebe6")                                                     # noqa: E731
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
tn = (d.get("treatment") or {}).get("name", "standard")

# ============================================================================================ 1. the rule, the windows and the comparison, drawn
SH, AN = [], []
ROW, X_END = 34.0, 1010
px_x = 1150.0 / X_END


def _shp(x0, x1, y0, y1, color, opacity=1.0, r_px=6.0, line=None):
    """A rounded bar as a path; x in the units of the figure, y in rows."""
    rx, ry = min(r_px / px_x, (x1 - x0) / 2), min(r_px / ROW, (y1 - y0) / 2)
    SH.append(dict(type="path", fillcolor=color, opacity=opacity, line=(line or dict(width=0)),
                   path=(f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
                         f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z")))


def _fcard(y0, y1, r_px=10.0):
    rx, ry = r_px / px_x, r_px / ROW
    x0, x1 = -12, X_END + 12
    SH.append(dict(type="path", fillcolor="#ffffff", line=dict(color=CARD_LINE, width=1), layer="below",
                   path=(f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
                         f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z")))


def _fhead(y, num, title, sub_=""):
    AN.append(dict(x=6, y=y, xanchor="left", yanchor="middle", showarrow=False, align="left", font=dict(size=14, color=INK),
                   text=f"<b style='font-size:17px;color:{NEUTRAL}'>{num}</b>&nbsp; <b>{title}</b>" + (f"<br><span style='font-size:12px;color:{MUTED}'>{_cap(sub_)}</span>" if sub_ else "")))


def _fnote(x, y, text, size=12, color=INK, anchor="center", **kw):
    AN.append(dict(x=x, y=y, xanchor=anchor, showarrow=False, text=text, font=dict(size=size, color=color), **kw))


def _farrow(x0, y0, x1, y1, color=MUTED):
    AN.append(dict(x=x1, y=y1, ax=x0, ay=y0, axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.6, arrowcolor=color, text=""))


fig_a = go.Figure()
fig_a.add_trace(go.Scatter(x=[0, X_END], y=[0, 0], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
# ---- card 1: the rule, with the path of this target
y1 = -0.6
_fcard(y1, y1 + 8.0)
_fhead(y1 + 1.0, "①", "The rule, written before the results", f"Applied by code to {ml_data.label(target)}. The path this target took is highlighted; a gate it never reached is grey")
refl = ml_data.model_label(ref)
evid_ok = d.get("status") == "chosen"
par_tested, par_passed = bool(par.get("tested")), bool(par.get("passed"))
gates = [
    ("Evidence", [f"At least {pol['min_days']} days", f"Coverage at least {pol['min_coverage']:.0%}"], f"{d['n_days']:,} days, {d['coverage']:.0%}", evid_ok),
    ("Rank", ["Mean absolute error", "On the same hours"], f"Best: {ml_data.model_label(d['best'])}", evid_ok),
    ("Parsimony", [f"The best must beat the", f"{refl} at {pol['alpha']:.0%}"],
     (f"p = {_p(par.get('p_value'))}, passed" if par_tested and par_passed else f"Not better: {refl} wins" if par_tested else f"The best is the {refl}"), evid_ok),
    ("Ties", ["Holm correction: one left", "wins; several: their " + str(pol["average_method"])],
     (f"{len(tied)} tied" + (", their " + str(pol["average_method"]) if len(tied) > 1 else ", it wins")) if (par_passed or not par_tested) and evid_ok else "Not reached", evid_ok and (par_passed or not par_tested)),
]
bx0, bw, gap_x, by0, bh = 0.0, 228.0, 32.0, y1 + 2.0, 3.6
for i, (title, lines, this, on) in enumerate(gates):
    x0 = bx0 + i * (bw + gap_x)
    _shp(x0, x0 + bw, by0, by0 + bh, REAL if on else NEUTRAL, 0.09 if on else 0.05, r_px=8)
    _shp(x0, x0 + 5, by0, by0 + bh, REAL if on else LUCK, 1.0, r_px=2)
    _fnote(x0 + 16, by0 + 0.55, f"<b>{i + 1}. {title}</b>", size=13, color=INK if on else MUTED, anchor="left")
    for j, ln in enumerate(lines):
        _fnote(x0 + 16, by0 + 1.3 + 0.5 * j, ln, size=11, color=theme.INK_SECONDARY if on else MUTED, anchor="left")
    _fnote(x0 + 16, by0 + 3.0, f"<b>{this}</b>", size=11, color=(REAL if on else MUTED), anchor="left")
    if i < len(gates) - 1:
        _farrow(x0 + bw + 2, by0 + bh / 2, x0 + bw + gap_x - 2, by0 + bh / 2, color=MUTED)
cy1 = by0 + bh + 1.2
_shp(0, X_END - 20, cy1 - 0.45, cy1 + 0.45, CHAMP, 0.92, r_px=8)
_fnote((X_END - 20) / 2, cy1, f"<b>Champion: {champ_txt}</b>", size=13, color="white")
if d.get("policy_digest"):
    _fnote(X_END - 20, cy1 + 0.95, f"Rule fingerprint {str(d['policy_digest'])[:12]}", size=10, color=MUTED, anchor="right")
# ---- card 2: the two windows
y2 = y1 + 8.45
_fcard(y2, y2 + 4.9)
_fhead(y2 + 1.0, "②", "Two windows, one honest test", "The choice is made on the development window. The locked year is scored once, and nothing was tuned or chosen on it")
dev = (d.get("source") or {}).get("window") or [None, None]
lkw = (lk or {}).get("window") or [None, None]
dev_days, lock_days = int(d["n_days"]), 365
tot_w = 960.0
wd = max(300.0, tot_w * dev_days / (dev_days + lock_days))
wl = tot_w - wd
cyt = y2 + 2.9
_shp(0, wd - 3, cyt - 0.5, cyt + 0.5, theme.BLUE, 0.62, r_px=6)
_shp(wd + 3, wd + wl, cyt - 0.5, cyt + 0.5, RED, 0.35, r_px=6)
_fnote(10, cyt, f"<b>Development window</b>: {dev_days:,} days", color=INK, anchor="left")
_fnote(wd + 13, cyt, f"<b>Locked year</b>: {lock_days} days", color=INK, anchor="left")
_fnote(10, cyt + 0.95, (f"{pd.Timestamp(dev[0]):%d-%m-%Y} to {pd.Timestamp(dev[1]):%d-%m-%Y}" if dev[0] else ""), size=11, color=MUTED, anchor="left")
_fnote(wd + 13, cyt + 0.95, (f"{pd.Timestamp(lkw[0]):%d-%m-%Y} to {pd.Timestamp(lkw[1]):%d-%m-%Y}" if lkw[0] else ""), size=11, color=MUTED, anchor="left")
_fnote(10, cyt + 1.55, "Ranking, tests and the champion are decided here", size=11, color=theme.INK_SECONDARY, anchor="left")
_fnote(wd + 13, cyt + 1.55, "Scored once, frozen by digest", size=11, color=RED, anchor="left")
SH.append(dict(type="line", x0=wd, x1=wd, y0=cyt - 0.75, y1=cyt + 0.75, line=dict(color="#a8a69d", width=1.4, dash="dot")))
# ---- card 3: what the champion is compared with
y3 = y2 + 5.35
_fcard(y3, y3 + 4.6)
_fhead(y3 + 1.0, "③", "What the champion is compared with", "All on exactly the same hours")
cmp_items = [(CHAMP, "Champion", "The chosen forecast"), (theme.BLUE, "Decision tree", "The simplest algorithm: what the extra ones buy"),
             (LUCK, "Naive forecasts", "No model at all: is machine learning worth having?")]
iw = 310.0
for i, (col, ttl, sub_) in enumerate(cmp_items):
    x0 = i * (iw + 25)
    _shp(x0, x0 + iw, y3 + 2.2, y3 + 4.0, col, 0.10, r_px=8)
    _shp(x0, x0 + 5, y3 + 2.2, y3 + 4.0, col, 1.0, r_px=2)
    _fnote(x0 + 16, y3 + 2.85, f"<b>{ttl}</b>", size=13, anchor="left")
    _fnote(x0 + 16, y3 + 3.5, sub_, size=10, color=theme.INK_SECONDARY, anchor="left")
end = y3 + 4.6
fig_a.layout.shapes = ()
theme.style_fig(fig_a, height=int((end + 0.9) * ROW + 12), xaxis_title=None, legend=False)
fig_a.update_yaxes(autorange=False, range=[end + 0.15, -0.75], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False, linewidth=0)
fig_a.update_xaxes(range=[-15, X_END + 15], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False)
fig_a.update_layout(margin=dict(t=2, b=2, l=10, r=10), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
fig_a.layout.shapes = SH
fig_a.layout.annotations = AN
st.plotly_chart(fig_a, width="stretch", config={"displayModeBar": False})

# ============================================================================================ 2. the decision
_section(f"The decision for {ml_data.label(target)}", "The evidence behind the choice, from the development window", num="④")
if tn in ("limited_evidence", "very_limited_evidence"):
    lab = d["treatment"]["label"].replace("first test day", "first validation day")
    st.markdown(f"<div style='margin:0 0 0.6rem;'>{_pill('Very limited evidence' if tn == 'very_limited_evidence' else 'Limited evidence', '#7a5200', '#fdf0d0')}"
                f"<span style='font-size:0.86rem; color:{theme.INK_SECONDARY}; margin-left:0.6rem;'>{_cap(lab)}. With {d['n_days']:,} development days only larger differences can be established, so ties are common.</span></div>", unsafe_allow_html=True)
rk = pd.DataFrame(d["ranking"])
vb = pd.DataFrame(d["versus_best"]) if d.get("versus_best") else pd.DataFrame()
_lc, _rc = st.columns([3, 2], gap="medium")
with _lc:
    _mini("The ranking", f"Mean daily error on the development window, {units}. The dotted line is the best")
    lad = rk[["model", "mean_loss"]].sort_values("mean_loss").reset_index(drop=True)
    notworse = set(tied)
    fl = go.Figure()
    k_tie = int(sum(1 for m_ in lad["model"] if m_ in notworse))
    if k_tie > 1:
        fl.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=-0.5, y1=k_tie - 0.5, fillcolor="rgba(154,152,144,0.14)", line_width=0, layer="below")
    best_l = float(lad["mean_loss"].iloc[0])
    fl.add_shape(type="line", x0=best_l, x1=best_l, y0=-0.5, y1=len(lad) - 0.5, line=dict(color=MUTED, width=1, dash="dot"))
    pv = {r_["model"]: r_ for _, r_ in vb.iterrows()} if len(vb) else {}
    fl.add_trace(go.Scatter(x=lad["mean_loss"], y=[ml_data.model_label(m_) for m_ in lad["model"]], mode="markers+text", showlegend=False,
                            marker=dict(size=15, color=[_alg(m_) for m_ in lad["model"]], line=dict(color="white", width=1.5)),
                            text=[f"{v_:.3f}" + ("" if m_ not in pv else f"   {'Cannot be told apart' if m_ in notworse else 'Significantly worse'} (p {_p(float(pv[m_]['p_holm']))})") for m_, v_ in zip(lad["model"], lad["mean_loss"])],
                            textposition="middle right", textfont=dict(size=11, color=theme.INK_SECONDARY), cliponaxis=False, hoverinfo="skip"))
    theme.style_fig(fl, height=int(46 * len(lad)) + 70, xaxis_title=f"Mean daily error ({units})", legend=False)
    fl.update_yaxes(autorange="reversed", showgrid=False, zeroline=False, fixedrange=True)
    spread = float(lad["mean_loss"].max()) - best_l
    fl.update_xaxes(range=[best_l - 0.15 * spread - 0.3, float(lad["mean_loss"].max()) + 0.9 * spread + 1.5], showgrid=False, zeroline=False)
    fl.update_layout(margin=dict(t=6, b=44, l=120, r=10))
    st.plotly_chart(fl, width="stretch", config={"displayModeBar": False})
with _rc:
    st.markdown(_panel(
        f"<div style='font-size:0.74rem; font-weight:600; color:{MUTED};'>Champion</div>"
        f"<div style='font-size:1.45rem; font-weight:800; color:{INK}; line-height:1.25; margin-bottom:0.4rem;'>{champ_txt}</div>"
        f"<div style='line-height:1.6; font-size:0.9rem;'>"
        f"<div><b>Development days:</b> {d['n_days']:,}</div><div><b>Best by mean loss:</b> {ml_data.model_label(d['best'])}</div>"
        f"<div><b>Against the {refl}:</b> " + (f"p = {_p(par.get('p_value'))}, " + ('passed' if par_passed else 'not better') if par_tested else "it is the best itself") + "</div>"
        f"<div><b>Cannot be told apart from the best:</b> " + (", ".join(ml_data.model_label(m_) for m_ in tied if m_ != d["best"]) or "none") + "</div></div>"), unsafe_allow_html=True)
if len(vb):
    _mini("Each algorithm against the best", f"Against {ml_data.model_label(d['best'])}, with the Holm correction over these comparisons")
    _table(["Algorithm", "Mean daily error", "Error of the best", "Difference", "DM statistic", "p-value", "p-value (Holm)", "Verdict"],
           [[f"{_dot(r_['model'])}{ml_data.model_label(r_['model'])}", f"{float(r_['mean_loss']):.3f}", f"{float(r_['best_loss']):.3f}",
             f"{round(float(r_['mean_loss']), 3) - round(float(r_['best_loss']), 3):+.3f}", f"{float(r_['statistic']):.2f}", _p(float(r_["p_value"])), _p(float(r_["p_holm"])),
             _pill("Real: worse than the best", "#006100", "#e3f1e3") if r_["distinguishable_from_best"] else _pill("Cannot be told apart", "#5a5953", "#ecebe6")] for _, r_ in vb.iterrows()])
if comb:
    _mini("Calculation for the tie", "Several algorithms cannot be told apart from the best, so the champion is their average, if the average is not worse")
    st.markdown(_panel(
        f"<div style='display:grid; grid-template-columns:8.5rem 1fr; row-gap:0.4rem; align-items:baseline;'>"
        f"<span style='font-weight:600; color:{INK};'>① The tied</span><span>{', '.join(ml_data.model_label(m_) for m_ in comb['members'])}</span>"
        f"<span style='font-weight:600; color:{INK};'>② Their {comb['method']}</span><span>against the best single algorithm: mean loss difference <b style='color:{INK};'>{comb['mean_diff']:+.3f}</b>, DM statistic <b style='color:{INK};'>{comb['statistic']:.2f}</b></span>"
        f"<span style='font-weight:600; color:{INK};'>③ Chance</span><span>p = <b style='color:{INK};'>{_p(comb['p_value'])}</b>: {'not significantly worse' if comb['accepted'] else 'significantly worse'}</span></div>"
        f"<div style='margin-top:0.6rem; font-size:1.1rem; font-weight:800; color:{REAL if comb['accepted'] else LUCK};'>"
        + ("The average is the champion" if comb["accepted"] else "The best single algorithm is the champion") + "</div>"), unsafe_allow_html=True)
with st.expander("The rule applied by code, in its own words"):
    st.markdown(_panel("".join(f"<div style='padding:0.25rem 0; border-bottom:1px solid #eeede8;'><b style='color:{INK};'>{i}.</b> {ml_data.humanize(r_)}</div>" for i, r_ in enumerate(d["reasons"], 1))), unsafe_allow_html=True)

# ============================================================================================ 3. the locked year
_section("The one look at the locked year", "Scored once, on a year that nothing was tuned or chosen on", num="⑤")
if lk is None:
    ml_ui.not_yet_run(f"the locked test of {ml_data.label(target)}")
else:
    s, rs = lk["locked_scores"], lk.get("reference_scores")
    real = ml_ui.cached(ml_data.locked_real_only)(target)                   # a target with days the auction published no price: the numbers use the real days only
    if real:
        ch_mae, rf_mae, p_lk, stat_lk = real["mae"], real["reference_mae"], real["p_value"], None
    else:
        ch_mae, rf_mae = s["mae"], (rs["mae"] if rs else None)
        dmr = lk.get("dm_vs_reference") or {}
        p_lk, stat_lk = dmr.get("p_value"), dmr.get("statistic")
    gain_lk = _gain(rf_mae, ch_mae) if rf_mae else None
    _lc, _rc = st.columns([5, 3], gap="medium")
    with _lc:
        _mini("Mean absolute error on the locked year", f"{ml_data.label(target)}, {units}" + (", real days only" if real else ""))
        fb = go.Figure(go.Bar(x=[f"{refl} (reference)", "Champion"] if rf_mae else ["Champion"], y=[rf_mae, ch_mae] if rf_mae else [ch_mae], marker_color=[theme.BLUE, CHAMP] if rf_mae else [CHAMP],
                              text=[f"{round(v_, 3):.3f}" for v_ in ([rf_mae, ch_mae] if rf_mae else [ch_mae])], textposition="outside", cliponaxis=False,
                              textfont=dict(size=12, color=theme.INK_SECONDARY), hoverinfo="skip"))
        theme.style_fig(fb, height=280, yaxis_title=f"MAE ({units})", legend=False)
        fb.update_yaxes(range=[0, float(max(rf_mae or 0, ch_mae)) * 1.18])
        fb.update_layout(margin=dict(t=24), bargap=0.6)
        st.plotly_chart(fb, width="stretch", config={"displayModeBar": False})
    with _rc:
        if rf_mae:
            _hd = lambda txt_: f"<div style='font-size:0.74rem; font-weight:600; color:{MUTED}; margin-top:0.55rem;'>{txt_}</div>"                    # noqa: E731
            st.markdown(_panel(
                f"<div style='font-size:0.74rem; font-weight:600; color:{MUTED}; padding:0.2rem 0;'>① The gain over the reference</div>"
                f"<div>({round(rf_mae, 3):.3f} − {round(ch_mae, 3):.3f}) ÷ {round(rf_mae, 3):.3f} = <b style='font-size:1.25rem; color:{INK};'>{gain_lk:.1f}%</b></div>"
                + _hd("② The test against the reference")
                + (f"<div>DM statistic <b style='color:{INK};'>{stat_lk:.2f}</b>, p = <b style='color:{INK};'>{_p(p_lk)}</b></div>" if stat_lk is not None else f"<div>p = <b style='color:{INK};'>{_p(p_lk)}</b></div>")
                + f"<div style='margin-top:0.7rem; font-size:1.2rem; font-weight:800; color:{REAL if (p_lk is not None and p_lk < 0.05 and gain_lk > 0) else LUCK};'>"
                + ("Real, not luck" if (p_lk is not None and p_lk < 0.05 and gain_lk > 0) else "Not distinguishable from luck") + "</div>"), unsafe_allow_html=True)
    _mini("What was frozen", "The locked test is scored once; the frozen data and the code revision are recorded")
    _table(["Window", "Days", "Lock status", "Code revision", "Data digest", "Locked loss ÷ development loss"],
           [[f"{pd.Timestamp(lk['window'][0]):%d-%m-%Y} to {pd.Timestamp(lk['window'][1]):%d-%m-%Y}", f"{lock_days}",
             _pill(str(lk.get("lock_status")), "#006100", "#e3f1e3") if lk.get("lock_status") == "verified" else _pill(str(lk.get("lock_status")), "#5a5953", "#ecebe6"),
             f"{lk['code']}", f"{str(lk.get('data_digest', ''))[:10]}", f"{lk['loss_ratio_locked_over_dev']:.2f}"]], first="minmax(11rem, 1.6fr)")
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
    if not lf.empty and target in set(lf["target"]):
        x = lf[lf["target"] == target].copy()
        x["ts"] = pd.to_datetime(x["ts_utc"], utc=True).dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
        days = sorted(x["ts"].dt.date.unique())
        _mini("A week of the locked year", "The real price, the champion's forecast and the reference's forecast. The models were refitted every 30 days on past data only")
        pick = st.select_slider("Week to inspect", options=days[::7], value=days[::7][len(days[::7]) // 2], key="ml8_week")
        w = x[(x["ts"].dt.date >= pick) & (x["ts"].dt.date < pick + pd.Timedelta(days=7))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w["ts"], y=w["y_true"], mode="lines", name="Real price", line=dict(color=GREY, width=3)))
        fig.add_trace(go.Scatter(x=w["ts"], y=w["champion"], mode="lines", name="Champion", line=dict(color=CHAMP, width=1.8)))
        fig.add_trace(go.Scatter(x=w["ts"], y=w["reference"], mode="lines", name=f"{refl} (reference)", line=dict(color=theme.BLUE, width=1.4, dash="dot")))
        theme.style_fig(fig, height=320, yaxis_title=units)
        _legend_top(fig)
        fig.update_layout(margin=dict(t=34))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ============================================================================================ 4. against a naive forecast
bm = bench.get(target)
_section("Against a naive forecast", "A decision tree shows what the extra algorithms buy. A naive forecast shows whether machine learning is worth having at all", num="⑥")
NAMES = {"champion": "Champion", "reference": f"{refl} (reference)", "base_price": "Day-ahead price, no spread", "naive_seasonal": "Seasonal naive", "naive_day": "Latest day available", "naive_week": "Last week"}
WHAT = {"champion": "The chosen forecast", "reference": "The simplest algorithm",
        "base_price": "The intraday price equals the day-ahead price", "naive_seasonal": "Latest day's price (Tue to Fri), last week's (Sat to Mon)",
        "naive_day": "The latest day's price known at forecast time", "naive_week": "The price of the same day last week"}
if not bm:
    ml_ui.not_yet_run(f"the naive benchmark of {ml_data.label(target)}")
else:
    stages = [(k_, t_) for k_, t_ in (("development", "Development window"), ("locked", "Locked year")) if bm.get(k_)]
    for stage, title in stages:
        b = bm[stage]
        sc = b["scores"]
        order = ["champion", "reference"] + sorted([n_ for n_ in sc if n_ not in ("champion", "reference")], key=lambda n_: sc[n_]["mae"])
        order = [n_ for n_ in order if n_ in sc]
        _mini(f"{title}", f"{b['window'][0]} to {b['window'][1]}, {round(b['hours'] / 24):,} days, MAE in {units}")
        fd = go.Figure()
        ch_v = round(sc["champion"]["mae"], 3)
        fd.add_shape(type="line", x0=ch_v, x1=ch_v, y0=-0.5, y1=len(order) - 0.5, line=dict(color=MUTED, width=1, dash="dot"))
        fd.add_trace(go.Scatter(x=[sc[n_]["mae"] for n_ in order], y=[NAMES.get(n_, n_) for n_ in order], mode="markers+text", showlegend=False,
                                marker=dict(size=15, color=[CHAMP if n_ == "champion" else theme.BLUE if n_ == "reference" else LUCK for n_ in order], line=dict(color="white", width=1.5)),
                                text=[f"{round(sc[n_]['mae'], 3):.3f}" for n_ in order], textposition="middle right", textfont=dict(size=11, color=theme.INK_SECONDARY), cliponaxis=False, hoverinfo="skip"))
        theme.style_fig(fd, height=int(44 * len(order)) + 64, xaxis_title=f"MAE ({units})", legend=False)
        mx = max(sc[n_]["mae"] for n_ in order)
        fd.update_yaxes(autorange="reversed", showgrid=False, zeroline=False, fixedrange=True)
        fd.update_xaxes(range=[ch_v - 0.12 * (mx - ch_v) - 0.5, mx + 0.16 * (mx - ch_v) + 1.0], showgrid=False, zeroline=False)
        fd.update_layout(margin=dict(t=6, b=44, l=190, r=10))
        st.plotly_chart(fd, width="stretch", config={"displayModeBar": False})
        rows = []
        for n_ in order:
            if n_ == "champion":
                continue
            g_ = _gain(sc[n_]["mae"], sc["champion"]["mae"])
            pv_ = b["dm_champion_vs"].get(n_, {}).get("p_value") if n_ != "champion" else None
            ok = pv_ is not None and pv_ < 0.05
            rows.append([f"{_dot(n_) if n_ == 'reference' else ''}{NAMES.get(n_, n_)}", WHAT.get(n_, ""), f"{round(sc[n_]['mae'], 3):.3f}", f"{sc[n_]['rmse']:.3f}", f"{g_:.1f}%", _p(pv_),
                         (_real_pill() if g_ > 0 else _pill("Real: champion is worse", "#8a1f1f", "#f7e3e3")) if ok else _luck_pill()])
        _table(["Forecast", "What it is", "MAE", "RMSE", "Champion's gain", "p-value", "Verdict"], rows, first="minmax(11rem, 1.3fr)")
        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    nv = [n_ for n_ in (bm[stages[0][0]]["scores"] if stages else {}) if n_ not in ("champion", "reference")]
    if nv:
        _mini("Calculation of the gain over one forecast")
        _cc, _cd = st.columns([1, 3], gap="medium")
        with _cc:
            _sel = st.selectbox("Forecast", nv, index=nv.index("naive_seasonal") if "naive_seasonal" in nv else 0, format_func=lambda n_: NAMES.get(n_, n_), key="ml8_naive", label_visibility="collapsed")
        parts = ""
        for stage, title in stages:
            sc = bm[stage]["scores"]
            if _sel in sc:
                parts += (f"<span style='font-weight:600; color:{INK};'>{title}</span>"
                          f"<span>({round(sc[_sel]['mae'], 3):.3f} − {round(sc['champion']['mae'], 3):.3f}) ÷ {round(sc[_sel]['mae'], 3):.3f} = "
                          f"<b style='font-size:1.1rem; color:{INK};'>{_gain(sc[_sel]['mae'], sc['champion']['mae']):.1f}%</b></span>")
        with _cd:
            st.markdown(_panel(f"<div style='display:grid; grid-template-columns:9rem 1fr; row-gap:0.45rem; align-items:baseline;'>{parts}</div>"), unsafe_allow_html=True)
    _line("Each forecast is scored on the hours where all of them exist, so the champion's MAE here can differ in the third decimal from the one in step ⑤. The naive forecasts are not tuned and not chosen, so they cannot be flattered by the selection.")
    if bm.get("locked") and bm["locked"].get("synthetic_days"):
        _line(f"The locked year has {bm['locked']['synthetic_days']} days on which the auction published no price. They are left out for every forecast, because a forecast built on the day-ahead price would be flattered by them.")

# ============================================================================================ 5. the outcome
_section("The outcome", num="⑦")
if bm and bm.get("locked") and lk is not None:
    sc = bm["locked"]["scores"]
    naive = [n_ for n_ in sc if n_ not in ("champion", "reference")]
    strongest = min(naive, key=lambda n_: sc[n_]["mae"]) if naive else None
    real_n = [n_ for n_ in naive if (bm["locked"]["dm_champion_vs"].get(n_, {}).get("p_value") or 1) < 0.05 and _gain(sc[n_]["mae"], sc["champion"]["mae"]) > 0]
    lines = [f"<div><b>Gain over the {refl} on the locked year:</b> " + (f"{gain_lk:.1f}%, p = {_p(p_lk)}" if gain_lk is not None else "–") + ".</div>"]
    if strongest:
        lines.append(f"<div><b>Gain over the strongest naive forecast ({NAMES.get(strongest, strongest)}):</b> {_gain(sc[strongest]['mae'], sc['champion']['mae']):.1f}%, p = {_p(bm['locked']['dm_champion_vs'].get(strongest, {}).get('p_value'))}.</div>")
        lines.append(f"<div><b>Real gain over the naive forecasts:</b> {len(real_n)} of {len(naive)}.</div>")
    st.markdown(f"<div style='display:flex; align-items:center; gap:1.3rem; background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0.8rem 1.2rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>"
                f"<div style='font-size:1.45rem; font-weight:800; color:{INK}; line-height:1.25; max-width:16rem;'>{champ_txt}</div>"
                f"<div style='line-height:1.6; font-size:0.9rem; color:{theme.INK_SECONDARY};'>{''.join(lines)}</div></div>", unsafe_allow_html=True)
else:
    ml_ui.not_yet_run("the locked test and the naive benchmark")
st.caption("The plant inputs (solar irradiance, temperature, natural inflow) have no champion decision and no locked year: irradiance and temperature are served by the weather forecast and inflow by a model chosen by walk-forward validation; the held-out year is shown on the Plant inputs page.")
