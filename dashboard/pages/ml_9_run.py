"""ML Lifecycle 9 - Run it: registry, champion promotion, monitoring, retraining schedule and the live switch. Told as a story in numbered steps."""
from __future__ import annotations

import datetime as dt
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

ml_ui.header("Run it: registry, monitoring, live", "A model in production needs a record of how it was made, a rule for replacing it, a watch on "
             "its errors and a safe way to switch it on. This page shows each.", step=8)
ml_ui.tour_bar(9)

# The colours mean one thing each, on every picture of this page (and of the other ML pages):
#   green is a pass or a real result, grey is "not applicable" or luck, amber is a warning or a simulation, red marks a threshold or a flag, near-black is the champion.
MUTED, INK, CHAMP = theme.INK_MUTED, theme.INK_PRIMARY, "#2b2a28"
NEUTRAL, GREY, REAL, LUCK, RED, AMBER = "#6b6a64", "#b9b7ae", theme.GREEN, "#9a9890", theme.STATUS_CRITICAL, "#eda100"
CARD_LINE = "#e3e1d9"
_cap = lambda h_: h_[:1].upper() + h_[1:] if h_[:1].islower() else h_                                                  # noqa: E731


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


_ok_pill = lambda t_: _pill(t_, "#006100", "#e3f1e3")                                                                  # noqa: E731
_off_pill = lambda t_: _pill(t_, "#5a5953", "#ecebe6")                                                                 # noqa: E731
_warn_pill = lambda t_: _pill(t_, "#7a5200", "#fdf0d0")                                                                # noqa: E731
_legend_top = lambda fig, y=1.08: fig.update_layout(legend=dict(orientation="h", x=0, y=y, xanchor="left", yanchor="bottom", font=dict(size=12, color=theme.INK_SECONDARY)))     # noqa: E731


def _flow(items: list) -> None:
    """A row of boxes joined by arrows, as HTML: each item is (title, [lines], accent colour)."""
    html = ""
    for i, (title, lines, col) in enumerate(items):
        html += (f"<div style='flex:1 1 0; min-width:0; background:{col}1a; border-left:4px solid {col}; border-radius:8px; padding:0.55rem 0.8rem;'>"
                 f"<div style='font-weight:700; color:{INK}; font-size:0.92rem;'>{i + 1}. {title}</div>"
                 + "".join(f"<div style='font-size:0.78rem; color:{theme.INK_SECONDARY}; line-height:1.4;'>{ln}</div>" for ln in lines) + "</div>")
        if i < len(items) - 1:
            html += f"<div style='align-self:center; color:{MUTED}; font-size:1.1rem; padding:0 0.1rem;'>→</div>"
    st.markdown(f"<div style='display:flex; gap:0.4rem; align-items:stretch; margin:0.3rem 0 0.6rem;'>{html}</div>", unsafe_allow_html=True)


snap = ml_ui.cached(lambda: ml_data.pack_json("registry_snapshot"))() or {}
daily = ml_ui.cached(lambda: ml_data.pack("daily_loss"))()
locked = ml_ui.cached(lambda: ml_data.pack("locked_forecasts"))()
facts = ml_ui.cached(lambda: ml_data.pack_json("facts"))() or {}
pol = ml_ui.cached(lambda: ml_data.config("champion_policy").get("promotion", {}))()
mon = ml_ui.cached(lambda: ml_data.config("monitoring"))()
live = ml_ui.cached(lambda: ml_data.config("live_models").get("targets", {}))()
runs = pd.DataFrame(snap.get("runs", []))
champs = snap.get("champions", {})
sch = mon.get("schedule", {})
trk = mon.get("tracking", {})
window = int(trk.get("window_days", 28))
q = float(trk.get("quantile", 0.95))
n_auto = sum(1 for v in live.values() if v["mode"] == "auto")
n_reg = sum(1 for v in live.values() if v["mode"] == "registry")
n_leg = sum(1 for v in live.values() if v["mode"] == "legacy")

# ============================================================================================ 1. the life of a champion, drawn
SH, AN = [], []
ROW, X_END = 34.0, 1010
px_x = 1150.0 / X_END


def _shp(x0, x1, y0, y1, color, opacity=1.0, r_px=6.0):
    """A rounded bar as a path; x in the units of the figure, y in rows."""
    rx, ry = min(r_px / px_x, (x1 - x0) / 2), min(r_px / ROW, (y1 - y0) / 2)
    SH.append(dict(type="path", fillcolor=color, opacity=opacity, line=dict(width=0),
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
# ---- card 1: the life of a champion
y1 = -0.6
_fcard(y1, y1 + 6.4)
_fhead(y1 + 1.0, "①", "The life of a champion", "Each step has its own rule, written before it is needed")
nodes = [
    ("Registry", [f"{len(runs)} experiments", "with digests"], theme.BLUE),
    ("Decision", ["Rule, in code:", f"{len(champs)} champions"], theme.AQUA),
    ("Promotion", [f"p &lt; {pol.get('alpha', 0.05):.0%}, {pol.get('min_improvement', 0.02):.0%} lower,", f"{pol.get('min_days_between_promotions', 60)} days apart"] if pol else ["A challenger must", "beat the champion"], theme.VIOLET),
    ("Live switch", [f"{n_auto + n_reg} on champion,", f"{n_leg} original"], theme.GREEN),
    ("Monitoring", [f"{window}-day error vs", f"the {q:.0%} level"], RED),
    ("Retraining", [f"Refit: {sch.get('refit_every_days', 30)} days", f"Re-tune: {sch.get('retune_every_days', 180)} days"], theme.ORANGE),
]
nw, ng, ny0, nh = 150.0, 22.0, y1 + 2.0, 2.9
for i, (ttl, lines, col) in enumerate(nodes):
    x0 = i * (nw + ng)
    _shp(x0, x0 + nw, ny0, ny0 + nh, col, 0.10, r_px=8)
    _shp(x0, x0 + 5, ny0, ny0 + nh, col, 1.0, r_px=2)
    _fnote(x0 + 13, ny0 + 0.55, f"<b>{i + 1}. {ttl}</b>", size=12, anchor="left")
    for j, ln in enumerate(lines):
        _fnote(x0 + 13, ny0 + 1.35 + 0.55 * j, ln, size=10, color=theme.INK_SECONDARY, anchor="left")
    if i < len(nodes) - 1:
        _farrow(x0 + nw + 2, ny0 + nh / 2, x0 + nw + ng - 2, ny0 + nh / 2)
_fnote(0, ny0 + nh + 0.8, "A flag from the monitor recommends a review and, if the market changed, a re-tune. Nothing changes the champion by itself.", size=11, color=MUTED, anchor="left")
# ---- card 2: the schedule over the locked year
y2 = y1 + 6.85
_fcard(y2, y2 + 5.0)
start = dt.date(2025, 10, 12)
if not locked.empty:
    start = pd.to_datetime(locked["ts_utc"], utc=True).dt.tz_convert("Europe/Madrid").dt.date.min()
refit_d, tune_d = int(sch.get("refit_every_days", 30)), int(sch.get("retune_every_days", 180))
span = 365
xs0, xs1 = 90.0, 980.0
dx = lambda day_: xs0 + (xs1 - xs0) * day_ / span                                                                       # noqa: E731
_fhead(y2 + 1.0, "②", "The schedule over one year", f"From {start:%d-%m-%Y}, the first day of the locked year, as if it were live")
cyr, cyt = y2 + 2.5, y2 + 3.6
_fnote(8, cyr, "Refit", size=12, color=theme.BLUE, anchor="left")
_fnote(8, cyt, "Re-tune", size=12, color=theme.ORANGE, anchor="left")
SH.append(dict(type="line", x0=xs0, x1=xs1, y0=cyr, y1=cyr, line=dict(color="#e3e1d9", width=1.5), layer="below"))
SH.append(dict(type="line", x0=xs0, x1=xs1, y0=cyt, y1=cyt, line=dict(color="#e3e1d9", width=1.5), layer="below"))
rd = [i * refit_d for i in range(span // refit_d + 1)]
td = [i * tune_d for i in range(span // tune_d + 1)]
fig_a.add_trace(go.Scatter(x=[dx(d_) for d_ in rd], y=[cyr] * len(rd), mode="markers", showlegend=False, marker=dict(size=11, color=theme.BLUE, line=dict(color="white", width=1.2)),
                           text=[f"Refit on {start + dt.timedelta(days=d_):%d-%m-%Y}" for d_ in rd], hoverinfo="text"))
fig_a.add_trace(go.Scatter(x=[dx(d_) for d_ in td], y=[cyt] * len(td), mode="markers", showlegend=False, marker=dict(size=14, color=theme.ORANGE, symbol="diamond", line=dict(color="white", width=1.2)),
                           text=[f"Re-tune on {start + dt.timedelta(days=d_):%d-%m-%Y}" for d_ in td], hoverinfo="text"))
for d_ in td:
    _fnote(dx(d_), cyt + 0.7, f"{start + dt.timedelta(days=d_):%d-%m-%Y}", size=10, color=MUTED, anchor="center")
_fnote(dx(span), cyr - 0.7, f"{start + dt.timedelta(days=span - 1):%d-%m-%Y}", size=10, color=MUTED, anchor="right")
# ---- card 3: the safety chain of the live switch
y3 = y2 + 5.45
_fcard(y3, y3 + 5.5)
_fhead(y3 + 1.0, "③", "Checks before a champion runs live", "All three must hold, every time a champion is loaded")
checks = ["File matches its<br>registered checksum", "Belongs to the<br>active version", "Feature definitions<br>are unchanged"]
cw, cg, cy0, ch = 235.0, 26.0, y3 + 2.0, 1.7
for i, ttl in enumerate(checks):
    x0 = i * (cw + cg)
    _shp(x0, x0 + cw, cy0, cy0 + ch, theme.GREEN, 0.10, r_px=8)
    _shp(x0, x0 + 5, cy0, cy0 + ch, theme.GREEN, 1.0, r_px=2)
    _fnote(x0 + 16, cy0 + ch / 2, f"<b>{i + 1}.</b> {ttl}", size=11, anchor="left", align="left")
    _farrow(x0 + cw + 2, cy0 + ch / 2, x0 + cw + cg - 2, cy0 + ch / 2)
x0 = 3 * (cw + cg)
_shp(x0, X_END, cy0, cy0 + ch, CHAMP, 0.92, r_px=8)
_fnote((x0 + X_END) / 2, cy0 + ch / 2, "<b>Champion runs</b>", size=12, color="white")
_fnote(0, cy0 + ch + 0.75, "If one fails: the original forecaster runs with a warning (fallback mode), or an error is raised (champion-only mode)", size=11, color=MUTED, anchor="left")
end = y3 + 5.5
fig_a.layout.shapes = ()
theme.style_fig(fig_a, height=int((end + 0.9) * ROW + 12), xaxis_title=None, legend=False)
fig_a.update_yaxes(autorange=False, range=[end + 0.15, -0.75], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False, linewidth=0)
fig_a.update_xaxes(range=[-15, X_END + 15], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, showline=False)
fig_a.update_layout(margin=dict(t=2, b=2, l=10, r=10), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
fig_a.layout.shapes = SH
fig_a.layout.annotations = AN
st.plotly_chart(fig_a, width="stretch", config={"displayModeBar": False})

# ============================================================================================ 2. the registry
_section("The experiment registry", "Every run is recorded with its scores, the digests of its configuration and data, and the code version", num="④")
if runs.empty:
    ml_ui.not_yet_run("the registry snapshot")
else:
    rep_n = int((runs["attempts"] > 1).sum())
    ok_n = int((runs["reproduced"] == True).sum())          # noqa: E712
    bad_n = int((runs["reproduced"] == False).sum())        # noqa: E712
    _line(f"<b>{len(runs)}</b> experiments recorded, <b>{rep_n}</b> run more than once: <b>{ok_n}</b> gave exactly the same scores, <b>{bad_n}</b> did not. "
          "An experiment is identified by what defines it; running it again is a new attempt, and the registry checks that all attempts agree.")

    def _rows(frame):
        out = []
        for _, r_ in frame.iterrows():
            rp = r_["reproduced"]
            pill = _off_pill("Single attempt") if pd.isna(rp) else _ok_pill("Reproduced exactly") if bool(rp) else _warn_pill("Not reproduced")
            out.append([f"{ml_data.run_label(r_['kind'])}, {ml_data.label(r_['target'])}, {ml_data.run_label(r_['stage']).lower()}",
                        f"{pd.Timestamp(r_['window_start']):%d-%m-%Y} to {pd.Timestamp(r_['window_end']):%d-%m-%Y}", f"{int(r_['attempts'])}", pill,
                        f"<span style='font-family:monospace;'>{str(r_['run_key'])[:8]}</span>"])
        return out
    heads = ["Experiment", "Window", "Attempts", "Reproducibility", "Run key"]
    wdt = "minmax(11rem, 2fr) minmax(9.5rem, 1.5fr) minmax(3.5rem, 0.6fr) minmax(8rem, 1.1fr) minmax(5.5rem, 0.8fr)"
    rep_rows = runs[runs["attempts"] > 1]
    if len(rep_rows):
        _mini("The experiments that were run more than once")
        _table(heads, _rows(rep_rows), widths=wdt, left=(0, 1))
        _line("The Optuna day-ahead experiment, run twice, gave identical predictions: it is the one reproduced exactly. The random-search control is shown as not reproduced on purpose: "
              "its second attempt ran with the numerical guard, which changed one unstable row (see step ⑩).")
    with st.expander(f"All {len(runs)} experiments"):
        _table(heads, _rows(runs), widths=wdt, left=(0, 1))

# ============================================================================================ 3. champions and the promotion rule
_section("Champions and the promotion rule", "A challenger replaces the champion only if all three conditions hold, fixed before any challenge", num="⑤")
if champs:
    _mini("The registered champions")
    rows = []
    for t_, vs in sorted(champs.items(), key=lambda kv: ml_data.TARGETS.index(kv[0]) if kv[0] in ml_data.TARGETS else 99):
        for v_ in vs:
            pr = str(v_.get("promoted_utc", ""))
            try:
                pr = f"{pd.Timestamp(pr):%d-%m-%Y %H:%M}"
            except (ValueError, TypeError):
                pass
            rows.append([ml_data.label(t_), f"{v_['version']}", _ok_pill(str(v_["status"]).capitalize()) if str(v_["status"]) in ("active", "champion") else _off_pill(str(v_["status"]).capitalize()), pr])
    _table(["Target", "Version", "Status", "Promoted (UTC)"], rows)
else:
    _line("No champion is registered yet: the decisions exist, and registering a champion is a deliberate step. The registry holds the versions, the fitted models with their checksums and an append-only event log.")
if pol:
    _mini("The promotion rule", "Every challenge, accepted or rejected, is logged with its numbers")
    _flow([("Clearly better", [f"Loss at least {pol['min_improvement']:.0%} lower", "No churn for tiny gains"], theme.VIOLET),
           ("Significantly better", [f"Diebold-Mariano test, last {pol['window_days']} days", f"Better at {pol['alpha']:.0%}"], theme.VIOLET),
           ("Not too soon", [f"At least {pol['min_days_between_promotions']} days", "since the last promotion"], theme.VIOLET),
           ("Promotion", ["All three hold", "The old champion is retired"], CHAMP)])
    _mini("Two example challengers", "Illustrative numbers, not real challenges: the champion's loss is 17.80")
    ex = [("Challenger A", 17.40, 0.01, 75), ("Challenger B", 17.55, 0.03, 75), ("Challenger C", 17.30, 0.001, 20)]
    champ_loss = 17.80
    rows = []
    for nm, loss_, p_, days_ in ex:
        imp = 100 * (round(champ_loss, 2) - round(loss_, 2)) / round(champ_loss, 2)
        c1, c2, c3 = imp >= 100 * pol["min_improvement"], p_ < pol["alpha"], days_ >= pol["min_days_between_promotions"]
        mk = lambda ok_, txt_: (_ok_pill(f"✓ {txt_}") if ok_ else _off_pill(f"✗ {txt_}"))                                   # noqa: E731
        rows.append([nm, f"{loss_:.2f}", mk(c1, f"{imp:.1f}% lower"), mk(c2, f"p = {p_:g}"), mk(c3, f"{days_} days"), _ok_pill("Promoted") if (c1 and c2 and c3) else _off_pill("Rejected")])
    _table(["Challenger", "Loss", f"At least {pol['min_improvement']:.0%} lower", f"p below {pol['alpha']:.0%}", f"At least {pol['min_days_between_promotions']} days since the last", "Result"], rows,
           widths="minmax(8rem, 1.2fr) minmax(4rem, 0.6fr) repeat(3, minmax(7.5rem, 1.2fr)) minmax(6rem, 0.9fr)")
    _line("The improvement is worked out from the losses as shown: " + M(r"\frac{17.80-17.40}{17.80}=2.2\%") + ". A challenger that fails any one condition is rejected, and the rejection is logged.")

# ============================================================================================ 4. monitoring
_section("Monitoring: replay over the locked year", f"The rolling {window}-day error of the champion against what normal looked like before going live", num="⑥")
targets = sorted(set(locked["target"]), key=lambda t: ml_data.TARGETS.index(t)) if not locked.empty else []
if not targets:
    ml_ui.not_yet_run("the locked-year forecasts")
else:
    tgt = ml_ui.target_picker(targets=targets)
    sim = st.toggle("Simulate a market change (not real data)", key="ml9_sim")
    shock, factor = None, 1.0
    if sim:
        lo, hi = dt.date(2025, 10, 12), dt.date(2026, 10, 11)
        _s1, _s2 = st.columns(2, gap="large")
        with _s1:
            shock = str(st.slider("From day", lo, hi, dt.date(2026, 4, 1), key="ml9_shock"))
        with _s2:
            factor = st.slider("Errors grow by a factor of", 1.0, 3.0, 2.0, 0.1, key="ml9_factor")
    rep = ml_data.monitoring_replay(locked, daily, tgt, window, q, shock, factor)
    if rep is None:
        st.info("Not enough development days to set a threshold for this target.")
    else:
        units = ml_data.UNITS[tgt]
        dd = rep["daily"].dropna(subset=["rolling_mae"])
        _lc, _rc = st.columns([5, 3], gap="medium")
        with _lc:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dd["date"], y=dd["rolling_mae"], mode="lines", name=f"Rolling {window}-day MAE", line=dict(color=CHAMP, width=2)))
            fig.add_trace(go.Scatter(x=[dd["date"].min(), dd["date"].max()], y=[rep["level"]] * 2, mode="lines", name="Development level", line=dict(color=MUTED, width=1.5, dash="dot")))
            fig.add_trace(go.Scatter(x=[dd["date"].min(), dd["date"].max()], y=[rep["threshold"]] * 2, mode="lines", name=f"Threshold ({q:.0%} of development windows)", line=dict(color=RED, width=1.5, dash="dash")))
            flagged = dd[dd["flag"]]
            if len(flagged):
                fig.add_trace(go.Scatter(x=flagged["date"], y=flagged["rolling_mae"], mode="markers", name="Flag", marker=dict(color=RED, size=6)))
            if sim and shock:
                fig.add_shape(type="line", x0=pd.Timestamp(shock), x1=pd.Timestamp(shock), y0=0, y1=1, yref="paper", line=dict(color=AMBER, width=2))
                fig.add_annotation(x=pd.Timestamp(shock), y=1, yref="paper", text="Simulated change", showarrow=False, yanchor="bottom", font=dict(color="#7a5200"))
            theme.style_fig(fig, height=340, yaxis_title=f"MAE ({units})")
            _legend_top(fig, 1.02)
            fig.update_layout(margin=dict(t=64))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with _rc:
            lv, th = round(rep["level"], 3), round(rep["threshold"], 3)
            fl = rep["flagged_days"]
            if sim:
                verdict = _warn_pill("Simulation: not a result")
            elif fl:
                verdict = _warn_pill("Flag raised: review recommended")
            else:
                verdict = _ok_pill("No flag")
            _hd = lambda txt_: f"<div style='font-size:0.74rem; font-weight:600; color:{MUTED}; margin-top:0.55rem;'>{txt_}</div>"                    # noqa: E731
            st.markdown(_panel(
                f"<div style='font-size:0.74rem; font-weight:600; color:{MUTED}; padding:0.2rem 0;'>① The development level</div>"
                f"<div>The mean of the rolling {window}-day MAE before going live = <b style='font-size:1.1rem; color:{INK};'>{lv:.3f}</b></div>"
                + _hd("② The threshold")
                + f"<div>The {q:.0%} quantile of the same statistic = <b style='font-size:1.1rem; color:{INK};'>{th:.3f}</b>, {100 * (th - lv) / lv:+.1f}% above the level</div>"
                + _hd("③ Days above the threshold")
                + f"<div><b style='font-size:1.1rem; color:{INK};'>{fl}</b>" + (f", the first on {rep['first_flag']:%d-%m-%Y}" if rep["first_flag"] is not None else "") + "</div>"
                + f"<div style='margin-top:0.7rem;'>{verdict}</div>"), unsafe_allow_html=True)
        if sim:
            _line("The errors after the chosen day were multiplied. This shows how the flag reacts; it is not a result.")
        elif fl:
            _line(f"In the real locked year the rolling error crosses the threshold on <b>{fl} day(s)</b>, first on {rep['first_flag']:%d-%m-%Y}. That is the monitor doing its job: it raises a flag and recommends a review and, "
                  "if the cause is a change in the market, a re-tuning. It never changes the champion by itself.")
        _line("The full rule also uses a bias flag, a coverage flag and a comparison with the reference. A degraded status needs the reference to be better, or both level and bias to be off.")

# ============================================================================================ 5. the retraining schedule
if sch:
    _section("Retraining schedule", "A refit retrains on the latest data and reuses the tuned settings; the search runs again at every re-tune", num="⑦")
    st.markdown(_panel(
        f"<div style='display:grid; grid-template-columns:9rem 1fr; row-gap:0.45rem; align-items:baseline;'>"
        f"<span style='font-weight:600; color:{INK};'>Refit</span><span>Every <b style='font-size:1.1rem; color:{INK};'>{refit_d}</b> days, settings reused</span>"
        f"<span style='font-weight:600; color:{INK};'>Re-tune</span><span>Every <b style='font-size:1.1rem; color:{INK};'>{tune_d}</b> days, the search runs again</span>"
        f"<span style='font-weight:600; color:{INK};'>Refits per re-tune</span><span>{M(rf'\frac{{{tune_d}}}{{{refit_d}}}=')} <b style='font-size:1.1rem; color:{INK};'>{tune_d // refit_d}</b></span></div>"), unsafe_allow_html=True)
    _line("The live schedule equals the one used in the evaluation, checked when the configuration is loaded, so live behaviour is the behaviour that was scored.")

# ============================================================================================ 6. the live switch
_section("Switching a forecast on, safely", "A target is switched by editing one setting, after the champion has been scored against the original forecaster on the same days", num="⑧")
if live:
    mode_pill = {"legacy": lambda: _off_pill("Original forecaster"), "auto": lambda: _ok_pill("Champion, original as fallback"), "registry": lambda: _ok_pill("Champion only")}
    rows = []
    for t_ in ml_data.TARGETS:
        v_ = live.get(t_)
        reg = _ok_pill("Registered") if t_ in champs else _off_pill("None yet")
        if v_ is None:
            rows.append([ml_data.label(t_), _warn_pill("Cannot be switched"), reg])
        else:
            rows.append([ml_data.label(t_), mode_pill.get(v_["mode"], lambda: _off_pill(v_["mode"]))(), reg])
    _table(["Forecast", "Currently uses", "Champion in the registry"], rows, widths="minmax(10rem, 1.4fr) minmax(14rem, 1.6fr) minmax(8rem, 1fr)", left=(0,))
    _line("mFRR cannot be switched: the original mFRR forecast is the capacity price, while the framework's target is the activation price. "
          "aFRR is switched one side at a time: the up and the down price are separate targets, and a side without a champion keeps the original forecaster.")
    _mini("The safety rules")
    _table(["Rule", "What it guarantees"],
           [["Three modes", "Original forecaster; champion with the original as fallback (a warning is raised); champion only (an error is raised if it cannot run)"],
            ["Verified load", "The champion is loaded only if the file matches its registered checksum, belongs to the active version and the feature definitions have not changed"],
            ["One setting per target", "A target is switched by editing one setting, after the champion was scored against the original on the same days"],
            ["Complete data", "Live forecasts, fits and monitoring read the cleaned and completed data, so a gap in a feed does not stop a forecast"]],
           widths="minmax(10rem, 1fr) minmax(20rem, 3.4fr)", left=(0, 1))

# ============================================================================================ 7. engineering
_section("Engineering", num="⑨")
_table(["What", "Count", "Where it counts"],
       [["Automated tests", f"{facts.get('test_functions', '–')}", f"in {facts.get('test_files', '–')} test files"],
        ["Framework modules", f"{facts.get('framework_modules', '–')}", "the reusable forecasting and evaluation code"],
        ["Command-line tools", f"{facts.get('tools', '–')}", "the steps can be run and repeated from the terminal"],
        ["Configuration files", f"{facts.get('config_files', '–')}", "the rules on this page are read from settings"]],
       widths="minmax(10rem, 1.2fr) minmax(5rem, 0.6fr) minmax(16rem, 2.6fr)", left=(0, 2))

# ============================================================================================ 8. the numerical failure
_section("A numerical failure, found and guarded", "A boosted model can leave the range of the prices it was trained on", num="⑩")
_flow([("What went wrong", ["One tuned XGBoost, with an outlier-robust loss (pseudo-Huber), in one fold (April 2023)", "Forecasts up to 5e21"], RED),
       ("The guard", ["Every forecast is limited to the training prices plus a margin", "In training, in the tuning validation and in the live champion"], theme.BLUE),
       ("The proof", ["The Optuna experiment, rerun with the guard, is bit-identical: no forecast was ever limited", "The 720 limited forecasts of the control are exactly the incident"], theme.GREEN)])

st.caption("The plant inputs are not registered as champions and are not monitored here: irradiance and temperature come from the weather forecast, and the inflow forecaster is retrained daily on the full history.")
