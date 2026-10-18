"""ML Lifecycle 1 - Data and quality: the sources, the audited raw zone, the quality gate, missing and empty days. Told as a story in numbered steps."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fill_legend
import ml_data
import ml_ui
import theme

ml_ui.header("Data and quality", "", step=0)
ml_ui.tour_bar(1)

# The colours mean one thing each: green is a pass or a good state, amber is a repair or a warning, red is a failure, grey is "nothing to report".
MUTED, INK = theme.INK_MUTED, theme.INK_PRIMARY
NEUTRAL, GREY, REAL = "#6b6a64", "#b9b7ae", theme.GREEN
CARD_LINE = "#e3e1d9"
_cap = lambda h_: h_[:1].upper() + h_[1:] if h_[:1].islower() else h_                                                  # noqa: E731
_dmy = lambda x: f"{str(x)[8:10]}-{str(x)[5:7]}-{str(x)[:4]}"                                                          # noqa: E731


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
    """A white rounded card, as on the Tuning lab."""
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


_ok_pill = lambda t_: _pill(t_, "#006100", "#e3f1e3")                                                                  # noqa: E731
_off_pill = lambda t_: _pill(t_, "#5a5953", "#ecebe6")                                                                 # noqa: E731
_warn_pill = lambda t_: _pill(t_, "#7a5200", "#fdf0d0")                                                                # noqa: E731
_bad_pill = lambda t_: _pill(t_, "#8a1f1f", "#f7e3e3")                                                                 # noqa: E731
_legend_top = lambda fig, y=1.08: fig.update_layout(legend=dict(orientation="h", x=0, y=y, xanchor="left", yanchor="bottom", font=dict(size=12, color=theme.INK_SECONDARY)))     # noqa: E731

SERIES_GROUP = {"da_price": "Day-ahead price", "ida_price": "Intraday auction price", "xbid_price": "XBID price", "afrr_price": "aFRR price",
                "mfrr_price": "mFRR price", "imbalance_price": "Imbalance price"}


def _group(text: str) -> str:
    """'ida_price session 1' as 'Intraday auction price, session 1'."""
    m_ = re.match(r"^([a-z_]+?)(?: session (\d))?$", text.strip())
    if not m_:
        return text
    base = SERIES_GROUP.get(m_.group(1), m_.group(1).replace("_", " ").capitalize())
    return base + (f", session {m_.group(2)}" if m_.group(2) else "")


manifest = ml_ui.cached(lambda: ml_data.pack("raw_manifest_summary"))()
gate = ml_ui.cached(lambda: ml_data.pack_json("quality_gate"))()
status = ml_ui.cached(lambda: ml_data.pack("load_status_monthly"))()

# ============================================================================================ 1. data sources
_section("Data sources", "The raw zone is audited: every source is recorded with its days and its file versions", num="①")
if manifest.empty:
    ml_ui.not_yet_run("the data pack")
else:
    rows = []
    for _, r_ in manifest.iterrows():
        span = (pd.Timestamp(r_["last_key"]) - pd.Timestamp(r_["first_key"])).days + 1
        miss = int(span - r_["days"])
        rows.append([ml_data.source_label(r_["source"]), f"{int(r_['days']):,}", _dmy(r_["first_key"]), _dmy(r_["last_key"]), f"{span:,}",
                     _ok_pill("None") if miss == 0 else _warn_pill(f"{miss:,} days ({100 * miss / span:.1f}%)"),
                     _ok_pill("None") if int(r_["versions_above_1"]) == 0 else _warn_pill(f"{int(r_['versions_above_1']):,}")])
    _table(["Source", "Days held", "First day", "Last day", "Days in the span", "Missing in the span", "Files with several versions"], rows,
           widths="minmax(10rem, 1.7fr) minmax(4rem, 0.7fr) minmax(5.2rem, 0.9fr) minmax(5.2rem, 0.9fr) minmax(4rem, 0.7fr) minmax(7rem, 1.1fr) minmax(5.5rem, 0.8fr)")
    _line("Missing in the span = days in the span − days held, where the span runs from the first to the last day. A missing day is either a day the source never published or one that failed to load; see step ⑤.")

# ============================================================================================ 2. the quality gate
_section("The quality gate", "The latest check of the data: it passes, or it says why not", num="②")
if gate is None:
    ml_ui.not_yet_run("the quality check")
else:
    passed = bool(gate.get("passed"))
    st.markdown(_panel(
        f"<div style='display:flex; align-items:center; gap:1.2rem; flex-wrap:wrap;'>"
        f"<div style='font-size:1.5rem; font-weight:800; color:{REAL if passed else theme.STATUS_CRITICAL};'>{'Passed' if passed else 'Failed'}</div>"
        f"<div style='line-height:1.6; font-size:0.9rem;'><div><b>Window checked:</b> {_dmy(gate['window'][0])} to {_dmy(gate['window'][1])}</div>"
        f"<div><b>Run on:</b> {_dmy(gate.get('today', ''))}</div></div></div>"), unsafe_allow_html=True)
    reasons = gate.get("reasons", [])
    if reasons:
        _mini("Why it failed")
        _table(["Reason"], [[ml_data.plain_text(r_)] for r_ in reasons], widths="1fr", left=(0,))
    latest = gate.get("latest_days") or {}
    today = pd.Timestamp(gate.get("today")) if gate.get("today") else None
    if latest and today is not None:
        _mini("The latest day held for each series", f"Compared with the day the check ran, {_dmy(gate['today'])}")
        lrows = []
        for k_, v_ in latest.items():
            behind = (today - pd.Timestamp(v_)).days
            lrows.append([SERIES_GROUP.get(k_, k_.replace("_", " ").capitalize()), _dmy(v_), f"{behind}", _ok_pill("Up to date") if behind <= 1 else _warn_pill("Behind")])
        _table(["Series", "Latest day", "Days behind", "Status"], lrows, widths="minmax(12rem, 2fr) minmax(6rem, 1fr) minmax(5rem, 0.8fr) minmax(6rem, 1fr)")
        _line("Up to date means the latest day held is at most one day behind the day the check ran.")
    repairs = []
    for t_ in gate.get("repairs", []):
        m_ = re.match(r"^(.*?): (\d+)/(\d+) days missing \([\d.]+%\), (.*)$", t_)
        if m_:
            n_, N_ = int(m_.group(2)), int(m_.group(3))
            repairs.append([_group(m_.group(1)), f"{n_:,}", f"{N_:,}", f"{100 * n_ / N_:.1f}%", _warn_pill(_cap(m_.group(4)))])
        else:
            repairs.append([t_, "–", "–", "–", "–"])
    if repairs:
        _mini("Repairs made by the gate", "A missing day is filled by the data curation, so a model never meets a gap")
        _table(["Series", "Days missing", "Days in the window", "Share missing", "What was done"], repairs, widths="minmax(12rem, 2fr) repeat(3, minmax(5rem, 0.8fr)) minmax(11rem, 1.6fr)")

# ============================================================================================ 3. checking, cleaning and imputation
_section("Checking, cleaning and imputation", "Data errors are removed, then each missing hour is filled by one stated method", num="③")
cur = ml_ui.cached(lambda: ml_data.pack("curation_summary"))()
if cur.empty:
    ml_ui.not_yet_run("the curation summary")
else:
    _mini("What was checked", "Per series, in hours")
    rows = []
    for _, r_ in cur.iterrows():
        sh = 100 * r_["filled"] / r_["hours"] if r_["hours"] else 0.0
        rows.append([ml_data.series_label(r_["series"]), f"{int(r_['hours']):,}", f"{int(r_['published']):,}", f"{int(r_['errors_removed']):,}", f"{int(r_['filled']):,}",
                     (_ok_pill("0.00%") if r_["filled"] == 0 else _warn_pill(f"{sh:.2f}%"))])
    _table(["Series", "Hours in the series", "Published", "Data errors removed", "Hours imputed", "Share imputed"], rows,
           widths="minmax(10rem, 1.6fr) repeat(4, minmax(4.8rem, 0.9fr)) minmax(5rem, 0.9fr)")
    _line("Share imputed = hours imputed ÷ hours in the series. A series with 0.00% needed no filling.")
    gaps = ml_ui.cached(lambda: ml_data.pack("curation_gaps"))()
    if not gaps.empty:
        _mini("How the gaps were filled", "The number of separate gaps, with the hours in brackets: 187 small gaps read faster than 213 hours")
        method_column = {"straight line between the neighbours": "Straight line", "last value carried forward": "Carried forward",
                         "day-ahead price plus the typical spread": "Day-ahead price + spread", "same hour a week earlier": "Same hour earlier"}
        per = gaps.groupby(["series", "method"]).agg(n=("hours", "size"), h=("hours", "sum"))
        grows = []
        for _, r_ in cur.iterrows():
            cells = [f"{int(per.loc[(r_['series'], mth), 'n']):,} ({int(per.loc[(r_['series'], mth), 'h']):,})" if (r_["series"], mth) in per.index else "–" for mth in method_column]
            if any(c_ != "–" for c_ in cells):
                grows.append([ml_data.series_label(r_["series"])] + cells)
        _table(["Series"] + [f"{v_}: gaps (hours)" for v_ in method_column.values()], grows, widths="minmax(11rem, 1.6fr) repeat(4, minmax(6rem, 1fr))")
        _line("A series that needed no filling is left out. The pictures below show each method and the exact condition for using it.")
    st.markdown(fill_legend.html_block(), unsafe_allow_html=True)

# ============================================================================================ 4. plant input series
_section("Plant input series", "Irradiance and temperature are downloaded; inflow comes from a file", num="④")
_plant = [(key, name, ml_data.plant_input(key)) for key, name, _ in ml_data.PLANT_INPUTS]
if not any(r for _, _, r in _plant):
    ml_ui.not_yet_run("the plant input description (tools/evaluate_plant_inputs.py)")
else:
    rows = []
    for key, name, r in _plant:
        d = (r or {}).get("data")
        if not d:
            continue
        tot = int(d["measured"]) + int(d["filled"])
        rows.append([f"{name} ({d['unit']}, {r['resolution']})", "ERA5 reanalysis, then recent days (Open-Meteo)" if key in ("ghi", "temperature") else "Guadiana inflow research dataset",
                     f"{_dmy(d['first'])} to {_dmy(d['last_measured'])}", f"{int(d['measured']):,}", _warn_pill(f"{int(d['filled']):,} ({100 * d['filled'] / tot:.1f}%)")])
    _table(["Series", "Source", "Measured from and to", "Measured values", "Imputed values (share)"], rows,
           widths="minmax(10rem, 1.5fr) minmax(11rem, 1.7fr) minmax(9.5rem, 1.4fr) minmax(5rem, 0.8fr) minmax(6.5rem, 1fr)", left=(0, 1))
    _line("Where a day is missing the file holds imputed values: a synthetic clear-sky curve for today's irradiance and temperature, the monthly mean for inflow. Only measured values are used as a test target.")

# ============================================================================================ 5. missing and empty days
_section("Missing and empty days", "The state of every delivery day, month by month, for one source", num="⑤")
if status.empty:
    ml_ui.not_yet_run("the load log summary")
else:
    srcs = sorted(status["source"].unique())
    _sc, _ = st.columns([3, 2], gap="medium")
    with _sc:
        src = st.selectbox("Source", srcs, index=srcs.index("omie_ida1") if "omie_ida1" in set(status["source"]) else 0, key="ml1_src", format_func=ml_data.source_label)
    s = status[status["source"] == src].pivot_table(index="month", columns="status", values="days", aggfunc="sum").fillna(0)
    order = [c_ for c_ in ("loaded", "unchanged", "empty_publication", "not_published", "rejected", "failed") if c_ in s.columns] + [c_ for c_ in s.columns if c_ not in ("loaded", "unchanged", "empty_publication", "not_published", "rejected", "failed")]
    palette = {"loaded": theme.GREEN, "unchanged": "#8fc98f", "empty_publication": theme.YELLOW, "not_published": theme.ORANGE, "rejected": theme.VIOLET, "failed": theme.STATUS_CRITICAL}
    fig = go.Figure()
    for c_ in order:
        fig.add_trace(go.Bar(x=s.index, y=s[c_], name=c_.replace("_", " ").capitalize(), marker_color=palette.get(c_, MUTED)))
    theme.style_fig(fig, height=320, yaxis_title="Days", barmode="stack")
    _legend_top(fig)
    fig.update_layout(margin=dict(t=34))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    tot = s.sum().sum()
    _mini(f"The days of {ml_data.source_label(src)} by state", f"{int(tot):,} days in all")
    srows = []
    for c_ in order:
        n_ = int(s[c_].sum())
        good = c_ in ("loaded", "unchanged")
        srows.append([f"<span style='color:{palette.get(c_, MUTED)}'>■</span>&nbsp;{c_.replace('_', ' ').capitalize()}", f"{n_:,}", f"{100 * n_ / tot:.1f}%",
                      _ok_pill("Usable") if good else _warn_pill("Needs a repair or a note")])
    _table(["State", "Days", "Share of days", "Meaning"], srows, widths="minmax(12rem, 2fr) repeat(2, minmax(5rem, 0.8fr)) minmax(10rem, 1.6fr)")
    _line("Loaded and unchanged days are usable. The other states are days the source did not publish, published empty, or that were rejected or failed to load; the data curation fills the gaps (step ③).")
