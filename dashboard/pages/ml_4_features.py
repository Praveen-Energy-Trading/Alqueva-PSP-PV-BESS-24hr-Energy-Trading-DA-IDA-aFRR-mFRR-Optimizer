"""ML Lifecycle 4 - Features: a catalogue with written availability, feature sets. Told as a story in numbered steps."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ml_data
import ml_ui
import theme

ml_ui.header("Features", "", step=3)
ml_ui.tour_bar(4)

# The colours mean one thing each: a group of features keeps one colour; green is available, grey is "does not apply", red is "too new to be known", amber is "not in the information set".
MUTED, INK, NEUTRAL = theme.INK_MUTED, theme.INK_PRIMARY, "#6b6a64"
REAL, BAD = theme.STATUS_GOOD, theme.STATUS_CRITICAL
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


def _line(html_: str) -> None:
    """One short note under a table or chart, the key numbers in bold."""
    st.markdown(f"<div style='font-size:0.86rem; color:{theme.INK_SECONDARY}; margin:0.25rem 0 0.9rem;'>{html_}</div>", unsafe_allow_html=True)


def _panel(inner: str) -> str:
    """A white rounded card, as on the Tuning lab."""
    return (f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0.55rem 1.1rem 0.8rem; box-shadow:0 1px 2px rgba(0,0,0,0.04); "
            f"font-size:0.9rem; color:{theme.INK_SECONDARY};'>{inner}</div>")


def _table(headers: list, rows: list, widths: str | None = None, left: tuple = (0,), max_height: str | None = None) -> None:
    """A table as on the Tuning lab: muted header, thin row lines, the first column left, the others centred; a long table scrolls under a fixed header."""
    grid = f"display:grid; grid-template-columns:{widths or f'minmax(9rem, 1.5fr) repeat({len(headers) - 1}, minmax(0, 1fr))'}; align-items:center; column-gap:0.6rem;"
    head = (f"<div style='{grid} padding:0.6rem 0 0.4rem; border-bottom:1px solid {CARD_LINE}; font-size:0.72rem; color:{MUTED}; font-weight:600; line-height:1.2; "
            f"{'position:sticky; top:0; background:#fff; z-index:1;' if max_height else ''}'>"
            + "".join(f"<span style='{'' if j in left else 'text-align:center;'}'>{h}</span>" for j, h in enumerate(headers)) + "</div>")
    body = "".join(f"<div style='{grid} min-height:2.5rem; font-size:0.86rem; border-bottom:1px solid #eeede8;'>"
                   + "".join(f"<span style='{'' if j in left else 'text-align:center;'} color:{INK if str(c) != '–' else MUTED};'>{c}</span>" for j, c in enumerate(row)) + "</div>" for row in rows)
    inner = f"<div style='max-height:{max_height}; overflow-y:auto;'>{head}{body}</div>" if max_height else f"{head}{body}"
    st.markdown(f"<div style='background:#fff; border:1px solid {CARD_LINE}; border-radius:12px; padding:0 1.1rem 0.1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);'>{inner}</div>", unsafe_allow_html=True)


def _pill(text: str, color: str, bg: str) -> str:
    return f"<span style='background:{bg}; color:{color}; border-radius:999px; padding:0.1rem 0.6rem; font-size:0.78rem; font-weight:600; white-space:nowrap;'>{text}</span>"


_ok_pill = lambda t_: _pill(t_, "#006100", "#e3f1e3")                                                                  # noqa: E731
_off_pill = lambda t_: _pill(t_, "#5a5953", "#ecebe6")                                                                 # noqa: E731
_warn_pill = lambda t_: _pill(t_, "#7a5200", "#fdf0d0")                                                                # noqa: E731
_bad_pill = lambda t_: _pill(t_, "#8a1f1f", "#f7e3e3")                                                                 # noqa: E731

cat = ml_ui.cached(lambda: ml_data.config("feature_catalogue"))()
avail = ml_ui.cached(lambda: ml_data.pack("feature_availability"))()
feats = pd.DataFrame(cat.get("features", []))
if not avail.empty and not feats.empty:
    avail = avail[avail["feature"].isin(feats["id"])]                 # the pack may still list features that are no longer in the catalogue
if feats.empty:
    ml_ui.not_yet_run("the feature catalogue")
    st.stop()

pick_col, _ = st.columns([1, 2])
with pick_col:
    target = ml_ui.target_picker()
tlabel = ml_data.label(target)
_desc = dict(zip(feats["id"], feats["description"]))


def fname(fid: str) -> str:
    """A feature by what it is (its description), not by its identifier."""
    return _cap(str(_desc.get(fid, fid)))


g = feats.groupby("group").size().sort_values(ascending=False)
GCOL = {grp_: theme.CATEGORICAL[i % 8] for i, grp_ in enumerate(g.index)}                                              # one colour per group, on every picture
usable = set(avail[(avail["target"] == target) & (avail["available"])]["feature"]) if not avail.empty else set(feats["id"])
n_all, n_ok = len(feats), len(usable & set(feats["id"]))

# ============================================================================================ 1. the catalogue
_section("The catalogue", f"{n_all} features, each written down once with its group, its kind and what it is", num="①")
_mini("Features per group")
fig = go.Figure(go.Bar(x=g.values, y=[ml_data.pretty(x_) for x_ in g.index], orientation="h", marker_color=[GCOL[x_] for x_ in g.index], text=[str(v_) for v_ in g.values],
                       textposition="outside", cliponaxis=False, textfont=dict(size=12, color=theme.INK_SECONDARY), hoverinfo="skip"))
fig.update_yaxes(autorange="reversed")
theme.style_fig(fig, height=44 * len(g) + 50, xaxis_title="Features", legend=False)
fig.update_xaxes(range=[0, float(g.max()) * 1.12])
fig.update_layout(margin=dict(t=6))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
_line(f"The groups add up to <b>{int(g.sum())}</b> features: " + " + ".join(f"{v_}" for v_ in g.values) + f" = {int(g.sum())}.")

_mini("Every feature", f"Marked for {tlabel}: the features it may use come first and are numbered")
grp = st.multiselect("Groups", list(g.index), default=list(g.index), key="ml4_groups", format_func=ml_data.pretty)
show = feats[feats["group"].isin(grp)]
show = show.assign(ok=show["id"].isin(usable)).sort_values("ok", ascending=False, kind="stable")
rows = []
for i, (_, r_) in enumerate(show.iterrows(), 1):
    rows.append([str(i) if r_["ok"] else "–", fname(r_["id"]), f"<span style='color:{GCOL[r_['group']]}'>■</span>&nbsp;{ml_data.pretty(r_['group'])}", ml_data.pretty(r_["kind"]),
                 _ok_pill("Available") if r_["ok"] else _off_pill("Not available")])
_table(["No.", "Feature", "Group", "Kind", "This target"], rows, widths="minmax(2.5rem, 0.4fr) minmax(12rem, 3fr) minmax(7rem, 1.2fr) minmax(7rem, 1.2fr) minmax(7.5rem, 1fr)", left=(1, 2, 3), max_height="26rem")

# ============================================================================================ 2. availability
_section("Availability: what each target may use", "A feature is available only if everything it reads was published before the forecast is made", num="②")
if avail.empty:
    ml_ui.not_yet_run("the availability matrix")
else:
    a = avail[avail["group"].isin(grp)]
    order = [t for t in ml_data.TARGETS if t in set(a["target"])]
    cnt = a.groupby(["group", "target"])["available"].agg(["sum", "count"]).reset_index()
    ok = cnt.pivot(index="group", columns="target", values="sum")[order]
    tot = cnt.pivot(index="group", columns="target", values="count")[order]
    text = ok.astype(int).astype(str) + " of " + tot.astype(int).astype(str)
    _mini("Available features, by group and target", "Green: all of the group; red: none of it; the number is how many of the group")
    fig = go.Figure(go.Heatmap(z=(ok / tot).values, x=[ml_data.short(t) for t in order], y=[ml_data.pretty(x_) for x_ in ok.index], text=text.values,
                               texttemplate="%{text}", colorscale=[[0, "#f1c8c8"], [0.5, "#f7e7b0"], [1, "#a9d8a9"]], showscale=False, zmin=0, zmax=1, xgap=3, ygap=3,
                               hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    theme.style_fig(fig, height=70 + 42 * len(ok), legend=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    sel = avail[avail["target"] == target]
    n_bad = int((~sel["available"]).sum())
    _mini(f"The features of {tlabel}, by group", f"{n_ok} available + {n_bad} not available = {n_ok + n_bad} features")
    grows = []
    for grp_ in g.index:
        s_ = sel[sel["group"] == grp_]
        if s_.empty:
            continue
        k_, n_ = int(s_["available"].sum()), len(s_)
        mg = s_.loc[s_["available"] & s_["margin_min"].notna(), "margin_min"]
        margin = "–" if mg.empty else (f"{mg.min() / 60:,.1f} hours" if mg.min() <= 72 * 60 else f"{mg.min() / 1440:,.1f} days")
        pill = _ok_pill("All") if k_ == n_ else _bad_pill("None") if k_ == 0 else _warn_pill("Some")
        grows.append([f"<span style='color:{GCOL[grp_]}'>■</span>&nbsp;{ml_data.pretty(grp_)}", f"{k_}", f"{n_}", f"{100 * k_ / n_:.0f}%", pill, margin])
    _table(["Group", "Available", "In the group", "Share available", "Status", "Smallest safety margin"], grows,
           widths="minmax(9rem, 1.6fr) minmax(4.5rem, 0.7fr) minmax(4.5rem, 0.7fr) minmax(5rem, 0.8fr) minmax(5rem, 0.8fr) minmax(7rem, 1.1fr)")
    _line("The safety margin is the time between the moment the newest value a feature reads was published and the moment the forecast is made; the smallest one in the group is shown. A dash means there is nothing to measure: the group reads no published series, or none of it is available.")
    bad = sel[~sel["available"]].copy()
    if len(bad):
        def _kind(r_: str):
            r_ = str(r_)
            if r_.startswith("reads "):
                return "Too new", _bad_pill("Too new")
            if "information set" in r_:
                return "Not in the information set", _warn_pill("Not in the information set")
            return "Does not apply", _off_pill("Does not apply")
        bad["why"] = bad["reason"].map(lambda r_: _kind(r_)[0])
        bad["pill"] = bad["reason"].map(lambda r_: _kind(r_)[1])
        _mini(f"Why {len(bad)} features are not available for {tlabel}")
        counts = bad["why"].value_counts()
        pill_of = {"Too new": _bad_pill("Too new"), "Not in the information set": _warn_pill("Not in the information set"), "Does not apply": _off_pill("Does not apply")}
        crows = [[pill_of.get(k_, k_), f"{int(v_)}", f"{100 * v_ / len(bad):.0f}%"] for k_, v_ in counts.items()]
        _table(["Reason", "Features", "Share of the unavailable"], crows, widths="minmax(12rem, 2fr) minmax(5rem, 1fr) minmax(7rem, 1.4fr)")
        _line("Too new: the feature reads a series up to a day that is later than the target may use, which would leak information from after the forecast is made. "
              "Not in the information set: the series is not one the target is allowed to see. Does not apply: the feature belongs to another kind of target.")
        with st.expander(f"The {len(bad)} features, one by one"):
            _table(["Feature", "Group", "Reason", "Detail"],
                   [[fname(r_["feature"]), f"<span style='color:{GCOL.get(r_['group'], MUTED)}'>■</span>&nbsp;{ml_data.pretty(r_['group'])}", r_["pill"], _cap(ml_data.plain_text(r_["reason"]))] for _, r_ in bad.iterrows()],
                   widths="minmax(12rem, 2fr) minmax(7rem, 1fr) minmax(9rem, 1.2fr) minmax(14rem, 2.6fr)", left=(0, 1, 3), max_height="22rem")
