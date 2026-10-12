"""
ml_ui.py — the shared components of the ML Lifecycle pages: header, target picker, status badges and the interview-mode tour. Streamlit only; the data comes from ml_data.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import ml_data
import theme

PAGES = [("ml_0_overview", "Overview"), ("ml_1_data", "Data and quality"), ("ml_2_market", "Market analysis"),
         ("ml_3_formulation", "Problem formulation"), ("ml_4_features", "Features"), ("ml_5_evaluation", "Evaluation design"),
         ("ml_6_tuning", "Tuning lab"), ("ml_7_arena", "Model arena"), ("ml_8_decision", "Decision and locked test"),
         ("ml_9_run", "Run it: registry, monitoring, live")]

STATE_COLOR = {"done": theme.STATUS_GOOD, "partial": theme.STATUS_WARNING, "not run": theme.STATUS_NEUTRAL}


def cached(fn):
    """Cache a data-layer function for the session (the files are committed and change only with a commit)."""
    return st.cache_data(show_spinner=False)(fn)


def interview_mode() -> bool:
    return bool(st.session_state.get("ml_interview_mode", False))


def sidebar_controls() -> None:
    st.sidebar.toggle("Interview mode", key="ml_interview_mode",
                      help="Shows a tour bar to step through the ten pages.")


def header(title: str, subtitle: str, step: Optional[int] = None) -> None:
    """Page title and one-sentence purpose (`step` is the page number, used by the interview-mode tour only)."""
    theme.inject_css()
    sidebar_controls()
    st.title(title)
    st.markdown(f'<div style="color:{theme.INK_SECONDARY}; font-size:1.02rem; margin-top:-0.4rem; margin-bottom:0.6rem;">{subtitle}</div>',
                unsafe_allow_html=True)


def fmt_p(p) -> str:
    """A p-value for display: '<0.001' for tiny ones, three decimals otherwise, a dash if missing."""
    import math
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def cards(items, ncols: int = 4, pre_formatted: bool = False) -> None:
    """Metric cards with whole numbers shown without decimals (see theme.metric_cards)."""
    if not pre_formatted:
        items = [(k, f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else (f"{v:,.2f}" if isinstance(v, float) else v)) for k, v in items]
    theme.metric_cards(items, ncols=ncols, pre_formatted=True)


def table(df, compact: bool = False, height=None, formats: Optional[dict] = None, wrap: bool = False) -> None:
    """A table that fits: numbers right-aligned with a sensible number of decimals, missing text as a dash, columns sized to their
    content (compact) or sharing the page width, and no scrolling inside the table unless a height is given."""
    import pandas as pd
    df = df.copy()
    cfg = {}
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_bool_dtype(col):
            df[c] = col.map({True: "yes", False: "no"}).fillna("–")
            cfg[c] = st.column_config.TextColumn(str(c))
        elif pd.api.types.is_numeric_dtype(col):
            fmt = (formats or {}).get(c)
            if fmt is None:
                vals = col.dropna()
                fmt = "%d" if len(vals) and (vals % 1 == 0).all() and vals.abs().max() < 1e9 else ("%.3g" if len(vals) and vals.abs().max() < 1e-2 else "%.2f")
            if col.isna().any():                              # a gap would show as 'None': write the numbers out and use a dash
                df[c] = [("–" if pd.isna(v) else (f"{v:,.0f}" if fmt == "%d" else fmt % v)) for v in col]
                cfg[c] = st.column_config.TextColumn(str(c))
            else:
                cfg[c] = st.column_config.NumberColumn(str(c), format=fmt)
        else:
            df[c] = col.astype(object).where(col.notna(), "–")
            cfg[c] = st.column_config.TextColumn(str(c))
    if height is None and len(df) <= 20:                      # show every row of a short table; longer ones scroll
        height = 38 + 35 * len(df)
    if wrap:                                                  # text-heavy tables: wrapped text, every row shown
        st.table(df.set_index(df.columns[0]))                 # the first column labels the rows
        return
    kw = {"height": height} if height else {}
    st.dataframe(df, hide_index=True, width="content" if compact else "stretch", column_config=cfg, **kw)


def not_yet_run(what: str) -> None:
    st.info(f"Not yet run: {what}. This page fills in as soon as the stage has produced its files.")


def badge(text: str, state: str = "done") -> str:
    c = STATE_COLOR.get(state, theme.STATUS_NEUTRAL)
    return (f'<span style="background:{c}22; color:{c}; border:1px solid {c}55; border-radius:999px; padding:2px 10px; '
            f'font-size:0.8rem; font-weight:600;">{text}</span>')


def target_picker(key: str = "ml_target", targets: Iterable[str] = ml_data.TARGETS, default: str = "da_price") -> str:
    """One selectbox whose choice is shared by all ML pages (session state)."""
    opts = list(targets)
    if key not in st.session_state or st.session_state[key] not in opts:
        st.session_state[key] = default if default in opts else opts[0]
    return st.selectbox("Target", opts, key=key, format_func=ml_data.label)


def tour_bar(step: int) -> None:
    """In interview mode: previous and next buttons through the ten pages."""
    if not interview_mode():
        return
    c1, c2, c3 = st.columns([1, 6, 1])
    for col, idx, text, icon in ((c1, step - 1, "Previous", "⬅️"), (c3, step + 1, "Next", "➡️")):
        with col:
            if 0 <= idx < len(PAGES):
                try:
                    st.page_link(f"pages/{PAGES[idx][0]}.py", label=text, icon=icon)
                except Exception:                                    # not inside the navigation (a single page run on its own)
                    st.caption(f"{text}: {PAGES[idx][1]}")
    with c2:
        st.caption("Interview mode: " + " > ".join(f"**{n}**" if i == step else n for i, (_, n) in enumerate(PAGES)))
