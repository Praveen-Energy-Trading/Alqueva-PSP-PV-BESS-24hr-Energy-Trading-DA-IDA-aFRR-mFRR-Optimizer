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

PAGES = [("ml_1_data", "Data and quality"), ("ml_2_market", "Market analysis"),
         ("ml_3_formulation", "Problem formulation"), ("ml_4_features", "Features"), ("ml_5_evaluation", "Evaluation design"),
         ("ml_6_tuning", "Tuning lab"), ("ml_7_arena", "Model arena"), ("ml_8_decision", "Decision and locked test"),
         ("ml_9_run", "Run it: monitoring, live"), ("ml_10_plant_inputs", "Plant inputs: PV and inflow")]

STATE_COLOR = {"done": theme.STATUS_GOOD, "partial": theme.STATUS_WARNING, "not run": theme.STATUS_NEUTRAL}


def cached(fn):
    """Cache a data-layer function, for at most a minute: the pack and result files change when a stage is re-run or a commit is pulled,
    and the page must then show the new numbers without restarting the server."""
    return st.cache_data(show_spinner=False, ttl=60)(fn)


def interview_mode() -> bool:
    return bool(st.session_state.get("ml_interview_mode", False))


def sidebar_controls() -> None:
    st.sidebar.toggle("Interview mode", key="ml_interview_mode",
                      help="Shows a tour bar to step through the pages.")


def header(title: str, subtitle: str, step: Optional[int] = None) -> None:
    """Page title and one-sentence purpose (`step` is the page number, used by the interview-mode tour only)."""
    theme.inject_css()
    sidebar_controls()
    st.title(title)
    if subtitle:
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


def table(df, compact: bool = False, height=None, formats: Optional[dict] = None, wrap: bool = False, fit: bool = True, stretch: bool = False, nowrap: bool = False) -> None:
    """A table whose columns are as wide as their content and whose headers wrap onto two or more lines (see html_table.py). Numbers are
    right-aligned with a sensible number of decimals, a missing value is a dash; a table of more than 20 rows scrolls inside a frame, a
    shorter one shows every row. `stretch` makes the table as wide as the page. `compact`, `wrap` and `fit` are kept for the callers and no longer change anything."""
    import html_table
    if height is None and len(df) > 20:
        height = 520
    st.markdown(html_table.html_table(df, formats=formats, height=height, stretch=stretch, nowrap=nowrap), unsafe_allow_html=True)


def table_halves(df, **kwargs) -> None:
    """A long table as two halves side by side (less height, the width is used); each half is as wide as its column."""
    half = (len(df) + 1) // 2
    left, right = st.columns(2)
    with left:
        table(df.iloc[:half], stretch=True, **kwargs)
    with right:
        table(df.iloc[half:], stretch=True, **kwargs)


def figure(path) -> None:
    """Show a figure at the full width of the page: the SVG next to a PNG if there is one (sharp at any zoom), else the PNG itself."""
    import base64
    from pathlib import Path
    p = Path(path)
    svg = p.with_suffix(".svg")
    if svg.exists():
        data = base64.b64encode(svg.read_bytes()).decode("ascii")
        # Streamlit sets object-fit: scale-down on images, which never enlarges an SVG beyond its natural size; fill the box instead
        st.markdown('<style>img[src^="data:image/svg+xml"] { object-fit: fill !important; }</style>'
                    f'<img src="data:image/svg+xml;base64,{data}" width="100%" alt="{p.stem}">', unsafe_allow_html=True)
    else:
        st.image(str(p), width="stretch")


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
    """In interview mode: previous and next buttons through the pages."""
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
