"""theme.py — single source of visual identity for the dashboard.

Every page imports colors and style_fig() from here instead of hardcoding
plotly colors ("steelblue", "darkorange", ...), so the seven pages read as
one system instead of independently-styled charts. Palette is the validated
categorical/status reference (fixed hue order, colorblind-safe adjacent
pairs) — see the dataviz skill's references/palette.md for the source.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Categorical palette — fixed order, never cycled/reassigned per-chart.
# ---------------------------------------------------------------------------
BLUE    = "#2a78d6"
ORANGE  = "#eb6834"
AQUA    = "#1baf7a"
YELLOW  = "#eda100"
MAGENTA = "#e87ba4"
GREEN   = "#008300"
VIOLET  = "#4a3aa7"
RED     = "#e34948"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# Status palette — reserved for state (PASS/WARN/FAIL), never series identity.
STATUS_GOOD     = "#0ca30c"
STATUS_WARNING  = "#fab219"
STATUS_SERIOUS  = "#ec835a"
STATUS_CRITICAL = "#d03b3b"
STATUS_NEUTRAL  = "#898781"

STATUS_COLOR = {"PASS": STATUS_GOOD, "SKIP": STATUS_NEUTRAL, "WARN": STATUS_WARNING, "FAIL": STATUS_CRITICAL}
STATUS_ICON  = {"PASS": "\U0001F7E2", "SKIP": "⚪", "WARN": "\U0001F7E1", "FAIL": "\U0001F534"}

# Semantic roles — reused consistently across pages so "generation" and
# "pumping" (or aFRR/mFRR, up/down) always mean the same color everywhere in
# the app, not just within one chart.
COLOR_GEN   = BLUE     # generation / turbine / DA
COLOR_PUMP  = ORANGE   # pumping / mFRR / IDA
COLOR_PRICE = VIOLET   # price overlays
COLOR_UP    = AQUA     # up-regulation
COLOR_DOWN  = MAGENTA  # down-regulation

# ---------------------------------------------------------------------------
# Chart chrome (matches .streamlit/config.toml light theme)
# ---------------------------------------------------------------------------
INK_PRIMARY   = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED     = "#898781"
GRIDLINE      = "#e1e0d9"
SURFACE       = "#fcfcfb"

FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def style_fig(fig: go.Figure, *, height: int = 380, yaxis_title: str | None = None,
              xaxis_title: str | None = None, legend: bool = True,
              barmode: str | None = None) -> go.Figure:
    """Apply the shared chart chrome. Call last, after adding traces."""
    layout_kwargs = dict(
        height=height,
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family=FONT_FAMILY, color=INK_PRIMARY),
        showlegend=legend,
    )
    if legend:
        layout_kwargs["legend"] = dict(orientation="h", y=-0.18, font=dict(color=INK_SECONDARY))
    if barmode:
        layout_kwargs["barmode"] = barmode
    fig.update_layout(**layout_kwargs)
    fig.update_yaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, tickfont=dict(color=INK_MUTED),
                      title_text=yaxis_title, title_font=dict(color=INK_SECONDARY))
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK_MUTED),
                      title_text=xaxis_title, title_font=dict(color=INK_SECONDARY))
    return fig


CSS = f"""
<style>
    .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}
    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {GRIDLINE};
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }}
    [data-testid="stMetricLabel"] {{ color: {INK_SECONDARY}; font-weight: 500; font-size: 0.85rem; }}
    [data-testid="stMetricValue"] {{ color: {INK_PRIMARY}; }}
    h1, h2, h3 {{ letter-spacing: -0.01em; }}
    h1 {{ font-weight: 700; }}
    hr {{ margin: 1.7rem 0; border-color: {GRIDLINE}; }}
    [data-testid="stSidebar"] {{ border-right: 1px solid {GRIDLINE}; }}
    [data-testid="stSidebar"] h1 {{ font-size: 1.25rem; }}
    div[data-testid="stExpander"] {{ border: 1px solid {GRIDLINE}; border-radius: 10px; }}
    code {{ color: {COLOR_GEN}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# Streamlit reruns the whole script on every autorefresh tick, and the
# front end remounts the page body each time — the browser has no reason to
# keep scroll position across that, so a live-refreshing page keeps jumping
# back to the top mid-read. There's no first-class Streamlit API for this;
# the standard workaround is a components.v1.html iframe that reaches into
# the parent document (same-origin, so this is allowed) to save scrollTop on
# every scroll event and restore it right after each rerender.
_SCROLL_RESTORE_JS = """
<script>
(function() {
    try {
        var doc = window.parent.document;
        var key = "alqueva_dash_scroll_y";
        var main = doc.querySelector('section.main') || doc.documentElement;
        var saved = sessionStorage.getItem(key);
        if (saved !== null) {
            main.scrollTop = parseFloat(saved);
        }
        main.addEventListener('scroll', function() {
            sessionStorage.setItem(key, main.scrollTop);
        }, { passive: true });
    } catch (e) { /* best-effort only */ }
})();
</script>
"""


def inject_scroll_restore() -> None:
    """Best-effort: keeps scroll position stable across autorefresh reruns.
    Call once per page, after inject_css()."""
    import streamlit.components.v1 as components
    components.html(_SCROLL_RESTORE_JS, height=0)
