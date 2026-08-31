"""theme.py - single source of visual identity for the dashboard.

Every page imports colors and style_fig() from here instead of hardcoding
plotly colors ("steelblue", "darkorange", ...), so the seven pages read as
one system instead of independently-styled charts. Palette is the validated
categorical/status reference (fixed hue order, colorblind-safe adjacent
pairs) - see the dataviz skill's references/palette.md for the source.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Categorical palette - fixed order, never cycled/reassigned per-chart.
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

# Status palette - reserved for state (PASS/WARN/FAIL), never series identity.
STATUS_GOOD     = "#0ca30c"
STATUS_WARNING  = "#fab219"
STATUS_SERIOUS  = "#ec835a"
STATUS_CRITICAL = "#d03b3b"
STATUS_NEUTRAL  = "#898781"

STATUS_COLOR = {"PASS": STATUS_GOOD, "SKIP": STATUS_NEUTRAL, "WARN": STATUS_WARNING, "FAIL": STATUS_CRITICAL}
STATUS_ICON  = {"PASS": "\U0001F7E2", "SKIP": "⚪", "WARN": "\U0001F7E1", "FAIL": "\U0001F534"}

# Semantic roles - reused consistently across pages so "generation" and
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


def rgba(hex_color: str, alpha: float) -> str:
    """'#1baf7a' -> 'rgba(27,175,122,0.15)'. Plotly's fillcolor rejects
    8-digit hex-with-alpha, so callers needing a translucent fill (e.g. area
    charts) must go through this instead of string-concatenating hex."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


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
# front end remounts the page body each time - the browser has no reason to
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
        // Streamlit's actual scrolling element changed from "section.main"
        // to a data-testid-keyed one across versions -- selecting the wrong
        // one silently falls back to documentElement, which doesn't scroll
        // at all, so restore does nothing and every rerun snaps to top.
        // Try all of them, oldest-first, so this survives future renames too.
        var main = doc.querySelector('section.main')
            || doc.querySelector('[data-testid="stMain"]')
            || doc.querySelector('[data-testid="stAppViewContainer"]')
            || doc.documentElement;

        function restore() {
            var saved = sessionStorage.getItem(key);
            if (saved !== null) {
                main.scrollTop = parseFloat(saved);
            }
        }

        // Widgets stream into the DOM over the following second or two as
        // Streamlit renders each one -- a single scrollTop assignment right
        // now gets wiped out as soon as more content is appended below. A
        // fixed-interval poll can miss the exact moment that happens (or
        // stop too early on a slow-loading page), so watch for the DOM
        // actually changing instead and re-clamp scroll the instant it
        // does -- event-driven, not a guess at timing.
        var mo = new MutationObserver(function() { restore(); });
        mo.observe(main, { childList: true, subtree: true });
        var settleTimer = setTimeout(function() { mo.disconnect(); }, 3000);
        restore();  // also apply once immediately, in case nothing mutates after this

        // Saving on every 'scroll' event is what breaks this: a rerun's own
        // automatic scroll reset ALSO fires a 'scroll' event on this same
        // listener (still attached from the previous mount, since removing
        // an iframe doesn't retroactively unregister callbacks it already
        // attached to the parent document) -- so the reset can overwrite
        // the good saved position with the bad one before restore() above
        // ever gets a chance to reapply it. Debouncing the save doesn't fix
        // this either: the reset can land before the debounce timer fires,
        // so it ends up saving the ALREADY-reset value.
        //
        // 'wheel'/'touchmove' fire at the START of a genuine user gesture,
        // before the browser has necessarily applied the resulting scroll
        // yet -- so they mark "a real gesture is happening" rather than
        // supplying a trustworthy position themselves. 'scroll' fires AFTER
        // the position actually changes, so it's the only event with an
        // accurate value -- but only trust it while within a short window
        // of a real gesture; a 'scroll' event with no recent wheel/touch
        // behind it is a rerun's own reset, not the user, and gets ignored.
        var gestureUntil = 0;
        function markGesture() {
            gestureUntil = Date.now() + 400;
            // The user is genuinely scrolling right now -- stop reapplying
            // the OLD saved position, or the observer above fights the
            // user's own scroll (re-clamping back on every mutation it
            // causes), which reads as the page being stuck/unresponsive.
            mo.disconnect();
            clearTimeout(settleTimer);
        }
        // capture: true -- Plotly charts (and other embedded widgets) often
        // call stopPropagation() on wheel/touch events to handle their own
        // zoom/pan, which would otherwise stop this listener from ever
        // seeing a scroll that started over a chart. Capture-phase fires on
        // the way DOWN to the target, before any bubble-phase handler
        // (Plotly's included) gets a chance to stop it.
        main.addEventListener('wheel', markGesture, { passive: true, capture: true });
        main.addEventListener('touchmove', markGesture, { passive: true, capture: true });
        main.addEventListener('scroll', function() {
            if (Date.now() < gestureUntil) {
                sessionStorage.setItem(key, main.scrollTop);
            }
        }, { passive: true });
    } catch (e) { /* best-effort only */ }
})();
</script>
"""


def _auto_correct_refresh_toggle() -> None:
    """app.py already turns the auto-refresh toggle off the moment a live
    run finishes -- but only on a FULL script rerun, via a was-live/now-live
    transition it has to personally witness. A page sitting on its own
    @st.fragment(run_every=...) tick never re-executes app.py, so if the
    user stays on one page the whole time (the common case), app.py never
    gets another chance to observe that transition and the toggle keeps
    polling forever -- every tick briefly blanks and remounts each widget's
    iframe, which is the flicker this was built to fix.

    Deliberately NOT the same witnessed-transition check app.py uses --
    this runs from inside a fragment tick, which may be the first time this
    exact session ever evaluates liveness for this date (e.g. the run had
    already gone stale before this tab's first render), so there may be no
    prior state to compare against. Unconditional instead: if the toggle is
    on and the pipeline is not live, right now, turn it off. Costs the
    minor "leave it on to watch for a future run" use case app.py's
    docstring mentions, but that's a rare deliberate act the user can just
    redo -- an auto-refresh that silently never turns itself off is the
    worse failure mode."""
    selected_date = st.session_state.get("selected_date")
    if not selected_date or not st.session_state.get("auto_refresh_toggle"):
        return
    import data  # local import: theme.py must stay import-cycle-free of data.py
    # data.is_pipeline_active() is file-based (checks the log on disk) and
    # lags a subprocess that was JUST launched by runner.start() in this
    # same session - Popen() returns before the child has written its first
    # line, so this watchdog was winning a race against start()'s own
    # auto_refresh_toggle=True and immediately flipping it back off, every
    # time, before the file ever caught up. session_state.runner["running"]
    # is set synchronously by start() with no file round-trip, so OR it in
    # as an immediate signal this session already knows about.
    runner_state = st.session_state.get("runner") or {}
    session_knows_its_running = bool(runner_state.get("running"))
    if not data.is_pipeline_active(selected_date) and not session_knows_its_running:
        st.session_state["auto_refresh_toggle"] = False
        st.rerun(scope="app")


def inject_scroll_restore() -> None:
    """Best-effort: keeps scroll position stable across autorefresh reruns.
    Call on every rerun -- both the full-script one in app.py AND once more
    inside each page's own @st.fragment(run_every=...) body, since a
    fragment's periodic tick reruns only that fragment, not app.py.

    components.html() renders an iframe from a literal HTML string; when
    that string is byte-identical to the previous call (as it always was
    here), Streamlit's frontend reuses the existing iframe node instead of
    remounting it, so the <script> inside never re-executes after the
    first time. That's harmless for a full-script rerun (the whole DOM,
    iframe included, gets replaced anyway) but silently no-ops on a
    fragment's own run_every tick, which patches only the fragment's
    subtree -- so nothing was left to counter Streamlit's own scroll reset
    on that tick, and the page would jump back on a timer even though a
    user's own click looked perfectly fixed. A trailing HTML comment with
    a fresh timestamp makes each call's content different, forcing a real
    remount (and a real script re-run) every single time."""
    _auto_correct_refresh_toggle()
    import time
    import streamlit.components.v1 as components
    components.html(_SCROLL_RESTORE_JS + f"<!-- {time.time()} -->", height=0)


def fmt_metric(value: object) -> str:
    """Whole-number floats (day counts, pass counts) print as plain
    integers -- '180.00' reads like a measurement with false precision,
    '180' reads like the count it actually is. Genuine decimals keep 2dp."""
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def metric_cards(items: list[tuple[str, object]], ncols: int = 4, pre_formatted: bool = False) -> None:
    """Custom wrap-text cards instead of st.metric -- st.metric truncates
    both long labels ('VaR(95%) - 5th-pct P&L (...') and long value strings
    ('1,015,15...') with an ellipsis once a card gets narrow, silently
    hiding the actual number. Wrapping instead of truncating costs a little
    vertical space but never loses information.

    pre_formatted=True skips fmt_metric() -- pass already-formatted strings
    (e.g. "1,015,154 EUR") when the value needs a unit suffix or other
    formatting fmt_metric doesn't handle."""
    import streamlit as st
    cols = st.columns(min(ncols, len(items)))
    for i, (label, value) in enumerate(items):
        with cols[i % len(cols)]:
            display = value if pre_formatted else fmt_metric(value)
            st.markdown(
                f'<div style="background:{SURFACE}; border:1px solid {GRIDLINE}; '
                f'border-radius:10px; padding:0.7rem 0.9rem; margin-bottom:8px; min-height:84px;">'
                f'<div style="font-size:12.5px; line-height:1.35; color:{INK_SECONDARY}; '
                f'margin-bottom:4px;">{label}</div>'
                f'<div style="font-size:22px; font-weight:500; color:{INK_PRIMARY}; '
                f'word-break:break-word;">{display}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def auto_refresh_interval() -> int | None:
    """Seconds between auto-reruns for the CALLING page's own
    @st.fragment(run_every=...), or None to disable. Set once per full
    script run by app.py's sidebar (auto-refresh toggle + interval), read
    fresh by each page every time it's re-entered -- see app.py for why
    this replaced streamlit_autorefresh's whole-script reruns."""
    return st.session_state.get("_auto_refresh_interval_s")
