"""
Alqueva PSP-PV-BESS Trading Dashboard — entry point.

Multipage app organised by audience/purpose (health -> money -> decision
rationale -> risk/constraints -> backtest -> raw evidence), not by which
Excel sheet a number happens to live in. Every page reads through
dashboard/data.py, which mirrors whatever run_production.py has already
written to runtime/reports, runtime/logs and runtime/audit — this app never
runs pipeline logic itself except via the explicit in-app runner on the
Run & Monitor page.

Launch:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from streamlit_autorefresh import st_autorefresh

import data
import runner
import theme

st.set_page_config(
    page_title="Alqueva PSP-PV-BESS Trading Dashboard",
    page_icon="⚡",
    layout="wide",
)
theme.inject_css()
theme.inject_scroll_restore()

runner.init_state()
runner.poll()  # before we list dates below, so a just-finished run's report is picked up this same rerun

# ---------------------------------------------------------------------------
# Shared sidebar — date selection + refresh, used by every page
# ---------------------------------------------------------------------------

st.sidebar.title("⚡ Alqueva Trading Pipeline")
st.sidebar.caption("PSP + PV + BESS — DA / IDA / XBID / aFRR / mFRR")

if st.sidebar.button("🔄 Refresh now", width="stretch"):
    st.cache_data.clear()

st.sidebar.markdown("---")

dates, report_dates = data.available_dates()
selected_date = None
if not dates:
    st.sidebar.warning("No runs found yet.")
    st.session_state["selected_date"] = None
else:
    def _label(d: str) -> str:
        if d in report_dates:
            return d
        # No Excel report doesn't necessarily mean "still running" — a run
        # can finish (with a failure) before ever reaching the analytics
        # phase that writes the report. run_phase_state distinguishes a
        # genuinely in-progress run (including one paused on interactive
        # input) from a stale completed result left by a previous run for
        # this same date, so the label never claims something stale is live
        # or something live is stale.
        state = data.run_phase_state(d)
        if state == "running":
            return f"{d}  (running live)"
        if state == "idle_running":
            return f"{d}  (running — paused for input?)"
        if state == "stopped":
            return f"{d}  (stopped — Ctrl+C or crash)"
        if state == "none":
            return f"{d}  (no run record)"
        status = data.load_run_status(d)
        n_fail = sum(1 for r in status["results"] if r["status"] == "FAIL")
        if n_fail:
            return f"{d}  (finished — FAILED, no report)"
        return f"{d}  (finished — no report)"

    selected_date = st.sidebar.selectbox(
        "Delivery date", dates, index=0, format_func=_label, key="date_select",
    )
    st.session_state["selected_date"] = selected_date
    st.session_state["report_ready"] = selected_date in report_dates

# is_pipeline_active checks the log file's own mtime — source-agnostic, so a
# VS Code run, a plain terminal run, or the in-app runner ALL count as
# "live" the same way. This is what makes the dashboard an actual mirror of
# the terminal rather than something that only catches up once a run
# finishes: whenever the log is moving, the dashboard polls fast too,
# regardless of who started the process.
pipeline_live = bool(selected_date) and data.is_pipeline_active(selected_date)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.toggle(
    "⏱️ Auto-refresh while pipeline runs", key="auto_refresh_toggle",
    value=True,  # a fresh tab should live-follow whatever's running, no extra click
    help="Polls the report/log/audit files on disk and reloads whatever "
         "changed — on by default. Turn off for a static snapshot.",
)
if auto_refresh:
    if pipeline_live:
        interval_s = 2
        st.sidebar.success("🟢 LIVE — mirroring an active run (any source: "
                            "VS Code, terminal, or this dashboard)")
    else:
        interval_s = st.sidebar.slider("Refresh interval (seconds)", 3, 30, 5)
    st_autorefresh(interval=interval_s * 1000, key="pipeline_autorefresh")
    st.sidebar.caption(f"Auto-refreshing every {interval_s}s — mtime-based, "
                        f"so only changed files are actually reloaded.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Reads run_production.py's saved outputs — this app doesn't run trades "
    "on its own except via the explicit Run & Monitor page."
)

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

pages = {
    "Operate": [
        st.Page("pages/0_overview.py", title="Overview", icon="⚡", default=True),
        st.Page("pages/1_run_monitor.py", title="Run & Monitor", icon="🚀"),
        st.Page("pages/2_trading_desk.py", title="Trading Desk", icon="💰"),
        st.Page("pages/3_decision_rationale.py", title="Decision Rationale", icon="🧭"),
        st.Page("pages/4_risk_constraints.py", title="Risk & Constraints", icon="⚠️"),
    ],
    "Analyse": [
        st.Page("pages/5_backtest_risk.py", title="Backtest & Portfolio Risk", icon="📈"),
        st.Page("pages/6_figures.py", title="Figures", icon="🖼️"),
        st.Page("pages/7_console_log.py", title="Console Log", icon="🖥️"),
    ],
}
pg = st.navigation(pages)
pg.run()
