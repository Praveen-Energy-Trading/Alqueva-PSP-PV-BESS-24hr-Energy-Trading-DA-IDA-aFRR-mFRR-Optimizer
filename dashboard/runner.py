"""
runner.py — in-app pipeline launcher.

Launches run_production.py as a subprocess and streams its stdout live into
the page. A background thread does the blocking readline() loop (Streamlit
reruns the whole script on every interaction, so nothing here can block the
main thread); the thread just appends to a plain list living in
st.session_state, which the next rerun reads and renders. --auto is
required (no interactive A/R or ENTER prompts — those need a real terminal,
which a subprocess pipe can't drive cleanly).

Only used by pages/1_run_monitor.py, but state init/poll must happen in
app.py (the entry point that runs on every navigation) so a run started on
one page keeps streaming correctly if the user switches pages mid-run.
"""
from __future__ import annotations

import subprocess
import sys
import threading

import streamlit as st

from data import REPO_ROOT


def init_state() -> None:
    if "runner" not in st.session_state:
        st.session_state.runner = {
            "proc": None, "thread": None, "lines": [], "running": False,
            "date": None, "return_code": None,
        }


def _reader_thread(proc: subprocess.Popen, lines: list[str]) -> None:
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
    proc.stdout.close()


def start(date_arg: str, synthetic: bool) -> None:
    runner_state = st.session_state.runner
    if runner_state["running"]:
        return
    # -u: unbuffered stdout. Without it, Python fully block-buffers stdout
    # when it isn't a TTY (i.e. whenever it's piped, as here) — the live
    # console would just show "waiting for output..." until the process
    # exited and flushed everything at once.
    cmd = [sys.executable, "-u", str(REPO_ROOT / "run_production.py"), "--date", date_arg, "--auto"]
    if synthetic:
        cmd.append("--synthetic")
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines: list[str] = []
    thread = threading.Thread(target=_reader_thread, args=(proc, lines), daemon=True)
    thread.start()
    st.session_state.runner = {
        "proc": proc, "thread": thread, "lines": lines, "running": True,
        "date": date_arg, "return_code": None,
    }
    st.session_state["auto_refresh_toggle"] = True  # so the console streams without an extra click


def stop() -> None:
    runner_state = st.session_state.runner
    if runner_state["proc"] is not None and runner_state["running"]:
        runner_state["proc"].terminate()


def poll() -> None:
    runner_state = st.session_state.runner
    if runner_state["proc"] is not None and runner_state["running"]:
        rc = runner_state["proc"].poll()
        if rc is not None:
            runner_state["running"] = False
            runner_state["return_code"] = rc
            st.cache_data.clear()  # new report/log/status on disk — drop stale cache
