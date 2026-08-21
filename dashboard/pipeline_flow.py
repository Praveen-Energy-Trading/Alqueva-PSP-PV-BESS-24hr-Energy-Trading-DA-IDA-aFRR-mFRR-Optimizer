"""pipeline_flow.py — a faithful, real-data trace of run_production.py's
actual 19-phase execution, not an illustration of it. The phase list below
mirrors run_production.py's own _PHASES table exactly (key, label, group) --
same keys, same order, same grouping by phase-number prefix (1-3, 4, 5, 6)
that the real code already uses. Per-phase status/detail/elapsed comes
straight from that day's real run_status_<date>.json (data.py::
load_run_status), the same file Run & Monitor's health banner reads --
nothing here is invented or re-simulated.
"""
from __future__ import annotations

import html as _html
import json as _json
import re as _re

import theme

# Matches the real structured log line format written by _Tee/get_logger
# throughout the pipeline: "2026-08-20 18:45:21 | INFO    | phase1.da |
# Solved in 2.48s | energy revenue 486,852.69 EUR". Everything else in the
# log (bid tables, raw print() output) is real too, but isn't a discrete
# "the program just did X" milestone -- deliberately not matched here, see
# render_log_trace_card's docstring for why.
_LOG_LINE_RE = _re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*(\w+)\s*\|\s*([\w.]+)\s*\|\s*(.*)$"
)


def _log_line_icon(level: str, message: str) -> str:
    """Best-effort icon per real log line, from its real level + content --
    not a classification the pipeline itself makes, just a visual cue over
    real text."""
    if level == "ERROR":
        return "&#128308;"
    if level == "WARNING":
        return "&#128993;"
    msg = message.lower()
    if "solved" in msg:
        return "&#129518;"
    if "revenue" in msg or "capacity revenue" in msg:
        return "&#128176;"
    if "submitted" in msg or "saved (stub submit)" in msg:
        return "&#128228;"
    if "no re-bid" in msg or "no_order" in msg or "no order" in msg:
        return "&#9208;"
    if "done" in msg or "solved" in msg:
        return "&#9989;"
    if "tradable" in msg:
        return "&#128269;"
    return "&#9654;"


def parse_log_trace(log_text: str) -> list[dict]:
    """Extracts every real structured milestone line from the pipeline's
    actual stdout log (data.py::load_log), in real order. Returns [] for an
    empty/missing log -- never fabricates lines that weren't actually
    printed."""
    rows = []
    for line in log_text.splitlines():
        m = _LOG_LINE_RE.match(line)
        if not m:
            continue
        ts, level, tag, message = m.groups()
        rows.append({"ts": ts, "level": level, "tag": tag, "message": message})
    return rows


def render_log_trace_card(log_text: str) -> str:
    """Typewriter-reveal animation of the real pipeline log's structured
    milestone lines (see parse_log_trace) -- real timestamps, real phase
    tags, real messages, straight from that day's actual stdout. The other
    ~97% of the raw log (bid tables, per-ISP print output) is real too but
    is data, not a narrative milestone -- included as a raw-log toggle
    instead of auto-animated, since scrolling through 96 bid rows one at a
    time teaches nothing a chart doesn't already show better."""
    rows = parse_log_trace(log_text)
    if not rows:
        return f'''<div style="font-family:system-ui,-apple-system,'Segoe UI',sans-serif; padding:2rem; text-align:center; color:{theme.INK_MUTED};">
        No structured log lines found for this date yet. Run the pipeline, then Refresh.</div>'''

    rows_json = _json.dumps(rows)
    line_html = "".join(
        f'''<div class="lt-line" style="display:flex; gap:10px; padding:4px 0; opacity:0;">
          <span style="font-size:13px; flex-shrink:0;">{_log_line_icon(r["level"], r["message"])}</span>
          <span style="font-size:10.5px; color:{theme.INK_MUTED}; flex-shrink:0; font-family:monospace;">{_html.escape(r["ts"][11:])}</span>
          <span style="font-size:10.5px; color:{theme.INK_SECONDARY}; flex-shrink:0; font-family:monospace; min-width:120px;">{_html.escape(r["tag"])}</span>
          <span style="font-size:11.5px; color:{theme.INK_PRIMARY};">{_html.escape(r["message"])}</span>
        </div>'''
        for r in rows
    )

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Live trace</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:11.5px; padding:3px 10px; border-radius:6px; font-weight:500;">Real log lines, real order</span>
    </div>
    <div style="font-size:18px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:8px;">Pipeline execution, line by line</div>
    <div id="lt-scroll" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE}; border-radius:8px; padding:10px 14px; height:340px; overflow-y:auto;">
      {line_html}
    </div>
    <div style="display:flex; justify-content:flex-end; margin-top:10px;">
      <button class="gt-replay" id="lt-play-btn">&#9654; Replay</button>
    </div>
  </div>
</div>
<style>.gt-replay {{ font-size:12px; padding:4px 12px; border:1px solid {theme.GRIDLINE}; border-radius:6px; background:{theme.SURFACE}; cursor:pointer; }}</style>
<script>
(function() {{
  var lines = Array.from(document.querySelectorAll('.lt-line'));
  var scrollBox = document.getElementById('lt-scroll');
  var btn = document.getElementById('lt-play-btn');
  function reveal() {{
    btn.disabled = true;
    lines.forEach(function(l) {{ l.style.opacity = '0'; }});
    scrollBox.scrollTop = 0;
    var i = 0;
    function step() {{
      if (i >= lines.length) {{ btn.disabled = false; return; }}
      lines[i].style.transition = 'opacity 0.25s';
      lines[i].style.opacity = '1';
      scrollBox.scrollTop = lines[i].offsetTop - scrollBox.clientHeight + 60;
      i++;
      setTimeout(step, 260);
    }}
    step();
  }}
  btn.addEventListener('click', reveal);
  reveal();
}})();
</script>'''

# Mirrors run_production.py's _PHASES table (key, label) grouped into the
# same 4 numbered stages the real phase-number prefixes already imply
# (1-3, 4A-4D, 5A-5C, 6A-6D) -- not a simplification invented for this
# widget, just the code's own structure made visible.
_GROUPS = [
    ("Day-ahead and capacity decisions", [
        ("da", "Day-ahead bidding (OMIE DA)"),
        ("afrr", "aFRR capacity offer (PICASSO/REN)"),
        ("mfrr", "mFRR capacity offer (MARI)"),
    ]),
    ("Intraday re-optimization", [
        ("ida1", "IDA1 intraday re-optimisation"),
        ("ida2", "IDA2 intraday re-optimisation"),
        ("ida3", "IDA3 intraday re-optimisation"),
        ("xbid_w1", "XBID continuous (D-1 18:30)"),
        ("xbid_w2", "XBID continuous (D-1 22:30)"),
        ("xbid_w3", "XBID continuous (D 03:00)"),
        ("xbid_w4", "XBID continuous (D 06:00)"),
        ("xbid_w5", "XBID continuous (D 09:30)"),
        ("xbid_w6", "XBID continuous (D 12:00)"),
    ]),
    ("Real-time delivery and activation", [
        ("realtime", "RT dispatch simulation (96 ISPs)"),
        ("afrr_activation", "aFRR activation response"),
        ("mfrr_activation", "mFRR activation response"),
    ]),
    ("Settlement and reporting", [
        ("energy_settlement", "Energy settlement (DA / IDA)"),
        ("reserve_settlement", "Reserve settlement (aFRR / mFRR)"),
        ("imbalance_settlement", "Imbalance settlement (REN balance)"),
        ("analytics", "Analytics + KPI report + Excel"),
    ]),
]
_GROUP_ICON = {"Day-ahead and capacity decisions": "&#128200;", "Intraday re-optimization": "&#128260;",
               "Real-time delivery and activation": "&#9889;", "Settlement and reporting": "&#9878;"}


def render_pipeline_flow_card(run_status: dict | None, delivery_date: str) -> str:
    """Scrolling, credits-style trace of the real pipeline run for
    delivery_date. Every phase shown is real (from run_production.py's own
    _PHASES table); every status/detail/elapsed is real (from that day's
    run_status_<date>.json, when it exists). If run_status is None (no
    completed run recorded for this date), every phase renders as
    'not run' rather than fabricating a status."""
    results_by_key = {}
    if run_status and run_status.get("results"):
        results_by_key = {r["key"]: r for r in run_status["results"]}

    group_blocks = []
    for group_name, phases in _GROUPS:
        rows_html = []
        n_pass = n_total = 0
        for key, label in phases:
            n_total += 1
            r = results_by_key.get(key)
            if r is None:
                status, detail, elapsed = "SKIP", "not recorded for this run", None
            else:
                status, detail, elapsed = r["status"], r.get("detail", ""), r.get("elapsed")
            if status == "PASS":
                n_pass += 1
            color = theme.STATUS_COLOR.get(status, theme.INK_MUTED)
            icon = theme.STATUS_ICON.get(status, "&#9899;")
            elapsed_txt = f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else "&mdash;"
            rows_html.append(f'''
            <div style="display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid {theme.GRIDLINE};">
              <span style="font-size:11.5px; color:{theme.INK_PRIMARY};">{icon} {_html.escape(label)}</span>
              <span style="font-size:10.5px; color:{color}; font-weight:500; white-space:nowrap;">{status} &middot; {elapsed_txt}{(' &middot; ' + _html.escape(str(detail)[:40])) if detail else ''}</span>
            </div>''')
        group_blocks.append(f'''
        <div class="stage" style="padding:22px 4px;">
          <div style="text-align:center; margin-bottom:10px;">
            <div style="font-size:20px;">{_GROUP_ICON[group_name]}</div>
            <div style="font-size:14px; font-weight:500; color:{theme.INK_PRIMARY}; margin-top:4px;">{_html.escape(group_name)}</div>
            <div class="stagenum" style="font-size:10.5px; color:{theme.INK_MUTED};">{n_pass}/{n_total} passed</div>
          </div>
          <div style="max-width:460px; margin:0 auto;">{''.join(rows_html)}</div>
        </div>
        <svg width="20" height="40" style="display:block; margin:0 auto;"><line x1="10" y1="0" x2="10" y2="40" stroke="{theme.GRIDLINE}" stroke-width="2" stroke-dasharray="4,5" class="pf-flow"/></svg>''')

    # Drop the trailing connector after the last group.
    reel_inner = "".join(group_blocks)
    if reel_inner.count("<svg") > 0:
        last_svg = reel_inner.rfind("<svg")
        reel_inner = reel_inner[:last_svg]

    run_meta = ""
    if run_status:
        n_pass_total = sum(1 for r in run_status.get("results", []) if r["status"] == "PASS")
        n_total_total = len(run_status.get("results", []))
        run_meta = (f"{n_pass_total}/{n_total_total} phases passed &middot; "
                    f"mode={_html.escape(str(run_status.get('mode', '?')))} &middot; "
                    f"finished {_html.escape(str(run_status.get('finished_at', '?')))}")
    else:
        run_meta = "No completed run recorded for this date yet."

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div style="background:#2C2C2A; border-radius:16px; padding:14px 14px 22px;">
    <div style="display:flex; gap:6px; margin-bottom:10px;">
      <div style="width:8px; height:8px; border-radius:50%; background:#E24B4A;"></div>
      <div style="width:8px; height:8px; border-radius:50%; background:#EF9F27;"></div>
      <div style="width:8px; height:8px; border-radius:50%; background:#639922;"></div>
    </div>
    <div id="pf-screen" style="background:{theme.SURFACE}; border-radius:8px; height:520px; overflow:hidden; position:relative;">
      <div id="pf-reel" style="position:absolute; left:0; right:0; top:0; padding:20px 18px; transition:transform 1.1s cubic-bezier(0.4,0,0.2,1);">
        {reel_inner}
        <div style="height:60px;"></div>
      </div>
    </div>
  </div>
  <div style="display:flex; align-items:center; justify-content:space-between; margin-top:10px;">
    <span style="font-size:11.5px; color:{theme.INK_SECONDARY};">{run_meta}</span>
    <button class="gt-replay" id="pf-play-btn">&#9654; Replay</button>
  </div>
</div>
<style>.gt-replay {{ font-size:12px; padding:4px 12px; border:1px solid {theme.GRIDLINE}; border-radius:6px; background:{theme.SURFACE}; cursor:pointer; }}</style>
<script>
(function() {{
  var reel = document.getElementById('pf-reel');
  var stages = Array.from(document.querySelectorAll('.stage'));
  var flows = Array.from(document.querySelectorAll('.pf-flow'));
  var btn = document.getElementById('pf-play-btn');
  var dash = 0;
  setInterval(function() {{
    dash = (dash - 1) % 18;
    flows.forEach(function(f) {{ f.setAttribute('stroke-dashoffset', dash); }});
  }}, 60);
  btn.addEventListener('click', function() {{
    btn.disabled = true;
    reel.style.transition = 'none';
    reel.style.transform = 'translateY(0px)';
    void reel.offsetHeight;
    reel.style.transition = 'transform 1.1s cubic-bezier(0.4,0,0.2,1)';
    var offsets = stages.map(function(s) {{ return s.offsetTop - 30; }});
    var i = 0;
    function step() {{
      if (i >= offsets.length) {{ btn.disabled = false; return; }}
      reel.style.transform = 'translateY(-' + offsets[i] + 'px)';
      setTimeout(function() {{ i++; step(); }}, 1650);
    }}
    step();
  }});
}})();
</script>'''
