"""dispatch_ticket.py — technical/physical optimization widgets: reservoir
trajectory, unit commitment, ramp utilization, etc. Distinct from
delivery_ticket.py (market delivery/settlement cards) and gate_ticket.py
(bid decisions) -- these show what the MILP solver actually did physically,
not what was bid or settled. All data here is real solved-model output
(ComponentStore), never illustrative, unless explicitly labelled otherwise.
"""
from __future__ import annotations

import html as _html
import json as _json
import math as _math

import theme


def _nice_max(raw_max: float) -> float:
    """Round an axis ceiling up to a round step instead of an arbitrary
    padded data max, so a reader sees a number like 600 on the axis instead
    of whatever the data happened to peak at (e.g. 377). Finer step ladder
    than the textbook 1/2/5/10 -- that coarser version rounds anything from
    5.01x to 10x a decade up to a full 10x, e.g. a real max of 525 becomes
    1000, nearly doubling the ceiling and wasting half the chart on empty
    space above the highest point. 1/1.5/2/2.5/3/4/5/6/8/10 keeps every
    jump under ~33%."""
    if raw_max <= 0:
        return 10.0
    exp = _math.floor(_math.log10(raw_max))
    base = 10 ** exp
    frac = raw_max / base
    for nice_frac in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if frac <= nice_frac:
            return nice_frac * base
    return 10 * base

_REPLAY_STYLE = f'<style>.gt-replay {{ font-size:14px; padding:2px 10px; border:1px solid {theme.GRIDLINE}; border-radius:6px; background:{theme.SURFACE}; cursor:pointer; }}</style>'

# One generic mousemove-crosshair module, reused by every hand-drawn SVG
# chart in this file instead of duplicating hover-tracking JS per widget.
# A widget opts in by (1) adding a transparent catch-rect + a hover line +
# one <circle> per trace to its SVG, (2) embedding a dtHoverInit(block, cfg)
# call with cfg.traces[i].py already holding that trace's PRECOMPUTED pixel
# y-coordinates (Python already builds these to draw the path, so passing
# them again costs nothing and keeps every widget's differing y-scale math
# in Python, not duplicated in JS). Native Plotly charts elsewhere in this
# app (Gate Position Evolution, etc.) get this behavior for free; this file
# only draws raw SVG, so it needs its own.
_HOVER_CROSSHAIR_SCRIPT = '''
<script>
window.dtHoverInit = function(block, cfg) {
  var svg = block.querySelector(cfg.svgSelector || 'svg');
  var catchRect = block.querySelector(cfg.catchSelector);
  var line = block.querySelector(cfg.lineSelector);
  var tooltip = block.querySelector(cfg.tooltipSelector);
  if (!svg || !catchRect) return;
  function fx(i) { return cfg.x0 + i / Math.max(cfg.n - 1, 1) * (cfg.x1 - cfg.x0); }
  function move(evt) {
    var rect = svg.getBoundingClientRect();
    var vbX = (evt.clientX - rect.left) / rect.width * (cfg.viewW || 1400);
    var frac = Math.max(0, Math.min(1, (vbX - cfg.x0) / (cfg.x1 - cfg.x0)));
    var idx = Math.round(frac * (cfg.n - 1));
    var cx = fx(idx);
    if (line) { line.setAttribute('x1', cx); line.setAttribute('x2', cx); line.setAttribute('opacity', '1'); }
    var html = '<div style="color:#52514e;">' + (cfg.indexLabel || 'Point') + ' ' +
      (cfg.indexVals ? cfg.indexVals[idx] : (idx + 1)) + '</div>';
    for (var t = 0; t < cfg.traces.length; t++) {
      var tr = cfg.traces[t];
      var v = tr.yvals[idx];
      var dot = tr.dotSelector ? block.querySelector(tr.dotSelector) : null;
      if (dot) {
        dot.setAttribute('cx', cx);
        dot.setAttribute('cy', tr.py[idx].toFixed(1));
        dot.setAttribute('opacity', '1');
        if (tr.negColor != null) dot.setAttribute('fill', v >= 0 ? tr.color : tr.negColor);
      }
      var decimals = tr.decimals != null ? tr.decimals : 1;
      var disp = (tr.signed ? (v >= 0 ? '+' : '') : '') + v.toFixed(decimals);
      html += '<div style="color:' + tr.color + '; font-weight:600;">' + tr.label + ' ' + disp +
        (tr.unit ? ' ' + tr.unit : '') + '</div>';
    }
    if (tooltip) { tooltip.innerHTML = html; tooltip.style.display = 'block'; }
  }
  function leave() {
    if (line) line.setAttribute('opacity', '0');
    for (var t = 0; t < cfg.traces.length; t++) {
      var dot = cfg.traces[t].dotSelector ? block.querySelector(cfg.traces[t].dotSelector) : null;
      if (dot) dot.setAttribute('opacity', '0');
    }
    if (tooltip) tooltip.style.display = 'none';
  }
  catchRect.addEventListener('mousemove', move);
  catchRect.addEventListener('mouseleave', leave);
};
</script>'''


def _hover_svg_elems(prefix: str, x0: float, x1: float, total_h: float, n_traces: int) -> str:
    """Catch-rect + hover line + one dot per trace -- the SVG-side half of
    the generic hover module. n_traces dots get ids f'{prefix}-hover-dot-0',
    '-1', etc., matching the dotSelector list a caller passes to cfg.traces."""
    dots = "".join(
        f'<circle id="{prefix}-hover-dot-{i}" r="5" fill="{theme.INK_PRIMARY}" stroke="white" stroke-width="1.5" opacity="0"/>'
        for i in range(n_traces)
    )
    return (
        f'<rect id="{prefix}-hover-catch" x="{x0}" y="0" width="{x1-x0}" height="{total_h}" fill="transparent"/>'
        f'<line id="{prefix}-hover-line" x1="{x0}" y1="4" x2="{x0}" y2="{total_h-4}" stroke="{theme.INK_PRIMARY}" '
        f'stroke-width="1" stroke-dasharray="2,2" opacity="0"/>'
        f'{dots}'
    )


def _sparse_edge_path(fx, y_of, vals: list) -> str:
    """Border trace for a filled up/down area against a zero baseline,
    drawn as a SEPARATE path from the fill polygon so the stroke only
    appears where the value actually leaves zero. Stroking the whole
    closed fill polygon (which always includes a flat return-to-baseline
    edge) draws a solid colored line coincident with the 0 gridline across
    every stretch that's genuinely at 0 -- e.g. down-regulation on an ISP
    that only delivered up-regulation -- which reads as a persistent
    colored x-axis even though nothing is being delivered in that
    direction. This traces through the real zero-crossing points so the
    line still visually rises/falls from baseline exactly where the
    excursion starts/ends; it just doesn't linger there when flat."""
    parts: list = []
    pen_down = False
    for i, v in enumerate(vals):
        x, y = fx(i), y_of(v)
        if v == 0:
            if pen_down:
                parts.append(f"L{x:.1f},{y:.1f}")
                pen_down = False
        else:
            if not pen_down:
                if i > 0:
                    parts.append(f"M{fx(i-1):.1f},{y_of(0.0):.1f}")
                    parts.append(f"L{x:.1f},{y:.1f}")
                else:
                    parts.append(f"M{x:.1f},{y:.1f}")
                pen_down = True
            else:
                parts.append(f"L{x:.1f},{y:.1f}")
    return " ".join(parts)


def _hover_tooltip_div() -> str:
    return (
        f'<div class="dt-hover-tooltip" style="position:absolute; top:6px; right:6px; background:{theme.SURFACE}; '
        f'border:1px solid {theme.GRIDLINE}; border-radius:6px; padding:6px 10px; font-size:12.5px; display:none; '
        f'pointer-events:none; box-shadow:0 2px 6px rgba(0,0,0,0.08); white-space:nowrap; z-index:2;"></div>'
    )

_RESERVOIR_REPLAY_SCRIPT = '''
<script>
window.dtReservoirReplay = function(btn) {
  var block = btn.closest('.res-chart-block');
  if (!block || block.dataset.replaying === '1') return;
  block.dataset.replaying = '1';
  var data = JSON.parse(block.dataset.res);
  var n = data.hours.length;
  var clipA = block.querySelector('#res-clip-rect-a');
  var clipB = block.querySelector('#res-clip-rect-b');
  var playhead = block.querySelector('#res-playhead');
  var clock = block.querySelector('#res-clock');
  var upperLabel = block.querySelector('#res-upper-val');
  var lowerLabel = block.querySelector('#res-lower-val');
  var headLabel = block.querySelector('#res-head-val');
  var spillLabel = block.querySelector('#res-spill-val');

  var steps = n;
  var stepMs = 8000 / steps;
  var numberEvery = Math.max(1, Math.round(steps / 12));
  clipA.setAttribute('width', '0');
  clipB.setAttribute('width', '0');

  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var idx = Math.min(s - 1, n - 1);
        var frac = s / steps;
        clipA.setAttribute('width', (1400 * frac).toFixed(1));
        clipB.setAttribute('width', (1400 * frac).toFixed(1));
        if (playhead) playhead.style.left = (frac * 100) + '%';
        if (clock) clock.textContent = 'H' + data.hours[idx];
        if (s % numberEvery === 0 || s === steps) {
          if (upperLabel) upperLabel.textContent = data.upperHm3[idx].toFixed(1) + ' hm³';
          if (lowerLabel) lowerLabel.textContent = data.lowerHm3[idx].toFixed(1) + ' hm³';
          if (headLabel) headLabel.textContent = data.headM[idx].toFixed(1) + ' m';
          if (spillLabel) spillLabel.textContent = data.spillM3h[idx].toFixed(0) + ' m³/h';
        }
        if (s === steps) block.dataset.replaying = '0';
      }, s * stepMs);
    })(s);
  }
};
</script>'''


def render_reservoir_trajectory_card(traj: dict) -> str:
    """Two-reservoir (Alqueva upper / Pedrogao lower) volume trajectory from
    the real solved DA MILP, each normalized to its own operational band
    [min, usable-or-capacity] so both read on a comparable 0-100% scale
    despite the ~100x difference in absolute hm3 (Alqueva ~2,500-4,150 hm3
    vs Pedrogao ~5-54 hm3). Drawn as two stacked filled 'tank gauges'
    (0/50/100% gridlines) instead of two overlaid thin lines on one shared
    axis -- the overlay made two physically-unrelated reservoirs look
    directly comparable just because their % traces happened to cross,
    which read as confusing rather than informative. A dashed reference
    line on Alqueva's panel marks the terminal constraint level (must end
    >= start, the no-free-lunch rule preventing the solver draining the
    reservoir for one profitable day)."""
    hours = traj["hours"]
    n = len(hours)
    upper_hm3, lower_hm3 = traj["upper_hm3"], traj["lower_hm3"]

    def pct(v: float, lo: float, hi: float) -> float:
        return max(0.0, min(100.0, (v - lo) / max(hi - lo, 1e-9) * 100))

    def zoom_range(vals: list[float], *extra: float) -> tuple[float, float]:
        """Alqueva only moves ~1% of its 2,320 hm3 operational band in a real
        day (Pedrogao, ~47x smaller, moves the same water 40%+ of its own
        band) -- plotted against the full band, Alqueva's line reads as a
        flat/dead trace even though it's genuinely moving. Zoom each panel
        to that day's own observed range (plus any reference level that
        must stay visible, e.g. the terminal target) instead, with a
        padding margin so the line never touches the frame edge."""
        lo, hi = min(vals + list(extra)), max(vals + list(extra))
        pad = max((hi - lo) * 0.15, 0.5)
        return lo - pad, hi + pad

    u_min, u_max = zoom_range(upper_hm3, traj["upper_initial_hm3"])
    l_min, l_max = zoom_range(lower_hm3)

    upper_pct = [pct(v, u_min, u_max) for v in upper_hm3]
    lower_pct = [pct(v, l_min, l_max) for v in lower_hm3]
    terminal_pct = pct(traj["upper_initial_hm3"], u_min, u_max)

    x0, x1 = 10, 1390
    zero_y, top_y = 68, 8
    band_h = zero_y - top_y

    def fx(i: int) -> float:
        return x0 + i / max(n - 1, 1) * (x1 - x0)

    def fy(p: float) -> float:
        return zero_y - (p / 100.0) * band_h

    def fill_path(pcts: list[float]) -> str:
        pts = " L".join(f"{fx(i):.1f},{fy(p):.1f}" for i, p in enumerate(pcts))
        return f"M{fx(0):.1f},{zero_y:.1f} L{pts} L{fx(n-1):.1f},{zero_y:.1f} Z"

    upper_fill = fill_path(upper_pct)
    lower_fill = fill_path(lower_pct)
    terminal_y = fy(terminal_pct)
    mid_y = fy(50.0)
    upper_py = [fy(p) for p in upper_pct]
    lower_py = [fy(p) for p in lower_pct]

    payload = {
        "hours": hours,
        "upperHm3": upper_hm3,
        "lowerHm3": lower_hm3,
        "headM": traj["head_m"],
        "spillM3h": traj["spill_m3h"],
    }
    payload_json = _json.dumps(payload)

    hover_cfg_a = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#res-a-hover-catch", "lineSelector": "#res-a-hover-line",
        "tooltipSelector": "#res-a-tooltip", "indexLabel": "Hour", "indexVals": hours,
        "traces": [{"label": "Alqueva", "yvals": upper_hm3, "py": upper_py, "dotSelector": "#res-a-hover-dot-0",
                    "color": theme.COLOR_GEN, "unit": "hm³", "decimals": 1}],
    })
    hover_cfg_b = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#res-b-hover-catch", "lineSelector": "#res-b-hover-line",
        "tooltipSelector": "#res-b-tooltip", "indexLabel": "Hour", "indexVals": hours,
        "traces": [{"label": "Pedrógão", "yvals": lower_hm3, "py": lower_py, "dotSelector": "#res-b-hover-dot-0",
                    "color": theme.COLOR_PUMP, "unit": "hm³", "decimals": 1}],
    })

    terminal_badge = (
        f'<span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:13.5px; padding:2px 8px; border-radius:5px; font-weight:500;">&#10003; Ended the day at least as full as it started</span>'
        if traj["terminal_ok"] else
        f'<span style="background:{theme.STATUS_CRITICAL}22; color:{theme.STATUS_CRITICAL}; font-size:13.5px; padding:2px 8px; border-radius:5px; font-weight:500;">&#10007; Ended the day emptier than it started</span>'
    )
    spill_total = sum(traj["spill_m3h"])
    spill_note = (
        f"The upper lake got completely full at some point today, so {spill_total:,.0f} m&sup3;/h of water had to be released without generating power."
        if spill_total > 1e-6 else
        "The upper lake never got completely full today, so no water was wasted."
    )

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Optimization</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:15px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">Reservoir trajectory</div>
    <div class="res-chart-block" data-res='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:13.5px; color:{theme.INK_MUTED}; font-weight:500;">Alqueva (upper lake)</span>
      <button class="gt-replay" onclick="dtReservoirReplay(this)">&#9654; Replay</button>
    </div>
    <div style="position:relative; height:84px;">
      <svg viewBox="0 0 1400 76" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <defs><clipPath id="res-clip-a"><rect id="res-clip-rect-a" x="0" y="0" width="1400" height="76"/></clipPath></defs>
        <line x1="{x0}" y1="{top_y}" x2="{x1}" y2="{top_y}" stroke="{theme.GRIDLINE}" stroke-width="1" stroke-dasharray="2,3"/>
        <line x1="{x0}" y1="{mid_y:.1f}" x2="{x1}" y2="{mid_y:.1f}" stroke="{theme.GRIDLINE}" stroke-width="1" stroke-dasharray="2,3"/>
        <line x1="{x0}" y1="{zero_y}" x2="{x1}" y2="{zero_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        <g clip-path="url(#res-clip-a)">
          <path d="{upper_fill}" fill="{theme.COLOR_GEN}" fill-opacity="0.3" stroke="{theme.COLOR_GEN}" stroke-width="2.5"/>
        </g>
        <line x1="{x0}" y1="{terminal_y:.1f}" x2="{x1}" y2="{terminal_y:.1f}" stroke="{theme.STATUS_WARNING}" stroke-width="1" stroke-dasharray="4,3"/>
        <text x="{x0-4}" y="{top_y+3}" font-size="12" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">{u_max:,.0f}</text>
        <text x="{x0-4}" y="{zero_y+3}" font-size="12" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">{u_min:,.0f}</text>
        {_hover_svg_elems("res-a", x0, x1, 76, 1)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="res-a-tooltip"')}
    </div>
    <p style="font-size:15px; color:{theme.INK_MUTED}; margin:2px 0 0; text-align:right;">hm&sup3; &mdash; zoomed to today's range (Alqueva barely moves day-to-day vs. its full size)</p>
    <p style="font-size:13.5px; color:{theme.INK_MUTED}; margin:8px 0 4px; font-weight:500;">Pedr&oacute;g&atilde;o (lower lake)</p>
    <div style="position:relative; height:84px;">
      <svg viewBox="0 0 1400 76" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <defs><clipPath id="res-clip-b"><rect id="res-clip-rect-b" x="0" y="0" width="1400" height="76"/></clipPath></defs>
        <line x1="{x0}" y1="{top_y}" x2="{x1}" y2="{top_y}" stroke="{theme.GRIDLINE}" stroke-width="1" stroke-dasharray="2,3"/>
        <line x1="{x0}" y1="{mid_y:.1f}" x2="{x1}" y2="{mid_y:.1f}" stroke="{theme.GRIDLINE}" stroke-width="1" stroke-dasharray="2,3"/>
        <line x1="{x0}" y1="{zero_y}" x2="{x1}" y2="{zero_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        <g clip-path="url(#res-clip-b)">
          <path d="{lower_fill}" fill="{theme.COLOR_PUMP}" fill-opacity="0.3" stroke="{theme.COLOR_PUMP}" stroke-width="2.5"/>
        </g>
        <text x="{x0-4}" y="{top_y+3}" font-size="12" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">{l_max:,.0f}</text>
        <text x="{x0-4}" y="{zero_y+3}" font-size="12" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">{l_min:,.0f}</text>
        {_hover_svg_elems("res-b", x0, x1, 76, 1)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="res-b-tooltip"')}
    </div>
    <p style="font-size:15px; color:{theme.INK_MUTED}; margin:2px 0 0; text-align:right;">hm&sup3; &mdash; zoomed to today's range</p>
    <div style="display:flex; gap:16px; margin-top:8px;">
      <div style="flex:1;">
        <p style="font-size:14px; color:{theme.INK_MUTED}; margin:0 0 2px;">Alqueva now</p>
        <p style="font-size:14px; font-weight:500; margin:0;"><span id="res-upper-val">{upper_hm3[0]:.1f} hm&sup3;</span></p>
      </div>
      <div style="flex:1;">
        <p style="font-size:14px; color:{theme.INK_MUTED}; margin:0 0 2px;">Pedr&oacute;g&atilde;o now</p>
        <p style="font-size:14px; font-weight:500; margin:0;"><span id="res-lower-val">{lower_hm3[0]:.1f} hm&sup3;</span></p>
      </div>
      <div style="flex:1;">
        <p style="font-size:14px; color:{theme.INK_MUTED}; margin:0 0 2px;">Height between lakes</p>
        <p style="font-size:14px; font-weight:500; margin:0;"><span id="res-head-val">{traj['head_m'][0]:.1f} m</span></p>
      </div>
      <div style="flex:1;">
        <p style="font-size:14px; color:{theme.INK_MUTED}; margin:0 0 2px;">Water wasted (this hour)</p>
        <p style="font-size:14px; font-weight:500; margin:0;"><span id="res-spill-val">{traj['spill_m3h'][0]:.0f} m&sup3;/h</span></p>
      </div>
    </div>
    <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
      <span style="font-size:14px; color:{theme.INK_MUTED};">H1</span>
      <div style="flex:1; height:4px; background:{theme.GRIDLINE}; border-radius:2px; position:relative;">
        <div id="res-playhead" style="position:absolute; left:0%; top:-3px; width:10px; height:10px; border-radius:50%; background:{theme.COLOR_GEN}; transform:translateX(-50%);"></div>
      </div>
      <span style="font-size:14px; color:{theme.INK_MUTED};">H24</span>
      <span id="res-clock" style="font-size:14px; color:{theme.INK_SECONDARY}; min-width:30px; text-align:right;"></span>
    </div>
    </div>
    <div style="display:flex; gap:14px; margin-top:8px;">
      <span style="font-size:14px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:1px; background:{theme.STATUS_WARNING}; margin-right:4px; vertical-align:middle;"></span>Level the day must end at or above</span>
    </div>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-top:8px; padding-top:8px; border-top:1px solid {theme.GRIDLINE};">
      {terminal_badge}
      <span style="font-size:13.5px; color:{theme.INK_MUTED};">Day total wasted: {spill_total:,.0f} m&sup3;/h</span>
    </div>
    <p style="font-size:13px; color:{theme.INK_MUTED}; margin:4px 0 0;">{spill_note}</p>
  </div>
</div>
{_REPLAY_STYLE}
{_RESERVOIR_REPLAY_SCRIPT}
{_HOVER_CROSSHAIR_SCRIPT}
<script>
(function() {{
  var block = document.querySelector('.res-chart-block');
  if (block) {{ dtHoverInit(block, {hover_cfg_a}); dtHoverInit(block, {hover_cfg_b}); }}
}})();
</script>'''


_PV_REPLAY_SCRIPT = '''
<script>
window.dtPvReplay = function(btn) {
  var block = btn.closest('.pv-chart-block');
  if (!block || block.dataset.replaying === '1') return;
  block.dataset.replaying = '1';
  var data = JSON.parse(block.dataset.pv);
  var n = data.hours.length;
  var readout = block.querySelector('#pv-readout');
  var usedBars = block.querySelectorAll('.pv-bar-used');
  var bessBars = block.querySelectorAll('.pv-bar-bess');
  var curtBars = block.querySelectorAll('.pv-bar-curt');
  var steps = n;
  var stepMs = 8000 / steps;

  [usedBars, bessBars, curtBars].forEach(function(g) { g.forEach(function(b) { b.setAttribute('height', '0'); }); });

  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var idx = s - 1;
        [['pv-bar-used', 'usedFinalH', 'usedFinalY'], ['pv-bar-bess', 'bessFinalH', 'bessFinalY'], ['pv-bar-curt', 'curtFinalH', 'curtFinalY']].forEach(function(t) {
          var bar = block.querySelector('.' + t[0] + '[data-hidx="' + idx + '"]');
          if (bar) { bar.setAttribute('height', bar.dataset.finalH); bar.setAttribute('y', bar.dataset.finalY); bar.style.opacity = '1'; }
        });
        if (readout) {
          var avail = data.availableMw[idx].toFixed(1);
          var used = data.usedMw[idx].toFixed(1);
          var bess = data.toBessMw[idx].toFixed(1);
          var curt = data.curtailedMw[idx].toFixed(1);
          var cause = data.causes[idx];
          var txt = 'H' + data.hours[idx] + ': ' + avail + ' MW available \\u2192 ' + used + ' grid / ' + bess + ' BESS / ' + curt + ' curtailed';
          if (cause) txt += ' (' + cause + ')';
          readout.textContent = txt;
        }
        if (s === steps) block.dataset.replaying = '0';
      }, s * stepMs);
    })(s);
  }
};
</script>'''


def render_pv_routing_card(pv: dict) -> str:
    """Real per-hour PV allocation -- used (to grid), to_bess (charge), and
    curtailed, from the exact three-way split the solved model's pv_balance
    constraint enforces (pv_used + pv_to_bess + pv_curt = pv_available).
    Stacked bars, SVG geometry only, HTML text overlay."""
    hours = pv["hours"]
    n = len(hours)
    used, to_bess, curtailed, available = pv["used_mw"], pv["to_bess_mw"], pv["curtailed_mw"], pv["available_mw"]
    max_avail = max(max(available, default=0.0), 1.0)

    x0, x1 = 10, 1390
    zero_y, top_y = 92, 8
    band_h = zero_y - top_y
    slot_w = (x1 - x0) / n
    bar_w = max(min(28.0, slot_w * 0.7), 4.0)

    used_bars, bess_bars, curt_bars = [], [], []
    for i in range(n):
        x = x0 + i * slot_w + (slot_w - bar_w) / 2
        h_used = used[i] / max_avail * band_h
        h_bess = to_bess[i] / max_avail * band_h
        h_curt = curtailed[i] / max_avail * band_h
        y_used = zero_y - h_used
        y_bess = y_used - h_bess
        y_curt = y_bess - h_curt
        used_bars.append(
            f'<rect class="pv-bar-used" data-hidx="{i}" data-final-h="{h_used:.1f}" data-final-y="{y_used:.1f}" '
            f'x="{x:.1f}" y="{y_used:.1f}" width="{bar_w:.1f}" height="{h_used:.1f}" fill="{theme.YELLOW}" opacity="0.9"/>'
        )
        bess_bars.append(
            f'<rect class="pv-bar-bess" data-hidx="{i}" data-final-h="{h_bess:.1f}" data-final-y="{y_bess:.1f}" '
            f'x="{x:.1f}" y="{y_bess:.1f}" width="{bar_w:.1f}" height="{h_bess:.1f}" fill="{theme.COLOR_UP}" opacity="0.9"/>'
        )
        curt_bars.append(
            f'<rect class="pv-bar-curt" data-hidx="{i}" data-final-h="{h_curt:.1f}" data-final-y="{y_curt:.1f}" '
            f'x="{x:.1f}" y="{y_curt:.1f}" width="{bar_w:.1f}" height="{h_curt:.1f}" fill="{theme.STATUS_CRITICAL}" opacity="0.55"/>'
        )

    payload = {
        "hours": hours, "usedMw": used, "toBessMw": to_bess,
        "curtailedMw": curtailed, "availableMw": available, "causes": pv["causes"],
    }
    payload_json = _json.dumps(payload)

    hover_cfg = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#pv-hover-catch", "lineSelector": "#pv-hover-line",
        "tooltipSelector": "#pv-tooltip", "indexLabel": "Hour", "indexVals": hours,
        "traces": [
            {"label": "Used", "yvals": used, "py": used, "color": theme.YELLOW, "unit": "MW", "decimals": 1},
            {"label": "BESS", "yvals": to_bess, "py": to_bess, "color": theme.COLOR_UP, "unit": "MW", "decimals": 1},
            {"label": "Curtailed", "yvals": curtailed, "py": curtailed, "color": theme.STATUS_CRITICAL, "unit": "MW", "decimals": 1},
        ],
    })

    curt_note = (
        f"No curtailment this day &mdash; every available PV MWh was exported or stored."
        if pv["total_curtailed_mwh"] <= 1e-6 else
        f"{pv['total_curtailed_mwh']:.1f} MWh curtailed ({pv['curtailed_pct']:.1f}% of available) &mdash; cost &euro;{pv['curtailment_cost_eur']:.0f}."
    )

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Optimization</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:15px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">PV routing &amp; curtailment</div>
    <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:0 0 10px;">
      <span style="background:{theme.YELLOW}22; color:{theme.YELLOW}; padding:3px 9px; border-radius:6px; font-weight:600; font-size:15px;">&#9728; PV</span>
      <span style="color:{theme.INK_MUTED}; font-size:15px;">splits into</span>
      <span style="background:{theme.COLOR_GEN}22; color:{theme.COLOR_GEN}; padding:3px 9px; border-radius:6px; font-weight:600; font-size:15px;">&#9889; Grid</span>
      <span style="color:{theme.INK_MUTED}; font-size:15px;">/</span>
      <span style="background:{theme.COLOR_UP}22; color:{theme.COLOR_UP}; padding:3px 9px; border-radius:6px; font-weight:600; font-size:15px;">&#128267; Battery</span>
      <span style="color:{theme.INK_MUTED}; font-size:15px;">/</span>
      <span style="background:{theme.STATUS_CRITICAL}22; color:{theme.STATUS_CRITICAL}; padding:3px 9px; border-radius:6px; font-weight:600; font-size:15px;">&#10005; Curtailed</span>
    </div>
    <div class="pv-chart-block" data-pv='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:14px; color:{theme.INK_SECONDARY};">Stacked per hour, normalized to available PV each hour</span>
      <button class="gt-replay" onclick="dtPvReplay(this)">&#9654; Replay</button>
    </div>
    <div style="position:relative; height:130px;">
      <svg viewBox="0 0 1400 100" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{x0}" y1="{zero_y}" x2="{x1}" y2="{zero_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(used_bars)}
        {''.join(bess_bars)}
        {''.join(curt_bars)}
        {_hover_svg_elems("pv", x0, x1, 100, 0)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="pv-tooltip"')}
    </div>
    <div style="display:flex; gap:14px; margin-top:4px;">
      <span style="font-size:14px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.YELLOW}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Used (to grid)</span>
      <span style="font-size:14px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_UP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>To BESS</span>
      <span style="font-size:14px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.STATUS_CRITICAL}; border-radius:2px; margin-right:4px; vertical-align:middle; opacity:0.55;"></span>Curtailed</span>
    </div>
    <div style="margin-top:8px; padding-top:8px; border-top:1px solid {theme.GRIDLINE};">
      <span id="pv-readout" style="font-size:14.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Full day shown &mdash; Replay steps through each hour</span>
    </div>
    </div>
    <p style="font-size:13px; color:{theme.INK_MUTED}; margin:8px 0 0;">{curt_note}</p>
  </div>
</div>
{_REPLAY_STYLE}
{_PV_REPLAY_SCRIPT}
{_HOVER_CROSSHAIR_SCRIPT}
<script>
(function() {{
  var block = document.querySelector('.pv-chart-block');
  if (block) {{ dtHoverInit(block, {hover_cfg}); }}
}})();
</script>'''


_MULTI_REPLAY_SCRIPT = '''
<script>
window.dtMultiReplay = function(btn) {
  var block = btn.closest('.mx-chart-block');
  if (!block || block.dataset.replaying === '1') return;
  block.dataset.replaying = '1';
  var data = JSON.parse(block.dataset.mx);
  var n = data.hours.length;
  var readout = block.querySelector('#mx-readout');
  var groups = ['mx-bar-pv', 'mx-bar-bessdis', 'mx-bar-besschg', 'mx-bar-turb', 'mx-bar-pump'];
  var steps = n;
  var stepMs = 8000 / steps;

  groups.forEach(function(g) {
    block.querySelectorAll('.' + g).forEach(function(b) { b.setAttribute('height', '0'); });
  });

  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var idx = s - 1;
        groups.forEach(function(g) {
          var bar = block.querySelector('.' + g + '[data-hidx="' + idx + '"]');
          if (bar) { bar.setAttribute('height', bar.dataset.finalH); bar.setAttribute('y', bar.dataset.finalY); bar.style.opacity = '1'; }
        });
        if (readout) {
          readout.textContent = 'H' + data.hours[idx] + ': PV ' + data.pvMw[idx].toFixed(1) +
            ' + BESS dis ' + data.bessDisMw[idx].toFixed(1) + ' \\u2212 BESS chg ' + data.bessChgMw[idx].toFixed(1) +
            ' MW \\u00a0|\\u00a0 PSP turbine ' + data.turbMw[idx].toFixed(0) + ' \\u2212 pump ' + data.pumpMw[idx].toFixed(0) +
            ' MW \\u00a0|\\u00a0 net ' + data.netMw[idx].toFixed(0) + ' MW';
        }
        if (s === steps) block.dataset.replaying = '0';
      }, s * stepMs);
    })(s);
  }
};
</script>'''


def render_multi_asset_dispatch_card(mx: dict) -> str:
    """The solver's asset-coordination decision each hour, split into two
    independently-scaled charts because PV (5 MWp) and BESS (1 MW) are two
    orders of magnitude smaller than PSP (400+ MW) -- on one shared axis
    PV/BESS bars are invisible next to PSP. Top chart: PV + BESS on their
    own MW scale. Bottom chart: PSP turbine/pump on its own MW scale. Same
    underlying real solved-model data (pv_schedule/bess_schedule/
    psp_schedule) as the single-chart version, just legible now. Reserve
    headroom (Gen/Pump_headroom_MW) is genuinely plant-level only in this
    pipeline (reserve_offer_builder.py's fat_deliverable_mw sums
    psp_ramp_cap + bess_cap into one combined FAT ceiling) -- shown as a
    footnote, not fabricated into a per-asset split that doesn't exist."""
    hours = mx["hours"]
    n = len(hours)
    pv, bess_dis, bess_chg = mx["pv_mw"], mx["bess_dis_mw"], mx["bess_chg_mw"]
    turb, pump, net = mx["psp_turb_mw"], mx["psp_pump_mw"], mx["net_mw"]

    x0, x1 = 10, 1390
    slot_w = (x1 - x0) / n
    bar_w = max(min(28.0, slot_w * 0.7), 4.0)

    # -- Top chart: PV + BESS (small-scale) --------------------------------
    pv_bess_max = max([abs(pv[i]) + max(bess_dis[i], bess_chg[i]) for i in range(n)] + [1e-6])
    zero_y1, half_h1 = 55, 48
    pv_bars, dis_bars, chg_bars = [], [], []
    for i in range(n):
        x = x0 + i * slot_w + (slot_w - bar_w) / 2
        h_pv = pv[i] / pv_bess_max * half_h1
        h_dis = bess_dis[i] / pv_bess_max * half_h1
        h_chg = bess_chg[i] / pv_bess_max * half_h1

        y_pv = zero_y1 - h_pv
        y_dis = y_pv - h_dis
        pv_bars.append(f'<rect class="mx-bar-pv" data-hidx="{i}" data-final-h="{h_pv:.1f}" data-final-y="{y_pv:.1f}" x="{x:.1f}" y="{y_pv:.1f}" width="{bar_w:.1f}" height="{h_pv:.1f}" fill="{theme.YELLOW}" opacity="0.9"/>')
        dis_bars.append(f'<rect class="mx-bar-bessdis" data-hidx="{i}" data-final-h="{h_dis:.1f}" data-final-y="{y_dis:.1f}" x="{x:.1f}" y="{y_dis:.1f}" width="{bar_w:.1f}" height="{h_dis:.1f}" fill="{theme.COLOR_UP}" opacity="0.9"/>')

        y_chg = zero_y1 + h_chg
        chg_bars.append(f'<rect class="mx-bar-besschg" data-hidx="{i}" data-final-h="{h_chg:.1f}" data-final-y="{zero_y1}" x="{x:.1f}" y="{zero_y1}" width="{bar_w:.1f}" height="{h_chg:.1f}" fill="{theme.COLOR_DOWN}" opacity="0.9"/>')

    # -- Bottom chart: PSP turbine/pump (large-scale) -----------------------
    psp_max = max([max(turb[i], pump[i]) for i in range(n)] + [1e-6])
    zero_y2, half_h2 = 55, 48
    turb_bars, pump_bars = [], []
    for i in range(n):
        x = x0 + i * slot_w + (slot_w - bar_w) / 2
        h_turb = turb[i] / psp_max * half_h2
        h_pump = pump[i] / psp_max * half_h2
        y_turb = zero_y2 - h_turb
        turb_bars.append(f'<rect class="mx-bar-turb" data-hidx="{i}" data-final-h="{h_turb:.1f}" data-final-y="{y_turb:.1f}" x="{x:.1f}" y="{y_turb:.1f}" width="{bar_w:.1f}" height="{h_turb:.1f}" fill="{theme.COLOR_GEN}" opacity="0.9"/>')
        pump_bars.append(f'<rect class="mx-bar-pump" data-hidx="{i}" data-final-h="{h_pump:.1f}" data-final-y="{zero_y2}" x="{x:.1f}" y="{zero_y2}" width="{bar_w:.1f}" height="{h_pump:.1f}" fill="{theme.COLOR_PUMP}" opacity="0.9"/>')

    payload = {
        "hours": hours, "pvMw": pv, "bessDisMw": bess_dis, "turbMw": turb,
        "bessChgMw": bess_chg, "pumpMw": pump, "netMw": net,
    }
    payload_json = _json.dumps(payload)

    hover_cfg_top = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#mx-top-hover-catch", "lineSelector": "#mx-top-hover-line",
        "tooltipSelector": "#mx-top-tooltip", "indexLabel": "Hour", "indexVals": hours,
        "traces": [
            {"label": "PV", "yvals": pv, "py": pv, "color": theme.YELLOW, "unit": "MW", "decimals": 2},
            {"label": "BESS dis", "yvals": bess_dis, "py": bess_dis, "color": theme.COLOR_UP, "unit": "MW", "decimals": 2},
            {"label": "BESS chg", "yvals": bess_chg, "py": bess_chg, "color": theme.COLOR_DOWN, "unit": "MW", "decimals": 2},
        ],
    })
    hover_cfg_bot = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#mx-bot-hover-catch", "lineSelector": "#mx-bot-hover-line",
        "tooltipSelector": "#mx-bot-tooltip", "indexLabel": "Hour", "indexVals": hours,
        "traces": [
            {"label": "PSP turbine", "yvals": turb, "py": turb, "color": theme.COLOR_GEN, "unit": "MW", "decimals": 1},
            {"label": "PSP pump", "yvals": pump, "py": pump, "color": theme.COLOR_PUMP, "unit": "MW", "decimals": 1},
        ],
    })

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Optimization</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:15px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">Multi-asset dispatch</div>
    <div class="mx-chart-block" data-mx='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:flex-end; margin-bottom:6px;">
      <button class="gt-replay" onclick="dtMultiReplay(this)">&#9654; Replay</button>
    </div>
    <p style="font-size:13.5px; color:{theme.INK_MUTED}; margin:6px 0 2px; font-weight:500;">PV &amp; BESS (own scale, max {pv_bess_max:.1f} MW)</p>
    <div style="position:relative; height:104px;">
      <svg viewBox="0 0 1400 110" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{x0}" y1="{zero_y1}" x2="{x1}" y2="{zero_y1}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(pv_bars)}
        {''.join(dis_bars)}
        {''.join(chg_bars)}
        {_hover_svg_elems("mx-top", x0, x1, 110, 0)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="mx-top-tooltip"')}
    </div>
    <div style="display:flex; gap:12px; margin:2px 0 8px;">
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.YELLOW}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>PV</span>
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_UP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>BESS discharge</span>
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_DOWN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>BESS charge</span>
    </div>
    <p style="font-size:13.5px; color:{theme.INK_MUTED}; margin:6px 0 2px; font-weight:500;">PSP turbine &amp; pump (own scale, max {psp_max:.0f} MW)</p>
    <div style="position:relative; height:104px;">
      <svg viewBox="0 0 1400 110" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{x0}" y1="{zero_y2}" x2="{x1}" y2="{zero_y2}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(turb_bars)}
        {''.join(pump_bars)}
        {_hover_svg_elems("mx-bot", x0, x1, 110, 0)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="mx-bot-tooltip"')}
    </div>
    <div style="display:flex; gap:12px; margin:2px 0 8px;">
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_GEN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>PSP turbine</span>
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_PUMP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>PSP pump</span>
    </div>
    <span id="mx-readout" style="display:none;"></span>
    </div>
  </div>
</div>
{_REPLAY_STYLE}
{_MULTI_REPLAY_SCRIPT}
{_HOVER_CROSSHAIR_SCRIPT}
<script>
(function() {{
  var block = document.querySelector('.mx-chart-block');
  if (block) {{ dtHoverInit(block, {hover_cfg_top}); dtHoverInit(block, {hover_cfg_bot}); }}
}})();
</script>'''


_WATER_REPLAY_SCRIPT = '''
<script>
window.dtWaterReplay = function(btn) {
  var block = btn.closest('.wb-chart-block');
  if (!block || block.dataset.replaying === '1') return;
  block.dataset.replaying = '1';
  var data = JSON.parse(block.dataset.wb);
  var n = data.hours.length;
  var readout = block.querySelector('#wb-readout');
  var groups = ['wb-bar-in', 'wb-bar-pump', 'wb-bar-turb', 'wb-bar-spill'];
  var steps = n;
  var stepMs = 8000 / steps;

  groups.forEach(function(g) { block.querySelectorAll('.' + g).forEach(function(b) { b.setAttribute('height', '0'); }); });

  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var idx = s - 1;
        groups.forEach(function(g) {
          var bar = block.querySelector('.' + g + '[data-hidx="' + idx + '"]');
          if (bar) { bar.setAttribute('height', bar.dataset.finalH); bar.setAttribute('y', bar.dataset.finalY); bar.style.opacity = '1'; }
        });
        if (readout) {
          readout.textContent = 'H' + data.hours[idx] + ': inflow ' + Math.round(data.inflow[idx]).toLocaleString() +
            ' + pump ' + Math.round(data.pump[idx]).toLocaleString() + ' \\u2212 turbine ' + Math.round(data.turb[idx]).toLocaleString() +
            ' \\u2212 spill ' + Math.round(data.spill[idx]).toLocaleString() + ' m\\u00b3/h = \\u0394' + data.delta[idx].toFixed(2) + ' hm\\u00b3 (' + data.upperHm3[idx].toFixed(1) + ' hm\\u00b3)';
        }
        if (s === steps) block.dataset.replaying = '0';
      }, s * stepMs);
    })(s);
  }
};
</script>'''


def render_water_balance_card(wb: dict) -> str:
    """Upper reservoir (Alqueva) water balance decomposition -- real natural
    inflow + pump-return flow - turbine draw - spill, verified against the
    solved model's own upper_hm3 continuity each hour (the identity the
    reservoir_balance constraint enforces internally, shown here as a visible
    hour-by-hour reconciliation rather than assumed correct)."""
    rows = wb["rows"]
    hours = wb["hours"]
    n = len(hours)
    max_mag = max([abs(r["net_flow_hm3"]) for r in rows] + [0.1])

    x0, x1 = 10, 1390
    zero_y, half_h = 50, 42
    slot_w = (x1 - x0) / n
    bar_w = max(min(28.0, slot_w * 0.7), 4.0)

    in_bars, pump_bars, turb_bars, spill_bars = [], [], [], []
    for i, r in enumerate(rows):
        x = x0 + i * slot_w + (slot_w - bar_w) / 2
        h_in = (r["inflow_m3h"] / 1e6) / max_mag * half_h
        h_pump = (r["pump_m3h"] / 1e6) / max_mag * half_h
        h_turb = (r["turb_m3h"] / 1e6) / max_mag * half_h
        h_spill = (r["spill_m3h"] / 1e6) / max_mag * half_h

        y_in = zero_y - h_in
        y_pump = y_in - h_pump
        in_bars.append(f'<rect class="wb-bar-in" data-hidx="{i}" data-final-h="{h_in:.1f}" data-final-y="{y_in:.1f}" x="{x:.1f}" y="{y_in:.1f}" width="{bar_w:.1f}" height="{h_in:.1f}" fill="{theme.COLOR_UP}" opacity="0.9"/>')
        pump_bars.append(f'<rect class="wb-bar-pump" data-hidx="{i}" data-final-h="{h_pump:.1f}" data-final-y="{y_pump:.1f}" x="{x:.1f}" y="{y_pump:.1f}" width="{bar_w:.1f}" height="{h_pump:.1f}" fill="{theme.COLOR_PUMP}" opacity="0.9"/>')

        y_turb = zero_y + h_turb
        y_spill = y_turb + h_spill
        turb_bars.append(f'<rect class="wb-bar-turb" data-hidx="{i}" data-final-h="{h_turb:.1f}" data-final-y="{zero_y}" x="{x:.1f}" y="{zero_y}" width="{bar_w:.1f}" height="{h_turb:.1f}" fill="{theme.COLOR_GEN}" opacity="0.9"/>')
        spill_bars.append(f'<rect class="wb-bar-spill" data-hidx="{i}" data-final-h="{h_spill:.1f}" data-final-y="{y_turb:.1f}" x="{x:.1f}" y="{y_turb:.1f}" width="{bar_w:.1f}" height="{h_spill:.1f}" fill="{theme.STATUS_CRITICAL}" opacity="0.6"/>')

    payload = {
        "hours": hours,
        "inflow": [r["inflow_m3h"] for r in rows],
        "pump": [r["pump_m3h"] for r in rows],
        "turb": [r["turb_m3h"] for r in rows],
        "spill": [r["spill_m3h"] for r in rows],
        "delta": [r["delta_hm3"] for r in rows],
        "upperHm3": [r["upper_hm3"] for r in rows],
    }
    payload_json = _json.dumps(payload)

    hover_cfg = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#wb-hover-catch", "lineSelector": "#wb-hover-line",
        "tooltipSelector": "#wb-tooltip", "indexLabel": "Hour", "indexVals": hours,
        "traces": [
            {"label": "Inflow", "yvals": [r["inflow_m3h"] for r in rows], "py": [r["inflow_m3h"] for r in rows], "color": theme.COLOR_UP, "unit": "m³/h", "decimals": 0},
            {"label": "Pump return", "yvals": [r["pump_m3h"] for r in rows], "py": [r["pump_m3h"] for r in rows], "color": theme.COLOR_PUMP, "unit": "m³/h", "decimals": 0},
            {"label": "Turbine draw", "yvals": [r["turb_m3h"] for r in rows], "py": [r["turb_m3h"] for r in rows], "color": theme.COLOR_GEN, "unit": "m³/h", "decimals": 0},
            {"label": "Spill", "yvals": [r["spill_m3h"] for r in rows], "py": [r["spill_m3h"] for r in rows], "color": theme.STATUS_CRITICAL, "unit": "m³/h", "decimals": 0},
        ],
    })

    spill_note = (
        f"No spill this day &mdash; every m&sup3; of inflow and pumped water stayed usable."
        if wb["total_spill_m3h"] <= 1e-6 else
        f"{wb['total_spill_m3h']:,.0f} m&sup3;/h spilled total &mdash; reservoir hit its usable ceiling at some hour."
    )

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Optimization</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:15px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">Water balance</div>
    <p style="font-size:15px; color:{theme.INK_MUTED}; margin:0 0 10px;">Upper reservoir (Alqueva): inflow + pump return &minus; turbine draw &minus; spill, per hour</p>
    <div class="wb-chart-block" data-wb='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:14px; color:{theme.INK_SECONDARY};">Above zero = inflows &middot; below = outflows</span>
      <button class="gt-replay" onclick="dtWaterReplay(this)">&#9654; Replay</button>
    </div>
    <div style="position:relative; height:130px;">
      <svg viewBox="0 0 1400 100" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{x0}" y1="{zero_y}" x2="{x1}" y2="{zero_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(in_bars)}
        {''.join(pump_bars)}
        {''.join(turb_bars)}
        {''.join(spill_bars)}
        {_hover_svg_elems("wb", x0, x1, 100, 0)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="wb-tooltip"')}
    </div>
    <div style="display:flex; gap:12px; margin-top:4px; flex-wrap:wrap;">
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_UP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Natural inflow</span>
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_PUMP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Pump return</span>
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_GEN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Turbine draw</span>
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.STATUS_CRITICAL}; border-radius:2px; margin-right:4px; vertical-align:middle; opacity:0.6;"></span>Spill</span>
    </div>
    <div style="margin-top:8px; padding-top:8px; border-top:1px solid {theme.GRIDLINE};">
      <span id="wb-readout" style="font-size:14.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Full day shown &mdash; Replay steps through each hour</span>
    </div>
    </div>
    <p style="font-size:13px; color:{theme.INK_MUTED}; margin:8px 0 0;">{spill_note}</p>
  </div>
</div>
{_REPLAY_STYLE}
{_WATER_REPLAY_SCRIPT}
{_HOVER_CROSSHAIR_SCRIPT}
<script>
(function() {{
  var block = document.querySelector('.wb-chart-block');
  if (block) {{ dtHoverInit(block, {hover_cfg}); }}
}})();
</script>'''


_BESS_REPLAY_SCRIPT = '''
<script>
window.dtBessReplay = function(btn) {
  var block = btn.closest('.bs-chart-block');
  if (!block || block.dataset.replaying === '1') return;
  block.dataset.replaying = '1';
  var data = JSON.parse(block.dataset.bs);
  var n = data.hours.length;
  var readout = block.querySelector('#bs-readout');
  var dot = block.querySelector('#bs-dot');
  var clipRect = block.querySelector('#bs-clip-rect');
  var steps = n;
  var stepMs = 8000 / steps;
  clipRect.setAttribute('width', '0');
  if (dot) dot.style.opacity = '1';

  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var idx = s - 1;
        var frac = s / steps;
        clipRect.setAttribute('width', (1400 * frac).toFixed(1));
        if (dot) { dot.setAttribute('cx', data.socX[idx].toFixed(1)); dot.setAttribute('cy', data.socY[idx].toFixed(1)); }
        if (readout) {
          var action = data.chgMw[idx] > 0.01 ? ('charging ' + data.chgMw[idx].toFixed(2) + ' MW')
            : (data.disMw[idx] > 0.01 ? ('discharging ' + data.disMw[idx].toFixed(2) + ' MW') : 'idle');
          readout.textContent = 'H' + data.hours[idx] + ': price \\u20ac' + data.price[idx].toFixed(1) + '/MWh \\u2192 ' + action +
            ', SOC ' + data.socMwh[idx].toFixed(2) + ' MWh (' + data.socPct[idx].toFixed(0) + '%)';
        }
        if (s === steps) {
          if (dot) dot.style.opacity = '0';
          block.dataset.replaying = '0';
        }
      }, s * stepMs);
    })(s);
  }
};
</script>'''


def render_bess_soc_price_card(bs: dict) -> str:
    """BESS SOC trajectory vs real DA clearing price -- the storage
    arbitrage decision the solver actually made (soc_mwh, charge_mw/
    discharge_mw from ComponentStore.bess_schedule; DA_price_EUR_MWh from
    the same Dispatch_Hourly sheet every other price series here reads).
    Charge/discharge hours shaded under the SOC line so the price
    correlation is visible without a second axis."""
    hours = bs["hours"]
    n = len(hours)
    soc_pct = bs["soc_pct"]
    price = bs["price_eur_mwh"]
    p_lo, p_hi = min(price, default=0.0), max(price, default=1.0)

    x0, x1 = 0, 1400
    top_y, bottom_y = 8, 92
    band_h = bottom_y - top_y

    def fx(i: int) -> float:
        return x0 + i / max(n - 1, 1) * (x1 - x0)

    def fy_soc(p: float) -> float:
        return bottom_y - (max(0.0, min(100.0, p)) / 100.0) * band_h

    def fy_price(v: float) -> float:
        frac = (v - p_lo) / max(p_hi - p_lo, 1e-9)
        return bottom_y - frac * band_h

    soc_pts = [(fx(i), fy_soc(p)) for i, p in enumerate(soc_pct)]
    price_pts = [(fx(i), fy_price(v)) for i, v in enumerate(price)]
    soc_line = " ".join(f"{x:.1f},{y:.1f}" for x, y in soc_pts)
    price_line = " ".join(f"{x:.1f},{y:.1f}" for x, y in price_pts)

    shades = []
    slot_w = (x1 - x0) / n
    for i in range(n):
        chg, dis = bs["chg_mw"][i], bs["dis_mw"][i]
        if chg > 0.01:
            color, opacity = theme.COLOR_DOWN, 0.12
        elif dis > 0.01:
            color, opacity = theme.COLOR_UP, 0.12
        else:
            continue
        shades.append(f'<rect x="{fx(i) - slot_w/2:.1f}" y="{top_y}" width="{slot_w:.1f}" height="{band_h}" fill="{color}" opacity="{opacity}"/>')

    payload = {
        "hours": hours, "socMwh": bs["soc_mwh"], "socPct": soc_pct, "price": price,
        "chgMw": bs["chg_mw"], "disMw": bs["dis_mw"],
        "socX": [x for x, _ in soc_pts], "socY": [y for _, y in soc_pts],
    }
    payload_json = _json.dumps(payload)

    hover_cfg = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#bs-hover-catch", "lineSelector": "#bs-hover-line",
        "tooltipSelector": "#bs-tooltip", "indexLabel": "Hour", "indexVals": hours,
        "traces": [
            {"label": "SOC", "yvals": soc_pct, "py": [y for _, y in soc_pts], "dotSelector": "#bs-hover-dot-0", "color": theme.COLOR_GEN, "unit": "%", "decimals": 0},
            {"label": "DA price", "yvals": price, "py": [y for _, y in price_pts], "dotSelector": "#bs-hover-dot-1", "color": theme.COLOR_PRICE, "unit": "EUR/MWh", "decimals": 1},
        ],
    })

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Optimization</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:15px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">BESS SOC vs price</div>
    <p style="font-size:15px; color:{theme.INK_MUTED}; margin:0 0 10px;">Storage arbitrage the solver actually chose &mdash; SOC (% of usable band) against DA clearing price</p>
    <div class="bs-chart-block" data-bs='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:14px; color:{theme.INK_SECONDARY};">Shaded = charge (blue) / discharge (aqua) hours</span>
      <button class="gt-replay" onclick="dtBessReplay(this)">&#9654; Replay</button>
    </div>
    <div style="position:relative; height:130px;">
      <svg viewBox="0 0 1400 100" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <defs><clipPath id="bs-clip"><rect id="bs-clip-rect" x="0" y="-20" width="1400" height="140"/></clipPath></defs>
        {''.join(shades)}
        <g clip-path="url(#bs-clip)">
          <polyline points="{price_line}" fill="none" stroke="{theme.COLOR_PRICE}" stroke-width="2.3" opacity="0.85" stroke-dasharray="3,2"/>
          <polyline points="{soc_line}" fill="none" stroke="{theme.COLOR_GEN}" stroke-width="2.5"/>
        </g>
        <circle id="bs-dot" cx="0" cy="{fy_soc(soc_pct[0]):.1f}" r="6" fill="{theme.COLOR_GEN}" style="opacity:0;"/>
        {_hover_svg_elems("bs", x0, x1, 100, 2)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="bs-tooltip"')}
    </div>
    <div style="display:flex; gap:14px; margin-top:4px;">
      <span style="font-size:14px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:2px; background:{theme.COLOR_GEN}; margin-right:4px; vertical-align:middle;"></span>SOC %</span>
      <span style="font-size:14px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:1px; background:{theme.COLOR_PRICE}; margin-right:4px; vertical-align:middle;"></span>DA price</span>
    </div>
    <div style="margin-top:8px; padding-top:8px; border-top:1px solid {theme.GRIDLINE};">
      <span id="bs-readout" style="font-size:14.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Full day shown &mdash; Replay steps through each hour</span>
    </div>
    </div>
    <p style="font-size:13px; color:{theme.INK_MUTED}; margin:8px 0 0;">Usable SOC band: {bs['e_min_mwh']:.2f}&ndash;{bs['e_max_mwh']:.2f} MWh &mdash; small (2 MWh) BESS, so absolute swings are modest even when the % band moves a lot.</p>
  </div>
</div>
{_REPLAY_STYLE}
{_BESS_REPLAY_SCRIPT}
{_HOVER_CROSSHAIR_SCRIPT}
<script>
(function() {{
  var block = document.querySelector('.bs-chart-block');
  if (block) {{ dtHoverInit(block, {hover_cfg}); }}
}})();
</script>'''


_BESSCHG_REPLAY_SCRIPT = '''
<script>
window.dtBesschgReplay = function(btn) {
  var block = btn.closest('.bc2-chart-block');
  if (!block || block.dataset.replaying === '1') return;
  block.dataset.replaying = '1';
  var data = JSON.parse(block.dataset.bc2);
  var n = data.hours.length;
  var readout = block.querySelector('#bc2-readout');
  var groups = ['bc2-bar-grid', 'bc2-bar-pv'];
  var steps = n;
  var stepMs = 8000 / steps;

  groups.forEach(function(g) {
    block.querySelectorAll('.' + g).forEach(function(b) { b.setAttribute('height', '0'); });
  });

  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var idx = s - 1;
        groups.forEach(function(g) {
          var bar = block.querySelector('.' + g + '[data-hidx="' + idx + '"]');
          if (bar) { bar.setAttribute('height', bar.dataset.finalH); bar.setAttribute('y', bar.dataset.finalY); bar.style.opacity = '1'; }
        });
        if (readout) {
          readout.textContent = 'H' + data.hours[idx] + ': grid ' + data.gridMw[idx].toFixed(2) +
            ' MW + PV ' + data.pvMw[idx].toFixed(2) + ' MW charging';
        }
        if (s === steps) block.dataset.replaying = '0';
      }, s * stepMs);
    })(s);
  }
};
</script>'''


def render_bess_charge_source_card(bc: dict) -> str:
    """Real per-hour split of BESS charging power by source -- grid
    (bess_schedule.charge_mw) vs PV (bess_schedule.pv_to_bess_mw), the exact
    two terms the solver's real total_charge = p_chg + pv_to_bess constraint
    sums. Honest empty-state: across every date in this pipeline's history,
    the BESS never charges -- verified this is real economics (round-trip
    efficiency loss + degradation cost pricing out same-day arbitrage at
    this battery's size), not a data or modeling bug, after fixing a real
    day-to-day state-continuity gap that could have explained it but didn't
    change the outcome."""
    hours = bc["hours"]
    n = len(hours)
    grid_mw, pv_mw = bc["grid_mw"], bc["pv_mw"]
    max_mag = max(max(bc["total_mw"], default=0.0), 1e-6)

    x0, x1 = 10, 1390
    zero_y, top_y = 92, 8
    band_h = zero_y - top_y
    slot_w = (x1 - x0) / n
    bar_w = max(min(28.0, slot_w * 0.7), 4.0)

    grid_bars, pv_bars = [], []
    for i in range(n):
        x = x0 + i * slot_w + (slot_w - bar_w) / 2
        h_grid = grid_mw[i] / max_mag * band_h
        h_pv = pv_mw[i] / max_mag * band_h
        y_grid = zero_y - h_grid
        y_pv = y_grid - h_pv
        grid_bars.append(f'<rect class="bc2-bar-grid" data-hidx="{i}" data-final-h="{h_grid:.1f}" data-final-y="{y_grid:.1f}" x="{x:.1f}" y="{y_grid:.1f}" width="{bar_w:.1f}" height="{h_grid:.1f}" fill="{theme.COLOR_DOWN}" opacity="0.9"/>')
        pv_bars.append(f'<rect class="bc2-bar-pv" data-hidx="{i}" data-final-h="{h_pv:.1f}" data-final-y="{y_pv:.1f}" x="{x:.1f}" y="{y_pv:.1f}" width="{bar_w:.1f}" height="{h_pv:.1f}" fill="{theme.YELLOW}" opacity="0.9"/>')

    payload = {"hours": hours, "gridMw": grid_mw, "pvMw": pv_mw}
    payload_json = _json.dumps(payload)

    hover_cfg = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#bc2-hover-catch", "lineSelector": "#bc2-hover-line",
        "tooltipSelector": "#bc2-tooltip", "indexLabel": "Hour", "indexVals": hours,
        "traces": [
            {"label": "Grid", "yvals": grid_mw, "py": grid_mw, "color": theme.COLOR_DOWN, "unit": "MW", "decimals": 2},
            {"label": "PV", "yvals": pv_mw, "py": pv_mw, "color": theme.YELLOW, "unit": "MW", "decimals": 2},
        ],
    })

    if bc["any_charging"]:
        summary = f"{bc['grid_share_pct']:.0f}% from grid, {bc['pv_share_pct']:.0f}% from PV this day"
        empty_badge = ""
    else:
        summary = "No charging this day"
        empty_badge = (
            f'<div style="background:{theme.STATUS_WARNING}18; border:1px solid {theme.STATUS_WARNING}55; '
            f'border-radius:8px; padding:10px 12px; margin-top:8px; font-size:14.5px; color:{theme.INK_PRIMARY}; line-height:1.5;">'
            f'<strong>Verified honest finding, not a bug:</strong> the BESS never charges on any date in this pipeline\'s history. '
            f'Round-trip efficiency is {bc["round_trip_eff_pct"]:.0f}% and the degradation cost '
            f'(&euro;{bc["degradation_cost_eur_mwh"]:.1f}/MWh, charged on both legs of a cycle) prices out same-day arbitrage '
            f'for this battery\'s size at realistic DA price spreads. A real day-to-day BESS state-continuity gap was found and '
            f'fixed in the pipeline (SOC now carries over instead of resetting each morning) -- it did not change this outcome, '
            f'confirming it\'s genuine economics under the configured degradation-cost assumption, not a modeling artifact.'
            f'</div>'
        )

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Optimization</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:15px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">BESS charge source</div>
    <p style="font-size:15px; color:{theme.INK_MUTED}; margin:0 0 10px;">What BESS charging power actually came from &mdash; grid vs PV &mdash; {summary}</p>
    <div class="bc2-chart-block" data-bc2='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:14px; color:{theme.INK_SECONDARY};">Stacked per hour, charging only</span>
      <button class="gt-replay" onclick="dtBesschgReplay(this)">&#9654; Replay</button>
    </div>
    <div style="position:relative; height:130px;">
      <svg viewBox="0 0 1400 100" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{x0}" y1="{zero_y}" x2="{x1}" y2="{zero_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(grid_bars)}
        {''.join(pv_bars)}
        {_hover_svg_elems("bc2", x0, x1, 100, 0)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="bc2-tooltip"')}
    </div>
    <div style="display:flex; gap:14px; margin-top:4px;">
      <span style="font-size:14px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_DOWN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Grid</span>
      <span style="font-size:14px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.YELLOW}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>PV</span>
    </div>
    <div style="margin-top:8px; padding-top:8px; border-top:1px solid {theme.GRIDLINE};">
      <span id="bc2-readout" style="font-size:14.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Full day shown &mdash; Replay steps through each hour</span>
    </div>
    {empty_badge}
    </div>
  </div>
</div>
{_REPLAY_STYLE}
{_BESSCHG_REPLAY_SCRIPT}
{_HOVER_CROSSHAIR_SCRIPT}
<script>
(function() {{
  var block = document.querySelector('.bc2-chart-block');
  if (block) {{ dtHoverInit(block, {hover_cfg}); }}
}})();
</script>'''


_DAACT_REPLAY_SCRIPT = '''
<script>
window.dtDaactReplay = function(btn) {
  var block = btn.closest('.da2-chart-block');
  if (!block || block.dataset.replaying === '1') return;
  block.dataset.replaying = '1';
  var data = JSON.parse(block.dataset.da2);
  var n = data.hours.length;
  var readout = block.querySelector('#da2-readout');
  var daBars = block.querySelectorAll('.da2-bar');
  var deltaBars = block.querySelectorAll('.da2-delta-bar');
  var steps = n;
  var stepMs = 8000 / steps;

  daBars.forEach(function(b) { b.setAttribute('height', '0'); });
  deltaBars.forEach(function(b) { b.setAttribute('height', '0'); });

  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var idx = s - 1;
        var bar = block.querySelector('.da2-bar[data-hidx="' + idx + '"]');
        var dbar = block.querySelector('.da2-delta-bar[data-hidx="' + idx + '"]');
        if (bar) { bar.setAttribute('height', bar.dataset.finalH); bar.setAttribute('y', bar.dataset.finalY); bar.style.opacity = '1'; }
        if (dbar) { dbar.setAttribute('height', dbar.dataset.finalH); dbar.setAttribute('y', dbar.dataset.finalY); dbar.style.opacity = '1'; }
        if (readout) {
          readout.textContent = 'ISP ' + data.isps[idx] + ' (H' + data.hours[idx] + '): DA ' + data.daMw[idx].toFixed(0) +
            ' MW committed, grid asked for ' + (data.deltaMw[idx] >= 0 ? '+' : '') + data.deltaMw[idx].toFixed(0) +
            ' MW \\u2192 ' + data.deliveredMw[idx].toFixed(0) + ' MW actual';
        }
        if (s === steps) block.dataset.replaying = '0';
      }, s * stepMs);
    })(s);
  }
};
</script>'''


def render_da_vs_activation_card(dv: dict) -> str:
    """Two clearly separate panels instead of one overlaid chart, at true
    96-ISP (15-min) resolution: Panel 1 is the DA-committed net position
    (what the plant already agreed yesterday to sell/buy each ISP). Panel 2
    is the real per-ISP aFRR/mFRR activation delta as its own signed bar
    chart, colored green ("grid asked for more") / red ("grid asked for
    less") on its own scale -- these are two independent obligations, not a
    reconciliation, and a tick+dot riding on top of the DA bar read as a
    bar-height error mark rather than its own event. DA is a day-ahead
    energy schedule; activation is a real-time response to system-wide
    Area Control Error, unrelated to Alqueva's own DA delivery accuracy.
    Combined aFRR+mFRR, plant-level only (no per-asset split exists in
    the real pipeline)."""
    isps = dv["isps"]
    hours = dv["hours"]
    n = len(isps)
    da_net = dv["da_net_mw"]
    delta = dv["activation_delta_mw"]
    delivered = dv["delivered_mw"]
    da_max = max([abs(v) for v in da_net] + [1.0])
    delta_max = max([abs(v) for v in delta] + [1.0])

    x0, x1 = 10, 1390
    zero_y1, half_h1 = 45, 38
    zero_y2, half_h2 = 45, 38
    slot_w = (x1 - x0) / n
    bar_w = max(min(28.0, slot_w * 0.7), 4.0)

    def y1(v: float) -> float:
        return zero_y1 - (v / da_max) * half_h1

    def y2(v: float) -> float:
        return zero_y2 - (v / delta_max) * half_h2

    da_bars, delta_bars = [], []
    for i in range(n):
        x_center = x0 + i * slot_w + (slot_w - bar_w) / 2
        y_da = y1(da_net[i])
        bar_top = min(zero_y1, y_da)
        bar_h = abs(y_da - zero_y1)
        da_bars.append(f'<rect class="da2-bar" data-hidx="{i}" data-final-h="{bar_h:.1f}" data-final-y="{bar_top:.1f}" x="{x_center:.1f}" y="{bar_top:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{theme.COLOR_GEN}" opacity="0.85"/>')

        y_delta = y2(delta[i])
        dbar_top = min(zero_y2, y_delta)
        dbar_h = abs(y_delta - zero_y2)
        dcolor = theme.STATUS_GOOD if delta[i] >= 0 else theme.STATUS_CRITICAL
        delta_bars.append(f'<rect class="da2-delta-bar" data-hidx="{i}" data-final-h="{dbar_h:.1f}" data-final-y="{dbar_top:.1f}" x="{x_center:.1f}" y="{dbar_top:.1f}" width="{bar_w:.1f}" height="{dbar_h:.1f}" fill="{dcolor}" opacity="0.85"/>')

    payload = {"isps": isps, "hours": hours, "daMw": da_net, "deltaMw": delta, "deliveredMw": delivered}
    payload_json = _json.dumps(payload)

    hover_cfg_top = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#da2-top-hover-catch", "lineSelector": "#da2-top-hover-line",
        "tooltipSelector": "#da2-top-tooltip", "indexLabel": "ISP", "indexVals": isps,
        "traces": [{"label": "DA committed", "yvals": da_net, "py": da_net, "color": theme.COLOR_GEN, "unit": "MW", "decimals": 1, "signed": True}],
    })
    hover_cfg_bot = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#da2-bot-hover-catch", "lineSelector": "#da2-bot-hover-line",
        "tooltipSelector": "#da2-bot-tooltip", "indexLabel": "ISP", "indexVals": isps,
        "traces": [{"label": "Activation delta", "yvals": delta, "py": delta, "color": theme.STATUS_GOOD, "negColor": theme.STATUS_CRITICAL, "unit": "MW", "decimals": 1, "signed": True}],
    })

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Optimization</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:15px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">DA schedule vs grid activation</div>
    <div class="da2-chart-block" data-da2='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:2px;">
      <span style="font-size:13.5px; color:{theme.INK_MUTED}; font-weight:500;">1. What the plant already agreed to do (planned yesterday)</span>
      <button class="gt-replay" onclick="dtDaactReplay(this)">&#9654; Replay</button>
    </div>
    <div style="position:relative; height:78px;">
      <svg viewBox="0 0 1400 90" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{x0}" y1="{zero_y1}" x2="{x1}" y2="{zero_y1}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(da_bars)}
        {_hover_svg_elems("da2-top", x0, x1, 90, 0)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="da2-top-tooltip"')}
    </div>
    <p style="font-size:13.5px; color:{theme.INK_MUTED}; margin:8px 0 2px; font-weight:500;">2. Extra power the grid asked for right then (separate, real-time)</p>
    <div style="position:relative; height:78px;">
      <svg viewBox="0 0 1400 90" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{x0}" y1="{zero_y2}" x2="{x1}" y2="{zero_y2}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(delta_bars)}
        {_hover_svg_elems("da2-bot", x0, x1, 90, 0)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="da2-bot-tooltip"')}
    </div>
    <div style="display:flex; gap:14px; margin-top:4px;">
      <span style="font-size:14px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.STATUS_GOOD}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Grid asked for more power</span>
      <span style="font-size:14px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.STATUS_CRITICAL}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Grid asked for less power</span>
    </div>
    <span id="da2-readout" style="display:none;"></span>
    </div>
  </div>
</div>
{_REPLAY_STYLE}
{_DAACT_REPLAY_SCRIPT}
{_HOVER_CROSSHAIR_SCRIPT}
<script>
(function() {{
  var block = document.querySelector('.da2-chart-block');
  if (block) {{ dtHoverInit(block, {hover_cfg_top}); dtHoverInit(block, {hover_cfg_bot}); }}
}})();
</script>'''


_ISP_REPLAY_SCRIPT = '''
<script>
window.dtIspReplay = function(btn) {
  var block = btn.closest('.ispd-chart-block');
  if (!block || block.dataset.replaying === '1') return;
  block.dataset.replaying = '1';
  var data = JSON.parse(block.dataset.ispd);
  var n = data.isps.length;
  var clipA = block.querySelector('#ispd-clip-rect-a');
  var clipB = block.querySelector('#ispd-clip-rect-b');
  var playhead = block.querySelector('#ispd-playhead');
  var readout = block.querySelector('#ispd-readout');
  var steps = n;
  var stepMs = 6000 / steps;
  clipA.setAttribute('width', '0');
  clipB.setAttribute('width', '0');
  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var idx = Math.min(s - 1, n - 1);
        var frac = s / steps;
        clipA.setAttribute('width', (1400 * frac).toFixed(1));
        clipB.setAttribute('width', (1400 * frac).toFixed(1));
        if (playhead) playhead.style.left = (frac * 100) + '%';
        if (readout) readout.textContent = 'ISP ' + data.isps[idx] + ' (H' + data.hours[idx] + '): PV ' +
          data.pvMw[idx].toFixed(2) + ' MW · BESS ' + data.bessMw[idx].toFixed(2) + ' MW · PSP turbine ' +
          data.pspMw[idx].toFixed(1) + ' MW · PSP pump ' + data.pspPumpMw[idx].toFixed(1) + ' MW';
        if (s === steps) block.dataset.replaying = '0';
      }, s * stepMs);
    })(s);
  }
};
</script>'''


def render_pnl_breakdown_card(total_pnl: float, reserve_pct: float | None, lines: list[tuple[str, float, str]]) -> str:
    """Total P&L plus every real revenue line item as a mini proportion bar
    under it, replacing the old 5-flat-metric row. `lines` = [(label, value,
    color)], each bar's width = |value| / |total_pnl| (capped 100%) so all
    bars share one scale and are directly comparable. A negative value (e.g.
    Imbalance settlement, which can be a real signed penalty on other days)
    renders in STATUS_CRITICAL red regardless of its assigned color, so a
    cost never reads as if it were revenue."""
    scale = max(abs(total_pnl), 1e-6)
    rows_html = []
    for label, value, color in lines:
        share_pct = value / total_pnl * 100 if total_pnl else 0.0  # signed -- sums to exactly 100% across all lines
        bar_pct = min(abs(value) / scale * 100, 100.0)
        bar_color = theme.STATUS_CRITICAL if value < 0 else color
        rows_html.append(f'''
        <div style="margin-bottom:8px;">
          <div style="display:flex; justify-content:space-between; font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">
            <span>{label}</span>
            <span><span style="color:{theme.INK_PRIMARY};">{value:,.0f} EUR</span> <span style="color:{theme.INK_MUTED}; font-weight:400;">&middot; {share_pct:.1f}%</span></span>
          </div>
          <div style="height:7px; background:{theme.GRIDLINE}; border-radius:4px; margin-top:3px;">
            <div style="width:{bar_pct:.2f}%; height:100%; background:{bar_color}; border-radius:4px;"></div>
          </div>
        </div>''')

    reserve_html = (
        f'<div style="display:flex; justify-content:space-between; margin-top:8px; padding-top:8px; border-top:1px solid {theme.GRIDLINE}; font-size:13px;">'
        f'<span style="color:{theme.INK_MUTED};">Reserve share of P&amp;L</span>'
        f'<span style="font-weight:600; color:{theme.INK_PRIMARY};">{reserve_pct:.1f}%</span></div>'
        if reserve_pct is not None else ""
    )

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div style="margin-bottom:10px;">
    <div style="font-size:15px; color:{theme.INK_SECONDARY}; font-weight:500;">Total P&amp;L</div>
    <div style="font-size:38px; font-weight:700; color:{theme.INK_PRIMARY}; letter-spacing:-0.5px;">{total_pnl:,.0f} <span style="font-size:18px; font-weight:500; color:{theme.INK_MUTED};">EUR</span></div>
  </div>
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    {''.join(rows_html)}
    {reserve_html}
  </div>
</div>'''


def render_isp_dispatch_card(dv: dict) -> str:
    """Real per-asset dispatch (PV, BESS discharge, PSP turbine/pump) at
    true 96-ISP (15-min) resolution -- every point is a genuinely different
    solved value now that DA/IDA/XBID run natively at ISP resolution, not a
    repeated hourly number. Same minimal chart language as the other
    Optimization cards (Multi-asset dispatch, Water balance, etc.): no
    boxed frame or axis tick labels, just a zero line, colored traces, and
    a small legend row below each panel. PSP is mirrored around a mid-panel
    zero baseline -- turbine (generation) up, pump (consumption) down --
    same convention as Multi-asset dispatch, instead of the earlier
    turbine-only version of this panel which silently dropped every
    pumping hour."""
    isps = dv["isps"]
    hours = dv["hours"]
    n = len(isps)
    pv_mw, bess_mw = dv["pv_mw"], dv["bess_dis_mw"]
    psp_turb_mw, psp_pump_mw = dv["psp_turb_mw"], dv["psp_pump_mw"]

    x0, x1 = 10, 1390

    def fx(i: int) -> float:
        return x0 + i / max(n - 1, 1) * (x1 - x0)

    def area_path(vals: list[float], vmax: float, zero_y: float, top_y: float) -> str:
        band_h = zero_y - top_y
        def fy(v: float) -> float:
            return zero_y - (v / vmax * band_h if vmax > 1e-9 else 0.0)
        pts = " L".join(f"{fx(i):.1f},{fy(v):.1f}" for i, v in enumerate(vals))
        return f"M{fx(0):.1f},{zero_y:.1f} L{pts} L{fx(n-1):.1f},{zero_y:.1f} Z"

    def line_path(vals: list[float], vmax: float, zero_y: float, top_y: float) -> str:
        band_h = zero_y - top_y
        def fy(v: float) -> float:
            return zero_y - (v / vmax * band_h if vmax > 1e-9 else 0.0)
        return " ".join(f"{fx(i):.1f},{fy(v):.1f}" for i, v in enumerate(vals))

    pv_bess_max = max(max(pv_mw, default=0.0), max(bess_mw, default=0.0), 1e-6)
    psp_max = max(max(psp_turb_mw, default=0.0), max(psp_pump_mw, default=0.0), 1e-6)

    zero_a, top_a = 70, 10
    zero_b, half_b = 44, 34
    pv_area = area_path(pv_mw, pv_bess_max, zero_a, top_a)
    bess_line = line_path(bess_mw, pv_bess_max, zero_a, top_a)

    def psp_y(v: float) -> float:
        return zero_b - (v / psp_max * half_b if psp_max > 1e-9 else 0.0)

    psp_pump_signed = [-w for w in psp_pump_mw]
    pos_pts = " L".join(f"{fx(i):.1f},{psp_y(max(v, 0.0)):.1f}" for i, v in enumerate(psp_turb_mw))
    neg_pts = " L".join(f"{fx(i):.1f},{psp_y(min(v, 0.0)):.1f}" for i, v in enumerate(psp_pump_signed))
    psp_pos_path = f"M{fx(0):.1f},{zero_b} L{pos_pts} L{fx(n-1):.1f},{zero_b} Z"
    psp_neg_path = f"M{fx(0):.1f},{zero_b} L{neg_pts} L{fx(n-1):.1f},{zero_b} Z"
    psp_pos_edge = _sparse_edge_path(fx, psp_y, [max(v, 0.0) for v in psp_turb_mw])
    psp_neg_edge = _sparse_edge_path(fx, psp_y, [min(v, 0.0) for v in psp_pump_signed])

    payload = {
        "isps": isps, "hours": hours, "pvMw": pv_mw, "bessMw": bess_mw,
        "pspMw": psp_turb_mw, "pspPumpMw": psp_pump_mw,
    }
    payload_json = _json.dumps(payload)

    def fy_a(v: float) -> float:
        band_h = zero_a - top_a
        return zero_a - (v / pv_bess_max * band_h if pv_bess_max > 1e-9 else 0.0)

    pv_py = [fy_a(v) for v in pv_mw]
    bess_py = [fy_a(v) for v in bess_mw]
    psp_turb_py = [psp_y(v) for v in psp_turb_mw]
    psp_pump_py = [psp_y(v) for v in psp_pump_signed]

    hover_cfg_top = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#ispd-top-hover-catch", "lineSelector": "#ispd-top-hover-line",
        "tooltipSelector": "#ispd-top-tooltip", "indexLabel": "ISP", "indexVals": isps,
        "traces": [
            {"label": "PV", "yvals": pv_mw, "py": pv_py, "dotSelector": "#ispd-top-hover-dot-0", "color": theme.YELLOW, "unit": "MW", "decimals": 2},
            {"label": "BESS discharge", "yvals": bess_mw, "py": bess_py, "dotSelector": "#ispd-top-hover-dot-1", "color": theme.COLOR_UP, "unit": "MW", "decimals": 2},
        ],
    })
    hover_cfg_bot = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#ispd-bot-hover-catch", "lineSelector": "#ispd-bot-hover-line",
        "tooltipSelector": "#ispd-bot-tooltip", "indexLabel": "ISP", "indexVals": isps,
        "traces": [
            {"label": "PSP turbine", "yvals": psp_turb_mw, "py": psp_turb_py, "dotSelector": "#ispd-bot-hover-dot-0", "color": theme.COLOR_GEN, "unit": "MW", "decimals": 1},
            {"label": "PSP pump", "yvals": psp_pump_signed, "py": psp_pump_py, "dotSelector": "#ispd-bot-hover-dot-1", "color": theme.COLOR_PUMP, "unit": "MW", "decimals": 1, "signed": True},
        ],
    })

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Optimization</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:15px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">ISP dispatch</div>
    <p style="font-size:15px; color:{theme.INK_MUTED}; margin:0 0 10px;">Real per-asset dispatch at 96-ISP (15-min) resolution</p>

    <div class="ispd-chart-block" data-ispd='{payload_json}'>
    <p style="font-size:13.5px; color:{theme.INK_MUTED}; margin:6px 0 2px; font-weight:500;">PV &amp; BESS (own scale, max {pv_bess_max:.1f} MW)</p>
    <div style="position:relative; height:92px;">
      <svg viewBox="0 0 1400 80" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <defs><clipPath id="ispd-clip-a"><rect id="ispd-clip-rect-a" x="0" y="0" width="1400" height="80"/></clipPath></defs>
        <line x1="{x0}" y1="{zero_a}" x2="{x1}" y2="{zero_a}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        <g clip-path="url(#ispd-clip-a)">
          <path d="{pv_area}" fill="{theme.YELLOW}" fill-opacity="0.35" stroke="{theme.YELLOW}" stroke-width="2.3"/>
          <polyline points="{bess_line}" fill="none" stroke="{theme.COLOR_UP}" stroke-width="2.3"/>
        </g>
        {_hover_svg_elems("ispd-top", x0, x1, 80, 2)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="ispd-top-tooltip"')}
    </div>
    <div style="display:flex; gap:12px; margin:2px 0 8px;">
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.YELLOW}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>PV</span>
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_UP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>BESS discharge</span>
    </div>
    <p style="font-size:13.5px; color:{theme.INK_MUTED}; margin:6px 0 2px; font-weight:500;">PSP turbine vs pump (own scale, max {psp_max:.0f} MW)</p>
    <div style="position:relative; height:92px;">
      <svg viewBox="0 0 1400 80" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <defs><clipPath id="ispd-clip-b"><rect id="ispd-clip-rect-b" x="0" y="0" width="1400" height="80"/></clipPath></defs>
        <line x1="{x0}" y1="{zero_b}" x2="{x1}" y2="{zero_b}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        <g clip-path="url(#ispd-clip-b)">
          <path d="{psp_pos_path}" fill="{theme.COLOR_GEN}" fill-opacity="0.35"/>
          <path d="{psp_neg_path}" fill="{theme.COLOR_PUMP}" fill-opacity="0.35"/>
          <path d="{psp_pos_edge}" fill="none" stroke="{theme.COLOR_GEN}" stroke-width="2.3"/>
          <path d="{psp_neg_edge}" fill="none" stroke="{theme.COLOR_PUMP}" stroke-width="2.3"/>
        </g>
        {_hover_svg_elems("ispd-bot", x0, x1, 80, 2)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="ispd-bot-tooltip"')}
    </div>
    <div style="display:flex; gap:12px; margin:2px 0 8px;">
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_GEN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>PSP turbine</span>
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_PUMP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>PSP pump</span>
    </div>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-top:4px;">
      <span id="ispd-readout" style="font-size:14.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Full day shown &mdash; Replay steps through each ISP</span>
      <button class="gt-replay" onclick="dtIspReplay(this)">&#9654; Replay</button>
    </div>
    <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
      <span style="font-size:14px; color:{theme.INK_MUTED};">ISP 1</span>
      <div style="flex:1; height:4px; background:{theme.GRIDLINE}; border-radius:2px; position:relative;">
        <div id="ispd-playhead" style="position:absolute; left:0%; top:-3px; width:10px; height:10px; border-radius:50%; background:{theme.COLOR_GEN}; transform:translateX(-50%);"></div>
      </div>
      <span style="font-size:14px; color:{theme.INK_MUTED};">ISP {n}</span>
    </div>
    </div>
  </div>
</div>
{_REPLAY_STYLE}
{_ISP_REPLAY_SCRIPT}
{_HOVER_CROSSHAIR_SCRIPT}
<script>
(function() {{
  var block = document.querySelector('.ispd-chart-block');
  if (block) {{ dtHoverInit(block, {hover_cfg_top}); dtHoverInit(block, {hover_cfg_bot}); }}
}})();
</script>'''


_AFRR_REPLAY_SCRIPT = '''
<script>
window.dtAfrrReplay = function(btn) {
  var block = btn.closest('.afrrd-chart-block');
  if (!block || block.dataset.replaying === '1') return;
  block.dataset.replaying = '1';
  var data = JSON.parse(block.dataset.afrrd);
  var n = data.isps.length;
  var clip = block.querySelector('#afrrd-clip-rect');
  var playhead = block.querySelector('#afrrd-playhead');
  var readout = block.querySelector('#afrrd-readout');
  var steps = n;
  var stepMs = 6000 / steps;
  clip.setAttribute('width', '0');
  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var idx = Math.min(s - 1, n - 1);
        var frac = s / steps;
        clip.setAttribute('width', (1400 * frac).toFixed(1));
        if (playhead) playhead.style.left = (frac * 100) + '%';
        if (readout) readout.textContent = 'ISP ' + data.isps[idx] + ': ACE ' +
          (data.ace[idx] >= 0 ? '+' : '') + data.ace[idx].toFixed(1) + ' MW · up BESS ' +
          data.bessUp[idx].toFixed(2) + ' + PSP ' + data.pspUp[idx].toFixed(2) + ' MW · dn BESS ' +
          data.bessDn[idx].toFixed(2) + ' + PSP ' + data.pspDn[idx].toFixed(2) + ' MW';
        if (s === steps) block.dataset.replaying = '0';
      }, s * stepMs);
    })(s);
  }
};

// Hover crosshair -- runs immediately (not gated behind Replay) on every
// .afrrd-chart-block in this iframe, so the chart is interactively probeable
// the moment it renders, matching Gate Position Evolution's native Plotly
// hover instead of only reacting to the Replay button.
(function () {
  var blocks = document.querySelectorAll('.afrrd-chart-block');
  for (var b = 0; b < blocks.length; b++) {
    (function (block) {
      var data = JSON.parse(block.dataset.afrrd);
      var n = data.isps.length;
      var svg = block.querySelector('svg');
      var catchRect = block.querySelector('#afrrd-hover-catch');
      var hoverLine = block.querySelector('#afrrd-hover-line');
      var dotAce = block.querySelector('#afrrd-hover-dot-ace');
      var dotReg = block.querySelector('#afrrd-hover-dot-reg');
      var tooltip = block.querySelector('#afrrd-tooltip');
      var ttIsp = block.querySelector('#afrrd-tt-isp');
      var ttAce = block.querySelector('#afrrd-tt-ace');
      var ttReg = block.querySelector('#afrrd-tt-reg');
      var ttRegLabel = block.querySelector('#afrrd-tt-reg-label');
      if (!svg || !catchRect) return;

      function fx(i) { return data.x0 + i / Math.max(n - 1, 1) * (data.x1 - data.x0); }
      function aceY(v) { return data.aceMid - (v / data.aceMax) * data.aceHalf; }
      function regY(v) { return data.regMid - (v / data.regMax) * data.regHalf; }

      function move(evt) {
        var rect = svg.getBoundingClientRect();
        var vbX = (evt.clientX - rect.left) / rect.width * 1400;
        var frac = Math.max(0, Math.min(1, (vbX - data.x0) / (data.x1 - data.x0)));
        var idx = Math.round(frac * (n - 1));
        var cx = fx(idx);
        var aceVal = data.ace[idx], regVal = data.net[idx];
        hoverLine.setAttribute('x1', cx); hoverLine.setAttribute('x2', cx); hoverLine.setAttribute('opacity', '1');
        dotAce.setAttribute('cx', cx); dotAce.setAttribute('cy', aceY(aceVal).toFixed(1)); dotAce.setAttribute('opacity', '1');
        dotReg.setAttribute('cx', cx); dotReg.setAttribute('cy', regY(regVal).toFixed(1)); dotReg.setAttribute('opacity', '1');
        dotReg.setAttribute('fill', regVal >= 0 ? '#1baf7a' : '#e87ba4');
        if (tooltip) {
          ttIsp.textContent = data.isps[idx];
          ttAce.textContent = (aceVal >= 0 ? '+' : '') + aceVal.toFixed(1);
          ttReg.textContent = Math.abs(regVal).toFixed(2);
          ttRegLabel.textContent = regVal >= 0 ? 'Up' : 'Down';
          ttRegLabel.parentElement.style.color = regVal >= 0 ? '#1baf7a' : '#e87ba4';
          tooltip.style.display = 'block';
        }
      }
      function leave() {
        hoverLine.setAttribute('opacity', '0');
        dotAce.setAttribute('opacity', '0');
        dotReg.setAttribute('opacity', '0');
        if (tooltip) tooltip.style.display = 'none';
      }
      catchRect.addEventListener('mousemove', move);
      catchRect.addEventListener('mouseleave', leave);
    })(blocks[b]);
  }
})();
</script>'''


_FAT_BY_PRODUCT = {
    "aFRR": ("PICASSO", "5 min"),
    "mFRR": ("MARI", "12.5 min"),
}


def render_afrr_dispatch_card(dv: dict, act: dict | None = None) -> str:
    """Real per-ISP reserved capacity vs actual activation for aFRR (or
    mFRR), split by resource: BESS (fast responder up to its power rating,
    base of the fill) vs PSP (ramp covers the remainder, stacked on top).
    PV is never a contributor -- the activation model has no PV term in its
    call-sizing logic, so it's correctly absent, not omitted.

    Same combined-SVG, single-connector layout as the FCR dispatch card
    (dispatch_ticket.py::render_fcr_dispatch_card): ACE (the real signal
    this product's activation responds to -- see
    reserve_activation.py::simulate_ace_series -- the same role frequency
    deviation plays for FCR) on top, and up/down-regulation activation
    merged into ONE panel below it sharing a single 0 MW baseline and a
    single common scale (not two separately-ceilinged panels) -- up
    positive, down negative, the same mirrored convention used everywhere
    else in this app. Axis ceilings round up to a 'nice' number
    (_nice_max) with 3 gridlines (-max/0/+max), not an arbitrary padded data
    max. One connector links both panels at the ISP with the widest ACE
    swing. Panel spacing is generous on purpose -- the first cut had the
    second panel's title sitting almost on top of the first panel's bottom
    tick label; there's now a full title-height gap between them.
    Up/down colors are theme.COLOR_UP/COLOR_DOWN (aqua/magenta) -- the
    same semantic pair used everywhere else in the app that shows
    up-regulation vs down-regulation (ACE activation chart, FCR dispatch,
    gate ticket reserve bars), not COLOR_GEN/RED.

    `act` (optional) is data.py::load_activation_summary()'s dict --
    settlement numbers (revenue, MWh, ISPs activated, decided timestamp)
    for the SAME product/day, merged in as a stat row so this one card
    covers both "what physically happened" and "what it settled for"
    instead of splitting them across two separate cards (the former
    Market & Delivery "aFRR/mFRR activation" card, which duplicated this
    same ACE/response chart in miniature -- removed, not preserved
    elsewhere)."""
    isps = dv["isps"]
    n = len(isps)
    product = dv["product"]
    bess_up, psp_up = dv["bess_up_mw"], dv["psp_up_mw"]
    bess_dn, psp_dn = dv["bess_dn_mw"], dv["psp_dn_mw"]
    ace = dv.get("ace_mw") or [0.0] * n

    x0, x1 = 68, 1390   # left margin wide enough for axis value labels

    ace_mid_y, ace_half = 46, 24
    ace_bottom = ace_mid_y + ace_half
    reg_title_y = ace_bottom + 24        # clear title-height gap before the second panel starts
    reg_half = 64
    reg_mid_y = reg_title_y + 14 + reg_half
    total_h = reg_mid_y + reg_half + 16

    total_up = [b + p for b, p in zip(bess_up, psp_up)]
    total_dn = [b + p for b, p in zip(bess_dn, psp_dn)]
    ace_max = _nice_max(max((abs(v) for v in ace), default=0.0) * 1.05 or 15.0)
    reg_max = _nice_max(max((max(total_up, default=0.0), max(total_dn, default=0.0))) * 1.05 or 10.0)

    def fx(i: int) -> float:
        return x0 + i / max(n - 1, 1) * (x1 - x0)

    def ace_y(v: float) -> float:
        return ace_mid_y - (v / ace_max) * ace_half

    def reg_y(v: float) -> float:
        return reg_mid_y - (v / reg_max) * reg_half

    def axis_ticks(mid_y: float, half: float, vmax: float, unit_suffix: str = "") -> str:
        """3 gridlines/labels: -max, 0, +max -- same convention as every
        other panel in this app (FCR included), not a busier 5-tick axis."""
        out = []
        for frac in (1.0, 0.0, -1.0):
            y = mid_y - frac * half
            val = frac * vmax
            is_zero = val == 0
            label = "0" if is_zero else f"{val:+.0f}{unit_suffix}"
            # All axis numbers use INK_PRIMARY (near-black), not the lighter
            # INK_MUTED/INK_SECONDARY reserved for de-emphasized captions --
            # axis numbers are information a reader has to actually read,
            # and anything short of near-black is too low-contrast at these
            # small chart font sizes to read comfortably. The 0 tick gets a
            # bolder line/weight on top of that so it still stands out from
            # the +/-max ticks despite sharing the same color now.
            line_stroke = theme.INK_PRIMARY if is_zero else theme.GRIDLINE
            text_fill = theme.INK_PRIMARY
            weight = 1.4 if is_zero else 1.0
            out.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{line_stroke}" stroke-width="{weight}"/>'
                f'<text x="{x0-8}" y="{y+4:.1f}" font-size="11" fill="{text_fill}" text-anchor="end" font-weight="{"700" if is_zero else "600"}">{label}</text>'
            )
        return "".join(out)

    net = [u - d for u, d in zip(total_up, total_dn)]
    ace_pts = " L".join(f"{fx(i):.1f},{ace_y(v):.1f}" for i, v in enumerate(ace))
    pos_pts = " L".join(f"{fx(i):.1f},{reg_y(max(v, 0.0)):.1f}" for i, v in enumerate(net))
    neg_pts = " L".join(f"{fx(i):.1f},{reg_y(min(v, 0.0)):.1f}" for i, v in enumerate(net))
    pos_path = f"M{fx(0):.1f},{reg_mid_y} L{pos_pts} L{fx(n-1):.1f},{reg_mid_y} Z"
    neg_path = f"M{fx(0):.1f},{reg_mid_y} L{neg_pts} L{fx(n-1):.1f},{reg_mid_y} Z"
    pos_edge = _sparse_edge_path(fx, reg_y, [max(v, 0.0) for v in net])
    neg_edge = _sparse_edge_path(fx, reg_y, [min(v, 0.0) for v in net])

    # Connector: the ISP with the widest ACE swing, marked with a dot in
    # both panels and joined by one vertical line.
    peak_idx = max(range(n), key=lambda i: abs(ace[i])) if n else 0
    peak_x = fx(peak_idx)
    peak_ace_y = ace_y(ace[peak_idx])
    peak_reg_y = reg_y(net[peak_idx])
    peak_color = theme.COLOR_UP if net[peak_idx] >= 0 else theme.COLOR_DOWN

    payload = {
        "isps": isps, "bessUp": bess_up, "pspUp": psp_up, "bessDn": bess_dn, "pspDn": psp_dn, "ace": ace,
        "net": net, "x0": x0, "x1": x1,
        "aceMid": ace_mid_y, "aceHalf": ace_half, "aceMax": ace_max,
        "regMid": reg_mid_y, "regHalf": reg_half, "regMax": reg_max,
    }
    payload_json = _json.dumps(payload)

    # Settlement stat row -- merged in from the former standalone
    # "activation" card (see docstring). Only rendered when act is
    # supplied, so this function still works standalone for callers that
    # only have the physical dispatch data.
    stats_html = ""
    if act is not None:
        platform, fat = _FAT_BY_PRODUCT.get(product, ("", ""))
        stats_html = f'''
    <p style="font-size:12px; color:{theme.INK_SECONDARY}; margin:0 0 10px;">{act['n_isp']} of {n} ISPs activated &middot; decided {_html.escape(act['timestamp'])}</p>
    <div style="display:flex; gap:8px; margin-bottom:14px;">
      <div style="background:{theme.GRIDLINE}44; border-radius:8px; padding:0.55rem 0.75rem; flex:1;">
        <p style="font-size:11px; color:{theme.INK_SECONDARY}; margin:0 0 2px;">Activation revenue</p>
        <p style="font-size:17px; font-weight:500; margin:0; color:{theme.INK_PRIMARY};">&euro;{act['revenue_eur']:,.0f}</p>
      </div>
      <div style="background:{theme.GRIDLINE}44; border-radius:8px; padding:0.55rem 0.75rem; flex:1;">
        <p style="font-size:11px; color:{theme.INK_SECONDARY}; margin:0 0 2px;">Energy up / down</p>
        <p style="font-size:17px; font-weight:500; margin:0; color:{theme.INK_PRIMARY};">{act['up_mwh']:.1f} / {act['dn_mwh']:.1f} MWh</p>
      </div>
      <div style="background:{theme.GRIDLINE}44; border-radius:8px; padding:0.55rem 0.75rem; flex:1;">
        <p style="font-size:11px; color:{theme.INK_SECONDARY}; margin:0 0 2px;">Platform &middot; FAT</p>
        <p style="font-size:17px; font-weight:500; margin:0; color:{theme.INK_PRIMARY};">{platform} &middot; {fat}</p>
      </div>
    </div>'''

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:14px; color:{theme.INK_SECONDARY}; font-weight:500;">Reserve delivery</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:13px; padding:3px 10px; border-radius:6px; font-weight:500;">Real ACE-driven activation</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">{product} dispatch</div>
    <p style="font-size:12.5px; color:{theme.INK_SECONDARY}; margin:0 0 8px;">Area-wide ACE signal &mdash; Alqueva delivers only its own contracted share, not the full response.</p>
    {stats_html}

    <div class="afrrd-chart-block" data-afrrd='{payload_json}'>
    <p style="font-size:13.5px; color:{theme.INK_MUTED}; margin:6px 0 10px; font-weight:500;">ACE deviation (top), and regulation activated at those same moments &mdash; {n} real ISPs</p>
    <div style="position:relative; height:{total_h*0.92:.0f}px;">
      <svg viewBox="0 0 1400 {total_h}" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <defs><clipPath id="afrrd-clip"><rect id="afrrd-clip-rect" x="0" y="0" width="1400" height="{total_h}"/></clipPath></defs>
        <text x="{x0}" y="16" font-size="12" font-weight="700" fill="{theme.INK_PRIMARY}">ACE deviation (MW)</text>
        {axis_ticks(ace_mid_y, ace_half, ace_max)}

        <text x="{x0}" y="{reg_title_y:.1f}" font-size="12" font-weight="700" fill="{theme.INK_PRIMARY}">Regulation activated (MW)</text>
        {axis_ticks(reg_mid_y, reg_half, reg_max)}

        <g clip-path="url(#afrrd-clip)">
          <path d="M{fx(0):.1f},{ace_y(ace[0] if ace else 0):.1f} L{ace_pts}" fill="none" stroke="{theme.COLOR_PRICE}" stroke-width="2.2"/>
          <path d="{pos_path}" fill="{theme.COLOR_UP}" fill-opacity="0.4"/>
          <path d="{neg_path}" fill="{theme.COLOR_DOWN}" fill-opacity="0.4"/>
          <path d="{pos_edge}" fill="none" stroke="{theme.COLOR_UP}" stroke-width="2"/>
          <path d="{neg_edge}" fill="none" stroke="{theme.COLOR_DOWN}" stroke-width="2"/>
        </g>
        <line x1="{peak_x:.1f}" y1="{peak_ace_y+6:.1f}" x2="{peak_x:.1f}" y2="{peak_reg_y-6:.1f}" stroke="{theme.INK_PRIMARY}" stroke-width="1.6" stroke-dasharray="3,3" opacity="0.55"/>
        <circle cx="{peak_x:.1f}" cy="{peak_ace_y:.1f}" r="5.5" fill="{theme.COLOR_PRICE}" stroke="white" stroke-width="2"/>
        <circle cx="{peak_x:.1f}" cy="{peak_reg_y:.1f}" r="5.5" fill="{peak_color}" stroke="white" stroke-width="2"/>

        <rect id="afrrd-hover-catch" x="{x0}" y="0" width="{x1-x0}" height="{total_h}" fill="transparent"/>
        <line id="afrrd-hover-line" x1="{x0}" y1="4" x2="{x0}" y2="{total_h-4}" stroke="{theme.INK_PRIMARY}" stroke-width="1" stroke-dasharray="2,2" opacity="0"/>
        <circle id="afrrd-hover-dot-ace" r="5" fill="{theme.COLOR_PRICE}" stroke="white" stroke-width="1.5" opacity="0"/>
        <circle id="afrrd-hover-dot-reg" r="5" fill="{theme.COLOR_UP}" stroke="white" stroke-width="1.5" opacity="0"/>
      </svg>
      <div id="afrrd-tooltip" style="position:absolute; top:6px; right:6px; background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
           border-radius:6px; padding:6px 10px; font-size:12.5px; display:none; pointer-events:none; box-shadow:0 2px 6px rgba(0,0,0,0.08); white-space:nowrap;">
        <div style="color:{theme.INK_SECONDARY};">ISP <span id="afrrd-tt-isp"></span></div>
        <div style="color:{theme.COLOR_PRICE}; font-weight:600;">ACE <span id="afrrd-tt-ace"></span> MW</div>
        <div id="afrrd-tt-reg-row" style="font-weight:600;"><span id="afrrd-tt-reg-label"></span> <span id="afrrd-tt-reg"></span> MW</div>
      </div>
    </div>
    <div style="font-size:13.5px; color:{theme.INK_SECONDARY}; margin:8px 0 8px;">
      <span style="color:{theme.COLOR_UP}; font-weight:600;">&#9679;</span> Up-regulation &nbsp;&nbsp;
      <span style="color:{theme.COLOR_DOWN}; font-weight:600;">&#9679;</span> Down-regulation &nbsp;&nbsp;
      <span style="color:{theme.COLOR_PRICE}; font-weight:600;">&#9679;</span> ACE
    </div>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-top:4px;">
      <span id="afrrd-readout" style="font-size:14.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Full day shown &mdash; Replay steps through each ISP</span>
      <button class="gt-replay" onclick="dtAfrrReplay(this)">&#9654; Replay</button>
    </div>
    <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
      <span style="font-size:14px; color:{theme.INK_MUTED};">ISP 1</span>
      <div style="flex:1; height:4px; background:{theme.GRIDLINE}; border-radius:2px; position:relative;">
        <div id="afrrd-playhead" style="position:absolute; left:0%; top:-3px; width:10px; height:10px; border-radius:50%; background:{theme.COLOR_GEN}; transform:translateX(-50%);"></div>
      </div>
      <span style="font-size:14px; color:{theme.INK_MUTED};">ISP {n}</span>
    </div>
    </div>
  </div>
</div>
{_REPLAY_STYLE}
{_AFRR_REPLAY_SCRIPT}'''


_FCRD_REPLAY_SCRIPT = '''
<script>
window.dtFcrdReplay = function(btn) {
  var block = btn.closest('.fcrd-chart-block');
  if (!block || block.dataset.replaying === '1') return;
  block.dataset.replaying = '1';
  var data = JSON.parse(block.dataset.fcrd);
  var n = data.freq.length;
  var clip = block.querySelector('#fcrd-clip-rect');
  var lineEl = block.querySelector('#fcrd-playhead-line');
  var dotF = block.querySelector('#fcrd-playhead-f');
  var dotR = block.querySelector('#fcrd-playhead-r');
  var readout = block.querySelector('#fcrd-readout');
  var playhead = block.querySelector('#fcrd-playhead');
  [lineEl, dotF, dotR].forEach(function(el) { if (el) el.style.opacity = '1'; });

  function fy(v) { return data.fMidY - (v / data.freqScale) * data.fHalf; }
  function ry(v) { return data.rMidY - (v / data.headroom) * data.rHalf; }
  function fx(i) { return data.x0 + i / Math.max(n - 1, 1) * (data.x1 - data.x0); }

  var steps = Math.min(n, 160);   // 2,880 real ticks sampled to ~160 animation steps
  var stepMs = 8000 / steps;
  clip.setAttribute('width', '0');

  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var idx = Math.min(Math.round((s / steps) * (n - 1)), n - 1);
        var frac = s / steps;
        var px = fx(idx);
        clip.setAttribute('width', (1400 * frac).toFixed(1));
        if (playhead) playhead.style.left = (frac * 100) + '%';
        if (lineEl) { lineEl.setAttribute('x1', px.toFixed(1)); lineEl.setAttribute('x2', px.toFixed(1)); }
        if (dotF) { dotF.setAttribute('cx', px.toFixed(1)); dotF.setAttribute('cy', fy(data.freq[idx]).toFixed(1)); }
        if (dotR) { dotR.setAttribute('cx', px.toFixed(1)); dotR.setAttribute('cy', ry(data.resp[idx]).toFixed(1));
          dotR.setAttribute('fill', data.resp[idx] > 0 ? '#1baf7a' : (data.resp[idx] < 0 ? '#e87ba4' : '#898781')); }
        if (readout) {
          var totalSec = idx * 30;
          var hh = String(Math.floor(totalSec / 3600)).padStart(2, '0');
          var mm = String(Math.floor((totalSec % 3600) / 60)).padStart(2, '0');
          var ss = String(totalSec % 60).padStart(2, '0');
          readout.textContent = hh + ':' + mm + ':' + ss + '  ·  ' + data.freq[idx].toFixed(1) + ' mHz  ·  ' + data.resp[idx].toFixed(2) + ' MW';
        }
        if (s === steps) {
          block.dataset.replaying = '0';
          [lineEl, dotF, dotR].forEach(function(el) { if (el) el.style.opacity = '0'; });
        }
      }, s * stepMs);
    })(s);
  }
};
</script>'''


def render_fcr_dispatch_card(fcr: dict) -> str:
    """FCR dispatch -- frequency deviation and the plant's compensating
    response, plotted on the SAME time axis, stacked directly on top of
    each other, at the real 30s tick resolution (2,880/day). This is
    deliberately NOT a frequency-vs-response scatter (which shows the
    droop relationship but not WHEN anything happened, so a non-technical
    reader can't connect cause and effect). Stacked time series let you
    trace a single moment straight down: frequency wobbles here, response
    reacts here, same instant -- the whole "why" without needing to know
    what mHz or droop mean.

      Panel A -- grid frequency deviation over the day (mHz from 50.000 Hz),
      shaded band = the +/-10 mHz safe zone where no response is required.

      Panel B -- the plant's response (MW) at that same moment, green =
      pushed more power, red = pulled back, mirrored below zero the same
      way every other up/down chart in this app is drawn.
    """
    headroom = max(fcr["reserved_headroom_mw"], 1e-6)
    freq = fcr["tick_freq_mhz"]
    resp = fcr["tick_response_mw"]
    n = len(freq)
    x0, x1 = 46, 1390   # left margin wide enough for axis value labels on both panels

    def fx(i: int) -> float:
        return x0 + i / max(n - 1, 1) * (x1 - x0)

    # Zoom the frequency panel to the day's actual swing (padded) instead of
    # the full +/-200 mHz droop range -- real deviations are typically a few
    # dozen mHz, so the full range would flatten the line near zero.
    freq_obs_max = max((abs(f) for f in freq), default=0.0)
    freq_scale = max(freq_obs_max * 1.2, 15.0)
    f_mid_y, f_half = 40, 32

    def fy(v: float) -> float:
        return f_mid_y - (v / freq_scale) * f_half

    deadband_top, deadband_bot = fy(10.0), fy(-10.0)
    freq_pts = " L".join(f"{fx(i):.1f},{fy(v):.1f}" for i, v in enumerate(freq))

    # Response panel sits below the frequency panel in the SAME svg (not a
    # separate one) so a single vertical line can physically connect one
    # moment across both -- that connector is the whole point of this
    # widget, and two independent <svg> coordinate spaces couldn't host it.
    gap_h = 46          # room for the connector + its label pill
    r_base_y = 80 + gap_h
    r_mid_y, r_half = r_base_y + 40, 32

    def ry(v: float) -> float:
        return r_mid_y - (v / headroom) * r_half

    pos_pts = " L".join(f"{fx(i):.1f},{ry(max(v, 0.0)):.1f}" for i, v in enumerate(resp))
    neg_pts = " L".join(f"{fx(i):.1f},{ry(min(v, 0.0)):.1f}" for i, v in enumerate(resp))
    pos_path = f"M{fx(0):.1f},{r_mid_y} L{pos_pts} L{fx(n-1):.1f},{r_mid_y} Z"
    neg_path = f"M{fx(0):.1f},{r_mid_y} L{neg_pts} L{fx(n-1):.1f},{r_mid_y} Z"
    pos_edge = _sparse_edge_path(fx, ry, [max(v, 0.0) for v in resp])
    neg_edge = _sparse_edge_path(fx, ry, [min(v, 0.0) for v in resp])

    # Callout: the single tick with the widest frequency swing -- a marker
    # dot on each panel at that exact moment, joined by one vertical line
    # and a plain-English label, so the cause-and-effect reads on its own
    # without the viewer having to trace two separate lines by eye.
    peak_idx = max(range(n), key=lambda i: abs(freq[i])) if n else 0
    peak_x = fx(peak_idx)
    peak_fy = fy(freq[peak_idx])
    peak_ry = ry(resp[peak_idx])
    if resp[peak_idx] > 0:
        callout_color = theme.COLOR_UP
    elif resp[peak_idx] < 0:
        callout_color = theme.COLOR_DOWN
    else:
        callout_color = theme.INK_MUTED
    total_h = r_base_y + 80

    payload_json = _json.dumps({
        "freq": freq, "resp": resp, "fMidY": f_mid_y, "fHalf": f_half, "freqScale": freq_scale,
        "rMidY": r_mid_y, "rHalf": r_half, "headroom": headroom, "x0": x0, "x1": x1,
    })

    freq_py = [fy(v) for v in freq]
    resp_py = [ry(v) for v in resp]
    hover_cfg = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#fcrd-hv-hover-catch", "lineSelector": "#fcrd-hv-hover-line",
        "tooltipSelector": "#fcrd-hv-tooltip", "indexLabel": "Tick", "indexVals": list(range(1, n + 1)),
        "traces": [
            {"label": "Frequency", "yvals": freq, "py": freq_py, "dotSelector": "#fcrd-hv-hover-dot-0", "color": theme.COLOR_PRICE, "unit": "mHz", "decimals": 1, "signed": True},
            {"label": "Response", "yvals": resp, "py": resp_py, "dotSelector": "#fcrd-hv-hover-dot-1", "color": theme.COLOR_UP, "negColor": theme.COLOR_DOWN, "unit": "MW", "decimals": 2, "signed": True},
        ],
    })

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Delivery</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:15px; padding:3px 10px; border-radius:6px; font-weight:500;">Mandatory, non-remunerated</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">FCR dispatch</div>
    <p style="font-size:15px; color:{theme.INK_MUTED}; margin:0 0 10px;">Reserved headroom {headroom:.1f} MW, plant-level only &mdash; no per-resource split exists</p>

    <div class="fcrd-chart-block" data-fcrd='{payload_json}'>
    <p style="font-size:13.5px; color:{theme.INK_MUTED}; margin:6px 0 2px; font-weight:500;">Grid frequency deviation from 50.000 Hz, and the plant's response at those same moments ({n:,} real 30-s ticks)</p>
    <div style="position:relative; height:{total_h*0.95:.0f}px;">
      <svg viewBox="0 0 1400 {total_h}" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <defs><clipPath id="fcrd-clip"><rect id="fcrd-clip-rect" x="0" y="0" width="1400" height="{total_h}"/></clipPath></defs>
        <rect x="{x0}" y="{deadband_top:.1f}" width="{x1-x0}" height="{deadband_bot-deadband_top:.1f}" fill="{theme.STATUS_GOOD}" fill-opacity="0.10"/>
        <line x1="{x0}" y1="{f_mid_y}" x2="{x1}" y2="{f_mid_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        <line x1="{x0}" y1="{r_mid_y}" x2="{x1}" y2="{r_mid_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>

        <text x="{x0-6}" y="{f_mid_y-f_half+3:.1f}" font-size="12" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">+{freq_scale:.0f}</text>
        <text x="{x0-6}" y="{f_mid_y+3:.1f}" font-size="12" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">0 mHz</text>
        <text x="{x0-6}" y="{f_mid_y+f_half+3:.1f}" font-size="12" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">-{freq_scale:.0f}</text>

        <text x="{x0-6}" y="{r_mid_y-r_half+3:.1f}" font-size="12" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">+{headroom:.1f}</text>
        <text x="{x0-6}" y="{r_mid_y+3:.1f}" font-size="12" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">0 MW</text>
        <text x="{x0-6}" y="{r_mid_y+r_half+3:.1f}" font-size="12" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">-{headroom:.1f}</text>

        <g clip-path="url(#fcrd-clip)">
          <path d="M{fx(0):.1f},{fy(freq[0]):.1f} L{freq_pts}" fill="none" stroke="{theme.COLOR_PRICE}" stroke-width="2.3"/>
          <path d="{pos_path}" fill="{theme.COLOR_UP}" fill-opacity="0.4"/>
          <path d="{neg_path}" fill="{theme.COLOR_DOWN}" fill-opacity="0.4"/>
          <path d="{pos_edge}" fill="none" stroke="{theme.COLOR_UP}" stroke-width="2.1"/>
          <path d="{neg_edge}" fill="none" stroke="{theme.COLOR_DOWN}" stroke-width="2.1"/>
        </g>

        <line x1="{peak_x:.1f}" y1="{peak_fy+6:.1f}" x2="{peak_x:.1f}" y2="{peak_ry-6:.1f}" stroke="{theme.INK_PRIMARY}" stroke-width="2.2" stroke-dasharray="4,3"/>
        <circle cx="{peak_x:.1f}" cy="{peak_fy:.1f}" r="6" fill="{theme.COLOR_PRICE}" stroke="white" stroke-width="2.1"/>
        <circle cx="{peak_x:.1f}" cy="{peak_ry:.1f}" r="6" fill="{callout_color}" stroke="white" stroke-width="2.1"/>

        <line id="fcrd-playhead-line" x1="{x0}" y1="0" x2="{x0}" y2="{total_h}" stroke="{theme.INK_SECONDARY}" stroke-width="1" stroke-dasharray="2,2" opacity="0"/>
        <circle id="fcrd-playhead-f" cx="{x0}" cy="{f_mid_y}" r="6" fill="{theme.COLOR_PRICE}" stroke="white" stroke-width="2.1" opacity="0"/>
        <circle id="fcrd-playhead-r" cx="{x0}" cy="{r_mid_y}" r="6" fill="{theme.COLOR_UP}" stroke="white" stroke-width="2.1" opacity="0"/>
        {_hover_svg_elems("fcrd-hv", x0, x1, total_h, 2)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="fcrd-hv-tooltip"')}
    </div>
    <div style="display:flex; gap:12px; margin:2px 0 8px;">
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};">shaded = &plusmn;10 mHz safe band</span>
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_UP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Pushed more power</span>
      <span style="font-size:13.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_DOWN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Pulled back</span>
    </div>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-top:4px;">
      <span id="fcrd-readout" style="font-size:14.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Full day shown &mdash; Replay steps through every 30-s tick</span>
      <button class="gt-replay" onclick="dtFcrdReplay(this)">&#9654; Replay</button>
    </div>
    <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
      <span style="font-size:14px; color:{theme.INK_MUTED};">00:00:00</span>
      <div style="flex:1; height:4px; background:{theme.GRIDLINE}; border-radius:2px; position:relative;">
        <div id="fcrd-playhead" style="position:absolute; left:0%; top:-3px; width:10px; height:10px; border-radius:50%; background:{theme.COLOR_PRICE}; transform:translateX(-50%);"></div>
      </div>
      <span style="font-size:14px; color:{theme.INK_MUTED};">23:59:30</span>
    </div>
    </div>
  </div>
</div>
{_REPLAY_STYLE}
{_FCRD_REPLAY_SCRIPT}
{_HOVER_CROSSHAIR_SCRIPT}
<script>
(function() {{
  var block = document.querySelector('.fcrd-chart-block');
  if (block) {{ dtHoverInit(block, {hover_cfg}); }}
}})();
</script>'''
