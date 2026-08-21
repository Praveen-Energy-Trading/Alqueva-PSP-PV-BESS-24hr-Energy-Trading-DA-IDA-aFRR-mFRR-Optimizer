"""delivery_ticket.py — renders phase 4A/4B/4C delivery cards (RT dispatch,
aFRR activation, mFRR activation). Distinct from gate_ticket.py: these
phases don't "decide" Submitted/Held, they simulate delivery of whatever
was already committed, so the card shape is metrics + a trace, not a
decision pill. See dashboard/data.py::load_rt_delivery /
load_activation_summary for the fields.
"""
from __future__ import annotations

import html as _html
import json as _json

import theme
from dispatch_ticket import _HOVER_CROSSHAIR_SCRIPT, _hover_svg_elems, _hover_tooltip_div, _sparse_edge_path

# Shared by both delivery cards below. Each render_*_card() is its own
# components.html() iframe (a separate document), so unlike gate_ticket.py's
# shared <style>/<script> in one page, this has to be re-emitted per card --
# but there's still only one strip per iframe, so a plain id-free onclick
# scoped via closest('.dt-strip-scope') is enough (no XBID-style multi-block
# collision to worry about here).
_REPLAY_STYLE = f'<style>.gt-replay {{ font-size:11px; padding:2px 10px; border:1px solid {theme.GRIDLINE}; border-radius:6px; background:{theme.SURFACE}; cursor:pointer; }}</style>'
_REPLAY_SCRIPT = '''
<script>
window.dtReplay = function(btn) {
  // .dt-strip-scope is the next sibling of the header row btn sits in
  // (not an ancestor of btn), so closest() wouldn't find it -- walk to
  // the header row's own next sibling instead.
  var scope = btn.parentElement.nextElementSibling;
  if (!scope || scope.dataset.replaying === '1') return;
  scope.dataset.replaying = '1';
  var bars = Array.prototype.slice.call(scope.querySelectorAll('[data-kind="dt-bar"]'));
  bars.forEach(function(el) {
    var isDn = el.getAttribute('data-dir') === 'dn';
    el.setAttribute('height', '0');
    if (!isDn) el.setAttribute('y', (parseFloat(el.getAttribute('data-final-y')) + parseFloat(el.getAttribute('data-final-h'))).toFixed(1));
  });
  var n = bars.length || 1;
  var stepMs = 1400 / n;
  bars.forEach(function(el, i) {
    setTimeout(function() {
      el.setAttribute('y', el.getAttribute('data-final-y'));
      el.setAttribute('height', el.getAttribute('data-final-h'));
      if (i === bars.length - 1) scope.dataset.replaying = '0';
    }, i * stepMs);
  });
  if (!bars.length) scope.dataset.replaying = '0';
};
</script>'''


def _deviation_strip(rows: list[dict], plot_x0: float, plot_x1: float, max_dev: float) -> str:
    """Deviation-per-ISP bar row: over-delivered (actual > scheduled) in
    aqua above the zero line, under-delivered in magenta below it — the
    same up/down color convention used for reserve activation elsewhere in
    this dashboard. Scaled to the ticket's own max_deviation_mw so a single
    outlier ISP doesn't flatten every other bar to invisible."""
    n = len(rows)
    slot_w = (plot_x1 - plot_x0) / max(n, 1)
    bar_w = max(min(6.0, slot_w * 0.75), 2.0)
    # Strip is now 100px tall (was 24px) -- at the old height even a bar at
    # 100% of max_dev was only ~11px, barely more than a hairline. Bigger
    # bars read as an actual chart instead of a thin sparkline squeezed
    # under the metric cards.
    zero_y, half_h = 50, 46
    scale = half_h / max(max_dev, 1e-6)

    bars = []
    for i, r in enumerate(rows):
        dev = r["actual_mw"] - r["scheduled_mw"]
        if abs(dev) < 1e-9:
            continue
        x = plot_x0 + i * slot_w + (slot_w - bar_w) / 2
        h = min(abs(dev) * scale, half_h)
        if dev > 0:
            y = zero_y - h
            bars.append(f'<rect data-kind="dt-bar" data-final-y="{y:.1f}" data-final-h="{h:.1f}" data-dir="up" '
                        f'x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="1" fill="{theme.COLOR_UP}"/>')
        else:
            bars.append(f'<rect data-kind="dt-bar" data-final-y="{zero_y:.1f}" data-final-h="{h:.1f}" data-dir="dn" '
                        f'x="{x:.1f}" y="{zero_y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="1" fill="{theme.COLOR_DOWN}"/>')

    hover_cfg = _json.dumps({
        "n": n, "x0": plot_x0, "x1": plot_x1, "viewW": 1400,
        "catchSelector": "#dt-strip-hover-catch", "lineSelector": "#dt-strip-hover-line",
        "tooltipSelector": "#dt-strip-tooltip", "indexLabel": "ISP", "indexVals": list(range(1, n + 1)),
        "traces": [{"label": "Deviation", "yvals": [r["actual_mw"] - r["scheduled_mw"] for r in rows],
                    "py": [r["actual_mw"] - r["scheduled_mw"] for r in rows], "color": theme.COLOR_UP,
                    "negColor": theme.COLOR_DOWN, "unit": "MW", "decimals": 2, "signed": True}],
    })

    max_label = f"{max_dev:.1f}"
    # HTML overlay for the +/-max labels, not SVG <text> -- same reasoning
    # as gate_ticket.py's _bars_svg: SVG's uniform scaling shrinks
    # font-size along with fitting the container's unpredictable width,
    # which is what made these unreadably small once the card widened.
    return f'''
    <div style="display:flex; align-items:center; justify-content:space-between; margin:10px 0 4px;">
      <p style="font-size:11px; color:{theme.INK_SECONDARY}; margin:0;">Deviation per ISP (MW)</p>
      <button class="gt-replay" onclick="dtReplay(this)">&#9654; Replay</button>
    </div>
    <div class="dt-strip-scope" style="position:relative; height:100px; padding-left:30px; box-sizing:border-box;">
      <svg viewBox="0 0 1400 100" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{plot_x0}" y1="{zero_y}" x2="{plot_x1}" y2="{zero_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(bars)}
        {_hover_svg_elems("dt-strip", plot_x0, plot_x1, 100, 0)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="dt-strip-tooltip"')}
      <div style="position:absolute; left:0; top:0; font-size:12px; font-weight:600; color:{theme.INK_PRIMARY};">+{max_label}</div>
      <div style="position:absolute; left:0; top:50%; transform:translateY(-50%); font-size:12px; font-weight:600; color:{theme.INK_PRIMARY};">0</div>
      <div style="position:absolute; left:0; bottom:0; font-size:12px; font-weight:600; color:{theme.INK_PRIMARY};">-{max_label}</div>
    </div>
    <div style="display:flex; gap:14px; margin-top:2px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_UP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Over-delivered</span>
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_DOWN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Under-delivered</span>
    </div>
    {_HOVER_CROSSHAIR_SCRIPT}
    <script>
    (function() {{
      var block = document.querySelector('.dt-strip-scope');
      if (block) {{ dtHoverInit(block, {hover_cfg}); }}
    }})();
    </script>'''


def render_rt_card(rt: dict, fig_num: int = 1) -> str:
    rows = rt["rows"]
    n = len(rows)
    # Wide viewBox (matches gate_ticket.py's _bars_svg) so the trace and
    # deviation strip actually fill the card's now-full-width container
    # instead of being scaled down and centered by SVG's default
    # preserveAspectRatio behavior.
    plot_x0, plot_x1 = 40, 1380
    slot_w = (plot_x1 - plot_x0) / max(n, 1)

    deviation_strip = _deviation_strip(rows, plot_x0, plot_x1, rt["max_deviation_mw"])

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Delivery</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:500;">{rt['n_isp']}/{rt['n_isp']} ISPs</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">ISP real-time dispatch</div>
    <p style="font-size:12px; color:{theme.INK_MUTED}; margin:0 0 10px;">decided {_html.escape(rt['timestamp'])}</p>
    <div style="display:flex; gap:10px; margin-bottom:12px;">
      <div style="background:{theme.SURFACE}; border-radius:8px; padding:0.6rem 0.8rem; flex:1; border:1px solid {theme.GRIDLINE};">
        <p style="font-size:12px; color:{theme.INK_SECONDARY}; margin:0 0 2px;">Total abs. deviation</p>
        <p style="font-size:18px; font-weight:500; margin:0;">{rt['total_abs_deviation_mwh']:.1f} MWh</p>
      </div>
      <div style="background:{theme.SURFACE}; border-radius:8px; padding:0.6rem 0.8rem; flex:1; border:1px solid {theme.GRIDLINE};">
        <p style="font-size:12px; color:{theme.INK_SECONDARY}; margin:0 0 2px;">Max deviation</p>
        <p style="font-size:18px; font-weight:500; margin:0;">{rt['max_deviation_mw']:.2f} MW</p>
      </div>
    </div>
    {deviation_strip}
    <p style="font-size:10.5px; color:{theme.INK_SECONDARY}; text-align:center; margin:8px 0 0;">
      <b>Fig. {fig_num}.</b>&nbsp; ISP dispatch deviation from plan, MW per ISP.
    </p>
  </div>
</div>
{_REPLAY_STYLE}
{_REPLAY_SCRIPT}'''


_FAT_BY_PRODUCT = {
    "aFRR": ("PICASSO", "5 min"),
    "mFRR": ("MARI", "12.5 min"),
}


def _ace_replay_chart(summary: dict, product: str) -> str:
    """ACE (Area Control Error) and this product's activated response,
    stacked as two time-aligned panels with a connector at the peak-ACE
    moment -- the same layout as the FCR dispatch card's frequency/response
    chart (see dispatch_ticket.py::render_fcr_dispatch_card), because ACE is
    to aFRR/mFRR what frequency deviation is to FCR: the real signal the
    product responds to (see reserve_activation.py::simulate_ace_series),
    so the two widgets should read as one family, not as different chart
    languages for the same idea. 96 ISPs (this product's real resolution),
    not FCR's 2,880 30s ticks, but Replay runs at the same relative pace.
    The direction gauge + numeric readouts sit alongside, unchanged."""
    ace = summary["ace_mw"]
    resp = summary["response_mw"]
    n = len(resp)
    ref_mw = max(summary.get("max_offer_mw", 0.0), 1e-6)
    ace_max = max(max((abs(a) for a in ace), default=0.0), 15.0)

    x0, x1 = 40, 1400

    def fx(i: int) -> float:
        return x0 + i / max(n - 1, 1) * (x1 - x0)

    a_mid_y, a_half = 26, 20
    gap_h = 16
    b_base_y = (a_mid_y + a_half) + gap_h
    b_mid_y, b_half = b_base_y + 20, 20
    total_h = b_mid_y + b_half + 6

    def ay(v: float) -> float:
        return a_mid_y - (v / ace_max) * a_half

    def by(v: float) -> float:
        return b_mid_y - (v / ref_mw) * b_half

    ace_pts = " L".join(f"{fx(i):.1f},{ay(v):.1f}" for i, v in enumerate(ace))
    pos_pts = " L".join(f"{fx(i):.1f},{by(max(v, 0.0)):.1f}" for i, v in enumerate(resp))
    neg_pts = " L".join(f"{fx(i):.1f},{by(min(v, 0.0)):.1f}" for i, v in enumerate(resp))
    pos_path = f"M{fx(0):.1f},{b_mid_y} L{pos_pts} L{fx(n-1):.1f},{b_mid_y} Z"
    neg_path = f"M{fx(0):.1f},{b_mid_y} L{neg_pts} L{fx(n-1):.1f},{b_mid_y} Z"
    pos_edge = _sparse_edge_path(fx, by, [max(v, 0.0) for v in resp])
    neg_edge = _sparse_edge_path(fx, by, [min(v, 0.0) for v in resp])

    # Connector: the ISP with the widest ACE swing, marked on both panels
    # and joined by one line -- same "trace this moment down" idea as FCR.
    peak_idx = max(range(n), key=lambda i: abs(ace[i])) if n else 0
    peak_x = fx(peak_idx)
    peak_ay = ay(ace[peak_idx])
    peak_by = by(resp[peak_idx])
    if resp[peak_idx] > 0:
        callout_color = theme.COLOR_UP
    elif resp[peak_idx] < 0:
        callout_color = theme.COLOR_DOWN
    else:
        callout_color = theme.INK_MUTED

    resp_json = "[" + ",".join(f"{v:.2f}" for v in resp) + "]"
    ace_json = "[" + ",".join(f"{v:.1f}" for v in ace) + "]"

    hover_prefix = f"ace-{product}"
    ace_py = [ay(v) for v in ace]
    resp_py = [by(v) for v in resp]
    hover_cfg = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": f"#{hover_prefix}-hover-catch", "lineSelector": f"#{hover_prefix}-hover-line",
        "tooltipSelector": f"#{hover_prefix}-tooltip", "indexLabel": "ISP", "indexVals": list(range(1, n + 1)),
        "traces": [
            {"label": "ACE", "yvals": ace, "py": ace_py, "dotSelector": f"#{hover_prefix}-hover-dot-0", "color": theme.COLOR_PUMP, "unit": "MW", "decimals": 1, "signed": True},
            {"label": "Response", "yvals": resp, "py": resp_py, "dotSelector": f"#{hover_prefix}-hover-dot-1", "color": theme.COLOR_UP, "negColor": theme.COLOR_DOWN, "unit": "MW", "decimals": 2, "signed": True},
        ],
    })

    gx, gy, gr = 60.0, 65.0, 45.0

    return f'''
    <div class="ace-chart-block">
    <div style="display:flex; align-items:center; justify-content:space-between; margin:14px 0 6px;">
      <p style="font-size:12px; color:{theme.INK_PRIMARY}; margin:0; font-weight:500;">ACE-driven activation response</p>
      <button class="gt-replay" onclick="dtAceReplay(this)">&#9654; Replay</button>
    </div>
    <div style="background:{theme.STATUS_WARNING}22; border-left:3px solid {theme.STATUS_WARNING}; padding:6px 10px; margin-bottom:8px;">
      <span style="font-size:11.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Synthetic ACE, not real TSO SCADA &mdash; illustrative of the signal this product responds to, not a live grid feed.</span>
    </div>
    <div style="display:flex; gap:16px; align-items:center;">
      <div class="dt-ace-scope" data-ref="{ref_mw}" data-resp='{resp_json}' data-ace='{ace_json}'
           data-gx="{gx}" data-gy="{gy}" data-gr="{gr}"
           data-ace-max="{ace_max}" data-a-mid="{a_mid_y}" data-a-half="{a_half}"
           data-b-mid="{b_mid_y}" data-b-half="{b_half}" data-total-h="{total_h:.0f}"
           style="position:relative; height:{total_h:.0f}px; box-sizing:border-box; flex:1; min-width:0;">
        <svg viewBox="0 0 1400 {total_h:.0f}" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
          <defs><clipPath id="ace-clip-{product}"><rect id="ace-clip-rect-{product}" x="0" y="0" width="0" height="{total_h:.0f}"/></clipPath></defs>
          <line x1="{x0}" y1="{a_mid_y}" x2="{x1}" y2="{a_mid_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
          <line x1="{x0}" y1="{b_mid_y}" x2="{x1}" y2="{b_mid_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
          <text x="{x0-4}" y="{a_mid_y-a_half+3:.1f}" font-size="10" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">+{ace_max:.0f}</text>
          <text x="{x0-4}" y="{a_mid_y+3:.1f}" font-size="10" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">0 MW</text>
          <text x="{x0-4}" y="{a_mid_y+a_half+3:.1f}" font-size="10" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">-{ace_max:.0f}</text>
          <text x="{x0-4}" y="{b_mid_y-b_half+3:.1f}" font-size="10" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">+{ref_mw:.0f}</text>
          <text x="{x0-4}" y="{b_mid_y+3:.1f}" font-size="10" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">0 MW</text>
          <text x="{x0-4}" y="{b_mid_y+b_half+3:.1f}" font-size="10" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">-{ref_mw:.0f}</text>
          <g clip-path="url(#ace-clip-{product})">
            <path d="M{fx(0):.1f},{ay(ace[0] if ace else 0):.1f} L{ace_pts}" fill="none" stroke="{theme.COLOR_PRICE}" stroke-width="1.5"/>
            <path d="{pos_path}" fill="{theme.COLOR_UP}" fill-opacity="0.4"/>
            <path d="{neg_path}" fill="{theme.COLOR_DOWN}" fill-opacity="0.4"/>
            <path d="{pos_edge}" fill="none" stroke="{theme.COLOR_UP}" stroke-width="1.3"/>
            <path d="{neg_edge}" fill="none" stroke="{theme.COLOR_DOWN}" stroke-width="1.3"/>
          </g>
          <line x1="{peak_x:.1f}" y1="{peak_ay+5:.1f}" x2="{peak_x:.1f}" y2="{peak_by-5:.1f}" stroke="{theme.INK_PRIMARY}" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.7"/>
          <circle cx="{peak_x:.1f}" cy="{peak_ay:.1f}" r="3.5" fill="{theme.COLOR_PRICE}" stroke="white" stroke-width="1.2"/>
          <circle cx="{peak_x:.1f}" cy="{peak_by:.1f}" r="3.5" fill="{callout_color}" stroke="white" stroke-width="1.2"/>
          <circle id="ace-dot-{product}" cx="0" cy="{b_mid_y}" r="4" fill="{theme.COLOR_UP}" style="opacity:0;"/>
          {_hover_svg_elems(hover_prefix, x0, x1, total_h, 2)}
        </svg>
        {_hover_tooltip_div().replace('dt-hover-tooltip"', f'dt-hover-tooltip" id="{hover_prefix}-tooltip"')}
      </div>
      <div style="width:1px; align-self:stretch; background:{theme.GRIDLINE};"></div>
      <div style="width:130px; flex-shrink:0; text-align:center;">
        <svg viewBox="0 0 120 70" style="width:110px; height:64px;">
          <path d="M {gx - gr} {gy} A {gr} {gr} 0 0 1 {gx} {gy - gr}" fill="none" stroke="{theme.COLOR_DOWN}" stroke-width="9"/>
          <path d="M {gx} {gy - gr} A {gr} {gr} 0 0 1 {gx + gr} {gy}" fill="none" stroke="{theme.COLOR_UP}" stroke-width="9"/>
          <line id="ace-needle-{product}" x1="{gx}" y1="{gy}" x2="{gx}" y2="{gy - gr * 0.85:.1f}" stroke="{theme.INK_PRIMARY}" stroke-width="2"/>
          <circle cx="{gx}" cy="{gy}" r="3.5" fill="{theme.INK_PRIMARY}"/>
          <text x="{gx - gr:.1f}" y="{gy + 4:.1f}" font-size="7" fill="{theme.COLOR_DOWN}">DOWN</text>
          <text x="{gx + gr:.1f}" y="{gy + 4:.1f}" font-size="7" fill="{theme.COLOR_UP}" text-anchor="end">UP</text>
        </svg>
        <p style="font-size:18px; font-weight:500; margin:2px 0 0;"><span id="ace-mw-{product}">0.0</span> <span style="font-size:11px; font-weight:400; color:{theme.INK_MUTED};">MW</span></p>
        <p id="ace-pct-{product}" style="font-size:10px; color:{theme.INK_MUTED}; margin:0;">0% of offered</p>
        <p id="ace-val-{product}" style="font-size:11px; color:{theme.COLOR_PRICE}; margin:4px 0 0;">ACE 0 MW</p>
      </div>
    </div>
    <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
      <span style="font-size:11px; color:{theme.INK_MUTED};">00:00</span>
      <div style="flex:1; height:4px; background:{theme.GRIDLINE}; border-radius:2px; position:relative;">
        <div id="ace-playhead-{product}" style="position:absolute; left:0%; top:-3px; width:10px; height:10px; border-radius:50%; background:{theme.COLOR_UP}; transform:translateX(-50%);"></div>
      </div>
      <span style="font-size:11px; color:{theme.INK_MUTED};">24:00</span>
      <span id="ace-clock-{product}" style="font-size:11px; color:{theme.INK_SECONDARY}; min-width:36px; text-align:right;"></span>
    </div>
    </div>
    <div style="display:flex; gap:14px; margin-top:8px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:2px; background:{theme.COLOR_PUMP}; margin-right:4px; vertical-align:middle;"></span>Area Control Error</span>
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:2px; background:{theme.COLOR_UP}; margin-right:4px; vertical-align:middle;"></span>Plant response</span>
    </div>
    {_HOVER_CROSSHAIR_SCRIPT}
    <script>
    (function() {{
      var block = document.querySelector('.dt-ace-scope');
      if (block) {{ dtHoverInit(block, {hover_cfg}); }}
    }})();
    </script>'''


_ACE_REPLAY_SCRIPT = '''
<script>
window.dtAceReplay = function(btn) {
  var block = btn.closest('.ace-chart-block');
  var scope = block ? block.querySelector('.dt-ace-scope') : null;
  if (!scope || scope.dataset.replaying === '1') return;
  scope.dataset.replaying = '1';
  var clipRect = scope.querySelector('rect[id^="ace-clip-rect-"]');
  var dot = scope.querySelector('circle[id^="ace-dot-"]');
  var needle = block.querySelector('line[id^="ace-needle-"]');
  var mwLabel = block.querySelector('span[id^="ace-mw-"]');
  var pctLabel = block.querySelector('p[id^="ace-pct-"]');
  var aceLabel = block.querySelector('p[id^="ace-val-"]');
  var playhead = block.querySelector('div[id^="ace-playhead-"]');
  var clock = block.querySelector('span[id^="ace-clock-"]');
  var resp = JSON.parse(scope.dataset.resp);
  var ace = JSON.parse(scope.dataset.ace);
  var ref = parseFloat(scope.dataset.ref);
  var gx = parseFloat(scope.dataset.gx), gy = parseFloat(scope.dataset.gy), gr = parseFloat(scope.dataset.gr);
  var bMid = parseFloat(scope.dataset.bMid), bHalf = parseFloat(scope.dataset.bHalf);
  var totalH = parseFloat(scope.dataset.totalH);
  var n = resp.length;
  // Same 10s duration and ~500ms (~20 updates total) number-readout cadence
  // as the FCR card's replay, so all three widgets (FCR/aFRR/mFRR) feel like
  // one consistent playback speed rather than each having its own pace.
  var steps = n;
  var stepMs = 10000 / steps;
  var numberEvery = Math.max(1, Math.round(steps / 20));
  clipRect.setAttribute('height', totalH.toFixed(0));
  clipRect.setAttribute('width', '0');
  if (dot) dot.style.opacity = '1';

  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var frac = s / steps;
        clipRect.setAttribute('width', (1400 * frac).toFixed(1));
        var idx = Math.min(Math.floor(frac * (n - 1)), n - 1);
        var x = frac * 1400;
        var respVal = resp[idx];
        var y = bMid - (respVal / ref) * bHalf;
        if (dot) { dot.setAttribute('cx', x.toFixed(1)); dot.setAttribute('cy', y.toFixed(1));
          dot.setAttribute('fill', respVal > 0 ? '#1baf7a' : (respVal < 0 ? '#e87ba4' : '#898781')); }
        if (playhead) playhead.style.left = (frac * 100) + '%';
        if (clock) {
          var totalSec = Math.floor(frac * 86400);
          var hh = String(Math.floor(totalSec / 3600)).padStart(2, '0');
          var mm = String(Math.floor((totalSec % 3600) / 60)).padStart(2, '0');
          clock.textContent = hh + ':' + mm;
        }
        if (s % numberEvery === 0 || s === steps) {
          if (needle) {
            var signedFrac = Math.max(-1, Math.min(respVal / ref, 1));
            var displayFrac = (signedFrac + 1) / 2;
            var angle = Math.PI * (1 - displayFrac);
            var nx = gx + gr * 0.85 * Math.cos(angle);
            var ny = gy - gr * 0.85 * Math.sin(angle);
            needle.setAttribute('x2', nx.toFixed(1));
            needle.setAttribute('y2', ny.toFixed(1));
          }
          if (mwLabel) mwLabel.textContent = (respVal >= 0 ? '+' : '') + respVal.toFixed(1);
          if (pctLabel) pctLabel.textContent = Math.round(Math.abs(respVal) / ref * 100) + '% of offered';
          if (aceLabel) aceLabel.textContent = 'ACE ' + (ace[idx] >= 0 ? '+' : '') + ace[idx].toFixed(1) + ' MW';
        }
        if (s === steps) {
          if (dot) dot.style.opacity = '0';
          scope.dataset.replaying = '0';
        }
      }, s * stepMs);
    })(s);
  }
};
</script>'''


def render_activation_card(summary: dict, product_label: str) -> str:
    ace_chart = _ace_replay_chart(summary, product_label)
    platform, fat = _FAT_BY_PRODUCT.get(product_label, ("", ""))
    fat_badge = (
        f'<span style="background:{theme.INK_MUTED}22; color:{theme.INK_SECONDARY}; font-size:10.5px; '
        f'padding:2px 8px; border-radius:5px; font-weight:500; white-space:nowrap; margin-left:8px;">'
        f'{platform} platform &middot; FAT {fat} (EBGL standard product)</span>'
        if platform else ""
    )
    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Delivery</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:500;">Activated</span>
    </div>
    <div style="display:flex; align-items:center; margin-bottom:2px;">
      <span style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY};">{_html.escape(product_label)} activation response</span>
      {fat_badge}
    </div>
    <p style="font-size:12px; color:{theme.INK_MUTED}; margin:0 0 10px;">{summary['n_isp']} of 96 ISPs activated &middot; decided {_html.escape(summary['timestamp'])}</p>
    <div style="display:flex; gap:10px;">
      <div style="background:{theme.SURFACE}; border-radius:8px; padding:0.6rem 0.8rem; flex:1; border:1px solid {theme.GRIDLINE};">
        <p style="font-size:12px; color:{theme.INK_SECONDARY}; margin:0 0 2px;">Activation revenue</p>
        <p style="font-size:18px; font-weight:500; margin:0;">&euro;{summary['revenue_eur']:,.0f}</p>
      </div>
      <div style="background:{theme.SURFACE}; border-radius:8px; padding:0.6rem 0.8rem; flex:1; border:1px solid {theme.GRIDLINE};">
        <p style="font-size:12px; color:{theme.INK_SECONDARY}; margin:0 0 2px;">Energy up / down</p>
        <p style="font-size:18px; font-weight:500; margin:0;">{summary['up_mwh']:.1f} / {summary['dn_mwh']:.1f} MWh</p>
      </div>
    </div>
    {ace_chart}
  </div>
</div>
{_REPLAY_STYLE}
{_ACE_REPLAY_SCRIPT}'''


_AGC_REPLAY_SCRIPT = '''
<script>
window.dtAgcReplay = function(btn) {
  var block = btn.closest('.agc-chart-block');
  if (!block || block.dataset.replaying === '1') return;
  block.dataset.replaying = '1';
  var data = JSON.parse(block.dataset.agc);
  var names = data.names;
  var n = data.ace.length;
  var reqLabel = block.querySelector('#agc-required');
  var dirLabel = block.querySelector('#agc-direction');
  var aceLabel = block.querySelector('#agc-ace');
  var playhead = block.querySelector('#agc-playhead');
  var clock = block.querySelector('#agc-clock');
  // This widget updates far more per readout tick than FCR/aFRR/mFRR --
  // 6 provider rows x 2 numbers each, plus required/direction/ACE = 15
  // numbers per update, vs 3 for the other widgets. Same reading budget
  // (~167ms per number) scales the cadence to ~2000ms instead of ~500ms,
  // with total duration extended so the sweep still reads as continuous.
  var DURATION_MS = 16000;
  var TARGET_CADENCE_MS = 2000;
  var steps = n;
  var stepMs = DURATION_MS / steps;
  var numberEvery = Math.max(1, Math.round(TARGET_CADENCE_MS / stepMs));

  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var idx = Math.min(s - 1, n - 1);
        var frac = s / steps;
        if (playhead) playhead.style.left = (frac * 100) + '%';
        if (clock) {
          var totalSec = Math.floor(frac * 86400);
          var hh = String(Math.floor(totalSec / 3600)).padStart(2, '0');
          var mm = String(Math.floor((totalSec % 3600) / 60)).padStart(2, '0');
          clock.textContent = hh + ':' + mm;
        }
        names.forEach(function(name) {
          var row = block.querySelector('[data-provider="' + name + '"]');
          if (!row) return;
          var cap = data.capacity[name][idx];
          var dispatched = data.dispatched[name][idx];
          var price = data.price[name][idx];
          var pct = cap > 0 ? Math.min(100, (dispatched / cap) * 100) : 0;
          var fill = row.querySelector('.agc-fill');
          if (fill) fill.style.width = pct.toFixed(1) + '%';
          if (s % numberEvery === 0 || s === steps) {
            var readout = row.querySelector('.agc-readout');
            if (readout) readout.textContent = dispatched.toFixed(1) + ' / ' + cap.toFixed(0) + ' MW @ €' + price.toFixed(1);
          }
        });
        if (s % numberEvery === 0 || s === steps) {
          if (reqLabel) reqLabel.textContent = data.required[idx].toFixed(1) + ' MW';
          if (dirLabel) dirLabel.textContent = data.direction[idx];
          if (aceLabel) aceLabel.textContent = (data.ace[idx] >= 0 ? '+' : '') + data.ace[idx].toFixed(1) + ' MW';
        }
        if (s === steps) block.dataset.replaying = '0';
      }, s * stepMs);
    })(s);
  }
};
</script>'''


def render_agc_mechanism_card(demo: dict, product_label: str) -> str:
    """Illustrative AGC merit-order dispatch ladder -- NOT settlement data
    (see agc_mechanism_demo.py). Shows, per ISP, which providers in the
    control area get called and how much of the required regulation each
    covers, cheapest-first. Alqueva's row is real (its own offer/price);
    the other providers are clearly-labelled synthetic stand-ins, since no
    public data on Portugal's actual competitor fleet exists."""
    names = demo["provider_names"]
    payload = {
        "names": names,
        "ace": demo["ace_mw"],
        "required": demo["required_mw"],
        "direction": demo["direction"],
        "dispatched": demo["dispatched_by_provider"],
        "capacity": demo["capacity_by_provider"],
        "price": demo["price_by_provider"],
    }
    payload_json = _json.dumps(payload)

    rows = []
    for name in names:
        is_alqueva = "Alqueva" in name
        cap0 = demo["capacity_by_provider"][name][0]
        disp0 = demo["dispatched_by_provider"][name][0]
        price0 = demo["price_by_provider"][name][0]
        pct0 = min(100.0, (disp0 / cap0) * 100) if cap0 > 0 else 0.0
        border = f"2px solid {theme.COLOR_UP}" if is_alqueva else f"1px solid {theme.GRIDLINE}"
        fill_color = theme.COLOR_UP if is_alqueva else theme.COLOR_GEN
        label = f'<strong>{_html.escape(name)}</strong>' if is_alqueva else _html.escape(name)
        rows.append(f'''
        <div data-provider="{_html.escape(name)}" style="border:{border}; border-radius:8px; padding:6px 10px; margin-bottom:6px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-size:11.5px; color:{theme.INK_PRIMARY};">{label}</span>
            <span class="agc-readout" style="font-size:10.5px; color:{theme.INK_MUTED};">{disp0:.1f} / {cap0:.0f} MW @ &euro;{price0:.1f}</span>
          </div>
          <div style="height:6px; background:{theme.GRIDLINE}; border-radius:3px; overflow:hidden;">
            <div class="agc-fill" style="width:{pct0:.1f}%; height:100%; background:{fill_color};"></div>
          </div>
        </div>''')

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Mechanism</span>
      <span style="background:{theme.INK_MUTED}22; color:{theme.INK_SECONDARY}; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:500;">Illustrative, not settlement</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:8px;">{_html.escape(product_label)} AGC merit-order dispatch</div>
    <div class="agc-chart-block" data-agc='{payload_json}'>
    <div style="background:{theme.STATUS_WARNING}22; border-left:3px solid {theme.STATUS_WARNING}; padding:6px 10px; margin-bottom:10px;">
      <span style="font-size:11.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Shows how AGC picks providers cheapest-first to cover the area's regulation need. Alqueva's capacity/price are real; the other providers are synthetic (no public competitor data exists) &mdash; this does not feed settlement, which uses only Alqueva's own committed offer.</span>
    </div>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
      <div style="font-size:11px; color:{theme.INK_SECONDARY};">
        Required: <strong id="agc-required">{demo['required_mw'][0]:.1f} MW</strong>
        &middot; Direction: <strong id="agc-direction">{demo['direction'][0]}</strong>
        &middot; ACE: <strong id="agc-ace">{demo['ace_mw'][0]:+.1f} MW</strong>
      </div>
      <button class="gt-replay" onclick="dtAgcReplay(this)">&#9654; Replay</button>
    </div>
    {''.join(rows)}
    <div style="display:flex; align-items:center; gap:8px; margin-top:8px;">
      <span style="font-size:11px; color:{theme.INK_MUTED};">00:00</span>
      <div style="flex:1; height:4px; background:{theme.GRIDLINE}; border-radius:2px; position:relative;">
        <div id="agc-playhead" style="position:absolute; left:0%; top:-3px; width:10px; height:10px; border-radius:50%; background:{theme.COLOR_UP}; transform:translateX(-50%);"></div>
      </div>
      <span style="font-size:11px; color:{theme.INK_MUTED};">24:00</span>
      <span id="agc-clock" style="font-size:11px; color:{theme.INK_SECONDARY}; min-width:36px; text-align:right;"></span>
    </div>
    </div>
  </div>
</div>
{_REPLAY_STYLE}
{_AGC_REPLAY_SCRIPT}'''
