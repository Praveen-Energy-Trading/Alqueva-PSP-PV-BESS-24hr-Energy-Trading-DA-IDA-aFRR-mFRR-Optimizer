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

_FROZEN_GRAY = "#c3c2b7"

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
      </svg>
      <div style="position:absolute; left:0; top:0; font-size:12px; color:{theme.INK_MUTED};">+{max_label}</div>
      <div style="position:absolute; left:0; top:50%; transform:translateY(-50%); font-size:12px; color:{theme.INK_MUTED};">0</div>
      <div style="position:absolute; left:0; bottom:0; font-size:12px; color:{theme.INK_MUTED};">-{max_label}</div>
    </div>
    <div style="display:flex; gap:14px; margin-top:2px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_UP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Over-delivered</span>
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_DOWN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Under-delivered</span>
    </div>'''


def _fcr_replay_chart(fcr: dict) -> str:
    """Option E: grid frequency (dashed, behind, shaded deadband) overlaid
    with the plant's droop response (filled, in front) in ONE chart instead
    of two stacked ones -- cause and effect read in a single glance. Static
    by default (fully drawn); Replay resets a clip-path to width 0 and
    sweeps it back to full width via requestAnimationFrame over ~3s, with a
    dot tracking the current point on the response curve and a live
    HH:MM clock -- a real playback of the day's 2,880 actual 30s ticks,
    not a generic bar reveal."""
    freq = fcr["tick_freq_mhz"]
    resp = fcr["tick_response_mw"]
    n = len(resp)
    headroom = max(fcr["reserved_headroom_mw"], 1e-6)
    freq_max = max(max((abs(f) for f in freq), default=0.0), 15.0)

    x0, x1 = 0, 1400
    zero_y = 50
    ampl_freq = ampl_resp = 34  # same pixel amplitude for both -- neither trace gets a visual size advantage
    deadband_half_h = (10.0 / freq_max) * ampl_freq

    def fx(i: int) -> float:
        return x0 + i / max(n - 1, 1) * (x1 - x0)

    freq_pts = " ".join(f"{fx(i):.1f},{zero_y - (v / freq_max) * ampl_freq:.1f}" for i, v in enumerate(freq))
    resp_line_pts = [(fx(i), zero_y - (v / headroom) * ampl_resp) for i, v in enumerate(resp)]
    resp_line_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in resp_line_pts)
    resp_poly_str = resp_line_str + f" {x1},{zero_y:.1f} {x0},{zero_y:.1f}"

    resp_json = "[" + ",".join(f"{v:.2f}" for v in resp) + "]"
    freq_json = "[" + ",".join(f"{v:.1f}" for v in freq) + "]"
    # Gauge needle: fixed semicircular pivot/radius, angle 180deg (empty,
    # left) to 0deg (full headroom, right) driven purely by |response|/
    # headroom -- direction (up/down) is conveyed by the colored UP/DOWN
    # label next to it instead of needle side, since a magnitude gauge
    # reads more like a real speedometer than a signed one would.
    gx, gy, gr = 60.0, 65.0, 45.0

    ce_pool_mw = 3000.0
    pool_share_pct = headroom / ce_pool_mw * 100
    return f'''
    <div class="fcr-chart-block">
    <div style="display:flex; align-items:center; justify-content:space-between; margin:14px 0 6px;">
      <p style="font-size:12px; color:{theme.INK_PRIMARY}; margin:0; font-weight:500;">Grid frequency droop response</p>
      <button class="gt-replay" onclick="dtFcrReplay(this)">&#9654; Replay</button>
    </div>
    <div style="background:{theme.STATUS_WARNING}22; border-left:3px solid {theme.STATUS_WARNING}; padding:6px 10px; margin-bottom:8px;">
      <span style="font-size:11.5px; color:{theme.INK_PRIMARY}; font-weight:500;">This plant is {pool_share_pct:.2f}% of the Continental Europe FCR pool &mdash; it does not restore grid frequency alone. Its response tracks the deviation's shape (droop is proportional by design); it does not cause or fix it.</span>
    </div>
    <div style="display:flex; gap:16px; align-items:center;">
      <div class="dt-fcr-scope" data-headroom="{headroom}" data-resp='{resp_json}' data-freq='{freq_json}'
           data-gx="{gx}" data-gy="{gy}" data-gr="{gr}"
           style="position:relative; height:100px; box-sizing:border-box; flex:1; min-width:0;">
        <svg viewBox="0 0 1400 100" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
          <defs><clipPath id="fcr-clip"><rect id="fcr-clip-rect" x="0" y="-20" width="1400" height="140"/></clipPath></defs>
          <g clip-path="url(#fcr-clip)">
            <rect x="0" y="{zero_y - deadband_half_h:.1f}" width="1400" height="{2 * deadband_half_h:.1f}" fill="{_FROZEN_GRAY}" opacity="0.35"/>
            <line x1="0" y1="{zero_y}" x2="1400" y2="{zero_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
            <polygon points="{resp_poly_str}" fill="{theme.COLOR_UP}" opacity="0.3"/>
            <polyline points="{resp_line_str}" fill="none" stroke="{theme.COLOR_UP}" stroke-width="1.5"/>
            <polyline points="{freq_pts}" fill="none" stroke="{theme.COLOR_PUMP}" stroke-width="1.5" opacity="0.9" stroke-dasharray="3,2"/>
          </g>
          <circle id="fcr-dot" cx="0" cy="{zero_y}" r="4" fill="{theme.COLOR_UP}" style="opacity:0;"/>
        </svg>
      </div>
      <div style="width:1px; align-self:stretch; background:{theme.GRIDLINE};"></div>
      <div style="width:130px; flex-shrink:0; text-align:center;">
        <svg viewBox="0 0 120 70" style="width:110px; height:64px;">
          <path d="M {gx - gr} {gy} A {gr} {gr} 0 0 1 {gx} {gy - gr}" fill="none" stroke="{theme.COLOR_DOWN}" stroke-width="9"/>
          <path d="M {gx} {gy - gr} A {gr} {gr} 0 0 1 {gx + gr} {gy}" fill="none" stroke="{theme.COLOR_UP}" stroke-width="9"/>
          <line id="fcr-needle" x1="{gx}" y1="{gy}" x2="{gx}" y2="{gy - gr * 0.85:.1f}" stroke="{theme.INK_PRIMARY}" stroke-width="2"/>
          <circle cx="{gx}" cy="{gy}" r="3.5" fill="{theme.INK_PRIMARY}"/>
          <text x="{gx - gr:.1f}" y="{gy + 4:.1f}" font-size="7" fill="{theme.COLOR_DOWN}">DOWN</text>
          <text x="{gx + gr:.1f}" y="{gy + 4:.1f}" font-size="7" fill="{theme.COLOR_UP}" text-anchor="end">UP</text>
        </svg>
        <p style="font-size:18px; font-weight:500; margin:2px 0 0;"><span id="fcr-mw">0.0</span> <span style="font-size:11px; font-weight:400; color:{theme.INK_MUTED};">MW</span></p>
        <p id="fcr-pct" style="font-size:10px; color:{theme.INK_MUTED}; margin:0;">0% of headroom</p>
        <p id="fcr-mhz" style="font-size:11px; color:{theme.COLOR_PUMP}; margin:4px 0 0;">0 mHz</p>
      </div>
    </div>
    <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
      <span style="font-size:11px; color:{theme.INK_MUTED};">00:00</span>
      <div style="flex:1; height:4px; background:{theme.GRIDLINE}; border-radius:2px; position:relative;">
        <div id="fcr-playhead" style="position:absolute; left:0%; top:-3px; width:10px; height:10px; border-radius:50%; background:{theme.COLOR_UP}; transform:translateX(-50%);"></div>
      </div>
      <span style="font-size:11px; color:{theme.INK_MUTED};">24:00</span>
      <span id="fcr-clock" style="font-size:11px; color:{theme.INK_SECONDARY}; min-width:36px; text-align:right;"></span>
    </div>
    </div>
    <div style="display:flex; gap:14px; margin-top:8px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:2px; background:{theme.COLOR_PUMP}; margin-right:4px; vertical-align:middle;"></span>Grid frequency deviation</span>
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:2px; background:{theme.COLOR_UP}; margin-right:4px; vertical-align:middle;"></span>Plant response</span>
    </div>
    <div style="margin-top:10px;">
      <div style="display:flex; justify-content:space-between; font-size:10px; color:{theme.INK_MUTED}; margin-bottom:3px;">
        <span>Alqueva ({headroom:.0f} MW)</span>
        <span>Continental Europe FCR pool (~{ce_pool_mw:.0f} MW, ENTSO-E)</span>
      </div>
      <div style="height:8px; background:{theme.GRIDLINE}; border-radius:4px; overflow:hidden; display:flex;">
        <div style="width:{max(pool_share_pct, 1.5):.2f}%; background:{theme.COLOR_UP}; height:100%;"></div>
      </div>
    </div>
    '''


_FCR_REPLAY_SCRIPT = '''
<script>
window.dtFcrReplay = function(btn) {
  // Layout: header row (button) -> chart+gauge flex row -> playhead row -> legend row.
  // .dt-fcr-scope is the chart div INSIDE the chart+gauge row, so it's one
  // level deeper than it used to be before the gauge was added alongside
  // the chart -- playhead/clock are outside that row entirely, one level up.
  var block = btn.closest('.fcr-chart-block');
  var scope = block ? block.querySelector('.dt-fcr-scope') : null;
  if (!scope || scope.dataset.replaying === '1') return;
  scope.dataset.replaying = '1';
  var clipRect = scope.querySelector('#fcr-clip-rect');
  var dot = scope.querySelector('#fcr-dot');
  var needle = block.querySelector('#fcr-needle');
  var mwLabel = block.querySelector('#fcr-mw');
  var pctLabel = block.querySelector('#fcr-pct');
  var mhzLabel = block.querySelector('#fcr-mhz');
  var playhead = block.querySelector('#fcr-playhead');
  var clock = block.querySelector('#fcr-clock');
  var resp = JSON.parse(scope.dataset.resp);
  var freq = JSON.parse(scope.dataset.freq);
  var headroom = parseFloat(scope.dataset.headroom);
  var gx = parseFloat(scope.dataset.gx), gy = parseFloat(scope.dataset.gy), gr = parseFloat(scope.dataset.gr);
  var n = resp.length;

  // setTimeout-stepped, NOT requestAnimationFrame -- rAF only fires while
  // the tab is actually compositing visible frames, so it silently never
  // progresses in a background/inactive tab (verified: clip stayed at
  // width 0 indefinitely in exactly that state). setTimeout keeps firing
  // regardless, the same reason gate_ticket.py's bar-reveal animations
  // avoid CSS animation-delay in favor of setTimeout.
  //
  // Duration deliberately slow: at the original 3000ms/90-step pace the
  // needle and MW/%/mHz numbers were changing ~30x/second -- far faster
  // than a human can read three simultaneous numbers, so it just looked
  // like flicker. Line-sweep stays smooth (160 steps, ~62.5ms each = 10s
  // total), but the gauge numbers are throttled to update only every 8th
  // step (~500ms, 2x/second) -- reading three fields at once needs closer
  // to half a second each, not a quarter, to actually register before the
  // next value replaces it.
  var steps = 160;
  var stepMs = 10000 / steps;
  var numberEvery = 8;
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
        var y = 50 - (respVal / headroom) * 34;
        if (dot) { dot.setAttribute('cx', x.toFixed(1)); dot.setAttribute('cy', y.toFixed(1)); }
        if (playhead) playhead.style.left = (frac * 100) + '%';
        if (clock) {
          var totalSec = Math.floor(frac * 86400);
          var hh = String(Math.floor(totalSec / 3600)).padStart(2, '0');
          var mm = String(Math.floor((totalSec % 3600) / 60)).padStart(2, '0');
          clock.textContent = hh + ':' + mm;
        }
        // Gauge: split bidirectional dial -- left half (coral) = DOWN,
        // right half (green) = UP, needle rests pointing straight up at
        // zero. displayFrac 0->1 maps response/headroom -1->+1 onto the
        // semicircle's 180deg(left)->0deg(right) sweep, so which side the
        // needle leans IS the direction -- no text label needed to read it,
        // unlike the single-magnitude version this replaced.
        // Throttled to numberEvery steps (see above) so the reading is
        // legible instead of flickering through 2,880 values in 8s.
        if (s % numberEvery === 0 || s === steps) {
          if (needle) {
            var signedFrac = Math.max(-1, Math.min(respVal / headroom, 1));
            var displayFrac = (signedFrac + 1) / 2;
            var angle = Math.PI * (1 - displayFrac);
            var nx = gx + gr * 0.85 * Math.cos(angle);
            var ny = gy - gr * 0.85 * Math.sin(angle);
            needle.setAttribute('x2', nx.toFixed(1));
            needle.setAttribute('y2', ny.toFixed(1));
          }
          if (mwLabel) mwLabel.textContent = (respVal >= 0 ? '+' : '') + respVal.toFixed(1);
          if (pctLabel) pctLabel.textContent = Math.round(Math.abs(respVal) / headroom * 100) + '% of headroom';
          if (mhzLabel) mhzLabel.textContent = (freq[idx] >= 0 ? '+' : '') + freq[idx].toFixed(1) + ' mHz';
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


def render_fcr_card(fcr: dict) -> str:
    """FCR droop-response compliance card. No revenue metric on purpose --
    FCR is non-remunerated in Portugal (mandatory grid-code obligation, not
    a market-procured product), so showing a fake €0 next to the other
    cards' real euro figures would misleadingly imply this one just happens
    to earn nothing rather than structurally never earning anything. See
    dashboard/data.py::load_fcr_activation and fcr_activation.py."""
    n = len(fcr["rows"])
    chart = _fcr_replay_chart(fcr)
    reserved_budget_mwh = fcr["reserved_headroom_mw"] * 24
    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Delivery</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:500;">{fcr['n_isp_responding']}/{n} ISPs responding</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">FCR droop response</div>
    <p style="font-size:12px; color:{theme.INK_MUTED}; margin:0 0 10px;">Mandatory, non-remunerated (Portuguese grid code) &middot; reserved headroom {fcr['reserved_headroom_mw']:.1f} MW</p>
    {chart}
    <div style="display:flex; gap:14px; margin-top:10px; padding-top:10px; border-top:1px solid {theme.GRIDLINE};">
      <div><p style="font-size:10px; color:{theme.INK_MUTED}; margin:0;">Activated (up / down)</p><p style="font-size:15px; font-weight:500; margin:0;">{fcr['up_mwh']:.1f} / {fcr['dn_mwh']:.1f} MWh</p></div>
      <div><p style="font-size:10px; color:{theme.INK_MUTED}; margin:0;">Reserved all day</p><p style="font-size:15px; font-weight:500; margin:0;">{reserved_budget_mwh:.0f} MWh</p></div>
      <div><p style="font-size:10px; color:{theme.INK_MUTED}; margin:0;">Max response</p><p style="font-size:15px; font-weight:500; margin:0;">{fcr['max_response_mw']:.2f} MW</p></div>
    </div>
  </div>
</div>
{_REPLAY_STYLE}
{_FCR_REPLAY_SCRIPT}'''


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
    """ACE (Area Control Error, background, dashed) overlaid with this
    product's activated response (foreground) in one chart, same Option E
    layout as the FCR card's frequency/droop chart -- and for the same
    reason: ACE is the actual signal aFRR/mFRR activation responds to (see
    reserve_activation.py::simulate_ace_series), so overlaying cause and
    effect is honest here the way it wasn't for a made-up frequency line.
    96 ISPs (this product's real resolution), not FCR's 2,880 30s ticks, but
    Replay runs at the same 10s pace and ~500ms numeric-readout cadence as
    the FCR card so all three widgets feel consistent."""
    ace = summary["ace_mw"]
    resp = summary["response_mw"]
    n = len(resp)
    ref_mw = max(summary.get("max_offer_mw", 0.0), 1e-6)
    ace_max = max(max((abs(a) for a in ace), default=0.0), 15.0)

    x0, x1 = 0, 1400
    zero_y = 50
    ampl_ace = ampl_resp = 34

    def fx(i: int) -> float:
        return x0 + i / max(n - 1, 1) * (x1 - x0)

    ace_pts = " ".join(f"{fx(i):.1f},{zero_y - (v / ace_max) * ampl_ace:.1f}" for i, v in enumerate(ace))
    resp_line_pts = [(fx(i), zero_y - (v / ref_mw) * ampl_resp) for i, v in enumerate(resp)]
    resp_line_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in resp_line_pts)
    resp_poly_str = resp_line_str + f" {x1},{zero_y:.1f} {x0},{zero_y:.1f}"

    resp_json = "[" + ",".join(f"{v:.2f}" for v in resp) + "]"
    ace_json = "[" + ",".join(f"{v:.1f}" for v in ace) + "]"

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
           style="position:relative; height:100px; box-sizing:border-box; flex:1; min-width:0;">
        <svg viewBox="0 0 1400 100" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
          <defs><clipPath id="ace-clip-{product}"><rect id="ace-clip-rect-{product}" x="0" y="-20" width="1400" height="140"/></clipPath></defs>
          <g clip-path="url(#ace-clip-{product})">
            <line x1="0" y1="{zero_y}" x2="1400" y2="{zero_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
            <polygon points="{resp_poly_str}" fill="{theme.COLOR_UP}" opacity="0.3"/>
            <polyline points="{resp_line_str}" fill="none" stroke="{theme.COLOR_UP}" stroke-width="1.5"/>
            <polyline points="{ace_pts}" fill="none" stroke="{theme.COLOR_PUMP}" stroke-width="1.5" opacity="0.9" stroke-dasharray="3,2"/>
          </g>
          <circle id="ace-dot-{product}" cx="0" cy="{zero_y}" r="4" fill="{theme.COLOR_UP}" style="opacity:0;"/>
        </svg>
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
        <p id="ace-val-{product}" style="font-size:11px; color:{theme.COLOR_PUMP}; margin:4px 0 0;">ACE 0 MW</p>
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
    </div>'''


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
  var n = resp.length;
  // Same 10s duration and ~500ms (~20 updates total) number-readout cadence
  // as the FCR card's replay, so all three widgets (FCR/aFRR/mFRR) feel like
  // one consistent playback speed rather than each having its own pace.
  var steps = n;
  var stepMs = 10000 / steps;
  var numberEvery = Math.max(1, Math.round(steps / 20));
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
        var y = 50 - (respVal / ref) * 34;
        if (dot) { dot.setAttribute('cx', x.toFixed(1)); dot.setAttribute('cy', y.toFixed(1)); }
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
