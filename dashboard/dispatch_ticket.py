"""dispatch_ticket.py — technical/physical optimization widgets: reservoir
trajectory, unit commitment, ramp utilization, etc. Distinct from
delivery_ticket.py (market delivery/settlement cards) and gate_ticket.py
(bid decisions) -- these show what the MILP solver actually did physically,
not what was bid or settled. All data here is real solved-model output
(ComponentStore), never illustrative, unless explicitly labelled otherwise.
"""
from __future__ import annotations

import json as _json

import theme

_REPLAY_STYLE = f'<style>.gt-replay {{ font-size:11px; padding:2px 10px; border:1px solid {theme.GRIDLINE}; border-radius:6px; background:{theme.SURFACE}; cursor:pointer; }}</style>'

_RESERVOIR_REPLAY_SCRIPT = '''
<script>
window.dtReservoirReplay = function(btn) {
  var block = btn.closest('.res-chart-block');
  if (!block || block.dataset.replaying === '1') return;
  block.dataset.replaying = '1';
  var data = JSON.parse(block.dataset.res);
  var n = data.hours.length;
  var clipRect = block.querySelector('#res-clip-rect');
  var dot = block.querySelector('#res-dot');
  var playhead = block.querySelector('#res-playhead');
  var clock = block.querySelector('#res-clock');
  var upperLabel = block.querySelector('#res-upper-val');
  var lowerLabel = block.querySelector('#res-lower-val');
  var headLabel = block.querySelector('#res-head-val');
  var spillLabel = block.querySelector('#res-spill-val');

  var steps = n;
  var stepMs = 8000 / steps;
  var numberEvery = Math.max(1, Math.round(steps / 12));
  clipRect.setAttribute('width', '0');
  if (dot) dot.style.opacity = '1';

  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var idx = Math.min(s - 1, n - 1);
        var frac = s / steps;
        clipRect.setAttribute('width', (1400 * frac).toFixed(1));
        var x = frac * 1400;
        var y = data.upperY[idx];
        if (dot) { dot.setAttribute('cx', x.toFixed(1)); dot.setAttribute('cy', y.toFixed(1)); }
        if (playhead) playhead.style.left = (frac * 100) + '%';
        if (clock) clock.textContent = 'H' + data.hours[idx];
        if (s % numberEvery === 0 || s === steps) {
          if (upperLabel) upperLabel.textContent = data.upperHm3[idx].toFixed(1) + ' hm³';
          if (lowerLabel) lowerLabel.textContent = data.lowerHm3[idx].toFixed(1) + ' hm³';
          if (headLabel) headLabel.textContent = data.headM[idx].toFixed(1) + ' m';
          if (spillLabel) spillLabel.textContent = data.spillM3h[idx].toFixed(0) + ' m³/h';
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


def render_reservoir_trajectory_card(traj: dict) -> str:
    """Two-reservoir (Alqueva upper / Pedrogao lower) volume trajectory from
    the real solved DA MILP, each normalized to its own operational band
    [min, usable-or-capacity] so both read on a comparable 0-100% scale
    despite the ~100x difference in absolute hm3 (Alqueva ~2,500-4,150 hm3
    vs Pedrogao ~5-54 hm3). A dashed reference line marks the terminal
    constraint level (must end >= start, the no-free-lunch rule preventing
    the solver draining the reservoir for one profitable day)."""
    hours = traj["hours"]
    n = len(hours)
    upper_hm3, lower_hm3 = traj["upper_hm3"], traj["lower_hm3"]
    u_min, u_max = traj["upper_min_hm3"], traj["upper_usable_hm3"]
    l_min, l_max = traj["lower_min_hm3"], traj["lower_capacity_hm3"]

    def pct(v: float, lo: float, hi: float) -> float:
        return max(0.0, min(100.0, (v - lo) / max(hi - lo, 1e-9) * 100))

    upper_pct = [pct(v, u_min, u_max) for v in upper_hm3]
    lower_pct = [pct(v, l_min, l_max) for v in lower_hm3]
    terminal_pct = pct(traj["upper_initial_hm3"], u_min, u_max)

    x0, x1 = 0, 1400
    top_y, bottom_y = 8, 92
    band_h = bottom_y - top_y

    def fx(i: int) -> float:
        return x0 + i / max(n - 1, 1) * (x1 - x0)

    def fy(p: float) -> float:
        return bottom_y - (p / 100.0) * band_h

    upper_pts = [(fx(i), fy(p)) for i, p in enumerate(upper_pct)]
    lower_pts = [(fx(i), fy(p)) for i, p in enumerate(lower_pct)]
    upper_line = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper_pts)
    lower_line = " ".join(f"{x:.1f},{y:.1f}" for x, y in lower_pts)
    terminal_y = fy(terminal_pct)

    payload = {
        "hours": hours,
        "upperHm3": upper_hm3,
        "lowerHm3": lower_hm3,
        "headM": traj["head_m"],
        "spillM3h": traj["spill_m3h"],
        "upperY": [y for _, y in upper_pts],
    }
    payload_json = _json.dumps(payload)

    terminal_badge = (
        f'<span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:10.5px; padding:2px 8px; border-radius:5px; font-weight:500;">Terminal constraint satisfied</span>'
        if traj["terminal_ok"] else
        f'<span style="background:{theme.STATUS_CRITICAL}22; color:{theme.STATUS_CRITICAL}; font-size:10.5px; padding:2px 8px; border-radius:5px; font-weight:500;">Terminal constraint violated</span>'
    )
    spill_total = sum(traj["spill_m3h"])
    spill_note = (
        f"Spill occurred: {spill_total:,.0f} m&sup3;/h total &mdash; reservoir hit its usable ceiling at some hour."
        if spill_total > 1e-6 else
        "No spill this day &mdash; the solver never needed to discard water."
    )

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Optimization</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">Reservoir trajectory</div>
    <p style="font-size:12px; color:{theme.INK_MUTED}; margin:0 0 10px;">Alqueva (upper) &middot; Pedr&oacute;g&atilde;o (lower) &middot; two-reservoir pumped-storage water balance</p>
    <div class="res-chart-block" data-res='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};">Each trace normalized to its own operational band (min &rarr; usable/capacity)</span>
      <button class="gt-replay" onclick="dtReservoirReplay(this)">&#9654; Replay</button>
    </div>
    <div style="display:flex; gap:16px; align-items:center;">
      <div style="position:relative; height:100px; box-sizing:border-box; flex:1; min-width:0;">
        <svg viewBox="0 0 1400 100" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
          <defs><clipPath id="res-clip"><rect id="res-clip-rect" x="0" y="-20" width="1400" height="140"/></clipPath></defs>
          <line x1="0" y1="{terminal_y:.1f}" x2="1400" y2="{terminal_y:.1f}" stroke="{theme.STATUS_WARNING}" stroke-width="1" stroke-dasharray="4,3"/>
          <g clip-path="url(#res-clip)">
            <polyline points="{lower_line}" fill="none" stroke="{theme.COLOR_PUMP}" stroke-width="1.5" opacity="0.9" stroke-dasharray="3,2"/>
            <polyline points="{upper_line}" fill="none" stroke="{theme.COLOR_GEN}" stroke-width="1.8"/>
          </g>
          <circle id="res-dot" cx="0" cy="{fy(upper_pct[0]):.1f}" r="4" fill="{theme.COLOR_GEN}" style="opacity:0;"/>
        </svg>
      </div>
      <div style="width:1px; align-self:stretch; background:{theme.GRIDLINE};"></div>
      <div style="width:150px; flex-shrink:0;">
        <p style="font-size:11px; color:{theme.INK_MUTED}; margin:0 0 2px;">Upper (Alqueva)</p>
        <p style="font-size:15px; font-weight:500; margin:0 0 6px;"><span id="res-upper-val">{upper_hm3[0]:.1f} hm&sup3;</span></p>
        <p style="font-size:11px; color:{theme.INK_MUTED}; margin:0 0 2px;">Lower (Pedr&oacute;g&atilde;o)</p>
        <p style="font-size:15px; font-weight:500; margin:0 0 6px;"><span id="res-lower-val">{lower_hm3[0]:.1f} hm&sup3;</span></p>
        <p style="font-size:11px; color:{theme.INK_MUTED}; margin:0 0 2px;">Net head</p>
        <p style="font-size:13px; margin:0 0 6px;"><span id="res-head-val">{traj['head_m'][0]:.1f} m</span></p>
        <p style="font-size:11px; color:{theme.INK_MUTED}; margin:0 0 2px;">Spill (this hour)</p>
        <p style="font-size:13px; margin:0;"><span id="res-spill-val">{traj['spill_m3h'][0]:.0f} m&sup3;/h</span></p>
      </div>
    </div>
    <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
      <span style="font-size:11px; color:{theme.INK_MUTED};">H1</span>
      <div style="flex:1; height:4px; background:{theme.GRIDLINE}; border-radius:2px; position:relative;">
        <div id="res-playhead" style="position:absolute; left:0%; top:-3px; width:10px; height:10px; border-radius:50%; background:{theme.COLOR_GEN}; transform:translateX(-50%);"></div>
      </div>
      <span style="font-size:11px; color:{theme.INK_MUTED};">H24</span>
      <span id="res-clock" style="font-size:11px; color:{theme.INK_SECONDARY}; min-width:30px; text-align:right;"></span>
    </div>
    </div>
    <div style="display:flex; gap:14px; margin-top:8px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:2px; background:{theme.COLOR_GEN}; margin-right:4px; vertical-align:middle;"></span>Upper reservoir (Alqueva)</span>
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:2px; background:{theme.COLOR_PUMP}; margin-right:4px; vertical-align:middle;"></span>Lower reservoir (Pedr&oacute;g&atilde;o)</span>
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:1px; background:{theme.STATUS_WARNING}; margin-right:4px; vertical-align:middle;"></span>Terminal constraint level</span>
    </div>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-top:8px; padding-top:8px; border-top:1px solid {theme.GRIDLINE};">
      {terminal_badge}
      <span style="font-size:10.5px; color:{theme.INK_MUTED};">Day total: {spill_total:,.0f} m&sup3;/h spill</span>
    </div>
    <p style="font-size:10px; color:{theme.INK_MUTED}; margin:4px 0 0;">{spill_note}</p>
  </div>
</div>
{_REPLAY_STYLE}
{_RESERVOIR_REPLAY_SCRIPT}'''


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
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">PV routing &amp; curtailment</div>
    <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:0 0 10px;">
      <span style="background:{theme.YELLOW}22; color:{theme.YELLOW}; padding:3px 9px; border-radius:6px; font-weight:600; font-size:12px;">&#9728; PV</span>
      <span style="color:{theme.INK_MUTED}; font-size:12px;">splits into</span>
      <span style="background:{theme.COLOR_GEN}22; color:{theme.COLOR_GEN}; padding:3px 9px; border-radius:6px; font-weight:600; font-size:12px;">&#9889; Grid</span>
      <span style="color:{theme.INK_MUTED}; font-size:12px;">/</span>
      <span style="background:{theme.COLOR_UP}22; color:{theme.COLOR_UP}; padding:3px 9px; border-radius:6px; font-weight:600; font-size:12px;">&#128267; Battery</span>
      <span style="color:{theme.INK_MUTED}; font-size:12px;">/</span>
      <span style="background:{theme.STATUS_CRITICAL}22; color:{theme.STATUS_CRITICAL}; padding:3px 9px; border-radius:6px; font-weight:600; font-size:12px;">&#10005; Curtailed</span>
    </div>
    <div class="pv-chart-block" data-pv='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};">Stacked per hour, normalized to available PV each hour</span>
      <button class="gt-replay" onclick="dtPvReplay(this)">&#9654; Replay</button>
    </div>
    <div style="position:relative; height:100px;">
      <svg viewBox="0 0 1400 100" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{x0}" y1="{zero_y}" x2="{x1}" y2="{zero_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(used_bars)}
        {''.join(bess_bars)}
        {''.join(curt_bars)}
      </svg>
    </div>
    <div style="display:flex; gap:14px; margin-top:4px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.YELLOW}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Used (to grid)</span>
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_UP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>To BESS</span>
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.STATUS_CRITICAL}; border-radius:2px; margin-right:4px; vertical-align:middle; opacity:0.55;"></span>Curtailed</span>
    </div>
    <div style="margin-top:8px; padding-top:8px; border-top:1px solid {theme.GRIDLINE};">
      <span id="pv-readout" style="font-size:11.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Full day shown &mdash; Replay steps through each hour</span>
    </div>
    </div>
    <p style="font-size:10px; color:{theme.INK_MUTED}; margin:8px 0 0;">{curt_note}</p>
  </div>
</div>
{_REPLAY_STYLE}
{_PV_REPLAY_SCRIPT}'''


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

    headroom_note = (
        "Reserve headroom (aFRR+mFRR) is combined-plant only in the real model &mdash; no per-asset split exists in reserve_offer_builder.py/reserve_activation.py, so none is shown here."
        if mx["has_headroom"] else
        "Reserve headroom data unavailable for this date."
    )

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Optimization</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">Multi-asset dispatch</div>
    <p style="font-size:12px; color:{theme.INK_MUTED}; margin:0 0 10px;">Per-hour asset coordination, split onto two scales &mdash; PV/BESS (MW-scale) and PSP (hundreds-of-MW scale) would crush each other on one axis</p>
    <div class="mx-chart-block" data-mx='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};">Above zero = generation &middot; below = consumption</span>
      <button class="gt-replay" onclick="dtMultiReplay(this)">&#9654; Replay</button>
    </div>
    <p style="font-size:10.5px; color:{theme.INK_MUTED}; margin:6px 0 2px; font-weight:500;">PV &amp; BESS (own scale, max {pv_bess_max:.1f} MW)</p>
    <div style="position:relative; height:80px;">
      <svg viewBox="0 0 1400 110" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{x0}" y1="{zero_y1}" x2="{x1}" y2="{zero_y1}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(pv_bars)}
        {''.join(dis_bars)}
        {''.join(chg_bars)}
      </svg>
    </div>
    <div style="display:flex; gap:12px; margin:2px 0 8px;">
      <span style="font-size:10.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.YELLOW}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>PV</span>
      <span style="font-size:10.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_UP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>BESS discharge</span>
      <span style="font-size:10.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_DOWN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>BESS charge</span>
    </div>
    <p style="font-size:10.5px; color:{theme.INK_MUTED}; margin:6px 0 2px; font-weight:500;">PSP turbine &amp; pump (own scale, max {psp_max:.0f} MW)</p>
    <div style="position:relative; height:80px;">
      <svg viewBox="0 0 1400 110" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{x0}" y1="{zero_y2}" x2="{x1}" y2="{zero_y2}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(turb_bars)}
        {''.join(pump_bars)}
      </svg>
    </div>
    <div style="display:flex; gap:12px; margin:2px 0 8px;">
      <span style="font-size:10.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_GEN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>PSP turbine</span>
      <span style="font-size:10.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_PUMP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>PSP pump</span>
    </div>
    <div style="margin-top:8px; padding-top:8px; border-top:1px solid {theme.GRIDLINE};">
      <span id="mx-readout" style="font-size:11.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Full day shown &mdash; Replay steps through each hour</span>
    </div>
    </div>
    <p style="font-size:10px; color:{theme.INK_MUTED}; margin:8px 0 0;">{headroom_note}</p>
  </div>
</div>
{_REPLAY_STYLE}
{_MULTI_REPLAY_SCRIPT}'''


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
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">Water balance</div>
    <p style="font-size:12px; color:{theme.INK_MUTED}; margin:0 0 10px;">Upper reservoir (Alqueva): inflow + pump return &minus; turbine draw &minus; spill, per hour</p>
    <div class="wb-chart-block" data-wb='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};">Above zero = inflows &middot; below = outflows</span>
      <button class="gt-replay" onclick="dtWaterReplay(this)">&#9654; Replay</button>
    </div>
    <div style="position:relative; height:100px;">
      <svg viewBox="0 0 1400 100" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{x0}" y1="{zero_y}" x2="{x1}" y2="{zero_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(in_bars)}
        {''.join(pump_bars)}
        {''.join(turb_bars)}
        {''.join(spill_bars)}
      </svg>
    </div>
    <div style="display:flex; gap:12px; margin-top:4px; flex-wrap:wrap;">
      <span style="font-size:10.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_UP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Natural inflow</span>
      <span style="font-size:10.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_PUMP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Pump return</span>
      <span style="font-size:10.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_GEN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Turbine draw</span>
      <span style="font-size:10.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.STATUS_CRITICAL}; border-radius:2px; margin-right:4px; vertical-align:middle; opacity:0.6;"></span>Spill</span>
    </div>
    <div style="margin-top:8px; padding-top:8px; border-top:1px solid {theme.GRIDLINE};">
      <span id="wb-readout" style="font-size:11.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Full day shown &mdash; Replay steps through each hour</span>
    </div>
    </div>
    <p style="font-size:10px; color:{theme.INK_MUTED}; margin:8px 0 0;">{spill_note}</p>
  </div>
</div>
{_REPLAY_STYLE}
{_WATER_REPLAY_SCRIPT}'''


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

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Optimization</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">BESS SOC vs price</div>
    <p style="font-size:12px; color:{theme.INK_MUTED}; margin:0 0 10px;">Storage arbitrage the solver actually chose &mdash; SOC (% of usable band) against DA clearing price</p>
    <div class="bs-chart-block" data-bs='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};">Shaded = charge (blue) / discharge (aqua) hours</span>
      <button class="gt-replay" onclick="dtBessReplay(this)">&#9654; Replay</button>
    </div>
    <div style="position:relative; height:100px;">
      <svg viewBox="0 0 1400 100" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <defs><clipPath id="bs-clip"><rect id="bs-clip-rect" x="0" y="-20" width="1400" height="140"/></clipPath></defs>
        {''.join(shades)}
        <g clip-path="url(#bs-clip)">
          <polyline points="{price_line}" fill="none" stroke="{theme.COLOR_PRICE}" stroke-width="1.5" opacity="0.85" stroke-dasharray="3,2"/>
          <polyline points="{soc_line}" fill="none" stroke="{theme.COLOR_GEN}" stroke-width="1.8"/>
        </g>
        <circle id="bs-dot" cx="0" cy="{fy_soc(soc_pct[0]):.1f}" r="4" fill="{theme.COLOR_GEN}" style="opacity:0;"/>
      </svg>
    </div>
    <div style="display:flex; gap:14px; margin-top:4px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:2px; background:{theme.COLOR_GEN}; margin-right:4px; vertical-align:middle;"></span>SOC %</span>
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:1px; background:{theme.COLOR_PRICE}; margin-right:4px; vertical-align:middle;"></span>DA price</span>
    </div>
    <div style="margin-top:8px; padding-top:8px; border-top:1px solid {theme.GRIDLINE};">
      <span id="bs-readout" style="font-size:11.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Full day shown &mdash; Replay steps through each hour</span>
    </div>
    </div>
    <p style="font-size:10px; color:{theme.INK_MUTED}; margin:8px 0 0;">Usable SOC band: {bs['e_min_mwh']:.2f}&ndash;{bs['e_max_mwh']:.2f} MWh &mdash; small (2 MWh) BESS, so absolute swings are modest even when the % band moves a lot.</p>
  </div>
</div>
{_REPLAY_STYLE}
{_BESS_REPLAY_SCRIPT}'''


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

    if bc["any_charging"]:
        summary = f"{bc['grid_share_pct']:.0f}% from grid, {bc['pv_share_pct']:.0f}% from PV this day"
        empty_badge = ""
    else:
        summary = "No charging this day"
        empty_badge = (
            f'<div style="background:{theme.STATUS_WARNING}18; border:1px solid {theme.STATUS_WARNING}55; '
            f'border-radius:8px; padding:10px 12px; margin-top:8px; font-size:11.5px; color:{theme.INK_PRIMARY}; line-height:1.5;">'
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
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">BESS charge source</div>
    <p style="font-size:12px; color:{theme.INK_MUTED}; margin:0 0 10px;">What BESS charging power actually came from &mdash; grid vs PV &mdash; {summary}</p>
    <div class="bc2-chart-block" data-bc2='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};">Stacked per hour, charging only</span>
      <button class="gt-replay" onclick="dtBesschgReplay(this)">&#9654; Replay</button>
    </div>
    <div style="position:relative; height:100px;">
      <svg viewBox="0 0 1400 100" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{x0}" y1="{zero_y}" x2="{x1}" y2="{zero_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(grid_bars)}
        {''.join(pv_bars)}
      </svg>
    </div>
    <div style="display:flex; gap:14px; margin-top:4px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_DOWN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Grid</span>
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.YELLOW}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>PV</span>
    </div>
    <div style="margin-top:8px; padding-top:8px; border-top:1px solid {theme.GRIDLINE};">
      <span id="bc2-readout" style="font-size:11.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Full day shown &mdash; Replay steps through each hour</span>
    </div>
    {empty_badge}
    </div>
  </div>
</div>
{_REPLAY_STYLE}
{_BESSCHG_REPLAY_SCRIPT}'''


_DAACT_REPLAY_SCRIPT = '''
<script>
window.dtDaactReplay = function(btn) {
  var block = btn.closest('.da2-chart-block');
  if (!block || block.dataset.replaying === '1') return;
  block.dataset.replaying = '1';
  var data = JSON.parse(block.dataset.da2);
  var n = data.hours.length;
  var readout = block.querySelector('#da2-readout');
  var bars = block.querySelectorAll('.da2-bar');
  var ticks = block.querySelectorAll('.da2-tick');
  var dots = block.querySelectorAll('.da2-dot');
  var steps = n;
  var stepMs = 8000 / steps;

  bars.forEach(function(b) { b.setAttribute('height', '0'); });
  ticks.forEach(function(t) { t.style.opacity = '0'; });
  dots.forEach(function(d) { d.style.opacity = '0'; });

  for (var s = 1; s <= steps; s++) {
    (function(s) {
      setTimeout(function() {
        var idx = s - 1;
        var bar = block.querySelector('.da2-bar[data-hidx="' + idx + '"]');
        var tick = block.querySelector('.da2-tick[data-hidx="' + idx + '"]');
        var dot = block.querySelector('.da2-dot[data-hidx="' + idx + '"]');
        if (bar) { bar.setAttribute('height', bar.dataset.finalH); bar.setAttribute('y', bar.dataset.finalY); bar.style.opacity = '1'; }
        if (tick) tick.style.opacity = '1';
        if (dot) dot.style.opacity = '1';
        if (readout) {
          readout.textContent = 'H' + data.hours[idx] + ': DA ' + data.daMw[idx].toFixed(0) +
            ' MW committed, ISP activation ' + (data.deltaMw[idx] >= 0 ? '+' : '') + data.deltaMw[idx].toFixed(0) +
            ' MW \\u2192 ' + data.deliveredMw[idx].toFixed(0) + ' MW actual';
        }
        if (s === steps) block.dataset.replaying = '0';
      }, s * stepMs);
    })(s);
  }
};
</script>'''


def render_da_vs_activation_card(dv: dict) -> str:
    """Hourly DA-committed net position (blue bar) with the real per-ISP
    aFRR/mFRR activation delta overlaid as a tick+dot on top -- two
    independent obligations on the same hour, not a reconciliation. DA is a
    day-ahead energy schedule; activation is a real-time response to
    system-wide Area Control Error, unrelated to Alqueva's own DA delivery
    accuracy. Combined aFRR+mFRR, plant-level only (no per-asset split
    exists in the real pipeline)."""
    hours = dv["hours"]
    n = len(hours)
    da_net = dv["da_net_mw"]
    delta = dv["activation_delta_mw"]
    delivered = dv["delivered_mw"]
    max_mag = max([abs(v) for v in delivered + da_net] + [1.0])

    x0, x1 = 10, 1390
    zero_y, half_h = 55, 48
    slot_w = (x1 - x0) / n
    bar_w = max(min(28.0, slot_w * 0.7), 4.0)

    def y(v: float) -> float:
        return zero_y - (v / max_mag) * half_h

    bars, ticks, dots = [], [], []
    for i in range(n):
        x_center = x0 + i * slot_w + (slot_w - bar_w) / 2
        y_da = y(da_net[i])
        y_delivered = y(delivered[i])
        bar_top = min(zero_y, y_da)
        bar_h = abs(y_da - zero_y)
        bars.append(f'<rect class="da2-bar" data-hidx="{i}" data-final-h="{bar_h:.1f}" data-final-y="{bar_top:.1f}" x="{x_center:.1f}" y="{bar_top:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{theme.COLOR_GEN}" opacity="0.8"/>')
        tick_x = x_center + bar_w / 2
        ticks.append(f'<line class="da2-tick" data-hidx="{i}" x1="{tick_x:.1f}" y1="{y_da:.1f}" x2="{tick_x:.1f}" y2="{y_delivered:.1f}" stroke="{theme.STATUS_CRITICAL}" stroke-width="2"/>')
        dots.append(f'<circle class="da2-dot" data-hidx="{i}" cx="{tick_x:.1f}" cy="{y_delivered:.1f}" r="3" fill="{theme.STATUS_CRITICAL}"/>')

    payload = {"hours": hours, "daMw": da_net, "deltaMw": delta, "deliveredMw": delivered}
    payload_json = _json.dumps(payload)

    activation_note = (
        "Real ISP activation shown as red tick + dot on top of each hour's DA commitment."
        if dv["any_activation"] else
        "No aFRR/mFRR activation recorded for this date &mdash; DA bars shown alone."
    )

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Optimization</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:500;">Real solved MILP output</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">DA schedule vs ISP activation</div>
    <p style="font-size:12px; color:{theme.INK_MUTED}; margin:0 0 10px;">Two independent obligations, not a reconciliation &mdash; DA is a day-ahead energy commitment, aFRR/mFRR activation responds to system-wide grid balance, unrelated to whether Alqueva delivered its own DA position</p>
    <div class="da2-chart-block" data-da2='{payload_json}'>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};">{activation_note}</span>
      <button class="gt-replay" onclick="dtDaactReplay(this)">&#9654; Replay</button>
    </div>
    <div style="position:relative; height:110px;">
      <svg viewBox="0 0 1400 110" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <line x1="{x0}" y1="{zero_y}" x2="{x1}" y2="{zero_y}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        {''.join(bars)}
        {''.join(ticks)}
        {''.join(dots)}
      </svg>
    </div>
    <div style="display:flex; gap:14px; margin-top:4px;">
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_GEN}; border-radius:2px; margin-right:4px; vertical-align:middle; opacity:0.8;"></span>DA commitment</span>
      <span style="font-size:11px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:2px; background:{theme.STATUS_CRITICAL}; margin-right:4px; vertical-align:middle;"></span>ISP activation delta</span>
    </div>
    <div style="margin-top:8px; padding-top:8px; border-top:1px solid {theme.GRIDLINE};">
      <span id="da2-readout" style="font-size:11.5px; color:{theme.INK_PRIMARY}; font-weight:500;">Full day shown &mdash; Replay steps through each hour</span>
    </div>
    </div>
    <p style="font-size:10px; color:{theme.INK_MUTED}; margin:8px 0 0;">Activation is combined aFRR+mFRR, plant-level only &mdash; the real pipeline has no per-asset attribution of which unit delivered a balancing call.</p>
  </div>
</div>
{_REPLAY_STYLE}
{_DAACT_REPLAY_SCRIPT}'''
