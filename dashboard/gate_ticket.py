"""gate_ticket.py - renders data.latest_gate_ticket() as a self-contained,
animated IEEE-style figure card. One components.html block (not
st.markdown) on purpose: the Replay button's JS and the elements it
animates must live in the same DOM, which an iframe guarantees and
markdown-injected <script> tags do not (Streamlit strips them).
"""
from __future__ import annotations

import html as _html
import json as _json

import theme
from dispatch_ticket import _HOVER_CROSSHAIR_SCRIPT, _hover_svg_elems, _hover_tooltip_div

_STATUS_BG = {"good": "#eaf3de", "neutral": "#f1efe8", "critical": "#fcebeb"}
_STATUS_FG = {"good": "#173404", "neutral": "#444441", "critical": "#4a1b0c"}
_STATUS_DOT = {"good": theme.STATUS_GOOD, "neutral": theme.STATUS_NEUTRAL, "critical": theme.STATUS_CRITICAL}
_CAPTION_GATE_NAME = {
    "DA": "Day-ahead", "IDA1": "IDA1", "IDA2": "IDA2", "IDA3": "IDA3",
    "XBID": "XBID", "AFRR": "aFRR", "MFRR": "mFRR",
}
_TRADABLE_LIGHT = "#b5d4f4"
_FROZEN_GRAY = "#c3c2b7"


def gate_caption_name(gate: str) -> str:
    """Short display name for a gate key (e.g. "DA" -> "Day-ahead")."""
    return _CAPTION_GATE_NAME.get(gate, gate)


def _bars_svg(ticket: dict) -> tuple[str, str, str, int, str]:
    """Returns (svg_groups_markup, y_axis_labels_html, hour_row_html, n_bars)
    for the hourly bars + tick marks. Geometry only in the SVG (bars/lines,
    rendered with preserveAspectRatio="none" so they stretch to fill the
    card's full width exactly) -- no <text> inside it. Text is real HTML
    instead: SVG's default uniform scaling shrinks font-size right along
    with fitting an unpredictable container width, which is what made every
    label (hour ticks, axis values, "Hour of Day") render unreadably small
    once the card itself became much wider. HTML text has no such coupling."""
    hourly = ticket["hourly"]
    n = len(hourly)
    if n == 0:
        return "", "", "", 0, ""

    plot_x0, plot_x1 = 0, 1000
    plot_y0, plot_y1, zero_y = 4, 96, 50
    slot_w = (plot_x1 - plot_x0) / n
    bar_w = min(10, slot_w * 0.62)

    if ticket["is_reserve"]:
        vals = [max(h["up_mw"], h["dn_mw"]) for h in hourly]
    else:
        vals = [abs(h["volume_mwh"]) for h in hourly]
    max_abs = max(vals) if vals and max(vals) > 0 else 1.0
    scale = (zero_y - plot_y0) / max_abs

    bars = []
    for i, h in enumerate(hourly):
        x = plot_x0 + i * slot_w + (slot_w - bar_w) / 2
        # No CSS animation/animation-delay here on purpose. An earlier
        # version staggered the reveal with per-element animation-delay  - 
        # correct in principle, but CSS animation timing is unreliable once
        # the browser tab loses focus for even a moment mid-animation (a
        # near-certainty for this user's actual workflow, which constantly
        # switches to VS Code and back): longer-delay bars could end up
        # permanently stuck invisible. No class in the markup at all here;
        # the Replay button's JS drives the reveal with setTimeout, which  - 
        # unlike CSS animation-delay - reliably fires even across a
        # visibility change.
        if ticket["is_reserve"]:
            up, dn = h["up_mw"], h["dn_mw"]
            if up > 0:
                bh = up * scale
                bars.append(f'<rect data-kind="gt-bar" data-final-y="{zero_y - bh:.1f}" data-final-h="{bh:.1f}" '
                            f'x="{x:.1f}" y="{zero_y - bh:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{theme.COLOR_UP}"/>')
            if dn > 0:
                bh = dn * scale
                bars.append(f'<rect data-kind="gt-bar-dn" data-final-y="{zero_y:.1f}" data-final-h="{bh:.1f}" '
                            f'x="{x:.1f}" y="{zero_y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{theme.COLOR_DOWN}"/>')
        else:
            v = h["volume_mwh"]
            bh = abs(v) * scale
            if v >= 0:
                bars.append(f'<rect data-kind="gt-bar" data-final-y="{zero_y - bh:.1f}" data-final-h="{bh:.1f}" '
                            f'x="{x:.1f}" y="{zero_y - bh:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{theme.COLOR_GEN}"/>')
            else:
                bars.append(f'<rect data-kind="gt-bar-dn" data-final-y="{zero_y:.1f}" data-final-h="{bh:.1f}" '
                            f'x="{x:.1f}" y="{zero_y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{theme.COLOR_PUMP}"/>')

    # Sparse hour ticks: first, quarter points, last - never every hour.
    hours = [h["hour"] for h in hourly]
    tick_idxs = sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1]))
    ticks = []
    for idx in tick_idxs:
        x = plot_x0 + idx * slot_w + slot_w / 2
        ticks.append(f'<line x1="{x:.1f}" y1="{plot_y1}" x2="{x:.1f}" y2="{plot_y1 - 5}" stroke="#0b0b0b" stroke-width="1"/>')

    max_label = f"{max_abs:.0f}"
    playhead_x0 = plot_x0 + slot_w / 2
    playhead_x1 = plot_x0 + (n - 1) * slot_w + slot_w / 2

    svg = f'''
      <line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y1}" stroke="#0b0b0b" stroke-width="1.25"/>
      <line x1="{plot_x0}" y1="{plot_y1}" x2="{plot_x1}" y2="{plot_y1}" stroke="#0b0b0b" stroke-width="1.25"/>
      <line x1="{plot_x0}" y1="{zero_y}" x2="{plot_x1}" y2="{zero_y}" stroke="#0b0b0b" stroke-width="0.75" stroke-dasharray="1.5,1.5" opacity="0.5"/>
      {''.join(ticks)}
      {''.join(bars)}
      <line data-kind="gt-playhead" data-x0="{playhead_x0:.1f}" data-x1="{playhead_x1:.1f}"
            x1="{playhead_x0:.1f}" y1="{plot_y0}" x2="{playhead_x0:.1f}" y2="{plot_y1}"
            stroke="#0b0b0b" stroke-width="1" stroke-dasharray="2,2" style="opacity:0;"/>
      {_hover_svg_elems("gt-bars", plot_x0, plot_x1, 116, 0)}
    '''

    if ticket["is_reserve"]:
        hover_traces = [
            {"label": "Up", "yvals": [h["up_mw"] for h in hourly], "py": [h["up_mw"] for h in hourly], "color": theme.COLOR_UP, "unit": "MW", "decimals": 2},
            {"label": "Down", "yvals": [h["dn_mw"] for h in hourly], "py": [h["dn_mw"] for h in hourly], "color": theme.COLOR_DOWN, "unit": "MW", "decimals": 2},
        ]
    else:
        hover_traces = [
            {"label": "Volume", "yvals": [h["volume_mwh"] for h in hourly], "py": [h["volume_mwh"] for h in hourly],
             "color": theme.COLOR_GEN, "negColor": theme.COLOR_PUMP, "unit": "MWh", "decimals": 2, "signed": True},
        ]
    hover_cfg = _json.dumps({
        "n": n, "x0": plot_x0, "x1": plot_x1, "viewW": 1000,
        "catchSelector": "#gt-bars-hover-catch", "lineSelector": "#gt-bars-hover-line",
        "tooltipSelector": "#gt-bars-tooltip", "indexLabel": "Hour", "indexVals": hours,
        "traces": hover_traces,
    })
    hover_tooltip = _hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="gt-bars-tooltip"')
    hover_script = f'''
    {_HOVER_CROSSHAIR_SCRIPT}
    <script>
    (function() {{
      var block = document.querySelector('.gt-bars-chart-block');
      if (block) {{ dtHoverInit(block, {hover_cfg}); }}
    }})();
    </script>'''

    # Real HTML labels, positioned by percentage over the chart box (which
    # is position:relative in the caller) -- see docstring for why not SVG
    # <text>. y-axis values sit to the left at their exact fraction of the
    # box height; hour ticks sit in a plain flex row below (always evenly
    # spaced across the 5 sparse tick points, so no percentage math needed
    # there -- justify-content:space-between does it).
    def _y_pct(y: float) -> float:
        return y / 116 * 100

    hour_labels = "".join(
        f'<span style="font-size:12px; color:{theme.INK_MUTED};">{hours[idx]}</span>' for idx in tick_idxs
    )
    overlay = f'''
      <div style="position:absolute; left:0; top:{_y_pct(plot_y0):.1f}%; transform:translateY(-50%); font-size:12px; font-weight:600; color:{theme.INK_PRIMARY};">{max_label}</div>
      <div style="position:absolute; left:0; top:{_y_pct(zero_y):.1f}%; transform:translateY(-50%); font-size:12px; font-weight:600; color:{theme.INK_PRIMARY};">0</div>
      <div style="position:absolute; left:0; top:{_y_pct(plot_y1):.1f}%; transform:translateY(-50%); font-size:12px; font-weight:600; color:{theme.INK_PRIMARY};">-{max_label}</div>
    '''
    hour_row = f'<div style="display:flex; justify-content:space-between; margin-top:4px;">{hour_labels}</div>'
    overlay = hover_tooltip + overlay
    return svg, overlay, hour_row, n, hover_script


def _standard_chart_block(ticket: dict, delivery_date: str, fig_num: int) -> str:
    """DA/aFRR/mFRR: the full 24h absolute bar chart + Replay button."""
    unit = "MW" if ticket["is_reserve"] else "MWh"
    bars_svg, y_axis_html, hour_row_html, n_hours, hover_script = _bars_svg(ticket)
    if ticket["is_reserve"]:
        legend_items = [(theme.COLOR_UP, "Up"), (theme.COLOR_DOWN, "Down")]
    else:
        legend_items = [(theme.COLOR_PUMP, "Buy (pump)"), (theme.COLOR_GEN, "Sell (generate)")]
    legend_html = "".join(
        f'<span style="font-size:11px; color:{theme.INK_SECONDARY};">'
        f'<span style="display:inline-block; width:9px; height:9px; background:{color}; '
        f'border-radius:2px; margin-right:4px; vertical-align:middle;"></span>{label}</span>'
        for color, label in legend_items
    )
    return f'''
    <div style="display:flex; align-items:center; justify-content:space-between; margin:14px 0 8px;">
      <p style="font-size:12px; color:{theme.INK_SECONDARY}; margin:0;">Hourly {'reserve capacity' if ticket['is_reserve'] else 'bid volume'} ({unit})</p>
      <button class="gt-replay" id="gt-replay-btn">&#9654; Replay</button>
    </div>
    <div class="gt-bars-chart-block" style="position:relative; height:100px; padding-left:26px; box-sizing:border-box;">
      <svg viewBox="0 0 1000 116" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        {bars_svg}
      </svg>
      {y_axis_html}
    </div>
    <div style="padding-left:26px;">{hour_row_html}</div>
    <div style="display:flex; gap:14px; margin:6px 0 0;">
      {legend_html}
    </div>
    <p style="font-size:10.5px; color:{theme.INK_SECONDARY}; text-align:center; margin:4px 0 0;">
      <b>Fig. {fig_num}.</b>&nbsp; {_html.escape(gate_caption_name(ticket['gate']))} hourly {'reserve capacity' if ticket['is_reserve'] else 'bid volume'}, delivery {_html.escape(delivery_date)}.
    </p>
    {hover_script}
    ''', n_hours


def _rebid_block(info: dict, wrapper_id: str = "", hidden: bool = False) -> str:
    """IDA1-3/XBID: the improvement-vs-threshold comparison (the number
    that actually drove the decision) plus which hours were tradable and
    which of those actually traded - instead of the 24h absolute chart,
    which usually looks identical to DA's since most hours are frozen or
    simply unchanged. See dashboard/data.py::_build_ticket for the fields.
    wrapper_id/hidden let XBID render one block per check-window and toggle
    which is visible client-side (see _xbid_windows_block)."""
    info = info or {}
    improvement = info.get("improvement_eur")
    threshold = info.get("threshold_eur")
    tradable_hours = sorted(info.get("tradable_hours") or [])
    traded_hours = sorted(info.get("traded_hours") or [])
    n_total = info.get("n_total_hours") or 24
    cleared = improvement is not None and threshold is not None and improvement >= threshold

    bars_html = ""
    caption_html = ""
    if improvement is not None or threshold is not None:
        # Full-width horizontal scale instead of two separate vertical bars  - 
        # the two vertical bars left most of the card's (now much wider)
        # width empty on the right; a single 0-to-max track puts improvement
        # and threshold on the same line so "did it clear the bar" reads as
        # one glance at where the fill ends relative to the marker, and the
        # track uses the full card width the same way the tradable-hours
        # strip below it already does.
        #
        # Labels are plain HTML spans positioned with left:%, NOT SVG <text>.
        # An earlier version put the labels inside the SVG with a wide
        # viewBox (needed so the bar itself stretches full width) - but SVG's
        # default "meet" scaling shrinks EVERYTHING (including font-size) by
        # whichever of width/height is more constrained, so a viewBox
        # authored wide enough to fill an unpredictable container width
        # shrinks the text along with it. HTML text has no such coupling:
        # font-size stays exactly what's specified regardless of the SVG's
        # own scaling next to it.
        max_val = max(abs(improvement or 0), abs(threshold or 0), 1.0) * 1.12
        bar_color = theme.COLOR_PRICE if cleared else theme.STATUS_NEUTRAL

        fill_val = max(improvement or 0, 0)
        fill_pct = max(min(fill_val / max_val, 1.0), 0.0) * 100
        threshold_pct = max(min((threshold or 0) / max_val, 1.0), 0.0) * 100

        track_svg = (
            f'<svg viewBox="0 0 1000 20" preserveAspectRatio="none" '
            f'style="width:100%; height:20px; display:block;">'
            f'<rect x="0" y="0" width="1000" height="20" rx="4" fill="{theme.SURFACE}" stroke="{theme.GRIDLINE}" stroke-width="1"/>'
            f'<rect data-kind="gt-rebid-fill" data-final-w="{fill_pct * 10:.1f}" '
            f'x="0" y="0" width="{fill_pct * 10:.1f}" height="20" rx="4" fill="{bar_color}"/>'
            f'</svg>'
        )

        # Label anchor near either edge: a centered label (translateX(-50%))
        # at threshold_pct close to 0 or 100 overflows past the track and
        # gets clipped by the card edge -- switch to left/right anchoring
        # (matching the improvement label's existing left-edge handling
        # below) instead of centering, once the marker is within 8% of
        # either end. The marker LINE itself always stays exactly at
        # threshold_pct; only the text label's anchor point changes.
        labels = []
        if threshold is not None:
            if threshold_pct < 8:
                th_transform = "translateX(0)"
            elif threshold_pct > 92:
                th_transform = "translateX(-100%)"
            else:
                th_transform = "translateX(-50%)"
            labels.append(
                f'<div data-kind="gt-rebid-marker" style="position:absolute; left:{threshold_pct:.1f}%; top:-20px; transform:{th_transform}; '
                f'font-size:13px; font-weight:500; color:{theme.STATUS_GOOD}; white-space:nowrap;">&euro;{threshold:,.0f} threshold</div>'
                f'<div data-kind="gt-rebid-marker" style="position:absolute; left:{threshold_pct:.1f}%; top:0; bottom:0; width:2px; '
                f'background:{theme.STATUS_GOOD}; transform:translateX(-1px);"></div>'
            )
        if improvement is not None:
            imp_transform = "translateX(-100%)" if fill_pct > 12 else "translateX(0)"
            labels.append(
                f'<div style="position:absolute; left:{fill_pct:.1f}%; bottom:-20px; transform:{imp_transform}; '
                f'font-size:13px; font-weight:500; color:{bar_color}; white-space:nowrap;">&euro;{improvement:,.0f} improvement</div>'
            )

        bars_html = (
            f'<div style="display:flex; align-items:center; justify-content:space-between; margin-top:14px;">'
            f'<p style="font-size:12px; color:{theme.INK_SECONDARY}; margin:0;">Improvement vs re-bid threshold</p>'
            f'<button class="gt-replay" onclick="gtReplayRebid(this)">&#9654; Replay</button>'
            f'</div>'
            f'<div style="position:relative; margin:12px 0 26px;">{track_svg}{"".join(labels)}</div>'
        )
        if improvement is not None and threshold is not None:
            caption = "Cleared the bar - re-bid submitted" if cleared else "Didn&rsquo;t clear the bar - position held"
            caption_html = f'<p style="font-size:11px; color:{bar_color}; text-align:center; margin:8px 0 0;">{caption}</p>'

    hours_html = ""
    if tradable_hours:
        strip_id = f"gt-strip{('-' + wrapper_id) if wrapper_id else ''}"
        strip_vb_w = 1400
        seg_w = strip_vb_w / n_total
        traded_set = set(traded_hours)
        tradable_set = set(tradable_hours)
        segs = []
        status_labels = []
        for h in range(1, n_total + 1):
            x = (h - 1) * seg_w
            if h in traded_set:
                color = theme.COLOR_PRICE
                status_labels.append("Traded")
            elif h in tradable_set:
                color = _TRADABLE_LIGHT
                status_labels.append("Tradable")
            else:
                color = _FROZEN_GRAY
                status_labels.append("Frozen")
            segs.append(f'<rect data-kind="gt-tradable-seg" data-final-fill="{color}" '
                        f'x="{x:.1f}" y="4" width="{seg_w * 0.8:.1f}" height="10" rx="1" fill="{color}"/>')
        legend_bits = [
            f'<span style="font-size:10px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:8px; height:8px; background:{_FROZEN_GRAY}; border-radius:2px; margin-right:4px;"></span>Frozen</span>',
            f'<span style="font-size:10px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:8px; height:8px; background:{_TRADABLE_LIGHT}; border-radius:2px; margin-right:4px;"></span>Tradable</span>',
        ]
        if traded_hours:
            legend_bits.append(
                f'<span style="font-size:10px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:8px; height:8px; background:{theme.COLOR_PRICE}; border-radius:2px; margin-right:4px;"></span>Traded</span>'
            )
        strip_label = (f"Hours traded ({len(traded_hours)} of {len(tradable_hours)} tradable)" if traded_hours
                        else f"Tradable hours ({len(tradable_hours)} of {n_total})")
        # Categorical strip (Frozen/Tradable/Traded per hour) -- the color
        # alone already tells the full story, so this doesn't reuse the
        # generic numeric dtHoverInit module (which formats values with
        # toFixed). Just enough bespoke JS to snap a line to the hovered
        # hour and show its status as text; no dots, nothing numeric.
        status_json = _json.dumps(status_labels)
        hours_html = f'''
        <div style="border-top:1px solid {theme.GRIDLINE}; padding-top:10px; margin-top:12px;">
          <p style="font-size:11px; color:{theme.INK_SECONDARY}; margin:0 0 6px;">{strip_label}</p>
          <div style="position:relative;">
            <svg viewBox="0 0 {strip_vb_w} 22" preserveAspectRatio="none" style="width:100%; height:20px; display:block;">
              {''.join(segs)}
              <rect id="{strip_id}-hover-catch" x="0" y="0" width="{strip_vb_w}" height="22" fill="transparent"/>
              <line id="{strip_id}-hover-line" x1="0" y1="0" x2="0" y2="22" stroke="{theme.INK_PRIMARY}" stroke-width="1" stroke-dasharray="2,2" opacity="0"/>
            </svg>
            {_hover_tooltip_div().replace('dt-hover-tooltip"', f'dt-hover-tooltip" id="{strip_id}-tooltip"')}
          </div>
          <div style="display:flex; justify-content:space-between; margin-top:2px;">
            <span style="font-size:9px; color:{theme.INK_MUTED};">H1</span>
            <span style="font-size:9px; color:{theme.INK_MUTED};">H{n_total}</span>
          </div>
          <div style="display:flex; gap:12px; margin-top:6px;">{''.join(legend_bits)}</div>
        </div>
        <script>
        (function() {{
          var catchEl = document.getElementById("{strip_id}-hover-catch");
          var line = document.getElementById("{strip_id}-hover-line");
          var tooltip = document.getElementById("{strip_id}-tooltip");
          var statuses = {status_json};
          var n = statuses.length;
          var vbW = {strip_vb_w};
          if (!catchEl) return;
          catchEl.addEventListener('mousemove', function(evt) {{
            var svg = catchEl.ownerSVGElement;
            var rect = svg.getBoundingClientRect();
            var vbX = (evt.clientX - rect.left) / rect.width * vbW;
            var idx = Math.max(0, Math.min(n - 1, Math.floor(vbX / (vbW / n))));
            var cx = (idx + 0.5) * (vbW / n);
            line.setAttribute('x1', cx); line.setAttribute('x2', cx); line.setAttribute('opacity', '1');
            tooltip.innerHTML = '<div style="color:{theme.INK_SECONDARY};">Hour ' + (idx + 1) + '</div>' +
              '<div style="font-weight:600;">' + statuses[idx] + '</div>';
            tooltip.style.display = 'block';
          }});
          catchEl.addEventListener('mouseleave', function() {{
            line.setAttribute('opacity', '0');
            tooltip.style.display = 'none';
          }});
        }})();
        </script>'''
    elif improvement is not None and threshold is not None:
        # No tradable-hours data on this event (an older run, before this
        # field was added) -- say so plainly instead of just omitting the
        # section, so it doesn't read as "nothing traded" by default.
        hours_html = f'''
        <div style="border-top:1px solid {theme.GRIDLINE}; padding-top:10px; margin-top:12px;">
          <p style="font-size:11px; color:{theme.INK_SECONDARY}; margin:0 0 6px;">Hours traded</p>
          <div style="display:flex; align-items:center; gap:8px; background:{theme.SURFACE}; border-radius:6px; padding:8px 10px; border:1px solid {theme.GRIDLINE};">
            <span style="font-size:11px; color:{theme.INK_MUTED};">Not recorded for this decision (older run)</span>
          </div>
        </div>'''

    body = bars_html + caption_html + hours_html
    style = "display:none;" if hidden else ""
    id_bit = f' id="{wrapper_id}"' if wrapper_id else ""
    # gt-rebid-scope is what the Replay button's onclick walks up to via
    # closest() -- always present (not just when wrapper_id is set) so a
    # single-window IDA1-3 card's button can find its own elements the same
    # way an XBID window's button finds only its own, not another window's.
    return f'<div class="gt-rebid-scope"{id_bit} style="{style}">{body}</div>'


def _xbid_windows_block(windows: list[dict]) -> str:
    """XBID re-checks several times a day (W1, W2, ...) - one pill per
    window above the usual improvement-vs-threshold block, defaulting to
    the latest. Pure client-side toggle (no Streamlit rerun) so it stays
    self-contained like the Replay button. Only rendered when there's more
    than one window; a single-window day just shows the block directly."""
    if len(windows) <= 1:
        return _rebid_block(windows[0]["rebid_info"] if windows else {})

    pills = []
    blocks = []
    last_idx = len(windows) - 1
    for i, w in enumerate(windows):
        dot = _STATUS_DOT[w["status_class"]]
        active = i == last_idx
        pills.append(
            f'<button class="gt-xbid-pill{" gt-xbid-pill-active" if active else ""}" '
            f'data-win-idx="{i}" onclick="gtSelectXbidWindow({i})">'
            f'<span style="display:inline-block; width:6px; height:6px; border-radius:50%; '
            f'background:{dot}; margin-right:5px;"></span>{_html.escape(w["window"])}'
            f'<span style="opacity:0.6; margin-left:4px;">{_html.escape(w["decision"])}</span></button>'
        )
        blocks.append(_rebid_block(w["rebid_info"], wrapper_id=f"gt-xbid-block-{i}", hidden=not active))

    pills_html = "".join(pills)
    blocks_html = "".join(blocks)
    return f'''
    <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:2px;">{pills_html}</div>
    {blocks_html}
    <style>
      .gt-xbid-pill {{ font-size:10.5px; padding:2px 9px; border:1px solid {theme.GRIDLINE}; border-radius:12px;
                        background:{theme.SURFACE}; color:{theme.INK_SECONDARY}; cursor:pointer; }}
      .gt-xbid-pill-active {{ border-color:{theme.INK_PRIMARY}; color:{theme.INK_PRIMARY}; font-weight:500; }}
    </style>
    <script>
    function gtSelectXbidWindow(idx) {{
      document.querySelectorAll('.gt-xbid-pill').forEach(function(el, i) {{
        el.classList.toggle('gt-xbid-pill-active', i === idx);
      }});
      document.querySelectorAll('[id^="gt-xbid-block-"]').forEach(function(el) {{
        el.style.display = (el.id === 'gt-xbid-block-' + idx) ? '' : 'none';
      }});
    }}
    </script>'''


def render(ticket: dict, delivery_date: str, fig_num: int = 1) -> str:
    """Returns the full self-contained HTML for the gate ticket card."""
    is_rebid = bool(ticket.get("is_rebid_gate"))

    bg = _STATUS_BG[ticket["status_class"]]
    fg = _STATUS_FG[ticket["status_class"]]
    dot = _STATUS_DOT[ticket["status_class"]]
    ref_bit = f'OMIE/MARI ref {_html.escape(ticket["ref"])} &middot; ' if ticket.get("ref") else ""

    # Rebid gates show improvement inside their own comparison bars below  - 
    # showing it again as a plain metric card up top would just repeat the
    # same number without the threshold it's being judged against.
    revenue_items = ticket.get("revenue_items", [])
    if is_rebid:
        revenue_items = [(l, v) for l, v in revenue_items if l != "Improvement vs committed"]
    revenue_cards = "".join(
        f'<div style="background:{theme.SURFACE}; border-radius:8px; padding:0.6rem 0.8rem;">'
        f'<p style="font-size:12px; color:{theme.INK_SECONDARY}; margin:0 0 2px;">{_html.escape(rlabel)}</p>'
        f'<p style="font-size:18px; font-weight:500; margin:0;">&euro;{rvalue:,.0f}</p></div>'
        for rlabel, rvalue in revenue_items
    )
    # Reserve-capacity gates (aFRR/mFRR): add an avg clearing price card
    # alongside the revenue card. cap_up_eur_mw/cap_dn_eur_mw come straight
    # from ReserveStore -- the actual per-hour price this capacity cleared
    # at, not a derived number -- weighted by the MW offered at that price
    # so an hour with a big offer counts more than a token 0.1 MW hour.
    if ticket.get("is_reserve") and ticket.get("hourly"):
        total_mw = sum(h["up_mw"] + h["dn_mw"] for h in ticket["hourly"])
        if total_mw > 0:
            weighted = sum(h["up_mw"] * h["cap_up_eur_mw"] + h["dn_mw"] * h["cap_dn_eur_mw"]
                           for h in ticket["hourly"])
            avg_price = weighted / total_mw
            revenue_cards += (
                f'<div style="background:{theme.SURFACE}; border-radius:8px; padding:0.6rem 0.8rem;">'
                f'<p style="font-size:12px; color:{theme.INK_SECONDARY}; margin:0 0 2px;">Avg clearing price</p>'
                f'<p style="font-size:18px; font-weight:500; margin:0;">&euro;{avg_price:,.2f}'
                f'<span style="font-size:11px; color:{theme.INK_MUTED};">/MW/h</span></p></div>'
            )

    revenue_bit = (f'<div style="display:flex; gap:10px; margin-top:10px; flex-wrap:wrap;">{revenue_cards}</div>'
                   if revenue_cards else "")

    phase_bit = (f'<span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">'
                 f'{_html.escape(ticket["phase_label"])}</span>' if ticket.get("phase_label") else "")

    label = _html.escape(ticket["label"])
    decision = _html.escape(ticket["decision"])
    ts = _html.escape(ticket["timestamp"])
    # Same "Gate closes (CET): HH:MM <-- submit before this" info the
    # terminal prints at the DA approval prompt (see
    # trader_approval_prompt.py) -- surfaced here too so it isn't lost the
    # moment the terminal scrolls past it. None for gates data.py couldn't
    # honestly resolve a display string for (shouldn't happen for the 7
    # known gates, but a ticket from an unrecognised gate key degrades
    # gracefully rather than erroring).
    gate_close_bit = (
        f'<p style="font-size:12px; color:{theme.INK_SECONDARY}; margin:0 0 4px;">'
        f'&#128274; {_html.escape(ticket["gate_close"])}</p>'
        if ticket.get("gate_close") else ""
    )

    if is_rebid:
        xbid_windows = ticket.get("xbid_windows")
        body = _xbid_windows_block(xbid_windows) if xbid_windows else _rebid_block(ticket.get("rebid_info"))
        replay_script = f'''
<script>
(function() {{
  // Same setTimeout-driven reveal as the standard bar chart's Replay
  // button (see below) and for the same reason: no CSS animation-delay,
  // since that stalls permanently once the tab loses focus mid-play.
  // Scoped via closest('.gt-rebid-scope') from the clicked button, not a
  // shared id, because XBID renders one hidden block per check-window in
  // the same DOM -- a shared id would only ever reach the first one.
  window.gtReplayRebid = function(btn) {{
    var scope = btn.closest('.gt-rebid-scope');
    if (!scope || scope.dataset.replaying === '1') return;
    scope.dataset.replaying = '1';

    var fillEl = scope.querySelector('[data-kind="gt-rebid-fill"]');
    var markers = scope.querySelectorAll('[data-kind="gt-rebid-marker"]');
    var segs = Array.prototype.slice.call(scope.querySelectorAll('[data-kind="gt-tradable-seg"]'));

    if (fillEl) fillEl.setAttribute('width', '0');
    markers.forEach(function(el) {{ el.style.opacity = '0'; }});
    segs.forEach(function(el) {{ el.setAttribute('fill', '{_FROZEN_GRAY}'); }});

    setTimeout(function() {{
      if (fillEl) fillEl.setAttribute('width', fillEl.getAttribute('data-final-w'));
      markers.forEach(function(el) {{ el.style.opacity = '1'; }});
    }}, 150);

    var n = segs.length || 1;
    var stepMs = 1200 / n;
    segs.forEach(function(el, i) {{
      setTimeout(function() {{
        el.setAttribute('fill', el.getAttribute('data-final-fill'));
        if (i === segs.length - 1) scope.dataset.replaying = '0';
      }}, 550 + i * stepMs);
    }});
    if (!segs.length) setTimeout(function() {{ scope.dataset.replaying = '0'; }}, 700);
  }};
}})();
</script>'''
    else:
        body, n_hours = _standard_chart_block(ticket, delivery_date, fig_num)
        replay_script = f'''
<script>
(function() {{
  // The server-rendered SVG already shows every bar at its real final
  // height/position (see _bars_svg) -- nothing runs on load, so the page
  // is a static figure by default. Replay uses plain setTimeout to reveal
  // bars hour-by-hour, NOT CSS animation-delay -- a long CSS animation
  // reliably stalls once the browser tab loses focus mid-play (this
  // user's workflow constantly alternates between VS Code and the
  // browser), leaving later-hour bars permanently invisible. setTimeout
  // has no such failure mode: every callback still fires once the tab
  // regains focus, even if throttled while hidden.
  var replaying = false;
  function playAll() {{
    if (replaying) return;
    replaying = true;
    var bars = Array.prototype.slice.call(document.querySelectorAll('[data-kind="gt-bar"], [data-kind="gt-bar-dn"]'));
    var playhead = document.querySelector('[data-kind="gt-playhead"]');
    var n = bars.length || 1;
    var stepMs = 4200 / n;

    bars.forEach(function(el) {{
      var isDn = el.getAttribute('data-kind') === 'gt-bar-dn';
      el.setAttribute('height', '0');
      el.setAttribute('y', isDn ? el.getAttribute('data-final-y') : (parseFloat(el.getAttribute('data-final-y')) + parseFloat(el.getAttribute('data-final-h'))).toFixed(1));
    }});
    if (playhead) {{
      playhead.style.opacity = '1';
      playhead.setAttribute('x1', playhead.getAttribute('data-x0'));
      playhead.setAttribute('x2', playhead.getAttribute('data-x0'));
    }}

    bars.forEach(function(el, i) {{
      setTimeout(function() {{
        el.setAttribute('y', el.getAttribute('data-final-y'));
        el.setAttribute('height', el.getAttribute('data-final-h'));
        if (playhead) {{
          var frac = (i + 1) / n;
          var x0 = parseFloat(playhead.getAttribute('data-x0'));
          var x1 = parseFloat(playhead.getAttribute('data-x1'));
          var x = x0 + (x1 - x0) * frac;
          playhead.setAttribute('x1', x.toFixed(1));
          playhead.setAttribute('x2', x.toFixed(1));
        }}
        if (i === bars.length - 1) {{ replaying = false; }}
      }}, i * stepMs);
    }});
  }}
  var btn = document.getElementById('gt-replay-btn');
  if (btn) btn.addEventListener('click', playAll);
}})();
</script>'''

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <style>
    @keyframes gt-pulse {{ 0%,100% {{ opacity:1; transform:scale(1); }} 50% {{ opacity:0.4; transform:scale(0.7); }} }}
    @keyframes gt-in {{ from {{ opacity:0; transform:translateY(6px); }} to {{ opacity:1; transform:translateY(0); }} }}
    .gt-card {{ animation: gt-in 0.4s ease-out; background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
                border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box; }}
    .gt-live {{ animation: gt-pulse 1.4s ease-in-out infinite; }}
    .gt-replay {{ font-size:11px; padding:2px 10px; border:1px solid {theme.GRIDLINE}; border-radius:6px;
                  background:{theme.SURFACE}; cursor:pointer; }}
  </style>
  <div class="gt-card">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <div style="display:flex; align-items:center; gap:8px;">
        <span class="gt-live" style="width:8px; height:8px; border-radius:50%; background:{dot}; display:inline-block;"></span>
        {phase_bit if phase_bit else f'<span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Latest gate action</span>'}
      </div>
      <span style="background:{bg}; color:{fg}; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:500;">{decision}</span>
    </div>
    <div style="display:flex; align-items:baseline; gap:10px; margin-bottom:2px;">
      <span style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY};">{label}</span>
    </div>
    <p style="font-size:12px; color:{theme.INK_MUTED}; margin:0 0 4px;">{ref_bit}decided {ts}</p>
    {gate_close_bit}
    {revenue_bit}
    {body}
  </div>
</div>
{replay_script}
'''
