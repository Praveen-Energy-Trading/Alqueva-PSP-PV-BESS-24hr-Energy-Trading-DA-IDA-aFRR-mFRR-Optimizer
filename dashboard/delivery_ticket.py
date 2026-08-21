"""delivery_ticket.py — renders phase 4A/4B/4C delivery cards. Distinct
from gate_ticket.py: these phases don't "decide" Submitted/Held, they
simulate delivery of whatever was already committed, so the card shape is
metrics + a trace, not a decision pill. See dashboard/data.py::load_rt_delivery
for the fields.

aFRR/mFRR activation settlement (revenue, MWh, ISPs activated) used to have
its own standalone card here, with a miniature ACE-vs-response chart that
duplicated dispatch_ticket.py::render_afrr_dispatch_card's fuller version in
smaller/less-detailed form. Removed -- that settlement data is now a stat
row merged directly into render_afrr_dispatch_card (data.py::
load_activation_summary's dict passed in as its optional `act` argument),
so there's one card per product instead of two overlapping ones.
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


def render_imbalance_settlement_card(im: dict) -> str:
    """Imbalance settlement mechanism explainer: why deviating from a
    committed DA schedule costs money, and why it costs more to buy back a
    shortfall than it earns to sell a surplus (dual pricing). Two panels
    sharing an x-axis: imbalance MW per ISP on top (same actual - scheduled
    trace as the 'ISP dispatch' widget, real, not recomputed differently),
    short/long reference price per ISP below (illustrative DA-multiplier
    fallback formula -- see data.py::load_imbalance_settlement for why this
    doesn't call REN's live price API from a page render). Net EUR and
    total MWh in the stat row ARE the real settled numbers from the audit
    trail, whatever price source that actual run used."""
    isps, hours = im["isps"], im["hours"]
    n = len(isps)
    imbalance_mw = im["imbalance_mw"]
    short_price, long_price = im["short_price"], im["long_price"]
    n_long = sum(1 for v in imbalance_mw if v > 1e-6)
    n_short = sum(1 for v in imbalance_mw if v < -1e-6)

    x0, x1 = 46, 1390

    def fx(i: int) -> float:
        return x0 + i / max(n - 1, 1) * (x1 - x0)

    imb_mid_y, imb_half = 40, 30
    price_title_y = imb_mid_y + imb_half + 22
    price_top, price_bot = price_title_y + 14, price_title_y + 74
    total_h = price_bot + 12
    imb_max = max((abs(v) for v in imbalance_mw), default=0.0) or 1.0
    price_max = max(max(short_price, default=0.0), max(long_price, default=0.0), 1.0) * 1.08

    def imb_y(v: float) -> float:
        return imb_mid_y - (v / imb_max) * imb_half

    def price_y(v: float) -> float:
        return price_bot - (v / price_max) * (price_bot - price_top)

    pos_pts = " L".join(f"{fx(i):.1f},{imb_y(max(v, 0.0)):.1f}" for i, v in enumerate(imbalance_mw))
    neg_pts = " L".join(f"{fx(i):.1f},{imb_y(min(v, 0.0)):.1f}" for i, v in enumerate(imbalance_mw))
    pos_path = f"M{fx(0):.1f},{imb_mid_y} L{pos_pts} L{fx(n-1):.1f},{imb_mid_y} Z"
    neg_path = f"M{fx(0):.1f},{imb_mid_y} L{neg_pts} L{fx(n-1):.1f},{imb_mid_y} Z"
    pos_edge = _sparse_edge_path(fx, imb_y, [max(v, 0.0) for v in imbalance_mw])
    neg_edge = _sparse_edge_path(fx, imb_y, [min(v, 0.0) for v in imbalance_mw])
    short_pts = " L".join(f"{fx(i):.1f},{price_y(v):.1f}" for i, v in enumerate(short_price))
    long_pts = " L".join(f"{fx(i):.1f},{price_y(v):.1f}" for i, v in enumerate(long_price))

    short_py = [price_y(v) for v in short_price]
    long_py = [price_y(v) for v in long_price]
    imb_py = [imb_y(v) for v in imbalance_mw]
    hover_cfg = _json.dumps({
        "n": n, "x0": x0, "x1": x1, "viewW": 1400,
        "catchSelector": "#imb-hover-catch", "lineSelector": "#imb-hover-line",
        "tooltipSelector": "#imb-tooltip", "indexLabel": "ISP", "indexVals": isps,
        "traces": [
            {"label": "Imbalance", "yvals": imbalance_mw, "py": imb_py, "dotSelector": "#imb-hover-dot-0",
             "color": theme.COLOR_UP, "negColor": theme.COLOR_DOWN, "unit": "MW", "decimals": 2, "signed": True},
            {"label": "Short price (buy back)", "yvals": short_price, "py": short_py, "dotSelector": "#imb-hover-dot-1",
             "color": theme.STATUS_CRITICAL, "unit": "EUR/MWh", "decimals": 1},
            {"label": "Long price (sell)", "yvals": long_price, "py": long_py, "dotSelector": "#imb-hover-dot-2",
             "color": theme.STATUS_GOOD, "unit": "EUR/MWh", "decimals": 1},
        ],
    })

    net_color = theme.STATUS_CRITICAL if (im["net_eur"] or 0) < 0 else theme.STATUS_GOOD
    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Mechanism</span>
      <span style="background:{theme.INK_MUTED}22; color:{theme.INK_SECONDARY}; font-size:11.5px; padding:3px 10px; border-radius:6px; font-weight:500;">Real settlement + illustrative reference price</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">Imbalance settlement</div>
    <p style="font-size:12.5px; color:{theme.INK_SECONDARY}; margin:0 0 4px;">Deviating from the day-ahead schedule settles at a different buy price and sell price, not the DA price &mdash; that asymmetry is the mechanism.</p>
    <p style="font-size:12px; color:{theme.INK_MUTED}; margin:0 0 10px;">decided {_html.escape(im['timestamp'])}</p>
    <div style="display:flex; gap:8px; margin-bottom:14px;">
      <div style="background:{theme.SURFACE}; border-radius:8px; padding:0.55rem 0.75rem; flex:1; border:1px solid {theme.GRIDLINE};">
        <p style="font-size:11px; color:{theme.INK_SECONDARY}; margin:0 0 2px;">Net imbalance settlement</p>
        <p style="font-size:17px; font-weight:500; margin:0; color:{net_color};">{'-' if (im['net_eur'] or 0) < 0 else ''}&euro;{abs(im['net_eur'] or 0):,.0f}</p>
      </div>
      <div style="background:{theme.SURFACE}; border-radius:8px; padding:0.55rem 0.75rem; flex:1; border:1px solid {theme.GRIDLINE};">
        <p style="font-size:11px; color:{theme.INK_SECONDARY}; margin:0 0 2px;">Total imbalance volume</p>
        <p style="font-size:17px; font-weight:500; margin:0; color:{theme.INK_PRIMARY};">{im['total_imbalance_mwh'] or 0:.1f} MWh</p>
      </div>
      <div style="background:{theme.SURFACE}; border-radius:8px; padding:0.55rem 0.75rem; flex:1; border:1px solid {theme.GRIDLINE};">
        <p style="font-size:11px; color:{theme.INK_SECONDARY}; margin:0 0 2px;">Long ISPs / short ISPs</p>
        <p style="font-size:17px; font-weight:500; margin:0; color:{theme.INK_PRIMARY};">{n_long} / {n_short}</p>
      </div>
    </div>
    <div class="imb-chart-block" style="position:relative; height:{total_h*0.92:.0f}px;">
      <svg viewBox="0 0 1400 {total_h}" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
        <text x="{x0}" y="14" font-size="12" font-weight="700" fill="{theme.INK_PRIMARY}">Imbalance (MW)</text>
        <line x1="{x0}" y1="{imb_mid_y - imb_half:.1f}" x2="{x1}" y2="{imb_mid_y - imb_half:.1f}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        <line x1="{x0}" y1="{imb_mid_y:.1f}" x2="{x1}" y2="{imb_mid_y:.1f}" stroke="{theme.INK_PRIMARY}" stroke-width="1.2"/>
        <line x1="{x0}" y1="{imb_mid_y + imb_half:.1f}" x2="{x1}" y2="{imb_mid_y + imb_half:.1f}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        <text x="{x0-6}" y="{imb_mid_y-imb_half+3:.1f}" font-size="11" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">+{imb_max:.1f}</text>
        <text x="{x0-6}" y="{imb_mid_y+4:.1f}" font-size="11" fill="{theme.INK_PRIMARY}" font-weight="700" text-anchor="end">0</text>
        <text x="{x0-6}" y="{imb_mid_y+imb_half+3:.1f}" font-size="11" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">-{imb_max:.1f}</text>
        <path d="{pos_path}" fill="{theme.COLOR_UP}" fill-opacity="0.4"/>
        <path d="{neg_path}" fill="{theme.COLOR_DOWN}" fill-opacity="0.4"/>
        <path d="{pos_edge}" fill="none" stroke="{theme.COLOR_UP}" stroke-width="1.8"/>
        <path d="{neg_edge}" fill="none" stroke="{theme.COLOR_DOWN}" stroke-width="1.8"/>

        <text x="{x0}" y="{price_title_y:.1f}" font-size="12" font-weight="700" fill="{theme.INK_PRIMARY}">Reference price &mdash; illustrative (EUR/MWh)</text>
        <line x1="{x0}" y1="{price_top:.1f}" x2="{x1}" y2="{price_top:.1f}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        <line x1="{x0}" y1="{price_bot:.1f}" x2="{x1}" y2="{price_bot:.1f}" stroke="{theme.GRIDLINE}" stroke-width="1"/>
        <text x="{x0-6}" y="{price_top+3:.1f}" font-size="11" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">{price_max:.0f}</text>
        <text x="{x0-6}" y="{price_bot+3:.1f}" font-size="11" fill="{theme.INK_PRIMARY}" font-weight="600" text-anchor="end">0</text>
        <path d="M{fx(0):.1f},{price_y(short_price[0] if short_price else 0):.1f} L{short_pts}" fill="none" stroke="{theme.STATUS_CRITICAL}" stroke-width="1.8"/>
        <path d="M{fx(0):.1f},{price_y(long_price[0] if long_price else 0):.1f} L{long_pts}" fill="none" stroke="{theme.STATUS_GOOD}" stroke-width="1.8"/>

        {_hover_svg_elems("imb", x0, x1, total_h, 3)}
      </svg>
      {_hover_tooltip_div().replace('dt-hover-tooltip"', 'dt-hover-tooltip" id="imb-tooltip"')}
    </div>
    <div style="display:flex; gap:14px; margin-top:8px;">
      <span style="font-size:11.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_UP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Long (over-delivered)</span>
      <span style="font-size:11.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_DOWN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Short (under-delivered)</span>
      <span style="font-size:11.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:2px; background:{theme.STATUS_CRITICAL}; margin-right:4px; vertical-align:middle;"></span>Short price (buy back)</span>
      <span style="font-size:11.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:14px; height:2px; background:{theme.STATUS_GOOD}; margin-right:4px; vertical-align:middle;"></span>Long price (sell)</span>
    </div>
    <div style="background:{theme.STATUS_WARNING}22; border-left:3px solid {theme.STATUS_WARNING}; padding:6px 10px; margin-top:10px;">
      <span style="font-size:11px; color:{theme.INK_PRIMARY}; font-weight:500;">Short price &gt; long price by design (see config: fallback_short_factor 1.20x DA, fallback_long_factor 0.85x DA) &mdash; that gap is what makes deviating from your schedule a net cost, not a coin flip.</span>
    </div>
    {_HOVER_CROSSHAIR_SCRIPT}
    <script>
    (function() {{
      var block = document.querySelector('.imb-chart-block');
      if (block) {{ dtHoverInit(block, {hover_cfg}); }}
    }})();
    </script>
  </div>
</div>'''


def render_capacity_vs_activation_card(cv: dict) -> str:
    """Two separate real payments per reserve product, side by side: paid
    for OFFERING the reserve (capacity, whether or not the TSO ever calls
    it) vs paid only for MW actually DELIVERED when called (activation /
    energy). See data.py::load_capacity_vs_activation for exactly which
    real audit fields feed each bar -- nothing here is illustrative or
    invented, both numbers are the same ones already shown elsewhere
    (gate ticket capacity revenue, activation card revenue), just placed
    next to each other so the split itself -- the concept a newcomer to
    reserve markets most needs -- is visible at a glance."""
    rows = cv["rows"]
    max_total = max((r["capacity_eur"] + r["activation_eur"] for r in rows), default=0.0) or 1.0

    bar_rows = []
    for r in rows:
        total = r["capacity_eur"] + r["activation_eur"]
        cap_pct = (r["capacity_eur"] / max_total) * 100
        act_pct = (r["activation_eur"] / max_total) * 100
        total_label = "&euro;0 (mandatory, unpaid)" if total < 1e-6 else f"&euro;{total:,.0f} total"
        bar_rows.append(f'''
        <div style="margin-bottom:12px;">
          <div style="display:flex; justify-content:space-between; font-size:12.5px; color:{theme.INK_SECONDARY}; margin-bottom:4px;">
            <span style="font-weight:500; color:{theme.INK_PRIMARY};">{_html.escape(r['product'])}</span>
            <span>{total_label}</span>
          </div>
          <div style="height:18px; background:{theme.GRIDLINE}; border-radius:4px; display:flex; overflow:hidden;">
            <div style="width:{cap_pct:.2f}%; height:100%; background:{theme.COLOR_GEN};" title="Capacity"></div>
            <div style="width:{act_pct:.2f}%; height:100%; background:{theme.COLOR_UP};" title="Activation"></div>
          </div>
        </div>''')

    return f'''
<div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div class="dt-card" style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE};
              border-radius:12px; padding:1rem 1.25rem; width:100%; box-sizing:border-box;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">
      <span style="font-size:13px; color:{theme.INK_SECONDARY}; font-weight:500;">Mechanism</span>
      <span style="background:{theme.STATUS_GOOD}22; color:{theme.STATUS_GOOD}; font-size:11.5px; padding:3px 10px; border-radius:6px; font-weight:500;">Real revenue, both halves</span>
    </div>
    <div style="font-size:20px; font-weight:500; color:{theme.INK_PRIMARY}; margin-bottom:2px;">Capacity vs activation revenue</div>
    <p style="font-size:12.5px; color:{theme.INK_SECONDARY}; margin:0 0 16px;">Two separate payments per reserve product: paid for being available, paid again only when actually called.</p>
    {''.join(bar_rows)}
    <div style="display:flex; gap:14px; margin-top:6px;">
      <span style="font-size:11.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_GEN}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Capacity (availability)</span>
      <span style="font-size:11.5px; color:{theme.INK_SECONDARY};"><span style="display:inline-block; width:9px; height:9px; background:{theme.COLOR_UP}; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Activation (energy)</span>
    </div>
  </div>
</div>'''


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
