"""delivery_ticket.py - renders phase 5A/5B/5C delivery cards. Distinct
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

The "Market & Delivery" overview section this file's cards used to live
under (ISP dispatch, imbalance settlement, capacity vs activation) has
been removed entirely -- render_imbalance_settlement_card now renders
under "Optimization &
Physical Dispatch" instead (see pages/0_overview.py). render_brp_
explainer_card and render_agc_mechanism_card (the AGC merit-order demo)
were removed outright along with that section, not relocated.
"""
from __future__ import annotations

import html as _html
import json as _json

import theme
from dispatch_ticket import _HOVER_CROSSHAIR_SCRIPT, _hover_svg_elems, _hover_tooltip_div, _sparse_edge_path

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
    <p style="font-size:12.5px; color:{theme.INK_SECONDARY}; margin:0 0 4px;">Deviating from the day-ahead schedule settles at a different buy price and sell price, not the DA price - that asymmetry is the mechanism.</p>
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
        <text x="{x0}" y="14" font-size="12" font-weight="700" fill="{theme.INK_PRIMARY}">Imbalance (MW) for each quarter-hour, ISP 1-96</text>
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

        <text x="{x0}" y="{price_title_y:.1f}" font-size="12" font-weight="700" fill="{theme.INK_PRIMARY}">Reference price - illustrative (EUR/MWh) for each quarter-hour</text>
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
      <span style="font-size:11px; color:{theme.INK_PRIMARY}; font-weight:500;">Short price &gt; long price by design (see config: fallback_short_factor 1.20x DA, fallback_long_factor 0.85x DA) - that gap is what makes deviating from your schedule a net cost, not a coin flip.</span>
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
