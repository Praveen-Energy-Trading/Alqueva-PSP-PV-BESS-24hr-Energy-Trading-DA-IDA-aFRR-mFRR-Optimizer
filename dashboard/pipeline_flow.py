"""Static reference diagrams explaining the real MILP formulation in
common_layer/optimisation_model/core_milp_builder.py + core_milp_solver.py.

These are NOT data-driven - every equation, constant, and field name here is
copied verbatim from the two source files (72 equations total, none
omitted). Nothing is fetched from ComponentStore/AuditStore, so there is no
caching or refresh logic: this page explains the model's structure, not a
day's result.

Equations are rendered as real HTML/CSS typeset math (serif italic
variables, upright roman for word-subscripts, a flex-column stack for true
sub+superscript placement, centered within each card) rather than
hand-placed SVG text - this avoids the class of overlap/positioning bugs
SVG dy/baseline-shift hacks produced, since the browser's own flex layout
reserves correct width for every superscript automatically. This is the
finalized style, verified in a real browser before being wired in here.
Variable names follow the user's paper nomenclature; anything not defined
there (efficiency-surface polynomial coefficients, normalised flow/head
grid coordinates) is modelled in the same notational style.
"""
from __future__ import annotations

_CSS = """
<style>
.eqwrap { font-family: 'Times New Roman', Times, serif; display: flex; flex-direction: column; align-items: center; }
.eqwrap .sec-box { border: 2px solid; border-radius: 14px; padding: 14px 16px 16px 16px; background: #FFFFFF; margin: 0;
                    display: block; width: 100%; max-width: 900px; box-sizing: border-box; }
.eqwrap .sec-box.hug { display: table; width: auto; margin-left: auto; margin-right: auto; }
.eqwrap .sec-title { font-size: 14px; font-weight: 600; margin: 0 0 10px 0; text-align: center; }
.eqwrap .sec-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; justify-content: center; margin-bottom: 4px; }
.eqwrap .card.wide { flex: 1 0 100%; max-width: 820px; }
.eqwrap .card { border-radius: 10px; padding: 8px 10px; display: inline-flex; flex-direction: column; align-items: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.08); text-align: center; box-sizing: border-box; }
.eqwrap .hdr { font-size: 10px; font-weight: 600; margin-bottom: 6px; white-space: normal; text-align: center;
                display: flex; align-items: center; justify-content: center; gap: 6px; }
.eqwrap .num-chip { display: inline-block; color: #fff; border-radius: 10px; padding: 1px 8px;
                      font-family: system-ui, sans-serif; font-size: 10px; font-weight: 700; white-space: nowrap; }
.eqwrap .tag { font-weight: 700; margin-right: 6px; }
.eqwrap .math { font-family: 'Times New Roman', Times, serif; font-weight: 700;
                 font-style: normal; font-size: 12px; white-space: nowrap; display: flex; align-items: center;
                 justify-content: center; flex-wrap: nowrap; position: relative; z-index: 1; }
.eqwrap .rm { font-style: normal; }
.eqwrap .stack { display: inline-flex; align-items: center; }
.eqwrap .stack .col { display: inline-flex; flex-direction: column; align-items: flex-start; text-align: left;
                        font-size: 0.6em; line-height: 1.15; margin-left: 0.08em; }
.eqwrap .bigop { display: inline-flex; flex-direction: column; align-items: center; padding: 0 0.15em; }
.eqwrap .bigop .op { font-size: 1.4em; font-style: normal; line-height: 0.7; }
.eqwrap .bigop .lim { font-size: 0.55em; font-style: normal; line-height: 1; }
.eqwrap .meaning { font-size: 12px; font-style: normal; font-weight: 400; color: #000;
                    margin-top: 7px; padding-top: 5px; border-top: 1px solid rgba(0,0,0,0.08); white-space: normal;
                    max-width: 260px; margin-left: auto; margin-right: auto; position: relative; z-index: 1; }
.eqwrap .legend { font-size: 10px; color: #5F5E5A; margin-top: 8px;
                   display: flex; justify-content: center; flex-wrap: wrap; gap: 4px 18px; width: 100%; }
.eqwrap .legend div { margin: 0; white-space: nowrap; }
.eqwrap .dot { display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:5px; }
.eqwrap .flow-arrow { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px;
                        margin: 0; text-align: center; }
.eqwrap .flow-arrow .connector { width: 2px; height: 22px; background: #9B96C9; position: relative; }
.eqwrap .flow-arrow .connector::after { content: ''; position: absolute; bottom: -1px; left: 50%; transform: translateX(-50%);
                        border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 9px solid #9B96C9; }
.eqwrap .flow-arrow .note { font-size: 11px; font-style: italic; color: #5F5E5A; max-width: 500px; }
</style>
"""

_ARROW = '<div class="flow-arrow"><div class="connector"></div></div>'
_ARROW_MERGE = ('<div class="flow-arrow"><div class="connector"></div>'
                 '<span class="note">BESS (E) and PV (F) are built in parallel, both feed into the reservoir/net-position equations below</span></div>')

# ---------------------------------------------------------------------------
# Small helpers so 72 equations can be composed from parts instead of
# hand-typed HTML each time.
# ---------------------------------------------------------------------------

def sym(base: str, sup: str = "", sub: str = "", color: str = "") -> str:
    """One math symbol with a correctly-placed superscript and/or subscript.

    A symbol with only ONE of sup/sub uses a real native <sup>/<sub> tag  - 
    browsers position these correctly by default (raised or lowered from
    the baseline), which a from-scratch flex layout kept getting subtly
    wrong (e.g. a lone superscript ending up vertically centered instead of
    raised, reading as "on the next line" rather than a proper superscript).

    A symbol with BOTH sup and sub still uses the flex-column stack, which
    is needed there to reserve horizontal width for the wider of the two so
    they don't overlap the next symbol - that dual case is where the
    original overlap bug actually was.
    """
    style = f' style="color:{color};"' if color else ""
    if sup and sub:
        sup_html = f'<span class="rm"{style}>{sup}</span>'
        sub_html = f'<span class="rm">{sub}</span>'
        col = f'<span class="col">{sup_html}{sub_html}</span>'
        return f'<span class="stack">{base}{col}</span>'
    if sup:
        return f'<span{style}>{base}<sup class="rm">{sup}</sup></span>'
    if sub:
        return f'<span>{base}<sub class="rm">{sub}</sub></span>'
    return f'<span{style}>{base}</span>'


def bigop(op: str = "&Sigma;", lim: str = "") -> str:
    return f'<span class="bigop"><span class="op">{op}</span><span class="lim">{lim or "&nbsp;"}</span></span>'


_SECTION_COLORS = {
    "A": ("#534AB7", "#F8F7FE", "#26215C"),
    "B": ("#0F6E56", "#F3FAF7", "#04342C"),
    "C": ("#185FA5", "#F0F6FC", "#042C53"),
    "D": ("#3B6D11", "#F2F7EC", "#173404"),
    "E": ("#993556", "#FCF3F7", "#4B1528"),
    "F": ("#854F0B", "#FDF8EE", "#412402"),
    "G": ("#185FA5", "#F0F6FC", "#042C53"),
    "H": ("#3B6D11", "#F2F7EC", "#173404"),
    "I": ("#534AB7", "#F8F7FE", "#26215C"),
    "S": ("#5F5E5A", "#F5F4F2", "#2C2C2A"),
}


def card(num: str, title: str, math_html: str, meaning: str, section: str,
          legend: list[tuple[str, str]] | None = None, wide: bool = False) -> str:
    """wide=True marks this card to be placed in its own dedicated row by
    section() (not stretched to fill that row - it keeps its natural content
    width and stays centered) - used for the handful of outlier equations (a
    long objective sentence, or one with a legend) whose width/height would
    otherwise make a shared row with much shorter cards look ragged."""
    border, bg, text = _SECTION_COLORS[section]
    legend_html = ""
    if legend:
        rows = "".join(f'<div><span class="dot" style="background:{c};"></span>{t}</div>' for c, t in legend)
        legend_html = f'<div class="legend">{rows}</div>'
    cls = "card wide" if wide else "card"
    num_chip = f'<span class="num-chip" style="background:{border};">{num}</span>' if num else ""
    return (
        f'<div class="{cls}" style="background:{bg}; border:1.5px solid {border}; color:{text};">'
        f'<div class="hdr">{num_chip}<span>{title}</span></div>'
        f'<div class="math">{math_html}</div>'
        f'{legend_html}'
        f'<div class="meaning">{meaning}</div>'
        f'</div>'
    )


def section(title: str, section_key: str, cards_html: list[str]) -> str:
    """One flowchart section: a big bordered container (matching the section's
    color) holding its smaller equation boxes inside - this nested-box
    structure, plus the arrows between sections, is what makes this an actual
    flowchart rather than a plain grouped list.

    Sections with more than one equation fill the full available width so
    the row can pack 2-3 cards across instead of shrink-wrapping to just
    however many happened to fit - that shrink-wrap was leaving a lot of
    unused margin on both sides. A section with only one equation (F) still
    hugs its own content instead of stretching to a nearly-empty box.
    """
    border, _bg, text = _SECTION_COLORS[section_key]
    box_cls = "sec-box hug" if len(cards_html) == 1 else "sec-box"
    return (
        f'<div class="{box_cls}" style="border-color:{border};">'
        f'<div class="sec-title" style="color:{text};">{title}</div>'
        f'<div class="sec-row">{"".join(cards_html)}</div>'
        f'</div>'
    )


# Shorthand colors used repeatedly for asset-tagged superscripts.
_TRB, _PMP, _BESS, _PV = "#0F6E56", "#993556", "#993556", "#854F0B"
R, L = "&#8594;", "&#8592;"  # rightward / leftward flow arrows


def _build_equations_content() -> str:
    """The stitched section HTML only, no CSS/wrapper - so it can be embedded
    directly alongside other content in the same document (e.g. combined with
    the Input Data table so a single continuous connector can be drawn between
    them, which isn't possible across separate component iframes)."""
    S = []  # list of section HTML blocks, in true build order

    # ---- A. Efficiency surface & geometry (12 equations) ----
    S.append(section("A &middot; Efficiency surface &amp; geometry", "A", [
        card("1", "Turbine efficiency polynomial",
             f'{sym("&eta;")} = {sym("c", sub="0")}+{sym("c", sub="1")}{"q&#770;"}+{sym("c", sub="2")}{"H&#770;"}'
             f'+{sym("c", sub="3")}{"q&#770;"}{"H&#770;"}+{sym("c", sub="4")}{"q&#770;"}&sup2;+{sym("c", sub="5")}{"H&#770;"}&sup2;',
             "6 fitted coefficients (not in the paper's table - modelled the same way as any other calibration constant)", "A"),
        card("2", "Normalised flow",
             f'{"q&#770;"} = ({sym("q", sub="f")} &minus; {sym("q", sub="f,min")}) / ({sym("q", sub="f,max")} &minus; {sym("q", sub="f,min")})',
             "Maps the real flow grid point onto [0,1] for the polynomial above", "A"),
        card("3", "Normalised head",
             f'{"H&#770;"} = ({sym("H", sub="h")} &minus; {sym("H", sub="min")}) / ({sym("H", sub="max")} &minus; {sym("H", sub="min")})',
             "Same idea, for the head grid point", "A"),
        card("4", "Turbine power at a grid point",
             f'{sym("&eta;", sub="table", sup="(TRB,f,h)")} &middot; &rho; &middot; g &middot; {sym("q", sub="f")} &middot; {sym("H", sub="h")} / {sym("3.6&times;10", sup="9")}',
             "Real physics: power = efficiency &times; density &times; gravity &times; flow &times; head", "A"),
        card("5", "Pump power at a grid point",
             f'(1/{sym("&eta;", sub="table", sup="(PMP,f,h)")}) &middot; &rho; &middot; g &middot; {sym("q", sub="f")} &middot; {sym("H", sub="h")} / {sym("3.6&times;10", sup="9")}',
             "Same physics, inverse efficiency since pumping consumes power", "A"),
        card("6", "Pump minimum power",
             f'{sym("P", sup="PMP", sub="min")} = {sym("P", sup="PMP", sub="rated")} &middot; ({sym("q", sup="PMP", sub="min")} / {sym("q", sup="PMP", sub="max")})',
             "Scales the rated power down by the pump's real minimum flow ratio", "A"),
        card("7", "Reservoir floor volume",
             f'{sym("Q", sup="res", sub="min")}',
             "The real minimum upper-reservoir volume, m&sup3; - the reference point head is measured from", "A"),
        card("8", "Reservoir usable range",
             f'&Delta;{sym("Q", sup="res")} = {sym("Q", sup="res", sub="max")} &minus; {sym("Q", sup="res", sub="min")}',
             "The real usable volume band the plant can operate within", "A"),
        card("9", "Head-volume gradient",
             f'dH/dQ = ({sym("H", sub="max")} &minus; {sym("H", sub="min")}) / &Delta;{sym("Q", sup="res")}',
             "Real bathymetric slope - how much head changes per m&sup3; of reservoir volume", "A"),
        card("10", "MWh-per-volume constant",
             f'&kappa; = {sym("P", sup="TRB", sub="rated")} / {sym("q", sup="TRB", sub="max")}',
             "Converts a m&sup3; of water moved into MWh, used later in the water-value term", "A"),
        card("11", "Turbine headroom cap",
             f'{sym("P", sup="TRB,cap")} = {sym("P", sup="TRB", sub="rated")} &minus; FRR headroom',
             "Rated capacity minus whatever FCR/FRR reserve is mandatorily withheld", "A"),
        card("12", "Pump headroom cap",
             f'{sym("P", sup="PMP,cap")} = {sym("P", sup="PMP", sub="rated")} &minus; FRR headroom',
             "Same, for the pump direction", "A"),
    ]))

    # ---- B. PSP unit commitment (12 equations) ----
    S.append(section("B &middot; PSP unit commitment", "B", [
        card("13", "Mode exclusivity",
             f'{sym("u", sup="TRB", sub="t")} + {sym("u", sup="PMP", sub="t")} &le; 1',
             "A unit can turbine or pump, never both, in the same ISP", "B"),
        card("14", "Turbine power upper bound",
             f'{sym("P", sup=f"TRB{R}DA", sub="t")} &le; {sym("P", sup="TRB", sub="rated")} &middot; {sym("u", sup="TRB", sub="t")}',
             "Power is zero unless the unit is committed on", "B"),
        card("15", "Turbine power lower bound",
             f'{sym("P", sup=f"TRB{R}DA", sub="t")} &ge; {sym("q", sup="TRB", sub="min")} &middot; {sym("u", sup="TRB", sub="t")}',
             "Once committed, a real minimum stable output applies", "B"),
        card("16", "Pump power upper bound",
             f'{sym("P", sup=f"PMP{L}DA", sub="t")} &le; {sym("P", sup="PMP", sub="rated")} &middot; {sym("u", sup="PMP", sub="t")}',
             "Same pattern, pumping direction", "B"),
        card("17", "Pump power lower bound",
             f'{sym("P", sup=f"PMP{L}DA", sub="t")} &ge; {sym("P", sup="PMP", sub="min")} &middot; {sym("u", sup="PMP", sub="t")}',
             "Real minimum pump draw once committed", "B"),
        card("18", "Turbine startup flag (t=first)",
             f'{sym("y", sup="TRB,start", sub="t")} &ge; {sym("u", sup="TRB", sub="t")}',
             "The first ISP of the horizon: coming on at all counts as a start", "B"),
        card("19", "Turbine startup flag",
             f'{sym("y", sup="TRB,start", sub="t")} &ge; {sym("u", sup="TRB", sub="t")} &minus; {sym("u", sup="TRB", sub="t-1")}',
             "Flags a real transition from off to on", "B"),
        card("20", "Pump startup flag (t=first)",
             f'{sym("y", sup="PMP,start", sub="t")} &ge; {sym("u", sup="PMP", sub="t")}',
             "Same idea, pumping", "B"),
        card("21", "Pump startup flag",
             f'{sym("y", sup="PMP,start", sub="t")} &ge; {sym("u", sup="PMP", sub="t")} &minus; {sym("u", sup="PMP", sub="t-1")}',
             "Same idea, pumping", "B"),
        card("22", "Minimum dwell window",
             'W = max(1, round(min_mode_hours / &Delta;t))',
             "Real minimum-run-time converted into a number of ISPs", "B"),
        card("23", "Turbine minimum dwell",
             f'{bigop(lim="&tau;&isin;window")}{sym("u", sup="TRB", sub="&tau;")} &ge; W &middot; {sym("y", sup="TRB,start", sub="t")}',
             "Once started, a turbine must stay on for the real minimum-mode duration", "B"),
        card("24", "Pump minimum dwell",
             f'{bigop(lim="&tau;&isin;window")}{sym("u", sup="PMP", sub="&tau;")} &ge; W &middot; {sym("y", sup="PMP,start", sub="t")}',
             "Same rule, pumping", "B"),
    ]))

    # ---- C. Head-volume geometry & McCormick linearisation (9 equations) ----
    S.append(section("C &middot; Head-volume geometry &amp; McCormick linearisation", "C", [
        card("25", "Net head from volume",
             f'{sym("H", sup="net", sub="t")} = {sym("H", sub="min")} + dH/dQ &middot; ({sym("Q", sup="res,up", sub="t")} &minus; {sym("Q", sup="res", sub="min")})',
             "Real hydraulic head is a linear function of how full the upper reservoir is", "C"),
        card("26/27", "Turbine active-head bounds",
             f'{sym("H&#770;", sup="TRB")} &isin; [{sym("H", sub="min", sup="safe")}, {sym("H", sub="max", sup="safe")}] &middot; {sym("u", sup="TRB", sub="t")}',
             "McCormick envelope keeping the linearised active head within the real safe operating range", "C"),
        card("28/29", "Turbine active-head linking",
             f'{sym("H&#770;", sup="TRB")} &asymp; {sym("H", sup="net", sub="t")} when {sym("u", sup="TRB", sub="t")}=1',
             "Ties the active-head variable to the real net head only while the unit runs", "C"),
        card("30/31/32/33", "Pump active-head (same pattern)",
             f'{sym("H&#770;", sup="PMP")} - identical structure to 26&ndash;29, for {sym("u", sup="PMP", sub="t")}',
             "The same 4-constraint McCormick envelope, mirrored for pumping mode", "C"),
    ]))

    # ---- D. Omega interpolation (8 equations) ----
    S.append(section("D &middot; Omega - bilinear efficiency-surface interpolation", "D", [
        card("34/35", "Interpolation weights sum to commitment",
             f'{bigop()}{sym("&omega;", sup="TRB")} = {sym("u", sup="TRB", sub="t")} &nbsp;&nbsp; (same for PMP)',
             "The grid weights for an off unit are forced to zero, on units sum to exactly 1", "D"),
        card("36/37", "Power from the interpolated grid",
             f'{sym("P", sup=f"TRB{R}DA", sub="t")} = {bigop()}{sym("&omega;", sup="TRB")} &middot; power({sym("q", sub="f")},{sym("H", sub="h")})',
             "Real dispatched power is a weighted blend of the pre-computed grid power values (same for pumping)", "D"),
        card("38/39", "Flow from the interpolated grid",
             f'{sym("q", sup="TRB", sub="t")} = {bigop()}{sym("&omega;", sup="TRB")} &middot; {sym("q", sub="f")}',
             "The actual flow is the same weighted blend, over the flow axis (same for pumping)", "D"),
        card("40/41", "Head consistency",
             f'{bigop()}{sym("&omega;", sup="TRB")} &middot; {sym("H", sub="h")} = {sym("H&#770;", sup="TRB")}',
             "Forces the interpolation weights to also reproduce the real active head (same for pumping)", "D"),
    ]))

    # ---- E. BESS (6 equations) ----
    S.append(section("E &middot; BESS", "E", [
        card("42", "Charge/discharge exclusivity",
             f'{sym("u", sup="BESS,ch", sub="t")} + {sym("u", sup="BESS,dis", sub="t")} &le; 1',
             "The battery can charge or discharge, never both, in one ISP", "E"),
        card("43", "Charge power cap",
             f'{sym("P", sup=f"BESS{L}DA", sub="t")} + {sym("P", sup=f"PV{R}BESS", sub="t")} &le; {sym("P", sup="BESS", sub="rated")} &middot; {sym("u", sup="BESS,ch", sub="t")}',
             "Grid-charging and PV-to-battery charging share the same rated power limit", "E"),
        card("44", "Discharge power cap",
             f'{sym("P", sup=f"BESS{R}DA", sub="t")} &le; {sym("P", sup="BESS", sub="rated")} &middot; {sym("u", sup="BESS,dis", sub="t")}',
             "Real inverter power limit while discharging", "E"),
        card("45", "PV-to-BESS flow cap",
             f'{sym("P", sup=f"PV{R}BESS", sub="t")} &le; {sym("P", sup="BESS", sub="rated")}',
             "PV charging the battery still can't exceed the battery's own rated power", "E"),
        card("46", "State-of-charge balance",
             f'{sym("E", sup="BESS", sub="t")} = {sym("E", sup="BESS", sub="t-1")} + {sym("&eta;", sub="ch")}&middot;chg&middot;&Delta;t &minus; dis&middot;&Delta;t/{sym("&eta;", sub="dis")}',
             "Battery level next ISP = level now + charge in &minus; discharge out, each scaled by round-trip loss", "E"),
        card("47/48", "SOC bounds",
             f'{sym("E", sup="BESS", sub="min")} &le; {sym("E", sup="BESS", sub="t")} &le; {sym("E", sup="BESS", sub="max")}',
             "The real battery can never go below its floor or above its cap", "E"),
    ]))

    # ---- F. PV balance (1 equation) ----
    S.append(section("F &middot; PV balance", "F", [
        card("49", "PV power split",
             f'{sym("P", sup=f"PV{R}DA", sub="t")} + {sym("P", sup=f"PV{R}BESS", sub="t")} + {sym("P", sup="PV,curt", sub="t")} = {sym("P", sup="PV,avail", sub="t")}',
             "Every available PV MW is sold, routed to the battery, or curtailed - a real solver decision, not a fixed rule", "F"),
    ]))

    # ---- G. Reservoir constraints (7 equations) ----
    S.append(section("G &middot; Reservoir constraints", "G", [
        card("50", "Upper reservoir balance",
             f'{sym("Q", sup="res,up", sub="t")} = {sym("Q", sup="res,up", sub="t-1")} + ({sym("q", sup="nat")} + {sym("q", sup="PMP")} &minus; {sym("q", sup="TRB")} &minus; {sym("q", sup="spill")}) &middot; &Delta;t',
             "Real water-volume bookkeeping: inflow and pumping add, turbining and spill remove", "G"),
        card("51", "Lower reservoir balance",
             f'{sym("Q", sup="res,low", sub="t")} = {sym("Q", sup="res,low", sub="t-1")} + ({sym("q", sup="TRB")} &minus; {sym("q", sup="PMP")}) &middot; &Delta;t',
             "The tailrace reservoir mirrors the upper one, turbining fills it, pumping drains it", "G"),
        card("52/53", "Upper reservoir bounds",
             f'{sym("Q", sup="res", sub="min")} &le; {sym("Q", sup="res,up", sub="t")} &le; {sym("Q", sup="res", sub="max")}',
             "The real physical volume band of the upper reservoir", "G"),
        card("54/55", "Lower reservoir bounds",
             f'{sym("Q", sup="res,low", sub="min")} &le; {sym("Q", sup="res,low", sub="t")} &le; {sym("Q", sup="res,low", sub="max")}',
             "Same, for the lower reservoir", "G"),
        card("56", "Terminal reservoir constraint",
             f'{sym("Q", sup="res,up", sub="T-end")} &ge; {sym("Q", sup="res", sub="init")}',
             "The plan can't end the horizon by draining the reservoir below where it started", "G",
             wide=True),
    ]))

    # ---- H. Net position / energy balance (4 equations) ----
    S.append(section("H &middot; Net position balance", "H", [
        card("57", "Net position identity",
             f'{bigop()}{sym("P", sup=f"{R}DA", sub="t")} = {bigop()}({sym("P", sup=f"TRB{R}DA", sub="t")} &minus; {sym("P", sup=f"PMP{L}DA", sub="t")}) '
             f'+ {sym("P", sup=f"PV{R}DA", sub="t")} + {sym("P", sup=f"BESS{R}DA", sub="t")} &minus; {sym("P", sup=f"BESS{L}DA", sub="t")}',
             "Every MW the plant puts on or takes off the grid, added up - holds for every single ISP", "H",
             legend=[(_TRB, "hydro turbine &minus; pump"), (_PV, "PV sold to market"), (_BESS, "BESS discharge &minus; charge")],
             wide=True),
        card("58", "Generation headroom",
             f'{bigop()}{sym("P", sup=f"{R}DA", sub="t")} &le; {sym("P", sup="TRB,cap")} &minus; {bigop()}{sym("P", sup=f"{R}uFRR")}',
             "Net export can't exceed generation capacity minus whatever aFRR/mFRR up-reserve is already awarded", "H"),
        card("59", "Pumping headroom",
             f'{bigop()}{sym("P", sup=f"{R}DA", sub="t")} &ge; &minus;({sym("P", sup="PMP,cap")} &minus; {bigop()}{sym("P", sup=f"{R}dFRR")})',
             "Same idea in the pumping direction, against down-reserve headroom", "H"),
        card("60", "IDA fixed-net lock",
             f'{bigop()}{sym("P", sup=f"{R}DA", sub="t")} = {sym("P", sup="fixed")} &nbsp; <span class="rm" style="font-size:0.6em;">(IDA gates only)</span>',
             "ISPs already committed at an earlier gate are locked; only the remaining ISPs stay free variables", "H",
             wide=True),
    ]))

    # ---- I. Objective (7 equations) ----
    S.append(section("I &middot; Objective - maximise revenue", "I", [
        card("61", "Energy revenue",
             f'{bigop(lim="t")}{sym("&lambda;", sup="DA", sub="t")} &middot; {sym("P", sup=f"{R}DA", sub="t")} &middot; &Delta;t',
             "Price &times; power sold, summed over every ISP", "I"),
        card("62", "Water value",
             f'{sym("c", sub="water")} &middot; &kappa; &middot; ({sym("Q", sup="res,up", sub="T-end")} &minus; {sym("Q", sup="res", sub="init")})',
             "Credits (or penalises) ending the horizon with more (or less) stored water than at the start", "I"),
        card("63", "PV curtailment penalty",
             f'{sym("c", sub="curt")} &middot; {bigop()}{sym("P", sup="PV,curt", sub="t")} &middot; &Delta;t',
             "Discourages wasting available solar power", "I"),
        card("64", "BESS degradation cost",
             f'{sym("c", sub="BESS")} &middot; {bigop()}(chg + {sym("P", sup=f"PV{R}BESS")} + dis) &middot; &Delta;t',
             "A real &euro;/MWh throughput cost applied to every MWh cycled through the battery", "I"),
        card("65", "Spillage penalty",
             f'{sym("c", sub="spill")} &middot; {bigop()}{sym("q", sup="spill")} &middot; &Delta;t',
             "Penalises wasting water that could have generated revenue", "I"),
        card("66", "Startup cost",
             f'{sym("c", sub="TRB")} &middot; {bigop()}{sym("y", sup="TRB,start")} + {sym("c", sub="PMP")} &middot; {bigop()}{sym("y", sup="PMP,start")}',
             "A real fixed cost charged every time a unit starts", "I"),
        card("67", "Objective",
             'Z = max( revenue + water value &minus; curtailment &minus; degradation &minus; spillage &minus; startup )',
             "Solved once, jointly, across the whole horizon - every other equation on this page feeds into this single number", "I",
             wide=True),
    ]))

    # ---- Solver: extraction (6 equations) ----
    S.append(section("J &middot; Solver extraction (core_milp_solver.py)", "S", [
        card("68/69", "Accumulated energy revenue",
             f'energy_rev += {sym("&lambda;", sup="DA", sub="t")} &middot; {sym("P", sup=f"{R}DA", sub="t")} &middot; &Delta;t',
             "The same equation as #61, accumulated ISP by ISP as the solved schedule is read back out", "S"),
        card("70", "Total charge power",
             f'total_charge = {sym("P", sup=f"BESS{L}DA")} + {sym("P", sup=f"PV{R}BESS")}',
             "Grid-charging and PV-charging combined, for reporting", "S"),
        card("71", "Power-weighted realised efficiency",
             f'{sym("&eta;", sub="pw")} = {bigop()}&omega;&middot;&eta;&middot;qH / {bigop()}&omega;&middot;qH',
             "The actual efficiency the unit achieved, weighted by how much flow/head it really ran at (0 if no flow)", "S"),
        card("72", "Binding constraint check",
             'binding = |slack| &lt; 1e&minus;6',
             "Flags which real constraints were actually active at the solved optimum, for diagnostics", "S"),
    ]))

    # Stitch sections together as a real flowchart: an arrow between each
    # consecutive section in true build order, with a note at the one real
    # fork/merge point (BESS and PV are independent of each other, both feed
    # into the reservoir/net-position equations that follow).
    stitched = [S[0]]
    for i in range(1, len(S)):
        stitched.append(_ARROW_MERGE if i == 6 else _ARROW)  # index 6 = "G" section, after E(4) and F(5)
        stitched.append(S[i])

    return "".join(stitched)


def build_equations_html() -> str:
    return _CSS + '<div class="eqwrap">' + _build_equations_content() + "</div>"


_INPUTS = """
<div style="font-family:system-ui,-apple-system,'Segoe UI',sans-serif; overflow-x:auto; background:transparent;">
<svg viewBox="0 0 900 700" style="width:100%; min-width:860px; height:auto; display:block;">
<defs>
<marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#888780"/>
</marker>
</defs>
<style>.f8{font-size:8px; font-family:monospace; fill:#2C2C2A;} .sec{font-size:12px; font-weight:500;} .tgt{font-size:8px; font-family:monospace; font-weight:700;}</style>
<text x="450" y="18" font-size="12" fill="#0b0b0b" text-anchor="middle" font-weight="500">Where every input value comes from - feeding the 72-equation model</text>

<rect x="20" y="30" width="270" height="220" rx="8" fill="#F3FAF7" stroke="#0F6E56"/>
<text x="32" y="48" class="sec" fill="#04342C">PSP unit config (static)</text>
<text x="32" y="62" font-size="7.5" fill="#555">psp_plant.yaml / PlantConfig</text>
<g class="f8" fill="#04342C">
<text x="32" y="80">P_TRB,rated / P_PMP,rated</text>
<text x="32" y="94">q_TRB,min/max, q_PMP,min/max</text>
<text x="32" y="108">c_TRB, c_PMP (commitment cost)</text>
<text x="32" y="122">min_mode_hours (dwell W)</text>
<text x="32" y="136">n_units_turb / n_units_pump</text>
<text x="32" y="150">FRR mandatory headroom</text>
</g>
<text x="32" y="228" class="tgt" fill="#0F6E56">&#8594; sections A, B, H</text>

<rect x="310" y="30" width="270" height="220" rx="8" fill="#F0F6FC" stroke="#185FA5"/>
<text x="322" y="48" class="sec" fill="#042C53">Reservoir geometry (static)</text>
<text x="322" y="62" font-size="7.5" fill="#555">reservoir_config.yaml</text>
<g class="f8" fill="#042C53">
<text x="322" y="80">Q_res,min, Q_res,max, Q_res,init</text>
<text x="322" y="94">H_min=54.7 m, H_max=73.0 m</text>
<text x="322" y="108">H_min,safe=52.1, H_max,safe=74.0 m</text>
<text x="322" y="122">c0&hellip;c5 (efficiency-surface coeffs)</text>
<text x="322" y="136">||F||=5, ||H||=5 (grid segments)</text>
<text x="322" y="150">Q_t-1,res,up (carried from prior ISP)</text>
</g>
<text x="322" y="228" class="tgt" fill="#185FA5">&#8594; sections A, C, G, I</text>

<rect x="600" y="30" width="280" height="220" rx="8" fill="#FCF3F7" stroke="#993556"/>
<text x="612" y="48" class="sec" fill="#4B1528">BESS config (static)</text>
<text x="612" y="62" font-size="7.5" fill="#555">bess_config.yaml</text>
<g class="f8" fill="#4B1528">
<text x="612" y="80">P_BESS,rated</text>
<text x="612" y="94">E_BESS,min / E_BESS,max</text>
<text x="612" y="108">eta_ch / eta_dis</text>
<text x="612" y="122">c_BESS (degradation cost)</text>
<text x="612" y="136">E_t-1,BESS (carried from previous ISP)</text>
</g>
<text x="612" y="170" class="tgt" fill="#993556">&#8594; section E</text>

<rect x="600" y="264" width="280" height="140" rx="8" fill="#F8F7FE" stroke="#534AB7"/>
<text x="612" y="282" class="sec" fill="#26215C">Economic params (static)</text>
<text x="612" y="296" font-size="7.5" fill="#555">economics_config.yaml</text>
<g class="f8" fill="#26215C">
<text x="612" y="314">c_water (water value)</text>
<text x="612" y="328">c_curt (PV curtailment penalty)</text>
<text x="612" y="342">c_spill (spillage penalty)</text>
<text x="612" y="356">&Delta;t (ISP length, 0.25 h)</text>
</g>
<text x="612" y="378" class="tgt" fill="#534AB7">&#8594; section I</text>

<rect x="20" y="264" width="270" height="290" rx="8" fill="#FDF8EE" stroke="#854F0B"/>
<text x="32" y="282" class="sec" fill="#412402">Forecasts (per-ISP, phase 1 output)</text>
<text x="32" y="296" font-size="7.5" fill="#555">da_price_pv_inflow_forecasting/</text>
<g class="f8" fill="#412402">
<text x="32" y="314">lambda_t,DA - day-ahead price, 96 ISPs</text>
<text x="32" y="344">P_t,PV,avail - PV availability forecast</text>
<text x="32" y="374">q_t,nat - natural reservoir inflow forecast</text>
<text x="32" y="404">P_uFRR / P_dFRR - aFRR/mFRR awards</text>
<text x="32" y="434">P_fixed - IDA-only, locked by prior gate</text>
</g>
<text x="32" y="480" class="tgt" fill="#854F0B">&#8594; sections A, F, G, H, I</text>

<rect x="310" y="264" width="270" height="140" rx="8" fill="#F2F7EC" stroke="#3B6D11"/>
<text x="322" y="282" class="sec" fill="#173404">Solved unknowns (decision vars)</text>
<text x="322" y="296" font-size="7.5" fill="#555">created empty, filled by solver</text>
<g class="f8" fill="#173404">
<text x="322" y="314">u_TRB/u_PMP, P_TRB-&gt;DA/P_PMP&lt;-DA,</text>
<text x="322" y="326">omega_TRB/omega_PMP, Q_res,up/Q_res,low,</text>
<text x="322" y="338">E_BESS, P_BESS&lt;-DA/P_BESS-&gt;DA,</text>
<text x="322" y="350">P_PV-&gt;DA/P_PV,curt, q_spill, y_start</text>
</g>
<text x="322" y="378" class="tgt" fill="#3B6D11">&#8594; sections B-I (not inputs - solver outputs)</text>

<line x1="450" y1="562" x2="450" y2="592" stroke="#888780" stroke-width="1.5" marker-end="url(#ar2)"/>

<rect x="20" y="594" width="860" height="70" rx="8" fill="#F5F4F2" stroke="#5F5E5A"/>
<text x="34" y="612" class="sec" fill="#2C2C2A">Real source at run time</text>
<text x="34" y="632" class="f8" fill="#2C2C2A">All static config loaded once per run. Forecasts &amp; carried state read fresh from ComponentStore</text>
<text x="34" y="646" class="f8" fill="#2C2C2A">at each gate (DA / IDA1-3 / XBID). No number here is fabricated - every field above is read directly from a real config file, a real trained forecast model's output, or the real prior-ISP state.</text>

</svg>
</div>
"""


def equation_blocks_html() -> str:
    """All 72 equations, real typeset HTML/CSS math (centered, no per-term
    callouts - the finalized style), grouped by section in build order."""
    return build_equations_html()


_TABLE_CSS = """
<style>
.eqwrap .gate-table { border-collapse: separate; border-spacing: 0 10px; width: 100%; max-width: 900px; }
.eqwrap .gate-table td { padding: 12px 16px; vertical-align: middle; }
.eqwrap .gate-table tr { box-shadow: 0 2px 5px rgba(0,0,0,0.08); }
.eqwrap .gate-table .gate-name { border-radius: 10px 0 0 10px; font-size: 13px; font-weight: 700;
                                   white-space: nowrap; width: 90px; }
.eqwrap .gate-table .gate-feeds { border-radius: 0 10px 10px 0; text-align: left; }
.eqwrap .gate-table .feed-line { display: flex; align-items: baseline; gap: 8px; }
.eqwrap .gate-table .feed-line + .feed-line { margin-top: 6px; }
.eqwrap .gate-table .feed-label { font-size: 13px; font-weight: 700; white-space: nowrap; }
.eqwrap .gate-table .feed-leader { flex: 1 1 auto; min-width: 12px; border-bottom: 1px dotted #C7C4BC;
                                     margin-bottom: 3px; }
.eqwrap .gate-table .feed-source { font-size: 10.5px; font-weight: 400; color: #5F5E5A;
                                     flex: 0 1 auto; min-width: 0; }
.eqwrap .flow-header { display: flex; align-items: center; justify-content: flex-end; margin-bottom: 10px; }
.eqwrap .flow-label { font-size: 13px; color: #5F5E5A; }
.eqwrap .flow-replay { display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 700;
                         padding: 6px 14px; border-radius: 20px; border: 1px solid #C7C4BC; background: #fff;
                         color: #2C2C2A; cursor: pointer; }
.eqwrap .flow-replay:hover { background: #F5F4F2; }
.eqwrap .flow-wrap { position: relative; padding-right: 70px; box-sizing: border-box; }
.eqwrap #flowSvg { position: absolute; top: 0; left: 0; overflow: visible; pointer-events: none; }
.eqwrap #plantModelHeading { font-size: 1.5rem; font-weight: 700; color: #000000; margin: 2.5rem 0 0.25rem; }
</style>
"""


def _feed_line(label: str, source: str) -> str:
    """One input, paired on a single line with the real file/source behind it -
    so the two never drift apart the way independent parallel <br> stacks can."""
    return (f'<div class="feed-line"><span class="feed-label">{label}</span>'
            f'<span class="feed-leader"></span><span class="feed-source">{source}</span></div>')


def _gate_row(gate: str, feed_lines: list[str], section_key: str) -> str:
    border, bg, text = _SECTION_COLORS[section_key]
    return (
        f'<tr style="background:{bg};">'
        f'<td class="gate-name" style="background:{border}; color:#fff;">{gate}</td>'
        f'<td class="gate-feeds" style="color:{text};">{"".join(feed_lines)}</td>'
        f'</tr>'
    )


_GATE_IDS = ["DA", "IDA1", "IDA2", "IDA3", "XBID"]
_GATE_SECTION_KEYS = ["F", "C", "C", "C", "A"]


def _build_gate_rows() -> str:
    """Every real input source, one row per gate, in build/run order."""
    rows = [
        _gate_row("DA", [
            _feed_line("Day-ahead price (&euro;/MWh)",
                       "historical OMIE/MIBEL day-ahead prices, 2020-2026"),
            _feed_line("Global solar radiation (W/m&sup2;) &amp; ambient temperature (&deg;C)",
                       "real on-site sensor data"),
            _feed_line("River inflow (m&sup3;/h)",
                       "real Guadiana river daily mean inflow"),
        ], "F"),
        _gate_row("IDA1", [
            _feed_line("IDA1 auction price (&euro;/MWh)",
                       "historical IDA1 intraday auction prices, 2024-2025"),
            _feed_line("Global solar radiation (W/m&sup2;) &amp; ambient temperature (&deg;C)",
                       "real on-site sensor data, refreshed for the IDA1 run"),
            _feed_line("River inflow (m&sup3;/h)",
                       "real Guadiana river daily mean inflow, refreshed for the IDA1 run"),
        ], "C"),
        _gate_row("IDA2", [
            _feed_line("IDA2 auction price (&euro;/MWh)",
                       "historical IDA2 intraday auction prices, 2024-2025"),
            _feed_line("Global solar radiation (W/m&sup2;) &amp; ambient temperature (&deg;C)",
                       "real on-site sensor data, refreshed for the IDA2 run"),
            _feed_line("River inflow (m&sup3;/h)",
                       "real Guadiana river daily mean inflow, refreshed for the IDA2 run"),
        ], "C"),
        _gate_row("IDA3", [
            _feed_line("IDA3 auction price (&euro;/MWh)",
                       "historical IDA3 intraday auction prices, 2024-2025"),
            _feed_line("Global solar radiation (W/m&sup2;) &amp; ambient temperature (&deg;C)",
                       "real on-site sensor data, refreshed for the IDA3 run"),
            _feed_line("River inflow (m&sup3;/h)",
                       "real Guadiana river daily mean inflow, refreshed for the IDA3 run"),
        ], "C"),
        _gate_row("XBID", [
            _feed_line("Continuous intraday price (&euro;/MWh)",
                       "historical XBID continuous intraday prices, 2024-2025"),
            _feed_line("Global solar radiation (W/m&sup2;) &amp; ambient temperature (&deg;C)",
                       "real on-site sensor data, refreshed for the XBID run"),
            _feed_line("River inflow (m&sup3;/h)",
                       "real Guadiana river daily mean inflow, refreshed for the XBID run"),
        ], "A"),
    ]
    return f'<table class="gate-table" id="gateTable">{"".join(rows)}</table>'


def build_inputs_html() -> str:
    """Every real input source, one row per gate, in build/run order - so a
    later step can draw a connector from a specific gate's row straight to
    the equations it feeds, instead of one flat undifferentiated card grid."""
    gate_colors = [_SECTION_COLORS[k][0] for k in _GATE_SECTION_KEYS]
    return _flow_widget_html(_build_gate_rows(), _GATE_IDS, gate_colors)


def _flow_widget_html(table: str, gate_ids: list[str], gate_colors: list[str], extra_content: str = "") -> str:
    """The gate table + play/replay button + connector, as one self-contained
    block. When extra_content is given (the plant-model heading + equations),
    it's appended inside the SAME positioned wrapper so the connector overlay
    can measure real coordinates and draw one continuous line all the way from
    the last gate row into the heading below - not possible across separate
    component iframes, which is why this exists as a single combined block
    rather than two independent ones."""
    body = f"""
<div class="flow-wrap" id="flowWrap">
  <div class="flow-header">
    <button class="flow-replay" id="flowReplayBtn"><span id="flowReplayIcon">&#9654;</span><span id="flowReplayText">Play</span></button>
  </div>
  <div class="flow-table-col">{table}</div>
  <svg id="flowSvg">
    <defs>
      <marker id="flowArrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
        <path d="M0,0 L8,4.5 L0,9 Z" fill="#9B96C9" id="flowArrowHead" />
      </marker>
    </defs>
    <path id="flowSpine" fill="none" stroke="#9B96C9" stroke-width="2.5" stroke-linecap="round" marker-end="url(#flowArrow)" />
    <path id="flowPulsePath" fill="none" stroke="none" />
    <circle id="flowPulse" r="4.5" opacity="0" />
  </svg>
  {extra_content}
</div>
<script>
(function() {{
  const gateIds = {gate_ids!r};
  const gateColors = {gate_colors!r};
  const wrap = document.getElementById('flowWrap');
  const table = document.getElementById('gateTable');
  const svg = document.getElementById('flowSvg');
  const spine = document.getElementById('flowSpine');
  const pulse = document.getElementById('flowPulse');
  const btn = document.getElementById('flowReplayBtn');
  const btnIcon = document.getElementById('flowReplayIcon');
  const btnText = document.getElementById('flowReplayText');
  let tribPaths = [];
  let running = false;
  let geom = {{ tribStartX: 0, spineX: 0, bendY: 0, endX: 0, endY: 0 }};
  const pulsePathEl = document.getElementById('flowPulsePath');

  function build() {{
    // The overlay svg is position:absolute inside a position:relative
    // wrapper, so its own height feeds back into wrap.scrollHeight. Zero it
    // before measuring, or every re-run (load/fonts.ready/300ms/1000ms/
    // resize/ResizeObserver) reads a height that includes its own last
    // output and drifts upward a little each pass - the actual cause of the
    // widget's content height creeping between page loads.
    svg.setAttribute('height', '0');
    const wrapRect = wrap.getBoundingClientRect();
    const wrapHeight = wrap.scrollHeight;
    svg.setAttribute('width', wrapRect.width);
    svg.setAttribute('height', wrapHeight);
    svg.setAttribute('viewBox', '0 0 ' + wrapRect.width + ' ' + wrapHeight);
    const rows = table.querySelectorAll('tr');
    const tableRect = table.getBoundingClientRect();
    const tribStartX = tableRect.right - wrapRect.left;
    const spineX = tribStartX + 45;
    const heading = document.getElementById('plantModelHeading');
    const firstSecBox = wrap.querySelector('.sec-box');
    const endY = firstSecBox ? (firstSecBox.getBoundingClientRect().top - wrapRect.top) :
      (heading ? (heading.getBoundingClientRect().top - wrapRect.top) : (tableRect.bottom - wrapRect.top));
    const endX = heading ? wrapRect.width / 2 : spineX;
    document.querySelectorAll('.flow-trib').forEach(function(el) {{ el.remove(); }});
    tribPaths = [];
    rows.forEach(function(row, i) {{
      const r = row.getBoundingClientRect();
      const y = (r.top - wrapRect.top) + r.height / 2;
      const d = 'M' + tribStartX + ',' + y + ' H ' + spineX;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', d);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', '#9B96C9');
      path.setAttribute('stroke-width', '1.5');
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('class', 'flow-trib');
      path.setAttribute('id', 'flowTrib' + i);
      svg.insertBefore(path, spine);
      tribPaths.push({{path: path, y: y}});
    }});
    const lastY = tribPaths.length ? tribPaths[tribPaths.length - 1].y : 0;
    const bendY = endX === spineX ? lastY : Math.max(lastY, endY - 30);
    spine.setAttribute('d', 'M' + spineX + ',' + (tribPaths[0] ? tribPaths[0].y : 0) +
      ' V ' + bendY + ' H ' + endX + ' V ' + endY);
    geom = {{ tribStartX: tribStartX, spineX: spineX, bendY: bendY, endX: endX, endY: endY }};
  }}

  function fullJourneyPath(i) {{
    // One unbroken path for gate i's whole journey - row, out to the shared
    // spine, down, and into the solver box - so the pulse glides across the
    // tributary-to-spine join as a single continuous line instead of two
    // separate animations that visibly seam/jump where they meet.
    const y = tribPaths[i] ? tribPaths[i].y : 0;
    return 'M' + geom.tribStartX + ',' + y + ' H ' + geom.spineX +
      ' V ' + geom.bendY + ' H ' + geom.endX + ' V ' + geom.endY;
  }}

  function resetAll() {{
    Array.from(table.querySelectorAll('tr')).forEach(function(row) {{ row.style.outline = 'none'; }});
    tribPaths.forEach(function(t) {{ t.path.setAttribute('stroke', '#9B96C9'); t.path.setAttribute('stroke-width', '1.5'); }});
    spine.setAttribute('stroke', '#9B96C9');
    document.getElementById('flowArrowHead').setAttribute('fill', '#9B96C9');
    pulse.setAttribute('opacity', '0');
    btnText.textContent = 'Play';
    btnIcon.innerHTML = '&#9654;';
  }}

  function wait(ms) {{
    // Polls stopRequested every 100ms instead of a single setTimeout(ms) -
    // every await wait(...) in play() (including scroll-wait durations that
    // scale with distance, up to several seconds each) was previously
    // un-interruptible, so Stop could take a very long time to actually
    // take effect. Bounds that to ~100ms regardless of the requested delay.
    return new Promise(function(resolve) {{
      if (ms <= 0) {{ resolve(); return; }}
      const stepMs = 100;
      let elapsed = 0;
      const id = setInterval(function() {{
        elapsed += stepMs;
        if (stopRequested || elapsed >= ms) {{ clearInterval(id); resolve(); }}
      }}, stepMs);
    }});
  }}

  function scrollFollowY(y) {{
    // Keep the viewport centered on the dot's CURRENT position every frame,
    // instead of jumping to the destination once up front and leaving the
    // page to sit still while the dot is still seconds away from arriving -
    // that's what made the scroll and the dot look disconnected.
    try {{
      const parentDoc = window.parent.document;
      const frameEl = window.frameElement;
      const scroller = parentDoc.querySelector('[data-testid="stMain"]');
      if (!scroller || !frameEl) {{ return; }}
      const frameRect = frameEl.getBoundingClientRect();
      const scrollerRect = scroller.getBoundingClientRect();
      const targetTop = scroller.scrollTop + (frameRect.top - scrollerRect.top) + y - (scroller.clientHeight / 2);
      scroller.scrollTop = Math.max(0, targetTop);
    }} catch (e) {{}}
  }}

  function animateAlong(path, color, duration, follow) {{
    // Checks stopRequested every frame - without this, a stop click (button
    // or click-anywhere) has no effect until the current gate's full
    // pulse-travel duration finishes on its own, which can be many seconds
    // to tens of seconds for gates further down the table. That made Stop
    // look unresponsive/stuck for up to a minute in practice.
    return new Promise(function(resolve) {{
      const len = path.getTotalLength();
      pulse.setAttribute('fill', color);
      pulse.setAttribute('opacity', '1');
      const start = performance.now();
      function step(now) {{
        if (stopRequested) {{ resolve(); return; }}
        const t = Math.min(1, (now - start) / duration);
        const pt = path.getPointAtLength(t * len);
        pulse.setAttribute('cx', pt.x);
        pulse.setAttribute('cy', pt.y);
        if (follow) {{ scrollFollowY(pt.y); }}
        if (t < 1) {{ requestAnimationFrame(step); }} else {{ resolve(); }}
      }}
      requestAnimationFrame(step);
    }});
  }}

  function getScrollerAndTarget(el) {{
    const parentDoc = window.parent.document;
    const frameEl = window.frameElement;
    const scroller = parentDoc.querySelector('[data-testid="stMain"]');
    if (!scroller || !frameEl) {{ return null; }}
    const rect = el.getBoundingClientRect();
    const frameRect = frameEl.getBoundingClientRect();
    const scrollerRect = scroller.getBoundingClientRect();
    const top = scroller.scrollTop + (frameRect.top - scrollerRect.top) + rect.top
      - (scroller.clientHeight / 2) + (rect.height / 2);
    return {{ scroller: scroller, top: Math.max(0, top) }};
  }}

  function scrollGlide(el, duration) {{
    // One smooth, continuous scroll toward el over the given duration - no
    // per-stop highlight/pause. A steady glide down through the content is
    // enough to read as it passes, and is far quicker than stopping on
    // every single equation card.
    return new Promise(function(resolve) {{
      const target = getScrollerAndTarget(el);
      if (!target) {{ resolve(); return; }}
      const scroller = target.scroller;
      const fromTop = scroller.scrollTop;
      const toTop = target.top;
      const start = performance.now();
      function step(now) {{
        const t = Math.min(1, (now - start) / duration);
        scroller.scrollTop = fromTop + (toTop - fromTop) * t;
        if (t < 1 && !stopRequested) {{ requestAnimationFrame(step); }} else {{ resolve(); }}
      }}
      requestAnimationFrame(step);
    }});
  }}

  function scrollToEl(el) {{
    try {{
      const target = getScrollerAndTarget(el);
      if (!target) {{ el.scrollIntoView({{behavior: 'smooth', block: 'center'}}); return 0; }}
      const scroller = target.scroller;
      const targetTop = target.top;
      const distance = Math.abs(Math.max(0, targetTop) - scroller.scrollTop);
      scroller.scrollTo({{top: Math.max(0, targetTop), behavior: 'smooth'}});
      return distance;
    }} catch (e) {{
      try {{ el.scrollIntoView({{behavior: 'smooth', block: 'center'}}); }} catch (e2) {{}}
      return 0;
    }}
  }}

  function scrollWaitMs(distance) {{
    // Native smooth-scroll duration scales with distance - a fixed wait is
    // long enough for a short hop but too short for a long one, so the next
    // scrollTo() retargets mid-flight and the browser visibly snaps instead
    // of gliding. Scale the pause with how far this particular jump is.
    return Math.min(2600, Math.max(900, distance * 1.1));
  }}

  let stopRequested = false;

  function hardReset() {{
    Array.from(table.querySelectorAll('tr')).forEach(function(row) {{ row.style.outline = 'none'; }});
    wrap.querySelectorAll('.sec-box').forEach(function(box) {{ box.style.outline = 'none'; }});
    wrap.querySelectorAll('.card').forEach(function(card) {{ card.style.outline = 'none'; }});
    resetAll();
  }}

  async function play() {{
    if (running) {{ stopRequested = true; return; }}
    running = true;
    stopRequested = false;
    build();
    hardReset();
    btnText.textContent = 'Playing';
    const cards = Array.from(wrap.querySelectorAll('.card'));
    await wait(scrollWaitMs(scrollToEl(table.querySelectorAll('tr')[0])));
    const rows = table.querySelectorAll('tr');
    for (let i = 0; i < gateIds.length && !stopRequested; i++) {{
      const color = gateColors[i];
      const row = rows[i];
      const trib = tribPaths[i].path;
      await wait(scrollWaitMs(scrollToEl(row)));
      if (stopRequested) break;
      row.style.outline = '2px solid ' + color;
      trib.setAttribute('stroke', color);
      trib.setAttribute('stroke-width', '2.5');
      spine.setAttribute('stroke', color);
      document.getElementById('flowArrowHead').setAttribute('fill', color);
      // One unbroken path and one animateAlong() call for the whole journey -
      // row to spine to solver box - so the dot glides through the join with
      // no seam, instead of two separate animations that visibly jump speed
      // where the tributary meets the spine. The scroll now follows the dot
      // every frame (see scrollFollowY) instead of jumping to the destination
      // once and leaving the page sitting still while the dot is still en
      // route - that mismatch was what made the scroll look disconnected.
      pulsePathEl.setAttribute('d', fullJourneyPath(i));
      const pulseSpeed = 0.18; // px per ms - slow enough to watch it glide, not blur past
      const journeyDuration = Math.max(3000, pulsePathEl.getTotalLength() / pulseSpeed);
      await animateAlong(pulsePathEl, color, journeyDuration, true);
      if (stopRequested) break;
      await wait(700);
      pulse.setAttribute('opacity', '0');
      // One smooth continuous scroll through the whole equation chain, all
      // the way to the last card (solver extraction) - stopping and
      // highlighting every single card took too long and read as blinking;
      // a steady glide down through the content is enough to follow along.
      if (cards.length) {{
        const lastCard = cards[cards.length - 1];
        const target = getScrollerAndTarget(lastCard);
        const glideDistance = target ? Math.abs(target.top - target.scroller.scrollTop) : 4000;
        const glideDuration = Math.min(30000, Math.max(6000, glideDistance * 6));
        await scrollGlide(lastCard, glideDuration);
      }}
      if (stopRequested) break;
      await wait(800);
      row.style.outline = 'none';
      trib.setAttribute('stroke', '#9B96C9');
      trib.setAttribute('stroke-width', '1.5');
      spine.setAttribute('stroke', '#9B96C9');
      document.getElementById('flowArrowHead').setAttribute('fill', '#9B96C9');
      await wait(600);
    }}
    hardReset();
    btnText.textContent = 'Replay';
    btnIcon.innerHTML = '&#8635;';
    running = false;
    stopRequested = false;
  }}

  function requestStop(e) {{
    if (!running) return;
    if (e.target && e.target.closest && e.target.closest('#flowReplayBtn')) return;
    stopRequested = true;
  }}

  btn.addEventListener('click', play);
  document.addEventListener('click', requestStop, true);
  try {{ window.parent.document.addEventListener('click', requestStop, true); }} catch (e) {{}}
  window.addEventListener('resize', build);
  build();
  // Re-measure after fonts/late layout settle - the initial synchronous
  // build() can run before web fonts finish loading, leaving the connector
  // pinned to slightly-stale row/table coordinates (visible as the line
  // appearing to "cut" short of the real row edge at normal zoom).
  if (document.fonts && document.fonts.ready) {{ document.fonts.ready.then(build); }}
  setTimeout(build, 300);
  setTimeout(build, 1000);
  if (window.ResizeObserver) {{
    new ResizeObserver(build).observe(table);
    new ResizeObserver(build).observe(wrap);
  }}
}})();
</script>
"""
    return _CSS + _TABLE_CSS + '<div class="eqwrap">' + body + '</div>'


def input_data_html() -> str:
    """Every real input source (static config, forecasts, carried state) feeding the model, mapped to sections."""
    return build_inputs_html()


def pipeline_and_model_html() -> str:
    """Input Data + Plant modeling & optimization combined into ONE component,
    so the gate-flow connector can be drawn as a single continuous line all the
    way into the equations heading - Streamlit's components.html() sandboxes
    each block in its own iframe, so two separate calls can never be bridged
    with a real connecting line, only an approximation drawn from the outside."""
    gate_colors = [_SECTION_COLORS[k][0] for k in _GATE_SECTION_KEYS]
    extra_content = (
        '<h3 id="plantModelHeading">Plant Modeling &amp; Optimization</h3>'
        + _build_equations_content()
    )
    return _flow_widget_html(_build_gate_rows(), _GATE_IDS, gate_colors, extra_content)


_LOOP_CSS = """
<style>
.plc-wrap { font-family: 'Times New Roman', Times, serif; max-width: 900px; margin: 10px auto 0 auto; }
.plc-row { display: flex; align-items: stretch; justify-content: center; gap: 0; width: 100%; box-sizing: border-box; }
.plc-card { flex: 1 1 0; min-width: 0; background: #ffffff; border: 1px solid #eceae4; border-top: 4px solid;
            border-radius: 12px; padding: 14px 6px 12px 6px; text-align: center; box-sizing: border-box;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.05); }
.plc-badge { width: 22px; height: 22px; border-radius: 50%; color: #fff; font-size: 11px;
             font-weight: 700; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px auto; }
.plc-title { font-size: 14px; font-weight: 700; margin-bottom: 4px; }
.plc-sub { font-size: 11.5px; color: #5f5e5a; }
.plc-arrow-h { flex: 0 0 20px; display: flex; align-items: center; justify-content: center; }
.plc-title { overflow-wrap: break-word; }
.plc-varrows-row { display: flex; align-items: center; justify-content: center; gap: 0; width: 100%; box-sizing: border-box; margin: 6px 0; }
.plc-varrow-slot { flex: 1 1 0; min-width: 0; display: flex; align-items: center; justify-content: center; }
.plc-varrow-gap { flex: 0 0 20px; }
.plc-model { max-width: 900px; margin: 0 auto; background: #F8F7FE; border: 1.5px solid #534AB7; border-radius: 14px;
             padding: 16px 20px; text-align: center; }
.plc-model .t { font-size: 15px; font-weight: 700; color: #000000; }
.plc-model .s { font-size: 12px; color: #3c3489; margin-top: 3px; }
.plc-head { display: flex; align-items: center; justify-content: flex-end; margin-bottom: 8px; }
.plc-play { display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 700;
            padding: 6px 14px; border-radius: 20px; border: 1px solid #C7C4BC; background: #fff;
            color: #2C2C2A; cursor: pointer; font-family: 'Times New Roman', Times, serif; }
.plc-play:hover { background: #F5F4F2; }
.plc-play:disabled { opacity: 0.5; cursor: default; }
.plc-card, .plc-model { transition: box-shadow 0.25s ease, transform 0.25s ease; }
.plc-card.plc-active, .plc-model.plc-active { box-shadow: 0 0 0 3px rgba(83,74,183,0.35), 0 4px 12px rgba(0,0,0,0.12);
                                                 transform: translateY(-2px); }
.plc-varrow-slot { opacity: 0.35; transition: opacity 0.25s ease; }
.plc-varrow-slot.plc-active { opacity: 1; }
</style>
"""


def _plc_card(num: int, title: str, sub: str, color: str, text_color: str) -> str:
    return (f'<div class="plc-card" id="plcCard{num}" style="border-top-color:{color};">'
            f'<div class="plc-badge" style="background:{color};">{num}</div>'
            f'<div class="plc-title" style="color:{text_color};">{title}</div>'
            f'<div class="plc-sub">{sub}</div></div>')


def _plc_arrow_h(color: str) -> str:
    return (f'<div class="plc-arrow-h"><svg width="22" height="16" viewBox="0 0 22 16">'
            f'<line x1="0" y1="8" x2="12" y2="8" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
            f'<path d="M10 2 L20 8 L10 14 Z" fill="{color}"/></svg></div>')


def _plc_varrow(color: str) -> str:
    return (f'<svg width="20" height="20" viewBox="0 0 20 20">'
            f'<line x1="10" y1="0" x2="10" y2="9" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
            f'<path d="M3 8 L10 19 L17 8 Z" fill="{color}"/></svg>')


def _plc_varrows_row(colors: list[str]) -> str:
    """One down-arrow per card, colored to match its own card, aligned under
    it via the same flex widths as the card row above."""
    slots = [f'<div class="plc-varrow-slot" id="plcArrow{i+1}">{_plc_varrow(c)}</div>'
             for i, c in enumerate(colors)]
    gap = '<div class="plc-varrow-gap"></div>'
    parts = [slots[0]]
    for s in slots[1:]:
        parts.append(gap)
        parts.append(s)
    return '<div class="plc-varrows-row">' + "".join(parts) + '</div>'


def gate_repetition_loop_html() -> str:
    """DA/IDA1/IDA2/IDA3/XBID - the only gates that call Plant modeling &
    optimization (core_milp_builder.py::build_core_model, confirmed 5 calls
    a day). aFRR/mFRR are deliberately not drawn as part of this chain -
    they read the committed position but never call this model; see the
    step-by-step widget below for what they actually do."""
    teal, blue, amber = "#1baf7a", "#185FA5", "#c9820f"
    teal_txt, blue_txt, amber_txt = "#0c6b4c", "#0c4a7a", "#7a4f09"
    gates = [
        (1, "DA bid", "Runs it", teal, teal_txt),
        (2, "IDA1", "Re-runs it", blue, blue_txt),
        (3, "IDA2", "Re-runs it", blue, blue_txt),
        (4, "IDA3", "Re-runs it", blue, blue_txt),
        (5, "XBID", "Continuous", amber, amber_txt),
    ]
    cards = [_plc_card(n, t, s, c, tc) for n, t, s, c, tc in gates]
    row = cards[0]
    for i, card in enumerate(cards[1:], start=1):
        row += _plc_arrow_h(gates[i][3]) + card
    return _LOOP_CSS + f"""
<div class="plc-wrap">
  <div class="plc-head">
    <button class="plc-play" id="plcPlayBtn" onclick="plcPlay()">&#9654; Play</button>
  </div>
  <div class="plc-row">
    {row}
  </div>
  {_plc_varrows_row([g[3] for g in gates])}
  <div class="plc-model" id="plcModel">
    <div class="t">Plant Modeling &amp; Optimization</div>
  </div>
</div>
<script>
function plcPlay() {{
  const btn = document.getElementById('plcPlayBtn');
  const cards = [1,2,3,4,5].map(n => document.getElementById('plcCard'+n));
  const arrows = [1,2,3,4,5].map(n => document.getElementById('plcArrow'+n));
  const model = document.getElementById('plcModel');
  btn.disabled = true;
  let i = 0;
  function clearAll() {{
    cards.forEach(n => n.classList.remove('plc-active'));
    arrows.forEach(a => a.classList.remove('plc-active'));
    model.classList.remove('plc-active');
  }}
  function step() {{
    clearAll();
    if (i < cards.length) {{
      // each gate lights up together with its own arrow into the plant
      // model, so "DA -> Plant" (and then IDA1 -> Plant, etc.) is what
      // actually animates, not the card alone.
      cards[i].classList.add('plc-active');
      arrows[i].classList.add('plc-active');
      model.classList.add('plc-active');
      i++;
      setTimeout(step, 700);
    }} else {{
      clearAll();
      btn.disabled = false;
    }}
  }}
  step();
}}
</script>
"""


_PLAIN_STEPS_CSS = """
<style>
.pls-wrap { font-family: 'Times New Roman', Times, serif; max-width: 700px; margin: 10px auto 0 auto; }
.pls-step { border-radius: 14px; padding: 18px 24px 20px 24px; margin: 0 auto;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.05);
            background: #ffffff; border: 1px solid #eceae4; border-left: 4px solid; }
.pls-step.aqua  { border-left-color: #1baf7a; }
.pls-step.amber { border-left-color: #eda100; }
.pls-head { display: flex; align-items: center; gap: 11px; margin: 0 0 12px 0; }
.pls-badge { flex: 0 0 auto; width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center;
             justify-content: center; font-size: 13px; font-weight: 700; color: #ffffff; }
.pls-step.aqua  .pls-badge { background: #1baf7a; }
.pls-step.amber .pls-badge { background: #eda100; }
.pls-title { font-size: 16px; font-weight: 700; letter-spacing: 0.1px; margin: 0; }
.pls-step.aqua  .pls-title { color: #0c6b4c; }
.pls-step.amber .pls-title { color: #8a6100; }
.pls-tag { font-size: 10.5px; font-weight: 600; letter-spacing: 0.4px; text-transform: uppercase;
           padding: 2px 9px; border-radius: 999px; margin-left: auto; }
.pls-step.aqua  .pls-tag { background: rgba(27,175,122,0.13); color: #0c6b4c; }
.pls-step.amber .pls-tag { background: rgba(237,161,0,0.15); color: #8a6100; }
.pls-list { margin: 0; padding: 0; list-style: none; }
.pls-list li { font-size: 13.5px; color: #33322f; line-height: 1.65; padding-left: 20px; position: relative; margin-bottom: 3px; }
.pls-list li:last-child { margin-bottom: 0; }
.pls-list li::before { content: '\\2192'; position: absolute; left: 0; top: 0; font-size: 12px; }
.pls-step.aqua  .pls-list li::before { color: #1baf7a; }
.pls-step.amber .pls-list li::before { color: #eda100; }
.pls-arrow { text-align: center; margin: 2px 0; line-height: 1; }
.pls-arrow svg { display: block; margin: 0 auto; }
.pls-note { text-align: center; font-size: 12.5px; font-style: italic; color: #6b6a66; margin: 14px 0 6px 0; line-height: 1.6; }
.pls-legend { text-align: center; font-size: 11.5px; color: #6b6a66; margin-top: 10px; padding-top: 10px;
              border-top: 1px solid #eceae4; }
.pls-legend .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin: 0 5px 0 14px; vertical-align: middle; }
.pls-legend .dot:first-child { margin-left: 0; }
.pls-legend .dot.aqua { background: #1baf7a; }
.pls-legend .dot.amber { background: #eda100; }
</style>
"""

def _pls_arrow(color: str) -> str:
    """A short colored connector between two step cards - a soft stem plus a
    filled chevron head, tinted to the step it's leading into."""
    return (f'<div class="pls-arrow"><svg width="20" height="30" viewBox="0 0 20 30">'
            f'<line x1="10" y1="0" x2="10" y2="15" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
            f'<path d="M3 14 L10 26 L17 14 Z" fill="{color}"/></svg></div>')


def pipeline_steps_plain_html() -> str:
    """DA/aFRR/mFRR/IDA1, one real step at a time, in plain language for a
    first-time reader. Teal = actually runs Plant modeling & optimization
    (DA, IDA1, and by extension IDA2/IDA3/XBID). Amber = a standalone sizing
    formula that never calls the optimizer (aFRR, mFRR) - see
    reserve_offer_builder.py::compute_price_aware_fraction and
    afrr_offer_builder.py/mfrr_offer_builder.py for the real code this maps to."""
    return _PLAIN_STEPS_CSS + f"""
<div class="pls-wrap">
  <div class="pls-step aqua">
    <div class="pls-head">
      <div class="pls-badge">1</div>
      <div class="pls-title">DA gate</div>
      <div class="pls-tag">Runs the optimizer</div>
    </div>
    <ul class="pls-list">
      <li>Guess tomorrow's price, sunshine and river inflow</li>
      <li>Run the optimizer to pick the best power plan, hour by hour</li>
      <li>Lock in that plan: how many MW to sell, at what price</li>
    </ul>
  </div>
  {_pls_arrow('#eda100')}
  <div class="pls-step amber">
    <div class="pls-head">
      <div class="pls-badge">2</div>
      <div class="pls-title">aFRR gate</div>
      <div class="pls-tag">Quick rule</div>
    </div>
    <ul class="pls-list">
      <li>Look at the power plan already locked in for tomorrow</li>
      <li>Predict the standby price with a model trained on real REN/PICASSO prices back to 2019</li>
      <li>Set aside some capacity as backup, using a quick rule</li>
      <li>Save that backup amount, within the plant's physical limits</li>
    </ul>
  </div>
  {_pls_arrow('#eda100')}
  <div class="pls-step amber">
    <div class="pls-head">
      <div class="pls-badge">3</div>
      <div class="pls-title">mFRR gate</div>
      <div class="pls-tag">Quick rule</div>
    </div>
    <ul class="pls-list">
      <li>Look at whatever capacity aFRR left free</li>
      <li>Predict the standby price with a model trained on real REN/MARI prices back to 2024</li>
      <li>Set aside more capacity as backup, using the same quick rule</li>
      <li>Save that backup amount too, within the plant's physical limits</li>
    </ul>
  </div>
  {_pls_arrow('#1baf7a')}
  <div class="pls-step aqua">
    <div class="pls-head">
      <div class="pls-badge">4</div>
      <div class="pls-title">IDA1 gate</div>
      <div class="pls-tag">Runs the optimizer</div>
    </div>
    <ul class="pls-list">
      <li>Look at the locked-in plan plus both backup amounts</li>
      <li>Guess this round's price</li>
      <li>Run the optimizer again, with less power now free to trade</li>
      <li>Check if the extra money beats the cost of changing the plan</li>
      <li>If yes, update the plan; if not, leave it as it is</li>
    </ul>
  </div>
  <div class="pls-note">IDA2, IDA3 and XBID repeat step 4, exactly the same way,<br/>each with a fresh price guess and the plan from the round before</div>
  <div class="pls-legend"><span class="dot aqua"></span>Runs the optimizer<span class="dot amber"></span>Quick rule, not the optimizer</div>
</div>
"""
