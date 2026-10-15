"""fill_legend.py - how a missing price is filled: for each method a small picture and the exact condition, in the order of the table columns.

Blue dots are measured values, hollow amber dots are filled values. The pictures show the idea only; their values are not data. The conditions
are those of config/data_curation.yaml and common_layer/data_quality/curation.py."""
from __future__ import annotations

import base64
import html

import theme

_MEASURED, _FILLED, _PRICE, _GREY = theme.BLUE, theme.YELLOW, theme.VIOLET, "#b4b2a9"


def _dot(x, y, filled=False):
    if filled:
        return f'<circle cx="{x}" cy="{y}" r="4.5" fill="none" stroke="{_FILLED}" stroke-width="2"/>'
    return f'<circle cx="{x}" cy="{y}" r="4" fill="{_MEASURED}"/>'


def _line(points, colour, width=2, dash=None):
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="{width}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>"


def _chart(body: str) -> str:
    """A mini chart as an image (an SVG inside an <img> is never touched by the page's HTML cleaning)."""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 10 285 104" font-family="system-ui, -apple-system, \'Segoe UI\', sans-serif">'
           f'<line x1="16" y1="108" x2="270" y2="108" stroke="{_GREY}" stroke-width="0.5"/>{body}</svg>')
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{data}" width="100%" alt="">'


def _straight() -> str:
    return _chart(_line([(16, 72), (42, 62), (68, 57)], _MEASURED) + _line([(172, 32), (198, 37), (224, 27), (250, 32)], _MEASURED)
                  + _line([(68, 57), (172, 32)], _FILLED, dash="4 4")
                  + "".join(_dot(x, y) for x, y in [(16, 72), (42, 62), (68, 57), (172, 32), (198, 37), (224, 27), (250, 32)])
                  + "".join(_dot(x, y, True) for x, y in [(94, 50.75), (120, 44.5), (146, 38.25)]))


def _carried() -> str:
    return _chart(_line([(16, 62), (42, 54), (68, 57)], _MEASURED) + _line([(224, 27), (250, 32)], _MEASURED) + _line([(68, 57), (198, 57)], _FILLED, dash="4 4")
                  + "".join(_dot(x, y) for x, y in [(16, 62), (42, 54), (68, 57), (224, 27), (250, 32)]) + "".join(_dot(x, 57, True) for x in (94, 120, 146, 172, 198)))


def _spread() -> str:
    da = [(16, 70), (39, 65), (62, 60), (85, 55), (108, 52), (131, 50), (154, 54), (177, 60), (200, 65), (223, 68), (246, 70)]
    return _chart(_line(da, _PRICE) + "".join(_dot(x, y) for x, y in [(16, 57), (39, 51), (62, 45), (85, 41), (177, 46), (200, 51), (223, 54), (246, 56)])
                  + "".join(_dot(x, y, True) for x, y in [(108, 38), (131, 36), (154, 40)])
                  + f'<line x1="131" y1="41" x2="131" y2="50" stroke="{_FILLED}" stroke-width="1"/>'
                  + f'<text x="146" y="22" font-size="12" fill="{theme.INK_SECONDARY}">+ typical spread</text>')


def _earlier() -> str:
    week = [(16, 55), (34, 45), (52, 40), (70, 35), (88, 25), (106, 20), (124, 28), (142, 38), (160, 45), (178, 50), (196, 55), (214, 48), (232, 42), (250, 38)]
    return _chart(_line(week, _GREY, 1.5) + _line([(16, 58), (34, 48), (52, 43), (70, 38)], _MEASURED) + _line([(196, 52), (214, 45), (232, 40), (250, 36)], _MEASURED)
                  + "".join(_dot(x, y) for x, y in [(16, 58), (34, 48), (52, 43), (70, 38), (196, 52), (214, 45), (232, 40), (250, 36)])
                  + "".join(_dot(x, y, True) for x, y in [(88, 25), (106, 20), (124, 28), (142, 38), (160, 45), (178, 50)]))


_CARDS = [
    ("Straight line", "the rule for every series", _straight,
     [("Gap length", "1 to 3 hours"), ("Condition", "a real value on both sides of the gap"), ("Value", "a straight line in time between the two neighbours")]),
    ("Carried forward", "exception: aFRR band prices (they change in blocks)", _carried,
     [("Gap length", "up to 48 hours"), ("Value", "the last known value, repeated"),
      ("Also", "the last resort for any series, when nothing a week, a day or two weeks earlier exists")]),
    ("Day-ahead price + spread", "exception: IDA1, IDA2, IDA3, XBID (forecast as a spread to the day-ahead price)", _spread,
     [("Gap length", "any length"), ("Condition", "the day-ahead price of that hour exists"),
      ("Value", "day-ahead price + median spread of that hour over the last 28 days (0 if there is no history)")]),
    ("Same hour earlier", "the rule for every series", _earlier,
     [("Gap length", "longer than 3 hours, no upper limit"), ("Condition", "an earlier value exists"), ("Value", "the same hour 1 week earlier; else yesterday; else 2 weeks earlier")]),
]

_NOTE = ("One rule for every series: a gap of 1 to 3 hours is joined by a straight line, a longer gap repeats the same hour from earlier. Two exceptions, each with a reason: "
         "aFRR band prices are carried forward (up to 48 hours) and IDA1-3 and XBID use the day-ahead price + spread. Wrong values are removed first and then count as missing. "
         "Genuine spikes and negative prices are kept.")


def html_block() -> str:
    border, muted, ink = theme.GRIDLINE, theme.INK_MUTED, theme.INK_PRIMARY
    cards = ""
    for title, tag, chart, rows in _CARDS:
        lines = "".join(f'<div style="display:grid; grid-template-columns:4.6rem 1fr; gap:0.5rem; padding:0.25rem 0; border-top:1px solid {border}; font-size:0.82rem; line-height:1.35;">'
                        f'<div style="color:{muted}; font-size:0.72rem; padding-top:0.08rem;">{html.escape(k)}</div><div style="color:{ink};">{html.escape(v)}</div></div>' for k, v in rows)
        cards += (f'<div style="border:1px solid {border}; border-radius:12px; background:{theme.SURFACE}; padding:0.55rem 0.8rem 0.3rem;">'
                  f'<div style="line-height:1.3; margin-bottom:0.25rem;"><span style="font-weight:600; font-size:0.93rem; color:{ink};">{html.escape(title)}</span>'
                  f'<span style="font-size:0.74rem; color:{muted};"> &middot; {html.escape(tag)}</span></div>'
                  f'<div style="display:flex; gap:0.8rem; align-items:center;"><div style="flex:0 0 34%; min-width:0;">{chart()}</div><div style="flex:1 1 0; min-width:0;">{lines}</div></div></div>')
    legend = (f'<div style="display:flex; gap:1.2rem; align-items:center; font-size:0.78rem; color:{theme.INK_SECONDARY}; margin:0.4rem 0 0.1rem;">'
              f'<span><span style="display:inline-block; width:0.7rem; height:0.7rem; border-radius:50%; background:{_MEASURED}; vertical-align:-1px;"></span> measured value</span>'
              f'<span><span style="display:inline-block; width:0.55rem; height:0.55rem; border-radius:50%; border:2px solid {_FILLED}; vertical-align:-1px;"></span> filled value</span></div>')
    return (f'<div style="margin:0.2rem 0 0.6rem 0;"><div style="display:grid; grid-template-columns:1fr 1fr; gap:0.6rem;">{cards}</div>{legend}'
            f'<div style="font-size:0.78rem; color:{theme.INK_SECONDARY}; line-height:1.45;">{html.escape(_NOTE)}</div></div>')
