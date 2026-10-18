r"""maths.py - formulas written in LaTeX, shown as real mathematics inside the HTML cards and tables of the dashboard.

`tex(r"\frac{1}{n}\sum_{i=1}^{n} \lvert e_i \rvert")` returns the formula as MathML, which the browser draws natively (a card made with
`st.markdown(..., unsafe_allow_html=True)` does not pass its text through the LaTeX renderer of Streamlit, so `$...$` would stay as typed).
The source of every formula is plain LaTeX: no Unicode symbols are typed in the code. A formula that cannot be converted is shown as its
LaTeX text, never as an error."""
from __future__ import annotations

import html
from functools import lru_cache

try:
    import latex2mathml.converter as _converter
except ImportError:                                                                       # the formula is then shown as LaTeX text
    _converter = None


@lru_cache(maxsize=1024)
def tex(latex: str, display: bool = False, size: float = 1.22) -> str:
    """A LaTeX formula as MathML, ready to be placed inside an HTML string."""
    if _converter is None:
        return f"<code>{html.escape(latex)}</code>"
    try:
        ml = _converter.convert(latex, display="block" if display else "inline")
    except Exception:                                                                     # noqa: BLE001 - a formula the converter does not know
        return f"<code>{html.escape(latex)}</code>"
    ml = ml.replace("<math ", "<math style=\"font-size:" + f"{size}em" + "; font-family:'Cambria Math','STIX Two Math','Latin Modern Math',serif;\" ", 1)
    head, _, rest = ml.partition(">")                                                     # a fraction keeps its full size inside a line of text
    return f"{head}><mstyle displaystyle=\"true\" scriptlevel=\"0\">{rest[:-len('</math>')]}</mstyle></math>"
