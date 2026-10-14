"""
frame_fit.py - every card of the dashboard that is drawn in its own frame (components.html) takes exactly the height of its content.

The cards used to be given a fixed height by the page that showed them, found by trial: too small and the bottom of the card was cut off, too large and
a blank band stayed under it. A short script is added to each card instead: it measures the card and sets the height of its own frame, and measures again
when the card changes size (a replay, a window resize).

    install()      wraps streamlit.components.v1.html once per process
"""
from __future__ import annotations

import functools

SCRIPT = """
<script>
(function () {
  function fit() {
    if (!window.frameElement) return;
    var bottom = 0, kids = document.body.children;
    for (var i = 0; i < kids.length; i++) {
      var tag = kids[i].tagName;
      if (tag === "SCRIPT" || tag === "STYLE") continue;
      bottom = Math.max(bottom, kids[i].getBoundingClientRect().bottom);
    }
    if (!bottom) return;
    var h = Math.ceil(bottom) + 4;
    if (Math.abs(h - (parseInt(window.frameElement.style.height) || 0)) > 1) {
      window.frameElement.style.height = h + "px";
      var box = window.frameElement.closest('[data-testid="stElementContainer"]');          // Streamlit also reserves the old height around the frame
      var node = window.frameElement.parentElement;
      while (node) {                                                                          // every wrapper up to and including the container
        node.style.setProperty("height", h + "px", "important"); node.style.setProperty("min-height", "0", "important"); node.style.setProperty("flex-basis", h + "px", "important");   // Streamlit's own rule is !important
        if (node === box) break;
        node = node.parentElement;
      }
    }
  }
  fit();
  window.addEventListener("load", fit);
  window.addEventListener("resize", fit);
  if (window.ResizeObserver) new ResizeObserver(fit).observe(document.body);
  setTimeout(fit, 150); setTimeout(fit, 600);
})();
</script>
"""


def install() -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    if getattr(st, "_frame_fit", False):
        return
    st._frame_fit = True
    original = components.html

    @functools.wraps(original)
    def html(source, *args, **kwargs):
        if isinstance(source, str) and "border-radius" in source:         # a visible card; the invisible script frames are left alone
            source = source + SCRIPT
        return original(source, *args, **kwargs)

    components.html = html
