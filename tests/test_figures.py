"""test_figures.py - the figure package writes its SVG files from the run's outputs, and skips (never fails) a figure whose input is missing."""
from __future__ import annotations

import pytest

import figures
from figures import _data as D


def test_generate_writes_valid_svgs():
    day = D.latest_date()
    if day is None or D.backtest() is None:
        pytest.skip("needs a solved day with a report and a backtest report in runtime/")
    written = figures.generate(day)
    assert len(written) == 8
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert text.lstrip().startswith("<?xml") and "<svg" in text
        assert "<text" not in text                  # text is converted to paths: the figure looks the same on every machine


def test_missing_day_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "RUNTIME", tmp_path)
    monkeypatch.setattr(figures._style, "OUT", tmp_path / "out") if hasattr(figures, "_style") else None
    assert isinstance(figures.generate("2000-01-01"), list)
