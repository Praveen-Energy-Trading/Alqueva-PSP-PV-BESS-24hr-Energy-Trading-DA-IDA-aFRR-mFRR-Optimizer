"""
test_xbid_backtest.py — real-price backtest extension to XBID (window W1),
the final step of the real production chain DA -> IDA1 -> IDA2 -> IDA3 -> XBID.
"""
from __future__ import annotations

import pytest

from phase_6_backtesting_and_validation.backtest_engine.historical_data_loader import (
    real_ida_price,
)
from phase_6_backtesting_and_validation.backtest_engine.backtest_runner import run_backtest
from phase_2d_xbid_continuous_intraday.xbid_milp_optimiser.xbid_optimiser import build_xbid_inputs

# 2026-08-21 has confirmed real DA/IDA1/IDA2/IDA3/XBID coverage.
_REAL_ALL_GATES_DATE = "2026-08-21"
# 2025-10-01: real DA, but before every intraday gate's real-data span
# starts (2026-01-01).
_REAL_DA_ONLY_DATE = "2025-10-01"


def test_real_ida_price_xbid_returns_real_value():
    r = real_ida_price("XBID", _REAL_ALL_GATES_DATE, list(range(1, 25)))
    assert r is not None
    assert set(r.keys()) == set(range(1, 25))


def test_real_ida_price_xbid_none_before_real_data_span():
    assert real_ida_price("XBID", _REAL_DA_ONLY_DATE, list(range(1, 25))) is None


def test_build_xbid_inputs_unknown_window_raises(cfg):
    with pytest.raises(ValueError):
        build_xbid_inputs(_REAL_ALL_GATES_DATE, cfg, "BOGUS_WINDOW")


def test_build_xbid_inputs_w1_shape(cfg):
    inputs, open_hours = build_xbid_inputs(_REAL_ALL_GATES_DATE, cfg, "W1")
    assert "da_prices" in inputs and "hours" in inputs
    assert set(open_hours).issubset(set(inputs["hours"]))
    assert len(open_hours) > 0


@pytest.mark.integration
def test_backtest_chains_xbid_after_ida3_on_real_coverage_date(cfg):
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping XBID backtest integration tests")

    res = run_backtest(_REAL_ALL_GATES_DATE, 1, cfg)
    row = res.rows[0]
    assert row["ida3_feasible"] is True
    assert row["xbid_feasible"] is True
    assert row["realised_xbid_price_source"] == "OMIE_LIVE"
    assert row["realised_xbid_revenue_eur"] is not None
    assert res.n_real_xbid_price_days == 1
    assert res.avg_realised_xbid_revenue_eur == pytest.approx(row["realised_xbid_revenue_eur"])


@pytest.mark.integration
def test_backtest_xbid_unavailable_before_real_data_span(cfg):
    """DA reports real while XBID (and every intraday gate before it in
    the chain) independently reports 'unavailable' -- never all-or-nothing,
    same invariant already verified for every other real-price component."""
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping XBID backtest integration tests")

    res = run_backtest(_REAL_DA_ONLY_DATE, 1, cfg)
    row = res.rows[0]
    assert row["feasible"] is True
    assert row["realised_price_source"] == "OMIE_LIVE"
    assert row["realised_xbid_price_source"] == "unavailable"
    assert row["realised_xbid_revenue_eur"] is None
