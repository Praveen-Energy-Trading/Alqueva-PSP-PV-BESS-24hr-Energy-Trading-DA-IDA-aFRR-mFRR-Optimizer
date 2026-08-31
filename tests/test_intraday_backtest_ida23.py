"""
test_intraday_backtest_ida23.py — real-price backtest extension to IDA2/IDA3
(chained DA -> IDA1 -> IDA2 -> IDA3, backtest_engine.backtest_runner's
_run_intraday_gate_backtest helper and historical_data_loader.real_ida_price
generalized to "IDA2"/"IDA3").
"""
from __future__ import annotations

import pytest

from phase_6_backtesting_and_validation.backtest_engine.historical_data_loader import (
    real_ida_price,
)
from phase_6_backtesting_and_validation.backtest_engine.backtest_runner import run_backtest

# 2026-08-21/22 have confirmed real DA + IDA1 + IDA2 + IDA3 coverage (all
# four archives overlap on these dates).
_REAL_ALL_GATES_DATE = "2026-08-21"
# 2026-08-01: real DA/IDA1/IDA2 coverage but before IDA3's real-data span
# starts (2026-08-15) -- IDA3 must independently report "unavailable".
_REAL_NO_IDA3_DATE = "2026-08-01"


def test_real_ida_price_ida2_returns_real_value():
    r = real_ida_price("IDA2", _REAL_ALL_GATES_DATE, list(range(1, 25)))
    assert r is not None
    assert set(r.keys()) == set(range(1, 25))


def test_real_ida_price_ida3_returns_real_value_on_covered_date():
    r = real_ida_price("IDA3", _REAL_ALL_GATES_DATE, list(range(1, 25)))
    assert r is not None
    assert set(r.keys()) == set(range(1, 25))


def test_real_ida_price_ida3_none_before_real_data_span():
    assert real_ida_price("IDA3", _REAL_NO_IDA3_DATE, list(range(1, 25))) is None


@pytest.mark.integration
def test_backtest_chains_all_three_intraday_gates_on_real_coverage_date(cfg):
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping intraday backtest integration tests")

    res = run_backtest(_REAL_ALL_GATES_DATE, 1, cfg)
    row = res.rows[0]
    assert row["feasible"] is True
    for gate in ("ida1", "ida2", "ida3"):
        assert row[f"{gate}_feasible"] is True
        assert row[f"realised_{gate}_revenue_eur"] is not None
        assert row[f"realised_{gate}_price_source"] == "OMIE_LIVE"
    assert res.n_real_ida1_price_days == 1
    assert res.n_real_ida2_price_days == 1
    assert res.n_real_ida3_price_days == 1


@pytest.mark.integration
def test_backtest_ida3_unavailable_independently_before_its_real_span(cfg):
    """DA/IDA1/IDA2 must report real while IDA3 independently reports
    'unavailable' (pre-real-data date) -- never all-or-nothing across gates,
    same invariant already verified for DA vs reserve capacity vs IDA1."""
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping intraday backtest integration tests")

    res = run_backtest(_REAL_NO_IDA3_DATE, 1, cfg)
    row = res.rows[0]
    assert row["feasible"] is True
    assert row["realised_price_source"] == "OMIE_LIVE"
    assert row["ida1_feasible"] is True
    assert row["realised_ida1_price_source"] == "OMIE_LIVE"
    assert row["ida2_feasible"] is True
    assert row["realised_ida2_price_source"] == "OMIE_LIVE"
    # IDA3 chain still solves (IDA2 was feasible) but has no real price yet.
    assert row["ida3_feasible"] is True
    assert row["realised_ida3_price_source"] == "unavailable"
    assert row["realised_ida3_revenue_eur"] is None
