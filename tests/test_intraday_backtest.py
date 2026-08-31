"""
test_intraday_backtest.py — real-price backtest extension for IDA1
(historical_data_loader.real_ida_price, generalized check_solution_quality,
and backtest_runner.py's IDA1 re-optimization block).
"""
from __future__ import annotations

import pytest

from phase_6_backtesting_and_validation.backtest_engine.historical_data_loader import (
    real_ida_price,
)
from phase_6_backtesting_and_validation.forecast_and_model_validation.milp_solution_quality_checker import (
    check_solution_quality,
)
from phase_6_backtesting_and_validation.backtest_engine.backtest_runner import run_backtest

from tests.conftest import make_inputs

# 2026-08-21 has confirmed real DA (OMIE_LIVE) and real IDA1 (OMIE_LIVE) coverage.
_REAL_DA_AND_IDA1_DATE = "2026-08-21"
# 2026-08-20 has confirmed real DA but IDA1 is one of the genuine gap dates
# (SYNTHETIC) -- same precedent as the aFRR/mFRR backfill gaps.
_REAL_DA_NO_REAL_IDA1_DATE = "2026-08-20"
# 2025-10-01: real DA, but before IDA1's real-data span starts (2026-01-01).
_REAL_DA_PRE_IDA1_DATE = "2025-10-01"
# Genuinely future / never-real date.
_NO_REAL_DATA_DATE = "2030-01-01"


# ── real_ida_price: all-or-nothing behavior ─────────────────────────────────

def test_real_ida_price_returns_real_value_for_known_real_date():
    r = real_ida_price("IDA1", _REAL_DA_AND_IDA1_DATE, list(range(1, 25)))
    assert r is not None
    assert set(r.keys()) == set(range(1, 25))
    assert all(isinstance(v, float) for v in r.values())


def test_real_ida_price_returns_none_for_gap_date():
    assert real_ida_price("IDA1", _REAL_DA_NO_REAL_IDA1_DATE, list(range(1, 25))) is None


def test_real_ida_price_returns_none_for_pre_real_data_date():
    assert real_ida_price("IDA1", _REAL_DA_PRE_IDA1_DATE, list(range(1, 25))) is None


def test_real_ida_price_returns_none_for_future_date():
    assert real_ida_price("IDA1", _NO_REAL_DATA_DATE, list(range(1, 25))) is None


def test_real_ida_price_unknown_gate_raises():
    with pytest.raises(ValueError):
        real_ida_price("NOT_A_REAL_GATE", _REAL_DA_AND_IDA1_DATE, list(range(1, 25)))


# ── check_solution_quality generalization ───────────────────────────────────

@pytest.mark.integration
def test_check_solution_quality_ida1_freezes_fixed_hours(cfg):
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping intraday backtest integration tests")

    inputs = make_inputs(cfg, price_pattern="arbitrage")
    fixed_net = {h: 50.0 for h in range(1, 9)}
    q = check_solution_quality(inputs, cfg, gate="IDA1", fixed_net_position=fixed_net)
    assert q.feasible
    for h, fixed_mw in fixed_net.items():
        assert q.gate_results.net_position_mw[h] == pytest.approx(fixed_mw, abs=1e-3)
    # tradable hours should not all be pinned to the same frozen value
    tradable_vals = [q.gate_results.net_position_mw[h] for h in range(9, 25)]
    assert any(abs(v - 50.0) > 1e-3 for v in tradable_vals)


@pytest.mark.integration
def test_check_solution_quality_da_default_unchanged(cfg):
    """Regression: existing DA-only callers (no new kwargs) must be
    byte-identical in behavior to before this change."""
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping intraday backtest integration tests")

    inputs = make_inputs(cfg, price_pattern="arbitrage")
    q_default = check_solution_quality(inputs, cfg)
    q_explicit = check_solution_quality(inputs, cfg, gate="DA",
                                        fixed_net_position=None,
                                        reserved_up_mw=None, reserved_dn_mw=None)
    assert q_default.feasible == q_explicit.feasible
    assert q_default.objective_eur == pytest.approx(q_explicit.objective_eur, rel=1e-9)
    for h in q_default.gate_results.net_position_mw:
        assert q_default.gate_results.net_position_mw[h] == pytest.approx(
            q_explicit.gate_results.net_position_mw[h], abs=1e-9)


# ── real end-to-end backtest ─────────────────────────────────────────────────

@pytest.mark.integration
def test_backtest_reports_real_ida1_revenue_on_real_coverage_date(cfg):
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping intraday backtest integration tests")

    res = run_backtest(_REAL_DA_AND_IDA1_DATE, 1, cfg)
    row = res.rows[0]
    assert row["feasible"] is True
    assert row["ida1_feasible"] is True
    assert row["realised_ida1_price_source"] == "OMIE_LIVE"
    assert row["realised_ida1_revenue_eur"] is not None
    assert isinstance(row["realised_ida1_revenue_eur"], float)
    assert res.n_real_ida1_price_days == 1
    assert res.avg_realised_ida1_revenue_eur == pytest.approx(row["realised_ida1_revenue_eur"])


@pytest.mark.integration
def test_backtest_da_real_but_ida1_unavailable_reports_independently(cfg):
    """DA revenue must report real while IDA1 independently reports
    'unavailable' -- never all-or-nothing across the row."""
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping intraday backtest integration tests")

    res = run_backtest(_REAL_DA_PRE_IDA1_DATE, 1, cfg)
    row = res.rows[0]
    assert row["feasible"] is True
    assert row["realised_price_source"] == "OMIE_LIVE"
    assert row["realised_revenue_eur"] is not None
    assert row["realised_ida1_price_source"] == "unavailable"
    assert row["realised_ida1_revenue_eur"] is None
