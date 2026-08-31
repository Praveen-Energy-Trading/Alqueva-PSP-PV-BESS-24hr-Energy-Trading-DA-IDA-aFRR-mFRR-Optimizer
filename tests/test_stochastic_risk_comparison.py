"""
test_stochastic_risk_comparison.py — real-price backtest comparing
expected-value vs CVaR-averse DA bidding on the IDENTICAL scenario fan,
both valued against the REAL settled price
(backtest_engine.backtest_runner.run_stochastic_risk_comparison).

Uses a module-scoped fixture for the real-price comparison (each solve
takes minutes at real 96-ISP DA scale -- see run_stochastic_risk_comparison's
own docstring for why n_scenarios defaults to 3, not 5) so the expensive
real solve runs ONCE per test session, not once per assertion.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from common_layer.optimisation_model import (
    build_core_model_stochastic, solve_core_model, extract_stochastic_results,
)
from phase_6_backtesting_and_validation.backtest_engine.backtest_runner import (
    run_stochastic_risk_comparison,
)

from tests.conftest import make_inputs

# 2026-08-21 has confirmed real DA price coverage.
_REAL_DA_DATE = "2026-08-21"
# Genuinely future date, beyond the real DA archive's coverage (confirmed
# real DA coverage runs up to 2026-08-28 as of this writing).
_NO_REAL_DA_DATE = "2026-09-10"


@pytest.fixture(scope="module")
def real_comparison(cfg):
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping risk-comparison integration tests")
    return run_stochastic_risk_comparison(_REAL_DA_DATE, 1, cfg, cvar_alpha=0.90)


@pytest.mark.integration
def test_real_price_day_produces_both_strategies_realised(real_comparison):
    res = real_comparison
    assert res.n_real_days == 1
    r = res.rows[0]
    assert r["price_source"] == "OMIE_LIVE"
    assert r["feasible_ev"] is True
    assert r["feasible_cvar"] is True
    assert r["realised_ev_eur"] is not None
    assert r["realised_cvar_eur"] is not None
    assert res.mean_realised_ev_eur is not None
    assert res.mean_realised_cvar_eur is not None
    assert res.risk_ev is not None
    assert res.risk_cvar is not None


@pytest.mark.integration
def test_cvar_strategy_does_not_beat_ev_strategy_on_mean(real_comparison):
    """CVaR trades some expected return for tail protection -- on the same
    scenario fan, its mean realised revenue should not exceed EV's by more
    than the solver's own MIP gap. Exact non-negativity does not hold
    numerically: both solves stop at a 0.5% MIP gap (cfg.solver.mip_gap),
    not exact optimality, so a small crossover within that gap is expected
    solver noise, not a violation of the CVaR<=mean property. Tolerance is
    relative (1%) rather than a tiny absolute constant, since realised
    revenue here is O(1e5)-O(1e6) EUR and a fixed few-EUR tolerance would
    be meaningless at that scale."""
    res = real_comparison
    tolerance = 0.01 * abs(res.mean_realised_ev_eur)
    assert res.cost_of_risk_aversion_eur >= -tolerance


@pytest.mark.integration
def test_unavailable_date_is_skipped_not_padded(cfg):
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping risk-comparison integration tests")

    res = run_stochastic_risk_comparison(_NO_REAL_DA_DATE, 1, cfg)
    assert res.n_real_days == 0
    assert res.rows[0]["price_source"] == "unavailable"
    assert res.mean_realised_ev_eur is None
    assert res.mean_realised_cvar_eur is None


@pytest.mark.integration
def test_ev_and_cvar_bids_identical_on_single_scenario(cfg):
    """Degenerate case: with only ONE scenario (no tail to protect
    against), the CVaR-averse objective must reduce to the same bid as
    expected-value -- proof CVaR only changes behavior when there is
    genuine scenario dispersion, not an arbitrary shift. Uses the small
    24-hour toy fixture (make_inputs), not real 96-ISP data, since this is
    a fast formulation sanity check, not a real-outcome measurement."""
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping risk-comparison integration tests")

    inputs = make_inputs(cfg, price_pattern="arbitrage")
    scenarios = {0: dict(inputs["da_prices"])}
    probabilities = {0: 1.0}

    ev_cfg = replace(cfg, stochastic=replace(cfg.stochastic, risk_measure="expected_value"))
    cvar_cfg = replace(cfg, stochastic=replace(cfg.stochastic, risk_measure="cvar", cvar_alpha=0.90))

    ev_model, ev_meta = build_core_model_stochastic(inputs, ev_cfg, scenarios, probabilities)
    solve_core_model(ev_model, ev_cfg, gate="DA")
    ev_results = extract_stochastic_results(ev_model, ev_meta)

    cvar_model, cvar_meta = build_core_model_stochastic(inputs, cvar_cfg, scenarios, probabilities)
    solve_core_model(cvar_model, cvar_cfg, gate="DA")
    cvar_results = extract_stochastic_results(cvar_model, cvar_meta)

    for h in ev_meta.hours:
        assert ev_results.net_position_mw[h] == pytest.approx(
            cvar_results.net_position_mw[h], abs=1e-3), f"bid mismatch at hour {h}"
