"""
test_stochastic_ida1.py — two-stage stochastic IDA1 MILP
(build_core_model_stochastic extended with fixed_net_position).

Mirrors tests/test_stochastic_da.py's fixture/skip/mark pattern, but
exercises the ida_reoptimiser-specific need: hours OUTSIDE the tradable
window must be frozen to a committed baseline (INV-11), identically across
every price scenario, since p_net[h] has no scenario index.
"""
from __future__ import annotations

import pytest

from common_layer.optimisation_model import (
    build_core_model, build_core_model_stochastic,
    solve_core_model, extract_results, extract_stochastic_results,
    generate_price_scenarios, bridge_stochastic_to_gate_results,
)

from tests.conftest import make_inputs

# Simulate IDA1's frozen pre-gate hours: the first 8 hours are already
# committed from DA and cannot be re-traded; only hours 9-24 are tradable.
_FIXED_NET = {h: 50.0 for h in range(1, 9)}


@pytest.fixture(scope="module")
def stoch_ida1_inputs(cfg):
    return make_inputs(cfg, price_pattern="arbitrage")


@pytest.fixture(scope="module")
def solved_stochastic_ida1(cfg, stoch_ida1_inputs):
    """Build + solve a small (3-scenario) stochastic IDA1-shaped model once
    per module, with a frozen pre-gate window. Skips if CPLEX unavailable."""
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping stochastic integration tests")

    scenarios, probabilities = generate_price_scenarios(
        stoch_ida1_inputs["da_prices"], mae=15.0, offsets=[-1.0, 0.0, 1.0])
    model, meta = build_core_model_stochastic(
        stoch_ida1_inputs, cfg, scenarios, probabilities,
        fixed_net_position=_FIXED_NET)
    solve_core_model(model, cfg, gate="IDA1")
    results = extract_stochastic_results(model, meta)
    return model, meta, results, scenarios, probabilities


@pytest.mark.integration
def test_stochastic_ida1_model_builds_and_solves(solved_stochastic_ida1):
    """Model builds and solves without error — the basic smoke test."""
    model, meta, results, scenarios, probabilities = solved_stochastic_ida1
    assert len(meta.scenarios) == 3
    assert set(results.net_position_mw.keys()) == set(meta.hours)


@pytest.mark.integration
def test_frozen_hours_match_fixed_net_position_in_every_scenario(solved_stochastic_ida1):
    """The whole point of adding fixed_net_position to the stochastic
    builder: frozen hours must equal the committed baseline exactly, and
    since p_net[h] has no scenario index, that holds identically for every
    scenario's recourse dispatch (there is only one p_net[h] to check, but
    confirm every scenario's implied dispatch nets to that same fixed value)."""
    model, meta, results, scenarios, probabilities = solved_stochastic_ida1
    for h, fixed_mw in _FIXED_NET.items():
        assert results.net_position_mw[h] == pytest.approx(fixed_mw, abs=1e-3), (
            f"hour {h} should be frozen at {fixed_mw} MW, got {results.net_position_mw[h]}")
    # Tradable hours (9-24) must NOT all equal the fixed value (a degenerate
    # freeze-everything bug would silently pass the check above too).
    tradable_vals = [results.net_position_mw[h] for h in meta.hours if h not in _FIXED_NET]
    assert any(abs(v - 50.0) > 1e-3 for v in tradable_vals), (
        "tradable hours look frozen too -- fixed_net_position may be over-applying")


@pytest.mark.integration
def test_every_scenario_respects_reservoir_and_soc_bounds(solved_stochastic_ida1, cfg):
    """Each scenario's independent recourse trajectory must stay physically
    feasible on its own, even with a frozen pre-gate window."""
    model, meta, results, scenarios, probabilities = solved_stochastic_ida1
    res = cfg.plant.reservoir
    bess = cfg.plant.bess
    for s in meta.scenarios:
        for h in meta.hours:
            d = results.per_scenario_dispatch[s][h]
            assert res.upper_min_hm3 - 1e-6 <= d["reservoir"]["upper_hm3"] <= res.upper_usable_hm3 + 1e-6
            assert res.lower_min_hm3 - 1e-6 <= d["reservoir"]["lower_hm3"] <= res.lower_capacity_hm3 + 1e-6
            assert bess.e_min_mwh - 1e-6 <= d["bess"]["soc_mwh"] <= bess.e_max_mwh + 1e-6


@pytest.mark.integration
def test_single_scenario_with_fixed_net_degenerates_to_deterministic(cfg, stoch_ida1_inputs):
    """Regression/sanity check: a 1-scenario stochastic run (scenario = the
    point forecast) with the same fixed_net_position must reproduce
    build_core_model's fixed_net_position bid within solver tolerance --
    proof the frozen-hour mechanism means the same thing in both models."""
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping stochastic integration tests")

    det_model, det_meta = build_core_model(
        stoch_ida1_inputs, cfg, fixed_net_position=_FIXED_NET)
    solve_core_model(det_model, cfg, gate="IDA1")
    det_results = extract_results(det_model, det_meta)

    scenarios = {0: dict(stoch_ida1_inputs["da_prices"])}
    probabilities = {0: 1.0}
    stoch_model, stoch_meta = build_core_model_stochastic(
        stoch_ida1_inputs, cfg, scenarios, probabilities,
        fixed_net_position=_FIXED_NET)
    solve_core_model(stoch_model, cfg, gate="IDA1")
    stoch_results = extract_stochastic_results(stoch_model, stoch_meta)

    for h in det_meta.hours:
        assert stoch_results.net_position_mw[h] == pytest.approx(
            det_results.net_position_mw[h], abs=1e-3), f"bid mismatch at hour {h}"
    assert stoch_results.objective_eur == pytest.approx(det_results.objective_eur, rel=1e-4)


@pytest.mark.integration
def test_bridge_stochastic_to_gate_results_shape(solved_stochastic_ida1):
    """bridge_stochastic_to_gate_results (moved to core_milp_solver.py so
    ida_reoptimiser.py can reuse it) must produce a GateResults with the
    same fields/shape a deterministic IDA1 solve would."""
    model, meta, results, scenarios, probabilities = solved_stochastic_ida1
    gr = bridge_stochastic_to_gate_results(results, meta)
    assert set(gr.net_position_mw.keys()) == set(meta.hours)
    assert set(gr.da_bids.keys()) == set(meta.hours)
    assert set(gr.psp_schedule.keys()) == set(meta.hours)
    assert set(gr.bess_schedule.keys()) == set(meta.hours)
    assert set(gr.pv_schedule.keys()) == set(meta.hours)
    assert set(gr.reservoir_trajectory.keys()) == set(meta.hours)
    # First-stage bid is the shared, scenario-independent decision -- must
    # pass through unchanged from the stochastic extraction.
    for h in meta.hours:
        assert gr.net_position_mw[h] == pytest.approx(results.net_position_mw[h], abs=1e-9)
    for h, fixed_mw in _FIXED_NET.items():
        assert gr.net_position_mw[h] == pytest.approx(fixed_mw, abs=1e-3)
