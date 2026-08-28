"""
test_stochastic_da.py — two-stage stochastic DA MILP (build_core_model_stochastic).

Mirrors the deterministic test pattern in conftest.py (`make_inputs` /
`solved` fixture): CPLEX-availability skip guard, session-scoped solve.

Marked @pytest.mark.integration (like the other full-solve tests) since it
runs the real MILP solver.
"""
from __future__ import annotations

import pytest

from common_layer.optimisation_model import (
    build_core_model, build_core_model_stochastic,
    solve_core_model, extract_results, extract_stochastic_results,
    generate_price_scenarios,
)

from tests.conftest import make_inputs


@pytest.fixture(scope="module")
def stoch_base_inputs(cfg):
    return make_inputs(cfg, price_pattern="arbitrage")


@pytest.fixture(scope="module")
def solved_stochastic(cfg, stoch_base_inputs):
    """Build + solve a small (3-scenario) stochastic DA model once per module.

    Skips the whole module if CPLEX is unavailable — same guard as the
    existing `solved` fixture in conftest.py.
    """
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping stochastic integration tests")

    scenarios, probabilities = generate_price_scenarios(
        stoch_base_inputs["da_prices"], mae=15.0, offsets=[-1.0, 0.0, 1.0])
    model, meta = build_core_model_stochastic(stoch_base_inputs, cfg, scenarios, probabilities)
    solve_core_model(model, cfg, gate="DA")
    results = extract_stochastic_results(model, meta)
    return model, meta, results, scenarios, probabilities


@pytest.mark.integration
def test_stochastic_model_builds_and_solves(solved_stochastic):
    """Model builds and solves without error — the basic smoke test."""
    model, meta, results, scenarios, probabilities = solved_stochastic
    assert len(meta.scenarios) == 3
    assert set(results.net_position_mw.keys()) == set(meta.hours)


@pytest.mark.integration
def test_first_stage_bid_is_scenario_independent(solved_stochastic):
    """The whole point of the formulation: p_net[h] must be ONE shared Var,
    not a scenario-indexed one -- confirm the model object only exposes a
    single value per hour (Pyomo would raise a KeyError on a second index
    if it were accidentally declared scenario-indexed)."""
    model, meta, results, scenarios, probabilities = solved_stochastic
    for h in meta.hours:
        # p_net is declared over m.H only; indexing with a scenario id must fail.
        with pytest.raises((KeyError, Exception)):
            _ = model.p_net[h, meta.scenarios[0]]
        # Plain single-index access must work.
        val = model.p_net[h]
        assert results.net_position_mw[h] == pytest.approx(float(val.value), abs=1e-6)


@pytest.mark.integration
def test_every_scenario_respects_reservoir_and_soc_bounds(solved_stochastic, cfg):
    """Each scenario's independent recourse trajectory must stay physically
    feasible on its own -- a two-stage model that let one scenario cheat
    the reservoir/SoC bounds would be a real, wrong model."""
    model, meta, results, scenarios, probabilities = solved_stochastic
    res = cfg.plant.reservoir
    bess = cfg.plant.bess
    for s in meta.scenarios:
        for h in meta.hours:
            d = results.per_scenario_dispatch[s][h]
            assert res.upper_min_hm3 - 1e-6 <= d["reservoir"]["upper_hm3"] <= res.upper_usable_hm3 + 1e-6
            assert res.lower_min_hm3 - 1e-6 <= d["reservoir"]["lower_hm3"] <= res.lower_capacity_hm3 + 1e-6
            assert bess.e_min_mwh - 1e-6 <= d["bess"]["soc_mwh"] <= bess.e_max_mwh + 1e-6


@pytest.mark.integration
def test_objective_is_probability_weighted_expected_value(solved_stochastic):
    """The reported objective must equal Σ prob[s] * per-scenario contribution
    -- not, say, the best-case or worst-case scenario's value alone."""
    model, meta, results, scenarios, probabilities = solved_stochastic
    from pyomo.environ import value as pyo_value
    expected_energy = sum(
        probabilities[s] * results.per_scenario_revenue_eur[s] for s in meta.scenarios)
    assert results.expected_energy_revenue_eur == pytest.approx(expected_energy, rel=1e-6)
    # Objective includes revenue plus water-value/penalty terms, so it need
    # not equal expected_energy exactly, but it must be a finite real number
    # of the right order of magnitude (sanity bound, not an exact identity).
    assert abs(results.objective_eur) < 1e7


@pytest.mark.integration
def test_single_scenario_degenerates_to_deterministic_bid(cfg, stoch_base_inputs):
    """Regression/sanity check: a 1-scenario stochastic run (scenario = the
    exact point forecast) must reproduce the existing deterministic
    build_core_model's bid within solver tolerance. This is the proof that
    the two-stage formulation is a genuine generalization of the existing,
    already-tested single-scenario model -- not a different model that
    happens to also run."""
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping stochastic integration tests")

    # Deterministic baseline.
    det_model, det_meta = build_core_model(stoch_base_inputs, cfg)
    solve_core_model(det_model, cfg, gate="DA")
    det_results = extract_results(det_model, det_meta)

    # Stochastic, single scenario == the point forecast, probability 1.0.
    scenarios = {0: dict(stoch_base_inputs["da_prices"])}
    probabilities = {0: 1.0}
    stoch_model, stoch_meta = build_core_model_stochastic(
        stoch_base_inputs, cfg, scenarios, probabilities)
    solve_core_model(stoch_model, cfg, gate="DA")
    stoch_results = extract_stochastic_results(stoch_model, stoch_meta)

    for h in det_meta.hours:
        assert stoch_results.net_position_mw[h] == pytest.approx(
            det_results.net_position_mw[h], abs=1e-3), f"bid mismatch at hour {h}"
    assert stoch_results.objective_eur == pytest.approx(det_results.objective_eur, rel=1e-4)
