"""
test_cvar_stochastic.py — CVaR-averse objective for build_core_model_stochastic
(Rockafellar-Uryasev linearization, cfg.stochastic.risk_measure == "cvar").

Mirrors tests/test_stochastic_da.py's fixture/skip/mark pattern. Uses a
deliberately SKEWED scenario fan (not the symmetric default) so
expected-value and CVaR optimization can genuinely diverge -- a symmetric
fan makes both objectives numerically indistinguishable, which would prove
nothing about whether the new code path actually changes behavior.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from common_layer.optimisation_model import (
    build_core_model_stochastic, solve_core_model, extract_stochastic_results,
)

from tests.conftest import make_inputs

# Skewed fan: a long, deep downside tail (-40, -25, -10) and a shallow
# upside (+5, +10) around the point forecast -- CVaR at alpha=0.90 should
# weight the deep-downside scenarios much more than expected value does.
_SKEWED_OFFSETS_EUR = [-40.0, -25.0, -10.0, 5.0, 10.0]


def _skewed_scenarios(point_forecast):
    n = len(_SKEWED_OFFSETS_EUR)
    scenarios = {sid: {h: max(0.0, p + off) for h, p in point_forecast.items()}
                 for sid, off in enumerate(_SKEWED_OFFSETS_EUR)}
    probabilities = {sid: 1.0 / n for sid in scenarios}
    return scenarios, probabilities


@pytest.fixture(scope="module")
def cvar_base_inputs(cfg):
    return make_inputs(cfg, price_pattern="arbitrage")


def _cfg_with_risk(cfg, risk_measure, cvar_alpha=0.90):
    return replace(cfg, stochastic=replace(
        cfg.stochastic, risk_measure=risk_measure, cvar_alpha=cvar_alpha))


@pytest.fixture(scope="module")
def solved_cvar(cfg, cvar_base_inputs):
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping CVaR integration tests")

    cvar_cfg = _cfg_with_risk(cfg, "cvar", cvar_alpha=0.90)
    scenarios, probabilities = _skewed_scenarios(cvar_base_inputs["da_prices"])
    model, meta = build_core_model_stochastic(cvar_base_inputs, cvar_cfg, scenarios, probabilities)
    solve_core_model(model, cvar_cfg, gate="DA")
    results = extract_stochastic_results(model, meta)
    return model, meta, results, scenarios, probabilities


@pytest.fixture(scope="module")
def solved_expected_value(cfg, cvar_base_inputs):
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping CVaR integration tests")

    ev_cfg = _cfg_with_risk(cfg, "expected_value")
    scenarios, probabilities = _skewed_scenarios(cvar_base_inputs["da_prices"])
    model, meta = build_core_model_stochastic(cvar_base_inputs, ev_cfg, scenarios, probabilities)
    solve_core_model(model, ev_cfg, gate="DA")
    results = extract_stochastic_results(model, meta)
    return model, meta, results


@pytest.mark.integration
def test_cvar_model_builds_and_solves(solved_cvar):
    model, meta, results, scenarios, probabilities = solved_cvar
    assert meta.risk_measure == "cvar"
    assert meta.cvar_alpha == pytest.approx(0.90)
    assert results.risk_measure == "cvar"
    assert results.cvar_eur is not None
    assert results.eta_eur is not None


@pytest.mark.integration
def test_cvar_does_not_exceed_expected_value(solved_cvar):
    """Mathematical invariant, not an assumption about this model: CVaR of a
    distribution is always <= its mean. If this fails, the formulation is
    wrong, not just "different"."""
    model, meta, results, scenarios, probabilities = solved_cvar
    assert results.cvar_eur <= results.expected_energy_revenue_eur + 1e-3 or (
        # expected_energy_revenue_eur is energy-only; objective_eur / cvar_eur
        # include water value & penalties too, so compare against the full
        # probability-weighted profit instead for an apples-to-apples check.
        results.cvar_eur <= sum(
            probabilities[s] * results.per_scenario_profit_eur[s] for s in meta.scenarios) + 1e-3
    )


@pytest.mark.integration
def test_shortfall_constraints_hold_and_bind_on_worst_scenario(solved_cvar):
    """cvar_shortfall[s] >= max(0, eta - profit[s]) must hold for every
    scenario (the LP constraint the solver enforced), and at least one
    scenario -- the worst one(s) at this alpha -- must be (near) binding."""
    model, meta, results, scenarios, probabilities = solved_cvar
    from pyomo.environ import value as pyo_value
    eta = results.eta_eur
    slacks = []
    for s in meta.scenarios:
        shortfall = pyo_value(model.cvar_shortfall[s])
        profit = results.per_scenario_profit_eur[s]
        required = max(0.0, eta - profit)
        assert shortfall >= required - 1e-2, (
            f"scenario {s}: shortfall {shortfall} < required {required}")
        slacks.append(abs(shortfall - required))
    assert min(slacks) < 1.0, "no scenario is binding -- CVaR formulation may be inactive"


@pytest.mark.integration
def test_cvar_bid_diverges_from_expected_value_bid(solved_cvar, solved_expected_value):
    """Under the skewed fan, CVaR-mode and expected-value-mode must produce
    genuinely different first-stage bids -- proof the objective actually
    changes optimizer behavior, not just cosmetic new fields."""
    _, cvar_meta, cvar_results, _, _ = solved_cvar
    _, ev_meta, ev_results = solved_expected_value
    diffs = [abs(cvar_results.net_position_mw[h] - ev_results.net_position_mw[h])
             for h in cvar_meta.hours]
    assert max(diffs) > 0.5, (
        "CVaR and expected-value bids are numerically identical -- the risk "
        "measure had no effect on the skewed fan, formulation likely inert")


@pytest.mark.integration
def test_invalid_risk_measure_raises(cfg, cvar_base_inputs):
    bad_cfg = _cfg_with_risk(cfg, "bogus")
    scenarios, probabilities = _skewed_scenarios(cvar_base_inputs["da_prices"])
    with pytest.raises(ValueError):
        build_core_model_stochastic(cvar_base_inputs, bad_cfg, scenarios, probabilities)


@pytest.mark.integration
@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, 1.5, -0.1])
def test_invalid_cvar_alpha_raises(cfg, cvar_base_inputs, bad_alpha):
    bad_cfg = _cfg_with_risk(cfg, "cvar", cvar_alpha=bad_alpha)
    scenarios, probabilities = _skewed_scenarios(cvar_base_inputs["da_prices"])
    with pytest.raises(ValueError):
        build_core_model_stochastic(cvar_base_inputs, bad_cfg, scenarios, probabilities)


@pytest.mark.integration
def test_expected_value_default_unchanged(cfg, cvar_base_inputs):
    """Regression: with risk_measure='expected_value' (the shipped default),
    behavior must be identical to before this change -- the objective must
    equal the plain probability-weighted sum, with no eta/cvar_shortfall
    vars created at all."""
    exe = cfg.solver.resolve_executable()
    if exe is None:
        pytest.skip("CPLEX not found — skipping CVaR integration tests")

    ev_cfg = _cfg_with_risk(cfg, "expected_value")
    scenarios, probabilities = _skewed_scenarios(cvar_base_inputs["da_prices"])
    model, meta = build_core_model_stochastic(cvar_base_inputs, ev_cfg, scenarios, probabilities)
    solve_core_model(model, ev_cfg, gate="DA")
    results = extract_stochastic_results(model, meta)

    assert not hasattr(model, "eta")
    assert not hasattr(model, "cvar_shortfall")
    assert results.risk_measure == "expected_value"
    assert results.eta_eur is None
    assert results.cvar_eur is None
    expected = sum(probabilities[s] * results.per_scenario_profit_eur[s] for s in meta.scenarios)
    assert results.objective_eur == pytest.approx(expected, rel=1e-6)
