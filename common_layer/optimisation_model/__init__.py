"""
optimisation_model — the shared 24h portfolio MILP.

One model for DA and every IDA gate; build it with the right prices/forecasts
and (optionally) freeze hours an IDA cannot re-trade.

build_core_model_stochastic / extract_stochastic_results extend the DA gate
to a two-stage stochastic MILP (see core_milp_builder.py docstring) — opt-in
via config/stochastic.yaml, no effect on the deterministic path used above.
"""
from common_layer.optimisation_model.core_milp_builder import (
    build_core_model, CoreModelMeta,
    build_core_model_stochastic, StochasticModelMeta,
)
from common_layer.optimisation_model.core_milp_solver import (
    solve_core_model, extract_results, GateResults, SolveError,
    analyze_binding_constraints,
    extract_stochastic_results, StochasticGateResults,
    bridge_stochastic_to_gate_results,
)
from common_layer.optimisation_model.scenario_generator import (
    generate_price_scenarios, load_selected_model_mae, default_scenarios_for_da,
    default_scenarios_for_gate,
)

__all__ = [
    "build_core_model", "CoreModelMeta",
    "build_core_model_stochastic", "StochasticModelMeta",
    "solve_core_model", "extract_results", "GateResults", "SolveError",
    "analyze_binding_constraints",
    "extract_stochastic_results", "StochasticGateResults",
    "bridge_stochastic_to_gate_results",
    "generate_price_scenarios", "load_selected_model_mae", "default_scenarios_for_da",
    "default_scenarios_for_gate",
]
