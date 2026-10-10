"""
test_unit_status_chain.py — each unit's on/off status at the end of the previous day is
carried into the next day's initial state (ComponentStore.load_chained_initial_state).
"""
from __future__ import annotations

import pytest

from common_layer.database import component_store as cs
from common_layer.database.component_store import ComponentStore

DEFAULT = {
    "upper_reservoir_hm3": 2490.0,
    "lower_reservoir_hm3": 27.0,
    "bess_soc_frac": 0.5,
    "units_on_turb": [0, 0, 0, 0],
    "units_on_pump": [0, 0, 0, 0],
}
BOUNDS = {"upper_min_hm3": 2000.0, "upper_usable_hm3": 3000.0,
          "lower_min_hm3": 10.0, "lower_capacity_hm3": 100.0}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_repo_root", lambda: str(tmp_path))
    return ComponentStore()


def _save_prev(store, day, last_psp):
    store.save(
        day,
        psp_schedule={1: {"units_on_turb": [0, 0, 0, 0], "units_on_pump": [0, 0, 0, 0]},
                      2: last_psp},
        bess_schedule={1: {"soc_mwh": 1.0}, 2: {"soc_mwh": 0.5}},
        pv_schedule={}, reservoir_trajectory={1: {"upper_hm3": 2400.0, "lower_hm3": 30.0},
                                              2: {"upper_hm3": 2450.0, "lower_hm3": 40.0}},
        efficiency_per_hour={}, inflow_m3h={},
    )


def _chain(store, day):
    return store.load_chained_initial_state(day, 2.0, dict(DEFAULT), BOUNDS)


def test_last_period_unit_status_is_carried_over(store):
    _save_prev(store, "2026-10-10", {"units_on_turb": [1, 0, 1, 0], "units_on_pump": [0, 0, 0, 0]})
    s = _chain(store, "2026-10-11")
    assert s["units_on_turb"] == [1, 0, 1, 0]
    assert s["units_on_pump"] == [0, 0, 0, 0]
    assert s["upper_reservoir_hm3"] == pytest.approx(2450.0)     # reservoir chaining still works
    assert s["bess_soc_frac"] == pytest.approx(0.25)


def test_pumping_status_is_carried_over(store):
    _save_prev(store, "2026-10-10", {"units_on_turb": [0, 0, 0, 0], "units_on_pump": [1, 1, 0, 0]})
    assert _chain(store, "2026-10-11")["units_on_pump"] == [1, 1, 0, 0]


def test_no_previous_day_falls_back_to_the_default(store):
    s = _chain(store, "2026-10-11")
    assert s == DEFAULT


def test_old_record_without_unit_status_uses_the_default_units(store):
    _save_prev(store, "2026-10-10", {"turbine_mw": 100.0})        # no units_on_* keys
    s = _chain(store, "2026-10-11")
    assert s["units_on_turb"] == DEFAULT["units_on_turb"]
    assert s["units_on_pump"] == DEFAULT["units_on_pump"]
    assert s["upper_reservoir_hm3"] == pytest.approx(2450.0)


def test_unit_status_flows_into_the_milp_start_rule(store, cfg):
    """A unit running at midnight is not charged a start in the first period."""
    if cfg.solver.resolve_executable() is None:
        pytest.skip("CPLEX not found")
    import pyomo.environ as pyo
    from common_layer.optimisation_model import build_core_model, solve_core_model
    from tests.conftest import make_inputs
    _save_prev(store, "2026-10-10", {"units_on_turb": [1, 1, 1, 1], "units_on_pump": [0, 0, 0, 0]})
    chained = _chain(store, "2026-10-11")
    prices = {h: (150.0 if h <= 4 else 20.0) for h in range(1, 25)}
    inp = make_inputs(cfg, da_prices_override=prices)
    inp["da_prices"] = prices
    inp["initial_state"].update({k: chained[k] for k in ("units_on_turb", "units_on_pump")})
    m, _ = build_core_model(inp, cfg)
    solve_core_model(m, cfg, gate="DA")
    assert sum(pyo.value(m.start_turb[u, 1]) for u in m.U) == 0
