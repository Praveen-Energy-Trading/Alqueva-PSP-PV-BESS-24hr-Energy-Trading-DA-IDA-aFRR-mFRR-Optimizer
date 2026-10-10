"""
test_time_in_state.py — minimum up/down times that straddle midnight: the pipeline carries how
long each unit has already been on or off, and the model enforces the remainder in the first
periods of the next day.
"""
from __future__ import annotations

import dataclasses

import pyomo.environ as pyo
import pytest

from common_layer.database import component_store as cs
from common_layer.database.component_store import ComponentStore
from common_layer.optimisation_model import build_core_model, solve_core_model
from common_layer.optimisation_model.core_milp_builder import _initial_dwell_fixings, _hours_in_state
from tests.conftest import make_inputs

U = [1, 2, 3, 4]


# ---------- pure helper ----------

class TestInitialDwellFixings:
    H = list(range(1, 97))        # 15-minute periods

    def test_running_unit_must_finish_its_minimum_up_time(self):
        # on for 1 period, minimum up 2 h = 8 periods -> 7 more periods forced on
        out = _initial_dwell_fixings(U, self.H, 0.25, 2.0, 2.0, {1: 1, 2: 0, 3: 0, 4: 0},
                                     {1: 0.25, 2: None, 3: None, 4: None})
        assert out == [(1, h, 1) for h in range(1, 8)]

    def test_stopped_unit_must_finish_its_minimum_down_time(self):
        # off for 3 periods, minimum down 2 h = 8 periods -> 5 more periods forced off
        out = _initial_dwell_fixings(U, self.H, 0.25, 2.0, 2.0, {1: 0, 2: 0, 3: 0, 4: 0},
                                     {1: 0.75, 2: None, 3: None, 4: None})
        assert out == [(1, h, 0) for h in range(1, 6)]

    def test_nothing_is_forced_once_the_minimum_time_has_passed(self):
        out = _initial_dwell_fixings(U, self.H, 0.25, 2.0, 2.0, {1: 1, 2: 0, 3: 1, 4: 0},
                                     {1: 2.0, 2: 24.0, 3: 5.25, 4: 5.5})
        assert out == []

    def test_unknown_time_in_state_gives_no_constraint(self):
        assert _initial_dwell_fixings(U, self.H, 0.25, 2.0, 2.0, {u: 1 for u in U}, {u: None for u in U}) == []

    def test_hours_in_state_helper(self):
        assert _hours_in_state([1.5, None], U) == {1: 1.5, 2: None, 3: None, 4: None}
        assert _hours_in_state(None, U) == {u: None for u in U}


# ---------- component store ----------

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_repo_root", lambda: str(tmp_path))
    return ComponentStore()


def _prev_day(store, turb, pump):
    """24 hourly periods; turb / pump are 24-long lists of 4-unit status lists."""
    psp = {h: {"units_on_turb": turb[h - 1], "units_on_pump": pump[h - 1]} for h in range(1, 25)}
    store.save("2026-10-10", psp_schedule=psp,
               bess_schedule={1: {"soc_mwh": 1.0}, 24: {"soc_mwh": 1.0}}, pv_schedule={},
               reservoir_trajectory={1: {"upper_hm3": 2400.0, "lower_hm3": 30.0},
                                     24: {"upper_hm3": 2450.0, "lower_hm3": 40.0}},
               efficiency_per_hour={}, inflow_m3h={})


def test_time_in_state_is_counted_back_from_the_last_period(store):
    # unit 1 turbine: off for 20 h then on for the last 4 h; unit 2: on for 22 h then off for 2 h;
    # unit 3: on all day; unit 4: off all day
    turb = [[1 if h > 20 else 0, 0 if h > 22 else 1, 1, 0] for h in range(1, 25)]
    pump = [[0, 0, 0, 1 if h == 24 else 0] for h in range(1, 25)]
    _prev_day(store, turb, pump)
    s = store.load_chained_initial_state("2026-10-11", 2.0, {"upper_reservoir_hm3": 2490.0,
                                         "lower_reservoir_hm3": 27.0, "bess_soc_frac": 0.5})
    assert s["units_turb_hours_in_state"] == [4.0, 2.0, 24.0, 24.0]
    assert s["units_pump_hours_in_state"] == [24.0, 24.0, 24.0, 1.0]


def test_missing_unit_status_gives_no_time_in_state(store):
    store.save("2026-10-10", psp_schedule={1: {"turbine_mw": 1.0}, 2: {"turbine_mw": 1.0}},
               bess_schedule={1: {"soc_mwh": 1.0}}, pv_schedule={},
               reservoir_trajectory={1: {"upper_hm3": 2400.0, "lower_hm3": 30.0}},
               efficiency_per_hour={}, inflow_m3h={})
    s = store.load_chained_initial_state("2026-10-11", 2.0, {"upper_reservoir_hm3": 2490.0,
                                         "lower_reservoir_hm3": 27.0, "bess_soc_frac": 0.5})
    assert "units_turb_hours_in_state" not in s


# ---------- MILP ----------

def _psp(cfg, **kw):
    return dataclasses.replace(cfg, plant=dataclasses.replace(
        cfg.plant, psp=dataclasses.replace(cfg.plant.psp, **kw)))


def _solve(cfg, inputs):
    if cfg.solver.resolve_executable() is None:
        pytest.skip("CPLEX not found")
    m, _ = build_core_model(inputs, cfg)
    solve_core_model(m, cfg, gate="DA")
    return m


def test_unit_that_just_started_stays_on_for_the_rest_of_its_minimum_up_time(cfg):
    c = _psp(cfg, min_mode_hours=3, min_down_hours=1)
    inp = make_inputs(c, price_pattern="negative")          # turbining loses money all day
    inp["initial_state"].update({"units_on_turb": [1, 0, 0, 0],
                                 "units_turb_hours_in_state": [1.0, None, None, None]})
    m = _solve(c, inp)
    assert pyo.value(m.on_turb[1, 1]) == 1 and pyo.value(m.on_turb[1, 2]) == 1    # 3 h minimum - 1 h done
    free = make_inputs(c, price_pattern="negative")
    free["initial_state"].update({"units_on_turb": [1, 0, 0, 0]})                # time in state unknown
    m2 = _solve(c, free)
    assert pyo.value(m2.on_turb[1, 1]) == 0                                       # no constraint without it


def test_unit_that_just_stopped_stays_off_for_the_rest_of_its_minimum_down_time(cfg):
    c = _psp(cfg, min_down_hours=3, min_mode_hours=1)
    prices = {h: (150.0 if h <= 6 else 20.0) for h in range(1, 25)}
    inp = make_inputs(c, da_prices_override=prices)
    inp["da_prices"] = prices
    inp["initial_state"].update({"units_on_turb": [0, 0, 0, 0],
                                 "units_turb_hours_in_state": [1.0, None, None, None]})
    m = _solve(c, inp)
    assert pyo.value(m.on_turb[1, 1]) == 0 and pyo.value(m.on_turb[1, 2]) == 0    # 3 h minimum - 1 h done
    free = make_inputs(c, da_prices_override=prices)
    free["da_prices"] = prices
    m2 = _solve(c, free)
    assert pyo.value(m2.on_turb[1, 1]) == 1                                       # high price, nothing blocks it


def test_stochastic_model_accepts_the_time_in_state_inputs(cfg):
    from common_layer.optimisation_model import build_core_model_stochastic
    inp = make_inputs(cfg)
    inp["initial_state"].update({"units_on_turb": [1, 0, 0, 0],
                                 "units_turb_hours_in_state": [0.5, None, None, None]})
    scen = {0: dict(inp["da_prices"]), 1: {h: p * 1.2 for h, p in inp["da_prices"].items()}}
    m, _ = build_core_model_stochastic(inp, _psp(cfg, min_mode_hours=2), scen, {0: 0.5, 1: 0.5})
    assert hasattr(m, "init_dwell_turb")
