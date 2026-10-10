"""
test_plant_constraints.py — equation-audit items: initial unit status, minimum down time,
pump start cost and the fixed-speed pump switch (24-hour toy model, CPLEX).
"""
from __future__ import annotations

import dataclasses

import pyomo.environ as pyo
import pytest

from common_layer.optimisation_model import build_core_model, solve_core_model
from tests.conftest import make_inputs


def _psp(cfg, **kw):
    return dataclasses.replace(cfg, plant=dataclasses.replace(
        cfg.plant, psp=dataclasses.replace(cfg.plant.psp, **kw)))


def _solve(cfg, inputs):
    if cfg.solver.resolve_executable() is None:
        pytest.skip("CPLEX not found")
    m, meta = build_core_model(inputs, cfg)
    solve_core_model(m, cfg, gate="DA")
    return m, meta


def _spiky_prices():
    # alternating high / low prices tempt the optimiser to cycle units every hour
    return {h: (160.0 if h % 2 == 0 else 5.0) for h in range(1, 25)}


class TestInitialUnitStatus:
    def _inputs(self, cfg, on):
        prices = {h: (150.0 if h <= 4 else 20.0) for h in range(1, 25)}
        inp = make_inputs(cfg, da_prices_override=prices)
        inp["da_prices"] = prices
        if on:
            inp["initial_state"]["units_on_turb"] = [True, True, True, True]
        return inp

    def test_running_unit_is_not_charged_a_start_in_hour_one(self, cfg):
        m_off, _ = _solve(cfg, self._inputs(cfg, on=False))
        m_on, _ = _solve(cfg, self._inputs(cfg, on=True))
        starts_off = sum(pyo.value(m_off.start_turb[u, 1]) for u in m_off.U)
        starts_on = sum(pyo.value(m_on.start_turb[u, 1]) for u in m_on.U)
        assert starts_off >= 1          # cold plant must start units for the high-price hours
        assert starts_on == 0           # already-running units need no start


class TestMinimumDownTime:
    @staticmethod
    def _gaps(series):
        """Lengths of off-runs that sit between two on-periods."""
        gaps, run, seen_on = [], 0, False
        for v in series:
            if v >= 0.5:
                if seen_on and run:
                    gaps.append(run)
                run, seen_on = 0, True
            elif seen_on:
                run += 1
        return gaps

    def test_off_gaps_respect_the_minimum_down_time(self, cfg):
        c = _psp(cfg, min_down_hours=3, min_mode_hours=1)
        inp = make_inputs(c, da_prices_override=_spiky_prices())
        inp["da_prices"] = _spiky_prices()
        m, _ = _solve(c, inp)
        for var in (m.on_turb, m.on_pump):
            for u in m.U:
                gaps = self._gaps([pyo.value(var[u, h]) for h in m.H])
                assert all(g >= 3 for g in gaps), (u, gaps)

    def test_constraints_exist_only_when_enabled(self, cfg):
        off = _psp(cfg, min_down_hours=1)
        m, _ = build_core_model(make_inputs(off), off)
        assert not hasattr(m, "turb_min_down")
        on = _psp(cfg, min_down_hours=2)
        m2, _ = build_core_model(make_inputs(on), on)
        assert hasattr(m2, "turb_min_down") and hasattr(m2, "pump_min_down")


class TestPumpStartCost:
    def test_pump_start_cost_enters_the_objective(self, cfg):
        c0 = _psp(cfg, startup_cost_pump_eur=0.0, min_mode_hours=1, min_down_hours=1)
        c1 = _psp(cfg, startup_cost_pump_eur=500.0, min_mode_hours=1, min_down_hours=1)
        inp = make_inputs(c0, da_prices_override=_spiky_prices())
        inp["da_prices"] = _spiky_prices()
        m0, _ = _solve(c0, inp)
        m1, _ = _solve(c1, inp)
        obj0 = pyo.value(next(m0.component_data_objects(pyo.Objective, active=True)))
        obj1 = pyo.value(next(m1.component_data_objects(pyo.Objective, active=True)))
        n0 = sum(pyo.value(m0.start_pump[u, h]) for u in m0.U for h in m0.H)
        # a start cost can only lower the optimum, and by at most cost x the free solution's starts
        assert obj1 <= obj0 + 1e-6
        assert obj1 >= obj0 - 500.0 * n0 - 1.0


class TestFixedSpeedPump:
    def test_pumps_run_only_at_maximum_flow(self, cfg):
        c = _psp(cfg, pump_fixed_speed=True)
        inp = make_inputs(c, price_pattern="negative")      # prices below zero force pumping
        m, meta = _solve(c, inp)
        q_max = c.plant.psp.q_pump_max_m3h
        active = [(u, h) for u in m.U for h in m.H if pyo.value(m.on_pump[u, h]) > 0.5]
        assert active, "the negative-price day should pump"
        for u, h in active:
            assert pyo.value(m.q_pump[u, h]) == pytest.approx(q_max, rel=1e-6)

    def test_default_allows_variable_pump_flow(self, cfg):
        assert cfg.plant.psp.pump_fixed_speed is False
