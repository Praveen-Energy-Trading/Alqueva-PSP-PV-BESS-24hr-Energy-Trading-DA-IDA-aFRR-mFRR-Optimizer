"""
test_efficiency_adjacency.py — the turbine efficiency-surface interpolation weights must
use neighbouring head grid points only (SOS2 on the head axis, CPLEX); "full" adds both
axes for turbine and pump.

Without that rule the optimiser could blend non-neighbouring grid points and read off
more turbine power per m3 of water than the efficiency surface allows (about +2.6% on
a heavy generation day). These tests solve a real day and check the weights and the
resulting power against the continuous surface.
"""
from __future__ import annotations

import dataclasses

import pyomo.environ as pyo
import pytest

import common_layer.optimisation_model.core_milp_builder as C
from common_layer.optimisation_model import build_core_model, solve_core_model
from common_layer.optimisation_model.core_milp_builder import _adjacency_form
from tests.conftest import make_inputs


@pytest.fixture(scope="module")
def high_spread_solve(cfg):
    """A day with a deep price spread, so the plant generates at many flow levels."""
    if cfg.solver.resolve_executable() is None:
        pytest.skip("CPLEX not found")
    prices = {h: (140.0 + 8 * (h % 5) if 8 <= h <= 21 else 15.0) for h in range(1, 25)}
    inputs = make_inputs(cfg, da_prices_override=prices)
    inputs["da_prices"] = prices
    head_cfg = _with(cfg, "head")
    model, meta = build_core_model(inputs, head_cfg)
    solve_core_model(model, head_cfg, gate="DA")
    return model, meta


def _with(cfg, form):
    return dataclasses.replace(cfg, solver=dataclasses.replace(cfg.solver, efficiency_adjacency=form))


def _support(model, omega, u, h):
    return {(fi, hi) for fi in model.FI for hi in model.HI
            if pyo.value(omega[u, fi, hi, h]) > 1e-6}


class TestAdjacencyRule:
    def test_default_is_auto_which_means_head_with_cplex(self, cfg):
        assert cfg.solver.efficiency_adjacency == "auto"
        if cfg.solver.resolve_executable() is None:
            pytest.skip("CPLEX not found")
        assert _adjacency_form(cfg) == "head"

    def test_auto_turns_the_rule_on_with_cplex(self, cfg):
        if cfg.solver.resolve_executable() is None:
            pytest.skip("CPLEX not found")
        assert _adjacency_form(_with(cfg, "auto")) == "head"

    def test_invalid_setting_is_rejected(self, cfg):
        bad = dataclasses.replace(cfg, solver=dataclasses.replace(cfg.solver, efficiency_adjacency="nonsense"))
        with pytest.raises(ValueError):
            _adjacency_form(bad)

    def test_off_builds_no_sos_sets(self, cfg):
        off = _with(cfg, "off")
        model, _ = build_core_model(make_inputs(off), off)
        for name in ("sosf_trb", "sosh_trb", "sosf_pmp", "sosh_pmp"):
            assert not hasattr(model, name), name

    def test_model_contains_the_sos2_sets(self, cfg):
        if cfg.solver.resolve_executable() is None:
            pytest.skip("CPLEX not found")
        head = _with(cfg, "head")
        model, _ = build_core_model(make_inputs(head), head)
        assert hasattr(model, "sosh_trb")                       # turbine, head axis
        for name in ("sosf_trb", "sosf_pmp", "sosh_pmp"):       # not in the default
            assert not hasattr(model, name), name

    def test_full_form_adds_both_axes_for_turbine_and_pump(self, cfg):
        if cfg.solver.resolve_executable() is None:
            pytest.skip("CPLEX not found")
        full = _with(cfg, "full")
        model, _ = build_core_model(make_inputs(full), full)
        for name in ("sosf_trb", "sosh_trb", "sosf_pmp", "sosh_pmp"):
            assert hasattr(model, name), name

    def test_turbine_head_weights_are_neighbours(self, high_spread_solve):
        model, _ = high_spread_solve
        checked = 0
        for u in model.U:
            for h in model.H:
                s = _support(model, model.omega_trb, u, h)
                if not s:
                    continue
                checked += 1
                his = {b for _, b in s}
                assert max(his) - min(his) <= 1, (u, h, s)
        assert checked > 0

    def test_turbine_power_matches_the_continuous_surface(self, high_spread_solve):
        """Power read from the weights vs the efficiency surface at the same flow and head."""
        model, meta = high_spread_solve
        from common_layer.optimisation_model import core_milp_builder as B
        ratios = []
        for u in model.U:
            for h in model.H:
                w = {(fi, hi): pyo.value(model.omega_trb[u, fi, hi, h]) for fi in model.FI for hi in model.HI}
                if sum(w.values()) < 0.5:
                    continue
                q = sum(v * meta.flow_grid_trb[fi] for (fi, hi), v in w.items())
                hd = sum(v * meta.head_grid[hi] for (fi, hi), v in w.items())
                p_model = sum(v * meta.eff_trb[(fi, hi)] * B.RHO_WATER * B.G_GRAVITY
                              * meta.flow_grid_trb[fi] * meta.head_grid[hi] / B.CONV_M3H_TO_MW
                              for (fi, hi), v in w.items())
                a0, a1, a2, a3, a4, a5 = B.COEFFS_TRB
                fg, hg = meta.flow_grid_trb, meta.head_grid
                fn = (q - fg[0]) / (fg[-1] - fg[0])
                hn = (hd - hg[0]) / (hg[-1] - hg[0])
                eta = max(B.ETA_LO, min(a0 + a1*fn + a2*hn + a3*fn*hn + a4*fn**2 + a5*hn**2, B.ETA_HI))
                p_true = eta * B.RHO_WATER * B.G_GRAVITY * q * hd / B.CONV_M3H_TO_MW
                ratios.append(p_model / p_true)
        assert ratios, "no turbine unit-hours were active"
        # Before the fix the average overstatement was +2.6% (max +3.0%) on a heavy day.
        assert sum(ratios) / len(ratios) < 1.005
        assert max(ratios) < 1.015
