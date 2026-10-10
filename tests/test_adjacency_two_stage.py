"""
test_adjacency_two_stage.py — models with SOS2 adjacency sets are solved in two stages
(relaxed solve, fix the on/off decisions, then solve with SOS2). Regression tests for the
IDA3 failure (frozen hours rejected by CPLEX presolve) and the safe fallback.
"""
from __future__ import annotations

import dataclasses

import pyomo.environ as pyo
import pytest

import common_layer.optimisation_model.core_milp_solver as S
from common_layer.optimisation_model import build_core_model, solve_core_model
from tests.conftest import make_inputs


def _cfg(cfg, **solver_kw):
    return dataclasses.replace(cfg, solver=dataclasses.replace(cfg.solver, **solver_kw))


@pytest.fixture(scope="module")
def head_cfg(cfg):
    if cfg.solver.resolve_executable() is None:
        pytest.skip("CPLEX not found")
    return _cfg(cfg, efficiency_adjacency="head", adjacency_solve="two_stage")


def _inputs(cfg):
    prices = {h: (140.0 + 8 * (h % 5) if 8 <= h <= 21 else 15.0) for h in range(1, 25)}
    inp = make_inputs(cfg, da_prices_override=prices)
    inp["da_prices"] = prices
    return inp


def _obj(m):
    return pyo.value(next(m.component_data_objects(pyo.Objective, active=True)))


def test_two_stage_leaves_binaries_unfixed_and_sos_active(head_cfg):
    m, _ = build_core_model(_inputs(head_cfg), head_cfg)
    solve_core_model(m, head_cfg, gate="DA")
    assert not any(v.fixed for v in m.component_data_objects(pyo.Var))
    assert all(c.active for c in m.component_objects(pyo.SOSConstraint))


def test_a_solved_schedule_can_be_frozen_into_the_same_model(head_cfg):
    """Regression for the IDA3 failure: freezing a model's own first periods must stay feasible."""
    inp = _inputs(head_cfg)
    m, _ = build_core_model(inp, head_cfg)
    solve_core_model(m, head_cfg, gate="DA")
    frozen = {h: pyo.value(m.p_net[h]) for h in (1, 2, 3, 4, 5, 6)}
    m2, _ = build_core_model(inp, head_cfg, fixed_net_position=frozen)
    solve_core_model(m2, head_cfg, gate="IDA3")          # must not raise SolveError
    for h, v in frozen.items():
        assert pyo.value(m2.p_net[h]) == pytest.approx(v, abs=1e-3)


def test_two_stage_objective_is_close_to_the_joint_solve(head_cfg):
    inp = _inputs(head_cfg)
    m1, _ = build_core_model(inp, head_cfg)
    solve_core_model(m1, head_cfg, gate="DA")
    joint_cfg = _cfg(head_cfg, adjacency_solve="joint")
    m2, _ = build_core_model(inp, joint_cfg)
    solve_core_model(m2, joint_cfg, gate="DA")
    assert abs(_obj(m1) - _obj(m2)) / abs(_obj(m2)) < 0.01


def test_failed_second_stage_keeps_the_first_stage_solution(head_cfg, monkeypatch, capsys):
    real = S._run_solver
    calls = {"n": 0}

    def flaky(model, opt, name, cfg, gate, tl=None, has_sos=False):
        calls["n"] += 1
        if has_sos:
            raise S.SolveError("simulated stage-2 failure")
        return real(model, opt, name, cfg, gate, tl=tl, has_sos=has_sos)

    monkeypatch.setattr(S, "_run_solver", flaky)
    m, _ = build_core_model(_inputs(head_cfg), head_cfg)
    solve_core_model(m, head_cfg, gate="DA")             # must not raise
    assert calls["n"] == 2
    assert "adjacency stage failed" in capsys.readouterr().out
    assert pyo.value(m.p_net[10]) is not None             # stage-1 values are in the model
    assert not any(v.fixed for v in m.component_data_objects(pyo.Var))


def test_models_without_sos_take_the_normal_path(cfg):
    if cfg.solver.resolve_executable() is None:
        pytest.skip("CPLEX not found")
    off = _cfg(cfg, efficiency_adjacency="off")
    m, _ = build_core_model(make_inputs(off), off)
    assert not list(m.component_objects(pyo.SOSConstraint))
    solve_core_model(m, off, gate="DA")
