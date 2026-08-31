"""
test_live_bid_resettlement.py — store-free "live-bid real-price
re-settlement" (backtest_engine.live_bid_resettlement): value the ACTUAL
historically-committed position against REAL archived prices, no re-solve.

No CPLEX/integration marker needed -- there is no MILP build/solve in this
module at all, by design.
"""
from __future__ import annotations

import pytest

from phase_6_backtesting_and_validation.backtest_engine.live_bid_resettlement import (
    resettle_live_bid, _isp_delta_revenue, _prior_gate,
)

# 2026-08-21 has confirmed real DA + IDA1 + IDA2 + IDA3 + aFRR + mFRR
# coverage (established earlier this session, cross-checked against
# run_status_*.json in the exploration that scoped this feature).
_REAL_ALL_GATES_DATE = "2026-08-21"
# A genuinely future/never-real date.
_NO_REAL_DATA_DATE = "2030-01-01"


# ── _prior_gate ──────────────────────────────────────────────────────────

def test_prior_gate_chain():
    assert _prior_gate("DA") is None
    assert _prior_gate("IDA1") == "DA"
    assert _prior_gate("IDA2") == "IDA1"
    assert _prior_gate("IDA3") == "IDA2"
    assert _prior_gate("XBID") == "IDA3"


# ── _isp_delta_revenue: matches dispatch_sheet_builder.sum_delta_revenue math ──

def test_isp_delta_revenue_settles_incremental_volume_at_real_price():
    # ISP 1: DA committed 10 MW @ forecast 40 EUR/MWh; IDA1 re-trades to 15 MW.
    cur_pos = {1: {"volume_mwh": 15.0, "price_eur_mwh": 45.0}}
    prior_pos = {1: {"volume_mwh": 10.0}}
    real_price = {1: 50.0}
    realised, forecast, committed_mwh = _isp_delta_revenue(cur_pos, prior_pos, real_price)
    # delta = 15 - 10 = 5 MW
    assert forecast == pytest.approx(45.0 * 5.0)
    assert realised == pytest.approx(50.0 * 5.0)
    assert committed_mwh == pytest.approx(5.0)


def test_isp_delta_revenue_isp_with_no_real_price_contributes_zero_to_realised():
    cur_pos = {1: {"volume_mwh": 5.0, "price_eur_mwh": 40.0},
               2: {"volume_mwh": 8.0, "price_eur_mwh": 42.0}}
    prior_pos = {}
    real_price = {1: 55.0}   # ISP 2 has no real price
    realised, forecast, committed_mwh = _isp_delta_revenue(cur_pos, prior_pos, real_price)
    assert realised == pytest.approx(55.0 * 5.0)   # only ISP 1 counted
    assert forecast == pytest.approx(40.0 * 5.0 + 42.0 * 8.0)   # both counted
    assert committed_mwh == pytest.approx(13.0)


def test_isp_delta_revenue_negative_delta_pumping():
    cur_pos = {1: {"volume_mwh": -20.0, "price_eur_mwh": 10.0}}
    prior_pos = {1: {"volume_mwh": 0.0}}
    real_price = {1: 12.0}
    realised, forecast, committed_mwh = _isp_delta_revenue(cur_pos, prior_pos, real_price)
    assert forecast == pytest.approx(10.0 * -20.0)
    assert realised == pytest.approx(12.0 * -20.0)
    assert committed_mwh == pytest.approx(20.0)


# ── resettle_live_bid: real end-to-end against a real-coverage date ──────

def test_resettle_live_bid_no_commitments_returns_empty_result(monkeypatch, tmp_path):
    """A date with nothing in PositionStore/ReserveStore at all must return
    an empty-but-valid result, not raise."""
    import phase_6_backtesting_and_validation.backtest_engine.live_bid_resettlement as mod

    class _EmptyPositionStore:
        def load_position(self, delivery_date, gate):
            return {}
        def committed_position(self, delivery_date, as_of_gate=None):
            return {}

    class _EmptyReserveStore:
        def load_reserve(self, delivery_date, product):
            return {}

    monkeypatch.setattr(mod, "PositionStore", lambda: _EmptyPositionStore())
    monkeypatch.setattr(mod, "ReserveStore", lambda: _EmptyReserveStore())

    res = mod.resettle_live_bid(_NO_REAL_DATA_DATE)
    assert res.gates == []
    assert res.reserves == []
    assert res.total_realised_revenue_eur is None
    assert res.total_forecast_revenue_eur == 0.0
    assert res.n_gates_total == 0


def test_resettle_live_bid_gate_with_real_price_unavailable_reports_independently(monkeypatch):
    """DA has real coverage; IDA1 (deliberately a future/no-coverage date
    scenario simulated here) must independently report 'unavailable' while
    DA still reports real -- never all-or-nothing across gates, same
    invariant already established for the existing real-price backtest."""
    import phase_6_backtesting_and_validation.backtest_engine.live_bid_resettlement as mod
    from common_layer.utilities import date_utils as du

    day = du.parse_date(_REAL_ALL_GATES_DATE)
    isps = du.delivery_isps(day)
    da_pos = {isp: {"volume_mwh": 30.0, "price_eur_mwh": 41.0} for isp in isps}
    ida1_pos = {isp: {"volume_mwh": 35.0, "price_eur_mwh": 43.0} for isp in isps}

    class _FakePositionStore:
        def load_position(self, delivery_date, gate):
            if gate == "DA":
                return da_pos
            if gate == "IDA1":
                return ida1_pos
            return {}
        def committed_position(self, delivery_date, as_of_gate=None):
            if as_of_gate == "DA":
                return {isp: 30.0 for isp in isps}
            return {}

    class _EmptyReserveStore:
        def load_reserve(self, delivery_date, product):
            return {}

    monkeypatch.setattr(mod, "PositionStore", lambda: _FakePositionStore())
    monkeypatch.setattr(mod, "ReserveStore", lambda: _EmptyReserveStore())
    # Force real_ida_price to look unavailable regardless of real archive state,
    # so this test is deterministic and doesn't depend on IDA1's real-data window.
    monkeypatch.setattr(mod, "real_ida_price", lambda gate, date, hours: None)

    res = mod.resettle_live_bid(_REAL_ALL_GATES_DATE)
    by_gate = {g.gate: g for g in res.gates}
    assert by_gate["DA"].price_source == "OMIE_LIVE"
    assert by_gate["DA"].realised_revenue_eur is not None
    assert by_gate["IDA1"].price_source == "unavailable"
    assert by_gate["IDA1"].realised_revenue_eur is None
    # Total must be unavailable too since it's not all-gate-real.
    assert res.total_realised_revenue_eur is None
    # But forecast total (Trading-Desk-comparable figure) must still sum both.
    assert res.total_forecast_revenue_eur != 0.0


def test_reserve_capacity_realised_and_activation_flagged_excluded(monkeypatch):
    import phase_6_backtesting_and_validation.backtest_engine.live_bid_resettlement as mod

    afrr_offers = {h: {"up_mw": 5.0, "dn_mw": 3.0,
                       "cap_up_eur_mw": 8.0, "cap_dn_eur_mw": 6.0} for h in range(1, 25)}
    real_up = {h: 9.0 for h in range(1, 25)}
    real_dn = {h: 7.0 for h in range(1, 25)}

    class _EmptyPositionStore:
        def load_position(self, delivery_date, gate):
            return {}
        def committed_position(self, delivery_date, as_of_gate=None):
            return {}

    class _FakeReserveStore:
        def load_reserve(self, delivery_date, product):
            return afrr_offers if product == "aFRR" else {}

    monkeypatch.setattr(mod, "PositionStore", lambda: _EmptyPositionStore())
    monkeypatch.setattr(mod, "ReserveStore", lambda: _FakeReserveStore())
    monkeypatch.setattr(mod, "real_ren_capacity_price",
                        lambda date, hours, product: (real_up, real_dn) if product == "aFRR" else None)

    res = mod.resettle_live_bid(_REAL_ALL_GATES_DATE)
    assert len(res.reserves) == 1
    r = res.reserves[0]
    assert r.product == "aFRR"
    assert r.price_source == "REN_LIVE"
    assert r.activation_excluded is True
    expected_forecast = 24 * (5.0 * 8.0 + 3.0 * 6.0)
    expected_real = 24 * (5.0 * 9.0 + 3.0 * 7.0)
    assert r.forecast_capacity_revenue_eur == pytest.approx(expected_forecast)
    assert r.realised_capacity_revenue_eur == pytest.approx(expected_real)
