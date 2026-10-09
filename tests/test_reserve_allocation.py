"""
test_reserve_allocation.py — the replica allocation of the 524.4 MW plant:
65% DA energy / 20% aFRR / 14% mFRR / 1% FCR.

Checks the configured sizes, the physical limits by plant mode (turbining,
pumping, idle), the "no aFRR while idle" rule, the separate mFRR size cap, and
the mFRR future-band scenario price. No solver needed.
"""
from __future__ import annotations

import dataclasses

import pytest

from common_layer.configuration import load_config
from phase_3a_afrr_automatic_frequency_reserve.afrr_reserve_offer_builder.afrr_offer_builder import (
    build_afrr_offers,
)
from phase_3b_mfrr_manual_frequency_reserve.mfrr_reserve_offer_builder.mfrr_offer_builder import (
    build_mfrr_offers,
)

PLANT_MW = 524.4


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _offers(cfg, net_mw: float):
    committed = {1: net_mw}
    zero = {1: 0.0}
    a = build_afrr_offers(committed, zero, zero, cfg)
    m = build_mfrr_offers(committed, zero, zero,
                          {1: a[1].up_mw}, {1: a[1].dn_mw}, cfg)
    return a[1], m[1]


class TestConfiguredAllocation:
    def test_sizes_are_the_replica_allocation(self, cfg):
        assert cfg.market.afrr.max_offer_up_mw == 105.0      # 20% of 524.4
        assert cfg.market.afrr.max_offer_dn_mw == 105.0
        assert cfg.market.mfrr.max_offer_up_mw == 74.0       # 14% of 524.4
        assert cfg.market.mfrr.max_offer_dn_mw == 74.0
        assert cfg.plant.fcr.mandatory_headroom_mw == 5.0    # 1% of 524.4

    def test_percentages_add_to_the_plant(self, cfg):
        a = cfg.market.afrr.max_offer_up_mw
        m = cfg.market.mfrr.max_offer_up_mw
        f = cfg.plant.fcr.mandatory_headroom_mw
        da = PLANT_MW - a - m - f
        assert round(a / PLANT_MW * 100) == 20
        assert round(m / PLANT_MW * 100) == 14
        assert round(f / PLANT_MW * 100) == 1
        assert round(da / PLANT_MW * 100) == 65

    def test_mfrr_offers_all_leftover_headroom(self, cfg):
        assert cfg.market.mfrr.max_offer_fraction == 1.0

    def test_mfrr_band_is_paid_as_the_future_scenario_by_default(self, cfg):
        # default since 2026-10: priced at the aFRR band price (see market.yaml)
        assert cfg.market.mfrr.capacity_payment is True


class TestPhysicalLimitsByMode:
    def test_generating_near_full_load_leaves_little_up_headroom(self, cfg):
        a, m = _offers(cfg, 454.0)
        assert a.up_mw == pytest.approx(519.4 - 454.0)   # all of the headroom left
        assert a.dn_mw == 105.0                           # size cap binds
        assert m.up_mw == 0.0                             # aFRR already took the rest
        assert m.dn_mw == 74.0                            # mFRR cap binds

    def test_pumping_near_full_draw_leaves_little_down_headroom(self, cfg):
        a, m = _offers(cfg, -423.4)
        assert a.dn_mw == pytest.approx(-423.4 + 442.4)   # 19 MW of pump headroom
        assert a.up_mw == 105.0                           # can stop pumping
        assert m.dn_mw == 0.0
        assert m.up_mw == 74.0

    def test_no_double_selling_across_products(self, cfg):
        for net in (-400.0, -200.0, 0.0, 200.0, 450.0):
            a, m = _offers(cfg, net)
            assert net + a.up_mw + m.up_mw <= 519.4 + 1e-6
            assert net - a.dn_mw - m.dn_mw >= -442.4 - 1e-6


class TestIdleRule:
    def test_no_afrr_while_idle(self, cfg):
        a, _ = _offers(cfg, 0.0)
        assert (a.up_mw, a.dn_mw) == (0.0, 0.0)

    def test_afrr_offered_as_soon_as_a_unit_runs(self, cfg):
        a, _ = _offers(cfg, 250.0)
        assert a.up_mw > 0.0 and a.dn_mw > 0.0

    def test_mfrr_still_offered_while_idle(self, cfg):
        # 12.5 min is enough to start a unit, so mFRR is allowed from standstill
        _, m = _offers(cfg, 0.0)
        assert (m.up_mw, m.dn_mw) == (74.0, 74.0)


class TestMfrrSeparateCap:
    def test_mfrr_cap_is_independent_of_the_afrr_cap(self, cfg):
        wide = dataclasses.replace(cfg.market.afrr, max_offer_up_mw=300.0, max_offer_dn_mw=300.0)
        cfg2 = dataclasses.replace(cfg, market=dataclasses.replace(cfg.market, afrr=wide))
        _, m = _offers(cfg2, 250.0)
        assert m.up_mw <= 74.0 and m.dn_mw <= 74.0

    def test_mfrr_falls_back_to_the_afrr_cap_when_unset(self, cfg):
        none_cap = dataclasses.replace(cfg.market.mfrr, max_offer_up_mw=None, max_offer_dn_mw=None)
        cfg2 = dataclasses.replace(cfg, market=dataclasses.replace(cfg.market, mfrr=none_cap))
        _, m = _offers(cfg2, 0.0)
        assert m.up_mw == 105.0


class TestFutureBandScenario:
    def _scenario_cfg(self, cfg, on: bool):
        mf = dataclasses.replace(cfg.market.mfrr, capacity_payment=on)
        return dataclasses.replace(cfg, market=dataclasses.replace(cfg.market, mfrr=mf))

    def test_mfrr_capacity_price_is_zero_when_unpaid(self, cfg, monkeypatch):
        import common_layer.configuration as c
        from phase_6_backtesting_and_validation.backtest_engine.historical_data_loader import (
            real_ren_capacity_price,
        )
        monkeypatch.setattr(c, "load_config", lambda *a, **k: self._scenario_cfg(cfg, False))
        up, dn = real_ren_capacity_price("2026-09-10", list(range(1, 25)), "mFRR")
        assert set(up.values()) == {0.0} and set(dn.values()) == {0.0}

    def test_mfrr_capacity_is_paid_at_the_afrr_band_price_in_the_scenario(self, cfg, monkeypatch):
        import common_layer.configuration as c
        from phase_6_backtesting_and_validation.backtest_engine.historical_data_loader import (
            real_ren_capacity_price,
        )
        monkeypatch.setattr(c, "load_config", lambda *a, **k: self._scenario_cfg(cfg, True))
        hours = list(range(1, 25))
        m_up, m_dn = real_ren_capacity_price("2026-09-10", hours, "mFRR")
        a_up, a_dn = real_ren_capacity_price("2026-09-10", hours, "aFRR")
        assert m_up == a_up and m_dn == a_dn
        assert any(v > 0 for v in m_up.values())
