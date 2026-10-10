"""
test_pv_losses.py — PV DC-to-AC chain (inverter efficiency and system losses, PVWatts form).
"""
from __future__ import annotations

import dataclasses

import pytest

from common_layer.configuration import load_config
from common_layer.physical_plant_models.pv_production_model import PVModel


@pytest.fixture(scope="module")
def pv_cfg():
    return load_config().plant.pv


def _lossless(pv_cfg):
    return dataclasses.replace(pv_cfg, inverter_efficiency=1.0, system_losses_frac=0.0)


def test_configured_chain_is_the_pvwatts_default(pv_cfg):
    assert pv_cfg.inverter_efficiency == pytest.approx(0.96)
    assert pv_cfg.system_losses_frac == pytest.approx(0.1408)
    assert PVModel(pv_cfg, year=2026).system_factor == pytest.approx(0.96 * (1 - 0.1408))


def test_output_at_standard_conditions(pv_cfg):
    m = PVModel(pv_cfg, year=2026)
    expected = pv_cfg.peak_capacity_mw * m.degradation_factor * m.system_factor
    assert m.production_mw(1000.0, pv_cfg.t_ref_c) == pytest.approx(expected)


def test_losses_scale_the_lossless_output_by_the_system_factor(pv_cfg):
    with_losses = PVModel(pv_cfg, year=2026)
    lossless = PVModel(_lossless(pv_cfg), year=2026)
    for g, t in [(200.0, 20.0), (600.0, 35.0), (900.0, 45.0)]:
        assert with_losses.production_mw(g, t) == pytest.approx(
            lossless.production_mw(g, t) * with_losses.system_factor)


def test_disabling_the_chain_restores_the_original_formula(pv_cfg):
    cfg = _lossless(pv_cfg)
    m = PVModel(cfg, year=2026)
    g, t = 800.0, 40.0
    expected = (cfg.peak_capacity_mw * (g / cfg.g_ref_wm2)
                * (1 + cfg.temperature_coeff_per_c * (t - cfg.t_ref_c)) * m.degradation_factor)
    assert m.production_mw(g, t) == pytest.approx(expected)


def test_no_output_without_sun_and_never_above_the_degraded_peak(pv_cfg):
    m = PVModel(pv_cfg, year=2026)
    assert m.production_mw(0.0, 20.0) == 0.0
    assert m.production_mw(5000.0, 10.0) <= m.effective_peak_mw + 1e-9


def test_missing_settings_default_to_no_losses():
    from common_layer.configuration.plant_config import PVConfig
    d = dict(peak_capacity_mw=5.0, latitude=38.2, longitude=-7.5, temperature_coeff_per_c=-0.0045,
             t_ref_c=25.0, g_ref_wm2=1000.0, commission_year=2022, degradation_rate_per_year=0.005)
    assert PVModel(PVConfig.from_dict(d), year=2026).system_factor == 1.0
