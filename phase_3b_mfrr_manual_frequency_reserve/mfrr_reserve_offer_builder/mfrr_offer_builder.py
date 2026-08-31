"""
mfrr_offer_builder.py — build the mFRR capacity offer from headroom aFRR did not take.

mFRR = manual Frequency Restoration Reserve (FAT 12.5 min, MARI). It is slower and
lower-value than aFRR, so it is sized from the headroom REMAINING after the aFRR
commitment (passed in as reserved_up/dn). The offer is further limited to a
fraction of headroom (config mfrr.max_offer_fraction) to leave operating margin —
a discretionary trading-desk risk policy, NOT a REN/MPGGS rule (verified against
the real MPGGS Procedimento 19; see market.yaml's comment on max_offer_fraction
for the citation and what the real per-unit limit actually is).
"""
from __future__ import annotations

from typing import Dict, Optional

from common_layer.configuration.config_loader import AppConfig
from common_layer.optimisation_model.reserve_offer_builder import (
    build_reserve_offers, compute_price_aware_fraction, ReserveOffer,
)


def build_mfrr_offers(committed_net: Dict[int, float],
                      cap_up: Dict[int, float], cap_dn: Dict[int, float],
                      reserved_up: Dict[int, float], reserved_dn: Dict[int, float],
                      cfg: AppConfig,
                      da_price_eur_mwh: Optional[Dict[int, float]] = None,
                      ) -> Dict[int, ReserveOffer]:
    mf = cfg.market.mfrr
    # See afrr_offer_builder.build_afrr_offers: DA's already-cleared price is
    # the energy-side reference. An IDA1 forecast peek was tried and dropped
    # as not representative of real trading-desk practice.
    fraction_by_hour = None
    if mf.dynamic_allocation_enabled and da_price_eur_mwh:
        fraction_by_hour = {
            h: compute_price_aware_fraction(
                cap_price_eur_mw=cap_up.get(h, 0.0),
                da_price_eur_mwh=da_price_eur_mwh.get(h, 0.0),
                assumed_duty_cycle_h=mf.assumed_duty_cycle_h,
                min_fraction=mf.min_offer_fraction,
                max_fraction=mf.max_offer_fraction,   # today's fixed 0.20 stays the ceiling
            )
            for h in committed_net
        }
    return build_reserve_offers(
        product="mFRR",
        committed_net=committed_net,
        cap_prices_up=cap_up,
        cap_prices_dn=cap_dn,
        cfg=cfg,
        fat_min=mf.fat_min,                          # 12.5 min
        max_up_mw=cfg.market.afrr.max_offer_up_mw,   # mFRR shares the aFRR market-size cap; no separate config field
        max_dn_mw=cfg.market.afrr.max_offer_dn_mw,
        headroom_fraction=mf.max_offer_fraction,     # 0.20 of remaining headroom
        reserved_up=reserved_up,                     # subtract aFRR commitment
        reserved_dn=reserved_dn,
        headroom_fraction_by_hour=fraction_by_hour,
    )
