"""
afrr_offer_builder.py — build the aFRR capacity offer from leftover headroom.

aFRR is the FAST automatic reserve (FAT 5 min, restores frequency within the
+/- 0.200 Hz band, 49.800-50.200 Hz). Its real gate closes AFTER the DA energy
gate — CONFIRMED against REN's own MPGGS rulebook (Article 80(3): PDVD ->
aFRR band -> mFRR band, see run_afrr.py for the full citation) — so it is
sized from the plant's DA-committed position, taking first call on whatever
headroom DA left behind (higher value than mFRR). Offers are bounded by the
market max (config afrr.max_offer_up/dn_mw) and FAT deliverability, and sized
via the shared builder.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from common_layer.configuration.config_loader import AppConfig
from common_layer.optimisation_model.reserve_offer_builder import (
    build_reserve_offers, compute_price_aware_fraction, ReserveOffer,
)


def build_afrr_offers(committed_net: Dict[int, float],
                      cap_up: Dict[int, float], cap_dn: Dict[int, float],
                      cfg: AppConfig,
                      da_price_eur_mwh: Optional[Dict[int, float]] = None,
                      ) -> Dict[int, ReserveOffer]:
    a = cfg.market.afrr
    # Energy-side reference for the price-aware split: DA's already-cleared
    # price. An IDA1 price *forecast* was tried here (peeking at a market
    # that hasn't opened yet to size an earlier offer) and dropped as not
    # representative of real trading-desk practice.
    fraction_by_hour = None
    if a.dynamic_allocation_enabled and da_price_eur_mwh:
        fraction_by_hour = {
            h: compute_price_aware_fraction(
                cap_price_eur_mw=cap_up.get(h, 0.0),
                da_price_eur_mwh=da_price_eur_mwh.get(h, 0.0),
                assumed_duty_cycle_h=a.assumed_duty_cycle_h,
                min_fraction=a.min_offer_fraction,
                max_fraction=1.0,           # today's fixed ceiling stays the ceiling
            )
            for h in committed_net
        }
    return build_reserve_offers(
        product="aFRR",
        committed_net=committed_net,
        cap_prices_up=cap_up,
        cap_prices_dn=cap_dn,
        cfg=cfg,
        fat_min=a.fat_min,                 # 5 min
        max_up_mw=a.max_offer_up_mw,
        max_dn_mw=a.max_offer_dn_mw,
        headroom_fraction=1.0,             # aFRR has first call on DA's leftover headroom
        headroom_fraction_by_hour=fraction_by_hour,
    )
