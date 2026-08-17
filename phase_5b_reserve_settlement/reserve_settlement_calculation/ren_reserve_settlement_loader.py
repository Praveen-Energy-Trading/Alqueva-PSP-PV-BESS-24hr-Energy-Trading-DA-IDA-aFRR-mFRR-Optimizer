"""
ren_reserve_settlement_loader.py — reserve settlement inputs from REN data.

Pulls the two settlement components for a reserve product:
  * committed capacity offer (up/dn MW + cap price) from ReserveStore,
  * activated energy per ISP (up/dn MW + energy price) from ActivationStore.

The committed MW quantity is legitimately internal (our own accepted offer —
REN doesn't publish per-plant volumes), but the CAP PRICE we settle at should
be the real market-cleared price, not just our own bid replayed back at
ourselves. load_capacity_offer() reconciles: it takes our stored MW from
ReserveStore, but overwrites cap_up_eur_mw / cap_dn_eur_mw with REN's real
published clearing price for that hour when available (reusing the same live
fetchers already verified in picasso_afrr_price_loader.py /
mari_mfrr_price_loader.py), falling back to our own stored bid price when the
real price can't be fetched (network down, date not published yet, etc).

Activated MW/ISP volumes stay internal (ActivationStore — our own simulation
of what the TSO activated, since we have no real SCADA/REN telemetry access);
only the capacity-side price gets reconciled here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from common_layer.database import ReserveStore, ActivationStore
from common_layer.utilities.logging_utils import get_logger

log = get_logger(__name__)


def load_capacity_offer(delivery_date: str, product: str) -> Dict[int, dict]:
    offers = ReserveStore().load_reserve(delivery_date, product)
    real = _real_cap_prices(delivery_date, product, list(offers.keys()))
    if not real:
        return offers
    return {
        h: {**o, "cap_up_eur_mw": real[h][0], "cap_dn_eur_mw": real[h][1]}
        if h in real else o
        for h, o in offers.items()
    }


def load_activations(delivery_date: str, product: str) -> List[dict]:
    return ActivationStore().load(delivery_date, product)


def _real_cap_prices(delivery_date: str, product: str,
                     hours: List[int]) -> Optional[Dict[int, Tuple[float, float]]]:
    """Return {hour: (cap_up_eur_mw, cap_dn_eur_mw)} from REN's real published
    clearing price, or None if unavailable for this date/product."""
    dt = pd.Timestamp(delivery_date)
    try:
        if product == "aFRR":
            from phase_3a_afrr_automatic_frequency_reserve.afrr_price_forecasting.picasso_afrr_price_loader import (
                _download_ren_afrr,
            )
            cap_up, cap_dn = _download_ren_afrr(dt)
        elif product == "mFRR":
            from phase_3b_mfrr_manual_frequency_reserve.mfrr_price_forecasting.mari_mfrr_price_loader import (
                _download_ren_mfrr,
            )
            price = _download_ren_mfrr(dt)
            cap_up = cap_dn = price   # REN doesn't publish a separate mFRR up/dn split
        else:
            return None
    except Exception as exc:
        log.warning(f"[{product}] real cap-price reconciliation failed ({exc}); "
                    f"settling at our own stored offer price.")
        return None

    return {h: (cap_up[h], cap_dn[h]) for h in hours if h in cap_up and h in cap_dn}
