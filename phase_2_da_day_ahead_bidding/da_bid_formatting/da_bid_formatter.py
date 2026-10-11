"""
da_bid_formatter.py — turn the solved schedule into an OMIE DA bid.

For a price-taking self-schedule, each hour submits the optimised quantity. We
express sells (net generation) and buys (net pumping) as signed volume at the
forecast price, then render an OMIE-style hourly bid block plus a human table
for the trader. The real OMIE submission is XML over the participant API; that
serialisation is stubbed (clearly marked) since live submission is out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from common_layer.optimisation_model.core_milp_solver import GateResults


@dataclass
class DABid:
    hour: int
    volume_mwh: float          # + sell (generate), - buy (pump)
    price_eur_mwh: float
    side: str                  # "SELL" or "BUY" or "IDLE"


def format_da_bids(results: GateResults) -> List[DABid]:
    bids: List[DABid] = []
    for h in sorted(results.da_bids):
        b = results.da_bids[h]
        vol = b["volume_mwh"]
        side = "SELL" if vol > 1e-6 else "BUY" if vol < -1e-6 else "IDLE"
        bids.append(DABid(hour=h, volume_mwh=vol,
                          price_eur_mwh=b["price_eur_mwh"], side=side))
    return bids


def to_omie_payload(bids: List[DABid], delivery_date: str, unit_id: str = "ALQUEVA") -> dict:
    """Build the structured payload that the (stubbed) OMIE submitter would send.

    resolution is derived from the actual bid count -- 96 bids means this
    schedule was solved at real 15-min MTU granularity (MIBEL since
    2025-10-01), 24 (or 23/25 on DST) means hourly, matching whatever
    inputs["hours"]/dt_h the MILP was actually built with."""
    resolution = "15min" if len(bids) > 25 else "hourly"
    return {
        "market": "MIBEL_DA",
        "operator": "OMIE",
        "unit": unit_id,
        "delivery_date": delivery_date,
        "resolution": resolution,
        "bids": [
            {"hour": b.hour, "volume_mwh": round(b.volume_mwh, 3),
             "price_eur_mwh": round(b.price_eur_mwh, 2), "side": b.side}
            for b in bids
        ],
    }


def render_table(bids: List[DABid]) -> str:
    isp_mode = len(bids) > 25
    label, width = ("ISP", 5) if isp_mode else ("Hour", 5)
    lines = [f"  {label:<{width}} {'Side':<5} {'Volume MWh':>12} {'Price EUR/MWh':>15}",
             "  " + "-" * 40]
    for b in bids:
        tag = f"Q{b.hour:03d}" if isp_mode else f"H{b.hour:02d}"
        lines.append(f"  {tag:<5} {b.side:<5} {b.volume_mwh:>+12.2f} {b.price_eur_mwh:>15.2f}")
    return "\n".join(lines)
