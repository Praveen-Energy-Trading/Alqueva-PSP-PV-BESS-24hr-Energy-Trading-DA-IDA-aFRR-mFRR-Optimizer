"""
run_formulation_evidence.py — the measurements behind the Phase 5 formulation choices.

    python tools/run_formulation_evidence.py [--out DIR]

Writes CSV tables to runtime/analysis/phase5/ and prints a short summary. Read-only on the clean zone.
Questions answered (see common_layer/forecast_formulation/evidence.py):
  * which asinh scale equalises the day-over-day change across price levels
  * how much of the day-ahead error is one shift common to all 24 hours, and how many factors describe the day
  * spread to day-ahead versus level for each intraday market and the reserve prices
  * how the 15-minute day-ahead price moves inside the hour
  * when OMIE files are really issued
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.data_layer.clean_store import CleanStore  # noqa: E402
from common_layer.forecast_formulation import coherence, evidence  # noqa: E402
from common_layer.market_analysis import dependence, panel as pnl  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SCALES = [None, 5, 10, 20, 35, 50, 75, 100, 150, 250]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_REPO, "runtime", "analysis", "phase5"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clean = CleanStore()
    raw = pnl.build_panel(clean)
    p = pnl.add_calendar(raw)
    tables = {}

    da = raw["da_pt"].dropna()
    tables["asinh_scale"] = evidence.heteroscedasticity_by_scale(da, SCALES)
    resid = dependence.deseasonalise(p, "da_pt")
    rows, fac = [], []
    for label, r in (("all years", resid), ("2024 onward", resid[resid.index >= "2024-01-01"]),
                     ("2025-10 onward (15-min era)", resid[resid.index >= "2025-10-01"])):
        s = evidence.day_shift_share(r)
        s["sample"] = label
        rows.append(s)
        f = evidence.daily_shape_factors(r)
        f["sample"] = label
        fac.append(f)
    tables["day_shift_share"] = pd.DataFrame(rows)
    tables["daily_shape_factors"] = pd.concat(fac, ignore_index=True)

    sp = []
    for other in ("ida1_pt", "ida2_pt", "ida3_pt", "xbid_pt", "imb_short", "imb_long", "afrr_up", "afrr_dn", "mfrr_ap"):
        if raw[other].notna().sum() > 500:
            r = evidence.spread_vs_level(raw[other], raw["da_pt"])
            r["series"] = other
            sp.append(r)
    tables["spread_vs_level"] = pd.DataFrame(sp)

    # 15-minute day-ahead shape inside the hour
    q = clean.read("da_price")
    q = q[q["resolution_min"] == 15].copy()
    if len(q):
        t = pd.to_datetime(q["start_utc"], utc=True, format="%Y-%m-%dT%H:%M:%SZ")
        d = pd.DataFrame({"price": q["price_pt_eur_mwh"].to_numpy()}, index=pd.DatetimeIndex(t))
        d["local_hour"] = d.index.tz_convert(evidence.TZ).hour
        d["quarter"] = d.index.minute // 15
        full = d.groupby(d.index.floor("h"))["price"].transform("count") == 4
        out = coherence.within_hour_shape(d[full])
        tables["within_hour_shape"] = out["shape"]
        tables["flat_split_error"] = out["flat_split_error"]

    # publication times from the files themselves
    pub = []
    for ds in ("da_price", "xbid_price"):
        df = clean.read(ds)
        if "issued_utc" in df.columns and len(df):
            day = df.drop_duplicates("delivery_date")
            lags = evidence.publication_lags(day["issued_utc"], day["delivery_date"])
            s = evidence.summarise_publication(lags)
            s["dataset"] = ds
            pub.append(s)
    tables["publication"] = pd.DataFrame(pub)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for name, t in tables.items():
        t.to_csv(os.path.join(a.out, f"{name}.csv"), index=not isinstance(t.index, pd.RangeIndex))
        print(f"== {name}")
        print(t.round(3).to_string(index=not isinstance(t.index, pd.RangeIndex)))
        print()
    print("tables written to", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
