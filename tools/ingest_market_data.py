"""
ingest_market_data.py — load market data into the raw and clean zones (Phase 2 data layer).

Examples (from the repository root):
    python tools/ingest_market_data.py --start 2025-10-25 --end 2025-10-27
    python tools/ingest_market_data.py --start 2026-01-01 --end 2026-01-31 --sources omie_da ren_afrr
    python tools/ingest_market_data.py --rebuild --start 2025-10-25 --end 2025-10-27   # no network
    python tools/ingest_market_data.py --verify                                        # checksums only

Re-running is safe: days already in the raw zone are skipped (use --refresh to re-check the source).
The first long run (history back to 2019) is ~10 000 polite requests; start with a short window.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.data_layer import fetchers, ingest  # noqa: E402
from common_layer.data_layer.clean_store import CleanStore  # noqa: E402
from common_layer.data_layer.raw_store import RawStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=dt.date.fromisoformat)
    ap.add_argument("--end", type=dt.date.fromisoformat)
    ap.add_argument("--sources", nargs="+", default=list(ingest.SOURCES), choices=list(ingest.SOURCES))
    ap.add_argument("--refresh", action="store_true", help="re-download days already in the raw zone")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the clean zone from the raw zone only")
    ap.add_argument("--verify", action="store_true", help="verify raw-zone checksums and exit")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between requests (default 1)")
    a = ap.parse_args()

    raw, clean, log = RawStore(), CleanStore(), ingest.LoadLog()
    if a.verify:
        problems = raw.verify_all()
        print("raw zone OK" if not problems else "\n".join(problems))
        return 1 if problems else 0
    if not (a.start and a.end):
        ap.error("--start and --end are required")

    if a.rebuild:
        days = [a.start + dt.timedelta(days=i) for i in range((a.end - a.start).days + 1)]
        for src in a.sources:
            print(src, ingest.rebuild_clean(src, days, raw=raw, clean=clean, log=log))
        return 0

    f = fetchers.Fetcher(min_interval_s=a.interval)
    print(ingest.ingest_range(a.sources, a.start, a.end, raw=raw, clean=clean, fetcher=f, log=log,
                              refresh=a.refresh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
