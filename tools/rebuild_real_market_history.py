"""
rebuild_real_market_history.py — rebuild every price training file from the
official public sources, so no forecaster trains or is scored on generated data
that is labelled as real.

Why this exists
---------------
An audit on 2026-10-03 found that:
  * the hourly DA file (da_training_data_2020_2026.xlsx) held values that do
    not match OMIE for 2020-2024 and Oct 2025-Aug 2026, although every row was
    labelled OMIE_LIVE (e.g. 2022-03-08 H1: file 164.07, OMIE 450.95 EUR/MWh);
  * the pre-2026 IDA1/2/3, XBID, aFRR and mFRR rows came from the
    create_*_training_data.py generators (random processes) with a blank
    source label;
  * the 2026 IDA/XBID/aFRR/mFRR rows took their DA reference column from the
    corrupted hourly DA file, so their spreads were wrong.

What it does
------------
Downloads, per delivery date, from the same public endpoints the live loaders
already use, and rewrites each Excel file with the same sheet name, columns and
24-rows-per-day layout:

  DA hourly   OMIE  INT_PBC_EV_H_1_<dd>_<mm>_<yyyy>...TXT    2020-01-01 -> today
  IDA1/2/3    OMIE  marginalpibcpt_<yyyymmdd><session>.1      2024-06-13 -> today
  XBID        OMIE  precios_pibcic_<yyyymmdd>.1 (MedioPT)     2024-06-13 -> yesterday
  aFRR        REN   BaFRRPreco (band price, adjusted > initial) 2019-01-01 -> today
  mFRR        REN   MFRRPreco  (AP_PRECO, documented proxy)     2024-11-27 -> yesterday

Old files are hourly; since the 15-min switch they carry 96 quarter-hours,
which are averaged to hours. DST days (23/25 hours) are mapped onto 24 rows like
the existing files. Any date/hour that cannot be downloaded is gap-filled and
labelled SYNTHETIC (spread 0 for IDA/XBID, previous day for reserves) so the
lag features stay contiguous and the evaluation scripts can exclude it.

The DA reference column of every IDA/XBID/aFRR/mFRR file is taken from the
rebuilt hourly DA file, and spreads are recomputed from it.

Downloads are cached as JSON under runtime/cache/market_history/ (gitignored),
so an interrupted run resumes where it stopped.

Run:
    python tools/rebuild_real_market_history.py            # everything
    python tools/rebuild_real_market_history.py --only da,ida1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from common_layer.utilities.date_utils import portugal_today  # noqa: E402
from phase_3a_afrr_automatic_frequency_reserve.afrr_price_forecasting.picasso_afrr_price_loader import (  # noqa: E402
    _REN_API_KEY_B64,
)

_CACHE = os.path.join(_REPO, "runtime", "cache", "market_history")
_FCST = os.path.join(_REPO, "phase_1_da_day_ahead_bidding", "da_price_pv_inflow_forecasting")

DA_XLSX = os.path.join(_FCST, "da_training_data_2020_2026.xlsx")
DA_SHEET = "DA_Price_2020_2026"
ISP_XLSX = os.path.join(_FCST, "da_training_data_isp_2025_2026.xlsx")
ISP_SHEET = "DA_Price_ISP_2025_2026"
ISP_START = dt.date(2025, 10, 1)     # first OMIE day-ahead session with 96 quarter-hours

_IDA = {
    "ida1": ("01", os.path.join(_REPO, "phase_2a_ida1_intraday_auction_1", "ida1_price_forecasting",
                                "ida1_training_data_2024_2025.xlsx"), "IDA1_2024_2025"),
    "ida2": ("02", os.path.join(_REPO, "phase_2b_ida2_intraday_auction_2", "ida2_price_forecasting",
                                "ida2_training_data_2024_2025.xlsx"), "IDA2_2024_2025"),
    "ida3": ("03", os.path.join(_REPO, "phase_2c_ida3_intraday_auction_3", "ida3_price_forecasting",
                                "ida3_training_data_2024_2025.xlsx"), "IDA3_2024_2025"),
}
XBID_XLSX = os.path.join(_REPO, "phase_2d_xbid_continuous_intraday", "xbid_price_forecasting",
                         "xbid_training_data_2024_2025.xlsx")
XBID_SHEET = "XBID_2024_2025"
AFRR_XLSX = os.path.join(_REPO, "phase_3a_afrr_automatic_frequency_reserve", "afrr_price_forecasting",
                         "afrr_training_data_2019_2025.xlsx")
AFRR_SHEET = "AFRR_2019_2025"
MFRR_XLSX = os.path.join(_REPO, "phase_3b_mfrr_manual_frequency_reserve", "mfrr_price_forecasting",
                         "mfrr_training_data_2024_2025.xlsx")
MFRR_SHEET = "MFRR_2024_2025"

IDA_START = dt.date(2024, 6, 13)     # SIDC 3-session regime (MIBEL IDAs)
AFRR_START = dt.date(2019, 1, 1)
MFRR_START = dt.date(2024, 11, 27)   # REN accession to MARI
DA_START = dt.date(2020, 1, 1)

_WORKERS = 6
_RETRIES = 3


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _num_comma(s: str) -> float:
    return float(s.strip().replace(".", "").replace(",", "."))


def _to_24_hours(values: List[float]) -> Dict[int, float]:
    """Map a day's ordered values onto hours 1..24.

    Accepts hourly (23/24/25) or quarter-hourly (92/96/100) series. DST days
    follow OMIE's CET clock: the spring day lacks hour 3, the autumn day has it
    twice. Like the existing training files, the result always has 24 hours.
    """
    n = len(values)
    if n in (92, 96, 100):
        values = [sum(values[i:i + 4]) / 4.0 for i in range(0, n, 4)]
        n = len(values)
    if n == 23:
        values = values[:2] + [(values[1] + values[2]) / 2.0] + values[2:]
    elif n == 25:
        values = values[:2] + [(values[2] + values[3]) / 2.0] + values[4:]
    if len(values) != 24:
        raise ValueError(f"cannot map {n} values onto 24 hours")
    return {h: round(v, 2) for h, v in enumerate(values, start=1)}


def _to_96_quarters(values: List[float]) -> List[float]:
    """Map a quarter-hour day (92/96/100 values) onto 96 slots, DST like _to_24_hours."""
    n = len(values)
    if n == 92:      # spring: hour 3 missing -> its 4 quarters = mean of hours 2 and 4 quarters
        fill = [(values[4 + q] + values[8 + q]) / 2.0 for q in range(4)]
        values = values[:8] + fill + values[8:]
    elif n == 100:   # autumn: hour 3 repeated -> average the two occurrences
        merged = [(values[8 + q] + values[12 + q]) / 2.0 for q in range(4)]
        values = values[:8] + merged + values[16:]
    if len(values) != 96:
        raise ValueError(f"cannot map {n} quarter values onto 96 ISPs")
    return [round(v, 2) for v in values]


def _periods_to_hours(periods: Dict[int, float], n_day: Optional[int] = None) -> Dict[int, float]:
    """Map {period: price} (hourly 1..25 or quarter 1..100) to {hour: price}.

    Partial days (IDA3 covers only the afternoon, XBID hours without trades)
    keep only the hours that are fully present.
    """
    if not periods:
        return {}
    quarter = max(periods) > 25
    if not quarter:
        if len(periods) in (23, 25) and n_day:
            return _to_24_hours([periods[p] for p in sorted(periods)])
        return {p: round(v, 2) for p, v in periods.items() if 1 <= p <= 24}
    if len(periods) in (92, 100):
        return _to_24_hours([periods[p] for p in sorted(periods)])
    out = {}
    for h in range(1, 25):
        qs = [periods.get((h - 1) * 4 + q) for q in range(1, 5)]
        if all(q is not None for q in qs):
            out[h] = round(sum(qs) / 4.0, 2)
    return out


# ---------------------------------------------------------------------------
# Downloaders (one delivery date each). Return None when the source has no
# data for that date; raise only on transport errors (retried).
# ---------------------------------------------------------------------------

def _get(url: str, **kw):
    import requests
    resp = requests.get(url, timeout=25, **kw)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp


def dl_da(day: dt.date) -> Optional[dict]:
    dd, mm, yyyy = f"{day.day:02d}", f"{day.month:02d}", f"{day.year}"
    resp = _get(f"https://www.omie.es/sites/default/files/dados/AGNO_{yyyy}/MES_{mm}/TXT/"
                f"INT_PBC_EV_H_1_{dd}_{mm}_{yyyy}_{dd}_{mm}_{yyyy}.TXT")
    if resp is None:
        return None
    rows = {}
    for line in resp.text.splitlines():
        low = line.lower()
        if "precio marginal" not in low:
            continue
        vals = [_num_comma(p) for p in line.split(";")[1:] if p.strip()]
        key = "PT" if "portug" in low else "ES" if "espa" in low else None
        if key and key not in rows:
            rows[key] = vals
    if not rows:
        return None
    es = rows.get("ES") or rows.get("PT")
    pt = rows.get("PT") or es        # single coupled price when no PT row
    out = {"PT": _to_24_hours(pt), "ES": _to_24_hours(es)}
    if len(pt) in (92, 96, 100):     # 15-min era: keep the quarters for the ISP file
        out["PT_q"] = _to_96_quarters(pt)
    return out


def dl_ida(day: dt.date, session: str) -> Optional[dict]:
    resp = _get("https://www.omie.es/es/file-download?parents=marginalpibcpt&filename="
                f"marginalpibcpt_{day:%Y%m%d}{session}.1")
    if resp is None:
        return None
    periods = {}
    for line in resp.text.splitlines():
        p = [x.strip() for x in line.split(";")]
        if len(p) < 6 or not p[3].isdigit():
            continue
        try:
            periods[int(p[3])] = float(p[5])
        except ValueError:
            continue
    hours = _periods_to_hours(periods)
    return {"PT": hours} if hours else None


def dl_xbid(day: dt.date) -> Optional[dict]:
    resp = _get("https://www.omie.es/es/file-download?parents=precios_pibcic&filename="
                f"precios_pibcic_{day:%Y%m%d}.1")
    if resp is None:
        return None
    header, periods = None, {}
    for line in resp.content.decode("latin-1").splitlines():
        p = [x.strip() for x in line.split(";")]
        if p and p[0].lower() in ("año", "ano"):
            header = p
            continue
        if header is None or len(p) < len(header) or not p[0].isdigit():
            continue
        row = dict(zip(header, p))
        per = row.get("Periodo") or row.get("Hora")
        val = row.get("MedioPT", "")
        if not per or not val:
            continue          # hour/quarter without continuous trades
        try:
            periods[int(per)] = _num_comma(val)
        except ValueError:
            continue
    hours = _periods_to_hours(periods)
    return {"PT": hours} if hours else None


def _ren(api: str, day: dt.date):
    resp = _get(f"https://mercadoservices.ren.pt/api/{api}/Get{api}?language=PT&dayQuery={day.day}"
                f"&monthQuery={day.month}&yearQuery={day.year}&sWhere=",
                headers={"X-ApiKey": _REN_API_KEY_B64, "Accept": "application/json"})
    if resp is None:
        return None
    rows = resp.json()
    if isinstance(rows, str):
        rows = json.loads(rows)        # REN double-encodes the JSON body
    return rows if isinstance(rows, list) and rows else None


def _f(v):
    return float(v) if v not in (None, "") else None


def dl_afrr(day: dt.date) -> Optional[dict]:
    rows = _ren("BaFRRPreco", day)
    if rows is None:
        return None
    up, dn = {}, {}
    for r in rows:
        per = int(r["PERIODO"])
        u = _f(r.get("PRECO_AJUST_SUB")) or _f(r.get("PRECO_INI_SUB"))
        d = _f(r.get("PRECO_AJUST_DES")) or _f(r.get("PRECO_INI_DES"))
        if u is not None and d is not None:
            up[per], dn[per] = u, d
    n = len(rows)
    if len(up) != n:
        return None
    return {"up": _periods_to_hours(up, n), "dn": _periods_to_hours(dn, n)}


def dl_mfrr(day: dt.date) -> Optional[dict]:
    rows = _ren("MFRRPreco", day)
    if rows is None:
        return None
    ap = {int(r["PERIODO"]): _f(r.get("AP_PRECO")) for r in rows}
    ap = {k: v for k, v in ap.items() if v is not None}
    # Periods without an activation carry no AP_PRECO; keep the hours that are
    # complete -- build_reserve labels the missing hours SYNTHETIC.
    hours = _periods_to_hours(ap) if len(ap) != len(rows) else _periods_to_hours(ap, len(rows))
    return {"ap": hours} if hours else None


# ---------------------------------------------------------------------------
# Cached, parallel fetch
# ---------------------------------------------------------------------------

def fetch_all(name: str, days: List[dt.date], fn: Callable[[dt.date], Optional[dict]]) -> Dict[dt.date, Optional[dict]]:
    cdir = os.path.join(_CACHE, name)
    os.makedirs(cdir, exist_ok=True)
    out: Dict[dt.date, Optional[dict]] = {}
    todo = []
    for d in days:
        p = os.path.join(cdir, f"{d:%Y-%m-%d}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                out[d] = json.load(fh)
        else:
            todo.append(d)

    def one(d):
        for attempt in range(_RETRIES):
            try:
                return d, fn(d)
            except Exception as exc:  # transport / format error -> retry
                err = exc
                time.sleep(2 * (attempt + 1))
        print(f"  [{name}] {d} failed after {_RETRIES} tries: {err}")
        return d, "FAILED"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        futs = [ex.submit(one, d) for d in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            d, res = fut.result()
            if res == "FAILED":
                out[d] = None          # not cached -> retried on the next run
                continue
            out[d] = res
            if res is None and d >= portugal_today() - dt.timedelta(days=3):
                continue               # may simply not be published yet -> retry next run
            with open(os.path.join(cdir, f"{d:%Y-%m-%d}.json"), "w", encoding="utf-8") as fh:
                json.dump(res, fh)
            if i % 250 == 0:
                print(f"  [{name}] {i}/{len(todo)} downloaded ({time.time() - t0:.0f}s)")
    n_ok = sum(1 for d in days if out.get(d))
    print(f"[{name}] {n_ok}/{len(days)} dates with real data "
          f"({len(todo)} downloaded now, {len(days) - len(todo)} from cache)")
    return out


def _days(start: dt.date, end: dt.date) -> List[dt.date]:
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def _jkey(hours: dict) -> Dict[int, float]:
    return {int(k): v for k, v in hours.items()}


def _save(df: pd.DataFrame, path: str, sheet: str) -> None:
    df = df.sort_values([c for c in ("Date", "Hour") if c in df.columns]).reset_index(drop=True)
    # Drop trailing days that are entirely gap-filled (usually not published
    # yet): leaving them would make the live updater treat the file as current
    # and never fetch the real data once it appears.
    real_days = df.loc[df["source"] != "SYNTHETIC", "Date"]
    if not real_days.empty:
        df = df[df["Date"] <= real_days.max()].reset_index(drop=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=sheet, index=False)
    src = df["source"].value_counts().to_dict()
    print(f"  wrote {os.path.relpath(path, _REPO)}: {len(df):,} rows "
          f"{df['Date'].min():%Y-%m-%d} -> {df['Date'].max():%Y-%m-%d}  {src}")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_da(today: dt.date) -> pd.DataFrame:
    got = fetch_all("da", _days(DA_START, today), dl_da)
    rows, prev = [], None
    for d in _days(DA_START, today):
        res = got.get(d)
        if res:
            pt, es, src = _jkey(res["PT"]), _jkey(res["ES"]), "OMIE_LIVE"
            prev = (pt, es)
        elif prev:
            pt, es, src = prev[0], prev[1], "SYNTHETIC"
        else:
            continue
        for h in range(1, 25):
            rows.append({"Date": pd.Timestamp(d), "Hour": h, "price_DA_PT_EUR_MWh": pt[h],
                         "price_DA_ES_EUR_MWh": es[h], "source": src})
    df = pd.DataFrame(rows)
    _save(df, DA_XLSX, DA_SHEET)

    # 15-min file: same downloads, Portuguese row, all 96 quarter-hours.
    isp_rows, prev_q = [], None
    for d in _days(ISP_START, today):
        res = got.get(d)
        if res and res.get("PT_q"):
            q, src = res["PT_q"], "OMIE_LIVE"
            prev_q = q
        elif prev_q:
            q, src = prev_q, "SYNTHETIC"
        else:
            continue
        for i, v in enumerate(q, start=1):
            isp_rows.append({"Date": pd.Timestamp(d), "ISP": i, "price_DA_PT_EUR_MWh": v, "source": src})
    isp = pd.DataFrame(isp_rows)
    isp = isp.sort_values(["Date", "ISP"]).reset_index(drop=True)
    with pd.ExcelWriter(ISP_XLSX, engine="openpyxl") as w:
        isp.to_excel(w, sheet_name=ISP_SHEET, index=False)
    print(f"  wrote {os.path.relpath(ISP_XLSX, _REPO)}: {len(isp):,} rows "
          f"{isp['Date'].min():%Y-%m-%d} -> {isp['Date'].max():%Y-%m-%d}  {isp['source'].value_counts().to_dict()}")
    return df


def _da_lookup(da: pd.DataFrame) -> Dict[tuple, float]:
    return {(r.Date.date(), int(r.Hour)): r.price_DA_PT_EUR_MWh for r in da.itertuples()}


def build_intraday(key: str, path: str, sheet: str, price_col: str, days: List[dt.date],
                   got: Dict[dt.date, Optional[dict]], da_px: Dict[tuple, float],
                   product_hours: List[int]) -> None:
    """24 rows per day, the live loaders' layout. Hours the gate trades take
    the real clearing price (SYNTHETIC zero spread if OMIE has none); hours the
    gate does not trade (IDA3's frozen H1-H12) settle at the DA price with the
    day's own source label, exactly as omie_ida3_price_loader writes them."""
    rows = []
    for d in days:
        res = got.get(d)
        real = _jkey(res["PT"]) if res else {}
        day_src = "OMIE_LIVE" if real else "SYNTHETIC"
        for h in range(1, 25):
            da = da_px.get((d, h))
            if da is None:
                continue
            if h not in product_hours:
                px, src = da, day_src             # frozen hour: not traded in this gate
            elif h in real:
                px, src = real[h], "OMIE_LIVE"
            else:
                px, src = da, "SYNTHETIC"          # no trade/auction data -> zero spread
            rows.append({"Date": pd.Timestamp(d), "Hour": h, "price_DA_PT_EUR_MWh": da,
                         price_col: px, "spread_EUR_MWh": round(px - da, 2), "source": src})
    _save(pd.DataFrame(rows), path, sheet)


def build_reserve(name: str, path: str, sheet: str, days: List[dt.date],
                  got: Dict[dt.date, Optional[dict]], da_px: Dict[tuple, float],
                  up_key: str, dn_key: str) -> None:
    rows, prev = [], None
    for d in days:
        res = got.get(d)
        up = _jkey(res[up_key]) if res else {}
        dn = _jkey(res[dn_key]) if res else {}
        if not up and prev is None:
            continue
        for h in range(1, 25):
            if h in up and h in dn:
                u, w, src = up[h], dn[h], "REN_LIVE"
            elif up:                                    # partial day: day mean of real hours
                u = round(sum(up.values()) / len(up), 2)
                w = round(sum(dn.values()) / len(dn), 2)
                src = "SYNTHETIC"
            else:                                       # no data: carry previous day forward
                u, w, src = prev[0][h], prev[1][h], "SYNTHETIC"
            rows.append({"Date": pd.Timestamp(d), "Hour": h,
                         "price_DA_PT_EUR_MWh": da_px.get((d, h)),
                         "cap_up_EUR_MW": u, "cap_dn_EUR_MW": w, "source": src})
        day = {r["Hour"]: r for r in rows[-24:]}
        prev = ({h: day[h]["cap_up_EUR_MW"] for h in day}, {h: day[h]["cap_dn_EUR_MW"] for h in day})
    _save(pd.DataFrame(rows), path, sheet)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", default="da,ida1,ida2,ida3,xbid,afrr,mfrr")
    args = ap.parse_args()
    only = set(args.only.split(","))
    today = portugal_today()
    yesterday = today - dt.timedelta(days=1)
    print(f"Rebuilding real market history up to {today} (Portugal) -> {sorted(only)}")

    da = build_da(today) if "da" in only else pd.read_excel(DA_XLSX, sheet_name=DA_SHEET)
    da["Date"] = pd.to_datetime(da["Date"])
    da_px = _da_lookup(da)

    for key, (session, path, sheet) in _IDA.items():
        if key not in only:
            continue
        days = _days(IDA_START, today)
        got = fetch_all(key, days, lambda d, s=session: dl_ida(d, s))
        build_intraday(key, path, sheet, "price_IDA_PT_EUR_MWh", days, got, da_px,
                       list(range(13, 25)) if key == "ida3" else list(range(1, 25)))   # IDA3 = periods 49-96
    if "xbid" in only:
        days = _days(IDA_START, yesterday)
        got = fetch_all("xbid", days, dl_xbid)
        build_intraday("xbid", XBID_XLSX, XBID_SHEET, "price_XBID_PT_EUR_MWh", days, got, da_px,
                       list(range(1, 25)))
    if "afrr" in only:
        days = _days(AFRR_START, today)
        got = fetch_all("afrr", days, dl_afrr)
        build_reserve("afrr", AFRR_XLSX, AFRR_SHEET, days, got, da_px, "up", "dn")
    if "mfrr" in only:
        days = _days(MFRR_START, yesterday)
        got = fetch_all("mfrr", days, dl_mfrr)
        build_reserve("mfrr", MFRR_XLSX, MFRR_SHEET, days, got, da_px, "ap", "ap")


if __name__ == "__main__":
    main()
