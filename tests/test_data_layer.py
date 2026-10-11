"""
test_data_layer.py — Phase 2 exit criterion and the pieces behind it.

Uses real raw files saved in tests/fixtures/raw (including the three daylight-saving days
2025-03-30, 2025-10-26 and 2026-03-29). No network: the HTTP transport is faked.
"""
from __future__ import annotations

import datetime as dt
import os

import pandas as pd
import pytest

from common_layer.data_layer import calendar as cal
from common_layer.data_layer import fetchers, ingest, parsers, regimes
from common_layer.data_layer.clean_store import CleanStore
from common_layer.data_layer.contracts import DATASETS, freshness_issues, validate_day
from common_layer.data_layer.raw_store import RawIntegrityError, RawStore

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "raw")
D_SPRING, D_AUTUMN, D_SPRING26 = dt.date(2025, 3, 30), dt.date(2025, 10, 26), dt.date(2026, 3, 29)


def fx(name: str) -> bytes:
    with open(os.path.join(FIX, name), "rb") as fh:
        return fh.read()


# ------------------------------------------------------------------ calendar
def test_dst_period_counts_and_utc_starts():
    assert [cal.hours_in_day(d) for d in (D_SPRING, D_AUTUMN, dt.date(2025, 6, 1))] == [23, 25, 24]
    assert cal.expected_periods(D_SPRING26, 15) == 92 and cal.expected_periods(D_AUTUMN, 15) == 100
    # autumn day: local midnight is 22:00Z the day before; the two "hour 3" slots are distinct UTC times
    assert cal.period_start_utc(D_AUTUMN, 1, 60) == dt.datetime(2025, 10, 25, 22, tzinfo=dt.timezone.utc)
    assert cal.period_start_utc(D_AUTUMN, 3, 60) != cal.period_start_utc(D_AUTUMN, 4, 60)
    assert cal.period_start_utc(D_AUTUMN, 100, 15) == dt.datetime(2025, 10, 26, 22, 45, tzinfo=dt.timezone.utc)


def test_model_grid_policy_keeps_24_or_96_slots():
    assert len(cal.to_model_grid(list(range(23)), D_SPRING)) == 24
    assert len(cal.to_model_grid(list(range(25)), D_AUTUMN)) == 24
    assert len(cal.to_model_grid(list(range(92)), D_SPRING26)) == 96
    assert len(cal.to_model_grid(list(range(100)), D_AUTUMN)) == 96
    with pytest.raises(ValueError):
        cal.to_model_grid(list(range(50)), D_AUTUMN)


# ------------------------------------------------------------------- regimes
def test_regimes():
    assert regimes.regime_at(dt.date(2025, 9, 30))["da_resolution_min"] == 60
    assert regimes.regime_at(dt.date(2025, 10, 1))["da_resolution_min"] == 15
    assert regimes.regime_at(dt.date(2024, 1, 1))["ida_sessions"] == 6
    assert regimes.regime_at(dt.date(2025, 1, 1))["ida_sessions"] == 3
    assert regimes.regime_at(dt.date(2026, 5, 28))["price_floor_eur_mwh"] == -500.0
    assert regimes.regime_at(dt.date(2026, 5, 29))["price_floor_eur_mwh"] == -600.0
    assert regimes.regime_at(dt.date(2023, 1, 1))["iberian_exception"] is True
    assert regimes.regime_at(dt.date(2025, 1, 1))["iberian_exception"] is False


# ------------------------------------------------------------------- parsers
def test_da_parser_true_periods_on_dst_days():
    spring = parsers.parse_omie_da(fx("da_20250330.txt"), D_SPRING)
    autumn = parsers.parse_omie_da(fx("da_20251026.txt"), D_AUTUMN)
    spring26 = parsers.parse_omie_da(fx("da_20260329.txt"), D_SPRING26)
    assert (len(spring), set(spring.resolution_min)) == (23, {60})
    assert (len(autumn), set(autumn.resolution_min)) == (100, {15})
    assert (len(spring26), set(spring26.resolution_min)) == (92, {15})
    assert autumn.start_utc.is_unique and autumn.start_utc.is_monotonic_increasing
    assert spring.price_pt_eur_mwh.notna().all() and autumn.flow_es_to_pt_mw.notna().all()


def test_da_parser_rejects_wrong_day_and_garbage():
    with pytest.raises(parsers.ParseError):
        parsers.parse_omie_da(fx("da_20251026.txt"), dt.date(2025, 10, 27))
    with pytest.raises(parsers.ParseError):
        parsers.parse_omie_da(b"<html>Service unavailable</html>", D_AUTUMN)


def test_ida_xbid_and_ren_parsers():
    ida = parsers.parse_omie_ida(fx("ida1_20251026.txt"), D_AUTUMN, 1)
    assert len(ida) == 100 and set(ida.session) == {1}
    xb = parsers.parse_omie_xbid(fx("xbid_20251026.txt"), D_AUTUMN)
    assert len(xb) == 100
    no_trade = xb[xb.pt_traded == 0]
    assert len(no_trade) > 0 and no_trade.price_pt_mean_eur_mwh.isna().all()    # never a price of 0
    assert xb[xb.pt_traded == 1].price_pt_mean_eur_mwh.notna().all()
    for name, ds in (("ren_afrr_20260329.json", "afrr_price"), ("ren_MFRRPreco_20260329.json", "mfrr_price"),
                     ("ren_DesvioPreco_20260329.json", "imbalance_price")):
        df = parsers.parse_ren(fx(name), D_SPRING26, ds)
        assert len(df) == 92 and df.start_utc.iloc[0] == "2026-03-28T23:00:00Z"


def test_parsers_are_deterministic():
    a = parsers.parse_omie_da(fx("da_20251026.txt"), D_AUTUMN)
    b = parsers.parse_omie_da(fx("da_20251026.txt"), D_AUTUMN)
    pd.testing.assert_frame_equal(a, b)


# ----------------------------------------------------------------- raw store
def test_raw_store_versions_unchanged_and_integrity(tmp_path):
    raw = RawStore(str(tmp_path))
    r1 = raw.save("omie_da", "2025-10-26", b"first")
    assert (r1.status, r1.version) == ("new", 1)
    again = raw.save("omie_da", "2025-10-26", b"first")
    assert (again.status, again.version) == ("unchanged", 1)
    r2 = raw.save("omie_da", "2025-10-26", b"revised")
    assert (r2.status, r2.version) == ("new", 2)
    assert raw.read(raw.latest("omie_da", "2025-10-26")) == b"revised"
    assert raw.read(raw.versions("omie_da", "2025-10-26")[0]) == b"first"     # old version still there
    assert raw.verify_all() == []
    # tamper with a stored file: detected
    with open(os.path.join(raw.root, r1.path), "wb") as fh:
        fh.write(b"tampered")
    assert raw.verify_all() != []
    with pytest.raises(RawIntegrityError):
        raw.read(r1)


def test_raw_store_as_of_read(tmp_path):
    raw = RawStore(str(tmp_path))
    t1 = dt.datetime(2026, 1, 1, 12, tzinfo=dt.timezone.utc)
    t2 = dt.datetime(2026, 2, 1, 12, tzinfo=dt.timezone.utc)
    raw.save("omie_da", "2025-10-26", b"v1", retrieved_at=t1)
    raw.save("omie_da", "2025-10-26", b"v2", retrieved_at=t2)
    assert raw.latest("omie_da", "2025-10-26", as_of_utc=t1 + dt.timedelta(days=1)).version == 1
    assert raw.latest("omie_da", "2025-10-26").version == 2
    assert raw.latest("omie_da", "2025-10-26", as_of_utc=t1 - dt.timedelta(days=1)) is None


# ----------------------------------------------------------------- contracts
def test_contract_accepts_real_dst_days():
    for name, day, res in (("da_20250330.txt", D_SPRING, None), ("da_20251026.txt", D_AUTUMN, None),
                           ("da_20260329.txt", D_SPRING26, None)):
        r = validate_day("da_price", parsers.parse_omie_da(fx(name), day), day)
        assert r.ok, r.issues


def test_contract_rejects_missing_period_duplicates_and_bad_time():
    day = D_AUTUMN
    df = parsers.parse_omie_da(fx("da_20251026.txt"), day)
    assert not validate_day("da_price", df.iloc[:-1], day).ok                       # 99 of 100
    dup = pd.concat([df, df.iloc[[0]]])
    assert not validate_day("da_price", dup, day).ok
    shifted = df.copy()
    shifted["start_utc"] = shifted["start_utc"].shift(-1, fill_value="2025-10-27T00:00:00Z")
    assert not validate_day("da_price", shifted, day).ok


def test_contract_range_is_a_warning_not_an_error():
    day = D_AUTUMN
    df = parsers.parse_omie_da(fx("da_20251026.txt"), day)
    df.loc[5, "price_pt_eur_mwh"] = 99999.0
    r = validate_day("da_price", df, day)
    assert r.ok and "range" in r.flags()


def test_freshness():
    latest = {n: dt.date(2026, 10, 10) for n in DATASETS}
    assert freshness_issues(latest, dt.date(2026, 10, 10)) == []
    stale = dict(latest, afrr_price=dt.date(2026, 9, 1))
    assert any("afrr_price" in i.message for i in freshness_issues(stale, dt.date(2026, 10, 10)))


# ------------------------------------------------------- fetcher and ingest
class FakeHttp:
    """Serves saved files by URL; can be told to fail first, or to report 404."""

    def __init__(self, routes, fail_first=0):
        self.routes, self.fail_first, self.calls = routes, fail_first, []

    def __call__(self, url, headers, timeout):
        self.calls.append(url)
        if self.fail_first > 0:
            self.fail_first -= 1
            return 503, b""
        if url in self.routes:
            return 200, self.routes[url]
        return 404, b""


def fetcher_with(http, **kw):
    return fetchers.Fetcher(http, min_interval_s=0.0, backoff_s=0.0, sleep=lambda s: None, **kw)


def test_fetcher_retries_then_succeeds_and_404_is_not_published():
    url = fetchers.url_omie_da(D_AUTUMN)
    http = FakeHttp({url: fx("da_20251026.txt")}, fail_first=2)
    got = fetcher_with(http).get(url)
    assert got.attempts == 3 and len(got.content) > 1000
    with pytest.raises(fetchers.NotPublished):
        fetcher_with(FakeHttp({})).get(url)
    with pytest.raises(fetchers.FetchError):
        fetcher_with(FakeHttp({}, fail_first=99), retries=2).get(url)


def routes_for_autumn():
    return {
        fetchers.url_omie_da(D_AUTUMN): fx("da_20251026.txt"),
        fetchers.url_omie_ida(D_AUTUMN, 1): fx("ida1_20251026.txt"),
        fetchers.url_omie_xbid(D_AUTUMN): fx("xbid_20251026.txt"),
        fetchers.url_omie_da(D_SPRING26): fx("da_20260329.txt"),
        fetchers.url_ren("BaFRRPreco/GetBaFRRPreco", D_SPRING26): fx("ren_afrr_20260329.json"),
        fetchers.url_ren("MFRRPreco/GetMFRRPreco", D_SPRING26): fx("ren_MFRRPreco_20260329.json"),
        fetchers.url_ren("DesvioPreco/GetDesvioPreco", D_SPRING26): fx("ren_DesvioPreco_20260329.json"),
    }


@pytest.fixture
def zones(tmp_path):
    return (RawStore(str(tmp_path / "raw")), CleanStore(str(tmp_path / "clean.db")),
            ingest.LoadLog(str(tmp_path / "load_log.jsonl")))


def test_ingest_loads_every_source_and_is_idempotent(zones):
    raw, clean, log = zones
    http = FakeHttp(routes_for_autumn())
    f = fetcher_with(http)
    c1 = ingest.ingest_range(["omie_da", "omie_ida1", "omie_xbid"], D_AUTUMN, D_AUTUMN,
                             raw=raw, clean=clean, fetcher=f, log=log)
    assert c1 == {"loaded": 3}
    assert len(clean.read("da_price")) == 100 and len(clean.read("ida_price")) == 100
    digest = {d: clean.content_digest(d) for d in ("da_price", "ida_price", "xbid_price")}
    # second run: skipped, no new download
    n_calls = len(http.calls)
    c2 = ingest.ingest_range(["omie_da", "omie_ida1", "omie_xbid"], D_AUTUMN, D_AUTUMN,
                             raw=raw, clean=clean, fetcher=f, log=log)
    assert c2 == {"skipped": 3} and len(http.calls) == n_calls
    # forced refresh with identical bytes: "unchanged", clean data identical
    c3 = ingest.ingest_range(["omie_da"], D_AUTUMN, D_AUTUMN, raw=raw, clean=clean, fetcher=f, log=log,
                             refresh=True)
    assert c3 == {"unchanged": 1} and clean.content_digest("da_price") == digest["da_price"]
    assert {r["status"] for r in log.read()} == {"loaded", "skipped", "unchanged"}


def test_ingest_rejects_html_error_page_with_http_200(zones):
    raw, clean, log = zones
    http = FakeHttp({fetchers.url_omie_da(D_AUTUMN): b"<html><body>Error 500</body></html>"})
    out = ingest.ingest_day("omie_da", D_AUTUMN, raw=raw, clean=clean, fetcher=fetcher_with(http), log=log)
    assert out["status"] == "rejected" and raw.keys("omie_da") == [] and clean.days("da_price") == []


def test_ingest_reports_not_published_and_failed(zones):
    raw, clean, log = zones
    assert ingest.ingest_day("omie_da", D_AUTUMN, raw=raw, clean=clean, fetcher=fetcher_with(FakeHttp({})),
                             log=log)["status"] == "not_published"
    bad = fetcher_with(FakeHttp({}, fail_first=99), retries=1)
    assert ingest.ingest_day("omie_da", D_AUTUMN, raw=raw, clean=clean, fetcher=bad, log=log)["status"] == "failed"


def test_legacy_ida_days_are_not_mixed_in(zones):
    raw, clean, log = zones
    out = ingest.ingest_day("omie_ida1", dt.date(2024, 1, 10), raw=raw, clean=clean,
                            fetcher=fetcher_with(FakeHttp({})), log=log)
    assert out["status"] == "out_of_regime"


def test_revised_source_file_creates_new_version_and_updates_clean(zones):
    raw, clean, log = zones
    day = D_AUTUMN
    original = fx("da_20251026.txt")
    revised = original.replace(b"105,19", b"106,19")         # every occurrence, Spanish and Portuguese rows
    assert revised != original
    url = fetchers.url_omie_da(day)
    ingest.ingest_day("omie_da", day, raw=raw, clean=clean, fetcher=fetcher_with(FakeHttp({url: original})),
                      log=log)
    out = ingest.ingest_day("omie_da", day, raw=raw, clean=clean, fetcher=fetcher_with(FakeHttp({url: revised})),
                            log=log, refresh=True)
    assert out["status"] == "loaded" and out["raw_version"] == 2
    assert clean.read("da_price").price_pt_eur_mwh.iloc[0] == pytest.approx(106.19)


def test_rebuild_as_of_gives_the_state_known_at_that_time(zones):
    raw, clean, _ = zones
    day = D_AUTUMN
    original = fx("da_20251026.txt")
    t1 = dt.datetime(2026, 1, 27, 12, tzinfo=dt.timezone.utc)
    t2 = dt.datetime(2026, 3, 1, 12, tzinfo=dt.timezone.utc)
    raw.save("omie_da", day.isoformat(), original, retrieved_at=t1, ext=".txt")
    raw.save("omie_da", day.isoformat(), original.replace(b"105,19", b"106,19"), retrieved_at=t2, ext=".txt")
    ingest.rebuild_clean("omie_da", [day], raw=raw, clean=clean)
    assert clean.read("da_price").price_pt_eur_mwh.iloc[0] == pytest.approx(106.19)
    ingest.rebuild_clean("omie_da", [day], raw=raw, clean=clean, as_of_utc=t1 + dt.timedelta(days=1))
    assert clean.read("da_price").price_pt_eur_mwh.iloc[0] == pytest.approx(105.19)


def test_exit_criterion_clean_zone_rebuilds_exactly_from_raw(zones, tmp_path):
    """Delete the whole clean zone; rebuilding from the raw zone alone gives identical data."""
    raw, clean, log = zones
    f = fetcher_with(FakeHttp(routes_for_autumn()))
    plan = [("omie_da", D_AUTUMN), ("omie_ida1", D_AUTUMN), ("omie_xbid", D_AUTUMN), ("omie_da", D_SPRING26),
            ("ren_afrr", D_SPRING26), ("ren_mfrr", D_SPRING26), ("ren_imbalance", D_SPRING26)]
    for src, day in plan:
        assert ingest.ingest_day(src, day, raw=raw, clean=clean, fetcher=f, log=log)["status"] == "loaded"
    before = {ds: clean.content_digest(ds) for ds in DATASETS}

    rebuilt = CleanStore(str(tmp_path / "rebuilt.db"))                      # brand-new empty clean zone
    for src, day in plan:
        assert ingest.rebuild_clean(src, [day], raw=raw, clean=rebuilt) == {"loaded": 1}
    after = {ds: rebuilt.content_digest(ds) for ds in DATASETS}
    assert before == after
    assert raw.verify_all() == []
    # lineage: every clean row points to the raw file it came from
    lin = rebuilt.read("da_price", lineage=True)
    assert lin.raw_sha256.str.len().eq(64).all() and (lin.parser_version == parsers.PARSER_VERSION).all()


# ------------------------------------------- findings from the first live run
def test_ida3_covers_only_the_last_12_hours_and_passes_the_contract():
    ida3 = parsers.parse_omie_ida(fx("ida3_20251026.txt"), D_AUTUMN, 3)
    assert len(ida3) == 48 and ida3.period_index.min() == 53 and ida3.period_index.max() == 100
    assert validate_day("ida_price", ida3, D_AUTUMN).ok
    assert not validate_day("ida_price", ida3.iloc[1:], D_AUTUMN).ok          # 47 periods
    ida1 = parsers.parse_omie_ida(fx("ida1_20251026.txt"), D_AUTUMN, 1)
    assert not validate_day("ida_price", ida1.iloc[48:], D_AUTUMN).ok         # IDA1 must be the full day


def test_empty_intraday_file_is_reported_not_stored(zones):
    raw, clean, log = zones
    url = fetchers.url_omie_ida(dt.date(2025, 10, 27), 1)
    http = FakeHttp({url: fx("ida1_20251027_empty.txt")})
    out = ingest.ingest_day("omie_ida1", dt.date(2025, 10, 27), raw=raw, clean=clean,
                            fetcher=fetcher_with(http), log=log)
    assert out["status"] == "empty_publication" and raw.keys("omie_ida1") == []
