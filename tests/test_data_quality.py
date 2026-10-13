"""
test_data_quality.py — Phase 3: quality checks, flags, imputation rules, provenance, score and gate.

Series are built directly in the clean-zone format (no network). Real saved files are used where the
check is about real market behaviour (the repeated hour of the 2025-10-26 quarter-hour day).
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import pytest

from common_layer.data_layer import calendar as cal
from common_layer.data_layer import parsers
from common_layer.data_layer.clean_store import CleanStore
from common_layer.data_layer.ingest import LoadLog
from common_layer.data_layer.raw_store import RawStore
from common_layer.data_quality import checks, imputation, provenance, runner, scoring
from common_layer.data_quality.config import (DataQualityConfigError, load_quality_config,
                                              validate_quality_config)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "raw")
CFG = load_quality_config()
D0 = dt.date(2021, 3, 1)


def da_day(day: dt.date, prices, res: int = 60) -> pd.DataFrame:
    n = len(prices)
    assert n == cal.expected_periods(day, res)
    df = pd.DataFrame({
        "delivery_date": day.isoformat(), "period_index": range(1, n + 1), "resolution_min": res,
        "start_utc": [cal.period_start_utc(day, i, res).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(1, n + 1)],
        "price_pt_eur_mwh": prices})
    for c in ("buy_pt_mw", "sell_pt_mw"):
        df[c] = 100.0
    df["issued_utc"], df["flags"] = "", ""
    return df


def put(store: CleanStore, dataset: str, df: pd.DataFrame, flags: str = "") -> None:
    store.replace_day(dataset, df, raw_sha256="x" * 64, raw_version=1, parser_version="1",
                      ingest_id="t", quality_flags=flags)


def wavy(day: dt.date, base: float = 50.0, amp: float = 20.0, n: int = 24) -> list:
    d = day.toordinal() % 7
    return [round(base + amp * np.sin(2 * np.pi * (h + d) / n) + 0.01 * d, 2) for h in range(n)]


@pytest.fixture
def store(tmp_path):
    return CleanStore(str(tmp_path / "clean.db"))


def fill_da(store, days=40, start=D0, price_fn=None):
    out = []
    for i in range(days):
        d = start + dt.timedelta(days=i)
        n = cal.expected_periods(d, 60)
        p = price_fn(d, i) if price_fn else wavy(d, n=n)
        put(store, "da_price", da_day(d, p))
        out.append(d)
    return out


# ------------------------------------------------------------------- config
def test_config_is_valid_and_demands_a_written_imputation_rule():
    assert validate_quality_config(CFG) == []
    broken = copy.deepcopy(CFG)
    del broken["imputation"]["da_price.price_pt_eur_mwh"]
    assert any("no imputation rule" in p for p in validate_quality_config(broken))
    broken = copy.deepcopy(CFG)
    broken["imputation"]["afrr_price.price_adjusted_up"]["method"] = "magic"
    assert any("method must be" in p for p in validate_quality_config(broken))


# ----------------------------------------------------------- stuck / stale
def test_hourly_pinned_price_is_not_a_stuck_feed_but_a_constant_day_is(store):
    pinned = [1.95] * 9 + [5.0 + h for h in range(15)]                 # real behaviour seen in 2020-2021
    put(store, "da_price", da_day(D0, pinned))
    put(store, "da_price", da_day(D0 + dt.timedelta(days=1), [7.0] * 24))
    out = checks.check_flat_runs("da_price", store.read("da_price", lineage=True), CFG)
    assert set(out["delivery_date"]) == {(D0 + dt.timedelta(days=1)).isoformat()}


def test_duplicate_day_is_flagged_only_when_the_whole_day_repeats(store):
    a = wavy(D0)
    put(store, "da_price", da_day(D0, a))
    put(store, "da_price", da_day(D0 + dt.timedelta(days=1), a))                     # stale file served again
    put(store, "da_price", da_day(D0 + dt.timedelta(days=2), wavy(D0 + dt.timedelta(days=2), base=70)))
    out = checks.check_duplicate_day("da_price", store.read("da_price", lineage=True), CFG)
    assert list(out["delivery_date"]) == [(D0 + dt.timedelta(days=1)).isoformat()]


def test_dst_copy_check_on_real_autumn_day_and_on_a_forged_copy(store):
    day = dt.date(2025, 10, 26)
    real = parsers.parse_omie_da(open(os.path.join(FIX, "da_20251026.txt"), "rb").read(), day)
    put(store, "da_price", real)
    assert checks.check_dst_copy("da_price", store.read("da_price", lineage=True), CFG).empty
    forged = real.copy()
    forged.loc[12:15, "price_pt_eur_mwh"] = forged.loc[8:11, "price_pt_eur_mwh"].to_numpy()   # periods 13-16 = 9-12
    put(store, "da_price", forged)
    out = checks.check_dst_copy("da_price", store.read("da_price", lineage=True), CFG)
    assert list(out["check"]) == ["dst_copy_suspect"]


# ------------------------------------------------------ value flags (info)
def test_spike_and_negative_flags_are_information_and_causal(store):
    days = fill_da(store, 30)
    spike_day = days[-1]
    df = da_day(spike_day, wavy(spike_day))
    df.loc[10, "price_pt_eur_mwh"] = 600.0
    df.loc[11, "price_pt_eur_mwh"] = -40.0
    put(store, "da_price", df)
    out = checks.check_value_flags("da_price", store.read("da_price", lineage=True), CFG)
    day_rows = out[out["delivery_date"] == spike_day.isoformat()]
    assert {"spike_up", "negative_price"} <= set(day_rows["check"]) and (out["severity"] == "info").all()
    assert out[out["delivery_date"] < spike_day.isoformat()].empty           # calm history raises nothing
    # causality: changing later values does not change flags on earlier days
    later = spike_day + dt.timedelta(days=1)
    put(store, "da_price", da_day(later, [900.0] * 24))
    out2 = checks.check_value_flags("da_price", store.read("da_price", lineage=True), CFG)
    pd.testing.assert_frame_equal(out2[out2["delivery_date"] <= spike_day.isoformat()].reset_index(drop=True),
                                  out.reset_index(drop=True))


def test_level_jump_catches_a_unit_error(store):
    fill_da(store, 45, price_fn=lambda d, i: [round(v * (100 if i >= 40 else 1), 2) for v in wavy(d, n=cal.expected_periods(d, 60))])
    out = checks.check_level_jump("da_price", store.read("da_price", lineage=True), CFG)
    assert out["delivery_date"].min() == (D0 + dt.timedelta(days=40)).isoformat()


# ------------------------------------------------------------ cross-source
def test_cross_source_reference_matches_hourly_da_on_quarter_rows(store):
    d = dt.date(2025, 6, 10)                                                         # hourly DA, quarter REN rows
    prices = wavy(d)
    put(store, "da_price", da_day(d, prices))
    q = pd.DataFrame({"delivery_date": d.isoformat(), "period_index": range(1, 97), "resolution_min": 15,
                      "start_utc": [cal.period_start_utc(d, i, 15).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(1, 97)]})
    q["price_short_eur_mwh"] = q["price_long_eur_mwh"] = 1.0
    q["price_da_ref_eur_mwh"] = [prices[(i - 1) // 4] for i in range(1, 97)]
    put(store, "imbalance_price", q)
    frames = {"da_price": store.read("da_price"), "imbalance_price": store.read("imbalance_price")}
    assert checks.check_cross_source(frames, CFG).empty
    q["price_da_ref_eur_mwh"] = q["price_da_ref_eur_mwh"] / 100.0                 # cents instead of euros
    put(store, "imbalance_price", q)
    frames["imbalance_price"] = store.read("imbalance_price")
    out = checks.check_cross_source(frames, CFG)
    assert list(out["check"]) == ["cross_mismatch"]


# -------------------------------------------------------- gaps, score, gate
def test_gap_cause_comes_from_the_load_log(store):
    put(store, "da_price", da_day(D0, wavy(D0)))
    put(store, "da_price", da_day(D0 + dt.timedelta(days=3), wavy(D0)))
    status = {("omie_da", (D0 + dt.timedelta(days=1)).isoformat()): "not_published"}
    out = checks.check_gaps("da_price", store.read("da_price"), [(0, "omie_da")], D0, D0 + dt.timedelta(days=3), status)
    causes = dict(zip(out["delivery_date"], out["detail"]))
    assert causes == {(D0 + dt.timedelta(days=1)).isoformat(): "not_published",
                      (D0 + dt.timedelta(days=2)).isoformat(): "not_attempted"}


def test_score_penalises_warnings_not_information():
    uni = pd.DataFrame({"dataset": "da_price", "session": 0,
                        "delivery_date": ["2021-03-01", "2021-03-02", "2021-03-03", "2021-03-04"],
                        "present": [True, True, True, False]})
    iss = pd.DataFrame([
        checks._issue("da_price", "2021-03-01", 0, "spike_up", "info", 3.0),
        checks._issue("da_price", "2021-03-02", 0, "flat_run", "warning", 24.0),
        checks._issue("da_price", "2021-03-02", 0, "range", "warning", 1.0)], columns=checks.COLUMNS)
    s = scoring.daily_scores(uni, iss, CFG).set_index("delivery_date")["score"]
    assert s["2021-03-01"] == 100 and s["2021-03-02"] == 100 - 25 - 10 and s["2021-03-03"] == 100 and s["2021-03-04"] == 0


def run_da(store, tmp_path, days, mode="strict", **kw):
    cfg = copy.deepcopy(CFG)
    cfg["gate"]["mode"] = mode
    cfg["datasets"]["da_price"]["expected_start"] = D0.isoformat()
    return runner.run_quality(store, RawStore(str(tmp_path / "raw")), LoadLog(str(tmp_path / "log.jsonl")),
                              start=D0, end=D0 + dt.timedelta(days=days - 2), today=D0 + dt.timedelta(days=days - 1),
                              cfg=cfg, datasets=["da_price"])


def test_the_repair_gate_does_not_stop_a_run_for_missing_days_but_lists_the_repairs(store, tmp_path):
    gappy = CleanStore(str(tmp_path / "gappy.db"))
    for i in range(40):
        if not 10 <= i < 15:                                                   # 5 of 39 days missing: over the strict limit, under the repair ceiling
            d = D0 + dt.timedelta(days=i)
            put(gappy, "da_price", da_day(d, wavy(d, n=cal.expected_periods(d, 60))))
    assert not run_da(gappy, tmp_path, 40, mode="strict").gate.passed
    run = run_da(gappy, tmp_path, 40, mode="repair")
    assert run.gate.passed and not run.gate.reasons
    assert any("5/39 days missing" in r and "curation" in r for r in run.gate.repairs)
    cfg = copy.deepcopy(CFG)
    cfg["gate"]["mode"], cfg["gate"]["max_repaired_share"] = "repair", 0.05                 # a hole bigger than the ceiling is too little to repair from
    cfg["datasets"]["da_price"]["expected_start"] = D0.isoformat()
    red = runner.run_quality(gappy, RawStore(str(tmp_path / "raw")), LoadLog(str(tmp_path / "log.jsonl")), start=D0, end=D0 + dt.timedelta(days=38),
                             today=D0 + dt.timedelta(days=39), cfg=cfg, datasets=["da_price"])
    assert not red.gate.passed and any("too little real data" in r for r in red.gate.reasons)


def test_gate_green_on_clean_data_and_red_on_gaps_or_stale(store, tmp_path):
    fill_da(store, 40)
    run = run_da(store, tmp_path, 40)
    assert run.gate.passed, run.gate.reasons
    # a hole of 5 days (12.5%) is over the 2 % limit
    gappy = CleanStore(str(tmp_path / "gappy.db"))
    for i in range(40):
        if not 10 <= i < 15:
            d = D0 + dt.timedelta(days=i)
            put(gappy, "da_price", da_day(d, wavy(d, n=cal.expected_periods(d, 60))))
    red = run_da(gappy, tmp_path, 40)
    assert not red.gate.passed and any("missing" in r for r in red.gate.reasons)
    # stale: data end well before "today"
    cfg = copy.deepcopy(CFG)
    cfg["datasets"]["da_price"]["expected_start"] = D0.isoformat()
    stale = runner.run_quality(store, RawStore(str(tmp_path / "raw")), LoadLog(str(tmp_path / "log.jsonl")),
                               start=D0, end=D0 + dt.timedelta(days=38), today=D0 + dt.timedelta(days=60),
                               cfg=cfg, datasets=["da_price"])
    assert not stale.gate.passed and any("stale feed" in r for r in stale.gate.reasons)


def test_gate_red_when_a_raw_file_is_tampered(store, tmp_path):
    fill_da(store, 10)
    raw = RawStore(str(tmp_path / "raw"))
    rec = raw.save("omie_da", D0.isoformat(), b"data")
    with open(os.path.join(raw.root, rec.path), "wb") as fh:
        fh.write(b"tampered")
    cfg = copy.deepcopy(CFG)
    cfg["datasets"]["da_price"]["expected_start"] = D0.isoformat()
    run = runner.run_quality(store, raw, LoadLog(str(tmp_path / "log.jsonl")), start=D0,
                             end=D0 + dt.timedelta(days=8), today=D0 + dt.timedelta(days=9), cfg=cfg,
                             datasets=["da_price"])
    assert not run.gate.passed and any("checksum" in r for r in run.gate.reasons)


def test_require_green_blocks_missing_red_and_stale_results(store, tmp_path):
    qdir = str(tmp_path / "q")
    with pytest.raises(runner.DataQualityGateError, match="no data-quality result"):
        runner.require_green(qdir)
    fill_da(store, 40)
    green = run_da(store, tmp_path, 40)
    paths = runner.store_results(green, qdir, str(tmp_path / "rep"))
    assert runner.require_green(qdir)["passed"] is True
    assert os.path.exists(paths["report"]) and "GREEN" in open(paths["report"], encoding="utf-8").read()
    later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=48)
    with pytest.raises(runner.DataQualityGateError, match="old"):
        runner.require_green(qdir, now=later)
    g = json.load(open(paths["gate"]))
    g["passed"], g["reasons"] = False, ["da_price: test"]
    json.dump(g, open(paths["gate"], "w"))
    with pytest.raises(runner.DataQualityGateError, match="RED"):
        runner.require_green(qdir)


# --------------------------------------------------------------- imputation
def series(values, start="2026-01-01T00:00:00Z", res=15):
    t = pd.date_range(start, periods=len(values), freq=f"{res}min", tz="UTC")
    return pd.DataFrame({"start_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "resolution_min": res,
                         "price_adjusted_up": values})


def test_ffill_is_limited_and_marked():
    df = series([10.0, np.nan, np.nan, np.nan, np.nan, np.nan, 12.0])
    out = imputation.impute(df, "afrr_price", "price_adjusted_up", CFG)         # rule: ffill, 4 periods
    assert out["price_adjusted_up"].iloc[:5].tolist() == [10.0] * 5 and np.isnan(out["price_adjusted_up"].iloc[5])
    assert out["price_adjusted_up__imputed"].tolist() == [False, True, True, True, True, False, False]


def test_prev_day_uses_exactly_24h_earlier_and_none_leaves_gaps():
    n = 96 * 2
    vals = list(np.arange(n, dtype=float))
    vals[100] = np.nan
    df = series(vals).rename(columns={"price_adjusted_up": "price_programmed_activation"})
    out = imputation.impute(df, "mfrr_price", "price_programmed_activation", CFG)
    assert out["price_programmed_activation"].iloc[100] == 100 - 96
    assert bool(out["price_programmed_activation__imputed"].iloc[100])
    none_rule = series([1.0, np.nan, 3.0]).rename(columns={"price_adjusted_up": "price_pt_eur_mwh"})
    out2 = imputation.impute(none_rule, "da_price", "price_pt_eur_mwh", CFG)
    assert np.isnan(out2["price_pt_eur_mwh"].iloc[1]) and not out2["price_pt_eur_mwh__imputed"].any()


def test_imputation_never_uses_the_future():
    base = [5.0, np.nan, np.nan, 9.0, np.nan]
    a = imputation.impute(series(base), "afrr_price", "price_adjusted_up", CFG)
    changed = base[:3] + [1000.0, np.nan]                                        # change only later values
    b = imputation.impute(series(changed), "afrr_price", "price_adjusted_up", CFG)
    assert a["price_adjusted_up"].iloc[:3].tolist() == b["price_adjusted_up"].iloc[:3].tolist()


# --------------------------------------------------------------- provenance
def test_provenance_labels_and_training_filter():
    assert provenance.label("pv_weather", dt.date(2026, 6, 30), CFG) == "live"
    assert provenance.label("pv_weather", dt.date(2026, 7, 1), CFG) == "synthetic"
    assert provenance.label("inflow", dt.date(2026, 9, 1), CFG) == "synthetic"
    assert provenance.label("da_price", dt.date(2026, 9, 1), CFG) == "live"
    df = pd.DataFrame({"delivery_date": ["2026-06-30", "2026-07-01"], "ghi": [1.0, 2.0]})
    lab = provenance.label_frame(df, "pv_weather", cfg=CFG)
    assert provenance.trainable(lab)["delivery_date"].tolist() == ["2026-06-30"]
    with pytest.raises(KeyError):
        provenance.trainable(df)
