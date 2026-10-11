"""
test_phase0_pipeline.py — the daily data pipeline (phase_0_data_pipeline) and its hook into run_production.py.

Offline: the demo serves real saved files of the 25-hour day 2025-10-26 for all eight sources.
"""
from __future__ import annotations

import datetime as dt
import os
import time

import pytest

import run_production
from common_layer.data_layer.clean_store import CleanStore
from phase_0_data_pipeline import run_data_pipeline as p0
from phase_0_data_pipeline.demo import DEMO_DAY


def test_config_is_valid_and_covers_every_source():
    cfg = p0.load_pipeline_config()
    from common_layer.data_layer.ingest import SOURCES
    assert set(cfg["sources"]) == set(SOURCES)
    assert cfg["gate_mode"] in ("warn", "block")


def test_dry_run_plan_asks_for_tomorrows_day_ahead_only(tmp_path):
    today = dt.date(2026, 10, 11)
    plan = p0.run_daily(today=today, dry_run=True)["plan"]
    assert plan["omie_da"][1] == "2026-10-12"            # tomorrow's day-ahead results
    assert plan["omie_xbid"][1] == "2026-10-10"          # final the next day
    assert plan["omie_da"][0] == "2026-10-04"            # 7 days back


def test_demo_runs_green_on_the_25_hour_day_and_is_repeatable(tmp_path):
    work = str(tmp_path / "demo")
    first = p0.run_demo(work)
    assert first["status"] == "GREEN" and first["requests"] == 8
    assert sum(v.get("loaded", 0) for v in first["ingest"].values()) == 8
    clean = CleanStore(os.path.join(work, "clean", "market_clean.db"))
    assert len(clean.read("da_price")) == 100 and clean.days("da_price") == [DEMO_DAY.isoformat()]
    assert len(clean.read("ida_price")) == 100 + 100 + 48          # IDA1, IDA2 full day, IDA3 last 12 h
    digest = {d: clean.content_digest(d) for d in ("da_price", "afrr_price", "ida_price")}
    # second run on the same folders: the re-check finds identical files, nothing changes
    second = p0.run_demo(work)
    assert second["status"] == "GREEN"
    assert all(set(v) == {"unchanged"} for v in second["ingest"].values())
    clean2 = CleanStore(os.path.join(work, "clean", "market_clean.db"))
    assert {d: clean2.content_digest(d) for d in digest} == digest
    assert "GREEN" in open(first["report"], encoding="utf-8").read()


def test_lock_blocks_a_second_run_and_clears_a_stale_one(tmp_path):
    lock = str(tmp_path / "pipeline.lock")
    with p0.pipeline_lock(lock):
        with pytest.raises(p0.PipelineBusy):
            with p0.pipeline_lock(lock):
                pass
    assert not os.path.exists(lock)                                  # released
    open(lock, "w").write("crashed")
    old = time.time() - 24 * 3600
    os.utime(lock, (old, old))
    with p0.pipeline_lock(lock):                                     # stale lock is taken over
        pass


@pytest.mark.parametrize("gate_passed,mode,expected", [(True, "block", "OK"), (False, "warn", "DATA_WARN"),
                                                       (False, "block", "FAILED")])
def test_phase_entry_point_maps_the_gate_to_a_status(monkeypatch, gate_passed, mode, expected):
    cfg = p0.load_pipeline_config()
    cfg["gate_mode"] = mode
    monkeypatch.setattr(p0, "load_pipeline_config", lambda path=None: cfg)
    monkeypatch.setattr(p0, "run_daily", lambda c=None, **k: {
        "status": "GREEN" if gate_passed else "RED", "gate_passed": gate_passed,
        "reasons": [] if gate_passed else ["da_price: 5/7 days missing"], "ingest": {"omie_da": {"loaded": 2}},
        "report": "r.md"})
    assert p0.run_data_phase("12-10-2026")["status"] == expected


def test_phase_entry_point_skips_synthetic_runs_and_survives_a_busy_lock(monkeypatch):
    assert p0.run_data_phase("12-10-2026", use_synthetic=True)["status"] == "SKIPPED"

    def busy(*a, **k):
        raise p0.PipelineBusy("another run")
    monkeypatch.setattr(p0, "run_daily", busy)
    assert p0.run_data_phase("12-10-2026")["status"] == "DATA_WARN"


def test_orchestrator_runs_the_data_phase_first_and_treats_data_warn_as_a_warning():
    keys = [k for k, _, _ in run_production._PHASES]
    assert keys[0] == "data" and keys[1] == "da"
    assert "DATA_WARN" in run_production._WARN_CODES
    status, _ = run_production._dispatch("data", "12-10-2026", None, True, True)    # synthetic -> skipped
    assert status == run_production.SKIP
