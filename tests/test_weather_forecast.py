"""test_weather_forecast.py - the weather forecast the PV forecast is built on: archive first, live only for a day that has not started, else the fallback."""
from __future__ import annotations

import datetime as dt
import importlib
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "phase_2_da_day_ahead_bidding", "da_price_pv_inflow_forecasting"))
from common_layer.data_layer import weather_forecast as wf


def _archive(tmp_path, days=("2025-03-10",), drop_hour=None):
    rows = []
    for d in days:
        for h in range(24):
            if h == drop_hour:
                continue
            rows.append({"datetime_utc": pd.Timestamp(f"{d} {h:02d}:00", tz="UTC"), "ghi_fc_d1": 1.0, "ghi_fc_d2": 100.0 + h - (200 if h == 3 else 0),
                         "tamb_fc_d1": 1.0, "tamb_fc_d2": 10.0 + h})
    path = tmp_path / "fc.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    return str(path)


def test_the_archive_gives_the_two_day_old_forecast_by_hour_and_clips_irradiance(tmp_path, monkeypatch):
    monkeypatch.setattr(wf, "ARCHIVE", _archive(tmp_path))
    wf._archive_cache.clear()
    ghi, tamb = wf.archived("2025-03-10")
    assert ghi[1] == 100.0 and ghi[24] == 123.0 and tamb[1] == 10.0 and tamb[24] == 33.0
    assert ghi[4] == 0.0                                                              # hour 4 (UTC 03:00) was -97: irradiance is clipped at 0


def test_an_incomplete_or_missing_day_is_not_used(tmp_path, monkeypatch):
    monkeypatch.setattr(wf, "ARCHIVE", _archive(tmp_path, drop_hour=7))
    wf._archive_cache.clear()
    assert wf.archived("2025-03-10") is None and wf.archived("2025-03-11") is None
    monkeypatch.setattr(wf, "ARCHIVE", str(tmp_path / "nothing.csv.gz"))
    wf._archive_cache.clear()
    assert wf.archived("2025-03-10") is None


def test_a_past_day_is_never_asked_of_the_live_service(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the live service must not be called for a past day")))
    assert wf.live("2020-01-01") is None


def test_a_day_that_has_not_started_uses_the_live_service_and_a_failure_gives_none(monkeypatch):
    tomorrow = str(dt.date.today() + dt.timedelta(days=1))
    wf._live_cache.clear()

    class Ok:
        def raise_for_status(self): pass
        def json(self): return {"hourly": {"shortwave_radiation": [float(i) for i in range(24)], "temperature_2m": [15.0] * 24}}

    monkeypatch.setattr("requests.get", lambda *a, **k: Ok())
    ghi, tamb = wf.live(tomorrow)
    assert ghi[24] == 23.0 and tamb[1] == 15.0
    wf._live_cache.clear()
    monkeypatch.setattr("requests.get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    assert wf.live(tomorrow) is None


def test_the_pv_forecaster_takes_the_weather_forecast_first_and_falls_back_without_it(monkeypatch):
    m = importlib.import_module("pv_power_forecaster")
    fixed = ({h: 5.0 for h in range(1, 25)}, {h: 20.0 for h in range(1, 25)})
    monkeypatch.setattr(wf, "forecast", lambda d, allow_live=True: fixed)
    monkeypatch.setattr(m, "_fill_gaps", lambda d: (_ for _ in ()).throw(AssertionError("the model fallback must not run")))
    assert m._forecast_ghi_tamb("2025-03-10") == fixed
    monkeypatch.setattr(wf, "forecast", lambda d, allow_live=True: None)
    monkeypatch.setattr(m, "_fill_gaps", lambda d: (_ for _ in ()).throw(KeyError("fallback reached")))
    with pytest.raises(KeyError):
        m._forecast_ghi_tamb("2025-03-10")


# ------------------------------------------------------------------------------------------------ the newest days as they happened
def _hourly_payload(first_day, n_days, drop=None):
    times = [pd.Timestamp(first_day) + pd.Timedelta(hours=i) for i in range(24 * n_days) if drop is None or i != drop]
    return {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times], "shortwave_radiation": [100.0 + t.hour for t in times],
                       "temperature_2m": [10.0 + t.hour / 10 for t in times]}}


class _Reply:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._p


def test_recent_returns_only_complete_days_before_today(monkeypatch):
    today = dt.date.today()
    first = today - dt.timedelta(days=3)
    monkeypatch.setattr("requests.get", lambda *a, **k: _Reply(_hourly_payload(first, 4)))        # three past days and today
    df = wf.recent(first)
    assert sorted(set(df["datetime"].dt.date)) == [first, first + dt.timedelta(days=1), first + dt.timedelta(days=2)]      # today is not data
    assert len(df) == 72 and df["Hour"].min() == 1 and df["Hour"].max() == 24
    monkeypatch.setattr("requests.get", lambda *a, **k: _Reply(_hourly_payload(first, 4, drop=30)))      # one hour of the second day is missing
    assert sorted(set(wf.recent(first)["datetime"].dt.date)) == [first, first + dt.timedelta(days=2)]


def test_recent_gives_none_when_the_service_fails_or_the_days_are_too_old(monkeypatch):
    today = dt.date.today()
    monkeypatch.setattr("requests.get", lambda *a, **k: _Reply({}, status=500))
    assert wf.recent(today - dt.timedelta(days=3)) is None
    monkeypatch.setattr("requests.get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    assert wf.recent(today - dt.timedelta(days=3)) is None
    assert wf.recent(today - dt.timedelta(days=200)) is None and wf.recent(today) is None


def test_the_gap_fill_takes_real_recent_days_first_and_marks_the_rest_synthetic(monkeypatch):
    m = importlib.import_module("pv_power_forecaster")
    today = pd.Timestamp(dt.date.today())
    have = pd.DataFrame({"Date": [today - pd.Timedelta(days=3)] * 24, "Hour": list(range(1, 25)), "GHI": [50.0] * 24, "T_amb": [10.0] * 24, "source": ["ERA5_REANALYSIS"] * 24})
    saved = {}
    monkeypatch.setattr(m, "_load_excel", lambda: have.copy())
    monkeypatch.setattr(m, "_save_excel", lambda df: saved.setdefault("df", df))
    two_days = pd.DataFrame({"datetime": [today - pd.Timedelta(days=2) + pd.Timedelta(hours=h) for h in range(24)] + [today - pd.Timedelta(days=1) + pd.Timedelta(hours=h) for h in range(24)],
                             "Hour": list(range(1, 25)) * 2, "GHI": [200.0] * 48, "T_amb": [15.0] * 48})
    monkeypatch.setattr(wf, "recent", lambda a, b=None, **k: two_days)
    m._fill_gaps(str((today + pd.Timedelta(days=1)).date()))
    out = saved["df"]
    by_day = out.groupby(out["Date"].dt.date)["source"].agg(lambda s: set(s)).to_dict()
    assert by_day[(today - pd.Timedelta(days=2)).date()] == {"OPEN_METEO_RECENT"} and by_day[(today - pd.Timedelta(days=1)).date()] == {"OPEN_METEO_RECENT"}
    assert by_day[today.date()] == {"SYNTHETIC"}                                      # today is not over: the only synthetic day
    saved.clear()
    monkeypatch.setattr(wf, "recent", lambda a, b=None, **k: None)                    # the service is unreachable: synthetic, flagged
    m._fill_gaps(str((today + pd.Timedelta(days=1)).date()))
    assert set(saved["df"].loc[saved["df"]["Date"] > today - pd.Timedelta(days=3), "source"]) == {"SYNTHETIC"}
